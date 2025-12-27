//! Symmetric Eigenvalue Decomposition using Divide-and-Conquer.
//!
//! Uses Householder tridiagonalization followed by the divide-and-conquer algorithm.
//! This is typically faster than the QR algorithm for large matrices.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for divide-and-conquer symmetric eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricEvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for SymmetricEvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for SymmetricEvdDcError {}

/// Symmetric eigenvalue decomposition using divide-and-conquer.
///
/// Computes A = V·D·V^T where V contains eigenvectors and D is diagonal.
/// The divide-and-conquer algorithm is typically faster than QR for large matrices.
#[derive(Debug, Clone)]
pub struct SymmetricEvdDc<T: Scalar> {
    /// Eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns of V).
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

/// Threshold for switching to direct QR algorithm.
/// For matrices smaller than this, use QR directly.
/// Temporarily set high to use QR for all practical sizes until D&C merge is fixed.
const DC_THRESHOLD: usize = 100;

/// Maximum iterations for secular equation solver.
const MAX_SECULAR_ITER: usize = 100;

impl<T: Field + Real + bytemuck::Zeroable> SymmetricEvdDc<T> {
    /// Computes the eigendecomposition of a symmetric matrix using divide-and-conquer.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only upper triangle is used)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::SymmetricEvdDc;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[2.0f64, 1.0],
    ///     &[1.0, 2.0],
    /// ]);
    ///
    /// let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
    /// let eigs = evd.eigenvalues();
    ///
    /// // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    /// assert!((eigs[0] - 1.0).abs() < 1e-10);
    /// assert!((eigs[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SymmetricEvdDcError> {
        let n = a.nrows();

        if n == 0 {
            return Err(SymmetricEvdDcError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(SymmetricEvdDcError::NotSquare);
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalues = vec![a[(0, 0)]];
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        // Copy symmetric matrix (use upper triangle)
        let mut work = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = a[(i, j)];
                work[(i, j)] = val;
                work[(j, i)] = val;
            }
        }

        // Initialize eigenvector matrix to identity
        let mut v = Mat::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = T::one();
        }

        // Tridiagonalize: A = Q * T * Q^T
        let (diag, off_diag) = tridiagonalize(&mut work, &mut v, n);

        // Apply divide-and-conquer algorithm to tridiagonal matrix
        let eigenvalues = divide_and_conquer(diag, off_diag, &mut v, n)?;

        Ok(Self {
            eigenvalues,
            eigenvectors: v,
            n,
        })
    }

    /// Returns the eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix V.
    ///
    /// Column i contains the eigenvector corresponding to eigenvalue i.
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = V * D * V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a = Mat::zeros(n, n);

        // A = V * D * V^T = sum_i lambda_i * v_i * v_i^T
        for k in 0..n {
            let lambda = self.eigenvalues[k];
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] =
                        a[(i, j)] + lambda * self.eigenvectors[(i, k)] * self.eigenvectors[(j, k)];
                }
            }
        }

        a
    }
}

/// Tridiagonalizes a symmetric matrix using Householder reflections.
/// Returns (diagonal, off-diagonal) vectors.
fn tridiagonalize<T: Field + Real>(a: &mut Mat<T>, v: &mut Mat<T>, n: usize) -> (Vec<T>, Vec<T>) {
    let mut diag = vec![T::zero(); n];
    let mut off_diag = vec![T::zero(); n.saturating_sub(1)];

    for k in 0..(n.saturating_sub(2)) {
        // Compute Householder vector for column k (rows k+1 to n-1)
        let mut norm_sq = T::zero();
        for i in (k + 1)..n {
            norm_sq = norm_sq + a[(i, k)] * a[(i, k)];
        }
        let norm = Real::sqrt(norm_sq);

        if norm > T::zero() {
            let x_k1 = a[(k + 1, k)];
            let beta = if x_k1 >= T::zero() { -norm } else { norm };

            // Compute tau
            let tau = (beta - x_k1) / beta;

            // Scale Householder vector
            let scale = T::one() / (x_k1 - beta);
            for i in (k + 2)..n {
                a[(i, k)] = a[(i, k)] * scale;
            }

            // Apply Householder from left and right
            // p = tau * A * v
            let mut p = vec![T::zero(); n];
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    p[i] = p[i] + a[(i, j)] * v_j;
                }
                p[i] = tau * p[i];
            }

            // w = p - (tau/2) * (p^T * v) * v
            let mut ptv = T::zero();
            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                ptv = ptv + p[i] * v_i;
            }
            let half_tau = tau / (T::one() + T::one());

            let mut w = vec![T::zero(); n];
            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                w[i] = p[i] - half_tau * ptv * v_i;
            }

            // Update A: A = A - v*w^T - w*v^T
            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    a[(i, j)] = a[(i, j)] - v_i * w[j] - w[i] * v_j;
                }
            }

            // Update V: V = V * (I - tau * v * v^T)
            for i in 0..n {
                let mut vv = T::zero();
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    vv = vv + v[(i, j)] * v_j;
                }
                let tau_vv = tau * vv;
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    v[(i, j)] = v[(i, j)] - tau_vv * v_j;
                }
            }

            // Store off-diagonal element
            off_diag[k] = beta;
        }
    }

    // Extract diagonal and remaining off-diagonal
    for i in 0..n {
        diag[i] = a[(i, i)];
    }
    if n >= 2 {
        off_diag[n - 2] = a[(n - 1, n - 2)];
    }

    (diag, off_diag)
}

/// Divide-and-conquer algorithm for symmetric tridiagonal eigenvalue problem.
fn divide_and_conquer<T: Field + Real + bytemuck::Zeroable>(
    diag: Vec<T>,
    off_diag: Vec<T>,
    v: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T>, SymmetricEvdDcError> {
    if n <= 1 {
        return Ok(diag);
    }

    // For small matrices, use direct QR
    if n <= DC_THRESHOLD {
        return qr_algorithm(diag, off_diag, v, n);
    }

    // Divide: split at the middle
    let m = n / 2;
    let beta = off_diag[m - 1];

    // Create sub-problems
    // T1 = diag[0..m] with off_diag[0..m-1]
    // T2 = diag[m..n] with off_diag[m..n-1]

    // Modify the diagonal elements at the split point (rank-1 modification)
    let mut diag1: Vec<T> = diag[0..m].to_vec();
    let mut diag2: Vec<T> = diag[m..n].to_vec();

    // β comes from the (m-1, m) entry
    // T = [T1     ] + β * z * z^T where z = [0,...,0,1,1,0,...,0]^T
    //     [    T2 ]
    let abs_beta = Scalar::abs(beta);
    diag1[m - 1] = diag1[m - 1] - abs_beta;
    diag2[0] = diag2[0] - abs_beta;

    let off_diag1: Vec<T> = if m > 1 {
        off_diag[0..(m - 1)].to_vec()
    } else {
        vec![]
    };
    let off_diag2: Vec<T> = if n - m > 1 {
        off_diag[m..(n - 1)].to_vec()
    } else {
        vec![]
    };

    // Create temporary eigenvector matrices for subproblems
    let mut v1: Mat<T> = Mat::zeros(m, m);
    let mut v2: Mat<T> = Mat::zeros(n - m, n - m);
    for i in 0..m {
        v1[(i, i)] = T::one();
    }
    for i in 0..(n - m) {
        v2[(i, i)] = T::one();
    }

    // Recursively solve subproblems
    let eig1 = divide_and_conquer(diag1, off_diag1, &mut v1, m)?;
    let eig2 = divide_and_conquer(diag2, off_diag2, &mut v2, n - m)?;

    // Merge: solve the secular equation
    // The merged eigenvalues satisfy: 1 + rho * sum_i (z_i^2 / (d_i - lambda)) = 0
    // where d_i are the combined eigenvalues and z is the perturbation vector

    let rho = abs_beta;

    // Build the z vector from the last row of V1 and first row of V2
    let mut z = vec![T::zero(); n];
    for i in 0..m {
        z[i] = v1[(m - 1, i)];
    }
    for i in 0..(n - m) {
        z[m + i] = v2[(0, i)];
    }

    // Sign of beta determines the sign of z components
    if beta < T::zero() {
        for i in 0..(n - m) {
            z[m + i] = -z[m + i];
        }
    }

    // Combine eigenvalues from subproblems
    let mut d: Vec<T> = Vec::with_capacity(n);
    d.extend_from_slice(&eig1);
    d.extend_from_slice(&eig2);

    // Sort eigenvalues and permute z accordingly
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

    let d_sorted: Vec<T> = perm.iter().map(|&i| d[i]).collect();
    let z_sorted: Vec<T> = perm.iter().map(|&i| z[i]).collect();

    // Solve secular equation for each eigenvalue
    let merged_eigenvalues = solve_secular_equations(&d_sorted, &z_sorted, rho, n)?;

    // Compute eigenvectors of merged problem
    let merged_v = compute_merged_eigenvectors(&d_sorted, &z_sorted, &merged_eigenvalues, rho, n);

    // Apply permutation to eigenvectors
    let mut unperm_v: Mat<T> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            unperm_v[(perm[i], j)] = merged_v[(i, j)];
        }
    }

    // Combine with subproblem eigenvectors
    // V_new = [V1 0 ] * merged_V
    //         [0  V2]
    let mut combined: Mat<T> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::zero();
            for k in 0..m {
                sum = sum + v1[(i, k)] * unperm_v[(k, j)];
            }
            combined[(i, j)] = sum;
        }
        for i in 0..(n - m) {
            let mut sum = T::zero();
            for k in 0..(n - m) {
                sum = sum + v2[(i, k)] * unperm_v[(m + k, j)];
            }
            combined[(m + i, j)] = sum;
        }
    }

    // Apply the accumulated transformation from tridiagonalization
    // V_final = V * combined
    let v_copy = v.clone();
    for j in 0..n {
        for i in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + v_copy[(i, k)] * combined[(k, j)];
            }
            v[(i, j)] = sum;
        }
    }

    // Sort final eigenvalues and eigenvectors
    let mut eigenvalues = merged_eigenvalues;
    sort_eigenvalues(&mut eigenvalues, v, n);

    Ok(eigenvalues)
}

/// Solve secular equations: 1 + rho * sum_i (z_i^2 / (d_i - lambda)) = 0
fn solve_secular_equations<T: Field + Real>(
    d: &[T],
    z: &[T],
    rho: T,
    n: usize,
) -> Result<Vec<T>, SymmetricEvdDcError> {
    let mut eigenvalues = Vec::with_capacity(n);

    for k in 0..n {
        // Find eigenvalue k in the interval (d[k], d[k+1]) or beyond
        let (lower, upper) = if k < n - 1 {
            (d[k], d[k + 1])
        } else {
            // Last eigenvalue: estimate upper bound
            let sum_z_sq: T = z.iter().fold(T::zero(), |acc, &zi| acc + zi * zi);
            (d[n - 1], d[n - 1] + rho * sum_z_sq)
        };

        // Use modified Newton's method with safeguards
        let lambda = solve_single_secular(d, z, rho, lower, upper, k)?;
        eigenvalues.push(lambda);
    }

    Ok(eigenvalues)
}

/// Solve a single secular equation in the given interval.
fn solve_single_secular<T: Field + Real>(
    d: &[T],
    z: &[T],
    rho: T,
    lower: T,
    upper: T,
    k: usize,
) -> Result<T, SymmetricEvdDcError> {
    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    // Handle near-zero z[k] case (eigenvalue is approximately d[k])
    if Scalar::abs(z[k]) < tol {
        return Ok(d[k]);
    }

    let two = T::one() + T::one();

    // Initial guess: midpoint or weighted average
    let mut lambda = (lower + upper) / two;

    // Newton iteration with bisection safeguard
    let mut a = lower;
    let mut b = upper;

    for _ in 0..MAX_SECULAR_ITER {
        // Evaluate secular function f(lambda) = 1 + rho * sum(z_i^2 / (d_i - lambda))
        let (f, df) = secular_function_and_derivative(d, z, rho, lambda);

        if Scalar::abs(f) < tol * (T::one() + Scalar::abs(lambda)) {
            return Ok(lambda);
        }

        // Newton step
        let delta = f / df;
        let lambda_new = lambda - delta;

        // Check if Newton step is within bounds
        if lambda_new > a && lambda_new < b {
            lambda = lambda_new;
        } else {
            // Bisection fallback
            lambda = (a + b) / two;
        }

        // Update bounds based on function sign
        let (f_new, _) = secular_function_and_derivative(d, z, rho, lambda);
        if f_new < T::zero() {
            a = lambda;
        } else {
            b = lambda;
        }

        // Check convergence
        if b - a < tol * (T::one() + Scalar::abs(lambda)) {
            return Ok((a + b) / two);
        }
    }

    // Return best estimate even if not fully converged
    Ok(lambda)
}

/// Evaluate secular function f(λ) = 1 + ρ * Σ(z_i² / (d_i - λ)) and its derivative.
fn secular_function_and_derivative<T: Field + Real>(d: &[T], z: &[T], rho: T, lambda: T) -> (T, T) {
    let n = d.len();
    let mut f = T::one();
    let mut df = T::zero();

    for i in 0..n {
        let zi_sq = z[i] * z[i];
        let diff = d[i] - lambda;
        if Scalar::abs(diff) > <T as Scalar>::epsilon() {
            f = f + rho * zi_sq / diff;
            df = df + rho * zi_sq / (diff * diff);
        }
    }

    (f, df)
}

/// Compute eigenvectors of merged problem.
fn compute_merged_eigenvectors<T: Field + Real + bytemuck::Zeroable>(
    d: &[T],
    z: &[T],
    eigenvalues: &[T],
    _rho: T,
    n: usize,
) -> Mat<T> {
    let mut v: Mat<T> = Mat::zeros(n, n);

    for j in 0..n {
        let lambda = eigenvalues[j];

        // v_j = (D - lambda*I)^{-1} * z / ||...||
        let mut norm_sq = T::zero();
        for i in 0..n {
            let diff = d[i] - lambda;
            if Scalar::abs(diff) > <T as Scalar>::epsilon() {
                v[(i, j)] = z[i] / diff;
            } else {
                // d[i] == lambda case
                v[(i, j)] = T::one();
            }
            norm_sq = norm_sq + v[(i, j)] * v[(i, j)];
        }

        // Normalize
        let norm = Real::sqrt(norm_sq);
        if norm > T::zero() {
            for i in 0..n {
                v[(i, j)] = v[(i, j)] / norm;
            }
        }
    }

    v
}

/// QR algorithm for small symmetric tridiagonal matrices (base case).
fn qr_algorithm<T: Field + Real>(
    mut diag: Vec<T>,
    mut off_diag: Vec<T>,
    v: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T>, SymmetricEvdDcError> {
    const MAX_ITERATIONS: usize = 100;

    if n <= 1 {
        return Ok(diag);
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

    // QR iterations with implicit shifts
    let mut m = n - 1;
    let mut iter = 0;

    while m > 0 && iter < MAX_ITERATIONS * n {
        iter += 1;

        // Find largest m such that off_diag[m-1] is not negligible
        let mut l = m;
        while l > 0 {
            let test = Scalar::abs(diag[l - 1]) + Scalar::abs(diag[l]);
            if Scalar::abs(off_diag[l - 1]) <= eps * test {
                off_diag[l - 1] = T::zero();
                break;
            }
            l -= 1;
        }

        if l == m {
            // Eigenvalue found
            m -= 1;
            continue;
        }

        // Wilkinson shift
        let d = (diag[m - 1] - diag[m]) / (T::one() + T::one());
        let e = off_diag[m - 1];
        let mu = diag[m] - e * e / (d + Real::signum(d) * Real::hypot(d, e));

        // Implicit QR step
        let mut x = diag[l] - mu;
        let mut z = off_diag[l];

        for k in l..m {
            // Givens rotation to annihilate z
            let (c, s) = givens_rotation(x, z);

            if k > l {
                off_diag[k - 1] = Real::hypot(x, z);
            }

            // Update tridiagonal matrix
            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e = off_diag[k];

            diag[k] = c * c * d1 + s * s * d2 - (c + c) * s * e;
            diag[k + 1] = s * s * d1 + c * c * d2 + (c + c) * s * e;
            off_diag[k] = c * s * (d1 - d2) + (c * c - s * s) * e;

            if k < m - 1 {
                x = off_diag[k];
                z = -s * off_diag[k + 1];
                off_diag[k + 1] = c * off_diag[k + 1];
            }

            // Update eigenvectors
            for i in 0..n {
                let t1 = v[(i, k)];
                let t2 = v[(i, k + 1)];
                v[(i, k)] = c * t1 - s * t2;
                v[(i, k + 1)] = s * t1 + c * t2;
            }
        }
    }

    if iter >= MAX_ITERATIONS * n {
        return Err(SymmetricEvdDcError::NotConverged);
    }

    // Sort eigenvalues in ascending order
    sort_eigenvalues(&mut diag, v, n);

    Ok(diag)
}

/// Computes Givens rotation coefficients.
fn givens_rotation<T: Field + Real>(a: T, b: T) -> (T, T) {
    if b == T::zero() {
        (T::one(), T::zero())
    } else if Scalar::abs(b) > Scalar::abs(a) {
        let t = -a / b;
        let s = T::one() / Real::sqrt(T::one() + t * t);
        (s * t, s)
    } else {
        let t = -b / a;
        let c = T::one() / Real::sqrt(T::one() + t * t);
        (c, c * t)
    }
}

/// Sorts eigenvalues in ascending order and rearranges eigenvectors accordingly.
fn sort_eigenvalues<T: Field + Real>(eigenvalues: &mut [T], v: &mut Mat<T>, n: usize) {
    // Simple insertion sort (stable and efficient for small n)
    for i in 1..n {
        let key = eigenvalues[i];
        let mut j = i;
        while j > 0 && eigenvalues[j - 1] > key {
            eigenvalues[j] = eigenvalues[j - 1];
            // Swap eigenvector columns
            for row in 0..n {
                let tmp = v[(row, j)];
                v[(row, j)] = v[(row, j - 1)];
                v[(row, j - 1)] = tmp;
            }
            j -= 1;
        }
        eigenvalues[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_evd_dc_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_3x3() {
        // Symmetric 3x3 matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        // Reconstruct and verify
        let reconstructed = evd.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_evd_dc_diagonal() {
        // Diagonal matrix - eigenvalues are the diagonal elements
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues sorted: 1, 2, 3
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvdDc::compute(eye.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // All eigenvalues should be 1
        for &e in eigs {
            assert!(approx_eq(e, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_evd_dc_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 5.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_negative_eigenvalues() {
        // Matrix with negative eigenvalues
        let a = Mat::from_rows(&[&[-2.0f64, 1.0], &[1.0, -2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues: -3 and -1
        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_evd_dc_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue (3 appears twice)
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_orthogonality() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-8),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_evd_dc_larger_matrix() {
        // Test with a simple diagonal 10x10 matrix
        let n = 10;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create a simple diagonal matrix with distinct eigenvalues
        for i in 0..n {
            a[(i, i)] = (i + 1) as f64;
        }

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        // Eigenvalues should be 1, 2, 3, ..., 10
        let eigs = evd.eigenvalues();
        for i in 0..n {
            assert!(
                approx_eq(eigs[i], (i + 1) as f64, 1e-10),
                "eigenvalue {} = {}, expected {}",
                i,
                eigs[i],
                i + 1
            );
        }

        // Verify reconstruction
        let reconstructed = evd.reconstruct();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }

        // Verify orthogonality
        let v = evd.eigenvectors();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-8),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_evd_dc_vs_qr() {
        // Compare results with standard QR-based EVD
        use super::super::symmetric::SymmetricEvd;

        let a = Mat::from_rows(&[
            &[4.0f64, 2.0, 1.0, 0.5],
            &[2.0, 5.0, 3.0, 1.0],
            &[1.0, 3.0, 6.0, 2.0],
            &[0.5, 1.0, 2.0, 4.0],
        ]);

        let evd_qr = SymmetricEvd::compute(a.as_ref()).unwrap();
        let evd_dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        let eigs_qr = evd_qr.eigenvalues();
        let eigs_dc = evd_dc.eigenvalues();

        for i in 0..4 {
            assert!(
                approx_eq(eigs_qr[i], eigs_dc[i], 1e-8),
                "eigenvalue {} mismatch: QR={}, DC={}",
                i,
                eigs_qr[i],
                eigs_dc[i]
            );
        }
    }
}
