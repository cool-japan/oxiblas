//! Symmetric Eigenvalue Decomposition using Divide-and-Conquer.
//!
//! Uses Householder tridiagonalization followed by Cuppen's divide-and-conquer
//! algorithm for the symmetric tridiagonal eigenvalue problem.
//!
//! The tridiagonal matrix `T` is split into two sub-blocks `T1`, `T2` plus a
//! rank-one update `rho * z z^T`.  The two sub-blocks are solved recursively and
//! their eigensystems are merged by
//!
//! 1. deflating negligible components of `z` and (near-)equal eigenvalues
//!    (Gu-Eisenstat / LAPACK `dlaed2`-style deflation),
//! 2. solving the secular equation
//!    `f(lambda) = 1 + rho * sum_i z_i^2 / (d_i - lambda) = 0`
//!    for the non-deflated eigenvalues with a safeguarded rational-interpolation
//!    root finder (LAPACK `dlaed4`-style "middle way"),
//! 3. forming the eigenvectors from the *stabilised* rank-one vector recomputed
//!    via the Löwner formula (LAPACK `dlaed3`) and combining them with the
//!    block-diagonal sub-block eigenvectors — **not** by copying the block
//!    diagonal eigenvectors.
//!
//! This is typically faster than the QR algorithm for large matrices and, unlike
//! a naïve implementation, produces numerically orthogonal eigenvectors.

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

/// Sub-problem size at or below which the base-case QR solver is used.
///
/// This mirrors LAPACK's `SMLSIZ`.  For blocks larger than this the genuine
/// divide-and-conquer merge is used; for smaller blocks the (already correct and
/// numerically robust) tridiagonal QR iteration is cheaper.  The merge is exact
/// at every size, so this threshold only trades performance, never correctness.
const SMLSIZ: usize = 25;

/// Maximum number of iterations for a single secular-equation root.
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

        // Accumulated orthogonal transform from tridiagonalization: A = Q * T * Q^T.
        let mut q = Mat::eye(n);

        // Tridiagonalize.
        let (diag, off_diag) = tridiagonalize(&mut work, &mut q, n);

        // Solve the tridiagonal eigenproblem by divide-and-conquer, obtaining the
        // eigenvalues and the eigenvectors `z_mat` of the tridiagonal matrix.
        let (eigenvalues, z_mat) = tridiag_dc(diag, off_diag, SMLSIZ)?;

        // Eigenvectors of A are V = Q * z_mat.
        let mut v = Mat::zeros(n, n);
        for jc in 0..n {
            for r in 0..n {
                let mut acc = T::zero();
                for kk in 0..n {
                    acc = acc + q[(r, kk)] * z_mat[(kk, jc)];
                }
                v[(r, jc)] = acc;
            }
        }

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

/// Cuppen divide-and-conquer for the symmetric tridiagonal eigenvalue problem.
///
/// Given the diagonal `diag` (length `n`) and off-diagonal `off_diag` (length
/// `n-1`) of a symmetric tridiagonal matrix `T`, returns its eigenvalues (ascending)
/// and the matrix whose columns are the corresponding eigenvectors.
///
/// `smlsiz` is the sub-problem size at or below which the base-case QR solver is
/// used.  A value of `1` forces the divide-and-conquer merge at every level.
fn tridiag_dc<T: Field + Real + bytemuck::Zeroable>(
    diag: Vec<T>,
    off_diag: Vec<T>,
    smlsiz: usize,
) -> Result<(Vec<T>, Mat<T>), SymmetricEvdDcError> {
    let n = diag.len();

    if n == 0 {
        return Ok((Vec::new(), Mat::zeros(0, 0)));
    }
    if n == 1 {
        let mut z = Mat::zeros(1, 1);
        z[(0, 0)] = T::one();
        return Ok((diag, z));
    }

    // Base case: tridiagonal QR iteration with eigenvectors accumulated from the
    // identity.  Correct and numerically robust for small blocks.
    if n <= smlsiz {
        let mut z = Mat::eye(n);
        let eig = qr_algorithm(diag, off_diag, &mut z, n)?;
        return Ok((eig, z));
    }

    // Divide: split at the middle.
    let m = n / 2;
    let beta = off_diag[m - 1];
    let abs_beta = Scalar::abs(beta);

    // Rank-one tearing:  T = diag(T1, T2) + rho * v v^T  with  rho = |beta|  and
    // v = [0,..,0, 1, sign(beta), 0,..,0]^T  (ones at positions m-1 and m).
    // This removes |beta| from the coupling diagonal entries of each block.
    let mut diag1: Vec<T> = diag[0..m].to_vec();
    let mut diag2: Vec<T> = diag[m..n].to_vec();
    diag1[m - 1] = diag1[m - 1] - abs_beta;
    diag2[0] = diag2[0] - abs_beta;

    let off1: Vec<T> = if m > 1 {
        off_diag[0..(m - 1)].to_vec()
    } else {
        Vec::new()
    };
    let off2: Vec<T> = if n - m > 1 {
        off_diag[m..(n - 1)].to_vec()
    } else {
        Vec::new()
    };

    // Recursively solve the two sub-problems.
    let (eig1, q1) = tridiag_dc(diag1, off1, smlsiz)?;
    let (eig2, q2) = tridiag_dc(diag2, off2, smlsiz)?;

    // Merge the two eigensystems through the rank-one update.
    merge_rank_one(&eig1, &q1, &eig2, &q2, beta, m, n)
}

/// Merges the eigensystems of two sub-blocks joined by a rank-one update.
///
/// The combined matrix is `diag(Q1 D1 Q1^T, Q2 D2 Q2^T) = Q (D + rho z z^T) Q^T`
/// with `Q = diag(Q1, Q2)`, `D = diag(eig1, eig2)`, `rho = |beta|` and
/// `z = [last row of Q1 ; sign(beta) * first row of Q2]`.
///
/// Returns the eigenvalues (ascending) and the eigenvectors of the combined
/// tridiagonal block.
fn merge_rank_one<T: Field + Real + bytemuck::Zeroable>(
    eig1: &[T],
    q1: &Mat<T>,
    eig2: &[T],
    q2: &Mat<T>,
    beta: T,
    m: usize,
    n: usize,
) -> Result<(Vec<T>, Mat<T>), SymmetricEvdDcError> {
    // Assemble combined d, z and the block-diagonal eigenvector matrix.
    let mut d = vec![T::zero(); n];
    let mut z = vec![T::zero(); n];
    let mut qmat: Mat<T> = Mat::zeros(n, n);

    for a in 0..m {
        d[a] = eig1[a];
        z[a] = q1[(m - 1, a)];
        for r in 0..m {
            qmat[(r, a)] = q1[(r, a)];
        }
    }
    let sgn = if beta < T::zero() {
        -T::one()
    } else {
        T::one()
    };
    for a in 0..(n - m) {
        d[m + a] = eig2[a];
        z[m + a] = sgn * q2[(0, a)];
        for r in 0..(n - m) {
            qmat[(m + r, m + a)] = q2[(r, a)];
        }
    }

    let mut rho = Scalar::abs(beta);

    // If the coupling is (numerically) absent the problem is block diagonal.
    let mut znorm_sq = T::zero();
    for &za in z.iter() {
        znorm_sq = znorm_sq + za * za;
    }
    if !(rho > T::zero()) || !(znorm_sq > T::zero()) {
        let mut eigenvalues = d;
        let mut vmat = qmat;
        sort_eigenvalues(&mut eigenvalues, &mut vmat, n);
        return Ok((eigenvalues, vmat));
    }

    // Normalise z to unit length and fold the scale into rho.
    let znorm = Real::sqrt(znorm_sq);
    for za in z.iter_mut() {
        *za = *za / znorm;
    }
    rho = rho * znorm_sq;

    // Sort the combined eigenvalues ascending, permuting z and the columns of Q.
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&x, &y| d[x].partial_cmp(&d[y]).unwrap_or(std::cmp::Ordering::Equal));
    let d_sorted: Vec<T> = perm.iter().map(|&x| d[x]).collect();
    let z_sorted: Vec<T> = perm.iter().map(|&x| z[x]).collect();
    let mut q_sorted: Mat<T> = Mat::zeros(n, n);
    for (newc, &oldc) in perm.iter().enumerate() {
        for r in 0..n {
            q_sorted[(r, newc)] = qmat[(r, oldc)];
        }
    }
    let mut d = d_sorted;
    let mut z = z_sorted;
    let mut qmat = q_sorted;

    // Deflation tolerance (LAPACK dlaed2).
    let eps = <T as Scalar>::epsilon();
    let mut dmax = T::zero();
    let mut zmax = T::zero();
    for (&da, &za) in d.iter().zip(z.iter()) {
        let ad = Scalar::abs(da);
        if ad > dmax {
            dmax = ad;
        }
        let az = Scalar::abs(za);
        if az > zmax {
            zmax = az;
        }
    }
    let eight = T::from_f64(8.0).unwrap_or(T::one());
    let tol = eight * eps * if dmax > zmax { dmax } else { zmax };

    // Deflation pass.  `active` collects the poles kept in the secular equation;
    // `deflated` collects the eigenpairs that need no further work.
    let mut active: Vec<usize> = Vec::new();
    let mut deflated: Vec<usize> = Vec::new();
    let mut pj: Option<usize> = None;

    for s in 0..n {
        if rho * Scalar::abs(z[s]) <= tol {
            // Type-1 deflation: negligible z component -> d[s] is an eigenvalue.
            deflated.push(s);
            continue;
        }
        match pj {
            None => pj = Some(s),
            Some(p) => {
                let zp = z[p];
                let zc = z[s];
                let tau = Real::hypot(zc, zp);
                let c = zc / tau;
                let sn = zp / tau;
                let t = d[s] - d[p];
                if Scalar::abs(t * c * sn) <= tol {
                    // Type-2 deflation: (near-)equal poles.  Rotate columns p, s of
                    // Q so that the new z has a zero in position p; that column
                    // deflates with eigenvalue d[p].
                    for r in 0..n {
                        let qa = qmat[(r, p)];
                        let qb = qmat[(r, s)];
                        qmat[(r, p)] = c * qa - sn * qb;
                        qmat[(r, s)] = sn * qa + c * qb;
                    }
                    let dp_new = d[p] * c * c + d[s] * sn * sn;
                    let ds_new = d[p] * sn * sn + d[s] * c * c;
                    d[p] = dp_new;
                    d[s] = ds_new;
                    z[p] = T::zero();
                    z[s] = tau;
                    deflated.push(p);
                    pj = Some(s);
                } else {
                    active.push(p);
                    pj = Some(s);
                }
            }
        }
    }
    if let Some(p) = pj {
        active.push(p);
    }

    // Keep the active poles strictly ascending (rotations can perturb d slightly).
    active.sort_by(|&x, &y| d[x].partial_cmp(&d[y]).unwrap_or(std::cmp::Ordering::Equal));
    let k = active.len();

    let mut eigenvalues = vec![T::zero(); n];
    let mut vmat: Mat<T> = Mat::zeros(n, n);

    if k == 0 {
        // Everything deflated.
        for (idx, &col) in deflated.iter().enumerate() {
            eigenvalues[idx] = d[col];
            for r in 0..n {
                vmat[(r, idx)] = qmat[(r, col)];
            }
        }
        sort_eigenvalues(&mut eigenvalues, &mut vmat, n);
        return Ok((eigenvalues, vmat));
    }

    // Reduced secular problem.
    let dhat: Vec<T> = active.iter().map(|&a| d[a]).collect();
    let zhat: Vec<T> = active.iter().map(|&a| z[a]).collect();
    let zeta: Vec<T> = zhat.iter().map(|&zz| rho * zz * zz).collect();

    // Solve the secular equation for every active eigenvalue, recording the
    // accurate delta = dhat - lambda for each.
    let mut lambda = vec![T::zero(); k];
    let mut delta_mat: Vec<Vec<T>> = Vec::with_capacity(k);
    for l in 0..k {
        let (lam, del) = solve_secular_i(&dhat, &zeta, l, k)?;
        lambda[l] = lam;
        delta_mat.push(del);
    }

    // Gu-Eisenstat: recompute the magnitudes of z from the Löwner formula so the
    // eigenvectors are numerically orthogonal.
    //   rho * zt_a^2 = (lambda_a - dhat_a) * prod_{l != a} (lambda_l - dhat_a)/(dhat_l - dhat_a)
    let mut zt = vec![T::zero(); k];
    for a in 0..k {
        let mut prod = -delta_mat[a][a]; // lambda_a - dhat_a
        for l in 0..k {
            if l == a {
                continue;
            }
            let num = -delta_mat[l][a]; // lambda_l - dhat_a
            let den = dhat[l] - dhat[a]; // != 0 (poles are distinct)
            prod = prod * (num / den);
        }
        let val = prod / rho;
        let mag = if val > T::zero() {
            Real::sqrt(val)
        } else {
            T::zero()
        };
        zt[a] = if zhat[a] >= T::zero() { mag } else { -mag };
    }

    // Form the eigenvectors of the active problem and combine them with the
    // (rotated) block-diagonal eigenvectors held in `qmat`.
    for l in 0..k {
        // Secular eigenvector in the reduced basis: u_a = zt_a / (dhat_a - lambda_l).
        let mut u = vec![T::zero(); k];
        let mut nrm_sq = T::zero();
        for a in 0..k {
            let val = zt[a] / delta_mat[l][a];
            u[a] = val;
            nrm_sq = nrm_sq + val * val;
        }
        let nrm = Real::sqrt(nrm_sq);
        let inv = if nrm > T::zero() {
            T::one() / nrm
        } else {
            T::one()
        };

        for r in 0..n {
            let mut acc = T::zero();
            for (a, &ua) in u.iter().enumerate() {
                acc = acc + ua * qmat[(r, active[a])];
            }
            vmat[(r, l)] = acc * inv;
        }
        eigenvalues[l] = lambda[l];
    }

    // Append the deflated eigenpairs (columns of the rotated Q).
    for (idx, &col) in deflated.iter().enumerate() {
        let slot = k + idx;
        eigenvalues[slot] = d[col];
        for r in 0..n {
            vmat[(r, slot)] = qmat[(r, col)];
        }
    }

    sort_eigenvalues(&mut eigenvalues, &mut vmat, n);
    Ok((eigenvalues, vmat))
}

/// Solves the secular equation `1 + sum_j zeta_j / (d_j - lambda) = 0` for the
/// `i`-th eigenvalue (0-indexed).
///
/// The root lies in `(d[i], d[i+1])` for `i < k-1` and in
/// `(d[k-1], d[k-1] + sum(zeta))` for the last eigenvalue.  `zeta_j = rho * z_j^2`
/// with `zeta_j > 0`, and the poles `d` are strictly ascending.
///
/// Uses a safeguarded rational-interpolation ("middle way") iteration with
/// origin shifting, so the returned `delta[j] = d[j] - lambda` is accurate even
/// for eigenvalues extremely close to a pole.
fn solve_secular_i<T: Field + Real>(
    d: &[T],
    zeta: &[T],
    i: usize,
    k: usize,
) -> Result<(T, Vec<T>), SymmetricEvdDcError> {
    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();
    let eight = T::from_f64(8.0).unwrap_or(T::one());

    let mut sum_zeta = T::zero();
    for &zj in zeta.iter() {
        sum_zeta = sum_zeta + zj;
    }

    // Choose the origin (nearest pole) and the eta = lambda - d[orig] bracket.
    let (orig, mut elo, mut ehi) = if i + 1 < k {
        let gap = d[i + 1] - d[i];
        let mid = d[i] + gap / two;
        let mut fmid = T::one();
        for (&dj, &zj) in d.iter().zip(zeta.iter()) {
            fmid = fmid + zj / (dj - mid);
        }
        if fmid >= T::zero() {
            (i, T::zero(), gap)
        } else {
            (i + 1, -gap, T::zero())
        }
    } else {
        (k - 1, T::zero(), sum_zeta)
    };

    let d_orig = d[orig];
    let mut eta = (elo + ehi) / two;
    let mut delta = vec![T::zero(); k];

    for _iter in 0..MAX_SECULAR_ITER {
        for (deltaj, &dj) in delta.iter_mut().zip(d.iter()) {
            *deltaj = (dj - d_orig) - eta;
        }

        let mut psi = T::zero();
        let mut dpsi = T::zero();
        let mut phi = T::zero();
        let mut dphi = T::zero();
        let mut err = T::one();
        for (&zj, &dj) in zeta.iter().zip(delta.iter()).take(i + 1) {
            let t = zj / dj;
            psi = psi + t;
            dpsi = dpsi + t / dj;
            err = err + Scalar::abs(t);
        }
        for (&zj, &dj) in zeta.iter().zip(delta.iter()).skip(i + 1) {
            let t = zj / dj;
            phi = phi + t;
            dphi = dphi + t / dj;
            err = err + Scalar::abs(t);
        }
        let f = T::one() + psi + phi;

        // f is increasing in eta; maintain the sign bracket.
        if f >= T::zero() {
            ehi = eta;
        } else {
            elo = eta;
        }

        if !(Scalar::abs(f) > eight * eps * err) {
            break;
        }

        // Rational-interpolation step.
        let step_opt = if i + 1 < k {
            let dl = delta[i];
            let dr = delta[i + 1];
            let b1 = dpsi * dl * dl;
            let a1 = psi - dpsi * dl;
            let b2 = dphi * dr * dr;
            let a2 = phi - dphi * dr;
            let c0 = T::one() + a1 + a2;
            let bq = c0 * (dl + dr) + b1 + b2;
            let cq = c0 * dl * dr + b1 * dr + b2 * dl;
            solve_quadratic_in_range(c0, bq, cq, dl, dr)
        } else {
            let dl = delta[k - 1];
            let b1 = dpsi * dl * dl;
            let a1 = psi - dpsi * dl;
            let c0 = T::one() + a1 + phi;
            if c0 != T::zero() {
                let s = dl + b1 / c0;
                if s > dl { Some(s) } else { None }
            } else {
                None
            }
        };

        let eta_next = match step_opt {
            Some(s) => {
                let cand = eta + s;
                if cand > elo && cand < ehi {
                    cand
                } else {
                    (elo + ehi) / two
                }
            }
            None => (elo + ehi) / two,
        };

        if eta_next == eta {
            let bis = (elo + ehi) / two;
            if bis == eta {
                break;
            }
            eta = bis;
        } else {
            eta = eta_next;
        }
    }

    for (deltaj, &dj) in delta.iter_mut().zip(d.iter()) {
        *deltaj = (dj - d_orig) - eta;
    }

    let lambda = d_orig + eta;
    if !lambda.is_finite() {
        return Err(SymmetricEvdDcError::SecularEquationFailed);
    }
    Ok((lambda, delta))
}

/// Solves `a*s^2 - b*s + c = 0` and returns the root strictly inside `(lo, hi)`,
/// if any.  Used by the secular-equation rational interpolation.
fn solve_quadratic_in_range<T: Field + Real>(a: T, b: T, c: T, lo: T, hi: T) -> Option<T> {
    let two = T::one() + T::one();
    let four = two * two;
    if a == T::zero() {
        if b == T::zero() {
            return None;
        }
        let s = c / b;
        if s > lo && s < hi {
            return Some(s);
        }
        return None;
    }
    let disc = b * b - four * a * c;
    if disc < T::zero() {
        return None;
    }
    let sq = Real::sqrt(disc);
    let r1 = (b - sq) / (two * a);
    let r2 = (b + sq) / (two * a);
    if r1 > lo && r1 < hi {
        Some(r1)
    } else if r2 > lo && r2 < hi {
        Some(r2)
    } else {
        None
    }
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
                // The regenerated off-diagonal is r = c*x - s*z.  Using hypot(x, z)
                // here loses the sign, which leaves eigenvalues correct but corrupts
                // the accumulated eigenvectors.
                off_diag[k - 1] = c * x - s * z;
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

    /// Deterministic split-mix PRNG producing values in [-1, 1] (no external deps).
    struct Prng {
        state: u64,
    }

    impl Prng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    /// Builds a dense symmetric matrix with pseudo-random entries.
    fn random_symmetric(n: usize, seed: u64) -> Mat<f64> {
        let mut prng = Prng::new(seed);
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = prng.next_f64();
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        a
    }

    /// Builds A = H * diag(eigs) * H where H = I - 2 w w^T is a Householder
    /// reflector.  The eigenvalues of A are exactly `eigs` (with any repeats).
    fn symmetric_with_spectrum(eigs: &[f64], seed: u64) -> Mat<f64> {
        let n = eigs.len();
        let mut prng = Prng::new(seed);
        let mut w = vec![0.0f64; n];
        let mut norm_sq = 0.0;
        for wi in w.iter_mut() {
            *wi = prng.next_f64();
            norm_sq += *wi * *wi;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for wi in w.iter_mut() {
                *wi /= norm;
            }
        }
        // H = I - 2 w w^T ; A = H D H.
        let mut h: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let delta = if i == j { 1.0 } else { 0.0 };
                h[(i, j)] = delta - 2.0 * w[i] * w[j];
            }
        }
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += h[(i, k)] * eigs[k] * h[(j, k)];
                }
                a[(i, j)] = acc;
            }
        }
        a
    }

    /// Maximum |V^T V - I| over all entries.
    fn orthogonality_error(v: MatRef<'_, f64>, n: usize) -> f64 {
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                err = err.max((dot - expected).abs());
            }
        }
        err
    }

    /// Maximum residual |A v_i - lambda_i v_i| over all eigenpairs.
    fn residual_error(a: &Mat<f64>, evd: &SymmetricEvdDc<f64>, n: usize) -> f64 {
        let v = evd.eigenvectors();
        let eigs = evd.eigenvalues();
        let mut err = 0.0f64;
        for col in 0..n {
            let lambda = eigs[col];
            for row in 0..n {
                let mut av = 0.0;
                for k in 0..n {
                    av += a[(row, k)] * v[(k, col)];
                }
                err = err.max((av - lambda * v[(row, col)]).abs());
            }
        }
        err
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

    /// Compares divide-and-conquer eigenvalues against the QR reference and checks
    /// eigenvector orthogonality and residuals across a range of sizes, including
    /// sizes well above `SMLSIZ` that exercise the real D&C merge.
    #[test]
    fn test_evd_dc_reference_across_sizes() {
        use super::super::symmetric::SymmetricEvd;

        for (idx, &n) in [3usize, 5, 10, 50, 150, 300].iter().enumerate() {
            let a = random_symmetric(n, 0x1234_5678 + idx as u64 * 97);

            let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
            let qr = SymmetricEvd::compute(a.as_ref()).unwrap();

            let dc_eigs = dc.eigenvalues();
            let qr_eigs = qr.eigenvalues();

            // Scale tolerance by the spectral radius.
            let scale = qr_eigs.iter().fold(1.0f64, |acc, &e| acc.max(e.abs()));
            let eig_tol = 1e-7 * scale * (n as f64).sqrt();

            for i in 0..n {
                assert!(
                    (dc_eigs[i] - qr_eigs[i]).abs() <= eig_tol,
                    "n={}: eigenvalue {} mismatch DC={} QR={} (tol {})",
                    n,
                    i,
                    dc_eigs[i],
                    qr_eigs[i],
                    eig_tol
                );
            }

            let ortho = orthogonality_error(dc.eigenvectors(), n);
            assert!(
                ortho <= 1e-9 * (n as f64),
                "n={}: orthogonality error {} too large",
                n,
                ortho
            );

            let resid = residual_error(&a, &dc, n);
            assert!(
                resid <= 1e-7 * scale * (n as f64),
                "n={}: residual {} too large",
                n,
                resid
            );
        }
    }

    /// Directly exercises the divide-and-conquer merge at *every* level (smlsiz=1)
    /// for small tridiagonal matrices, comparing against the QR base solver.
    #[test]
    fn test_tridiag_dc_forced_merge_small() {
        for &n in &[2usize, 3, 5, 10] {
            let mut prng = Prng::new(0xABCD_00FF + n as u64);
            let diag: Vec<f64> = (0..n).map(|_| prng.next_f64() * 4.0).collect();
            let off: Vec<f64> = (0..n - 1).map(|_| prng.next_f64() * 2.0 + 0.5).collect();

            // Reference: QR base solver on the same tridiagonal.
            let mut z_ref = Mat::eye(n);
            let ref_eigs = qr_algorithm(diag.clone(), off.clone(), &mut z_ref, n).unwrap();

            // Forced divide-and-conquer (merge at every level).
            let (dc_eigs, dc_vec) = tridiag_dc(diag.clone(), off.clone(), 1).unwrap();

            for i in 0..n {
                assert!(
                    (dc_eigs[i] - ref_eigs[i]).abs() <= 1e-9 * (1.0 + ref_eigs[i].abs()),
                    "n={}: forced-merge eigenvalue {} mismatch DC={} REF={}",
                    n,
                    i,
                    dc_eigs[i],
                    ref_eigs[i]
                );
            }

            let ortho = orthogonality_error(dc_vec.as_ref(), n);
            assert!(
                ortho <= 1e-10 * (n as f64),
                "n={}: forced-merge orthogonality error {} too large",
                n,
                ortho
            );
        }
    }

    /// Exercises type-2 deflation in the merge path: a matrix built with many
    /// exactly-repeated eigenvalues but non-trivial (dense) coupling.
    #[test]
    fn test_evd_dc_deflation_repeated_spectrum() {
        // 40 eigenvalues with several repeats; SMLSIZ=25 forces one D&C merge.
        let mut eigs = vec![
            1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, -3.0, -3.0, -3.0, -3.0, -3.0, 7.0, 7.0, 0.0, 0.0,
            0.0, 5.5, 5.5, 9.0,
        ];
        // Pad up to 40 with a few more repeats and distinct values.
        while eigs.len() < 40 {
            let v = eigs.len() as f64 * 0.31 - 6.0;
            eigs.push(v);
            eigs.push(v); // ensure repeats appear across the split boundary
        }
        eigs.truncate(40);
        let n = eigs.len();

        let a = symmetric_with_spectrum(&eigs, 0x5151_2727);

        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let dc_eigs = dc.eigenvalues();

        let mut sorted = eigs.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let scale = sorted.iter().fold(1.0f64, |acc, &e| acc.max(e.abs()));
        for i in 0..n {
            assert!(
                (dc_eigs[i] - sorted[i]).abs() <= 1e-7 * scale * (n as f64).sqrt(),
                "eigenvalue {} mismatch DC={} expected={}",
                i,
                dc_eigs[i],
                sorted[i]
            );
        }

        let ortho = orthogonality_error(dc.eigenvectors(), n);
        assert!(
            ortho <= 1e-9 * (n as f64),
            "deflation orthogonality error {} too large",
            ortho
        );

        let resid = residual_error(&a, &dc, n);
        assert!(
            resid <= 1e-7 * scale * (n as f64),
            "deflation residual {} too large",
            resid
        );
    }

    /// f32 divide-and-conquer at a size that exercises the merge.
    #[test]
    fn test_evd_dc_f32_larger() {
        let n = 60usize;
        let mut prng = Prng::new(0x2468_ACE0);
        let mut a: Mat<f32> = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (prng.next_f64() as f32) * 2.0;
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }

        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = dc.eigenvalues();

        // Ascending order.
        for i in 0..n - 1 {
            assert!(eigs[i] <= eigs[i + 1] + 1e-3);
        }

        // Orthogonality (looser for f32).
        let v = dc.eigenvectors();
        let mut ortho = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f32;
                for k in 0..n {
                    dot += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                ortho = ortho.max((dot - expected).abs());
            }
        }
        assert!(
            ortho <= 1e-3 * n as f32,
            "f32 orthogonality {} too large",
            ortho
        );
    }
}
