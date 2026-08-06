//! # `IRAM` - select_shifts_general_group Methods
//!
//! This module contains method implementations for `IRAM`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::error::WhichEigenvalues;
use super::super::utils::{dot, norm};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::iram_type::IRAM;

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> IRAM<T> {
    /// Select wanted eigenvalues and unwanted shifts for general case.
    pub(super) fn select_shifts_general(
        &self,
        real: &[T],
        imag: &[T],
        nev: usize,
        p: usize,
    ) -> (Vec<usize>, Vec<T>, Vec<T>) {
        let n = real.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        let nev = nev.min(n);
        let p = p.min(n.saturating_sub(nev));

        // Compute magnitudes
        let mut indexed: Vec<(usize, T)> = real
            .iter()
            .zip(imag.iter())
            .enumerate()
            .map(|(i, (r, im))| {
                let mag = Real::sqrt(r.clone() * r.clone() + im.clone() * im.clone());
                (i, mag)
            })
            .collect();

        // Sort by magnitude based on selection criterion
        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // For algebraic, sort by real part
                let mut real_indexed: Vec<(usize, T)> = real
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.clone()))
                    .collect();
                match self.config.which {
                    WhichEigenvalues::LargestAlgebraic => {
                        real_indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        real_indexed.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
                indexed = real_indexed;
            }
        }

        let wanted_indices: Vec<usize> = indexed.iter().take(nev).map(|(i, _)| *i).collect();
        let unwanted_real: Vec<T> = indexed
            .iter()
            .skip(nev)
            .take(p)
            .map(|(i, _)| real[*i].clone())
            .collect();
        let unwanted_imag: Vec<T> = indexed
            .iter()
            .skip(nev)
            .take(p)
            .map(|(i, _)| imag[*i].clone())
            .collect();

        (wanted_indices, unwanted_real, unwanted_imag)
    }

    /// Apply implicit QR shifts for symmetric tridiagonal case.
    pub(super) fn apply_implicit_qr_shifts_symmetric(
        &self,
        alpha: &mut Vec<T>,
        beta: &mut Vec<T>,
        v: &mut Vec<Vec<T>>,
        f: &mut [T],
        shifts: &[T],
        nev: usize,
    ) {
        let n = v[0].len();
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for shift in shifts {
            let m = alpha.len();
            if m < 2 {
                break;
            }

            // Apply bulge-chasing QR step with shift
            // For tridiagonal, this is more efficient than full Hessenberg QR

            // First, compute the initial Givens rotation
            let mut g = alpha[0].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();

            for i in 0..m - 1 {
                let i_beta = if i < beta.len() {
                    beta[i].clone()
                } else {
                    T::zero()
                };
                let f_val = s.clone() * i_beta.clone();
                let b = c.clone() * i_beta;

                // Givens rotation to chase bulge
                if Scalar::abs(f_val.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f_val.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > 0 && i - 1 < beta.len() {
                        beta[i - 1] = f_val.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c.clone() * s.clone();
                } else {
                    s = f_val.clone() / g.clone();
                    let r = Real::sqrt(s.clone() * s.clone() + T::one());
                    if i > 0 && i - 1 < beta.len() {
                        beta[i - 1] = g.clone() * r.clone();
                    }
                    c = T::one() / r.clone();
                    s = s.clone() * c.clone();
                }

                // Update tridiagonal elements
                let p_val = s.clone() * alpha[i].clone() - c.clone() * b.clone();
                let next_alpha = if i + 1 < alpha.len() {
                    alpha[i + 1].clone()
                } else {
                    T::zero()
                };
                let r = (next_alpha.clone() - alpha[i].clone()) * s.clone()
                    + two.clone() * c.clone() * b.clone();

                alpha[i] = alpha[i].clone() + p_val.clone();
                g = c.clone() * r.clone() - b.clone();

                if i + 1 < alpha.len() {
                    alpha[i + 1] = next_alpha - s.clone() * r.clone();
                }

                // Apply Givens rotation to Lanczos vectors
                for k in 0..n {
                    if i + 1 < v.len() {
                        let tmp = c.clone() * v[i][k].clone() + s.clone() * v[i + 1][k].clone();
                        v[i + 1][k] = T::zero() - s.clone() * v[i][k].clone()
                            + c.clone() * v[i + 1][k].clone();
                        v[i][k] = tmp;
                    }
                }
            }

            // Update beta[m-2] if it exists
            if m >= 2 && m - 2 < beta.len() {
                beta[m - 2] = g;
            }
        }

        // Apply rotation to residual vector
        if !shifts.is_empty() {
            let m = alpha.len();
            if m > 0 && m <= v.len() {
                // The residual is updated implicitly through the Lanczos vectors
                let beta_last = if !beta.is_empty() {
                    beta.last().cloned().unwrap_or_else(T::zero)
                } else {
                    T::zero()
                };
                let last_v = v.last().cloned().unwrap_or_else(|| vec![T::zero(); n]);
                for i in 0..n {
                    f[i] = beta_last.clone() * last_v[i].clone();
                }
            }
        }

        // Truncate to nev
        alpha.truncate(nev);
        if !beta.is_empty() {
            beta.truncate(nev.saturating_sub(1));
        }
    }

    /// Apply the `p` implicit QR shifts of one IRAM restart to the general (non-symmetric)
    /// upper-Hessenberg factorization.
    ///
    /// Given the Arnoldi factorization `A*V_m = V_m*H_m + f*e_m^T`, this performs `p`
    /// shifted-QR sweeps that filter out the unwanted Ritz values, compressing the
    /// factorization from dimension `m = ncv` down to `k = nev`.
    ///
    /// Real unwanted Ritz values are applied as single **real** shifts (Francis single
    /// shift). Complex-conjugate unwanted Ritz pairs `(mu, conj(mu))` are applied
    /// together as a single **real double shift** (Francis double shift) using the
    /// quadratic factor `(H - mu*I)(H - conj(mu)*I) = H^2 - 2*Re(mu)*H + |mu|^2*I`,
    /// which keeps the whole sweep in real arithmetic. This is the standard,
    /// convergence-preserving treatment for spectra with genuine complex-conjugate
    /// pairs; applying only `Re(mu)` as a naive single shift (as an earlier version
    /// did, discarding the imaginary parts) degrades or breaks convergence.
    ///
    /// All sweeps are accumulated into an orthogonal matrix `q` so that
    /// `H_m^+ = q^T H_m q` and `V_m^+ = V_m q`. The restarted residual is then
    /// `f^+ = beta_k^+ * v_k^+ + sigma_k * f`, with `beta_k^+ = H_m^+[k][k-1]` the
    /// subdiagonal at the truncation boundary and `sigma_k = q[m-1][k-1]` the
    /// corresponding component of the last row of `q`.
    pub(super) fn apply_implicit_qr_shifts_general(
        &self,
        h: &mut [Vec<T>],
        v: &mut [Vec<T>],
        f: &mut [T],
        shifts_real: &[T],
        shifts_imag: &[T],
        nev: usize,
        ncv: usize,
    ) {
        let n = v[0].len();
        let m = v.len().min(ncv);
        if m < nev + 1 || m < 2 {
            return;
        }

        // Accumulated orthogonal transformation q (m x m), initialised to identity.
        let mut q: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();

        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        // Relative threshold: a shift is "complex" only if it carries an imaginary part
        // significant relative to the magnitude of the shift itself.
        let eps_scale = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        let mut idx = 0;
        while idx < shifts_real.len() {
            let mu_re = shifts_real[idx].clone();
            let mu_im = if idx < shifts_imag.len() {
                shifts_imag[idx].clone()
            } else {
                T::zero()
            };

            let imag_threshold = eps_scale.clone() * (T::one() + Scalar::abs(mu_re.clone()));
            if Scalar::abs(mu_im.clone()) <= imag_threshold {
                // Genuinely real unwanted Ritz value: single real Francis shift.
                self.francis_single_shift(h, &mut q, mu_re, m);
                idx += 1;
                continue;
            }

            // Complex Ritz value: try to pair it with its conjugate (the next shift).
            let pair_tol = eps_scale.clone()
                * (T::one() + Scalar::abs(mu_re.clone()) + Scalar::abs(mu_im.clone()));
            let has_conjugate = idx + 1 < shifts_real.len()
                && idx + 1 < shifts_imag.len()
                && Scalar::abs(shifts_real[idx + 1].clone() - mu_re.clone()) <= pair_tol
                && Scalar::abs(shifts_imag[idx + 1].clone() + mu_im.clone()) <= pair_tol;

            if has_conjugate {
                // Francis double shift with s = 2*Re(mu), t = |mu|^2.
                let s = two.clone() * mu_re.clone();
                let t = mu_re.clone() * mu_re.clone() + mu_im.clone() * mu_im.clone();
                self.francis_double_shift(h, &mut q, s, t, m);
                idx += 2;
            } else {
                // Conjugate partner is a wanted Ritz value (pair split across the
                // wanted/unwanted boundary). Applying the full quadratic would shift
                // out the wanted conjugate too, so fall back to a safe single real
                // shift at Re(mu).
                self.francis_single_shift(h, &mut q, mu_re, m);
                idx += 1;
            }
        }

        // Form V_m^+ = V_m * q (only the columns that survive matter, but recomputing
        // all of them keeps the basis consistent and is cheap for small m).
        let v_old: Vec<Vec<T>> = v[..m].to_vec();
        for (j, col) in v.iter_mut().enumerate().take(m) {
            let mut new_col = vec![T::zero(); n];
            for (l, v_old_l) in v_old.iter().enumerate() {
                let q_lj = q[l][j].clone();
                if Scalar::abs(q_lj.clone()) <= <T as Scalar>::epsilon() {
                    continue;
                }
                for row in 0..n {
                    new_col[row] = new_col[row].clone() + v_old_l[row].clone() * q_lj.clone();
                }
            }
            *col = new_col;
        }

        // Restarted residual f^+ = beta_k^+ * v_k^+ + sigma_k * f.
        let beta_k = h[nev][nev - 1].clone();
        let sigma_k = q[m - 1][nev - 1].clone();
        let f_old: Vec<T> = f.to_vec();
        for i in 0..n {
            f[i] = beta_k.clone() * v[nev][i].clone() + sigma_k.clone() * f_old[i].clone();
        }
    }

    /// One Francis single-shift implicit-QR sweep on the leading `m x m` block of the
    /// upper-Hessenberg matrix `h`, accumulating the transformation into `q`.
    ///
    /// The first reflector is built from the first column of `H - mu*I`
    /// (`[H[0][0] - mu, H[1][0]]`); the resulting bulge is then chased to the bottom
    /// with length-2 Householder reflectors. The net effect equals an explicit shifted
    /// QR step `H <- Q^T H Q` with `Q` the orthogonal factor of `H - mu*I`.
    pub(super) fn francis_single_shift(&self, h: &mut [Vec<T>], q: &mut [Vec<T>], mu: T, m: usize) {
        if m < 2 {
            return;
        }
        let mut x = h[0][0].clone() - mu;
        let mut y = h[1][0].clone();
        for k in 0..m - 1 {
            let (refl, beta) = Self::householder(&[x.clone(), y.clone()]);
            Self::apply_reflector_similarity(h, q, &refl, beta, k, m);
            if k + 2 < m {
                x = h[k + 1][k].clone();
                y = h[k + 2][k].clone();
            }
        }
    }

    /// One Francis double-shift (real) implicit-QR sweep on the leading `m x m` block of
    /// the upper-Hessenberg matrix `h`, accumulating the transformation into `q`.
    ///
    /// The sweep implicitly applies the complex-conjugate shift pair whose sum is `s`
    /// (`= 2*Re(mu)`) and product is `t` (`= |mu|^2`). The first reflector is built from
    /// the first column of `H^2 - s*H + t*I` (nonzero only in its first three entries),
    /// and the width-2 bulge it creates is chased down with length-3 Householder
    /// reflectors (length-2 at the very bottom). Everything stays in real arithmetic.
    pub(super) fn francis_double_shift(
        &self,
        h: &mut [Vec<T>],
        q: &mut [Vec<T>],
        s: T,
        t: T,
        m: usize,
    ) {
        if m < 2 {
            return;
        }

        // First column of H^2 - s*H + t*I (only the first three entries are nonzero for
        // an upper-Hessenberg H).
        let h00 = h[0][0].clone();
        let h01 = h[0][1].clone();
        let h10 = h[1][0].clone();
        let h11 = h[1][1].clone();
        let h21 = if m > 2 { h[2][1].clone() } else { T::zero() };

        let p0 = h00.clone() * h00.clone() + h01 * h10.clone() - s.clone() * h00.clone() + t;
        let p1 = h10.clone() * (h00 + h11 - s);
        let p2 = h10 * h21;

        for k in 0..m - 1 {
            let len = (m - k).min(3);
            let vec3 = if k == 0 {
                [p0.clone(), p1.clone(), p2.clone()]
            } else {
                let a = h[k][k - 1].clone();
                let b = h[k + 1][k - 1].clone();
                let c = if k + 2 < m {
                    h[k + 2][k - 1].clone()
                } else {
                    T::zero()
                };
                [a, b, c]
            };
            let (refl, beta) = Self::householder(&vec3[..len]);
            Self::apply_reflector_similarity(h, q, &refl, beta, k, m);
        }
    }

    /// Build a Householder reflector `P = I - beta*w*w^T` (length 2 or 3) that maps the
    /// input vector `x` to a multiple of `e_0`. Returns `(w, beta)`.
    pub(super) fn householder(x: &[T]) -> (Vec<T>, T) {
        let len = x.len();
        let mut norm_sq = T::zero();
        for xi in x {
            norm_sq = norm_sq + xi.clone() * xi.clone();
        }
        let xnorm = Real::sqrt(norm_sq);
        if xnorm <= <T as Scalar>::epsilon() {
            return (vec![T::zero(); len], T::zero());
        }
        // alpha = -sign(x[0]) * ||x|| for numerical stability.
        let alpha = if x[0] >= T::zero() {
            T::zero() - xnorm
        } else {
            xnorm
        };
        let mut w = x.to_vec();
        w[0] = w[0].clone() - alpha;
        let mut wtw = T::zero();
        for wi in &w {
            wtw = wtw + wi.clone() * wi.clone();
        }
        if wtw <= <T as Scalar>::epsilon() {
            return (vec![T::zero(); len], T::zero());
        }
        let beta = (T::one() + T::one()) / wtw;
        (w, beta)
    }

    /// Apply a Householder reflector `P = I - beta*w*w^T`, acting on indices
    /// `[k, k + w.len())`, as a similarity transform `H <- P^T H P` on the leading
    /// `m x m` block of `h`, and accumulate it into `q` (`q <- q P`). `P` is symmetric,
    /// so left and right applications use the same reflector.
    pub(super) fn apply_reflector_similarity(
        h: &mut [Vec<T>],
        q: &mut [Vec<T>],
        w: &[T],
        beta: T,
        k: usize,
        m: usize,
    ) {
        let len = w.len();
        if beta == T::zero() {
            return;
        }

        // Left application: H <- P H (rows k..k+len).
        for col in 0..m {
            let mut dot = T::zero();
            for (a, wa) in w.iter().enumerate() {
                dot = dot + wa.clone() * h[k + a][col].clone();
            }
            let scaled = beta.clone() * dot;
            for (a, wa) in w.iter().enumerate() {
                h[k + a][col] = h[k + a][col].clone() - wa.clone() * scaled.clone();
            }
        }

        // Right application: H <- H P (columns k..k+len).
        for row in 0..m {
            let mut dot = T::zero();
            for (a, wa) in w.iter().enumerate() {
                dot = dot + wa.clone() * h[row][k + a].clone();
            }
            let scaled = beta.clone() * dot;
            for (a, wa) in w.iter().enumerate() {
                h[row][k + a] = h[row][k + a].clone() - wa.clone() * scaled.clone();
            }
        }

        // Accumulate into q: q <- q P (columns k..k+len).
        for row in 0..m {
            let mut dot = T::zero();
            for (a, wa) in w.iter().enumerate() {
                dot = dot + wa.clone() * q[row][k + a].clone();
            }
            let scaled = beta.clone() * dot;
            for (a, wa) in w.iter().enumerate() {
                q[row][k + a] = q[row][k + a].clone() - wa.clone() * scaled.clone();
            }
        }

        let _ = len;
    }

    /// Compute eigenvectors for symmetric case.
    pub(super) fn compute_eigenvectors_symmetric(
        &self,
        lanczos_vectors: &[Vec<T>],
        ritz_vectors: &[Vec<T>],
        wanted_indices: &[usize],
        n: usize,
    ) -> Vec<Vec<T>> {
        let mut eigenvectors = Vec::with_capacity(wanted_indices.len());

        for &idx in wanted_indices {
            if idx >= ritz_vectors.len() {
                continue;
            }

            let y = &ritz_vectors[idx];
            let mut x = vec![T::zero(); n];

            // Transform from Lanczos basis to original basis: x = V * y
            for (j, vj) in lanczos_vectors.iter().enumerate() {
                if j < y.len() {
                    for i in 0..n {
                        x[i] = x[i].clone() + y[j].clone() * vj[i].clone();
                    }
                }
            }

            // Normalize
            let x_norm = norm(&x);
            if x_norm > <T as Scalar>::epsilon() {
                for xi in &mut x {
                    *xi = xi.clone() / x_norm.clone();
                }
            }

            eigenvectors.push(x);
        }

        eigenvectors
    }

    /// Compute a Ritz eigenvector of the leading `m x m` block of the upper
    /// Hessenberg matrix `h` associated with the eigenvalue whose real part is
    /// `lambda`, using inverse iteration on `(H - lambda*I)`.
    ///
    /// Because `lambda` is (nearly) an eigenvalue, `H - lambda*I` is close to
    /// singular; solving `(H - lambda*I) y_{k+1} = y_k` therefore amplifies the
    /// component of `y_k` along the requested eigenvector, and a couple of
    /// iterations converge to it. A single LU factorization (partial pivoting,
    /// with the pivots regularized to a small multiple of `||H||` so an exactly
    /// singular shift does not overflow) is reused across the iterations.
    ///
    /// The returned vector has length `m` and unit Euclidean norm.
    pub(crate) fn hessenberg_ritz_vector(&self, h: &[Vec<T>], m: usize, lambda: T) -> Vec<T> {
        if m == 0 {
            return Vec::new();
        }

        // Dense B = H - lambda*I.
        let mut b: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                if i < h.len() {
                    for (j, slot) in row.iter_mut().enumerate() {
                        if j < h[i].len() {
                            *slot = h[i][j].clone();
                        }
                    }
                }
                row[i] = row[i].clone() - lambda.clone();
                row
            })
            .collect();

        // Scale for pivot regularization: largest magnitude entry of B (>= 1).
        let mut scale = T::one();
        for row in &b {
            for entry in row {
                let a = Scalar::abs(entry.clone());
                if a > scale {
                    scale = a;
                }
            }
        }
        let eps = <T as Scalar>::epsilon();
        let pivot_floor = eps.clone() * scale;

        // LU factorization with partial pivoting, stored in place. `perm[i]` is
        // the original row now occupying factored row `i`.
        let mut perm: Vec<usize> = (0..m).collect();
        for k in 0..m {
            let mut pivot_row = k;
            let mut pivot_mag = Scalar::abs(b[k][k].clone());
            for i in (k + 1)..m {
                let cand = Scalar::abs(b[i][k].clone());
                if cand > pivot_mag {
                    pivot_mag = cand;
                    pivot_row = i;
                }
            }
            if pivot_row != k {
                b.swap(k, pivot_row);
                perm.swap(k, pivot_row);
            }

            // Regularize a (near-)zero pivot: the near-singular direction is the
            // eigenvector we are after, so we keep the factorization solvable
            // without destroying that direction.
            if Scalar::abs(b[k][k].clone()) <= pivot_floor {
                let signed = if b[k][k].clone() >= T::zero() {
                    pivot_floor.clone()
                } else {
                    T::zero() - pivot_floor.clone()
                };
                b[k][k] = signed;
            }

            let pivot = b[k][k].clone();
            for i in (k + 1)..m {
                let factor = b[i][k].clone() / pivot.clone();
                b[i][k] = factor.clone();
                for j in (k + 1)..m {
                    b[i][j] = b[i][j].clone() - factor.clone() * b[k][j].clone();
                }
            }
        }

        // Inverse iteration starting from a flat, generically non-orthogonal
        // seed so it has a component along the target eigenvector.
        let mut y = vec![T::one(); m];
        let seed_norm = norm(&y);
        if seed_norm > eps.clone() {
            for yi in &mut y {
                *yi = yi.clone() / seed_norm.clone();
            }
        }

        for _ in 0..3 {
            // Solve B*z = y using the stored LU factors: P applied to rhs, then
            // forward (unit lower L) and back (upper U) substitution.
            let mut z: Vec<T> = (0..m).map(|i| y[perm[i]].clone()).collect();
            for i in 0..m {
                let mut sum = z[i].clone();
                for j in 0..i {
                    sum = sum - b[i][j].clone() * z[j].clone();
                }
                z[i] = sum;
            }
            for i in (0..m).rev() {
                let mut sum = z[i].clone();
                for j in (i + 1)..m {
                    sum = sum - b[i][j].clone() * z[j].clone();
                }
                // Every U diagonal was regularized to magnitude >= pivot_floor > 0
                // above, so the division is always well defined. Dividing by the
                // (tiny, floored) pivot is exactly what amplifies the eigenvector
                // direction when the shift equals an eigenvalue.
                z[i] = sum / b[i][i].clone();
            }

            let z_norm = norm(&z);
            if z_norm <= eps.clone() {
                break;
            }
            for zi in &mut z {
                *zi = zi.clone() / z_norm.clone();
            }
            y = z;
        }

        y
    }

    /// Compute eigenvectors for the general (non-symmetric) case.
    ///
    /// For each selected Ritz value `lambda_idx` the eigenvector of the small
    /// upper Hessenberg matrix `H` is computed by inverse iteration on
    /// `(H - lambda_idx*I)` (see [`Self::hessenberg_ritz_vector`]) and then
    /// mapped back to the original space as `x = V*y`. The result is validated
    /// against the original operator `A` via the residual `||A*x - rho*x||`,
    /// where `rho = x^T A x` is the Rayleigh quotient; for real eigenvalues the
    /// shift is refined with `rho` (Rayleigh-quotient iteration) and the vector
    /// achieving the smallest residual is kept. This targets the *specific*
    /// requested eigenvalue instead of always converging to the dominant one.
    pub(super) fn compute_eigenvectors_general(
        &self,
        a: &CsrMatrix<T>,
        arnoldi_vectors: &[Vec<T>],
        h: &[Vec<T>],
        ritz_real: &[T],
        ritz_imag: &[T],
        wanted_indices: &[usize],
        n: usize,
        m: usize,
    ) -> Vec<Vec<T>> {
        let mut eigenvectors = Vec::with_capacity(wanted_indices.len());
        let eps = <T as Scalar>::epsilon();
        let res_tol = eps.clone() * T::from_f64(100.0).unwrap_or_else(T::one);
        let mut ax = vec![T::zero(); n];

        for &idx in wanted_indices {
            if idx >= m {
                continue;
            }

            let lambda_re = ritz_real.get(idx).cloned().unwrap_or_else(T::zero);
            let lambda_im = ritz_imag.get(idx).cloned().unwrap_or_else(T::zero);
            // A genuinely complex eigenvalue has no real eigenvector, so a
            // real-part shift cannot be refined to drive the residual to zero;
            // use a single inverse-iteration solve in that case.
            let is_complex = Scalar::abs(lambda_im) > eps.clone();
            let refine_steps = if is_complex { 1 } else { 4 };

            let mut shift = lambda_re;
            let mut best_x: Option<Vec<T>> = None;
            let mut best_res = T::zero();

            for step in 0..refine_steps {
                // Eigenvector of the small Hessenberg matrix for this Ritz value.
                let y = self.hessenberg_ritz_vector(h, m, shift.clone());

                // Map back to the original space: x = V * y.
                let mut x = vec![T::zero(); n];
                for (j, vj) in arnoldi_vectors.iter().enumerate() {
                    if j < y.len() {
                        for (i, xi) in x.iter_mut().enumerate() {
                            *xi = xi.clone() + y[j].clone() * vj[i].clone();
                        }
                    }
                }
                let x_norm = norm(&x);
                if x_norm <= eps.clone() {
                    break;
                }
                for xi in &mut x {
                    *xi = xi.clone() / x_norm.clone();
                }

                // Validate against the original operator A. With x normalized,
                // rho = x^T A x is the Rayleigh quotient and res = ||A x - rho x||.
                spmv(T::one(), a, &x, T::zero(), &mut ax);
                let rho = dot(&x, &ax);
                let mut res_sq = T::zero();
                for (i, axi) in ax.iter().enumerate() {
                    let ri = axi.clone() - rho.clone() * x[i].clone();
                    res_sq = res_sq + ri.clone() * ri.clone();
                }
                let res = Real::sqrt(res_sq);

                let improved = match best_x {
                    None => true,
                    Some(_) => res < best_res,
                };
                if improved {
                    best_res = res.clone();
                    best_x = Some(x);
                }

                // Rayleigh-quotient refinement of the shift for the next step.
                if !is_complex {
                    shift = rho;
                }
                if res <= res_tol || step + 1 == refine_steps {
                    break;
                }
            }

            eigenvectors.push(best_x.unwrap_or_else(|| vec![T::zero(); n]));
        }

        eigenvectors
    }

    /// Eigenvector and refined eigenvalue of the small Hessenberg matrix `H` for the Ritz
    /// value `re + i*im`, via Rayleigh-quotient iteration.
    ///
    /// Starting from the (approximate) Ritz value, each step solves the shifted system
    /// `(H - theta I) z = y` (complex inverse iteration), renormalises, and updates the
    /// shift to the Rayleigh quotient `theta = y^H H y` (with `||y|| = 1`). This converges
    /// cubically to an accurate eigenpair, so both the returned eigenvector and the
    /// refined eigenvalue are reliable even though the input Ritz value is not.
    ///
    /// Returns `(y_re, y_im, refined_re, refined_im)` with `y` normalised to unit 2-norm.
    pub(super) fn ritz_eigenvector(
        &self,
        h: &[Vec<T>],
        m: usize,
        re: T,
        im: T,
    ) -> (Vec<T>, Vec<T>, T, T) {
        if m == 0 {
            return (vec![], vec![], re, im);
        }

        let mut theta_re = re;
        let mut theta_im = im;

        // A "rich" real starting vector (unlikely to be orthogonal to the eigenvector).
        let mut y_re: Vec<T> = (0..m)
            .map(|i| T::from_usize(i + 1).unwrap_or_else(T::one))
            .collect();
        let mut y_im: Vec<T> = vec![T::zero(); m];
        Self::normalize_complex(&mut y_re, &mut y_im);

        let conv_tol = <T as Scalar>::epsilon() * T::from_f64(10.0).unwrap_or_else(T::one);

        for _ in 0..12 {
            let (mut z_re, mut z_im) = Self::complex_hessenberg_solve(
                h,
                m,
                theta_re.clone(),
                theta_im.clone(),
                &y_re,
                &y_im,
            );
            let mut nrm_sq = T::zero();
            for i in 0..m {
                nrm_sq =
                    nrm_sq + z_re[i].clone() * z_re[i].clone() + z_im[i].clone() * z_im[i].clone();
            }
            let nrm = Real::sqrt(nrm_sq);
            if nrm <= <T as Scalar>::epsilon() {
                break;
            }
            for i in 0..m {
                z_re[i] = z_re[i].clone() / nrm.clone();
                z_im[i] = z_im[i].clone() / nrm.clone();
            }
            y_re = z_re;
            y_im = z_im;

            // Rayleigh quotient theta = y^H (H y) (denominator y^H y = 1 after normalising).
            let mut hy_re = vec![T::zero(); m];
            let mut hy_im = vec![T::zero(); m];
            for i in 0..m {
                let mut sr = T::zero();
                let mut si = T::zero();
                for j in 0..m {
                    sr = sr + h[i][j].clone() * y_re[j].clone();
                    si = si + h[i][j].clone() * y_im[j].clone();
                }
                hy_re[i] = sr;
                hy_im[i] = si;
            }
            let mut num_re = T::zero();
            let mut num_im = T::zero();
            for i in 0..m {
                // conj(y_i) * (H y)_i
                num_re = num_re
                    + y_re[i].clone() * hy_re[i].clone()
                    + y_im[i].clone() * hy_im[i].clone();
                num_im = num_im + y_re[i].clone() * hy_im[i].clone()
                    - y_im[i].clone() * hy_re[i].clone();
            }

            let delta = Scalar::abs(num_re.clone() - theta_re.clone())
                + Scalar::abs(num_im.clone() - theta_im.clone());
            theta_re = num_re;
            theta_im = num_im;

            let scale = T::one() + Scalar::abs(theta_re.clone()) + Scalar::abs(theta_im.clone());
            if delta <= conv_tol.clone() * scale {
                break;
            }
        }

        (y_re, y_im, theta_re, theta_im)
    }

    /// Normalise a complex vector stored as `(real, imag)` in place to unit 2-norm.
    pub(super) fn normalize_complex(re: &mut [T], im: &mut [T]) {
        let mut nrm_sq = T::zero();
        for i in 0..re.len() {
            nrm_sq = nrm_sq + re[i].clone() * re[i].clone() + im[i].clone() * im[i].clone();
        }
        let nrm = Real::sqrt(nrm_sq);
        if nrm > <T as Scalar>::epsilon() {
            for i in 0..re.len() {
                re[i] = re[i].clone() / nrm.clone();
                im[i] = im[i].clone() / nrm.clone();
            }
        }
    }

    /// Solve the complex linear system `(H - (re + i*im) I) x = b` for the small dense
    /// (upper-Hessenberg) matrix `H`, with `b` given as `(b_re, b_im)`. Uses Gaussian
    /// elimination with partial pivoting; a tiny regulariser guards a (near-)singular
    /// pivot, which is expected here because `re + i*im` is an approximate eigenvalue.
    pub(super) fn complex_hessenberg_solve(
        h: &[Vec<T>],
        m: usize,
        re: T,
        im: T,
        b_re: &[T],
        b_im: &[T],
    ) -> (Vec<T>, Vec<T>) {
        let mut ar: Vec<Vec<T>> = (0..m).map(|i| h[i][..m].to_vec()).collect();
        let mut ai: Vec<Vec<T>> = (0..m).map(|_| vec![T::zero(); m]).collect();

        let mut scale = T::one();
        for row in ar.iter().take(m) {
            for val in row.iter().take(m) {
                let a = Scalar::abs(val.clone());
                if a > scale {
                    scale = a;
                }
            }
        }
        let reg = <T as Scalar>::epsilon() * scale;

        for i in 0..m {
            ar[i][i] = ar[i][i].clone() - re.clone();
            ai[i][i] = ai[i][i].clone() - im.clone();
        }

        let mut xr = b_re.to_vec();
        let mut xi = b_im.to_vec();

        // Forward elimination with partial pivoting by complex magnitude.
        for k in 0..m {
            let mut piv = k;
            let mut best = Self::cmag(ar[k][k].clone(), ai[k][k].clone());
            for i in k + 1..m {
                let mg = Self::cmag(ar[i][k].clone(), ai[i][k].clone());
                if mg > best {
                    best = mg;
                    piv = i;
                }
            }
            if piv != k {
                ar.swap(k, piv);
                ai.swap(k, piv);
                xr.swap(k, piv);
                xi.swap(k, piv);
            }
            if Self::cmag(ar[k][k].clone(), ai[k][k].clone()) <= reg {
                ar[k][k] = ar[k][k].clone() + reg.clone();
            }
            let dkr = ar[k][k].clone();
            let dki = ai[k][k].clone();
            for i in k + 1..m {
                let (lr, li) =
                    Self::cdiv(ar[i][k].clone(), ai[i][k].clone(), dkr.clone(), dki.clone());
                ar[i][k] = T::zero();
                ai[i][k] = T::zero();
                for j in k + 1..m {
                    let (pr, pi) =
                        Self::cmul(lr.clone(), li.clone(), ar[k][j].clone(), ai[k][j].clone());
                    ar[i][j] = ar[i][j].clone() - pr;
                    ai[i][j] = ai[i][j].clone() - pi;
                }
                let (pr, pi) = Self::cmul(lr, li, xr[k].clone(), xi[k].clone());
                xr[i] = xr[i].clone() - pr;
                xi[i] = xi[i].clone() - pi;
            }
        }

        // Back substitution.
        let mut out_r = vec![T::zero(); m];
        let mut out_i = vec![T::zero(); m];
        for k in (0..m).rev() {
            let mut sr = xr[k].clone();
            let mut si = xi[k].clone();
            for j in k + 1..m {
                let (pr, pi) = Self::cmul(
                    ar[k][j].clone(),
                    ai[k][j].clone(),
                    out_r[j].clone(),
                    out_i[j].clone(),
                );
                sr = sr - pr;
                si = si - pi;
            }
            let (qr, qi) = Self::cdiv(sr, si, ar[k][k].clone(), ai[k][k].clone());
            out_r[k] = qr;
            out_i[k] = qi;
        }

        (out_r, out_i)
    }

    /// Complex multiplication `(ar + i*ai) * (br + i*bi)`.
    pub(super) fn cmul(ar: T, ai: T, br: T, bi: T) -> (T, T) {
        (
            ar.clone() * br.clone() - ai.clone() * bi.clone(),
            ar * bi + ai * br,
        )
    }

    /// Complex division `(ar + i*ai) / (br + i*bi)`.
    pub(super) fn cdiv(ar: T, ai: T, br: T, bi: T) -> (T, T) {
        let denom = br.clone() * br.clone() + bi.clone() * bi.clone();
        (
            (ar.clone() * br.clone() + ai.clone() * bi.clone()) / denom.clone(),
            (ai * br - ar * bi) / denom,
        )
    }

    /// Complex magnitude `|ar + i*ai|`.
    pub(super) fn cmag(ar: T, ai: T) -> T {
        Real::sqrt(ar.clone() * ar.clone() + ai.clone() * ai.clone())
    }
}
