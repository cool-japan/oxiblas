//! Dense eigenvalue kernels for the generalized eigenvalue solver.
//!
//! These are the small dense linear-algebra primitives used by
//! [`GeneralizedEigen`](super::GeneralizedEigen): the Francis double-shift QR for real
//! upper Hessenberg matrices, the symmetric-tridiagonal QL eigenvalue solver with
//! inverse-iteration eigenvectors, implicit-restart shift application, and the shifted
//! Hessenberg/tridiagonal solves used by inverse iteration. They live in a dedicated
//! module purely to keep every source file within the workspace size policy.

use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::GeneralizedEigen;
use crate::csr::CsrMatrix;
use crate::linalg::eigenvalue::error::WhichEigenvalues;
use crate::linalg::eigenvalue::utils::{dot, givens_rotation, norm};
use crate::ops::spmv;

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> GeneralizedEigen<T> {
    /// Eigenvalues of a real upper Hessenberg matrix via the Francis double-shift QR
    /// algorithm (the classic `hqr` scheme).
    ///
    /// Returns `(real, imag)` parts. The double (implicit) shift keeps all arithmetic
    /// real while still resolving complex-conjugate eigenvalue pairs, which are read off
    /// the deflated 2x2 blocks. A single real shift cannot do this (it stagnates on
    /// complex spectra), which is why a full double-shift sweep is required here so that
    /// imaginary parts are never dropped.
    pub(super) fn solve_hessenberg_eigenvalues(&self, h: &[Vec<T>], n: usize) -> (Vec<T>, Vec<T>) {
        if n == 0 {
            return (vec![], vec![]);
        }

        // Dense working copy of the active n x n leading block.
        let mut a: Vec<Vec<T>> = (0..n)
            .map(|i| {
                let mut row = vec![T::zero(); n];
                for (j, cell) in row.iter_mut().enumerate() {
                    if i < h.len() && j < h[i].len() {
                        *cell = h[i][j].clone();
                    }
                }
                row
            })
            .collect();

        let mut wr = vec![T::zero(); n];
        let mut wi = vec![T::zero(); n];

        let eps = <T as Scalar>::epsilon();
        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let c075 = T::from_f64(0.75).unwrap_or_else(T::one);
        let c04375 = T::from_f64(0.4375).unwrap_or_else(T::one);

        // sign(mag, s) = |mag| with the sign of s.
        let signed = |mag: T, s: &T| -> T {
            if *s >= T::zero() {
                Scalar::abs(mag)
            } else {
                T::zero() - Scalar::abs(mag)
            }
        };

        // Norm of the Hessenberg part (used as a floor in the deflation test).
        let mut anorm = T::zero();
        for i in 0..n {
            let jstart = i.saturating_sub(1);
            for j in jstart..n {
                anorm = anorm + Scalar::abs(a[i][j].clone());
            }
        }

        let mut nn: isize = n as isize - 1;
        let mut t = T::zero();
        let mut its = 0usize;

        while nn >= 0 {
            // Search for a negligible subdiagonal, isolating the active block [l, nn].
            let mut l: isize = nn;
            while l >= 1 {
                let li = l as usize;
                let mut s =
                    Scalar::abs(a[li - 1][li - 1].clone()) + Scalar::abs(a[li][li].clone());
                if s <= eps.clone() * anorm.clone() {
                    s = anorm.clone();
                }
                if Scalar::abs(a[li][li - 1].clone()) <= eps.clone() * s {
                    a[li][li - 1] = T::zero();
                    break;
                }
                l -= 1;
            }
            if l < 0 {
                l = 0;
            }

            let nu = nn as usize;
            let x = a[nu][nu].clone();

            if l == nn {
                // One real eigenvalue has converged.
                wr[nu] = x + t.clone();
                wi[nu] = T::zero();
                nn -= 1;
                its = 0;
            } else if l == nn - 1 {
                // A 2x2 block has converged: two real or a complex-conjugate pair.
                let y = a[nu - 1][nu - 1].clone();
                let w = a[nu][nu - 1].clone() * a[nu - 1][nu].clone();
                let p = (y - x.clone()) / two.clone();
                let q = p.clone() * p.clone() + w.clone();
                let z = Real::sqrt(Scalar::abs(q.clone()));
                let x2 = x + t.clone();
                if q >= T::zero() {
                    let zz = p.clone() + signed(z, &p);
                    wr[nu - 1] = x2.clone() + zz.clone();
                    wr[nu] = if Scalar::abs(zz.clone()) > eps.clone() {
                        x2 - w / zz
                    } else {
                        x2 + zz
                    };
                    wi[nu - 1] = T::zero();
                    wi[nu] = T::zero();
                } else {
                    wr[nu - 1] = x2.clone() + p.clone();
                    wr[nu] = x2 + p;
                    wi[nu - 1] = T::zero() - z.clone();
                    wi[nu] = z;
                }
                nn -= 2;
                its = 0;
            } else {
                // No deflation yet. If we have stalled, force-extract the trailing 2x2
                // block to guarantee termination (rare: exceptional shifts are applied
                // at its == 10 and 20).
                if its >= 30 {
                    let y = a[nu - 1][nu - 1].clone();
                    let w = a[nu][nu - 1].clone() * a[nu - 1][nu].clone();
                    let p = (y - x.clone()) / two.clone();
                    let q = p.clone() * p.clone() + w.clone();
                    let z = Real::sqrt(Scalar::abs(q.clone()));
                    let x2 = x + t.clone();
                    if q >= T::zero() {
                        let zz = p.clone() + signed(z, &p);
                        wr[nu - 1] = x2.clone() + zz.clone();
                        wr[nu] = if Scalar::abs(zz.clone()) > eps.clone() {
                            x2 - w / zz
                        } else {
                            x2 + zz
                        };
                        wi[nu - 1] = T::zero();
                        wi[nu] = T::zero();
                    } else {
                        wr[nu - 1] = x2.clone() + p.clone();
                        wr[nu] = x2 + p;
                        wi[nu - 1] = T::zero() - z.clone();
                        wi[nu] = z;
                    }
                    nn -= 2;
                    its = 0;
                    continue;
                }

                // Form the (double) shift from the trailing 2x2 block.
                let mut xx = x.clone();
                let mut yy = a[nu - 1][nu - 1].clone();
                let mut ww = a[nu][nu - 1].clone() * a[nu - 1][nu].clone();

                if its == 10 || its == 20 {
                    // Exceptional shift to break stagnation.
                    t = t.clone() + xx.clone();
                    for i in 0..=nu {
                        a[i][i] = a[i][i].clone() - xx.clone();
                    }
                    let s = Scalar::abs(a[nu][nu - 1].clone())
                        + Scalar::abs(a[nu - 1][nu - 2].clone());
                    xx = c075.clone() * s.clone();
                    yy = xx.clone();
                    ww = T::zero() - c04375.clone() * s.clone() * s.clone();
                }
                its += 1;

                // Locate the start `m` of the bulge: two consecutive small subdiagonals.
                let mut p = T::zero();
                let mut q = T::zero();
                let mut r = T::zero();
                let mut m: isize = nn - 2;
                while m >= l {
                    let mi = m as usize;
                    let z = a[mi][mi].clone();
                    let rr = xx.clone() - z.clone();
                    let ss = yy.clone() - z.clone();
                    p = (rr.clone() * ss.clone() - ww.clone()) / a[mi + 1][mi].clone()
                        + a[mi][mi + 1].clone();
                    q = a[mi + 1][mi + 1].clone() - z.clone() - rr - ss;
                    r = a[mi + 2][mi + 1].clone();
                    let s2 =
                        Scalar::abs(p.clone()) + Scalar::abs(q.clone()) + Scalar::abs(r.clone());
                    if s2 > eps.clone() {
                        p = p.clone() / s2.clone();
                        q = q.clone() / s2.clone();
                        r = r.clone() / s2.clone();
                    }
                    if m == l {
                        break;
                    }
                    let u = Scalar::abs(a[mi][mi - 1].clone())
                        * (Scalar::abs(q.clone()) + Scalar::abs(r.clone()));
                    let v = Scalar::abs(p.clone())
                        * (Scalar::abs(a[mi - 1][mi - 1].clone())
                            + Scalar::abs(z.clone())
                            + Scalar::abs(a[mi + 1][mi + 1].clone()));
                    if u <= eps.clone() * v {
                        break;
                    }
                    m -= 1;
                }
                if m < l {
                    m = l;
                }
                let mu = m as usize;

                // Clear the entries below the sub-subdiagonal in the bulge region.
                for i in (mu + 2)..=nu {
                    a[i][i - 2] = T::zero();
                    if i != mu + 2 {
                        a[i][i - 3] = T::zero();
                    }
                }

                // Chase the bulge with Householder reflections (columns k = m ..= nn-1).
                let mut k = mu;
                while k < nu {
                    if k != mu {
                        p = a[k][k - 1].clone();
                        q = a[k + 1][k - 1].clone();
                        r = if k + 2 <= nu {
                            a[k + 2][k - 1].clone()
                        } else {
                            T::zero()
                        };
                        xx = Scalar::abs(p.clone())
                            + Scalar::abs(q.clone())
                            + Scalar::abs(r.clone());
                        if xx > eps.clone() {
                            p = p.clone() / xx.clone();
                            q = q.clone() / xx.clone();
                            r = r.clone() / xx.clone();
                        }
                    }
                    let s = signed(
                        Real::sqrt(
                            p.clone() * p.clone() + q.clone() * q.clone() + r.clone() * r.clone(),
                        ),
                        &p,
                    );
                    if Scalar::abs(s.clone()) > eps.clone() {
                        if k == mu {
                            if l != m {
                                a[k][k - 1] = T::zero() - a[k][k - 1].clone();
                            }
                        } else {
                            a[k][k - 1] = T::zero() - s.clone() * xx.clone();
                        }
                        p = p.clone() + s.clone();
                        let px = p.clone() / s.clone();
                        let py = q.clone() / s.clone();
                        let pz = r.clone() / s.clone();
                        let qq = q.clone() / p.clone();
                        let rr2 = r.clone() / p.clone();

                        // Row transformation.
                        for j in k..=nu {
                            let mut pp = a[k][j].clone() + qq.clone() * a[k + 1][j].clone();
                            if k + 2 <= nu {
                                pp = pp.clone() + rr2.clone() * a[k + 2][j].clone();
                                a[k + 2][j] = a[k + 2][j].clone() - pp.clone() * pz.clone();
                            }
                            a[k + 1][j] = a[k + 1][j].clone() - pp.clone() * py.clone();
                            a[k][j] = a[k][j].clone() - pp.clone() * px.clone();
                        }

                        // Column transformation.
                        let mmin = if nu < k + 3 { nu } else { k + 3 };
                        for i in (l as usize)..=mmin {
                            let mut pp =
                                px.clone() * a[i][k].clone() + py.clone() * a[i][k + 1].clone();
                            if k + 2 <= nu {
                                pp = pp.clone() + pz.clone() * a[i][k + 2].clone();
                                a[i][k + 2] = a[i][k + 2].clone() - pp.clone() * rr2.clone();
                            }
                            a[i][k + 1] = a[i][k + 1].clone() - pp.clone() * qq.clone();
                            a[i][k] = a[i][k].clone() - pp.clone();
                        }
                    }
                    k += 1;
                }
            }
        }

        (wr, wi)
    }

    /// Select wanted Ritz values and the unwanted ones (used as implicit-restart shifts).
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

        let mut indexed: Vec<(usize, T)> = real
            .iter()
            .zip(imag.iter())
            .enumerate()
            .map(|(i, (re, im))| {
                let mag = Real::sqrt(re.clone() * re.clone() + im.clone() * im.clone());
                (i, mag)
            })
            .collect();

        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // Algebraic orderings compare the real part.
                let mut real_indexed: Vec<(usize, T)> =
                    real.iter().enumerate().map(|(i, r)| (i, r.clone())).collect();
                match self.config.which {
                    WhichEigenvalues::LargestAlgebraic => {
                        real_indexed.sort_by(|x, y| {
                            y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        real_indexed.sort_by(|x, y| {
                            x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)
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

    /// One explicit shifted QR step on the leading `m x m` block of an upper Hessenberg
    /// matrix `h` (updated in place to `Q^T (h - shift I) Q + shift I`), returning the
    /// `m x m` orthogonal factor `Q`.
    pub(super) fn hessenberg_shifted_qr_step(
        &self,
        h: &mut [Vec<T>],
        m: usize,
        shift: &T,
    ) -> Vec<Vec<T>> {
        let mut q: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();
        if m == 0 {
            return q;
        }

        for i in 0..m {
            h[i][i] = h[i][i].clone() - shift.clone();
        }

        // QR of (h - shift I): Givens rotations zeroing the subdiagonal.
        let mut rots: Vec<(T, T)> = Vec::with_capacity(m.saturating_sub(1));
        for k in 0..m.saturating_sub(1) {
            let (c, s, r) = givens_rotation(h[k][k].clone(), h[k + 1][k].clone());
            h[k][k] = r;
            h[k + 1][k] = T::zero();
            for j in k + 1..m {
                let a1 = h[k][j].clone();
                let a2 = h[k + 1][j].clone();
                h[k][j] = c.clone() * a1.clone() + s.clone() * a2.clone();
                h[k + 1][j] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
            rots.push((c, s));
        }

        // Form R*Q by applying the rotations from the right, accumulating Q.
        for (k, (c, s)) in rots.into_iter().enumerate() {
            for row in h.iter_mut().take(m) {
                let a1 = row[k].clone();
                let a2 = row[k + 1].clone();
                row[k] = c.clone() * a1.clone() + s.clone() * a2.clone();
                row[k + 1] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
            for qrow in q.iter_mut().take(m) {
                let a1 = qrow[k].clone();
                let a2 = qrow[k + 1].clone();
                qrow[k] = c.clone() * a1.clone() + s.clone() * a2.clone();
                qrow[k + 1] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
        }

        for i in 0..m {
            h[i][i] = h[i][i].clone() + shift.clone();
        }

        q
    }

    /// Apply implicit QR shifts to the B-orthonormal Arnoldi factorization.
    ///
    /// Each unwanted Ritz value is applied as an exact shift in a full QR step,
    /// accumulating the orthogonal factor `Q` so that `H <- Q^T H Q` and `V <- V Q`. The
    /// Givens rotations are orthogonal in the coefficient space, so a B-orthonormal basis
    /// stays B-orthonormal. The kept basis is re-B-orthonormalized to counter the
    /// orthogonality drift that otherwise accumulates across restarts, and the compressed
    /// residual is formed with the Sorensen update
    /// `f^+ = H[nev][nev-1] * v_{nev} + Q[m-1][nev-1] * f` (then re-orthogonalized against
    /// the kept basis). Dropping the second term corrupts the factorization and produces
    /// false convergence.
    pub(super) fn apply_implicit_qr_shifts_b(
        &self,
        h: &mut [Vec<T>],
        v: &mut [Vec<T>],
        f: &mut [T],
        shifts_real: &[T],
        _shifts_imag: &[T],
        nev: usize,
        ncv: usize,
        b: &CsrMatrix<T>,
    ) {
        let n = v[0].len();
        let m0 = v.len().min(ncv);
        if m0 < 2 {
            return;
        }

        let f_old: Vec<T> = f.to_vec();

        // Accumulate the orthogonal factor Q = Q_1 Q_2 ... Q_p over all shifts.
        let mut q_total: Vec<Vec<T>> = (0..m0)
            .map(|i| {
                let mut row = vec![T::zero(); m0];
                row[i] = T::one();
                row
            })
            .collect();

        for shift in shifts_real {
            let q_step = self.hessenberg_shifted_qr_step(h, m0, shift);
            let mut q_new = vec![vec![T::zero(); m0]; m0];
            for i in 0..m0 {
                for kk in 0..m0 {
                    let qik = q_total[i][kk].clone();
                    if Scalar::abs(qik.clone()) <= <T as Scalar>::epsilon() {
                        continue;
                    }
                    for j in 0..m0 {
                        q_new[i][j] = q_new[i][j].clone() + qik.clone() * q_step[kk][j].clone();
                    }
                }
            }
            q_total = q_new;
        }

        // V <- V Q (recombine the first m0 Arnoldi vectors).
        let mut v_new: Vec<Vec<T>> = vec![vec![T::zero(); n]; m0];
        for (kk, vk) in v.iter().enumerate().take(m0) {
            for jj in 0..m0 {
                let coef = q_total[kk][jj].clone();
                if Scalar::abs(coef.clone()) <= <T as Scalar>::epsilon() {
                    continue;
                }
                for i in 0..n {
                    v_new[jj][i] = v_new[jj][i].clone() + coef.clone() * vk[i].clone();
                }
            }
        }
        for jj in 0..m0 {
            v[jj].clone_from(&v_new[jj]);
        }

        // Compressed residual (Sorensen), then re-B-orthogonalized against the kept basis
        // to remove any component that MGS rounding leaves along the retained vectors.
        if nev < v.len() {
            let h_sub = if nev < h.len() && nev > 0 && nev - 1 < h[nev].len() {
                h[nev][nev - 1].clone()
            } else {
                T::zero()
            };
            let sigma = if nev >= 1 && nev - 1 < m0 {
                q_total[m0 - 1][nev - 1].clone()
            } else {
                T::zero()
            };
            for i in 0..n {
                f[i] = h_sub.clone() * v[nev][i].clone() + sigma.clone() * f_old[i].clone();
            }
            for kk in 0..nev {
                let mut bvk = vec![T::zero(); n];
                spmv(T::one(), b, &v[kk], T::zero(), &mut bvk);
                let coef = dot(f, &bvk);
                for i in 0..n {
                    f[i] = f[i].clone() - coef.clone() * v[kk][i].clone();
                }
            }
        }
    }

    /// Approximate eigenvector of an upper Hessenberg matrix H for Ritz index `wi`.
    ///
    /// For a real Ritz value, two steps of (real) inverse iteration on `H - mu I` yield an
    /// accurate real eigenvector. For a complex Ritz value the eigenvector is genuinely
    /// complex and cannot be stored in a real vector, so a coordinate vector is used as a
    /// representative; the associated residual is then reported honestly.
    pub(super) fn hessenberg_eigenvector(
        &self,
        h: &[Vec<T>],
        m: usize,
        ritz_real: &[T],
        ritz_imag: &[T],
        wi: usize,
    ) -> Vec<T> {
        if wi >= ritz_imag.len() || Scalar::abs(ritz_imag[wi].clone()) > <T as Scalar>::epsilon() {
            let mut y = vec![T::zero(); m];
            if wi < m {
                y[wi] = T::one();
            }
            return y;
        }

        let mu = ritz_real[wi].clone();
        let perturb = <T as Scalar>::epsilon()
            * (Scalar::abs(mu.clone()) + T::one())
            * T::from_f64(10.0).unwrap_or_else(T::one);
        let shift = mu + perturb;

        let mut y = vec![T::one(); m];
        let mut y_norm = norm(&y);
        if y_norm > <T as Scalar>::epsilon() {
            for yi in &mut y {
                *yi = yi.clone() / y_norm.clone();
            }
        }

        for _ in 0..2 {
            y = self.solve_hessenberg_shifted(h, m, &shift, &y);
            y_norm = norm(&y);
            if y_norm > <T as Scalar>::epsilon() {
                for yi in &mut y {
                    *yi = yi.clone() / y_norm.clone();
                }
            }
        }

        y
    }

    /// Solve `(H - shift I) y = rhs` for an upper Hessenberg matrix H via a Givens QR
    /// factorization followed by back-substitution.
    pub(super) fn solve_hessenberg_shifted(&self, h: &[Vec<T>], m: usize, shift: &T, rhs: &[T]) -> Vec<T> {
        let mut a: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                for (j, cell) in row.iter_mut().enumerate() {
                    if i < h.len() && j < h[i].len() {
                        *cell = h[i][j].clone();
                    }
                }
                row[i] = row[i].clone() - shift.clone();
                row
            })
            .collect();

        let mut rhs_q = rhs.to_vec();

        for k in 0..m.saturating_sub(1) {
            let (c, s, r) = givens_rotation(a[k][k].clone(), a[k + 1][k].clone());
            a[k][k] = r;
            a[k + 1][k] = T::zero();
            for j in k + 1..m {
                let t = c.clone() * a[k][j].clone() + s.clone() * a[k + 1][j].clone();
                a[k + 1][j] =
                    T::zero() - s.clone() * a[k][j].clone() + c.clone() * a[k + 1][j].clone();
                a[k][j] = t;
            }
            let t = c.clone() * rhs_q[k].clone() + s.clone() * rhs_q[k + 1].clone();
            rhs_q[k + 1] =
                T::zero() - s.clone() * rhs_q[k].clone() + c.clone() * rhs_q[k + 1].clone();
            rhs_q[k] = t;
        }

        let eps = <T as Scalar>::epsilon();
        let mut y = vec![T::zero(); m];
        for i in (0..m).rev() {
            let mut sum = rhs_q[i].clone();
            for j in i + 1..m {
                sum = sum - a[i][j].clone() * y[j].clone();
            }
            let diag = if Scalar::abs(a[i][i].clone()) > eps.clone() {
                a[i][i].clone()
            } else if a[i][i].clone() >= T::zero() {
                eps.clone()
            } else {
                T::zero() - eps.clone()
            };
            y[i] = sum / diag;
        }

        y
    }

    /// Eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
    ///
    /// The eigenvalues are computed with the numerically stable implicit-QL algorithm
    /// (EISPACK `tql2` recurrence), and the eigenvectors are then recovered by inverse
    /// iteration on `(T - lambda I)`. Computing the vectors separately (rather than
    /// accumulating the QL rotations) keeps the eigenvectors consistent with the
    /// eigenvalues to working precision. The pairs are returned sorted by ascending
    /// eigenvalue, with `eigenvectors[k]` the eigenvector (in the tridiagonal basis) for
    /// `eigenvalues[k]`.
    pub(super) fn symmetric_tridiag_qr(&self, alpha: &[T], beta: &[T], m: usize) -> (Vec<T>, Vec<Vec<T>>) {
        if m == 0 {
            return (vec![], vec![]);
        }

        // --- Eigenvalues via implicit QL (only d, e are needed). ---
        let mut d = alpha[..m].to_vec();
        let mut e = if beta.len() >= m - 1 {
            beta[..m - 1].to_vec()
        } else {
            let mut e = beta.to_vec();
            e.resize(m.saturating_sub(1), T::zero());
            e
        };

        let max_qr_iter = 30 * m;
        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_qr_iter {
            // Locate the trailing unreduced block: `l` is its last index.
            let mut l = 0usize;
            for i in (0..m.saturating_sub(1)).rev() {
                if Scalar::abs(e[i].clone())
                    <= tol.clone() * (Scalar::abs(d[i].clone()) + Scalar::abs(d[i + 1].clone()))
                {
                    e[i] = T::zero();
                } else {
                    l = i + 1;
                    break;
                }
            }
            if l == 0 {
                break;
            }

            // Find the start `mstart` of that unreduced block.
            let mut mstart = l;
            for i in (0..l).rev() {
                if Scalar::abs(e[i].clone())
                    <= tol.clone() * (Scalar::abs(d[i].clone()) + Scalar::abs(d[i + 1].clone()))
                {
                    e[i] = T::zero();
                    mstart = i + 1;
                    break;
                }
                if i == 0 {
                    mstart = 0;
                }
            }

            // Wilkinson shift from the trailing 2x2 of the block.
            let pp = (d[l - 1].clone() - d[l].clone()) / two.clone();
            let rr = Real::sqrt(pp.clone() * pp.clone() + e[l - 1].clone() * e[l - 1].clone());
            let shift = if pp >= T::zero() {
                d[l].clone() - e[l - 1].clone() * e[l - 1].clone() / (pp.clone() + rr)
            } else {
                d[l].clone() - e[l - 1].clone() * e[l - 1].clone() / (pp.clone() - rr)
            };

            // Implicit QL sweep.
            let mut g = d[mstart].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();
            let mut p_val = T::zero();

            for i in mstart..l {
                let f_val = s.clone() * e[i].clone();
                let bb = c.clone() * e[i].clone();

                if Scalar::abs(f_val.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f_val.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > mstart {
                        e[i - 1] = f_val.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c.clone() * s.clone();
                } else {
                    s = f_val.clone() / g.clone();
                    let r = Real::sqrt(s.clone() * s.clone() + T::one());
                    if i > mstart {
                        e[i - 1] = g.clone() * r.clone();
                    }
                    c = T::one() / r.clone();
                    s = s.clone() * c.clone();
                }

                g = d[i].clone() - p_val.clone();
                let r = (d[i + 1].clone() - g.clone()) * s.clone()
                    + two.clone() * c.clone() * bb.clone();
                p_val = s.clone() * r.clone();
                d[i] = g.clone() + p_val.clone();
                g = c.clone() * r.clone() - bb.clone();
            }

            d[l] = d[l].clone() - p_val;
            e[l - 1] = g;
        }

        // Sort eigenvalues ascending.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvalues: Vec<T> = order.iter().map(|&i| d[i].clone()).collect();

        // --- Eigenvectors via inverse iteration on the tridiagonal T. ---
        let mut tri: Vec<Vec<T>> = vec![vec![T::zero(); m]; m];
        for i in 0..m {
            tri[i][i] = alpha[i].clone();
            if i > 0 && i - 1 < beta.len() {
                tri[i][i - 1] = beta[i - 1].clone();
            }
            if i + 1 < m && i < beta.len() {
                tri[i][i + 1] = beta[i].clone();
            }
        }

        let ritz_imag = vec![T::zero(); 1];
        let eigenvectors: Vec<Vec<T>> = eigenvalues
            .iter()
            .map(|lambda| {
                let ritz_real = vec![lambda.clone()];
                let mut y = self.hessenberg_eigenvector(&tri, m, &ritz_real, &ritz_imag, 0);
                // Normalize in the standard 2-norm.
                let yn = norm(&y);
                if yn > <T as Scalar>::epsilon() {
                    for yi in &mut y {
                        *yi = yi.clone() / yn.clone();
                    }
                }
                y
            })
            .collect();

        (eigenvalues, eigenvectors)
    }

    /// One shifted QR step on a symmetric tridiagonal matrix, returning the accumulated
    /// orthogonal factor `Q`, together with the new diagonal and off-diagonal.
    ///
    /// The step forms `T - shift*I = Q R` (Givens QR) and then `T_new = R Q + shift*I`,
    /// which is orthogonally similar to `T` and preserves the (symmetric) tridiagonal
    /// structure. `Q` is exactly the orthogonal factor, so `V_new = V Q` keeps the
    /// (B-orthonormal) Krylov basis consistent during an implicit restart.
    pub(super) fn tridiag_qr_step(alpha: &[T], beta: &[T], shift: T) -> (Vec<Vec<T>>, Vec<T>, Vec<T>) {
        let m = alpha.len();
        if m == 0 {
            return (vec![], vec![], vec![]);
        }

        // Dense symmetric tridiagonal T - shift*I.
        let mut t: Vec<Vec<T>> = vec![vec![T::zero(); m]; m];
        for i in 0..m {
            t[i][i] = alpha[i].clone() - shift.clone();
            if i > 0 && i - 1 < beta.len() {
                t[i][i - 1] = beta[i - 1].clone();
                t[i - 1][i] = beta[i - 1].clone();
            }
        }

        // Q accumulator (identity).
        let mut q: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();

        // Givens QR: zero the subdiagonal, remembering each rotation.
        let mut rots: Vec<(T, T)> = Vec::with_capacity(m.saturating_sub(1));
        for k in 0..m.saturating_sub(1) {
            let (c, s, r) = givens_rotation(t[k][k].clone(), t[k + 1][k].clone());
            t[k][k] = r;
            t[k + 1][k] = T::zero();
            for j in k + 1..m {
                let a1 = t[k][j].clone();
                let a2 = t[k + 1][j].clone();
                t[k][j] = c.clone() * a1.clone() + s.clone() * a2.clone();
                t[k + 1][j] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
            rots.push((c, s));
        }

        // Form R*Q by applying the rotations from the right, accumulating Q.
        for (k, (c, s)) in rots.into_iter().enumerate() {
            for row in t.iter_mut() {
                let a1 = row[k].clone();
                let a2 = row[k + 1].clone();
                row[k] = c.clone() * a1.clone() + s.clone() * a2.clone();
                row[k + 1] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
            for qrow in q.iter_mut() {
                let a1 = qrow[k].clone();
                let a2 = qrow[k + 1].clone();
                qrow[k] = c.clone() * a1.clone() + s.clone() * a2.clone();
                qrow[k + 1] = T::zero() - s.clone() * a1 + c.clone() * a2;
            }
        }

        // Restore the shift on the diagonal.
        for i in 0..m {
            t[i][i] = t[i][i].clone() + shift.clone();
        }

        let new_alpha: Vec<T> = (0..m).map(|i| t[i][i].clone()).collect();
        let new_beta: Vec<T> = (0..m.saturating_sub(1))
            .map(|i| t[i + 1][i].clone())
            .collect();

        (q, new_alpha, new_beta)
    }
}
