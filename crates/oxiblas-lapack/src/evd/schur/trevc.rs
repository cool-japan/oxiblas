//! Eigenvectors of a real upper quasi-triangular Schur factor (LAPACK `DTREVC`).
//!
//! Layout of the returned eigenvector matrices (same as LAPACK): column `k`
//! belongs to the eigenvalue at diagonal position `k`. A real eigenvalue owns a
//! real eigenvector column. A 2×2 diagonal block at rows/columns `(k, k + 1)`
//! holds a complex-conjugate pair `wr ± i·wi` (`wi > 0`); columns `k` and
//! `k + 1` hold the real and imaginary parts of the eigenvector `x` of
//! `wr + i·wi`, and the eigenvector of `wr − i·wi` is `conj(x)`.
//!
//! - Right eigenvectors satisfy `T·x = λ·x`.
//! - Left eigenvectors satisfy `yᴴ·T = λ·yᴴ` (for a real `T`, `Tᵀ·y = conj(λ)·y`).
//!
//! Near-singular diagonal pivots are perturbed to `smin = max(ulp·|λ|,
//! smlnum)` and the working vector is rescaled whenever the solution would
//! otherwise overflow, so defective and nearly defective matrices yield finite
//! eigenvectors with small residuals.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::qr::{big_number, cst, lanv2, small_number};

/// Returns `true` when rows/columns `(j, j + 1)` of `t` form a 2×2 diagonal
/// block, i.e. `t[j + 1, j]` is non-zero.
#[inline]
pub(crate) fn starts_block<T: Scalar>(t: &Mat<T>, j: usize) -> bool {
    j + 1 < t.nrows() && t[(j + 1, j)] != T::zero()
}

/// Result of [`laln2`]: `x[row][0]` / `x[row][1]` are the real / imaginary parts.
struct Solve2<T> {
    x: [[T; 2]; 2],
    scale: T,
    xnorm: T,
}

/// Robust complex division `(a + i·b) / (c + i·d)` (Smith's algorithm).
#[inline]
pub(crate) fn ladiv<T: Real>(a: T, b: T, c: T, d: T) -> (T, T) {
    if Scalar::abs(d) <= Scalar::abs(c) {
        let e = d / c;
        let f = c + d * e;
        ((a + b * e) / f, (b - a * e) / f)
    } else {
        let e = c / d;
        let f = d + c * e;
        ((b + a * e) / f, (b * e - a) / f)
    }
}

/// Pivot permutations of the complete-pivoting 2×2 solve (LAPACK `DLALN2`).
const IPIVOT: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
const RSWAP: [bool; 4] = [false, true, false, true];
const ZSWAP: [bool; 4] = [false, false, true, true];

/// Solves `(A − w·I)·X = scale·B` (or `(Aᵀ − w·I)·X = scale·B` when `ltrans`)
/// for a real `na × na` block `A` (`na ∈ {1, 2}`) and a real (`nw = 1`) or
/// complex (`nw = 2`, `w = wr + i·wi`) shift (LAPACK `DLALN2` with
/// `ca = d1 = d2 = 1`).
///
/// `A − w·I` is perturbed so that its smallest singular value is at least
/// `smin`, and `scale ≤ 1` is chosen so that `X` cannot overflow.
#[allow(clippy::too_many_arguments)]
fn laln2<T: Field + Real>(
    ltrans: bool,
    na: usize,
    nw: usize,
    smin: T,
    a: [[T; 2]; 2],
    b: [[T; 2]; 2],
    wr: T,
    wi: T,
) -> Solve2<T> {
    let zero = T::zero();
    let one = T::one();
    let smlnum = cst::<T>(2.0) * <T as Scalar>::min_positive();
    let bignum = one / smlnum;
    let smini = Float::max(smin, smlnum);
    let mut scale = one;
    let mut x = [[zero; 2]; 2];

    if na == 1 {
        if nw == 1 {
            // Real 1×1 system.
            let mut csr = a[0][0] - wr;
            let mut cnorm = Scalar::abs(csr);
            if cnorm < smini {
                csr = smini;
                cnorm = smini;
            }
            let bnorm = Scalar::abs(b[0][0]);
            if cnorm < one && bnorm > one && bnorm > bignum * cnorm {
                scale = one / bnorm;
            }
            x[0][0] = (b[0][0] * scale) / csr;
            let xnorm = Scalar::abs(x[0][0]);
            return Solve2 { x, scale, xnorm };
        }
        // Complex 1×1 system.
        let mut csr = a[0][0] - wr;
        let mut csi = -wi;
        let mut cnorm = Scalar::abs(csr) + Scalar::abs(csi);
        if cnorm < smini {
            csr = smini;
            csi = zero;
            cnorm = smini;
        }
        let bnorm = Scalar::abs(b[0][0]) + Scalar::abs(b[0][1]);
        if cnorm < one && bnorm > one && bnorm > bignum * cnorm {
            scale = one / bnorm;
        }
        let (xr, xi) = ladiv(scale * b[0][0], scale * b[0][1], csr, csi);
        x[0][0] = xr;
        x[0][1] = xi;
        let xnorm = Scalar::abs(xr) + Scalar::abs(xi);
        return Solve2 { x, scale, xnorm };
    }

    // 2×2 system: real part of C = A − w·I (or Aᵀ − w·I), column-major.
    let (c21, c12) = if ltrans {
        (a[0][1], a[1][0])
    } else {
        (a[1][0], a[0][1])
    };
    let crv = [a[0][0] - wr, c21, c12, a[1][1] - wr];

    if nw == 1 {
        // Real 2×2 system: Gaussian elimination with complete pivoting.
        let mut cmax = zero;
        let mut icmax = 0usize;
        for (j, &c) in crv.iter().enumerate() {
            if Scalar::abs(c) > cmax {
                cmax = Scalar::abs(c);
                icmax = j;
            }
        }
        if cmax < smini {
            // Use smini·I.
            let bnorm = Float::max(Scalar::abs(b[0][0]), Scalar::abs(b[1][0]));
            if smini < one && bnorm > one && bnorm > bignum * smini {
                scale = one / bnorm;
            }
            let temp = scale / smini;
            x[0][0] = temp * b[0][0];
            x[1][0] = temp * b[1][0];
            return Solve2 {
                x,
                scale,
                xnorm: temp * bnorm,
            };
        }
        let piv = IPIVOT[icmax];
        let ur11 = crv[icmax];
        let cr21 = crv[piv[1]];
        let ur12 = crv[piv[2]];
        let cr22 = crv[piv[3]];
        let ur11r = one / ur11;
        let lr21 = ur11r * cr21;
        let mut ur22 = cr22 - ur12 * lr21;
        if Scalar::abs(ur22) < smini {
            ur22 = smini;
        }
        let (br1, br2) = if RSWAP[icmax] {
            (b[1][0], b[0][0])
        } else {
            (b[0][0], b[1][0])
        };
        let br2 = br2 - lr21 * br1;
        let bbnd = Float::max(Scalar::abs(br1 * (ur22 * ur11r)), Scalar::abs(br2));
        if bbnd > one && Scalar::abs(ur22) < one && bbnd >= bignum * Scalar::abs(ur22) {
            scale = one / bbnd;
        }
        let xr2 = (br2 * scale) / ur22;
        let xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
        if ZSWAP[icmax] {
            x[0][0] = xr2;
            x[1][0] = xr1;
        } else {
            x[0][0] = xr1;
            x[1][0] = xr2;
        }
        let mut xnorm = Float::max(Scalar::abs(xr1), Scalar::abs(xr2));
        // Further scaling if norm(A)·norm(X) > overflow.
        if xnorm > one && cmax > one && xnorm > bignum / cmax {
            let temp = cmax / bignum;
            x[0][0] = temp * x[0][0];
            x[1][0] = temp * x[1][0];
            xnorm = temp * xnorm;
            scale = temp * scale;
        }
        return Solve2 { x, scale, xnorm };
    }

    // Complex 2×2 system: imaginary part of C is −wi·I.
    let civ = [-wi, zero, zero, -wi];
    let mut cmax = zero;
    let mut icmax = 0usize;
    for j in 0..4 {
        let mag = Scalar::abs(crv[j]) + Scalar::abs(civ[j]);
        if mag > cmax {
            cmax = mag;
            icmax = j;
        }
    }
    if cmax < smini {
        // Use smini·I.
        let bnorm = Float::max(
            Scalar::abs(b[0][0]) + Scalar::abs(b[0][1]),
            Scalar::abs(b[1][0]) + Scalar::abs(b[1][1]),
        );
        if smini < one && bnorm > one && bnorm > bignum * smini {
            scale = one / bnorm;
        }
        let temp = scale / smini;
        for (xr, br) in x.iter_mut().zip(b.iter()) {
            xr[0] = temp * br[0];
            xr[1] = temp * br[1];
        }
        return Solve2 {
            x,
            scale,
            xnorm: temp * bnorm,
        };
    }
    let piv = IPIVOT[icmax];
    let ur11 = crv[icmax];
    let ui11 = civ[icmax];
    let cr21 = crv[piv[1]];
    let ci21 = civ[piv[1]];
    let ur12 = crv[piv[2]];
    let ui12 = civ[piv[2]];
    let cr22 = crv[piv[3]];
    let ci22 = civ[piv[3]];
    let (ur11r, ui11r, lr21, li21, ur12s, ui12s, mut ur22, mut ui22);
    if icmax == 0 || icmax == 3 {
        // The off-diagonal entries of the pivoted C are real.
        if Scalar::abs(ur11) > Scalar::abs(ui11) {
            let temp = ui11 / ur11;
            ur11r = one / (ur11 * (one + temp * temp));
            ui11r = -temp * ur11r;
        } else {
            let temp = ur11 / ui11;
            ui11r = -one / (ui11 * (one + temp * temp));
            ur11r = -temp * ui11r;
        }
        lr21 = cr21 * ur11r;
        li21 = cr21 * ui11r;
        ur12s = ur12 * ur11r;
        ui12s = ur12 * ui11r;
        ur22 = cr22 - ur12 * lr21;
        ui22 = ci22 - ur12 * li21;
    } else {
        // The diagonal entries of the pivoted C are real.
        ur11r = one / ur11;
        ui11r = zero;
        lr21 = cr21 * ur11r;
        li21 = ci21 * ur11r;
        ur12s = ur12 * ur11r;
        ui12s = ui12 * ur11r;
        ur22 = cr22 - ur12 * lr21 + ui12 * li21;
        ui22 = -ur12 * li21 - ui12 * lr21;
    }
    let u22abs = Scalar::abs(ur22) + Scalar::abs(ui22);
    if u22abs < smini {
        ur22 = smini;
        ui22 = zero;
    }
    let (mut br1, mut bi1, mut br2, mut bi2) = if RSWAP[icmax] {
        (b[1][0], b[1][1], b[0][0], b[0][1])
    } else {
        (b[0][0], b[0][1], b[1][0], b[1][1])
    };
    br2 = br2 - lr21 * br1 + li21 * bi1;
    bi2 = bi2 - li21 * br1 - lr21 * bi1;
    let bbnd = Float::max(
        (Scalar::abs(br1) + Scalar::abs(bi1))
            * (u22abs * (Scalar::abs(ur11r) + Scalar::abs(ui11r))),
        Scalar::abs(br2) + Scalar::abs(bi2),
    );
    if bbnd > one && u22abs < one && bbnd >= bignum * u22abs {
        scale = one / bbnd;
        br1 = scale * br1;
        bi1 = scale * bi1;
        br2 = scale * br2;
        bi2 = scale * bi2;
    }
    let (xr2, xi2) = ladiv(br2, bi2, ur22, ui22);
    let xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
    let xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
    if ZSWAP[icmax] {
        x = [[xr2, xi2], [xr1, xi1]];
    } else {
        x = [[xr1, xi1], [xr2, xi2]];
    }
    let mut xnorm = Float::max(
        Scalar::abs(xr1) + Scalar::abs(xi1),
        Scalar::abs(xr2) + Scalar::abs(xi2),
    );
    // Further scaling if norm(A)·norm(X) > overflow.
    if xnorm > one && cmax > one && xnorm > bignum / cmax {
        let temp = cmax / bignum;
        for row in x.iter_mut() {
            row[0] = temp * row[0];
            row[1] = temp * row[1];
        }
        xnorm = temp * xnorm;
        scale = temp * scale;
    }
    Solve2 { x, scale, xnorm }
}

/// The 2×2 diagonal block of `t` at rows/columns `(j, j + 1)`.
#[inline]
fn block<T: Scalar>(t: &Mat<T>, j: usize) -> [[T; 2]; 2] {
    [
        [t[(j, j)], t[(j, j + 1)]],
        [t[(j + 1, j)], t[(j + 1, j + 1)]],
    ]
}

/// Machine constants shared by the eigenvector solvers.
struct Constants<T> {
    ulp: T,
    smlnum: T,
    bignum: T,
}

impl<T: Field + Real> Constants<T> {
    fn new(n: usize) -> Self {
        let ulp = <T as Scalar>::epsilon();
        let smlnum = small_number::<T>(n);
        let bignum = big_number(smlnum);
        Self {
            ulp,
            smlnum,
            bignum,
        }
    }
}

/// 1-norms of the strictly upper triangular part of each column of `t`.
fn column_norms<T: Field + Real>(t: &Mat<T>) -> Vec<T> {
    let n = t.nrows();
    let mut norms = vec![T::zero(); n];
    for (j, norm) in norms.iter_mut().enumerate() {
        for i in 0..j {
            *norm = *norm + Scalar::abs(t[(i, j)]);
        }
    }
    norms
}

/// Imaginary part `wi > 0` of the complex pair held by the standardized 2×2
/// block at `(j, j + 1)`.
#[inline]
fn pair_imag<T: Field + Real>(t: &Mat<T>, j: usize) -> T {
    Real::sqrt(Scalar::abs(t[(j, j + 1)])) * Real::sqrt(Scalar::abs(t[(j + 1, j)]))
}

#[inline]
fn scale_slice<T: Field>(x: &mut [T], s: T) {
    for v in x.iter_mut() {
        *v = *v * s;
    }
}

/// Right eigenvectors of a standardized real Schur factor `t`, in the LAPACK
/// layout described in the module documentation. Each eigenvector is scaled so
/// that its largest component (by `|re| + |im|`) has magnitude 1; entries
/// below the eigenvalue's diagonal block are zero.
pub(crate) fn right_eigenvectors_raw<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    let n = t.nrows();
    let mut vr = Mat::zeros(n, n);
    if n == 0 {
        return vr;
    }
    let c = Constants::<T>::new(n);
    let cnorm = column_norms(t);
    let mut wre = vec![T::zero(); n];
    let mut wim = vec![T::zero(); n];

    let mut ki = n;
    while ki > 0 {
        ki -= 1;
        if ki > 0 && starts_block(t, ki - 1) {
            // Complex pair at (ki − 1, ki): eigenvalue wr + i·wi with wi > 0.
            let k0 = ki - 1;
            let wr = t[(ki, ki)];
            let wi = pair_imag(t, k0);
            let smin = Float::max(c.ulp * (Scalar::abs(wr) + Scalar::abs(wi)), c.smlnum);
            wre[..=ki].fill(T::zero());
            wim[..=ki].fill(T::zero());
            // Initial solve of the 2×2 block (T_block − (wr + i·wi))·x = 0.
            if Scalar::abs(t[(k0, ki)]) >= Scalar::abs(t[(ki, k0)]) {
                wre[k0] = T::one();
                wim[ki] = wi / t[(k0, ki)];
            } else {
                wre[k0] = -wi / t[(ki, k0)];
                wim[ki] = T::one();
            }
            // Right-hand side.
            for k in 0..k0 {
                wre[k] = -wre[k0] * t[(k, k0)];
                wim[k] = -wim[ki] * t[(k, ki)];
            }
            // Solve (T[0..k0, 0..k0] − (wr + i·wi))·x = scale·rhs.
            let mut j = k0;
            while j > 0 {
                j -= 1;
                if j > 0 && starts_block(t, j - 1) {
                    let j1 = j - 1;
                    let rhs = [[wre[j1], wim[j1]], [wre[j], wim[j]]];
                    let s = laln2(false, 2, 2, smin, block(t, j1), rhs, wr, wi);
                    let mut x = s.x;
                    let mut scale = s.scale;
                    if s.xnorm > T::one() {
                        let beta = Float::max(cnorm[j1], cnorm[j]);
                        if beta > c.bignum / s.xnorm {
                            let rec = T::one() / s.xnorm;
                            for row in x.iter_mut() {
                                row[0] = row[0] * rec;
                                row[1] = row[1] * rec;
                            }
                            scale = scale * rec;
                        }
                    }
                    if scale != T::one() {
                        scale_slice(&mut wre[..=ki], scale);
                        scale_slice(&mut wim[..=ki], scale);
                    }
                    wre[j1] = x[0][0];
                    wre[j] = x[1][0];
                    wim[j1] = x[0][1];
                    wim[j] = x[1][1];
                    for r in 0..j1 {
                        wre[r] = wre[r] - x[0][0] * t[(r, j1)] - x[1][0] * t[(r, j)];
                        wim[r] = wim[r] - x[0][1] * t[(r, j1)] - x[1][1] * t[(r, j)];
                    }
                    j = j1;
                } else {
                    let rhs = [[wre[j], wim[j]], [T::zero(), T::zero()]];
                    let a = [[t[(j, j)], T::zero()], [T::zero(), T::zero()]];
                    let s = laln2(false, 1, 2, smin, a, rhs, wr, wi);
                    let (mut xr, mut xi) = (s.x[0][0], s.x[0][1]);
                    let mut scale = s.scale;
                    if s.xnorm > T::one() && cnorm[j] > c.bignum / s.xnorm {
                        xr = xr / s.xnorm;
                        xi = xi / s.xnorm;
                        scale = scale / s.xnorm;
                    }
                    if scale != T::one() {
                        scale_slice(&mut wre[..=ki], scale);
                        scale_slice(&mut wim[..=ki], scale);
                    }
                    wre[j] = xr;
                    wim[j] = xi;
                    for r in 0..j {
                        wre[r] = wre[r] - xr * t[(r, j)];
                        wim[r] = wim[r] - xi * t[(r, j)];
                    }
                }
            }
            let mut emax = T::zero();
            for r in 0..=ki {
                emax = Float::max(emax, Scalar::abs(wre[r]) + Scalar::abs(wim[r]));
            }
            let remax = T::one() / emax;
            for r in 0..=ki {
                vr[(r, k0)] = wre[r] * remax;
                vr[(r, ki)] = wim[r] * remax;
            }
            ki = k0;
        } else {
            // Real eigenvalue at ki.
            let wr = t[(ki, ki)];
            let smin = Float::max(c.ulp * Scalar::abs(wr), c.smlnum);
            wre[ki] = T::one();
            for k in 0..ki {
                wre[k] = -t[(k, ki)];
            }
            // Solve (T[0..ki, 0..ki] − wr)·x = scale·rhs.
            let mut j = ki;
            while j > 0 {
                j -= 1;
                if j > 0 && starts_block(t, j - 1) {
                    let j1 = j - 1;
                    let rhs = [[wre[j1], T::zero()], [wre[j], T::zero()]];
                    let s = laln2(false, 2, 1, smin, block(t, j1), rhs, wr, T::zero());
                    let (mut x1, mut x2) = (s.x[0][0], s.x[1][0]);
                    let mut scale = s.scale;
                    if s.xnorm > T::one() {
                        let beta = Float::max(cnorm[j1], cnorm[j]);
                        if beta > c.bignum / s.xnorm {
                            x1 = x1 / s.xnorm;
                            x2 = x2 / s.xnorm;
                            scale = scale / s.xnorm;
                        }
                    }
                    if scale != T::one() {
                        scale_slice(&mut wre[..=ki], scale);
                    }
                    wre[j1] = x1;
                    wre[j] = x2;
                    for r in 0..j1 {
                        wre[r] = wre[r] - x1 * t[(r, j1)] - x2 * t[(r, j)];
                    }
                    j = j1;
                } else {
                    let rhs = [[wre[j], T::zero()], [T::zero(), T::zero()]];
                    let a = [[t[(j, j)], T::zero()], [T::zero(), T::zero()]];
                    let s = laln2(false, 1, 1, smin, a, rhs, wr, T::zero());
                    let mut x = s.x[0][0];
                    let mut scale = s.scale;
                    if s.xnorm > T::one() && cnorm[j] > c.bignum / s.xnorm {
                        x = x / s.xnorm;
                        scale = scale / s.xnorm;
                    }
                    if scale != T::one() {
                        scale_slice(&mut wre[..=ki], scale);
                    }
                    wre[j] = x;
                    for r in 0..j {
                        wre[r] = wre[r] - x * t[(r, j)];
                    }
                }
            }
            let mut emax = T::zero();
            for &w in &wre[..=ki] {
                emax = Float::max(emax, Scalar::abs(w));
            }
            let remax = T::one() / emax;
            for r in 0..=ki {
                vr[(r, ki)] = wre[r] * remax;
            }
        }
    }
    vr
}

/// Left eigenvectors of a standardized real Schur factor `t`, in the LAPACK
/// layout described in the module documentation (`yᴴ·T = λ·yᴴ`, where the
/// pair columns hold `y` for the eigenvalue with positive imaginary part).
/// Each eigenvector is scaled so that its largest component (by
/// `|re| + |im|`) has magnitude 1; entries above the eigenvalue's diagonal
/// block are zero.
pub(crate) fn left_eigenvectors_raw<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    let n = t.nrows();
    let mut vl = Mat::zeros(n, n);
    if n == 0 {
        return vl;
    }
    let c = Constants::<T>::new(n);
    let cnorm = column_norms(t);
    let mut wre = vec![T::zero(); n];
    let mut wim = vec![T::zero(); n];

    let mut ki = 0;
    while ki < n {
        if starts_block(t, ki) {
            // Complex pair at (ki, ki + 1): solve for y with
            // (Tᵀ − (wr − i·wi))·y = 0, i.e. yᴴ·T = (wr + i·wi)·yᴴ.
            let k1 = ki + 1;
            let wr = t[(ki, ki)];
            let wi = pair_imag(t, ki);
            let smin = Float::max(c.ulp * (Scalar::abs(wr) + Scalar::abs(wi)), c.smlnum);
            wre[ki..].fill(T::zero());
            wim[ki..].fill(T::zero());
            if Scalar::abs(t[(ki, k1)]) >= Scalar::abs(t[(k1, ki)]) {
                wre[ki] = wi / t[(ki, k1)];
                wim[k1] = T::one();
            } else {
                wre[ki] = T::one();
                wim[k1] = -wi / t[(k1, ki)];
            }
            for k in (ki + 2)..n {
                wre[k] = -wre[ki] * t[(ki, k)];
                wim[k] = -wim[k1] * t[(k1, k)];
            }
            let mut vmax = T::one();
            let mut vcrit = c.bignum;
            let mut j = ki + 2;
            while j < n {
                if starts_block(t, j) {
                    let j2 = j + 1;
                    let beta = Float::max(cnorm[j], cnorm[j2]);
                    if beta > vcrit {
                        let rec = T::one() / vmax;
                        scale_slice(&mut wre[ki..], rec);
                        scale_slice(&mut wim[ki..], rec);
                        vmax = T::one();
                    }
                    let (mut sr1, mut si1, mut sr2, mut si2) =
                        (T::zero(), T::zero(), T::zero(), T::zero());
                    for r in (ki + 2)..j {
                        sr1 = sr1 + t[(r, j)] * wre[r];
                        si1 = si1 + t[(r, j)] * wim[r];
                        sr2 = sr2 + t[(r, j2)] * wre[r];
                        si2 = si2 + t[(r, j2)] * wim[r];
                    }
                    let rhs = [[wre[j] - sr1, wim[j] - si1], [wre[j2] - sr2, wim[j2] - si2]];
                    let s = laln2(true, 2, 2, smin, block(t, j), rhs, wr, -wi);
                    if s.scale != T::one() {
                        scale_slice(&mut wre[ki..], s.scale);
                        scale_slice(&mut wim[ki..], s.scale);
                    }
                    wre[j] = s.x[0][0];
                    wim[j] = s.x[0][1];
                    wre[j2] = s.x[1][0];
                    wim[j2] = s.x[1][1];
                    for row in s.x.iter() {
                        vmax =
                            Float::max(vmax, Float::max(Scalar::abs(row[0]), Scalar::abs(row[1])));
                    }
                    vcrit = c.bignum / vmax;
                    j += 2;
                } else {
                    if cnorm[j] > vcrit {
                        let rec = T::one() / vmax;
                        scale_slice(&mut wre[ki..], rec);
                        scale_slice(&mut wim[ki..], rec);
                        vmax = T::one();
                    }
                    let (mut sr, mut si) = (T::zero(), T::zero());
                    for r in (ki + 2)..j {
                        sr = sr + t[(r, j)] * wre[r];
                        si = si + t[(r, j)] * wim[r];
                    }
                    let rhs = [[wre[j] - sr, wim[j] - si], [T::zero(), T::zero()]];
                    let a = [[t[(j, j)], T::zero()], [T::zero(), T::zero()]];
                    let s = laln2(false, 1, 2, smin, a, rhs, wr, -wi);
                    if s.scale != T::one() {
                        scale_slice(&mut wre[ki..], s.scale);
                        scale_slice(&mut wim[ki..], s.scale);
                    }
                    wre[j] = s.x[0][0];
                    wim[j] = s.x[0][1];
                    vmax = Float::max(
                        vmax,
                        Float::max(Scalar::abs(s.x[0][0]), Scalar::abs(s.x[0][1])),
                    );
                    vcrit = c.bignum / vmax;
                    j += 1;
                }
            }
            let mut emax = T::zero();
            for r in ki..n {
                emax = Float::max(emax, Scalar::abs(wre[r]) + Scalar::abs(wim[r]));
            }
            let remax = T::one() / emax;
            for r in ki..n {
                vl[(r, ki)] = wre[r] * remax;
                vl[(r, k1)] = wim[r] * remax;
            }
            ki += 2;
        } else {
            // Real eigenvalue at ki: solve (T[ki+1.., ki+1..] − wr)ᵀ·y = rhs.
            let wr = t[(ki, ki)];
            let smin = Float::max(c.ulp * Scalar::abs(wr), c.smlnum);
            wre[ki] = T::one();
            for k in (ki + 1)..n {
                wre[k] = -t[(ki, k)];
            }
            let mut vmax = T::one();
            let mut vcrit = c.bignum;
            let mut j = ki + 1;
            while j < n {
                if starts_block(t, j) {
                    let j2 = j + 1;
                    let beta = Float::max(cnorm[j], cnorm[j2]);
                    if beta > vcrit {
                        let rec = T::one() / vmax;
                        scale_slice(&mut wre[ki..], rec);
                        vmax = T::one();
                    }
                    let (mut s1, mut s2) = (T::zero(), T::zero());
                    for r in (ki + 1)..j {
                        s1 = s1 + t[(r, j)] * wre[r];
                        s2 = s2 + t[(r, j2)] * wre[r];
                    }
                    let rhs = [[wre[j] - s1, T::zero()], [wre[j2] - s2, T::zero()]];
                    let s = laln2(true, 2, 1, smin, block(t, j), rhs, wr, T::zero());
                    if s.scale != T::one() {
                        scale_slice(&mut wre[ki..], s.scale);
                    }
                    wre[j] = s.x[0][0];
                    wre[j2] = s.x[1][0];
                    vmax = Float::max(
                        vmax,
                        Float::max(Scalar::abs(s.x[0][0]), Scalar::abs(s.x[1][0])),
                    );
                    vcrit = c.bignum / vmax;
                    j += 2;
                } else {
                    if cnorm[j] > vcrit {
                        let rec = T::one() / vmax;
                        scale_slice(&mut wre[ki..], rec);
                        vmax = T::one();
                    }
                    let mut s1 = T::zero();
                    for r in (ki + 1)..j {
                        s1 = s1 + t[(r, j)] * wre[r];
                    }
                    let rhs = [[wre[j] - s1, T::zero()], [T::zero(), T::zero()]];
                    let a = [[t[(j, j)], T::zero()], [T::zero(), T::zero()]];
                    let s = laln2(false, 1, 1, smin, a, rhs, wr, T::zero());
                    if s.scale != T::one() {
                        scale_slice(&mut wre[ki..], s.scale);
                    }
                    wre[j] = s.x[0][0];
                    vmax = Float::max(vmax, Scalar::abs(s.x[0][0]));
                    vcrit = c.bignum / vmax;
                    j += 1;
                }
            }
            let mut emax = T::zero();
            for &w in &wre[ki..] {
                emax = Float::max(emax, Scalar::abs(w));
            }
            let remax = T::one() / emax;
            for r in ki..n {
                vl[(r, ki)] = wre[r] * remax;
            }
            ki += 1;
        }
    }
    vl
}

/// Plane rotations `(k, cs, sn)` that bring a quasi-triangular `t` into
/// standardized Schur form: `t = G·t_std·Gᵀ` with `G` the product of the
/// rotations `[cs −sn; sn cs]` acting on rows/columns `(k, k + 1)`.
pub(crate) type BlockRotations<T> = Vec<(usize, T, T)>;

/// Returns a standardized copy of the quasi-triangular matrix `t` together
/// with the rotations applied, or `None` when `t` is already standardized
/// (every 2×2 block has equal diagonal entries and off-diagonal entries of
/// opposite sign).
///
/// Blocks are detected by a non-zero subdiagonal entry; a non-standard block
/// holding two real eigenvalues is split into upper triangular form.
pub(crate) fn standardize<T: Field + Real + bytemuck::Zeroable>(
    t: &Mat<T>,
) -> Option<(Mat<T>, BlockRotations<T>)> {
    let n = t.nrows();
    let is_standard =
        |j: usize| t[(j, j)] == t[(j + 1, j + 1)] && t[(j, j + 1)] * t[(j + 1, j)] < T::zero();
    let mut needs_work = false;
    let mut j = 0;
    while j < n {
        if starts_block(t, j) {
            if !is_standard(j) {
                needs_work = true;
                break;
            }
            j += 2;
        } else {
            j += 1;
        }
    }
    if !needs_work {
        return None;
    }

    let mut ts = Mat::zeros(n, n);
    for jj in 0..n {
        for i in 0..n {
            ts[(i, jj)] = t[(i, jj)];
        }
    }
    let mut rotations = Vec::new();
    let mut j = 0;
    while j < n {
        if !starts_block(&ts, j) {
            j += 1;
            continue;
        }
        let s = lanv2(
            ts[(j, j)],
            ts[(j, j + 1)],
            ts[(j + 1, j)],
            ts[(j + 1, j + 1)],
        );
        ts[(j, j)] = s.a;
        ts[(j, j + 1)] = s.b;
        ts[(j + 1, j)] = s.c;
        ts[(j + 1, j + 1)] = s.d;
        if !(s.cs == T::one() && s.sn == T::zero()) {
            for col in (j + 2)..n {
                let x = ts[(j, col)];
                let y = ts[(j + 1, col)];
                ts[(j, col)] = s.cs * x + s.sn * y;
                ts[(j + 1, col)] = s.cs * y - s.sn * x;
            }
            for row in 0..j {
                let x = ts[(row, j)];
                let y = ts[(row, j + 1)];
                ts[(row, j)] = s.cs * x + s.sn * y;
                ts[(row, j + 1)] = s.cs * y - s.sn * x;
            }
            rotations.push((j, s.cs, s.sn));
        }
        j += 2;
    }
    Some((ts, rotations))
}

/// Maps eigenvectors of a standardized factor back to the original
/// quasi-triangular factor: `v ← G·v` for the rotations of [`standardize`].
pub(crate) fn rotate_back<T: Field + Real>(v: &mut Mat<T>, rotations: &[(usize, T, T)]) {
    let ncols = v.ncols();
    for &(k, cs, sn) in rotations {
        for col in 0..ncols {
            let x = v[(k, col)];
            let y = v[(k + 1, col)];
            v[(k, col)] = cs * x - sn * y;
            v[(k + 1, col)] = sn * x + cs * y;
        }
    }
}

/// Scales every eigenvector (a real column, or the real/imaginary column pair
/// of a 2×2 block of `t`) to unit Euclidean norm.
pub(crate) fn normalize_unit<T: Field + Real>(v: &mut Mat<T>, t: &Mat<T>) {
    let n = t.nrows();
    let mut j = 0;
    while j < n {
        let width = if starts_block(t, j) { 2 } else { 1 };
        let mut big = T::zero();
        for col in j..j + width {
            for i in 0..n {
                big = Float::max(big, Scalar::abs(v[(i, col)]));
            }
        }
        if big > T::zero() && Float::is_finite(big) {
            let mut sum = T::zero();
            for col in j..j + width {
                for i in 0..n {
                    let s = v[(i, col)] / big;
                    sum = sum + s * s;
                }
            }
            let inv = T::one() / (big * Real::sqrt(sum));
            for col in j..j + width {
                for i in 0..n {
                    v[(i, col)] = v[(i, col)] * inv;
                }
            }
        }
        j += width;
    }
}

/// Computes right eigenvectors of a quasi-upper triangular matrix (LAPACK DTREVC).
///
/// The input `t` must be in real Schur form: upper quasi-triangular, with a
/// non-zero subdiagonal entry `t[j + 1, j]` exactly where a 2×2 diagonal block
/// holding a complex-conjugate eigenvalue pair starts (as produced by
/// [`Schur::compute`](super::Schur::compute)). A 2×2 block that is not in
/// standardized form is first standardized with a plane rotation (a block
/// with real eigenvalues is split), and the eigenvectors are rotated back.
///
/// # Returns
///
/// Matrix `V` with `T·V = V·D` column by column, each eigenvector normalized
/// to unit Euclidean norm. Column `k` belongs to the eigenvalue at diagonal
/// position `k`. For a 2×2 block at `(k, k + 1)` holding `wr ± i·wi`
/// (`wi > 0`), columns `k` and `k + 1` contain the real and imaginary parts of
/// the eigenvector `x` of `wr + i·wi`; the eigenvector of `wr − i·wi` is
/// `conj(x)`.
pub fn trevc_right<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    match standardize(t) {
        None => {
            let mut v = right_eigenvectors_raw(t);
            normalize_unit(&mut v, t);
            v
        }
        Some((ts, rotations)) => {
            let mut v = right_eigenvectors_raw(&ts);
            rotate_back(&mut v, &rotations);
            normalize_unit(&mut v, &ts);
            v
        }
    }
}

/// Computes left eigenvectors of a quasi-upper triangular matrix (LAPACK DTREVC).
///
/// The input requirements are those of [`trevc_right`].
///
/// # Returns
///
/// Matrix `U` of left eigenvectors, `uᴴ·T = λ·uᴴ` column by column (for a
/// real eigenvalue `uᵀ·T = λ·uᵀ`), each normalized to unit Euclidean norm.
/// The column layout is that of [`trevc_right`]: for a 2×2 block at
/// `(k, k + 1)`, columns `k` and `k + 1` hold the real and imaginary parts of
/// the left eigenvector `y` of `wr + i·wi` (`wi > 0`), and the left
/// eigenvector of `wr − i·wi` is `conj(y)`.
pub fn trevc_left<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    match standardize(t) {
        None => {
            let mut v = left_eigenvectors_raw(t);
            normalize_unit(&mut v, t);
            v
        }
        Some((ts, rotations)) => {
            let mut v = left_eigenvectors_raw(&ts);
            rotate_back(&mut v, &rotations);
            normalize_unit(&mut v, &ts);
            v
        }
    }
}
