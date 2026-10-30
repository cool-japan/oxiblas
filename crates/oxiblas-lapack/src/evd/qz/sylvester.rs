//! Generalized Sylvester solver of `DTGEX2`: `DTGSY2` (`TRANS = 'N'`,
//! `IJOB = 0`) with the complete-pivoting LU kernels `DGETC2`/`DGESC2`
//! (Netlib LAPACK 3.12.0, BSD-3).
//!
//! For the pairs `(A, D)` (`m × m`) and `(B, E)` (`n × n`), `A`, `B` upper
//! quasi-triangular and `D`, `E` upper triangular, `DTGSY2` solves
//!
//! ```text
//! A·R − L·B = scale·C
//! D·R − L·E = scale·F
//! ```
//!
//! block by block (1×1 and 2×2 diagonal blocks), each block system being a
//! `2×2`, `4×4` or `8×8` Kronecker system solved with `DGETC2`/`DGESC2`;
//! `0 < scale ≤ 1` avoids overflow. The `DLATDF` (`IJOB ≠ 0`) and
//! transposed paths are not used by `DTGEX2` and are not implemented.
//!
//! The operands live in `DTGEX2`'s `LDST = 4` work arrays ([`Small`], so
//! `m, n ≤ 4`) and the block systems in `LDZ = 8` arrays ([`Mat8`]).

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::util::Consts;

/// A `4 × 4` work array indexed `[row][column]` (`DTGEX2`'s `LDST = 4`).
pub(super) type Small<T> = [[T; 4]; 4];

/// An `8 × 8` work array indexed `[row][column]` (`DTGSY2`'s `LDZ = 8`).
pub(super) type Mat8<T> = [[T; 8]; 8];

/// Pivots (0-based) and `INFO` of [`getc2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Pivots {
    /// Row `i` was interchanged with row `ipiv[i]` (`IPIV`).
    pub(super) ipiv: [usize; 8],
    /// Column `j` was interchanged with column `jpiv[j]` (`JPIV`).
    pub(super) jpiv: [usize; 8],
    /// `0`, or the 1-based index `k` of the last pivot `U(k,k)` that was
    /// smaller than `SMIN` and has been replaced by `SMIN` (`INFO = k`).
    pub(super) info: usize,
}

/// Swaps rows `r1` and `r2` of `a` over the columns `0..n` (`DSWAP`).
fn swap_rows8<T: Copy>(a: &mut Mat8<T>, r1: usize, r2: usize, n: usize) {
    for c in 0..n {
        let tmp = a[r1][c];
        a[r1][c] = a[r2][c];
        a[r2][c] = tmp;
    }
}

/// Swaps columns `c1` and `c2` of `a` over the rows `0..n` (`DSWAP`).
fn swap_cols8<T: Copy>(a: &mut Mat8<T>, c1: usize, c2: usize, n: usize) {
    for row in a.iter_mut().take(n) {
        row.swap(c1, c2);
    }
}

/// `DGETC2`: `A = P·L·U·Q` with complete pivoting on the leading `n × n`
/// part of `a` (`n ≤ 8`), overwritten by the unit lower `L` (strictly
/// below the diagonal) and `U`. Pivots smaller than
/// `SMIN = max(eps · max|A|, SMLNUM)` are replaced by `SMIN` (`INFO = k`).
///
/// The pivot search scans rows in the outer loop with `.GE.`, so among
/// equal maxima the last one in row-major order is taken (LAPACK 3.12.0;
/// later versions scan columns first).
pub(super) fn getc2<T: Field + Real>(n: usize, a: &mut Mat8<T>, k: &Consts<T>) -> Pivots {
    let zero = T::zero();
    // Set constants to control overflow (BIGNUM = 1/SMLNUM is unused).
    let eps = k.eps;
    let smlnum = k.safmin / eps;
    let mut piv = Pivots {
        ipiv: [0; 8],
        jpiv: [0; 8],
        info: 0,
    };
    // Quick return if possible.
    if n == 0 {
        return piv;
    }
    // Handle the case N = 1 by itself.
    if n == 1 {
        if Scalar::abs(a[0][0]) < smlnum {
            piv.info = 1;
            a[0][0] = smlnum;
        }
        return piv;
    }

    // Factorize A using complete pivoting. Set pivots less than SMIN to SMIN.
    let mut smin = zero;
    for i in 0..n - 1 {
        // Find max element in matrix A (loops 10/20).
        let mut xmax = zero;
        let (mut ipv, mut jpv) = (i, i);
        for (ip, row) in a.iter().enumerate().take(n).skip(i) {
            for (jp, &x) in row.iter().enumerate().take(n).skip(i) {
                if Scalar::abs(x) >= xmax {
                    xmax = Scalar::abs(x);
                    ipv = ip;
                    jpv = jp;
                }
            }
        }
        if i == 0 {
            smin = Float::max(eps * xmax, smlnum);
        }

        // Swap rows.
        if ipv != i {
            swap_rows8(a, ipv, i, n);
        }
        piv.ipiv[i] = ipv;

        // Swap columns.
        if jpv != i {
            swap_cols8(a, jpv, i, n);
        }
        piv.jpiv[i] = jpv;

        // Check for singularity.
        if Scalar::abs(a[i][i]) < smin {
            piv.info = i + 1;
            a[i][i] = smin;
        }
        for j in (i + 1)..n {
            a[j][i] = a[j][i] / a[i][i];
        }
        // DGER(N−I, N−I, −ONE, A(I+1,I), 1, A(I,I+1), LDA, A(I+1,I+1), LDA).
        for jj in (i + 1)..n {
            let y = a[i][jj];
            if y != zero {
                let temp = -y;
                for ii in (i + 1)..n {
                    a[ii][jj] = a[ii][jj] + a[ii][i] * temp;
                }
            }
        }
    }

    if Scalar::abs(a[n - 1][n - 1]) < smin {
        piv.info = n;
        a[n - 1][n - 1] = smin;
    }

    // Set last pivots to N.
    piv.ipiv[n - 1] = n - 1;
    piv.jpiv[n - 1] = n - 1;
    piv
}

/// `IDAMAX`: index of the first entry of largest absolute value.
fn idamax<T: Field + Real>(x: &[T]) -> usize {
    let mut best = 0;
    let mut dmax = Scalar::abs(x[0]);
    for (i, &v) in x.iter().enumerate().skip(1) {
        if Scalar::abs(v) > dmax {
            best = i;
            dmax = Scalar::abs(v);
        }
    }
    best
}

/// `DGESC2`: solves `A·x = scale·rhs` with the factorization of [`getc2`]
/// (`rhs` is overwritten by `x`); returns `scale`, `0 < scale ≤ 1`, chosen
/// so that the solution does not overflow.
pub(super) fn gesc2<T: Field + Real>(
    n: usize,
    a: &Mat8<T>,
    rhs: &mut [T; 8],
    piv: &Pivots,
    k: &Consts<T>,
) -> T {
    let one = T::one();
    if n == 0 {
        return one;
    }
    // Set constant to control overflow (BIGNUM is unused).
    let smlnum = k.safmin / k.eps;

    // Apply permutations IPIV to RHS (DLASWP, forward).
    for i in 0..n - 1 {
        let ip = piv.ipiv[i];
        if ip != i {
            rhs.swap(i, ip);
        }
    }

    // Solve for L part.
    for i in 0..n - 1 {
        for j in (i + 1)..n {
            rhs[j] = rhs[j] - a[j][i] * rhs[i];
        }
    }

    // Solve for U part. Check for scaling.
    let mut scale = one;
    let imax = idamax(&rhs[..n]);
    if k.two * smlnum * Scalar::abs(rhs[imax]) > Scalar::abs(a[n - 1][n - 1]) {
        let temp = k.half / Scalar::abs(rhs[imax]);
        for x in rhs.iter_mut().take(n) {
            *x = temp * *x;
        }
        scale = scale * temp;
    }
    for i in (0..n).rev() {
        let temp = one / a[i][i];
        rhs[i] = rhs[i] * temp;
        for j in (i + 1)..n {
            rhs[i] = rhs[i] - rhs[j] * (a[i][j] * temp);
        }
    }

    // Apply permutations JPIV to the solution (DLASWP, backward).
    for i in (0..n - 1).rev() {
        let jp = piv.jpiv[i];
        if jp != i {
            rhs.swap(i, jp);
        }
    }
    scale
}

/// Coefficients of the generalized Sylvester equation of [`tgsy2`]: the
/// `m × m` pair `(a, d)` and the `n × n` pair `(b, e)` (`1 ≤ m, n ≤ 4`),
/// `a`, `b` upper quasi-triangular and `d`, `e` upper triangular.
#[derive(Debug, Clone, Copy)]
pub(super) struct SylvesterPencils<T> {
    /// Order of `(a, d)`.
    pub(super) m: usize,
    /// Order of `(b, e)`.
    pub(super) n: usize,
    /// `A`.
    pub(super) a: Small<T>,
    /// `B`.
    pub(super) b: Small<T>,
    /// `D`.
    pub(super) d: Small<T>,
    /// `E`.
    pub(super) e: Small<T>,
}

/// Result of [`tgsy2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Tgsy2<T> {
    /// `0 < scale ≤ 1`: `R`, `L` solve the system with right-hand sides
    /// `scale·C`, `scale·F`.
    pub(super) scale: T,
    /// `0`, or `> 0` when `(A, D)` and `(B, E)` have common or very close
    /// eigenvalues (a pivot of a block system was perturbed, `DTGSY2`
    /// `INFO`).
    pub(super) info: usize,
}

/// Diagonal block structure of the quasi-triangular `m × m` matrix `x`
/// (`DTGSY2` labels 10–40): block start rows followed by the sentinel `m`,
/// and the number of blocks.
fn block_starts<T: Field + Real>(x: &Small<T>, m: usize) -> ([usize; 5], usize) {
    let mut starts = [0usize; 5];
    let mut p = 0;
    let mut i = 0;
    while i < m {
        starts[p] = i;
        p += 1;
        if i + 1 == m {
            break;
        }
        i += if x[i + 1][i] != T::zero() { 2 } else { 1 };
    }
    starts[p] = m;
    (starts, p)
}

/// One `(I, J)` block system of `DTGSY2`: the Kronecker matrix `Z`
/// (`2·mb·nb` square) and right-hand side for the blocks starting at rows
/// `is` of `(A, D)` and `js` of `(B, E)`.
fn block_system<T: Field + Real>(
    p: &SylvesterPencils<T>,
    c: &Small<T>,
    f: &Small<T>,
    (is, mb): (usize, usize),
    (js, nb): (usize, usize),
) -> (Mat8<T>, [T; 8]) {
    let zero = T::zero();
    let (a, b, d, e) = (&p.a, &p.b, &p.d, &p.e);
    let (isp1, jsp1) = (is + 1, js + 1);
    let mut z = [[zero; 8]; 8];
    let mut rhs = [zero; 8];
    match (mb, nb) {
        (1, 1) => {
            // Build a 2-by-2 system Z * x = RHS.
            z[0][0] = a[is][is];
            z[1][0] = d[is][is];
            z[0][1] = -b[js][js];
            z[1][1] = -e[js][js];
            rhs[0] = c[is][js];
            rhs[1] = f[is][js];
        }
        (1, 2) => {
            // Build a 4-by-4 system Z * x = RHS.
            z[0][0] = a[is][is];
            z[2][0] = d[is][is];
            z[1][1] = a[is][is];
            z[3][1] = d[is][is];
            z[0][2] = -b[js][js];
            z[1][2] = -b[js][jsp1];
            z[2][2] = -e[js][js];
            z[3][2] = -e[js][jsp1];
            z[0][3] = -b[jsp1][js];
            z[1][3] = -b[jsp1][jsp1];
            z[3][3] = -e[jsp1][jsp1];
            rhs[0] = c[is][js];
            rhs[1] = c[is][jsp1];
            rhs[2] = f[is][js];
            rhs[3] = f[is][jsp1];
        }
        (2, 1) => {
            // Build a 4-by-4 system Z * x = RHS.
            z[0][0] = a[is][is];
            z[1][0] = a[isp1][is];
            z[2][0] = d[is][is];
            z[0][1] = a[is][isp1];
            z[1][1] = a[isp1][isp1];
            z[2][1] = d[is][isp1];
            z[3][1] = d[isp1][isp1];
            z[0][2] = -b[js][js];
            z[2][2] = -e[js][js];
            z[1][3] = -b[js][js];
            z[3][3] = -e[js][js];
            rhs[0] = c[is][js];
            rhs[1] = c[isp1][js];
            rhs[2] = f[is][js];
            rhs[3] = f[isp1][js];
        }
        _ => {
            // Build an 8-by-8 system Z * x = RHS.
            z[0][0] = a[is][is];
            z[1][0] = a[isp1][is];
            z[4][0] = d[is][is];
            z[0][1] = a[is][isp1];
            z[1][1] = a[isp1][isp1];
            z[4][1] = d[is][isp1];
            z[5][1] = d[isp1][isp1];
            z[2][2] = a[is][is];
            z[3][2] = a[isp1][is];
            z[6][2] = d[is][is];
            z[2][3] = a[is][isp1];
            z[3][3] = a[isp1][isp1];
            z[6][3] = d[is][isp1];
            z[7][3] = d[isp1][isp1];
            z[0][4] = -b[js][js];
            z[2][4] = -b[js][jsp1];
            z[4][4] = -e[js][js];
            z[6][4] = -e[js][jsp1];
            z[1][5] = -b[js][js];
            z[3][5] = -b[js][jsp1];
            z[5][5] = -e[js][js];
            z[7][5] = -e[js][jsp1];
            z[0][6] = -b[jsp1][js];
            z[2][6] = -b[jsp1][jsp1];
            z[6][6] = -e[jsp1][jsp1];
            z[1][7] = -b[jsp1][js];
            z[3][7] = -b[jsp1][jsp1];
            z[7][7] = -e[jsp1][jsp1];
            // Set up right hand side(s) (loop 80): columns of C, then of F.
            for jj in 0..nb {
                for r in 0..mb {
                    rhs[r + jj * mb] = c[is + r][js + jj];
                    rhs[mb * nb + r + jj * mb] = f[is + r][js + jj];
                }
            }
        }
    }
    (z, rhs)
}

/// `DTGSY2(TRANS = 'N', IJOB = 0)`: solves `A·R − L·B = scale·C`,
/// `D·R − L·E = scale·F` for the `m × n` matrices `R` (overwriting `c`) and
/// `L` (overwriting `f`).
///
/// The `(I, J)` subsystems are solved for `I = P, P−1, …, 1` (row blocks of
/// `A`, bottom up) within `J = 1, 2, …, Q` (column blocks of `B`), and the
/// solved blocks are substituted into the remaining equations.
pub(super) fn tgsy2<T: Field + Real>(
    p: &SylvesterPencils<T>,
    c: &mut Small<T>,
    f: &mut Small<T>,
    k: &Consts<T>,
) -> Tgsy2<T> {
    let (m, n) = (p.m, p.n);
    let (a, b, d, e) = (&p.a, &p.b, &p.d, &p.e);
    let one = T::one();
    let zero = T::zero();

    // Determine block structure of A and of B.
    let (ia, np) = block_starts(a, m);
    let (jb, nq) = block_starts(b, n);

    let mut scale = one;
    let mut info = 0;
    for jblk in 0..nq {
        let js = jb[jblk];
        let je = jb[jblk + 1] - 1;
        let nb = je - js + 1;
        for iblk in (0..np).rev() {
            let is = ia[iblk];
            let ie = ia[iblk + 1] - 1;
            let mb = ie - is + 1;
            let zdim = 2 * mb * nb;

            // Solve Z * x = RHS.
            let (mut z, mut rhs) = block_system(p, c, f, (is, mb), (js, nb));
            let piv = getc2(zdim, &mut z, k);
            if piv.info > 0 {
                info = piv.info;
            }
            let scaloc = gesc2(zdim, &z, &mut rhs, &piv, k);
            if scaloc != one {
                for col in 0..n {
                    for row in 0..m {
                        c[row][col] = scaloc * c[row][col];
                        f[row][col] = scaloc * f[row][col];
                    }
                }
                scale = scale * scaloc;
            }

            // Unpack solution vector(s): R(I, J) in RHS(1 : MB·NB) (column
            // major), L(I, J) in RHS(MB·NB+1 : 2·MB·NB).
            let lofs = mb * nb;
            for jj in 0..nb {
                for r in 0..mb {
                    c[is + r][js + jj] = rhs[r + jj * mb];
                    f[is + r][js + jj] = rhs[lofs + r + jj * mb];
                }
            }

            // Substitute R(I, J) and L(I, J) into the remaining equations:
            // C(1:IS−1, J) −= A(1:IS−1, I)·R(I, J), F(1:IS−1, J) −= D(1:IS−1, I)·R(I, J)
            // (DAXPY / DGER / DGEMV / DGEMM with ALPHA = −1, accumulated
            // column by column of R as the reference BLAS does).
            if iblk > 0 {
                for jj in 0..nb {
                    for l in 0..mb {
                        let x = rhs[l + jj * mb];
                        if x != zero {
                            let temp = -x;
                            for r in 0..is {
                                c[r][js + jj] = c[r][js + jj] + a[r][is + l] * temp;
                                f[r][js + jj] = f[r][js + jj] + d[r][is + l] * temp;
                            }
                        }
                    }
                }
            }
            // C(I, JE+1:N) += L(I, J)·B(J, JE+1:N), F(I, JE+1:N) += L(I, J)·E(J, JE+1:N).
            if jblk + 1 < nq {
                for col in (je + 1)..n {
                    for l in 0..nb {
                        let (bv, ev) = (b[js + l][col], e[js + l][col]);
                        for r in 0..mb {
                            let lv = rhs[lofs + r + l * mb];
                            c[is + r][col] = c[is + r][col] + lv * bv;
                            f[is + r][col] = f[is + r][col] + lv * ev;
                        }
                    }
                }
            }
        }
    }
    Tgsy2 { scale, info }
}
