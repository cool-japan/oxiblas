//! `DLAGV2`: generalized Schur factorization of a real 2×2 pencil `(A, B)`
//! with `B` upper triangular (Netlib LAPACK 3.12.0, BSD-3). `DTGEX2` uses
//! it to re-standardize the 2×2 diagonal blocks after a swap.
//!
//! With `Qᵀ = [csl snl; −snl csl]` and `Z = [csr −snr; snr csr]`:
//!
//! * real eigenvalues (including `0/0` and `1/0` types): `Qᵀ·A·Z` and
//!   `Qᵀ·B·Z` are upper triangular;
//! * a complex conjugate pair: `Qᵀ·B·Z` is diagonal (`DLASV2` of `B`).
//!
//! The reference documents `b11 ≥ b22 > 0` for the complex case, but its
//! code does not enforce the signs (`DLASV2` returns signed singular
//! values): a block entering with a negative diagonal in `B` can leave with
//! one. This translation keeps the reference behaviour; `DTGSEN`'s final
//! normalization (in `reorder`) makes the diagonal of `T` positive.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::lag2::{dlag2, dlasv2};
use super::util::{Consts, lapy2, lartg};

/// Output of [`dlagv2`].
#[derive(Debug, Clone, Copy)]
pub(super) struct Lagv2<T> {
    /// The "A-part" `Qᵀ·A·Z` of the generalized Schur form (`[row][col]`).
    pub(super) a: [[T; 2]; 2],
    /// The "B-part" `Qᵀ·B·Z` (upper triangular, diagonal for a complex
    /// pair).
    pub(super) b: [[T; 2]; 2],
    // The eigenvalue outputs are part of the DLAGV2 contract, but DTGEX2
    // discards them (the reference writes them into scratch arrays); only
    // the unit tests read them.
    /// `ALPHAR`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) alphar: [T; 2],
    /// `ALPHAI` (`alphai[1] = −alphai[0]`; zero for real eigenvalues).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) alphai: [T; 2],
    /// `BETA` (may be zero for infinite eigenvalues; one for a pair).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) beta: [T; 2],
    /// Cosine of the left rotation.
    pub(super) csl: T,
    /// Sine of the left rotation.
    pub(super) snl: T,
    /// Cosine of the right rotation.
    pub(super) csr: T,
    /// Sine of the right rotation.
    pub(super) snr: T,
}

/// `DROT` on the two rows of a 2×2 array.
fn rot_rows2<T: Field + Real>(m: &mut [[T; 2]; 2], c: T, s: T) {
    for j in 0..2 {
        let (x, y) = (m[0][j], m[1][j]);
        m[0][j] = c * x + s * y;
        m[1][j] = c * y - s * x;
    }
}

/// `DROT` on the two columns of a 2×2 array.
fn rot_cols2<T: Field + Real>(m: &mut [[T; 2]; 2], c: T, s: T) {
    for row in m.iter_mut() {
        let (x, y) = (row[0], row[1]);
        row[0] = c * x + s * y;
        row[1] = c * y - s * x;
    }
}

/// `DLAGV2` on the pencil `(a, b)` (`[row][column]`, `b` upper triangular).
///
/// `b[1][0]` is used as given (it takes part in the rotations, as in the
/// reference, but is not scaled) and is an exact zero on output.
pub(super) fn dlagv2<T: Field + Real>(a: [[T; 2]; 2], b: [[T; 2]; 2], k: &Consts<T>) -> Lagv2<T> {
    let one = T::one();
    let zero = T::zero();
    let abs = |x: T| Scalar::abs(x);
    let safmin = k.safmin;
    let ulp = k.eps;
    let (mut a, mut b) = (a, b);

    // Scale A.
    let anorm = Float::max(
        Float::max(abs(a[0][0]) + abs(a[1][0]), abs(a[0][1]) + abs(a[1][1])),
        safmin,
    );
    let ascale = one / anorm;
    a[0][0] = ascale * a[0][0];
    a[0][1] = ascale * a[0][1];
    a[1][0] = ascale * a[1][0];
    a[1][1] = ascale * a[1][1];

    // Scale B.
    let bnorm = Float::max(
        Float::max(abs(b[0][0]), abs(b[0][1]) + abs(b[1][1])),
        safmin,
    );
    let bscale = one / bnorm;
    b[0][0] = bscale * b[0][0];
    b[0][1] = bscale * b[0][1];
    b[1][1] = bscale * b[1][1];

    let mut wi = zero;
    let (mut scale1, mut wr1) = (one, zero);
    let (csl, snl, csr, snr);
    if abs(a[1][0]) <= ulp {
        // Check if A can be deflated.
        csl = one;
        snl = zero;
        csr = one;
        snr = zero;
        a[1][0] = zero;
        b[1][0] = zero;
    } else if abs(b[0][0]) <= ulp {
        // Check if B is singular.
        let (c, s, _) = lartg(a[0][0], a[1][0], k);
        csl = c;
        snl = s;
        csr = one;
        snr = zero;
        rot_rows2(&mut a, csl, snl);
        rot_rows2(&mut b, csl, snl);
        a[1][0] = zero;
        b[0][0] = zero;
        b[1][0] = zero;
    } else if abs(b[1][1]) <= ulp {
        let (c, s, _) = lartg(a[1][1], a[1][0], k);
        csr = c;
        snr = -s;
        rot_cols2(&mut a, csr, snr);
        rot_cols2(&mut b, csr, snr);
        csl = one;
        snl = zero;
        a[1][0] = zero;
        b[1][0] = zero;
        b[1][1] = zero;
    } else {
        // B is nonsingular, first compute the eigenvalues of (A, B).
        let g = dlag2(a, b, safmin, k);
        scale1 = g.scale1;
        wr1 = g.wr1;
        wi = g.wi;
        if wi == zero {
            // Two real eigenvalues, compute s·A − w·B.
            let h1 = scale1 * a[0][0] - wr1 * b[0][0];
            let h2 = scale1 * a[0][1] - wr1 * b[0][1];
            let h3 = scale1 * a[1][1] - wr1 * b[1][1];
            let rr = lapy2(h1, h2);
            let qq = lapy2(scale1 * a[1][0], h3);
            let (c, s, _) = if rr > qq {
                // Find right rotation matrix to zero 1,1 element of (sA − wB).
                lartg(h2, h1, k)
            } else {
                // Find right rotation matrix to zero 2,1 element of (sA − wB).
                lartg(h3, scale1 * a[1][0], k)
            };
            csr = c;
            snr = -s;
            rot_cols2(&mut a, csr, snr);
            rot_cols2(&mut b, csr, snr);

            // Compute inf norms of A and B.
            let h1 = Float::max(abs(a[0][0]) + abs(a[0][1]), abs(a[1][0]) + abs(a[1][1]));
            let h2 = Float::max(abs(b[0][0]) + abs(b[0][1]), abs(b[1][0]) + abs(b[1][1]));
            let (c, s, _) = if scale1 * h1 >= abs(wr1) * h2 {
                // Find left rotation matrix Q to zero out B(2,1).
                lartg(b[0][0], b[1][0], k)
            } else {
                // Find left rotation matrix Q to zero out A(2,1).
                lartg(a[0][0], a[1][0], k)
            };
            csl = c;
            snl = s;
            rot_rows2(&mut a, csl, snl);
            rot_rows2(&mut b, csl, snl);
            a[1][0] = zero;
            b[1][0] = zero;
        } else {
            // A pair of complex conjugate eigenvalues: first compute the SVD
            // of the matrix B.
            let sv = dlasv2(b[0][0], b[0][1], b[1][1], k);
            csl = sv.csl;
            snl = sv.snl;
            csr = sv.csr;
            snr = sv.snr;
            // Form (A, B) := Q(A, B)Zᵀ where Q is the left and Z the right
            // rotation matrix computed by DLASV2.
            rot_rows2(&mut a, csl, snl);
            rot_rows2(&mut b, csl, snl);
            rot_cols2(&mut a, csr, snr);
            rot_cols2(&mut b, csr, snr);
            b[1][0] = zero;
            b[0][1] = zero;
        }
    }

    // Unscaling.
    a[0][0] = anorm * a[0][0];
    a[1][0] = anorm * a[1][0];
    a[0][1] = anorm * a[0][1];
    a[1][1] = anorm * a[1][1];
    b[0][0] = bnorm * b[0][0];
    b[1][0] = bnorm * b[1][0];
    b[0][1] = bnorm * b[0][1];
    b[1][1] = bnorm * b[1][1];

    let (alphar, alphai, beta) = if wi == zero {
        ([a[0][0], a[1][1]], [zero, zero], [b[0][0], b[1][1]])
    } else {
        let ar = anorm * wr1 / scale1 / bnorm;
        let ai = anorm * wi / scale1 / bnorm;
        ([ar, ar], [ai, -ai], [one, one])
    };

    Lagv2 {
        a,
        b,
        alphar,
        alphai,
        beta,
        csl,
        snl,
        csr,
        snr,
    }
}
