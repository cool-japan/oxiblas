//! `DTGEX2`: swaps two adjacent diagonal blocks (1×1 or 2×2) of a real
//! generalized Schur pencil by an orthogonal equivalence (Netlib LAPACK
//! 3.12.0, BSD-3), the kernel of the reordering routines `DTGEXC`/`DTGSEN`.
//!
//! * `n1 = n2 = 1` (case 1): one Givens rotation from the right that makes
//!   the swapped `(S, T)` pair upper triangular up to a rounding-level
//!   `(2,1)` entry, and one from the left computed from `S` or `T`
//!   according to `|S22|·|T11| ≥ |S11|·|T22|`.
//! * otherwise (case 2): the generalized Sylvester equation
//!   `S11·R − L·S22 = scale·S12`, `T11·R − L·T22 = scale·T12` is solved
//!   (`DTGSY2`); the left factor comes from the QR factorization of
//!   `[−L; scale·I]`, the right one from the RQ factorization of
//!   `[scale·I, R]`; the swapped `T` part is re-triangularized by an RQ or a
//!   QR factorization (whichever leaves the smaller `S21`), and the new 2×2
//!   blocks are re-standardized with `DLAGV2`.
//!
//! A swap is accepted only after the weak stability test (`‖S21‖`, `‖T21‖`
//! below `thresh = max(20·eps·‖·‖_F, safmin/eps)` of the original blocks)
//! and the strong test (the backward error of the swap below the same
//! thresholds; `WANDS = .TRUE.`). A rejected swap leaves the pencil
//! untouched (`DTGEX2` `INFO = 1`).
//!
//! The Frobenius norms use the scaled sum of squares of `util::SumSq`
//! (classical `DLASSQ` recurrence); they only feed the acceptance tests.

use core::ops::Range;

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::lagv2::{Lagv2, dlagv2};
use super::sylvester::{Small, SylvesterPencils, tgsy2};
use super::util::{Consts, SumSq, larfg, lartg, rot_cols, rot_rows};

/// The generalized real Schur form being reordered in place: `S`, `T` and
/// the accumulated orthogonal factors `Q`, `Z` (the `A`, `B`, `Q`, `Z`
/// arguments of `DTGEX2`/`DTGEXC`/`DTGSEN` with `WANTQ = WANTZ = .TRUE.`),
/// so that `Q·S·Zᵀ` and `Q·T·Zᵀ` are invariant.
pub(super) struct SchurPencil<'a, T: Scalar> {
    /// Upper quasi-triangular `S` (`A`).
    pub(super) s: &'a mut Mat<T>,
    /// Upper triangular `T` (`B`).
    pub(super) t: &'a mut Mat<T>,
    /// Left factor `Q`.
    pub(super) q: &'a mut Mat<T>,
    /// Right factor `Z`.
    pub(super) z: &'a mut Mat<T>,
}

/// Why `DTGEX2` rejected a swap (`INFO = 1`); the pencil is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SwapRejection {
    /// `DTGSY2` reported common or very close eigenvalues of the two blocks
    /// (a pivot of its Kronecker system was perturbed).
    Sylvester,
    /// The weak stability test failed (`S21` or `T21` not negligible).
    Weak,
    /// The strong stability test failed (backward error too large).
    Strong,
}

/// Acceptance thresholds `THRESHA`, `THRESHB`.
#[derive(Debug, Clone, Copy)]
struct Thresholds<T> {
    a: T,
    b: T,
}

/// Which side and transposition an orthogonal factor is applied with
/// (`DORM2R`/`DORMR2` `SIDE`, `TRANS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Apply {
    /// `C := Q·C` (`SIDE = 'L'`, `TRANS = 'N'`).
    LeftQ,
    /// `C := Qᵀ·C` (`SIDE = 'L'`, `TRANS = 'T'`).
    LeftQt,
    /// `C := C·Q` (`SIDE = 'R'`, `TRANS = 'N'`).
    RightQ,
    /// `C := C·Qᵀ` (`SIDE = 'R'`, `TRANS = 'T'`).
    RightQt,
}

impl Apply {
    fn left(self) -> bool {
        matches!(self, Self::LeftQ | Self::LeftQt)
    }

    fn notrans(self) -> bool {
        matches!(self, Self::LeftQ | Self::RightQ)
    }
}

/// An all-zero work array (`DLASET('Full', LDST, LDST, ZERO, ZERO, ...)`).
fn small_zero<T: Field + Real>() -> Small<T> {
    [[T::zero(); 4]; 4]
}

/// Frobenius norm of `x(rows, cols)` (`DLASSQ` over the entries in
/// column-major order).
fn frob_small<T: Field + Real>(x: &Small<T>, rows: Range<usize>, cols: Range<usize>) -> T {
    let mut acc = SumSq::new();
    for j in cols {
        for row in &x[rows.clone()] {
            acc.push(row[j]);
        }
    }
    acc.norm()
}

/// `op(x)·op(y)` of the leading `m × m` parts (`DGEMM` with `ALPHA = 1`,
/// `BETA = 0`; inner products accumulated in increasing order).
fn mul_small<T: Field + Real>(
    m: usize,
    x: &Small<T>,
    tx: bool,
    y: &Small<T>,
    ty: bool,
) -> Small<T> {
    let mut out = small_zero();
    for (i, out_row) in out.iter_mut().enumerate().take(m) {
        for (j, out_ij) in out_row.iter_mut().enumerate().take(m) {
            let mut acc = T::zero();
            for l in 0..m {
                let xv = if tx { x[l][i] } else { x[i][l] };
                let yv = if ty { y[j][l] } else { y[l][j] };
                acc = acc + xv * yv;
            }
            *out_ij = acc;
        }
    }
    out
}

/// `DROT` on rows `r1`, `r2` of a work array over the columns `cols`.
fn rot_rows_small<T: Field + Real>(
    x: &mut Small<T>,
    r1: usize,
    r2: usize,
    cols: Range<usize>,
    c: T,
    s: T,
) {
    for j in cols {
        let (u, v) = (x[r1][j], x[r2][j]);
        x[r1][j] = c * u + s * v;
        x[r2][j] = c * v - s * u;
    }
}

/// `DROT` on columns `c1`, `c2` of a work array over the rows `rows`.
fn rot_cols_small<T: Field + Real>(
    x: &mut Small<T>,
    c1: usize,
    c2: usize,
    rows: Range<usize>,
    c: T,
    s: T,
) {
    for row in &mut x[rows] {
        let (u, v) = (row[c1], row[c2]);
        row[c1] = c * u + s * v;
        row[c2] = c * v - s * u;
    }
}

/// `DLARF('Left')`: `C(r0 .. r0+len, cols) := (I − tau·v·vᵀ)·C(...)`, with
/// `w = Cᵀ·v` and the rank-one update `C += v·(−tau·w)ᵀ` (reference BLAS
/// association). `tau = 0` leaves `C` unchanged.
fn larf_left<T: Field + Real>(c: &mut Small<T>, r0: usize, v: &[T], tau: T, cols: Range<usize>) {
    if tau == T::zero() {
        return;
    }
    for j in cols {
        let mut w = T::zero();
        for (l, &vl) in v.iter().enumerate() {
            w = w + c[r0 + l][j] * vl;
        }
        let temp = -tau * w;
        for (l, &vl) in v.iter().enumerate() {
            c[r0 + l][j] = c[r0 + l][j] + vl * temp;
        }
    }
}

/// `DLARF('Right')`: `C(rows, c0 .. c0+len) := C(...)·(I − tau·v·vᵀ)`, with
/// `w = C·v` and the rank-one update `C += w·(−tau·v)ᵀ`. `tau = 0` leaves
/// `C` unchanged.
fn larf_right<T: Field + Real>(c: &mut Small<T>, rows: Range<usize>, c0: usize, v: &[T], tau: T) {
    if tau == T::zero() {
        return;
    }
    for row in &mut c[rows] {
        let mut w = T::zero();
        for (l, &vl) in v.iter().enumerate() {
            w = w + row[c0 + l] * vl;
        }
        for (l, &vl) in v.iter().enumerate() {
            row[c0 + l] = row[c0 + l] + w * (-tau * vl);
        }
    }
}

/// `DGEQR2` on the leading `rows × cols` part of `a`: `A = Q·R` with
/// `Q = H(1)·…·H(k)`; `R` on and above the diagonal, the reflector vectors
/// below it. Returns `TAU`.
fn geqr2<T: Field + Real>(rows: usize, cols: usize, a: &mut Small<T>, k: &Consts<T>) -> [T; 4] {
    let mut tau = [T::zero(); 4];
    for i in 0..rows.min(cols) {
        // Generate elementary reflector H(i) to annihilate A(i+1:m, i).
        let len = rows - i - 1;
        let mut alpha = a[i][i];
        let mut x = [T::zero(); 4];
        for l in 0..len {
            x[l] = a[i + 1 + l][i];
        }
        tau[i] = larfg(&mut alpha, &mut x[..len], k);
        a[i][i] = alpha;
        for l in 0..len {
            a[i + 1 + l][i] = x[l];
        }
        if i + 1 < cols {
            // Apply H(i) to A(i:m, i+1:n) from the left.
            let mut v = [T::one(); 4];
            v[1..=len].copy_from_slice(&x[..len]);
            larf_left(a, i, &v[..=len], tau[i], (i + 1)..cols);
        }
    }
    tau
}

/// `DORG2R`: overwrites the leading `rows × cols` part of `a` (holding the
/// first `kref` reflectors of [`geqr2`]) with the `rows × cols` matrix `Q`
/// with orthonormal columns (`cols ≤ rows`).
fn org2r<T: Field + Real>(rows: usize, cols: usize, kref: usize, a: &mut Small<T>, tau: &[T; 4]) {
    let (zero, one) = (T::zero(), T::one());
    // Initialise columns k+1:n to columns of the unit matrix.
    for j in kref..cols {
        for row in a.iter_mut().take(rows) {
            row[j] = zero;
        }
        a[j][j] = one;
    }
    for i in (0..kref).rev() {
        // Apply H(i) to A(i:m, i:n) from the left.
        if i + 1 < cols {
            a[i][i] = one;
            let len = rows - i;
            let mut v = [zero; 4];
            for l in 0..len {
                v[l] = a[i + l][i];
            }
            larf_left(a, i, &v[..len], tau[i], (i + 1)..cols);
        }
        for row in a.iter_mut().take(rows).skip(i + 1) {
            row[i] = -tau[i] * row[i];
        }
        a[i][i] = one - tau[i];
        // Set A(1:i-1, i) to zero.
        for row in a.iter_mut().take(i) {
            row[i] = zero;
        }
    }
}

/// `DGERQ2` on the `rows × cols` submatrix of `a` starting at row `r0`
/// (`cols ≥ rows`): `A = R·Q` with `Q = H(1)·…·H(k)`; `R` in the upper
/// trapezoid, reflector `i` in row `rows − k + i` left of its unit entry
/// at column `cols − k + i`. Returns `TAU`.
fn gerq2<T: Field + Real>(
    r0: usize,
    rows: usize,
    cols: usize,
    a: &mut Small<T>,
    k: &Consts<T>,
) -> [T; 4] {
    let kmin = rows.min(cols);
    let mut tau = [T::zero(); 4];
    for i in (0..kmin).rev() {
        // Generate elementary reflector H(i) to annihilate
        // A(m−k+i, 1:n−k+i−1).
        let row = r0 + rows - kmin + i;
        let pc = cols - kmin + i;
        let mut alpha = a[row][pc];
        let mut x = [T::zero(); 4];
        x[..pc].copy_from_slice(&a[row][..pc]);
        tau[i] = larfg(&mut alpha, &mut x[..pc], k);
        a[row][pc] = alpha;
        a[row][..pc].copy_from_slice(&x[..pc]);
        // Apply H(i) to A(1:m−k+i−1, 1:n−k+i) from the right.
        let mut v = x;
        v[pc] = T::one();
        larf_right(a, r0..row, 0, &v[..=pc], tau[i]);
    }
    tau
}

/// `DORGR2`: overwrites the leading `rows × cols` part of `a` (its last
/// `kref` rows holding the reflectors of [`gerq2`]) with the `rows × cols`
/// matrix `Q` with orthonormal rows (`cols ≥ rows ≥ kref`).
fn orgr2<T: Field + Real>(rows: usize, cols: usize, kref: usize, a: &mut Small<T>, tau: &[T; 4]) {
    let (zero, one) = (T::zero(), T::one());
    if kref < rows {
        // Initialise rows 1:m−k to rows of the unit matrix.
        for j in 0..cols {
            for row in a.iter_mut().take(rows - kref) {
                row[j] = zero;
            }
            if j + rows >= cols && j + kref < cols {
                a[j + rows - cols][j] = one;
            }
        }
    }
    for i in 0..kref {
        let ii = rows - kref + i;
        let pc = cols - rows + ii;
        // Apply H(i) to A(1:m−k+i, 1:n−k+i) from the right.
        a[ii][pc] = one;
        let mut v = [zero; 4];
        v[..=pc].copy_from_slice(&a[ii][..=pc]);
        larf_right(a, 0..ii, 0, &v[..=pc], tau[i]);
        for x in a[ii].iter_mut().take(pc) {
            *x = -tau[i] * *x;
        }
        a[ii][pc] = one - tau[i];
        // Set A(m−k+i, n−k+i+1:n) to zero.
        for x in a[ii].iter_mut().take(cols).skip(pc + 1) {
            *x = zero;
        }
    }
}

/// `DORM2R(SIDE, TRANS, M, M, M, A, TAU, C)`: applies the orthogonal factor
/// of a [`geqr2`] factorization of an `m × m` matrix to the `m × m` `C`.
fn orm2r<T: Field + Real>(apply: Apply, m: usize, a: &Small<T>, tau: &[T; 4], c: &mut Small<T>) {
    let (left, notran) = (apply.left(), apply.notrans());
    let forward = (left && !notran) || (!left && notran);
    for step in 0..m {
        let i = if forward { step } else { m - 1 - step };
        let len = m - i;
        let mut v = [T::one(); 4];
        for l in 1..len {
            v[l] = a[i + l][i];
        }
        if left {
            // H(i) is applied to C(i:m, 1:n).
            larf_left(c, i, &v[..len], tau[i], 0..m);
        } else {
            // H(i) is applied to C(1:m, i:n).
            larf_right(c, 0..m, i, &v[..len], tau[i]);
        }
    }
}

/// `DORMR2(SIDE, TRANS, M, M, M, A, TAU, C)`: applies the orthogonal factor
/// of a [`gerq2`] factorization of an `m × m` matrix to the `m × m` `C`.
fn ormr2<T: Field + Real>(apply: Apply, m: usize, a: &Small<T>, tau: &[T; 4], c: &mut Small<T>) {
    let (left, notran) = (apply.left(), apply.notrans());
    let forward = (left && !notran) || (!left && notran);
    for step in 0..m {
        let i = if forward { step } else { m - 1 - step };
        // Reflector i: row i of A, unit entry at column NQ − K + i = i.
        let mut v = [T::one(); 4];
        v[..i].copy_from_slice(&a[i][..i]);
        if left {
            // H(i) is applied to C(1:m−k+i, 1:n).
            larf_left(c, 0, &v[..=i], tau[i], 0..m);
        } else {
            // H(i) is applied to C(1:m, 1:n−k+i).
            larf_right(c, 0..m, 0, &v[..=i], tau[i]);
        }
    }
}

/// `‖X(j1.., j1..) − left·mid·op(right)‖_F` over the `m × m` block of `src`
/// (the strong stability test; `op(right) = rightᵀ` when `right_t`).
fn swap_residual<T: Field + Real>(
    src: &Mat<T>,
    j1: usize,
    m: usize,
    (left, mid, right): (&Small<T>, &Small<T>, &Small<T>),
    right_t: bool,
) -> T {
    let work = mul_small(m, left, false, mid, false);
    let prod = mul_small(m, &work, false, right, right_t);
    let mut acc = SumSq::new();
    for j in 0..m {
        for (i, prod_row) in prod.iter().enumerate().take(m) {
            acc.push(src[(j1 + i, j1 + j)] - prod_row[j]);
        }
    }
    acc.norm()
}

/// `DTGEX2`: swaps the adjacent diagonal blocks of sizes `n1` (starting at
/// row `j1`) and `n2` of the Schur pencil, updating `Q` and `Z`.
///
/// `n1`, `n2` are 1 or 2 and the blocks must be blocks of the
/// quasi-triangular `S`. Returns the reason when the swap is rejected; the
/// pencil is then unchanged. The reference's quick returns (`n ≤ 1`, empty
/// or out-of-range blocks) are successful no-ops.
pub(super) fn tgex2<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    j1: usize,
    n1: usize,
    n2: usize,
    k: &Consts<T>,
) -> Result<(), SwapRejection> {
    let n = p.s.nrows();
    // Quick return if possible.
    if n <= 1 || n1 == 0 || n2 == 0 {
        return Ok(());
    }
    if n1 > n || j1 + n1 >= n {
        return Ok(());
    }
    let m = n1 + n2;
    debug_assert!(n1 <= 2 && n2 <= 2, "DTGEX2 swaps 1x1 and 2x2 blocks only");
    if m > 4 || j1 + m > n {
        return Ok(());
    }

    // Make a local copy of the selected block.
    let mut s = small_zero();
    let mut t = small_zero();
    for i in 0..m {
        for j in 0..m {
            s[i][j] = p.s[(j1 + i, j1 + j)];
            t[i][j] = p.t[(j1 + i, j1 + j)];
        }
    }

    // Compute threshold for testing acceptance of swapping
    // (THRESH = MAX(TWENTY·EPS·‖·‖_F, SMLNUM) since LAPACK 3.2.2).
    let eps = k.eps;
    let smlnum = k.safmin / eps;
    let dnorma = frob_small(&s, 0..m, 0..m);
    let dnormb = frob_small(&t, 0..m, 0..m);
    let twenty = T::from_u32(20).unwrap_or_else(|| k.four * k.four + k.four);
    let th = Thresholds {
        a: Float::max(twenty * eps * dnorma, smlnum),
        b: Float::max(twenty * eps * dnormb, smlnum),
    };

    if m == 2 {
        swap_1x1(p, j1, (s, t), th, k)
    } else {
        swap_blocks(p, (j1, n1, n2), (s, t), th, k)
    }
}

/// `DTGEX2` case 1: swap two 1×1 blocks with Givens rotations.
fn swap_1x1<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    j1: usize,
    (mut s, mut t): (Small<T>, Small<T>),
    th: Thresholds<T>,
    k: &Consts<T>,
) -> Result<(), SwapRejection> {
    let n = p.s.nrows();
    let m = 2;
    let mut li = small_zero();
    let mut ir = small_zero();

    // Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks using
    // Givens rotations and perform the swap tentatively.
    let f = s[1][1] * t[0][0] - t[1][1] * s[0][0];
    let g = s[1][1] * t[0][1] - t[1][1] * s[0][1];
    let sa = Scalar::abs(s[1][1]) * Scalar::abs(t[0][0]);
    let sb = Scalar::abs(s[0][0]) * Scalar::abs(t[1][1]);
    // DLARTG(F, G, IR(1,2), IR(1,1), DDUM).
    let (c, sn, _) = lartg(f, g, k);
    ir[0][1] = c;
    ir[0][0] = sn;
    ir[1][0] = -ir[0][1];
    ir[1][1] = ir[0][0];
    rot_cols_small(&mut s, 0, 1, 0..2, ir[0][0], ir[1][0]);
    rot_cols_small(&mut t, 0, 1, 0..2, ir[0][0], ir[1][0]);
    let (lc, ls, _) = if sa >= sb {
        lartg(s[0][0], s[1][0], k)
    } else {
        lartg(t[0][0], t[1][0], k)
    };
    li[0][0] = lc;
    li[1][0] = ls;
    rot_rows_small(&mut s, 0, 1, 0..2, li[0][0], li[1][0]);
    rot_rows_small(&mut t, 0, 1, 0..2, li[0][0], li[1][0]);
    li[1][1] = li[0][0];
    li[0][1] = -li[1][0];

    // Weak stability test: |S21| <= O(EPS F-norm((A))) and
    // |T21| <= O(EPS F-norm((B))).
    let weak = Scalar::abs(s[1][0]) <= th.a && Scalar::abs(t[1][0]) <= th.b;
    if !weak {
        return Err(SwapRejection::Weak);
    }

    // Strong stability test (WANDS):
    //   F-norm((A − QL·S·QRᵀ)) <= O(EPS·F-norm((A))) and
    //   F-norm((B − QL·T·QRᵀ)) <= O(EPS·F-norm((B))).
    let sa = swap_residual(p.s, j1, m, (&li, &s, &ir), true);
    let sb = swap_residual(p.t, j1, m, (&li, &t, &ir), true);
    if !(sa <= th.a && sb <= th.b) {
        return Err(SwapRejection::Strong);
    }

    // Update (A(J1:J1+M−1, M+J1:N), B(J1:J1+M−1, M+J1:N)) and
    // (A(1:J1−1, J1:J1+M), B(1:J1−1, J1:J1+M)).
    rot_cols(p.s, j1, j1 + 1, 0..(j1 + 2), ir[0][0], ir[1][0]);
    rot_cols(p.t, j1, j1 + 1, 0..(j1 + 2), ir[0][0], ir[1][0]);
    rot_rows(p.s, j1, j1 + 1, j1..n, li[0][0], li[1][0]);
    rot_rows(p.t, j1, j1 + 1, j1..n, li[0][0], li[1][0]);

    // Set N1-by-N2 (2,1)-blocks to ZERO.
    p.s[(j1 + 1, j1)] = T::zero();
    p.t[(j1 + 1, j1)] = T::zero();

    // Accumulate transformations into Q and Z.
    rot_cols(p.z, j1, j1 + 1, 0..n, ir[0][0], ir[1][0]);
    rot_cols(p.q, j1, j1 + 1, 0..n, li[0][0], li[1][0]);
    Ok(())
}

/// `DLAGV2` on the 2×2 diagonal block starting at `i`, written back into
/// `S` and `T`.
fn standardize_block<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    i: usize,
    k: &Consts<T>,
) -> Lagv2<T> {
    let blk = |m: &Mat<T>| {
        [
            [m[(i, i)], m[(i, i + 1)]],
            [m[(i + 1, i)], m[(i + 1, i + 1)]],
        ]
    };
    let g = dlagv2(blk(p.s), blk(p.t), k);
    for r in 0..2 {
        for c in 0..2 {
            p.s[(i + r, i + c)] = g.a[r][c];
            p.t[(i + r, i + c)] = g.b[r][c];
        }
    }
    g
}

/// `X(rows, j1 .. j1+m) := X(rows, j1 .. j1+m)·G` (`DGEMM('N', 'N')`).
fn mul_cols_right<T: Field + Real>(
    x: &mut Mat<T>,
    rows: Range<usize>,
    j1: usize,
    m: usize,
    g: &Small<T>,
) {
    for i in rows {
        let mut out = [T::zero(); 4];
        for (jj, o) in out.iter_mut().enumerate().take(m) {
            let mut acc = T::zero();
            for (l, g_row) in g.iter().enumerate().take(m) {
                acc = acc + x[(i, j1 + l)] * g_row[jj];
            }
            *o = acc;
        }
        for (jj, &o) in out.iter().enumerate().take(m) {
            x[(i, j1 + jj)] = o;
        }
    }
}

/// `X(j1 .. j1+m, cols) := Gᵀ·X(j1 .. j1+m, cols)` (`DGEMM('T', 'N')`).
fn mul_rows_left_t<T: Field + Real>(
    x: &mut Mat<T>,
    j1: usize,
    m: usize,
    cols: Range<usize>,
    g: &Small<T>,
) {
    for j in cols {
        let mut out = [T::zero(); 4];
        for (ii, o) in out.iter_mut().enumerate().take(m) {
            let mut acc = T::zero();
            for (l, g_row) in g.iter().enumerate().take(m) {
                acc = acc + g_row[ii] * x[(j1 + l, j)];
            }
            *o = acc;
        }
        for (ii, &o) in out.iter().enumerate().take(m) {
            x[(j1 + ii, j)] = o;
        }
    }
}

/// Re-standardizes the off-diagonal block `X(j1 .. j1+n2, j1+n2 .. j1+m)`
/// after `DLAGV2`: `X := W11ᵀ·X·TR22`, with the left rotations `wl` and the
/// right rotations `tr` (the two `DGEMM` pairs of `DTGEX2`).
fn restandardize_offdiag<T: Field + Real>(
    x: &mut Mat<T>,
    (j1, n1, n2): (usize, usize, usize),
    wl: &Small<T>,
    tr: &Small<T>,
) {
    // A(J1:J1+N2−1, J1+N2:J1+M−1) := WORK(1:N2, 1:N2)ᵀ·A(...).
    let mut tmp = small_zero();
    for (i, tmp_row) in tmp.iter_mut().enumerate().take(n2) {
        for (j, v) in tmp_row.iter_mut().enumerate().take(n1) {
            let mut acc = T::zero();
            for (l, wl_row) in wl.iter().enumerate().take(n2) {
                acc = acc + wl_row[i] * x[(j1 + l, j1 + n2 + j)];
            }
            *v = acc;
        }
    }
    for (i, tmp_row) in tmp.iter().enumerate().take(n2) {
        for (j, &v) in tmp_row.iter().enumerate().take(n1) {
            x[(j1 + i, j1 + n2 + j)] = v;
        }
    }
    // A(J1:J1+N2−1, J1+N2:J1+M−1) := A(...)·T(N2+1:M, N2+1:M).
    for (i, tmp_row) in tmp.iter_mut().enumerate().take(n2) {
        for (j, v) in tmp_row.iter_mut().enumerate().take(n1) {
            let mut acc = T::zero();
            for l in 0..n1 {
                acc = acc + x[(j1 + i, j1 + n2 + l)] * tr[n2 + l][n2 + j];
            }
            *v = acc;
        }
    }
    for (i, tmp_row) in tmp.iter().enumerate().take(n2) {
        for (j, &v) in tmp_row.iter().enumerate().take(n1) {
            x[(j1 + i, j1 + n2 + j)] = v;
        }
    }
}

/// `DTGEX2` case 2: swap a 1×1 and a 2×2 block, or two 2×2 blocks.
fn swap_blocks<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    (j1, n1, n2): (usize, usize, usize),
    (mut s, mut t): (Small<T>, Small<T>),
    th: Thresholds<T>,
    k: &Consts<T>,
) -> Result<(), SwapRejection> {
    let n = p.s.nrows();
    let m = n1 + n2;
    let (zero, one) = (T::zero(), T::one());
    let mut li = small_zero();
    let mut ir = small_zero();

    // Solve the generalized Sylvester equation
    //   S11·R − L·S22 = SCALE·S12
    //   T11·R − L·T22 = SCALE·T12
    // for R and L (DTGSY2; T12 enters in LI, S12 in IR(N2+1, N1+1)).
    let mut coef = SylvesterPencils {
        m: n1,
        n: n2,
        a: small_zero(),
        b: small_zero(),
        d: small_zero(),
        e: small_zero(),
    };
    let mut rmat = small_zero();
    let mut lmat = small_zero();
    for i in 0..n1 {
        for j in 0..n1 {
            coef.a[i][j] = s[i][j];
            coef.d[i][j] = t[i][j];
        }
        for j in 0..n2 {
            rmat[i][j] = s[i][n1 + j];
            lmat[i][j] = t[i][n1 + j];
        }
    }
    for i in 0..n2 {
        for j in 0..n2 {
            coef.b[i][j] = s[n1 + i][n1 + j];
            coef.e[i][j] = t[n1 + i][n1 + j];
        }
    }
    let sol = tgsy2(&coef, &mut rmat, &mut lmat, k);
    if sol.info != 0 {
        return Err(SwapRejection::Sylvester);
    }
    let scale = sol.scale;
    for i in 0..n1 {
        for j in 0..n2 {
            li[i][j] = lmat[i][j];
            ir[n2 + i][n1 + j] = rmat[i][j];
        }
    }

    // Compute orthogonal matrix QL:
    //   QLᵀ·LI = [TL; 0] where LI = [−L; SCALE·identity(N2)].
    for i in 0..n2 {
        for row in li.iter_mut().take(n1) {
            row[i] = -row[i];
        }
        li[n1 + i][i] = scale;
    }
    let taul = geqr2(m, n2, &mut li, k);
    org2r(m, m, n2, &mut li, &taul);

    // Compute orthogonal matrix RQ:
    //   IR·RQᵀ = [0 TR] where IR = [SCALE·identity(N1), R].
    for i in 0..n1 {
        ir[n2 + i][i] = scale;
    }
    let taur = gerq2(n2, n1, m, &mut ir, k);
    orgr2(m, m, n1, &mut ir, &taur);

    // Perform the swapping tentatively.
    let work = mul_small(m, &li, true, &s, false);
    s = mul_small(m, &work, false, &ir, true);
    let work = mul_small(m, &li, true, &t, false);
    t = mul_small(m, &work, false, &ir, true);
    let (mut scpy, mut tcpy, ircop, mut licop) = (s, t, ir, li);

    // Triangularize the B-part by an RQ factorization. Apply transformation
    // (from left) to A-part, giving S.
    let taur = gerq2(0, m, m, &mut t, k);
    ormr2(Apply::RightQt, m, &t, &taur, &mut s);
    ormr2(Apply::LeftQ, m, &t, &taur, &mut ir);
    // Compute F-norm(S21) in BRQA21. (T21 is 0.)
    let brqa21 = frob_small(&s, n2..m, 0..n2);

    // Triangularize the B-part by a QR factorization. Apply transformation
    // (from right) to A-part, giving S.
    let taul = geqr2(m, m, &mut tcpy, k);
    orm2r(Apply::LeftQt, m, &tcpy, &taul, &mut scpy);
    orm2r(Apply::RightQ, m, &tcpy, &taul, &mut licop);
    // Compute F-norm(S21) in BQRA21. (T21 is 0.)
    let bqra21 = frob_small(&scpy, n2..m, 0..n2);

    // Decide which method to use.
    //   Weak stability test: F-norm(S21) <= O(EPS·F-norm((S))).
    if bqra21 <= brqa21 && bqra21 <= th.a {
        s = scpy;
        t = tcpy;
        ir = ircop;
        li = licop;
    } else if brqa21 >= th.a {
        return Err(SwapRejection::Weak);
    }

    // Set lower triangle of B-part to zero.
    for c in 0..m {
        for row in t.iter_mut().take(m).skip(c + 1) {
            row[c] = zero;
        }
    }

    // Strong stability test (WANDS):
    //   F-norm((A − QL·S·QR)) <= O(EPS·F-norm((A))) and
    //   F-norm((B − QL·T·QR)) <= O(EPS·F-norm((B))).
    let sa = swap_residual(p.s, j1, m, (&li, &s, &ir), false);
    let sb = swap_residual(p.t, j1, m, (&li, &t, &ir), false);
    if !(sa <= th.a && sb <= th.b) {
        return Err(SwapRejection::Strong);
    }

    // The swap is accepted ("weakly" and "strongly"): set the N1-by-N2
    // (2,1)-block to zero and copy back the M-by-M diagonal block starting
    // at index J1 of (A, B).
    for row in s.iter_mut().take(m).skip(n2) {
        for v in row.iter_mut().take(n2) {
            *v = zero;
        }
    }
    for i in 0..m {
        for j in 0..m {
            p.s[(j1 + i, j1 + j)] = s[i][j];
            p.t[(j1 + i, j1 + j)] = t[i][j];
        }
    }

    // Standardize existing 2-by-2 blocks: WORK (here `wl`) collects the left
    // rotations, T (here `tr`) the right ones.
    let mut wl = small_zero();
    let mut tr = small_zero();
    wl[0][0] = one;
    tr[0][0] = one;
    if n2 > 1 {
        let g = standardize_block(p, j1, k);
        wl[0][0] = g.csl;
        wl[1][0] = g.snl;
        tr[0][0] = g.csr;
        tr[1][0] = g.snr;
        wl[0][1] = -wl[1][0];
        wl[1][1] = wl[0][0];
        tr[n2 - 1][n2 - 1] = tr[0][0];
        tr[0][1] = -tr[1][0];
    }
    wl[m - 1][m - 1] = one;
    tr[m - 1][m - 1] = one;
    if n1 > 1 {
        let g = standardize_block(p, j1 + n2, k);
        wl[n2][n2] = g.csl;
        wl[n2 + 1][n2] = g.snl;
        tr[n2][n2] = g.csr;
        tr[m - 1][m - 2] = g.snr;
        wl[m - 1][m - 1] = wl[n2][n2];
        wl[m - 2][m - 1] = -wl[n2 + 1][n2];
        tr[m - 1][m - 1] = tr[n2][n2];
        tr[m - 2][m - 1] = -tr[m - 1][m - 2];
    }
    restandardize_offdiag(p.s, (j1, n1, n2), &wl, &tr);
    restandardize_offdiag(p.t, (j1, n1, n2), &wl, &tr);
    li = mul_small(m, &li, false, &wl, false);
    ir = mul_small(m, &ir, true, &tr, false);

    // Accumulate transformations into Q and Z.
    mul_cols_right(p.q, 0..n, j1, m, &li);
    mul_cols_right(p.z, 0..n, j1, m, &ir);

    // Update (A(J1:J1+M−1, M+J1:N), B(J1:J1+M−1, M+J1:N)) and
    // (A(1:J1−1, J1:J1+M), B(1:J1−1, J1:J1+M)).
    if j1 + m < n {
        mul_rows_left_t(p.s, j1, m, (j1 + m)..n, &li);
        mul_rows_left_t(p.t, j1, m, (j1 + m)..n, &li);
    }
    if j1 > 0 {
        mul_cols_right(p.s, 0..j1, j1, m, &ir);
        mul_cols_right(p.t, 0..j1, j1, m, &ir);
    }
    Ok(())
}
