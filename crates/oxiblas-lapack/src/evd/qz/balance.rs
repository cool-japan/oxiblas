//! Permutation balancing of the pencil: `DGGBAL(JOB = 'P')` and
//! `DGGBAK(JOB = 'P')`.
//!
//! Only permutations are used (no diagonal scaling as in `JOB = 'B'`): a
//! permutation `Ã = Pₗ A Pᵣ` keeps `A = (Pₗᵀ Q̃) S (Pᵣ Z̃)ᵀ` with orthogonal
//! factors, so the back-permuted `Q` and `Z` are Schur vectors of the
//! original pencil, which diagonal scaling would destroy.

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::Mat;

/// Result of `DGGBAL('P')` (0-based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Permutation {
    /// First row/column of the unreduced middle block (`ILO − 1`).
    pub(super) ilo: usize,
    /// Last row/column of the unreduced middle block (`IHI − 1`).
    pub(super) ihi: usize,
    /// `LSCALE`: row `j` was interchanged with row `lperm[j]`
    /// (`lperm[j] = j` inside `ilo..=ihi`).
    pub(super) lperm: Vec<usize>,
    /// `RSCALE`: column `j` was interchanged with column `rperm[j]`
    /// (`rperm[j] = j` inside `ilo..=ihi`).
    pub(super) rperm: Vec<usize>,
}

/// True when `A(i,j)` or `B(i,j)` is nonzero (the pattern of `A ∪ B`).
fn nonzero<T: Field + Real>(a: &Mat<T>, b: &Mat<T>, i: usize, j: usize) -> bool {
    a[(i, j)] != T::zero() || b[(i, j)] != T::zero()
}

/// Swaps rows `r1` and `r2` of `m` over the columns `cols` (`DSWAP`).
fn swap_rows<T: Field + Real>(m: &mut Mat<T>, r1: usize, r2: usize, cols: core::ops::Range<usize>) {
    for j in cols {
        let tmp = m[(r1, j)];
        m[(r1, j)] = m[(r2, j)];
        m[(r2, j)] = tmp;
    }
}

/// Swaps columns `c1` and `c2` of `m` over the rows `rows` (`DSWAP`).
fn swap_cols<T: Field + Real>(m: &mut Mat<T>, c1: usize, c2: usize, rows: core::ops::Range<usize>) {
    for i in rows {
        let tmp = m[(i, c1)];
        m[(i, c1)] = m[(i, c2)];
        m[(i, c2)] = tmp;
    }
}

/// Row `i` restricted to columns `0..=l` has at most one nonzero (in the
/// pattern of `A ∪ B`): returns that column, or `l` when the row is empty
/// there (`DGGBAL` labels 30–70).
fn single_nonzero_in_row<T: Field + Real>(
    a: &Mat<T>,
    b: &Mat<T>,
    i: usize,
    l: usize,
) -> Option<usize> {
    match (0..l).find(|&j| nonzero(a, b, i, j)) {
        None => Some(l),
        Some(j0) => {
            if (j0 + 1..=l).any(|j| nonzero(a, b, i, j)) {
                None
            } else {
                Some(j0)
            }
        }
    }
}

/// Column `j` restricted to rows `k..=l` has at most one nonzero: returns
/// that row, or `l` when the column is empty there (`DGGBAL` labels
/// 100–140).
fn single_nonzero_in_col<T: Field + Real>(
    a: &Mat<T>,
    b: &Mat<T>,
    j: usize,
    k: usize,
    l: usize,
) -> Option<usize> {
    match (k..l).find(|&i| nonzero(a, b, i, j)) {
        None => Some(l),
        Some(i0) => {
            if (i0 + 1..=l).any(|i| nonzero(a, b, i, j)) {
                None
            } else {
                Some(i0)
            }
        }
    }
}

/// `DGGBAL(JOB = 'P')`: permutes `a` and `b` (in place) to isolate
/// eigenvalues, making the pencil block upper triangular with upper
/// triangular leading (`0..ilo`) and trailing (`ihi+1..n`) blocks.
///
/// Rows with a single nonzero in columns `0..=l` are moved to the bottom
/// (labels 20/30), then columns with a single nonzero in rows `k..=l` are
/// moved to the left (labels 90/100). `n` must be at least 1.
pub(super) fn ggbal_p<T: Field + Real>(a: &mut Mat<T>, b: &mut Mat<T>) -> Permutation {
    let n = a.nrows();
    let mut lperm: Vec<usize> = (0..n).collect();
    let mut rperm: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return Permutation {
            ilo: 0,
            ihi: n.saturating_sub(1),
            lperm,
            rperm,
        };
    }
    let mut k = 0usize;
    let mut l = n - 1;

    // Labels 20/30: find a row with one nonzero in columns 0..=l and swap it
    // (and the column of that nonzero) to position l.
    loop {
        let found = (0..=l)
            .rev()
            .find_map(|i| single_nonzero_in_row(a, b, i, l).map(|j| (i, j)));
        let Some((i, j)) = found else {
            break;
        };
        // Label 160/170 with M = L.
        lperm[l] = i;
        if i != l {
            swap_rows(a, i, l, k..n);
            swap_rows(b, i, l, k..n);
        }
        rperm[l] = j;
        if j != l {
            swap_cols(a, j, l, 0..l + 1);
            swap_cols(b, j, l, 0..l + 1);
        }
        // Label 20: L = L − 1; a 1×1 remainder ends the search.
        l -= 1;
        if l == 0 {
            lperm[0] = 0;
            rperm[0] = 0;
            return Permutation {
                ilo: k,
                ihi: l,
                lperm,
                rperm,
            };
        }
    }

    // Labels 90/100: find a column with one nonzero in rows k..=l and swap it
    // (and the row of that nonzero) to position k.
    loop {
        let found = (k..=l).find_map(|j| single_nonzero_in_col(a, b, j, k, l).map(|i| (i, j)));
        let Some((i, j)) = found else {
            break;
        };
        // Label 160/170 with M = K.
        lperm[k] = i;
        if i != k {
            swap_rows(a, i, k, k..n);
            swap_rows(b, i, k, k..n);
        }
        rperm[k] = j;
        if j != k {
            swap_cols(a, j, k, 0..l + 1);
            swap_cols(b, j, k, 0..l + 1);
        }
        // Label 90: K = K + 1. (Every row in k..=l keeps two nonzeros in
        // columns k..=l, so k < l holds here; the guard is defensive.)
        k += 1;
        if k >= l {
            break;
        }
    }

    // Label 190 (JOB = 'P'): no scaling inside ilo..=ihi.
    for j in k..=l {
        lperm[j] = j;
        rperm[j] = j;
    }
    Permutation {
        ilo: k,
        ihi: l,
        lperm,
        rperm,
    }
}

/// `DGGBAK(JOB = 'P')`: applies the inverse of the `DGGBAL` permutation
/// `perm` to the rows of `v` (`LSCALE` for left vectors / `Q`, `RSCALE` for
/// right vectors / `Z`): rows `i ↔ perm[i]` for `i = ilo−1` down to `0`,
/// then for `i = ihi+1` up to `n−1`.
pub(super) fn ggbak_p<T: Field + Real>(v: &mut Mat<T>, ilo: usize, ihi: usize, perm: &[usize]) {
    let n = v.nrows();
    let ncols = v.ncols();
    for i in (0..ilo).rev() {
        let k = perm[i];
        if k != i {
            swap_rows(v, i, k, 0..ncols);
        }
    }
    for i in (ihi + 1)..n {
        let k = perm[i];
        if k != i {
            swap_rows(v, i, k, 0..ncols);
        }
    }
}
