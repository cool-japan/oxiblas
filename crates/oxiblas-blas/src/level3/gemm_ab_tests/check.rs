//! Operands, the reference result and the comparisons of the correctness
//! pass.
//!
//! Route groups compare every variant (A included) with an independent
//! reference `C_ref` computed from the stored operands with a compensated
//! dot product in f64 (Dekker's TwoProduct and Knuth's TwoSum: as accurate as
//! a computation in twice the f64 precision, far beyond f32 and f64), with the
//! per-element tolerance
//!
//! `tol = min(γ_(k+2)·S, 8·u·√k·N2) + 2·u·|C_ref|`
//!
//! where `S = Σ_p |a_ip·b_pj|`, `N2 = √(Σ_p (a_ip·b_pj)²)`, `u` the unit
//! roundoff of the element type and `γ_j = j·u / (1 − j·u)` (`k` counts the
//! `beta·C` term too when `beta = 1`). The first term is the rigorous worst
//! case of any summation order; the second a statistical bound for sums of
//! pseudo-random terms of mixed sign (the harness's data), about twenty
//! standard deviations of recursive summation wide. The last term covers the
//! final rounding to the element type. Bitwise groups check A against the
//! reference too (computed column by column on several threads and never
//! stored whole, see [`check_streamed`]) and compare the candidates and the
//! control with A bit for bit.

use super::super::gemm_kernel::GemmKernel;
use super::super::trsm::Trans;
use super::specs::Beta;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;

/// The real element types the harness runs.
pub(super) trait AbReal: Field + GemmKernel + bytemuck::Zeroable + Send + Sync {
    /// Unit roundoff (half the machine epsilon).
    const UNIT_ROUNDOFF: f64;
    /// `x` rounded to this type.
    fn of(x: f64) -> Self;
    /// This value in f64 (exact).
    fn val(self) -> f64;
    /// The bits of this value, widened.
    fn bits(self) -> u64;
}

impl AbReal for f64 {
    const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;
    fn of(x: f64) -> Self {
        x
    }
    fn val(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl AbReal for f32 {
    const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;
    fn of(x: f64) -> Self {
        x as f32
    }
    fn val(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// Seeds of the operands A and B and of the initial C of a `beta = 1` case.
pub(super) const SEED_A: u64 = 1;
pub(super) const SEED_B: u64 = 2;
pub(super) const SEED_C: u64 = 3;

/// A deterministic value in `[-1, 1)` for element `(i, j)` of operand `seed`
/// (splitmix64 of the position).
pub(super) fn sample_value(i: usize, j: usize, seed: u64) -> f64 {
    let (i, j) = (i as u64, j as u64);
    let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(j.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    z = z.wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
}

/// Rows and columns of the stored matrix of an `rows × cols` `op(X)`.
pub(super) const fn stored_shape(trans: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match trans {
        Trans::NoTrans => (rows, cols),
        Trans::Trans | Trans::ConjTrans => (cols, rows),
    }
}

/// The stored operand of an `rows × cols` logical operand `op(X)`.
pub(super) fn operand<T: AbReal>(trans: Trans, rows: usize, cols: usize, seed: u64) -> Mat<T> {
    let (r, c) = stored_shape(trans, rows, cols);
    let mut x = Mat::<T>::zeros(r, c);
    for j in 0..c {
        (0..r).for_each(|i| x[(i, j)] = T::of(sample_value(i, j, seed)));
    }
    x
}

/// The C a correctness call starts from: NaN everywhere for `beta = 0` (C is
/// not read, so a NaN left over shows an element that was not written), the
/// values of seed [`SEED_C`] for `beta = 1`.
pub(super) fn initial_c<T: AbReal>(m: usize, n: usize, beta: Beta) -> Mat<T> {
    let mut c = Mat::<T>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            c[(i, j)] = match beta {
                Beta::Zero => T::of(f64::NAN),
                Beta::One => T::of(sample_value(i, j, SEED_C)),
            };
        }
    }
    c
}

/// `a` split into two halves with 26 significant bits each (Veltkamp).
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// `a·b = p + e` exactly (Dekker; no fused multiply-add needed).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// `a + b = s + e` exactly (Knuth).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// The compensated sums of one column's elements with their `S` and
/// `Σ term²`, as arrays over the rows (so that the loop over the rows of a
/// column runs on SIMD lanes; every element still gets its operations in the
/// same order).
#[derive(Debug, Clone, Default)]
struct ColumnAcc {
    sum: Vec<f64>,
    compensation: Vec<f64>,
    abs_sum: Vec<f64>,
    sq_sum: Vec<f64>,
}

/// Adds the term `x·y` to one element's sums.
#[inline]
fn add_term(
    (sum, compensation): (&mut f64, &mut f64),
    (abs_sum, sq_sum): (&mut f64, &mut f64),
    x: f64,
    y: f64,
) {
    let (p, e) = two_prod(x, y);
    let (s, q) = two_sum(*sum, p);
    *sum = s;
    *compensation += q + e;
    *abs_sum += p.abs();
    *sq_sum += p * p;
}

impl ColumnAcc {
    /// Empties the sums for a column of `m` rows.
    fn reset(&mut self, m: usize) {
        for part in [
            &mut self.sum,
            &mut self.compensation,
            &mut self.abs_sum,
            &mut self.sq_sum,
        ] {
            part.clear();
            part.resize(m, 0.0);
        }
    }

    /// Adds the term `x[i]·y` to every row `i`.
    fn add_column<T: AbReal>(&mut self, x: &[T], y: f64) {
        let sums = self.sum.iter_mut().zip(self.compensation.iter_mut());
        let squares = self.abs_sum.iter_mut().zip(self.sq_sum.iter_mut());
        for ((sums, squares), xi) in sums.zip(squares).zip(x) {
            add_term(sums, squares, xi.val(), y);
        }
    }

    /// Adds the terms `x[p]·y[p]`, in the order of `p`, to row `i`.
    fn add_dot<T: AbReal>(&mut self, i: usize, x: &[T], y: impl Iterator<Item = f64>) {
        let (Some(sum), Some(compensation), Some(abs_sum), Some(sq_sum)) = (
            self.sum.get_mut(i),
            self.compensation.get_mut(i),
            self.abs_sum.get_mut(i),
            self.sq_sum.get_mut(i),
        ) else {
            return;
        };
        for (xp, yp) in x.iter().zip(y) {
            let sums = (&mut *sum, &mut *compensation);
            add_term(sums, (&mut *abs_sum, &mut *sq_sum), xp.val(), yp);
        }
    }

    /// The finished element of row `i`.
    fn elem(&self, i: usize) -> RefElem {
        RefElem {
            value: self.sum[i] + self.compensation[i],
            abs_sum: self.abs_sum[i],
            sq_sum: self.sq_sum[i],
        }
    }

    /// The finished elements of every row.
    fn elems(&self) -> impl Iterator<Item = RefElem> + '_ {
        (0..self.sum.len()).map(|i| self.elem(i))
    }
}

/// One element of the reference: its value, `S` and `Σ term²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RefElem {
    pub(super) value: f64,
    pub(super) abs_sum: f64,
    pub(super) sq_sum: f64,
}

/// The reference result of one case (column-major `m × n`) and the number
/// of terms of every element.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Reference {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) terms: usize,
    pub(super) elems: Vec<RefElem>,
}

/// The operands of a product as the harness stores them.
#[derive(Clone, Copy)]
pub(super) struct Operands<'a, T: AbReal> {
    pub(super) trans_a: Trans,
    pub(super) trans_b: Trans,
    pub(super) a: &'a Mat<T>,
    pub(super) b: &'a Mat<T>,
    /// The initial C of a `beta = 1` product (`None`: `beta = 0`).
    pub(super) c0: Option<&'a Mat<T>>,
}

/// Column `j` of `x` as a slice (the harness's matrices are column-major
/// with adjacent rows).
fn col<T: AbReal>(x: &Mat<T>, j: usize) -> &[T] {
    let column = x.as_ref().col_as_slice(j);
    column.unwrap_or_else(|| panic!("column {j} of a harness matrix is not contiguous"))
}

/// Every column of `x` as a slice.
fn columns<T: AbReal>(x: &Mat<T>) -> Vec<&[T]> {
    (0..x.ncols()).map(|j| col(x, j)).collect()
}

/// The stored operands of a product as column slices, which threads can
/// share.
struct Columns<'a, T> {
    trans_a: Trans,
    trans_b: Trans,
    a: Vec<&'a [T]>,
    b: Vec<&'a [T]>,
    c0: Option<Vec<&'a [T]>>,
}

impl<'a, T: AbReal> Columns<'a, T> {
    fn of(ops: &Operands<'a, T>) -> Self {
        Self {
            trans_a: ops.trans_a,
            trans_b: ops.trans_b,
            a: columns(ops.a),
            b: columns(ops.b),
            c0: ops.c0.map(columns),
        }
    }

    /// Column `j` (of `m` rows) of the reference `op(A)·op(B) (+ C0)` into
    /// `acc` (reset first), walking the stored operands along their columns.
    /// Every element gets its terms in the order of `p`, then the `C0` term.
    fn reference_column(&self, j: usize, m: usize, acc: &mut ColumnAcc) {
        acc.reset(m);
        let trans_b = self.trans_b != Trans::NoTrans;
        // op(B)[p, j]: column j of B, or (B stored n × k) element j of
        // column p.
        let b_at = |p: usize| -> f64 {
            if trans_b {
                self.b[p][j].val()
            } else {
                self.b[j][p].val()
            }
        };
        if self.trans_a == Trans::NoTrans {
            // Column p of A is column p of op(A): all rows at once.
            for (p, a_col) in self.a.iter().enumerate() {
                acc.add_column(a_col, b_at(p));
            }
        } else {
            // A stored k × m: column i holds op(A)[i, :], a dot per row.
            for (i, a_col) in self.a.iter().enumerate() {
                acc.add_dot(i, a_col, (0..a_col.len()).map(&b_at));
            }
        }
        if let Some(c0) = &self.c0 {
            acc.add_column(c0[j], 1.0);
        }
    }
}

/// The reference `op(A)·op(B) (+ C0)` of `ops` (`m × k` by `k × n`).
pub(super) fn reference<T: AbReal>(ops: &Operands<'_, T>, [m, k, n]: [usize; 3]) -> Reference {
    let stored = Columns::of(ops);
    let mut acc = ColumnAcc::default();
    let mut elems = Vec::with_capacity(m * n);
    for j in 0..n {
        stored.reference_column(j, m, &mut acc);
        elems.extend(acc.elems());
    }
    let terms = k + usize::from(ops.c0.is_some());
    Reference { m, n, terms, elems }
}

/// `γ_j = j·u / (1 − j·u)` (infinite when `j·u >= 1`).
fn gamma(j: usize, u: f64) -> f64 {
    let ju = j as f64 * u;
    if ju < 1.0 {
        ju / (1.0 - ju)
    } else {
        f64::INFINITY
    }
}

/// The tolerance of one element of `terms` terms for unit roundoff `u` (see
/// the module documentation).
pub(super) fn tolerance(elem: &RefElem, terms: usize, u: f64) -> f64 {
    let rigorous = gamma(terms + 2, u) * elem.abs_sum;
    let statistical = 8.0 * u * (terms as f64).sqrt() * elem.sq_sum.sqrt();
    rigorous.min(statistical) + 2.0 * u * elem.value.abs()
}

/// Checks the element `(i, j)` of a result, `got`, against its reference
/// `elem` of `terms` terms; returns `|error| / tolerance` (0 for a zero
/// tolerance met exactly).
fn check_element<T: AbReal>(
    (i, j): (usize, usize),
    got: T,
    elem: &RefElem,
    terms: usize,
) -> Result<f64, String> {
    let tol = tolerance(elem, terms, T::UNIT_ROUNDOFF);
    let (gv, err) = (got.val(), (got.val() - elem.value).abs());
    if !(gv.is_finite() && err <= tol) {
        let want = elem.value;
        return Err(format!(
            "({i}, {j}): {gv:e} against the reference {want:e}: error {err:e} above the \
             tolerance {tol:e} (S {:e}, N2 {:e}, {terms} terms)",
            elem.abs_sum,
            elem.sq_sum.sqrt(),
        ));
    }
    Ok(if tol > 0.0 { err / tol } else { 0.0 })
}

/// The shape check of a result against `m × n`.
fn check_shape<T: AbReal>(got: &Mat<T>, (m, n): (usize, usize)) -> Result<(), String> {
    if (got.nrows(), got.ncols()) == (m, n) {
        Ok(())
    } else {
        Err(format!(
            "shape {}x{} against {m}x{n}",
            got.nrows(),
            got.ncols()
        ))
    }
}

/// Checks every element of `got` against `reference` with [`tolerance`];
/// returns the largest `|error| / tolerance` (the error names the first
/// element out of tolerance).
pub(super) fn check_reference<T: AbReal>(
    got: &Mat<T>,
    reference: &Reference,
) -> Result<f64, String> {
    let (m, n) = (reference.m, reference.n);
    check_shape(got, (m, n))?;
    let mut worst = 0.0_f64;
    for j in 0..n {
        for (i, g) in col(got, j).iter().enumerate() {
            let elem = &reference.elems[i + j * m];
            worst = worst.max(check_element((i, j), *g, elem, reference.terms)?);
        }
    }
    Ok(worst)
}

/// Multiply-adds of the reference from which [`check_streamed`] uses more
/// than one thread (one more thread per this many, up to the cores).
const STREAM_WORK_PER_THREAD: usize = 1 << 20;

/// [`check_reference`] of `got` against the reference of `ops` (`m × k` by
/// `k × n`), computed column by column and compared at once, never stored
/// whole: for A's result in the bitwise groups, whose products are too large
/// to keep a reference of. The columns are shared out among threads (up to
/// the available cores); the result is the one of [`check_reference`] (the
/// first failing column's error when several fail).
pub(super) fn check_streamed<T: AbReal>(
    ops: &Operands<'_, T>,
    [m, k, n]: [usize; 3],
    got: &Mat<T>,
) -> Result<f64, String> {
    check_shape(got, (m, n))?;
    let (stored, got_columns) = (Columns::of(ops), columns(got));
    let terms = k + usize::from(ops.c0.is_some());
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let work = m.saturating_mul(k).saturating_mul(n);
    let threads = cores.min(n).min(work / STREAM_WORK_PER_THREAD).max(1);
    let share = |first: usize| -> Result<f64, (usize, String)> {
        let mut acc = ColumnAcc::default();
        let mut worst = 0.0_f64;
        for j in (first..n).step_by(threads) {
            stored.reference_column(j, m, &mut acc);
            for (i, (elem, g)) in acc.elems().zip(got_columns[j]).enumerate() {
                let ratio = check_element((i, j), *g, &elem, terms);
                worst = worst.max(ratio.map_err(|e| (j, e))?);
            }
        }
        Ok(worst)
    };
    let shares: Vec<Result<f64, (usize, String)>> = std::thread::scope(|scope| {
        let share = &share;
        let handles: Vec<_> = (1..threads)
            .map(|first| scope.spawn(move || share(first)))
            .collect();
        let mut out = vec![share(0)];
        out.extend(handles.into_iter().map(|handle| {
            let joined = handle.join();
            joined.unwrap_or_else(|_| Err((0, "a reference thread panicked".to_string())))
        }));
        out
    });
    let mut worst = 0.0_f64;
    let mut first_error: Option<(usize, String)> = None;
    for share in shares {
        match share {
            Ok(ratio) => worst = worst.max(ratio),
            Err((j, e)) => {
                if first_error.as_ref().is_none_or(|(at, _)| j < *at) {
                    first_error = Some((j, e));
                }
            }
        }
    }
    match first_error {
        Some((_, e)) => Err(e),
        None => Ok(worst),
    }
}

/// Checks `got` against `base` bit for bit; the error names the first
/// mismatch.
pub(super) fn check_bitwise<T: AbReal>(got: &Mat<T>, base: &Mat<T>) -> Result<(), String> {
    let (m, n) = (base.nrows(), base.ncols());
    if (got.nrows(), got.ncols()) != (m, n) {
        return Err(format!(
            "shape {}x{} against {m}x{n}",
            got.nrows(),
            got.ncols()
        ));
    }
    for j in 0..n {
        for (i, (g, r)) in col(got, j).iter().zip(col(base, j)).enumerate() {
            if g.bits() != r.bits() {
                let (gv, gb, rv, rb) = (g.val(), g.bits(), r.val(), r.bits());
                return Err(format!(
                    "({i}, {j}): {gv:e} (bits {gb:#x}) against A's {rv:e} (bits {rb:#x}), bitwise"
                ));
            }
        }
    }
    Ok(())
}
