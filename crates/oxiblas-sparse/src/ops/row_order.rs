//! Row-order helpers for the CSR operations in [`super::functions`].
//!
//! A [`CsrMatrix`] row is duplicate-free but its column indices need not be
//! sorted (see [`CsrMatrix::new_unchecked`]). Operations that walk a row with
//! an ascending column cursor (`spadd`, `rank1_update`) therefore read rows
//! through [`SortedRow`], which borrows a row that is already sorted (the
//! common case, at the cost of one comparison per entry) and sorts a private
//! copy of any other row.

use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Scratch space for reading CSR rows in ascending column order.
pub(super) struct SortedRow<T> {
    order: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<T>,
}

impl<T: Scalar + Clone> SortedRow<T> {
    /// Creates empty scratch space.
    pub(super) fn new() -> Self {
        Self {
            order: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Returns row `i` of `a` as `(columns, values)` with the columns in
    /// strictly ascending order.
    ///
    /// A row that is already sorted is borrowed from `a`; any other row is
    /// copied into `self` and sorted there.
    pub(super) fn get<'s>(&'s mut self, a: &'s CsrMatrix<T>, i: usize) -> (&'s [usize], &'s [T]) {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let cols = &a.col_indices()[start..end];
        let vals = &a.values()[start..end];
        if cols.windows(2).all(|pair| pair[0] < pair[1]) {
            return (cols, vals);
        }

        self.order.clear();
        self.order.extend(0..cols.len());
        // Columns are distinct within a row (CSR contract), so an unstable sort
        // yields the same order as a stable one.
        self.order.sort_unstable_by_key(|&k| cols[k]);
        self.cols.clear();
        self.cols.extend(self.order.iter().map(|&k| cols[k]));
        self.vals.clear();
        self.vals
            .extend(self.order.iter().map(|&k| vals[k].clone()));
        (&self.cols, &self.vals)
    }
}

/// Appends the row `alpha * a + beta * b` to `out_cols` / `out_vals`.
///
/// Both input rows must list their columns in strictly ascending order (as
/// [`SortedRow::get`] returns them); the appended row is then strictly
/// ascending too. Results whose magnitude is at most `T::epsilon()` are
/// dropped.
pub(super) fn merge_scaled_rows<T: Scalar + Clone + Field>(
    alpha: &T,
    (a_cols, a_vals): (&[usize], &[T]),
    beta: &T,
    (b_cols, b_vals): (&[usize], &[T]),
    out_cols: &mut Vec<usize>,
    out_vals: &mut Vec<T>,
) {
    let mut push = |col: usize, val: T| {
        if Scalar::abs(val.clone()) > T::epsilon() {
            out_cols.push(col);
            out_vals.push(val);
        }
    };

    let (mut ia, mut ib) = (0, 0);
    loop {
        match (a_cols.get(ia), b_cols.get(ib)) {
            (Some(&ac), Some(&bc)) if ac < bc => {
                push(ac, alpha.clone() * a_vals[ia].clone());
                ia += 1;
            }
            (Some(&ac), Some(&bc)) if ac > bc => {
                push(bc, beta.clone() * b_vals[ib].clone());
                ib += 1;
            }
            (Some(&ac), Some(_)) => {
                push(
                    ac,
                    alpha.clone() * a_vals[ia].clone() + beta.clone() * b_vals[ib].clone(),
                );
                ia += 1;
                ib += 1;
            }
            (Some(&ac), None) => {
                push(ac, alpha.clone() * a_vals[ia].clone());
                ia += 1;
            }
            (None, Some(&bc)) => {
                push(bc, beta.clone() * b_vals[ib].clone());
                ib += 1;
            }
            (None, None) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::functions::{rank1_update, scale, spadd, transpose_csr};
    use crate::csr::CsrMatrix;

    /// `[2 0 1; 0 3 0]` with row 0 stored unsorted as columns `[2, 0]`.
    fn unsorted() -> CsrMatrix<f64> {
        CsrMatrix::new(2, 3, vec![0, 2, 3], vec![2, 0, 1], vec![1.0, 2.0, 3.0])
            .expect("unsorted but duplicate-free rows are valid CSR")
    }

    /// The same matrix with sorted rows.
    fn sorted() -> CsrMatrix<f64> {
        CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![2.0, 1.0, 3.0]).expect("valid CSR")
    }

    /// Checks the CSR contract with the validating constructor and that every
    /// row is strictly ascending.
    fn assert_valid_sorted(m: &CsrMatrix<f64>) {
        let rebuilt = CsrMatrix::new(
            m.nrows(),
            m.ncols(),
            m.row_ptrs().to_vec(),
            m.col_indices().to_vec(),
            m.values().to_vec(),
        );
        assert!(
            rebuilt.is_ok(),
            "result violates the CSR contract: {rebuilt:?}"
        );
        for i in 0..m.nrows() {
            let row = &m.col_indices()[m.row_ptrs()[i]..m.row_ptrs()[i + 1]];
            assert!(
                row.windows(2).all(|p| p[0] < p[1]),
                "row {i} not sorted: {row:?}"
            );
        }
    }

    // --- Regression: `spadd` merged rows as if they were sorted -------------
    //
    // With an unsorted row it emitted the same column twice (a duplicate entry
    // that `get` mis-resolves) and split the sum between the two entries.

    #[test]
    fn spadd_handles_unsorted_rows() {
        let a = unsorted();
        // B = [10 0 0; 0 0 5]
        let b = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![0, 2], vec![10.0, 5.0]).expect("valid");
        let c = spadd(1.0, &a, 1.0, &b);
        assert_valid_sorted(&c);
        assert_eq!(c.get(0, 0), Some(&12.0));
        assert_eq!(c.get(0, 2), Some(&1.0));
        assert_eq!(c.get(1, 1), Some(&3.0));
        assert_eq!(c.get(1, 2), Some(&5.0));
        assert_eq!(c.nnz(), 4);

        // Unsorted on the right-hand side, and on both sides at once.
        let c2 = spadd(1.0, &b, 1.0, &a);
        assert_valid_sorted(&c2);
        assert_eq!(c2.col_indices(), c.col_indices());
        assert_eq!(c2.values(), c.values());
        let c3 = spadd(2.0, &a, -1.0, &a);
        assert_valid_sorted(&c3);
        assert_eq!(c3.col_indices(), sorted().col_indices());
        assert_eq!(c3.values(), sorted().values());
    }

    #[test]
    fn spadd_unsorted_matches_sorted() {
        let b = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![1, 0, 0], vec![4.0, -2.0, 7.0])
            .expect("valid");
        let from_unsorted = spadd(0.5, &unsorted(), 3.0, &b);
        let from_sorted = spadd(0.5, &sorted(), 3.0, &b);
        assert_valid_sorted(&from_unsorted);
        assert_eq!(from_unsorted.row_ptrs(), from_sorted.row_ptrs());
        assert_eq!(from_unsorted.col_indices(), from_sorted.col_indices());
        assert_eq!(from_unsorted.values(), from_sorted.values());
        // (0, 0): 0.5 * 2 + 3 * -2 = -5; (0, 1): 3 * 4 = 12; (0, 2): 0.5.
        assert_eq!(from_unsorted.get(0, 0), Some(&-5.0));
        assert_eq!(from_unsorted.get(0, 1), Some(&12.0));
        assert_eq!(from_unsorted.get(0, 2), Some(&0.5));
    }

    // --- Regression: `rank1_update` dropped entries of unsorted rows --------
    //
    // It scanned columns `0..n` with a cursor that only advanced on a match, so
    // an entry stored before a larger column was never matched and vanished.

    #[test]
    fn rank1_update_keeps_entries_of_unsorted_rows() {
        let a = unsorted();
        // y = 0: the update is a no-op, so the result must equal `a`.
        let c = rank1_update(&a, 1.0, &[1.0, 1.0], &[0.0, 0.0, 0.0]);
        assert_valid_sorted(&c);
        assert_eq!(c.get(0, 0), Some(&2.0));
        assert_eq!(c.get(0, 2), Some(&1.0));
        assert_eq!(c.get(1, 1), Some(&3.0));
        assert_eq!(c.nnz(), 3);

        // A real update matches the sorted input exactly.
        let x = [1.0, -2.0];
        let y = [0.5, 1.0, 2.0];
        let from_unsorted = rank1_update(&a, 2.0, &x, &y);
        let from_sorted = rank1_update(&sorted(), 2.0, &x, &y);
        assert_valid_sorted(&from_unsorted);
        assert_eq!(from_unsorted.col_indices(), from_sorted.col_indices());
        assert_eq!(from_unsorted.values(), from_sorted.values());
        // (0, 0) = 2 + 2 * 1 * 0.5 = 3; (1, 2) = 0 + 2 * -2 * 2 = -8.
        assert_eq!(from_unsorted.get(0, 0), Some(&3.0));
        assert_eq!(from_unsorted.get(1, 2), Some(&-8.0));
    }

    /// The structure-copying and transposing operations keep the contract.
    #[test]
    fn scale_and_transpose_preserve_contract() {
        let a = unsorted();
        let scaled = scale(2.0, &a);
        assert!(
            CsrMatrix::new(
                2,
                3,
                scaled.row_ptrs().to_vec(),
                scaled.col_indices().to_vec(),
                scaled.values().to_vec()
            )
            .is_ok()
        );
        assert_eq!(scaled.get(0, 0), Some(&4.0));

        let t = transpose_csr(&a);
        assert_valid_sorted(&t);
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.get(2, 0), Some(&1.0));
        assert_eq!(t.get(0, 0), Some(&2.0));
        assert_eq!(t.get(1, 1), Some(&3.0));
    }
}
