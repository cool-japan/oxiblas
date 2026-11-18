//! Bit-for-bit tests of the transpose-aware small loops (`gemm_small_trans`)
//! against a verbatim copy of their 0.2.3 implementation.
//!
//! The axpy order (`op(A) = A` with adjacent rows, at least 8 rows) now keeps
//! blocks of a column of C in registers for all `p` (`small_trans_axpy` in
//! `gemm_small.rs`) instead of storing and reloading the column for every
//! `p` (where that pays, `axpy_blocks_pay`: columns of up to 64 rows, from
//! a depth that depends on the rows and the columns of C; elsewhere it runs
//! the loop of 0.2.3 `p` by `p`),
//! and the dot order for 3 to 7 such rows computes a whole column of C
//! at a time (`small_trans_dot_columns`) instead of one element after the
//! other. Every element still gets the same multiplies and adds in the same
//! order, so every result must have the bits of 0.2.3, including NaN
//! payloads wherever at most one operand of an operation is a NaN: the
//! promise of the legacy policy (`gemm_plan::LEGACY_POLICY`).
//!
//! Which NaN survives an operation on two NaNs is the operand order the
//! compiler gives the machine instruction, not the order in the source;
//! Rust and LLVM leave it unspecified, and the legacy policy promises only
//! the NaN positions there. Measured in aarch64 release builds with
//! distinct payloads: an add of two NaNs keeps the product's NaN in 0.2.3's
//! loop and in the `p` by `p` loop here, but the accumulator's in the
//! register blocks (8 to 64 rows, whichever order the source writes); a
//! multiply of two NaNs keeps A's NaN in 0.2.3's loop but `op(B)`'s in the
//! dot columns (3 to 7 rows). These tests run unoptimized builds, where the
//! source order decides; the mixed family, the only one in which two
//! different NaNs meet, therefore compares NaN positions only.

use super::{gemm_small_trans, op_dims, read_op, small_trans_axpy, small_trans_dot_columns};
use crate::level3::gemm_small::{axpy_blocks_pay, small_trans_axpy_blocks};
use crate::level3::trsm::Trans;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// The `gemm_small_trans` of 0.2.3, copied verbatim (only renamed, with its
/// `scale_matrix` and `SMALL_AXPY_MIN_ROWS = 8` inlined as they were).
fn gemm_small_trans_0_2_3<T: Field>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    let m = c.nrows();
    let n = c.ncols();
    let k = op_dims(a, trans_a).1;

    if alpha == T::zero() {
        // A and B are not read when alpha == 0 (reference BLAS semantics).
        scale_matrix_0_2_3(c, beta);
        return;
    }

    if T::is_real() && trans_a == Trans::NoTrans && a.col_stride() == 1 && m >= 8 {
        // C = beta * C first (C is not read when beta == 0).
        scale_matrix_0_2_3(c, beta);
        for j in 0..n {
            for p in 0..k {
                let t = alpha * read_op(b, trans_b, p, j);
                if let (Some(a_col), Some(c_col)) = (a.col_as_slice(p), c.col_as_slice_mut(j)) {
                    for (ci, &ai) in c_col.iter_mut().zip(a_col) {
                        *ci += ai * t;
                    }
                    continue;
                }
                // Not reached (`p < k`, `j < n`, unit row strides); kept so the
                // update never depends on that.
                for i in 0..m {
                    let v = c[(i, j)] + a[(i, p)] * t;
                    c.set(i, j, v);
                }
            }
        }
        return;
    }

    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            for p in 0..k {
                acc = acc + read_op(a, trans_a, i, p) * read_op(b, trans_b, p, j);
            }
            let scaled = alpha * acc;
            let val = if beta == T::zero() {
                scaled
            } else {
                scaled + beta * c[(i, j)]
            };
            c.set(i, j, val);
        }
    }
}

/// `scale_matrix` of 0.2.3 (unchanged since).
fn scale_matrix_0_2_3<T: Field>(c: &mut MatMut<'_, T>, beta: T) {
    if beta == T::zero() {
        c.fill_zero();
    } else if beta != T::one() {
        c.scale(beta);
    }
}

/// The two real element types, with what the tests need to build values.
trait Real: Field + bytemuck::Zeroable {
    fn of(x: f64) -> Self;
    fn to_bits64(self) -> u64;
    fn is_nan_value(self) -> bool;
    /// A quiet NaN with payload `p` (non-zero low bits), negative if `neg`.
    fn qnan(p: u32, neg: bool) -> Self;
    /// A signaling NaN with payload `p` (non-zero low bits).
    fn snan(p: u32) -> Self;
    fn inf() -> Self;
    fn tiny() -> Self;
}

impl Real for f64 {
    fn of(x: f64) -> Self {
        x
    }
    fn to_bits64(self) -> u64 {
        self.to_bits()
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn qnan(p: u32, neg: bool) -> Self {
        let sign = if neg { 1u64 << 63 } else { 0 };
        f64::from_bits(sign | 0x7ff8_0000_0000_0000 | u64::from(p & 0xffff))
    }
    fn snan(p: u32) -> Self {
        f64::from_bits(0x7ff0_0000_0000_0000 | u64::from((p & 0xffff) | 1))
    }
    fn inf() -> Self {
        f64::INFINITY
    }
    fn tiny() -> Self {
        // A subnormal: products with |x| < 1 underflow further.
        f64::from_bits(0x0000_0000_0000_1234)
    }
}

impl Real for f32 {
    fn of(x: f64) -> Self {
        x as f32
    }
    fn to_bits64(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn qnan(p: u32, neg: bool) -> Self {
        let sign = if neg { 1u32 << 31 } else { 0 };
        f32::from_bits(sign | 0x7fc0_0000 | (p & 0x3fff))
    }
    fn snan(p: u32) -> Self {
        f32::from_bits(0x7f80_0000 | (p & 0x3fff) | 1)
    }
    fn inf() -> Self {
        f32::INFINITY
    }
    fn tiny() -> Self {
        f32::from_bits(0x0000_1234)
    }
}

/// Which special values the operands carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Finite values, `±0` included.
    Plain,
    /// Distinct NaN payloads in A, at most one per row of `op(A)`.
    NanA,
    /// Distinct NaN payloads in B, at most one per column of `op(B)`.
    NanB,
    /// Distinct NaN payloads in C (seen when `beta != 0`).
    NanC,
    /// Infinities in A (at most one per row of `op(A)`) and zeros in B, so
    /// `inf * 0` and `inf - inf` give the default NaN.
    Inf,
    /// Subnormal values and products that underflow.
    Subnormal,
    /// All of the above at once (operations on two NaNs happen).
    Mixed,
}

const FAMILIES: [Family; 7] = [
    Family::Plain,
    Family::NanA,
    Family::NanB,
    Family::NanC,
    Family::Inf,
    Family::Subnormal,
    Family::Mixed,
];

/// A finite value of operand `salt` at `(i, j)`, `±0` included.
fn finite<T: Real>(i: usize, j: usize, salt: usize) -> T {
    let x = (i * 7 + j * 13 + salt * 5) % 23;
    match x {
        11 => T::of(0.0),
        12 => T::of(-0.0),
        _ => T::of(x as f64 / 11.0 - 1.0),
    }
}

/// `op(X)` element `(r, c)` of an operand of `family` (`which` is 0 for A,
/// 1 for B, 2 for C), before the transpose is applied to its storage.
fn value<T: Real>(family: Family, which: usize, r: usize, c: usize) -> T {
    let plain = finite::<T>(r, c, which);
    let tag = (r * 31 + c * 17 + which) as u32;
    let mixed = family == Family::Mixed;
    match which {
        // op(A) is m × k: at most one special value per row r, at column
        // r % 5 (rows of op(A) with fewer columns have none).
        0 if (family == Family::NanA || mixed) && c == r % 5 => {
            if r.is_multiple_of(3) {
                T::snan(tag + 2)
            } else {
                T::qnan(tag + 2, r % 2 == 1)
            }
        }
        0 if (family == Family::Inf || mixed) && c == (r + 2) % 5 => {
            if r.is_multiple_of(2) {
                T::inf()
            } else {
                T::of(-1.0) * T::inf()
            }
        }
        0 if (family == Family::Subnormal || mixed) && (r + c).is_multiple_of(3) => T::tiny(),
        // op(B) is k × n: at most one special value per column c, at row
        // c % 4.
        1 if (family == Family::NanB || mixed) && r == c % 4 => {
            T::qnan(tag + 3, c.is_multiple_of(2))
        }
        1 if (family == Family::Inf || mixed) && (r + c) % 4 == 2 => T::of(0.0),
        1 if (family == Family::Subnormal || mixed) && (r + c) % 2 == 1 => T::tiny(),
        2 if (family == Family::NanC || mixed) && (r + 2 * c).is_multiple_of(3) => {
            T::qnan(tag + 5, (r + c).is_multiple_of(2))
        }
        2 if (family == Family::Subnormal || mixed) && (r + c) % 4 == 1 => T::tiny(),
        _ => plain,
    }
}

/// A `rows × cols` view into a matrix with `pad` extra rows (a leading
/// dimension larger than `rows` when `pad > 0`), holding `op(X)` for an
/// operand stored transposed when `trans` is set.
fn operand<T: Real>(
    family: Family,
    which: usize,
    trans: Trans,
    op_rows: usize,
    op_cols: usize,
    pad: usize,
) -> Mat<T> {
    let (rows, cols) = if trans == Trans::NoTrans {
        (op_rows, op_cols)
    } else {
        (op_cols, op_rows)
    };
    let mut x: Mat<T> = Mat::zeros(rows + pad, cols);
    for j in 0..cols {
        for i in 0..rows + pad {
            x[(i, j)] = if i >= rows {
                // Padding rows: a value the product must never read.
                T::qnan(0x0bad, false)
            } else if trans == Trans::NoTrans {
                value(family, which, i, j)
            } else {
                value(family, which, j, i)
            };
        }
    }
    x
}

/// Runs both implementations on the same inputs; returns the two C
/// storages (the view's rows and the padding rows).
fn run_both<T: Real>(
    family: Family,
    (ta, tb): (Trans, Trans),
    (m, k, n): (usize, usize, usize),
    (alpha, beta): (T, T),
    pad: usize,
) -> (Mat<T>, Mat<T>) {
    let a_store = operand::<T>(family, 0, ta, m, k, pad);
    let b_store = operand::<T>(family, 1, tb, k, n, pad);
    let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
    let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
    let a = a_store.submatrix(0, 0, ar, ac);
    let b = b_store.submatrix(0, 0, br, bc);
    let c0 = operand::<T>(family, 2, Trans::NoTrans, m, n, pad);
    let mut c_new = c0.clone();
    let mut c_old = c0;
    gemm_small_trans(
        ta,
        tb,
        alpha,
        &a,
        &b,
        beta,
        &mut c_new.submatrix_mut(0, 0, m, n),
    );
    gemm_small_trans_0_2_3(
        ta,
        tb,
        alpha,
        &a,
        &b,
        beta,
        &mut c_old.submatrix_mut(0, 0, m, n),
    );
    (c_new, c_old)
}

/// Every element of the two C storages has the same bits (`exact`), or the
/// same NaN positions and the same bits wherever it is not a NaN.
fn compare<T: Real>(new: &Mat<T>, old: &Mat<T>, exact: bool, what: &str) {
    for j in 0..new.ncols() {
        for i in 0..new.nrows() {
            let (x, y) = (new[(i, j)], old[(i, j)]);
            let same = if exact || !(x.is_nan_value() || y.is_nan_value()) {
                x.to_bits64() == y.to_bits64()
            } else {
                x.is_nan_value() && y.is_nan_value()
            };
            assert!(
                same,
                "{what}: C({i}, {j}) is {:#x}, the reference gives {:#x}",
                x.to_bits64(),
                y.to_bits64()
            );
        }
    }
}

const COMBOS: [(Trans, Trans); 4] = [
    (Trans::NoTrans, Trans::NoTrans),
    (Trans::NoTrans, Trans::Trans),
    (Trans::Trans, Trans::NoTrans),
    (Trans::Trans, Trans::Trans),
];

/// Rows from 1 to 129: the dot order below 8 rows (the inline loop for one
/// and two rows, whole columns of C at a time from 3 to 7), register blocks
/// up to 64 rows (one block up to 16, then blocks of 16 and two blocks of 8
/// to 16), the `p` by `p` loop above 64 (65 to 129 rows). With the depths
/// of [`shapes_bitwise`], columns of 8 to 64 rows take both loops (`p` by
/// `p` in the shallow products, `axpy_blocks_pay`).
const SHAPE_ROWS: [usize; 36] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 24, 31, 32, 33, 40, 47,
    48, 49, 63, 64, 65, 66, 70, 80, 128, 129,
];

fn shapes_bitwise<T: Real>() {
    for m in SHAPE_ROWS {
        for k in [1usize, 2, 3, 4, 5, 7, 16, 17, 33] {
            for n in [1usize, 3, 9] {
                for combo in COMBOS {
                    for (alpha, beta) in [(1.0, 0.0), (0.75, 1.0), (-2.5, 0.5)] {
                        let (new, old) = run_both::<T>(
                            Family::Plain,
                            combo,
                            (m, k, n),
                            (T::of(alpha), T::of(beta)),
                            0,
                        );
                        let what = format!("{combo:?} {m}x{k}x{n} alpha={alpha} beta={beta}");
                        compare(&new, &old, true, &what);
                    }
                }
            }
        }
    }
}

fn special_values_bitwise<T: Real>() {
    for family in FAMILIES {
        for m in [2usize, 3, 4, 6, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65] {
            for k in [1usize, 5, 17] {
                for n in [1usize, 4] {
                    for combo in COMBOS {
                        for (alpha, beta) in [(1.0, 0.0), (-0.5, 1.0), (1.5, -2.0)] {
                            let (new, old) = run_both::<T>(
                                family,
                                combo,
                                (m, k, n),
                                (T::of(alpha), T::of(beta)),
                                0,
                            );
                            let what = format!(
                                "{family:?} {combo:?} {m}x{k}x{n} alpha={alpha} beta={beta}"
                            );
                            compare(&new, &old, family != Family::Mixed, &what);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_trans_axpy_bitwise_shapes_f64() {
    shapes_bitwise::<f64>();
}

#[test]
fn test_gemm_small_trans_axpy_bitwise_shapes_f32() {
    shapes_bitwise::<f32>();
}

#[test]
fn test_gemm_small_trans_axpy_bitwise_special_values_f64() {
    special_values_bitwise::<f64>();
}

#[test]
fn test_gemm_small_trans_axpy_bitwise_special_values_f32() {
    special_values_bitwise::<f32>();
}

/// The rows of C below its view (`m..` of every column) still hold the
/// padding value `operand` wrote there.
fn assert_padding_untouched<T: Real>(c: &Mat<T>, m: usize, what: &str) {
    for j in 0..c.ncols() {
        for i in m..c.nrows() {
            assert_eq!(
                c[(i, j)].to_bits64(),
                T::qnan(0x0bad, false).to_bits64(),
                "{what}: padding row {i} of column {j} was written"
            );
        }
    }
}

fn padded_views<T: Real>() {
    for m in [8usize, 10, 16, 17, 40, 64, 65] {
        for combo in COMBOS {
            let what = format!("padded {combo:?} m={m}");
            let (new, old) = run_both::<T>(
                Family::Plain,
                combo,
                (m, 6, 5),
                (T::of(1.25), T::of(-0.5)),
                3,
            );
            compare(&new, &old, true, &what);
            assert_padding_untouched(&new, m, &what);
        }
    }
}

/// Views with a leading dimension larger than their rows (A, B and C):
/// the same bits as 0.2.3, and the padding rows of C stay untouched (the
/// register blocks read and write exactly the rows of the view).
#[test]
fn test_gemm_small_trans_axpy_padded_views() {
    padded_views::<f64>();
    padded_views::<f32>();
}

/// An A whose rows are not adjacent (a transposed view used as `NoTrans`)
/// takes the dot order, as in 0.2.3; and `alpha == 0` reads neither A nor
/// B.
#[test]
fn test_gemm_small_trans_axpy_gates() {
    let (m, k, n) = (12usize, 7usize, 4usize);
    let a_t: Mat<f64> = operand::<f64>(Family::Plain, 0, Trans::Trans, m, k, 0);
    // `a_t` stores op(A)^T (k × m); its transposed view is m × k with
    // col_stride != 1.
    let a_view = a_t.as_ref().transpose();
    assert_ne!(a_view.col_stride(), 1);
    let b: Mat<f64> = operand::<f64>(Family::Plain, 1, Trans::Trans, k, n, 0);
    let b_view = b.as_ref();
    let c0: Mat<f64> = operand::<f64>(Family::NanC, 2, Trans::NoTrans, m, n, 0);
    for (alpha, beta) in [(1.0, 0.0), (0.5, 2.0), (0.0, 0.0), (0.0, 3.0)] {
        let mut c_new = c0.clone();
        let mut c_old = c0.clone();
        gemm_small_trans(
            Trans::NoTrans,
            Trans::Trans,
            alpha,
            &a_view,
            &b_view,
            beta,
            &mut c_new.as_mut(),
        );
        gemm_small_trans_0_2_3(
            Trans::NoTrans,
            Trans::Trans,
            alpha,
            &a_view,
            &b_view,
            beta,
            &mut c_old.as_mut(),
        );
        compare(
            &c_new,
            &c_old,
            true,
            &format!("strided A alpha={alpha} beta={beta}"),
        );
    }
}

/// `small_trans_axpy` on an A whose rows are not adjacent, which
/// `gemm_small_trans` never passes to it: no column of A is a slice, so
/// every step goes element by element (`axpy_rows_by_element` in the
/// register blocks; where the `p` by `p` loop runs, above 64 rows, in every
/// product of depth 2 with one column here and in some of depth 5, it hands
/// the whole product to `small_trans_axpy_by_element`). The operations per
/// element are those of
/// the slice paths, so the bits are those of the same call on a
/// column-major A with the same values.
fn strided_a_by_element<T: Real>() {
    for (k, n) in [(5usize, 3usize), (2, 1)] {
        strided_a_by_element_at::<T>(k, n);
    }
}

fn strided_a_by_element_at<T: Real>(k: usize, n: usize) {
    for m in [8usize, 13, 16, 31, 40, 64, 65, 80, 129] {
        for tb in [Trans::NoTrans, Trans::Trans] {
            // `a_t` stores op(A)^T (k × m); its transposed view is m × k
            // with col_stride != 1 and holds the values of `a_plain`.
            let a_t = operand::<T>(Family::Plain, 0, Trans::Trans, m, k, 0);
            let a_strided = a_t.as_ref().transpose();
            assert_ne!(a_strided.col_stride(), 1);
            let a_plain = operand::<T>(Family::Plain, 0, Trans::NoTrans, m, k, 0);
            let b = operand::<T>(Family::Plain, 1, tb, k, n, 0);
            let c0 = operand::<T>(Family::Plain, 2, Trans::NoTrans, m, n, 0);
            let alpha = T::of(1.5);
            let mut c_strided = c0.clone();
            let mut c_plain = c0;
            small_trans_axpy(alpha, a_strided, tb, b.as_ref(), c_strided.as_mut(), k);
            small_trans_axpy(alpha, a_plain.as_ref(), tb, b.as_ref(), c_plain.as_mut(), k);
            compare(
                &c_strided,
                &c_plain,
                true,
                &format!("strided A {tb:?} m={m} k={k}"),
            );
        }
    }
}

#[test]
fn test_gemm_small_trans_axpy_strided_a_by_element() {
    strided_a_by_element::<f64>();
    strided_a_by_element::<f32>();
}

/// `small_trans_axpy` on columns longer than 64 rows with an A of fewer rows
/// than C: no column of A is a slice of that many rows, so the `p` by `p`
/// loop stops before its first update and the element-by-element
/// continuation panics at its checked read of A; nothing is read out of
/// bounds.
#[test]
#[should_panic(expected = "out of bounds")]
fn test_gemm_small_trans_axpy_long_columns_check_a_rows() {
    // C is 70 × 2; A has 69 rows.
    let a: Mat<f64> = Mat::zeros(69, 3);
    let b: Mat<f64> = Mat::zeros(3, 2);
    let mut c: Mat<f64> = Mat::zeros(70, 2);
    small_trans_axpy(1.0, a.as_ref(), Trans::NoTrans, b.as_ref(), c.as_mut(), 3);
}

/// The same for a shallow product of shorter columns, which takes the `p`
/// by `p` loop too (`axpy_blocks_pay`: 40 rows at depth 2).
#[test]
#[should_panic(expected = "out of bounds")]
fn test_gemm_small_trans_axpy_shallow_columns_check_a_rows() {
    // C is 40 × 2; A has 39 rows.
    let a: Mat<f64> = Mat::zeros(39, 2);
    let b: Mat<f64> = Mat::zeros(2, 2);
    let mut c: Mat<f64> = Mat::zeros(40, 2);
    assert!(!axpy_blocks_pay::<f64>(40, 2, 2));
    small_trans_axpy(1.0, a.as_ref(), Trans::NoTrans, b.as_ref(), c.as_mut(), 2);
}

/// Where the register blocks are used (`axpy_blocks_pay`): the least depth
/// by the rows of a column, the columns of C and the element width (the
/// terms `(a, b)` of `axpy_blocks_terms`), and never above 64 rows.
#[test]
fn test_gemm_small_trans_axpy_blocks_pay_table() {
    // (rows, columns of C, least depth for f32, least depth for f64);
    // 0 = never.
    let table = [
        // 8 rows: (1, 3) for both.
        (8usize, 1usize, 4usize, 4usize),
        (8, 2, 3, 3),
        (8, 3, 2, 2),
        (8, 100, 2, 2),
        // 9 to 15 rows: (2, 1) for both.
        (9, 1, 3, 3),
        (9, 8, 3, 3),
        (15, 1, 3, 3),
        // 16 rows: f32 (2, 4), f64 (2, 1).
        (16, 1, 6, 3),
        (16, 2, 4, 3),
        (16, 3, 4, 3),
        (16, 4, 3, 3),
        (16, 8, 3, 3),
        // 17 to 32 rows: f32 (3, 2), f64 (3, 1).
        (20, 1, 5, 4),
        (20, 2, 4, 4),
        (24, 1, 5, 4),
        (32, 1, 5, 4),
        (32, 8, 4, 4),
        // 33 to 48 rows: f32 (5, 1), f64 (3, 1).
        (36, 1, 6, 4),
        (36, 8, 6, 4),
        (48, 1, 6, 4),
        // 49 to 64 rows: f32 (6, 1), f64 (4, 1).
        (52, 1, 7, 5),
        (64, 1, 7, 5),
        (64, 8, 7, 5),
        // 1 to 3 rows more than 16, 32 or 48: f32 (9, 1), f64 as the class.
        (17, 1, 10, 4),
        (17, 8, 10, 4),
        (19, 100, 10, 4),
        (33, 1, 10, 4),
        (35, 8, 10, 4),
        (49, 1, 10, 5),
        (51, 3, 10, 5),
        // More rows: never.
        (65, 1, 0, 0),
        (65, 8, 0, 0),
        (128, 100, 0, 0),
    ];
    for (rows, n, min_f32, min_f64) in table {
        for k in (0usize..=20).chain([100, usize::MAX]) {
            assert_eq!(
                axpy_blocks_pay::<f32>(rows, k, n),
                min_f32 != 0 && k >= min_f32,
                "f32 {rows} {k} {n}"
            );
            assert_eq!(
                axpy_blocks_pay::<f64>(rows, k, n),
                min_f64 != 0 && k >= min_f64,
                "f64 {rows} {k} {n}"
            );
        }
    }
    // No column of C: never (nothing to compute).
    assert!(!axpy_blocks_pay::<f32>(16, 100, 0));
    assert!(!axpy_blocks_pay::<f64>(16, 100, 0));
}

/// `small_trans_axpy_blocks` computes the product of every shape with the
/// operations of the loop of 0.2.3, also where `small_trans_axpy` does not
/// call it (shallow products, columns longer than 64 rows): the bits of
/// that loop (`beta = 1`, so that C is not scaled first).
fn blocks_equal_the_loop_of_0_2_3<T: Real>() {
    for m in SHAPE_ROWS {
        if m < 8 {
            continue;
        }
        for k in [1usize, 2, 3, 4, 5, 8, 17] {
            for n in [1usize, 3] {
                for tb in [Trans::NoTrans, Trans::Trans] {
                    let a = operand::<T>(Family::Plain, 0, Trans::NoTrans, m, k, 0);
                    let b = operand::<T>(Family::Plain, 1, tb, k, n, 0);
                    let c0 = operand::<T>(Family::Plain, 2, Trans::NoTrans, m, n, 0);
                    let alpha = T::of(-1.25);
                    let mut c_blocks = c0.clone();
                    let mut c_old = c0;
                    small_trans_axpy_blocks(
                        alpha,
                        &a.as_ref(),
                        tb,
                        &b.as_ref(),
                        &mut c_blocks.as_mut(),
                        k,
                    );
                    gemm_small_trans_0_2_3(
                        Trans::NoTrans,
                        tb,
                        alpha,
                        &a.as_ref(),
                        &b.as_ref(),
                        T::of(1.0),
                        &mut c_old.as_mut(),
                    );
                    compare(
                        &c_blocks,
                        &c_old,
                        true,
                        &format!("blocks against the loop of 0.2.3 {tb:?} {m}x{k}x{n}"),
                    );
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_trans_axpy_blocks_equal_the_loop_of_0_2_3() {
    blocks_equal_the_loop_of_0_2_3::<f64>();
    blocks_equal_the_loop_of_0_2_3::<f32>();
}

/// The same with an A of fewer columns than `k`: the loop stops at the
/// first missing column, and the continuation panics at its checked read.
#[test]
#[should_panic(expected = "out of bounds")]
fn test_gemm_small_trans_axpy_long_columns_check_a_cols() {
    // k = 4 (op(B) = B is 4 × 2); A has 3 columns.
    let a: Mat<f32> = Mat::zeros(70, 3);
    let b: Mat<f32> = Mat::zeros(4, 2);
    let mut c: Mat<f32> = Mat::zeros(70, 2);
    small_trans_axpy(1.0, a.as_ref(), Trans::NoTrans, b.as_ref(), c.as_mut(), 4);
}

/// The same call, looked at after its panic: the `p` by `p` loop has
/// updated the first column of C for the columns of A that exist, each
/// exactly once, the continuation has added nothing before its checked
/// read panicked, and the other columns of C are untouched.
#[test]
fn test_gemm_small_trans_axpy_long_columns_stop_at_the_missing_column() {
    let (m, k, n) = (70usize, 4usize, 2usize);
    let a_cols = 3usize;
    let a = operand::<f64>(Family::Plain, 0, Trans::NoTrans, m, a_cols, 0);
    let b = operand::<f64>(Family::Plain, 1, Trans::NoTrans, k, n, 0);
    let c0 = operand::<f64>(Family::Plain, 2, Trans::NoTrans, m, n, 0);
    let alpha = 1.5f64;
    let mut c = c0.clone();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        small_trans_axpy(alpha, a.as_ref(), Trans::NoTrans, b.as_ref(), c.as_mut(), k);
    }));
    assert!(
        outcome.is_err(),
        "an A with {a_cols} columns for k = {k} must panic"
    );
    // What the loop of 0.2.3 leaves behind at the same point.
    let mut want = c0;
    for p in 0..a_cols {
        let t = alpha * b[(p, 0)];
        for i in 0..m {
            want[(i, 0)] = a[(i, p)] * t + want[(i, 0)];
        }
    }
    compare(&c, &want, true, "C after the panic");
}

/// `small_trans_axpy` checks `op(B)` once at its entry: a `k` larger than
/// the rows of `op(B)` panics there with the check's message, before any
/// unchecked read of `op(B)` (in test builds an unchecked read out of bounds
/// would panic too, but with another message).
#[test]
#[should_panic(
    expected = "small transposed GEMM: op(B) has fewer than k rows or fewer columns than C"
)]
fn test_gemm_small_trans_axpy_checks_op_b_rows() {
    // op(A) = A is 8 × 4; op(B) = B^T is 3 × 2 (B stored 2 × 3): k = 4 > 3.
    let a: Mat<f64> = Mat::zeros(8, 4);
    let b: Mat<f64> = Mat::zeros(2, 3);
    let mut c: Mat<f64> = Mat::zeros(8, 2);
    small_trans_axpy(1.0, a.as_ref(), Trans::Trans, b.as_ref(), c.as_mut(), 4);
}

/// `small_trans_axpy`: a C with more columns than `op(B)` panics at the
/// entry check.
#[test]
#[should_panic(
    expected = "small transposed GEMM: op(B) has fewer than k rows or fewer columns than C"
)]
fn test_gemm_small_trans_axpy_checks_op_b_cols() {
    // op(B) = B is 4 × 2; C has 3 columns.
    let a: Mat<f32> = Mat::zeros(8, 4);
    let b: Mat<f32> = Mat::zeros(4, 2);
    let mut c: Mat<f32> = Mat::zeros(8, 3);
    small_trans_axpy(1.0, a.as_ref(), Trans::NoTrans, b.as_ref(), c.as_mut(), 4);
}

/// `small_trans_dot_columns`: a `k` larger than the rows of `op(B)` panics
/// at the entry check.
#[test]
#[should_panic(
    expected = "small transposed GEMM: op(B) has fewer than k rows or fewer columns than C"
)]
fn test_gemm_small_trans_dot_columns_checks_op_b_rows() {
    // op(A) = A is 5 × 4; op(B) = B is 3 × 2: k = 4 > 3.
    let a: Mat<f32> = Mat::zeros(5, 4);
    let b: Mat<f32> = Mat::zeros(3, 2);
    let mut c: Mat<f32> = Mat::zeros(5, 2);
    small_trans_dot_columns(
        1.0,
        a.as_ref(),
        Trans::NoTrans,
        b.as_ref(),
        0.0,
        c.as_mut(),
        4,
    );
}

/// `small_trans_dot_columns`: a C with more columns than `op(B)` panics at
/// the entry check.
#[test]
#[should_panic(
    expected = "small transposed GEMM: op(B) has fewer than k rows or fewer columns than C"
)]
fn test_gemm_small_trans_dot_columns_checks_op_b_cols() {
    // op(B) = B^T is 4 × 2 (B stored 2 × 4); C has 3 columns.
    let a: Mat<f64> = Mat::zeros(6, 4);
    let b: Mat<f64> = Mat::zeros(2, 4);
    let mut c: Mat<f64> = Mat::zeros(6, 3);
    small_trans_dot_columns(
        1.0,
        a.as_ref(),
        Trans::Trans,
        b.as_ref(),
        0.5,
        c.as_mut(),
        4,
    );
}

/// Both helpers accept an `op(B)` larger than `k × n` (the entry check asks
/// for at least `k` rows and `n` columns) and read only its leading `k × n`
/// part: the bits of a call on exactly that part.
fn helpers_read_leading_op_b<T: Real>() {
    let (k, n) = (5usize, 3usize);
    for tb in [Trans::NoTrans, Trans::Trans] {
        // op(B) of (k + 2) × (n + 1), and its leading k × n part.
        let big = operand::<T>(Family::Plain, 1, tb, k + 2, n + 1, 0);
        let exact = if tb == Trans::NoTrans {
            big.as_ref().submatrix(0, 0, k, n)
        } else {
            big.as_ref().submatrix(0, 0, n, k)
        };
        for m in [5usize, 12] {
            let a = operand::<T>(Family::Plain, 0, Trans::NoTrans, m, k, 0);
            let c0 = operand::<T>(Family::Plain, 2, Trans::NoTrans, m, n, 0);
            let (alpha, beta) = (T::of(1.5), T::of(-0.5));
            let mut c_big = c0.clone();
            let mut c_exact = c0.clone();
            if m >= 8 {
                small_trans_axpy(alpha, a.as_ref(), tb, big.as_ref(), c_big.as_mut(), k);
                small_trans_axpy(alpha, a.as_ref(), tb, exact, c_exact.as_mut(), k);
            } else {
                small_trans_dot_columns(
                    alpha,
                    a.as_ref(),
                    tb,
                    big.as_ref(),
                    beta,
                    c_big.as_mut(),
                    k,
                );
                small_trans_dot_columns(alpha, a.as_ref(), tb, exact, beta, c_exact.as_mut(), k);
            }
            compare(
                &c_big,
                &c_exact,
                true,
                &format!("larger op(B) {tb:?} m={m}"),
            );
        }
    }
}

#[test]
fn test_gemm_small_trans_helpers_read_leading_op_b() {
    helpers_read_leading_op_b::<f64>();
    helpers_read_leading_op_b::<f32>();
}
