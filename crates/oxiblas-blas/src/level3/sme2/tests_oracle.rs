//! Tests of `oracle.rs` and `pack.rs`: the oracle against `args::reference`
//! (and the smoke kernels against the oracle through the self-test
//! machinery), and the plain packers against the layouts of `args.rs`. The
//! tests of the self-test itself (`selftest.rs`) live in `tests_selftest.rs`.
//!
//! Everything here runs on any aarch64 host (no SME needed) except
//! `test_sme2_oracle_smoke_kernels_subnormals`, which runs the smoke kernels
//! of `tests_smoke.rs` through the self-test machinery and skips without
//! SME2.

use core::fmt::Debug;

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::{Epilogue, MR, MacroArgs, SmeElement, reference};
use super::oracle::{oracle_block, same_bits_or_nan};
use super::pack::{
    ColMajor, Span, a_pack_len, b_pack_len, block_args, block_beta, pack_a, pack_b, spans,
};
use super::selftest::{ALayout, Case, Data, KernelFn, self_test_cases};
use super::test_support::{Need, hardware_or_skip, sme_state};
use super::tests_smoke::{smoke_tile_f32, smoke_tile_f64};
use crate::level3::gemm::read_op;
use crate::level3::trsm::Trans;

/// The element types under test, with a conversion from `f64` (shared with
/// `tests_selftest.rs`).
pub(super) trait TestFloat: SmeElement + Field + Debug {
    /// `v` rounded to the type.
    fn of(v: f64) -> Self;
    /// Whether the value is subnormal.
    fn subnormal(self) -> bool;
}

impl TestFloat for f32 {
    fn of(v: f64) -> Self {
        v as f32
    }
    fn subnormal(self) -> bool {
        self.is_subnormal()
    }
}

impl TestFloat for f64 {
    fn of(v: f64) -> Self {
        v
    }
    fn subnormal(self) -> bool {
        self.is_subnormal()
    }
}

/// Deterministic test values: mixed signs and magnitudes, a few exact zeros
/// (the generator of `tests_contract.rs`).
fn values<T: TestFloat>(len: usize, seed: u64) -> Vec<T> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            let scale = [1.0, 1.0e-3, 37.0, 0.0][(state & 3) as usize];
            T::of((unit * 2.0 - 1.0) * scale)
        })
        .collect()
}

/// A NaN that no arithmetic produces, for borders.
fn border<T: TestFloat>() -> T {
    T::from_bits64(if T::NR == 16 {
        0x7fc0_beef
    } else {
        0x7ff8_0000_0000_beef
    })
}

/// The alpha/beta pairs of `tests_contract.rs`.
const PAIRS: [(f64, f64); 8] = [
    (1.0, 0.0),
    (1.0, -0.0),
    (1.0, 1.0),
    (-1.5, 0.0),
    (0.3, 0.0),
    (-1.5, 0.25),
    (1.0, -0.5),
    (2.0, 1.0),
];

// ---------------------------------------------------------------------------
// oracle.rs
// ---------------------------------------------------------------------------

/// Runs `oracle_block` and `reference` on the same block (A packed and in
/// place; B panels `kc · NR` apart and, when `n > NR`, also padded to
/// `kc · NR + 5` with NaN in the gaps) and requires identical bits
/// everywhere, border included.
fn oracle_vs_reference<T: TestFloat>(
    m: usize,
    n: usize,
    kc: usize,
    alpha: T,
    beta: T,
    generate: fn(usize, u64) -> Vec<T>,
) {
    let seed = (m * 1_000_003 + n * 1009 + kc) as u64;
    let a = generate(m * kc, seed);
    let b = generate(kc * n, seed + 1);
    let ldc = m + 3;
    let mode = Epilogue::select(alpha, beta);
    let start = generate(ldc * (n + 1), seed + 2);
    let mut c0 = vec![border::<T>(); ldc * (n + 1)];
    for j in 0..n {
        for i in 0..m {
            c0[i + j * ldc] = if mode.reads_c() {
                start[i + j * ldc]
            } else {
                T::from_bits64(u64::MAX >> 1)
            };
        }
    }
    let panels = m.div_ceil(MR);
    let mut packed_a = vec![T::ZERO; panels * MR * kc];
    for i in 0..m {
        for p in 0..kc {
            packed_a[(i / MR) * MR * kc + p * MR + i % MR] = a[i + p * m];
        }
    }
    let b_pads: &[usize] = if n > T::NR { &[0, 5] } else { &[0] };
    for (in_place, &b_pad) in [false, true]
        .into_iter()
        .flat_map(|in_place| b_pads.iter().map(move |pad| (in_place, pad)))
    {
        // B panels `kc · NR + b_pad` apart, NaN in the gaps.
        let b_panel = T::NR * kc + b_pad;
        let mut packed_b = vec![T::ZERO; n.div_ceil(T::NR) * b_panel];
        for (index, value) in packed_b.iter_mut().enumerate() {
            if index % b_panel >= T::NR * kc {
                *value = T::from_bits64(u64::MAX >> 1);
            }
        }
        for p in 0..kc {
            for j in 0..n {
                packed_b[(j / T::NR) * b_panel + p * T::NR + j % T::NR] = b[p * n + j];
            }
        }
        let mut got = c0.clone();
        let mut want = c0.clone();
        let base = MacroArgs {
            b_panel,
            ..MacroArgs::packed(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                got.as_mut_ptr(),
                ldc,
                m,
                n,
                kc,
                alpha,
                beta,
            )
        };
        let args = if in_place {
            MacroArgs {
                a: a.as_ptr(),
                a_depth: m,
                a_panel: MR,
                ..base
            }
        } else {
            base
        };
        // SAFETY: A holds `A(i, p)` for `i < m`, `p < kc` in the chosen
        // layout, B holds `ceil(n / NR)` panels of depth `kc`, `b_panel`
        // apart, C holds `ldc * (n + 1)` elements with `ldc ≥ m`.
        unsafe { oracle_block(&args) };
        let args = MacroArgs {
            c: want.as_mut_ptr(),
            ..args
        };
        // SAFETY: the same block with C in `want`.
        unsafe { reference(&args) };
        for (index, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.bits(),
                w.bits(),
                "{m}x{n}x{kc} {mode:?} in place {in_place} b_pad {b_pad}: C({}, {})",
                index % ldc,
                index / ldc
            );
            if index % ldc >= m || index / ldc >= n {
                assert_eq!(g.bits(), border::<T>().bits(), "outside the block");
            }
        }
    }
}

/// [`oracle_vs_reference`] over the grid of `tests_contract.rs`.
fn oracle_grid<T: TestFloat>() {
    for m in [1, 7, 9, 64, 65, 130] {
        for n in [1, 6, 8, 9, 17, 33] {
            for kc in [1, 7, 65] {
                for (alpha, beta) in PAIRS {
                    oracle_vs_reference(m, n, kc, T::of(alpha), T::of(beta), values::<T>);
                }
            }
        }
    }
}

#[test]
fn test_sme2_oracle_equals_reference_f32() {
    oracle_grid::<f32>();
}

#[test]
fn test_sme2_oracle_equals_reference_f64() {
    oracle_grid::<f64>();
}

#[test]
fn test_sme2_oracle_equals_reference_special_values() {
    const SPECIALS: [f64; 9] = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0e-310,
        -3.0,
        1.0e-41,
        0.7,
    ];
    /// Values drawn from the specials.
    fn specials<T: TestFloat>(len: usize, seed: u64) -> Vec<T> {
        values::<f64>(len, seed)
            .into_iter()
            .map(|v| T::of(SPECIALS[(v.abs() * 1.0e6) as usize % SPECIALS.len()]))
            .collect()
    }
    for (m, n, kc) in [(3, 2, 2), (9, 9, 3), (65, 17, 2)] {
        for (alpha, beta) in [(1.0, 0.0), (1.0, 1.0), (-2.0, 0.0), (0.5, -3.0), (1.0, 0.3)] {
            oracle_vs_reference(m, n, kc, alpha, beta, specials::<f64>);
            oracle_vs_reference(m, n, kc, alpha as f32, beta as f32, specials::<f32>);
        }
    }
}

#[test]
fn test_sme2_oracle_does_not_read_c_in_store_modes() {
    // Finite A and B, C full of NaN: M0 and M2 give finite results, M1 and
    // M3 give NaN everywhere.
    let (m, n, kc): (usize, usize, usize) = (70, 19, 5);
    let a = values::<f64>(m.div_ceil(MR) * MR * kc, 1);
    let b = values::<f64>(n.div_ceil(8) * 8 * kc, 2);
    for (alpha, beta, reads) in [
        (1.0, 0.0, false),
        (0.5, -0.0, false),
        (1.0, 1.0, true),
        (2.0, 0.5, true),
    ] {
        let mut c = vec![f64::NAN; m * n];
        let args = MacroArgs::packed(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m,
            m,
            n,
            kc,
            alpha,
            beta,
        );
        // SAFETY: whole panels of A and B, C of `m × n` with `ldc = m`.
        unsafe { oracle_block(&args) };
        assert!(
            c.iter().all(|v| v.is_nan() == reads),
            "alpha {alpha} beta {beta}"
        );
    }
}

#[test]
fn test_sme2_oracle_same_bits_or_nan() {
    assert!(same_bits_or_nan::<f32>(&[], &[]));
    assert!(same_bits_or_nan(&[1.0_f64, f64::NAN], &[1.0, -f64::NAN]));
    assert!(same_bits_or_nan(
        &[f32::from_bits(0x7fc0_0001)],
        &[f32::from_bits(0xffc0_1234)]
    ));
    assert!(!same_bits_or_nan(&[0.0_f64], &[-0.0]));
    assert!(!same_bits_or_nan(&[f64::NAN], &[1.0]));
    assert!(!same_bits_or_nan(&[1.0_f64], &[f64::NAN]));
    assert!(!same_bits_or_nan(&[1.0_f32], &[1.0, 2.0]));
    assert!(!same_bits_or_nan(&[1.0_f32, 2.0], &[1.0]));
    assert!(!same_bits_or_nan(
        &[1.0_f32],
        &[f32::from_bits(1.0_f32.to_bits() + 1)]
    ));
    assert!(!same_bits_or_nan(&[f64::INFINITY], &[f64::NEG_INFINITY]));
}

/// `oracle_block` with `mode = 4` and `mode = u64::MAX` equals
/// `args::reference` and the oracle in M3, bit for bit (the module docs of
/// `oracle.rs` treat a mode above 3 as M3).
fn mode_above_3<T: TestFloat>() {
    let (m, n, kc): (usize, usize, usize) = (70, 19, 5);
    let a = values::<T>(m.div_ceil(MR) * MR * kc, 21);
    let b = values::<T>(n.div_ceil(T::NR) * T::NR * kc, 22);
    let c0 = values::<T>(m * n, 23);
    let (alpha, beta) = (T::of(0.3), T::of(-0.7));
    let run = |mode: u64, use_reference: bool| {
        let mut c = c0.clone();
        let args = MacroArgs {
            mode,
            ..MacroArgs::packed(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m,
                m,
                n,
                kc,
                alpha,
                beta,
            )
        };
        if use_reference {
            // SAFETY: whole panels of A and B, C of `m × n` with `ldc = m`.
            unsafe { reference(&args) };
        } else {
            // SAFETY: as above.
            unsafe { oracle_block(&args) };
        }
        c
    };
    let m3 = run(Epilogue::ScaleCombine as u64, false);
    assert!(m3.iter().any(|v| v.bits() != T::ZERO.bits()));
    for mode in [4, u64::MAX] {
        let got = run(mode, false);
        let want = run(mode, true);
        for (index, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.bits(), w.bits(), "mode {mode}: reference, index {index}");
        }
        for (index, (g, w)) in got.iter().zip(&m3).enumerate() {
            assert_eq!(g.bits(), w.bits(), "mode {mode}: M3, index {index}");
        }
    }
}

#[test]
fn test_sme2_oracle_mode_above_3_is_m3() {
    mode_above_3::<f32>();
    mode_above_3::<f64>();
}

/// The cases and scalars the smoke kernels support: `m, n ≤ NR`, `kc = 1`,
/// M0 and M2.
fn smoke_cases(nr: usize) -> Vec<Case> {
    let mut cases = Vec::new();
    for (m, n) in [(nr, nr), (5, 3), (1, nr - 1)] {
        for (layout, lda_pad) in [(ALayout::Packed, 0), (ALayout::InPlace, 2)] {
            for data in [Data::Plain, Data::Special] {
                cases.push(Case {
                    m,
                    n,
                    kc: 1,
                    layout,
                    lda_pad,
                    b_pad: 0,
                    data,
                });
            }
        }
    }
    cases
}

/// M0 and M2 only (the smoke kernels scale in every mode but M0).
const SMOKE_SCALARS: [(f64, f64); 3] = [(1.0, 0.0), (0.3, 0.0), (-1.3, -0.0)];

#[test]
fn test_sme2_oracle_smoke_kernels_subnormals() {
    // A real FMOPA (and SVE FMUL in M2) on subnormal inputs, products and
    // results, through the self-test machinery: guards, NaN-filled C, A/B
    // unchanged, bitwise against the oracle.
    if hardware_or_skip("test_sme2_oracle_smoke_kernels_subnormals", Need::F32) {
        let smoke: KernelFn<f32> = smoke_tile_f32;
        // SAFETY: `hardware_or_skip` returned true, so SME2 with SVL 512 is
        // present and this test thread holds no SME state; the cases fit the
        // smoke kernel (`m, n ≤ 16`, `kc = 1`).
        let verdict = unsafe { self_test_cases(smoke, &smoke_cases(16), &SMOKE_SCALARS) };
        assert!(verdict, "f32 smoke kernel differs from the oracle");
        assert_eq!(sme_state(), Some((0, 0)));
    }
    if hardware_or_skip("test_sme2_oracle_smoke_kernels_subnormals", Need::F64) {
        let smoke: KernelFn<f64> = smoke_tile_f64;
        // SAFETY: as above, with FEAT_SME_F64F64 (`Need::F64`); `m, n ≤ 8`.
        let verdict = unsafe { self_test_cases(smoke, &smoke_cases(8), &SMOKE_SCALARS) };
        assert!(verdict, "f64 smoke kernel differs from the oracle");
        assert_eq!(sme_state(), Some((0, 0)));
    }
}

// ---------------------------------------------------------------------------
// pack.rs
// ---------------------------------------------------------------------------

/// Poison for packing buffers: a NaN no packer writes.
fn poison<T: TestFloat>() -> T {
    T::from_bits64(if T::NR == 16 {
        0x7fc0_0bad
    } else {
        0x7ff8_0000_0000_0bad
    })
}

/// `op(view)(i, j)` for every element agrees with the normalised operand.
fn assert_same_operand<T: TestFloat>(view: &MatRef<'_, T>, trans: Trans, op: &ColMajor<'_, T>) {
    let (rows, cols) = match trans {
        Trans::NoTrans => (view.nrows(), view.ncols()),
        _ => (view.ncols(), view.nrows()),
    };
    assert_eq!((op.rows(), op.cols()), (rows, cols));
    assert_eq!(op.stored().col_stride(), 1);
    assert_ne!(op.trans(), Trans::ConjTrans);
    for i in 0..rows {
        for j in 0..cols {
            let want = read_op(view, trans, i, j);
            let got = read_op(op.stored(), op.trans(), i, j);
            assert_eq!(got.bits(), want.bits(), "({i}, {j}) {trans:?}");
        }
    }
}

#[test]
fn test_sme2_pack_normalize_views() {
    let data = values::<f64>(400, 7);
    let trans_all = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    // (rows, cols, row_stride, col_stride, accepted, keeps the orientation)
    let views = [
        (5, 4, 9, 1, true, true),   // column-major, lda > rows
        (5, 4, 1, 7, true, false),  // row-major
        (1, 6, 3, 5, true, true),   // single row, both strides ≠ 1
        (6, 1, 3, 5, true, false),  // single column, both strides ≠ 1
        (1, 1, 4, 4, true, true),   // single element
        (5, 4, 13, 2, false, true), // fully strided
        (0, 4, 1, 1, false, true),  // empty
        (4, 0, 1, 1, false, true),  // empty
    ];
    for (rows, cols, row_stride, col_stride, accepted, keeps) in views {
        let Some(view) = MatRef::from_strided(&data, rows, cols, row_stride, col_stride) else {
            panic!("view {rows}x{cols}");
        };
        for trans in trans_all {
            let op = ColMajor::normalize(view, trans);
            assert_eq!(
                op.is_some(),
                accepted,
                "{rows}x{cols} {row_stride}/{col_stride}"
            );
            if let Some(op) = op {
                assert_same_operand(&view, trans, &op);
                let plain = if trans == Trans::NoTrans {
                    Trans::NoTrans
                } else {
                    Trans::Trans
                };
                let expected = if keeps {
                    plain
                } else if plain == Trans::NoTrans {
                    Trans::Trans
                } else {
                    Trans::NoTrans
                };
                assert_eq!(op.trans(), expected);
            }
        }
        // A transposed view normalises to the same operand.
        let transposed = view.transpose();
        if let (Some(a), Some(b)) = (
            ColMajor::normalize(view, Trans::Trans),
            ColMajor::normalize(transposed, Trans::NoTrans),
        ) {
            assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
            assert_same_operand(&transposed, Trans::NoTrans, &a);
            assert_same_operand(&transposed, Trans::NoTrans, &b);
            assert_same_operand(&view, Trans::Trans, &b);
        }
    }
}

/// The source operands of the packer tests: a stored `rows × cols` matrix
/// inside a larger buffer, column-major or row-major.
struct Source<T> {
    /// The buffer.
    data: Vec<T>,
    /// Stored rows.
    rows: usize,
    /// Stored columns.
    cols: usize,
    /// Row-major storage (`row_stride == 1`).
    row_major: bool,
}

impl<T: TestFloat> Source<T> {
    /// A source with spare rows and columns.
    fn new(rows: usize, cols: usize, row_major: bool, seed: u64) -> Self {
        Self {
            data: values(rows * cols + 3 * (rows + cols) + 8, seed),
            rows,
            cols,
            row_major,
        }
    }

    /// The view (leading dimension three more than needed).
    fn view(&self) -> MatRef<'_, T> {
        let view = if self.row_major {
            MatRef::from_strided(&self.data, self.rows, self.cols, 1, self.cols + 3)
        } else {
            MatRef::from_strided(&self.data, self.rows, self.cols, self.rows + 3, 1)
        };
        let Some(view) = view else {
            panic!("source view {}x{}", self.rows, self.cols);
        };
        view
    }
}

/// Checks a packed A block against `op(A)` and the layout of `args.rs`.
fn check_packed_a<T: TestFloat>(
    view: &MatRef<'_, T>,
    trans: Trans,
    rows: Span,
    depth: Span,
    buf: &AlignedVec<T>,
    len: usize,
) {
    for (index, value) in buf.as_slice().iter().enumerate() {
        if index >= len {
            assert_eq!(value.bits(), poison::<T>().bits(), "A suffix {index}");
            continue;
        }
        let panel = index / (MR * depth.len);
        let p = index % (MR * depth.len) / MR;
        let i = panel * MR + index % MR;
        let want = if i < rows.len {
            read_op(view, trans, rows.start + i, depth.start + p)
        } else {
            T::ZERO
        };
        assert_eq!(
            value.bits(),
            want.bits(),
            "A({i}, {p}) {trans:?} {rows:?} {depth:?}"
        );
    }
}

/// Checks a packed B block against `op(B)` and the layout of `args.rs`.
fn check_packed_b<T: TestFloat>(
    view: &MatRef<'_, T>,
    trans: Trans,
    depth: Span,
    cols: Span,
    buf: &AlignedVec<T>,
    len: usize,
) {
    let nr = T::NR;
    for (index, value) in buf.as_slice().iter().enumerate() {
        if index >= len {
            assert_eq!(value.bits(), poison::<T>().bits(), "B suffix {index}");
            continue;
        }
        let panel = index / (nr * depth.len);
        let p = index % (nr * depth.len) / nr;
        let j = panel * nr + index % nr;
        let want = if j < cols.len {
            read_op(view, trans, depth.start + p, cols.start + j)
        } else {
            T::ZERO
        };
        assert_eq!(
            value.bits(),
            want.bits(),
            "B({p}, {j}) {trans:?} {depth:?} {cols:?}"
        );
    }
}

/// Packs A blocks of every tail class from column- and row-major sources in
/// both transposes, at nonzero starts.
fn pack_a_layouts<T: TestFloat>() {
    for row_major in [false, true] {
        for trans in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
            // op(A) is 140 × 9 (rows tails 0, 1 and 63 fit at start 5).
            let (op_rows, op_cols) = (140, 9);
            let (rows, cols) = if trans == Trans::NoTrans {
                (op_rows, op_cols)
            } else {
                (op_cols, op_rows)
            };
            let source = Source::<T>::new(rows, cols, row_major, 11);
            let view = source.view();
            let Some(op) = ColMajor::normalize(view, trans) else {
                panic!("operand declined");
            };
            for block_rows in [64, 128, 65, 1, 63, 127] {
                for (depth_start, depth_len) in [(0, 1), (2, 5), (0, 9)] {
                    let rows_span = Span {
                        start: 5,
                        len: block_rows,
                    };
                    let depth = Span {
                        start: depth_start,
                        len: depth_len,
                    };
                    let Some(len) = a_pack_len(block_rows, depth_len) else {
                        panic!("length");
                    };
                    let mut buf = AlignedVec::filled(len + 37, poison::<T>());
                    let Some(packed) = pack_a(&op, rows_span, depth, &mut buf) else {
                        panic!("pack_a declined {rows_span:?} {depth:?}");
                    };
                    assert_eq!((packed.rows(), packed.depth()), (block_rows, depth_len));
                    assert_eq!(packed.data().len(), len);
                    check_packed_a(&view, trans, rows_span, depth, &buf, len);
                }
            }
        }
    }
}

/// Packs B blocks of every tail class from column- and row-major sources in
/// both transposes, at nonzero starts.
fn pack_b_layouts<T: TestFloat>() {
    let nr = T::NR;
    for row_major in [false, true] {
        for trans in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
            // op(B) is 9 × (3 · NR + 6).
            let (op_rows, op_cols) = (9, 3 * nr + 6);
            let (rows, cols) = if trans == Trans::NoTrans {
                (op_rows, op_cols)
            } else {
                (op_cols, op_rows)
            };
            let source = Source::<T>::new(rows, cols, row_major, 13);
            let view = source.view();
            let Some(op) = ColMajor::normalize(view, trans) else {
                panic!("operand declined");
            };
            for block_cols in [nr, 2 * nr, nr + 1, 1, nr - 1, 2 * nr - 1] {
                for (depth_start, depth_len) in [(0, 1), (3, 6), (0, 9)] {
                    let depth = Span {
                        start: depth_start,
                        len: depth_len,
                    };
                    let cols_span = Span {
                        start: 4,
                        len: block_cols,
                    };
                    let Some(len) = b_pack_len::<T>(depth_len, block_cols) else {
                        panic!("length");
                    };
                    let mut buf = AlignedVec::filled(len + 29, poison::<T>());
                    let Some(packed) = pack_b(&op, depth, cols_span, &mut buf) else {
                        panic!("pack_b declined {depth:?} {cols_span:?}");
                    };
                    assert_eq!((packed.depth(), packed.cols()), (depth_len, block_cols));
                    assert_eq!(packed.data().len(), len);
                    check_packed_b(&view, trans, depth, cols_span, &buf, len);
                }
            }
        }
    }
}

#[test]
fn test_sme2_pack_a_layout_f32() {
    pack_a_layouts::<f32>();
}

#[test]
fn test_sme2_pack_a_layout_f64() {
    pack_a_layouts::<f64>();
}

#[test]
fn test_sme2_pack_b_layout_f32() {
    pack_b_layouts::<f32>();
}

#[test]
fn test_sme2_pack_b_layout_f64() {
    pack_b_layouts::<f64>();
}

#[test]
fn test_sme2_pack_declines_bad_requests() {
    let source = Source::<f32>::new(70, 20, false, 3);
    let view = source.view();
    let Some(op) = ColMajor::normalize(view, Trans::NoTrans) else {
        panic!("operand declined");
    };
    let span = |start, len| Span { start, len };
    assert_eq!(a_pack_len(65, 3), Some(128 * 3));
    assert_eq!(b_pack_len::<f32>(3, 17), Some(32 * 3));
    assert_eq!(b_pack_len::<f64>(3, 17), Some(24 * 3));
    assert_eq!(a_pack_len(usize::MAX, 2), None);
    assert_eq!(b_pack_len::<f64>(usize::MAX, 2), None);
    let mut buf = AlignedVec::filled(1024, poison::<f32>());
    // Outside op(A), empty, overflowing, too small a buffer.
    assert!(pack_a(&op, span(60, 11), span(0, 4), &mut buf).is_none());
    assert!(pack_a(&op, span(0, 70), span(17, 4), &mut buf).is_none());
    assert!(pack_a(&op, span(0, 0), span(0, 4), &mut buf).is_none());
    assert!(pack_a(&op, span(0, 10), span(0, 0), &mut buf).is_none());
    assert!(pack_a(&op, span(usize::MAX, 2), span(0, 4), &mut buf).is_none());
    assert!(pack_a(&op, span(0, 65), span(0, 9), &mut buf).is_none());
    assert!(pack_b(&op, span(0, 71), span(0, 4), &mut buf).is_none());
    assert!(pack_b(&op, span(0, 4), span(19, 2), &mut buf).is_none());
    assert!(pack_b(&op, span(0, 0), span(0, 2), &mut buf).is_none());
    assert!(pack_b(&op, span(0, 70), span(0, 20), &mut buf).is_none());
    assert!(buf.iter().all(|v| v.to_bits() == poison::<f32>().to_bits()));
    // The largest accepted requests.
    assert!(pack_a(&op, span(0, 64), span(0, 16), &mut buf).is_some());
    assert!(pack_b(&op, span(0, 32), span(0, 20), &mut buf).is_some());
}

#[test]
fn test_sme2_pack_spans_block_beta_and_block_args() {
    let all: Vec<Span> = spans(10, 4).collect();
    assert_eq!(
        all,
        [
            Span { start: 0, len: 4 },
            Span { start: 4, len: 4 },
            Span { start: 8, len: 2 }
        ]
    );
    assert_eq!(spans(0, 4).count(), 0);
    assert_eq!(spans(3, 0).count(), 3);
    assert_eq!(
        spans(5, 64).collect::<Vec<_>>(),
        [Span { start: 0, len: 5 }]
    );
    assert_eq!(block_beta(Span { start: 0, len: 3 }, 0.25_f64), 0.25);
    assert_eq!(block_beta(Span { start: 3, len: 3 }, 0.25_f32), 1.0);

    let source = Source::<f64>::new(70, 20, false, 5);
    let Some(op) = ColMajor::normalize(source.view(), Trans::NoTrans) else {
        panic!("operand declined");
    };
    let (mut buf_a, mut buf_b) = (
        AlignedVec::<f64>::zeros(4096),
        AlignedVec::<f64>::zeros(4096),
    );
    let Some(a) = pack_a(
        &op,
        Span { start: 0, len: 10 },
        Span { start: 0, len: 4 },
        &mut buf_a,
    ) else {
        panic!("pack_a");
    };
    let Some(b) = pack_b(
        &op,
        Span { start: 0, len: 4 },
        Span { start: 0, len: 9 },
        &mut buf_b,
    ) else {
        panic!("pack_b");
    };
    let mut c_data = vec![0.0_f64; 40 * 12];
    let Some(mut c) = MatMut::from_strided(&mut c_data, 30, 12, 40) else {
        panic!("C view");
    };
    let Some(args) = block_args(&a, &b, &mut c, 7, 2, 0.5, 0.0) else {
        panic!("block_args");
    };
    assert_eq!((args.m, args.n, args.kc, args.ldc), (10, 9, 4, 40));
    assert_eq!(
        (args.a_depth, args.a_panel, args.b_panel),
        (MR, MR * 4, 4 * 8)
    );
    assert_eq!(args.mode, Epilogue::Scale as u64);
    assert_eq!(args.a, a.data().as_ptr());
    assert_eq!(args.b, b.data().as_ptr());
    assert_eq!(args.c, c.ptr_at_mut(7, 2));
    // A tight column-major C (`row_stride == rows`) whose rows the block
    // covers, with more than one column, is accepted with `ldc = rows`.
    let mut tight_data = vec![0.0_f64; 10 * 9];
    let Some(mut tight) = MatMut::from_column_major(&mut tight_data, 10, 9) else {
        panic!("tight C view");
    };
    let origin = tight.ptr_at_mut(0, 0);
    let Some(args) = block_args(&a, &b, &mut tight, 0, 0, 1.0, 0.0) else {
        panic!("tight C declined");
    };
    assert_eq!((args.m, args.n, args.ldc), (10, 9, 10));
    assert_eq!(args.c, origin);
    // The block leaves C.
    assert!(block_args(&a, &b, &mut c, 21, 0, 1.0, 0.0).is_none());
    assert!(block_args(&a, &b, &mut c, 0, 4, 1.0, 0.0).is_none());
    // Depths differ.
    let Some(b5) = pack_b(
        &op,
        Span { start: 0, len: 5 },
        Span { start: 0, len: 9 },
        &mut buf_b,
    ) else {
        panic!("pack_b");
    };
    assert!(block_args(&a, &b5, &mut c, 0, 0, 1.0, 0.0).is_none());
    // Overlapping columns (`row_stride < rows`), and a single column whose
    // stride does not matter.
    let Some(b2) = pack_b(
        &op,
        Span { start: 0, len: 4 },
        Span { start: 0, len: 2 },
        &mut buf_b,
    ) else {
        panic!("pack_b");
    };
    // SAFETY: the view addresses offsets `i + j · 5` for `i < 30`, `j < 2`,
    // inside `c_data`, which nothing else borrows; it is never written.
    let mut overlapping = unsafe { MatMut::new(c_data.as_mut_ptr(), 30, 2, 5) };
    assert!(block_args(&a, &b2, &mut overlapping, 0, 0, 1.0, 0.0).is_none());
    let Some(b1) = pack_b(
        &op,
        Span { start: 0, len: 4 },
        Span { start: 0, len: 1 },
        &mut buf_b,
    ) else {
        panic!("pack_b");
    };
    // SAFETY: as above, one column.
    let mut column = unsafe { MatMut::new(c_data.as_mut_ptr(), 30, 1, 0) };
    let Some(args) = block_args(&a, &b1, &mut column, 0, 0, 1.0, 1.0) else {
        panic!("single column");
    };
    assert_eq!((args.ldc, args.mode), (10, Epilogue::Add as u64));
}

/// The scalar reference of a whole blocked product: per element, per depth
/// block, `acc = Σ fma` from `+0` in ascending `p`, then the epilogue of the
/// block with `beta` on the first block and `1` afterwards.
fn blocked_reference<T: TestFloat>(
    a: &MatRef<'_, T>,
    trans_a: Trans,
    b: &MatRef<'_, T>,
    trans_b: Trans,
    c: &mut [T],
    ldc: usize,
    (m, n, k, kc): (usize, usize, usize, usize),
    alpha: T,
    beta: T,
) {
    for j in 0..n {
        for i in 0..m {
            let mut value = c[i + j * ldc];
            for depth in spans(k, kc) {
                let mut acc = T::ZERO;
                for p in depth.start..depth.start + depth.len {
                    acc = read_op(a, trans_a, i, p).fused(read_op(b, trans_b, p, j), acc);
                }
                let beta_eff = if depth.start == 0 { beta } else { T::ONE };
                let t = if alpha == T::ONE { acc } else { acc * alpha };
                value = if beta_eff == T::ZERO {
                    t
                } else if beta_eff == T::ONE {
                    t + value
                } else {
                    value.fused(beta_eff, t)
                };
            }
            c[i + j * ldc] = value;
        }
    }
}

/// A whole product computed block by block (pc, jc, ic) with `pack.rs`,
/// `MacroArgs::packed` and the oracle as the kernel, against
/// [`blocked_reference`], bit for bit, with C inside a border.
fn blocked_product<T: TestFloat>() {
    // (m, n, k, mc, nc, kc, ldc - m): the last two cover every row of a
    // tight C (`ldc == m`) with one row block of more than one column.
    let configs = [
        (130, 35, 70, 64, 19, 32, 4),
        (67, 9, 9, 50, 16, 4, 4),
        (5, 3, 1, 64, 8, 1, 0),
        (64, 2 * T::NR, 3, 128, T::NR, 3, 0),
    ];
    let pairs = [(1.0, 0.0), (0.3, -0.0), (1.0, 1.0), (-1.5, 0.25)];
    let trans_all = [Trans::NoTrans, Trans::Trans];
    for (m, n, k, mc, nc, kc, ldc_pad) in configs {
        for trans_a in trans_all {
            for (trans_b, b_row_major) in [(Trans::NoTrans, false), (Trans::Trans, true)] {
                let (ar, ac) = if trans_a == Trans::NoTrans {
                    (m, k)
                } else {
                    (k, m)
                };
                let (br, bc) = if trans_b == Trans::NoTrans {
                    (k, n)
                } else {
                    (n, k)
                };
                let a_src = Source::<T>::new(ar, ac, false, (m + k) as u64);
                let b_src = Source::<T>::new(br, bc, b_row_major, (n + k) as u64);
                let (a_view, b_view) = (a_src.view(), b_src.view());
                let (Some(a), Some(b)) = (
                    ColMajor::normalize(a_view, trans_a),
                    ColMajor::normalize(b_view, trans_b),
                ) else {
                    panic!("operands declined");
                };
                for (alpha, beta) in pairs {
                    let (alpha, beta) = (T::of(alpha), T::of(beta));
                    let ldc = m + ldc_pad;
                    let start = values::<T>(ldc * (n + 2), 99);
                    let mut c0 = vec![border::<T>(); ldc * (n + 2)];
                    let origin = ldc + 2;
                    for j in 0..n {
                        for i in 0..m {
                            c0[origin + i + j * ldc] = if beta == T::ZERO {
                                T::from_bits64(u64::MAX >> 1)
                            } else {
                                start[i + j * ldc]
                            };
                        }
                    }
                    let mut want = c0.clone();
                    blocked_reference(
                        &a_view,
                        trans_a,
                        &b_view,
                        trans_b,
                        &mut want[origin..],
                        ldc,
                        (m, n, k, kc),
                        alpha,
                        beta,
                    );

                    let mut got = c0.clone();
                    let Some(mut c) = MatMut::from_strided(&mut got[origin..], m, n, ldc) else {
                        panic!("C view");
                    };
                    let (Some(a_len), Some(b_len)) = (
                        a_pack_len(mc.min(m), kc.min(k)),
                        b_pack_len::<T>(kc.min(k), nc.min(n)),
                    ) else {
                        panic!("lengths");
                    };
                    let mut buf_a = AlignedVec::filled(a_len, poison::<T>());
                    let mut buf_b = AlignedVec::filled(b_len, poison::<T>());
                    for depth in spans(k, kc) {
                        let beta_eff = block_beta(depth, beta);
                        for cols in spans(n, nc) {
                            let Some(pb) = pack_b(&b, depth, cols, &mut buf_b) else {
                                panic!("pack_b");
                            };
                            for rows in spans(m, mc) {
                                let Some(pa) = pack_a(&a, rows, depth, &mut buf_a) else {
                                    panic!("pack_a");
                                };
                                let Some(args) = block_args(
                                    &pa, &pb, &mut c, rows.start, cols.start, alpha, beta_eff,
                                ) else {
                                    panic!("block_args");
                                };
                                // SAFETY: `block_args` validated the block:
                                // whole packed panels and a C block inside
                                // `c`, used while all borrows are alive.
                                unsafe { oracle_block(&args) };
                            }
                        }
                    }
                    for (index, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.bits(),
                            w.bits(),
                            "{m}x{n}x{k} blocks {mc}/{nc}/{kc} {trans_a:?}/{trans_b:?} \
                             alpha {alpha:?} beta {beta:?}: index {index}"
                        );
                    }
                    // The border kept its bits.
                    for (index, g) in got.iter().enumerate() {
                        let inside = index
                            .checked_sub(origin)
                            .is_some_and(|o| o % ldc < m && o / ldc < n);
                        if !inside {
                            assert_eq!(g.bits(), border::<T>().bits(), "border {index}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_pack_blocked_product_matches_reference_f32() {
    blocked_product::<f32>();
}

#[test]
fn test_sme2_pack_blocked_product_matches_reference_f64() {
    blocked_product::<f64>();
}
