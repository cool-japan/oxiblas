//! Tests of the few-rows kernels (`few_rows.rs`): the helpers every result
//! test uses, the plain-Rust tests (argument block, declines, the scalar
//! model and the self-check against wrong stand-ins, the test oracle against
//! the NEON blocked path) and the structure tests of the registered
//! templates. The result tests that execute the kernels are in
//! `tests_few_rows_grid.rs`.
//!
//! # How a case runs ([`check_case`])
//!
//! A [`Case`] is one call: `m`, `n`, `k`, the KC step, the orientation of B,
//! alpha and beta, the paddings of A, B and C, the first column of op(B)
//! the call uses, whether C starts as NaN, and the data ([`Fill`]).
//! [`Operands`] places every operand inside a larger buffer filled with a
//! SIGNALLING NaN (`TestElem::SENTINEL`): a guard column before and after
//! A and C, the padding rows of every column, the columns of the stored B
//! outside the span and its padding rows, a guard element before each
//! operand, and `WIDE_GUARD` more sentinels at both ends of every buffer.
//! Then C is computed three ways:
//!
//! - the kernel: [`few_rows_gemm`] (one region: the per-thread region
//!   counter of `few_rows.rs` moves by exactly one);
//! - the executable definition of the seams file: per depth block
//!   `args::reference` with A in place (`with_a_in_place`) and that block of
//!   op(B) packed by `pack::pack_b` ([`reference_blocks`]);
//! - the macro-kernel path of the blocked driver: per depth block
//!   `pack::pack_a`, `pack::pack_b`, `pack::block_args` and the macro-kernel
//!   ([`macro_blocks`], the sequence of `driver::run_part`).
//!
//! The block of C must equal both (`args::same_result`: bitwise, NaN
//! payloads aside), and every element outside the block must keep its
//! signalling-NaN bits: an out-of-bounds read-modify-write (M1 / M3) would
//! quiet it, an out-of-bounds store would overwrite it. Since the A guard
//! column after depth `k - 1` holds the sentinel too, a depth loop that runs
//! one step too many turns live elements into NaNs.
//!
//! # What no result test can see
//!
//! A forbidden READ feeds only lanes that are never stored (A rows `≥ m`,
//! B depth steps past the chunk, B columns past `n`, C in M0 / M2). The
//! structure tests parse the registered templates (`few_rows::SME_KERNELS`)
//! and prove which predicate governs every load, that the predicates are
//! built from `m`, the remaining depth and the live columns, and that the
//! depth-loop variants touch only the live C tiles ([`check_structure`]).

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::{Epilogue, MacroArgs, SmeElement, reference, same_result};
use super::driver::SmeGemm;
use super::few_rows::{
    FewRowsArgs, FewRowsKernel, OFF_A, OFF_ALPHA, OFF_B, OFF_B_TRANS, OFF_BETA, OFF_C, OFF_K,
    OFF_KC, OFF_LDA, OFF_LDB, OFF_LDC, OFF_M, OFF_MODE_FIRST, OFF_MODE_LATER, OFF_N, OFF_ONE,
    SME_KERNELS, SmeFewRows, few_rows_args, few_rows_gemm, max_rows, oracle_few_rows,
    regions_on_this_thread, self_check_with,
};
use super::lint::{lint_kernel, report};
use super::pack::{ColMajor, Span, block_args, block_beta, pack_a, pack_b, spans};
use super::test_support::{force_off, kernel_entries_here, kernel_ready, sme_state};
use super::tests_gemm_harness::{Rng, TestElem};
use super::tests_kernel_f32::predicate_uses;
use crate::level3::gemm::{BlockingSource, GemmBlocking, call_blocking, gemm_planned};
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan, GemmRoute};
use crate::level3::trsm::Trans;

// ---------------------------------------------------------------------------
// Helpers (shared with tests_few_rows_grid.rs)
// ---------------------------------------------------------------------------

/// An element type of the few-rows kernels with what the tests need.
pub(super) trait FrElem: TestElem + SmeFewRows {}

impl<T: TestElem + SmeFewRows> FrElem for T {}

/// The data of a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fill {
    /// Inexact values in `[-1, 1)`.
    Random,
    /// Random, plus `+Inf` and a quiet NaN in A, `-Inf` in B and, when C is
    /// read, `+Inf` and a NaN in C.
    NonFinite,
    /// A and B scaled into the subnormal range (C by the square), so that
    /// products, sums and results are subnormal.
    Subnormal,
    /// Small integers and zeros of both signs (exact products: the signs of
    /// zero results are those of the arithmetic, not of rounding).
    SignedZeros,
}

/// One call of the few-rows kernel (see the module docs).
#[derive(Debug, Clone, Copy)]
pub(super) struct Case<T> {
    /// Rows.
    pub(super) m: usize,
    /// Columns of the span.
    pub(super) n: usize,
    /// Depth.
    pub(super) k: usize,
    /// KC step.
    pub(super) kc: usize,
    /// op(B) = B^T (the stored B is `n_total × k`).
    pub(super) trans: bool,
    /// alpha.
    pub(super) alpha: T,
    /// beta.
    pub(super) beta: T,
    /// `lda - m`.
    pub(super) a_pad: usize,
    /// Leading dimension of the stored B minus its rows.
    pub(super) b_pad: usize,
    /// `ldc - m`.
    pub(super) c_pad: usize,
    /// First column of op(B) the call uses (the columns before it and one
    /// after the span exist in the stored B and hold the sentinel).
    pub(super) col0: usize,
    /// C starts as `CHECK_UNREAD` (a signalling NaN) when the first block
    /// does not read it.
    pub(super) c_nan: bool,
    /// The data.
    pub(super) fill: Fill,
    /// Seed of the data.
    pub(super) seed: u64,
}

impl<T: FrElem> Case<T> {
    /// A case with alpha 0.5, beta -0.5, small paddings, the span starting
    /// at column 1, random data.
    pub(super) fn new(m: usize, n: usize, k: usize, kc: usize, trans: bool) -> Self {
        Self {
            m,
            n,
            k,
            kc,
            trans,
            alpha: T::of(0.5),
            beta: T::of(-0.5),
            a_pad: 1,
            b_pad: 2,
            c_pad: 3,
            col0: 1,
            c_nan: false,
            fill: Fill::Random,
            seed: (m * 1_000_003 + n * 10_007 + k * 101 + kc) as u64 * 2 + u64::from(trans),
        }
    }

    /// With these scalars; C starts as NaN when beta is zero.
    pub(super) fn scalars(self, alpha: f64, beta: f64) -> Self {
        Self {
            alpha: T::of(alpha),
            beta: T::of(beta),
            c_nan: beta == 0.0,
            ..self
        }
    }

    /// With these paddings of A, B and C.
    pub(super) fn pads(self, a_pad: usize, b_pad: usize, c_pad: usize) -> Self {
        Self {
            a_pad,
            b_pad,
            c_pad,
            ..self
        }
    }

    /// With the span starting at stored column `col0`.
    pub(super) fn at_column(self, col0: usize) -> Self {
        Self { col0, ..self }
    }

    /// With this data.
    pub(super) fn filled(self, fill: Fill) -> Self {
        Self { fill, ..self }
    }

    /// Whether the first depth block reads C.
    pub(super) fn reads_c(&self) -> bool {
        Epilogue::select(self.alpha, self.beta).reads_c()
    }
}

/// Sentinel elements before and after every buffer of [`Operands`] beyond
/// the guard columns: one whole 4-vector group of `f32` (64 elements), so
/// that a store that runs a vector or a group too far lands on a sentinel
/// and fails the comparison instead of corrupting the allocator.
pub(super) const WIDE_GUARD: usize = 64;

/// The operands of a case inside their sentinel-filled buffers.
#[derive(Debug, Clone)]
pub(super) struct Operands<T> {
    /// The A buffer; `A(i, p) = a[a_off + i + p * lda]`.
    pub(super) a: Vec<T>,
    /// Offset of `A(0, 0)`.
    pub(super) a_off: usize,
    /// Leading dimension of A.
    pub(super) lda: usize,
    /// The B buffer; stored `(r, c)` at `b[b_off + r + c * ldb]`.
    pub(super) b: Vec<T>,
    /// Offset of the stored `(0, 0)`.
    pub(super) b_off: usize,
    /// Leading dimension of the stored B.
    pub(super) ldb: usize,
    /// Rows and columns of the stored B.
    pub(super) b_shape: (usize, usize),
    /// C before the call; `C(i, j) = c[c_off + i + j * ldc]`.
    pub(super) c: Vec<T>,
    /// Offset of `C(0, 0)`.
    pub(super) c_off: usize,
    /// Leading dimension of C.
    pub(super) ldc: usize,
}

impl<T: FrElem> Operands<T> {
    /// The buffers of `case` (see the module docs).
    pub(super) fn new(case: &Case<T>) -> Self {
        let (m, n, k) = (case.m, case.n, case.k);
        let mut rng = Rng::new(case.seed);
        let sentinel = T::SENTINEL;
        let shift = T::SUBNORMAL_SHIFT;
        let fill = case.fill;
        let value = move |rng: &mut Rng, exp: i32| -> T {
            match fill {
                Fill::SignedZeros => {
                    let bits = rng.next_u64();
                    match bits % 5 {
                        0 => T::of(0.0),
                        1 => T::of(-0.0),
                        _ => T::of(((bits >> 8) % 7) as f64 - 3.0),
                    }
                }
                Fill::Subnormal => T::of(rng.value::<T>().to_f64() * 2f64.powi(-exp)),
                _ => rng.value::<T>(),
            }
        };

        let lda = m + case.a_pad;
        let a_off = WIDE_GUARD + lda + 1;
        let mut a = vec![sentinel; a_off + lda * (k + 1) + 1 + WIDE_GUARD];
        for p in 0..k {
            for i in 0..m {
                a[a_off + i + p * lda] = value(&mut rng, shift);
            }
        }

        let n_total = case.col0 + n + 1;
        let b_shape = if case.trans {
            (n_total, k)
        } else {
            (k, n_total)
        };
        let ldb = b_shape.0 + case.b_pad;
        let b_off = WIDE_GUARD + ldb + 1;
        let mut b = vec![sentinel; b_off + ldb * (b_shape.1 + 1) + 1 + WIDE_GUARD];
        for p in 0..k {
            for j in 0..n {
                let at = if case.trans {
                    (case.col0 + j) + p * ldb
                } else {
                    p + (case.col0 + j) * ldb
                };
                b[b_off + at] = value(&mut rng, shift);
            }
        }

        let ldc = m + case.c_pad;
        let c_off = WIDE_GUARD + ldc + 1;
        let mut c = vec![sentinel; c_off + ldc * (n + 1) + 1 + WIDE_GUARD];
        for j in 0..n {
            for i in 0..m {
                c[c_off + i + j * ldc] = if case.c_nan && !case.reads_c() {
                    T::CHECK_UNREAD
                } else {
                    value(&mut rng, 2 * shift)
                };
            }
        }

        let mut ops = Self {
            a,
            a_off,
            lda,
            b,
            b_off,
            ldb,
            b_shape,
            c,
            c_off,
            ldc,
        };
        if case.fill == Fill::NonFinite {
            ops.plant_non_finite(case);
        }
        ops
    }

    /// `+Inf` and a quiet NaN in A, `-Inf` in B, and `+Inf` and a NaN in C
    /// when the first block reads it.
    fn plant_non_finite(&mut self, case: &Case<T>) {
        let (m, n, k) = (case.m, case.n, case.k);
        self.a[self.a_off + (m - 1) + (k / 2) * self.lda] = T::of(f64::INFINITY);
        if m > 1 || k > 1 {
            self.a[self.a_off + (k - 1) * self.lda] = T::of(f64::NAN);
        }
        let at = if case.trans {
            (case.col0 + n - 1) + (k - 1) * self.ldb
        } else {
            (k - 1) + (case.col0 + n - 1) * self.ldb
        };
        self.b[self.b_off + at] = T::of(f64::NEG_INFINITY);
        if case.reads_c() {
            self.c[self.c_off] = T::of(f64::INFINITY);
            self.c[self.c_off + (m - 1) + (n - 1) * self.ldc] = T::of(f64::NAN);
        }
    }

    /// op(A): the `m × k` column-major view.
    pub(super) fn a_view(&self, case: &Case<T>) -> MatRef<'_, T> {
        MatRef::from_strided(&self.a[self.a_off..], case.m, case.k, self.lda, 1)
            .unwrap_or_else(|| panic!("A view of {case:?}"))
    }

    /// The stored B (every column of it: the span and the sentinels).
    pub(super) fn b_stored(&self) -> MatRef<'_, T> {
        let (rows, cols) = self.b_shape;
        MatRef::from_strided(&self.b[self.b_off..], rows, cols, self.ldb, 1)
            .unwrap_or_else(|| panic!("B view {rows} x {cols}"))
    }

    /// op(B) as the driver normalises it.
    pub(super) fn b_op(&self, case: &Case<T>) -> ColMajor<'_, T> {
        let trans = if case.trans {
            Trans::Trans
        } else {
            Trans::NoTrans
        };
        ColMajor::normalize(self.b_stored(), trans).unwrap_or_else(|| panic!("op(B) of {case:?}"))
    }

    /// The `m × n` view of C inside `c` (a copy of [`Self::c`]).
    pub(super) fn c_view<'c>(&self, case: &Case<T>, c: &'c mut [T]) -> MatMut<'c, T> {
        MatMut::from_strided(&mut c[self.c_off..], case.m, case.n, self.ldc)
            .unwrap_or_else(|| panic!("C view of {case:?}"))
    }

    /// Whether element `at` of the C buffer is inside the block.
    pub(super) fn in_block(&self, case: &Case<T>, at: usize) -> bool {
        at >= self.c_off && {
            let offset = at - self.c_off;
            offset % self.ldc < case.m && offset / self.ldc < case.n
        }
    }
}

/// The span of op(B)'s columns of `case`.
pub(super) fn cols_of<T>(case: &Case<T>) -> Span {
    Span {
        start: case.col0,
        len: case.n,
    }
}

/// The per-block executable definition: `args::reference` once per depth
/// block, A in place, that block of op(B) packed by `pack::pack_b`. Writes
/// into `c` (a copy of the C buffer).
pub(super) fn reference_blocks<T: FrElem>(case: &Case<T>, ops: &Operands<T>, c: &mut [T]) {
    let b = ops.b_op(case);
    let cols = cols_of(case);
    let mut buf = AlignedVec::<T>::zeros(case.n.div_ceil(T::NR) * T::NR * case.kc.min(case.k));
    for depth in spans(case.k, case.kc) {
        let packed = pack_b(&b, depth, cols, &mut buf)
            .unwrap_or_else(|| panic!("pack_b of {depth:?} for {case:?}"));
        let a = ops.a[ops.a_off + depth.start * ops.lda..].as_ptr();
        let args = MacroArgs::packed(
            a,
            packed.data().as_ptr(),
            c[ops.c_off..].as_mut_ptr(),
            ops.ldc,
            case.m,
            case.n,
            depth.len,
            case.alpha,
            block_beta(depth, case.beta),
        )
        .with_a_in_place(a, ops.lda);
        // SAFETY: A in place: `A(i, p0 + p)` for `i < m ≤ 64` is inside the A
        // buffer; the packed block holds whole zero-padded panels of depth
        // `depth.len`; C is `ldc · n` elements from `c_off`, `ldc ≥ m`, and
        // overlaps neither.
        unsafe { reference(&args) };
    }
}

/// The macro-kernel path of the blocked driver (`driver::run_part` for one
/// part with MC ≥ m and NC ≥ n): per depth block `pack_a`, `pack_b`,
/// `block_args` and the macro-kernel. Executes SME2 code.
pub(super) fn macro_blocks<T: FrElem>(case: &Case<T>, ops: &Operands<T>, c: &mut [T]) {
    let a = ColMajor::normalize(ops.a_view(case), Trans::NoTrans)
        .unwrap_or_else(|| panic!("op(A) of {case:?}"));
    let b = ops.b_op(case);
    let cols = cols_of(case);
    let rows = Span {
        start: 0,
        len: case.m,
    };
    let kc = case.kc.min(case.k);
    let mut buf_a = AlignedVec::<T>::zeros(64 * kc);
    let mut buf_b = AlignedVec::<T>::zeros(case.n.div_ceil(T::NR) * T::NR * kc);
    let mut view = ops.c_view(case, c);
    for depth in spans(case.k, case.kc) {
        let packed_b = pack_b(&b, depth, cols, &mut buf_b)
            .unwrap_or_else(|| panic!("pack_b of {depth:?} for {case:?}"));
        let packed_a = pack_a(&a, rows, depth, &mut buf_a)
            .unwrap_or_else(|| panic!("pack_a of {depth:?} for {case:?}"));
        let args = block_args(
            &packed_a,
            &packed_b,
            &mut view,
            0,
            0,
            case.alpha,
            block_beta(depth, case.beta),
        )
        .unwrap_or_else(|| panic!("block_args of {depth:?} for {case:?}"));
        assert!(kernel_ready(), "thread not ready (macro path)");
        // SAFETY: the thread is ready (checked above; every caller is a
        // hardware test of `T`); `block_args` built the block from two
        // packed blocks of the same depth and the C view, so every address
        // of the layout is valid and C overlaps neither packed block.
        unsafe { <T as SmeGemm>::kernel(&args) };
    }
}

/// The blocked result of `case` (`gemm_planned` forced onto the blocked route with SME2 forced
/// off and an explicit blocking whose KC target is `case.kc`: the dispatched micro-kernel, NEON
/// or, under `force-scalar`, the portable one), and the KC step that call used (`call_blocking`'s
/// fitted value, which `balanced_block` may round).
pub(super) fn neon_blocked<T: FrElem>(case: &Case<T>, ops: &Operands<T>) -> (Vec<T>, usize) {
    let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
    let source = BlockingSource::Explicit(GemmBlocking {
        mc: 256,
        kc: case.kc,
        nc: 4096,
    });
    let blocks = call_blocking(source, &plan, case.m, case.k, case.n);
    let kc = blocks.sequential().kc.min(case.k);
    let stored = ops.b_stored();
    let (b_view, trans_b) = if case.trans {
        (stored.submatrix(case.col0, 0, case.n, case.k), Trans::Trans)
    } else {
        (
            stored.submatrix(0, case.col0, case.k, case.n),
            Trans::NoTrans,
        )
    };
    let mut c = ops.c.clone();
    {
        let _neon = force_off();
        gemm_planned(
            Trans::NoTrans,
            trans_b,
            case.alpha,
            ops.a_view(case),
            b_view,
            case.beta,
            ops.c_view(case, &mut c),
            Par::Seq,
            source,
            &plan,
        );
    }
    (c, kc)
}

/// Runs the kernel on `c` (a copy of the C buffer): exactly one streaming
/// region, the thread out of streaming mode with no ZA state afterwards.
pub(super) fn run_kernel<T: FrElem>(case: &Case<T>, ops: &Operands<T>, c: &mut [T]) {
    assert!(kernel_ready(), "thread not ready before the call");
    let b = ops.b_op(case);
    let regions = regions_on_this_thread();
    let entries = kernel_entries_here();
    // SAFETY: the thread is ready (checked above) and every caller is a
    // hardware test of `T` (`hardware_or_skip`); C is a copy, overlapping
    // neither A nor B.
    let done = unsafe {
        few_rows_gemm(
            ops.a_view(case),
            &b,
            cols_of(case),
            case.alpha,
            case.beta,
            case.kc,
            ops.c_view(case, c),
        )
    };
    assert!(done, "few_rows_gemm declined {case:?}");
    assert_eq!(
        regions_on_this_thread(),
        regions + 1,
        "not exactly one streaming region for {case:?}"
    );
    // Counted by the kernel itself, on this thread: exactly its one entry.
    assert_eq!(kernel_entries_here() - entries, 1, "entries: {case:?}");
    assert_eq!(sme_state(), Some((0, 0)), "SVCR / TPIDR2 after {case:?}");
    assert!(kernel_ready(), "thread not ready after the call");
}

/// Compares `got` with `want` (block: `same_result`; outside: the bits of
/// C before the call), panicking with `label` on the first difference.
pub(super) fn assert_c<T: FrElem>(
    case: &Case<T>,
    ops: &Operands<T>,
    got: &[T],
    want: &[T],
    label: &str,
) {
    assert_eq!(got.len(), want.len());
    for (at, (g, w)) in got.iter().zip(want).enumerate() {
        if ops.in_block(case, at) {
            let offset = at - ops.c_off;
            assert!(
                same_result(*g, *w),
                "{label}: C({}, {}) = {g:?} ({:#x}), want {w:?} ({:#x}); {case:?}",
                offset % ops.ldc,
                offset / ops.ldc,
                g.bits(),
                w.bits()
            );
        } else {
            assert_eq!(
                g.bits(),
                ops.c[at].bits(),
                "{label}: element {at} outside the block changed; {case:?}"
            );
        }
    }
}

/// Runs `case` on the kernel and checks it against the per-block reference
/// and the macro-kernel path (see the module docs). Hardware only.
pub(super) fn check_case<T: FrElem>(case: &Case<T>, label: &str) {
    check_ops(case, &Operands::new(case), label);
}

/// [`check_case`] on operands the caller may have edited.
pub(super) fn check_ops<T: FrElem>(case: &Case<T>, ops: &Operands<T>, label: &str) {
    let mut want = ops.c.clone();
    reference_blocks(case, ops, &mut want);
    let mut via_macro = ops.c.clone();
    macro_blocks(case, ops, &mut via_macro);
    assert_c(
        case,
        ops,
        &via_macro,
        &want,
        &format!("{label} (macro path vs reference)"),
    );
    let mut got = ops.c.clone();
    run_kernel(case, ops, &mut got);
    assert_c(case, ops, &got, &want, label);
    let finite = case.alpha.to_f64().is_finite() && case.beta.to_f64().is_finite();
    if case.fill == Fill::Random && case.c_nan && finite {
        // No NaN of the unread C may survive (finite data and scalars).
        for j in 0..case.n {
            for i in 0..case.m {
                let value = got[ops.c_off + i + j * ops.ldc];
                assert!(!value.nan(), "{label}: NaN at ({i}, {j}); {case:?}");
            }
        }
    }
}

/// `(alpha, beta)` of the seams grid: alpha in {1, -1, 0.5} × beta in
/// {0 (C NaN), -0, 1, -0.5}.
pub(super) const PAIRS: [(f64, f64); 12] = [
    (1.0, 0.0),
    (1.0, -0.0),
    (1.0, 1.0),
    (1.0, -0.5),
    (-1.0, 0.0),
    (-1.0, -0.0),
    (-1.0, 1.0),
    (-1.0, -0.5),
    (0.5, 0.0),
    (0.5, -0.0),
    (0.5, 1.0),
    (0.5, -0.5),
];

/// Depths of the seams grid.
pub(super) const DEPTHS: [usize; 10] = [1, 7, 8, 9, 15, 16, 17, 63, 64, 65];

/// KC steps of the seams grid (0 = `k`: one depth block).
pub(super) const KCS: [usize; 4] = [0, 32, 33, 64];

/// Column counts of the seams grid: `1..NR+1`, `2NR-1..2NR+1`, `4NR+3`.
pub(super) fn grid_columns<T: FrElem>() -> Vec<usize> {
    let nr = T::NR;
    (1..=nr + 1)
        .chain(2 * nr - 1..=2 * nr + 1)
        .chain([4 * nr + 3])
        .collect()
}

// ---------------------------------------------------------------------------
// The argument block and the entry point's checks (plain Rust)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_few_rows_max_rows() {
    assert_eq!(max_rows::<f32>(), 48);
    assert_eq!(max_rows::<f64>(), 32);
    // Three `.s` tiles of 16 rows (za3 transposes); four `.d` tiles of 8
    // rows (za4 transposes).
    assert_eq!(max_rows::<f32>(), 3 * <f32 as SmeElement>::TILE_ROWS);
    assert_eq!(max_rows::<f64>(), 4 * <f64 as SmeElement>::TILE_ROWS);
}

#[test]
fn test_sme2_few_rows_args_layout() {
    let offsets = [
        OFF_A,
        OFF_LDA,
        OFF_B,
        OFF_LDB,
        OFF_B_TRANS,
        OFF_C,
        OFF_LDC,
        OFF_M,
        OFF_N,
        OFF_K,
        OFF_KC,
        OFF_ALPHA,
        OFF_BETA,
        OFF_ONE,
        OFF_MODE_FIRST,
        OFF_MODE_LATER,
    ];
    for (index, offset) in offsets.iter().enumerate() {
        assert_eq!(*offset, 8 * index, "field {index}");
    }
    assert_eq!(core::mem::size_of::<FewRowsArgs<f32>>(), 128);
    assert_eq!(core::mem::size_of::<FewRowsArgs<f64>>(), 128);
    // `ldr x…, [x0, #off]` takes multiples of 8 up to 32760, `ldr w…` (the
    // f32 bit patterns) multiples of 4 up to 16380.
    assert!(
        offsets
            .iter()
            .all(|offset| offset % 8 == 0 && *offset <= 16_380)
    );
}

#[test]
fn test_sme2_few_rows_args_modes() {
    let values = [1.0f64, -1.0, 0.5, 0.0, -0.0, 2.0, f64::NAN, f64::INFINITY];
    for alpha in values {
        for beta in values {
            let (a, b) = (alpha as f32, beta as f32);
            let args = FewRowsArgs::<f32>::new(
                (core::ptr::null(), 0),
                (core::ptr::null(), 0, true),
                (core::ptr::null_mut(), 1),
                (1, 1, 1, 1),
                a,
                b,
            );
            assert_eq!(args.mode_first, Epilogue::select(a, b) as u64);
            assert_eq!(args.mode_later, Epilogue::select(a, 1.0f32) as u64);
            assert_eq!(args.alpha_bits, u64::from(a.to_bits()));
            assert_eq!(args.beta_bits, u64::from(b.to_bits()));
            assert_eq!(args.one_bits, u64::from(1.0f32.to_bits()));
            assert_eq!(args.b_trans, 1);
            let args = FewRowsArgs::<f64>::new(
                (core::ptr::null(), 0),
                (core::ptr::null(), 0, false),
                (core::ptr::null_mut(), 1),
                (1, 1, 1, 1),
                alpha,
                beta,
            );
            assert_eq!(args.mode_first, Epilogue::select(alpha, beta) as u64);
            assert_eq!(args.mode_later, Epilogue::select(alpha, 1.0) as u64);
            assert_eq!(args.one_bits, 1.0f64.to_bits());
            assert_eq!(args.b_trans, 0);
            // A later block never skips C: its beta is one.
            assert!(Epilogue::select(alpha, 1.0).reads_c());
        }
    }
}

/// `few_rows_args` for `case`'s views with these columns and KC step.
fn args_of<T: FrElem>(
    case: &Case<T>,
    ops: &Operands<T>,
    c: &mut [T],
    cols: Span,
    kc: usize,
) -> Option<FewRowsArgs<T>> {
    let b = ops.b_op(case);
    let mut view = ops.c_view(case, c);
    few_rows_args(
        &ops.a_view(case),
        &b,
        cols,
        case.alpha,
        case.beta,
        kc,
        &mut view,
    )
}

#[test]
fn test_sme2_few_rows_args_accepts_and_normalises() {
    for trans in [false, true] {
        let case = Case::<f32>::new(5, 3, 4, 2, trans).at_column(2);
        let ops = Operands::new(&case);
        let mut c = ops.c.clone();
        let args = args_of(&case, &ops, &mut c, cols_of(&case), case.kc)
            .unwrap_or_else(|| panic!("declined {case:?}"));
        assert_eq!((args.m, args.n, args.k, args.kc), (5, 3, 4, 2));
        assert_eq!(args.lda, ops.lda);
        assert_eq!(args.ldb, ops.ldb);
        assert_eq!(args.ldc, ops.ldc);
        assert_eq!(args.b_trans, u64::from(trans));
        assert_eq!(args.a, ops.a[ops.a_off..].as_ptr());
        let origin = if trans { 2 } else { 2 * ops.ldb };
        assert_eq!(args.b, ops.b[ops.b_off + origin..].as_ptr());
        assert_eq!(args.c.cast_const(), c[ops.c_off..].as_ptr());
        // kc above k is clamped to k.
        let args = args_of(&case, &ops, &mut c, cols_of(&case), 99)
            .unwrap_or_else(|| panic!("declined {case:?}"));
        assert_eq!(args.kc, 4);
    }
    // Strides of dimensions of length 1 address nothing and are normalised.
    let case = Case::<f64>::new(3, 1, 1, 1, false);
    let ops = Operands::new(&case);
    let mut c = ops.c.clone();
    let args = args_of(&case, &ops, &mut c, cols_of(&case), 1)
        .unwrap_or_else(|| panic!("declined {case:?}"));
    assert_eq!((args.lda, args.ldb, args.ldc), (0, 0, 3));
    let case = Case::<f64>::new(3, 2, 1, 1, true);
    let ops = Operands::new(&case);
    let mut c = ops.c.clone();
    let args = args_of(&case, &ops, &mut c, cols_of(&case), 1)
        .unwrap_or_else(|| panic!("declined {case:?}"));
    assert_eq!((args.lda, args.ldb, args.ldc), (0, 0, ops.ldc));
}

/// Every decline of `few_rows_args` (C untouched by construction: it only
/// takes addresses; `tests_few_rows_grid.rs` runs the same declines through
/// `few_rows_gemm` on hardware).
#[test]
fn test_sme2_few_rows_args_declines() {
    let case = Case::<f32>::new(4, 3, 5, 2, false);
    let ops = Operands::new(&case);
    let mut c = ops.c.clone();
    let span = cols_of(&case);
    for cols in [
        Span { start: 1, len: 0 },
        Span { start: 3, len: 3 },
        Span {
            start: usize::MAX,
            len: 2,
        },
    ] {
        assert!(args_of(&case, &ops, &mut c, cols, 2).is_none(), "{cols:?}");
    }
    assert!(args_of(&case, &ops, &mut c, span, 0).is_none());
    for alpha in [0.0f32, -0.0] {
        let zero = Case { alpha, ..case };
        assert!(args_of(&zero, &ops, &mut c, span, 2).is_none());
    }
    let b = ops.b_op(&case);
    let a = ops.a_view(&case);
    // More than max_rows rows, or a non-unit row step.
    let tall = vec![1.0f32; 49 * 5 * 2];
    let tall_view = MatRef::from_strided(&tall, 49, 5, 49, 1).unwrap_or_else(|| panic!("tall"));
    let strided = MatRef::from_strided(&tall, 4, 5, 40, 2).unwrap_or_else(|| panic!("strided"));
    let mut c_tall = vec![0.0f32; 49 * 3];
    for (view, rows) in [(tall_view, 49), (strided, 4)] {
        let mut c_view =
            MatMut::from_strided(&mut c_tall, rows, 3, rows).unwrap_or_else(|| panic!("C"));
        assert!(few_rows_args(&view, &b, span, 1.0, 0.0, 2, &mut c_view).is_none());
    }
    // A depth other than B's.
    let short_a = a.submatrix(0, 0, 4, 4);
    let mut c_view = ops.c_view(&case, &mut c);
    assert!(few_rows_args(&short_a, &b, span, 1.0, 0.0, 2, &mut c_view).is_none());
    // A C view of another shape, and a C with overlapping columns.
    let mut other = vec![0.0f32; 64];
    for (rows, cols, ld) in [(3, 3, 4), (4, 2, 4), (4, 3, 3)] {
        // SAFETY: a test-local buffer of 64 elements: every offset of a
        // `rows × cols` view with stride `ld` is below 16.
        let mut view = unsafe { MatMut::new(other.as_mut_ptr(), rows, cols, ld) };
        assert!(
            few_rows_args(&a, &b, span, 1.0, 0.0, 2, &mut view).is_none(),
            "{rows} x {cols} / {ld}"
        );
    }
    // The f64 limit is 32 rows.
    let wide = vec![1.0f64; 33 * 2];
    let wide_view = MatRef::from_strided(&wide, 33, 2, 33, 1).unwrap_or_else(|| panic!("wide"));
    let b64 = vec![1.0f64; 2 * 4];
    let b64_view = MatRef::from_strided(&b64, 2, 4, 2, 1).unwrap_or_else(|| panic!("b64"));
    let b64_op = ColMajor::normalize(b64_view, Trans::NoTrans).unwrap_or_else(|| panic!("b64"));
    let mut c64 = vec![0.0f64; 33 * 4];
    let all = Span { start: 0, len: 4 };
    {
        let mut c64_view = MatMut::from_strided(&mut c64, 33, 4, 33).unwrap_or_else(|| panic!("c"));
        assert!(few_rows_args(&wide_view, &b64_op, all, 1.0, 0.0, 2, &mut c64_view).is_none());
    }
    let ok_view = wide_view.submatrix(0, 0, 32, 2);
    let mut c64_ok = MatMut::from_strided(&mut c64, 32, 4, 33).unwrap_or_else(|| panic!("c"));
    assert!(few_rows_args(&ok_view, &b64_op, all, 1.0, 0.0, 2, &mut c64_ok).is_some());
}

/// More declines: no rows, and strides whose byte
/// arithmetic overflows (in a dimension longer than 1, so they address
/// elements), next to the same views accepted with sane strides.
#[test]
fn test_sme2_few_rows_args_declines_empty_and_overflow() {
    let case = Case::<f32>::new(4, 3, 5, 2, false);
    let ops = Operands::new(&case);
    let b = ops.b_op(&case);
    let span = cols_of(&case);
    let buffer = vec![1.0f32; 64];
    let mut c_buffer = vec![0.0f32; 64];
    // m == 0: an empty A (and C) view.
    let empty = MatRef::from_strided(&buffer, 0, 5, 1, 1).unwrap_or_else(|| panic!("empty A"));
    {
        let mut c_empty =
            MatMut::from_strided(&mut c_buffer, 0, 3, 1).unwrap_or_else(|| panic!("empty C"));
        assert!(few_rows_args(&empty, &b, span, 1.0, 0.0, 2, &mut c_empty).is_none());
    }
    // A huge depth stride with k > 1 (the view is never read: the call
    // declines before any address is formed).
    let huge = usize::MAX / 2;
    // SAFETY: the view is only inspected (strides, shape, pointer); nothing
    // dereferences it or offsets its pointer.
    let wide_a = unsafe { MatRef::new_strided(buffer.as_ptr(), 4, 5, huge, 1) };
    {
        let mut scratch = ops.c.clone();
        let mut c_view = ops.c_view(&case, &mut scratch);
        assert!(few_rows_args(&wide_a, &b, span, 1.0, 0.0, 2, &mut c_view).is_none());
    }
    // A huge leading dimension of the stored B between columns (NoTrans).
    // SAFETY: as above.
    let wide_b = unsafe { MatRef::new_strided(buffer.as_ptr(), 5, 6, huge, 1) };
    let wide_b = ColMajor::normalize(wide_b, Trans::NoTrans).unwrap_or_else(|| panic!("B"));
    {
        let mut scratch = ops.c.clone();
        let mut c_view = ops.c_view(&case, &mut scratch);
        assert!(
            few_rows_args(&ops.a_view(&case), &wide_b, span, 1.0, 0.0, 2, &mut c_view).is_none()
        );
    }
    // A huge column stride of C with more than one column.
    {
        // SAFETY: as above.
        let mut wide_c = unsafe { MatMut::new(c_buffer.as_mut_ptr(), 4, 3, huge) };
        assert!(few_rows_args(&ops.a_view(&case), &b, span, 1.0, 0.0, 2, &mut wide_c).is_none());
    }
    // The same huge strides in dimensions of length 1 are accepted and
    // normalised: A with k = 1, op(B) with one column, C with one column.
    let deep = Case::<f32>::new(4, 1, 1, 1, false);
    let deep_ops = Operands::new(&deep);
    // SAFETY: as above; a 4 x 1 view addresses only its first column.
    let a_k1 = unsafe { MatRef::new_strided(deep_ops.a[deep_ops.a_off..].as_ptr(), 4, 1, huge, 1) };
    let mut c1 = deep_ops.c.clone();
    // SAFETY: a 4 x 1 view addresses only its first column.
    let mut c_n1 = unsafe { MatMut::new(c1[deep_ops.c_off..].as_mut_ptr(), 4, 1, huge) };
    let b1 = deep_ops.b_op(&deep);
    let args = few_rows_args(&a_k1, &b1, cols_of(&deep), 1.0, 0.0, 1, &mut c_n1)
        .unwrap_or_else(|| panic!("declined the length-1 strides"));
    assert_eq!((args.lda, args.ldb, args.ldc), (0, 0, 4));
}

// ---------------------------------------------------------------------------
// The scalar model, the self-check and the test oracle (plain Rust)
// ---------------------------------------------------------------------------

/// `kernel` (plain Rust: the model or a stand-in) on `case`.
fn model<T: FrElem>(case: &Case<T>, ops: &Operands<T>, kernel: FewRowsKernel<T>) -> Vec<T> {
    let mut c = ops.c.clone();
    let args = args_of(case, ops, &mut c, cols_of(case), case.kc)
        .unwrap_or_else(|| panic!("declined {case:?}"));
    // SAFETY: `few_rows_args` accepted the views, so every address of the
    // block is valid; the kernels passed here are plain Rust.
    unsafe { kernel(&args) };
    c
}

fn model_matches_reference<T: FrElem>() {
    let mut index = 0usize;
    for m in [1, T::TILE_ROWS, T::TILE_ROWS + 1, max_rows::<T>()] {
        for n in [1, T::NR, T::NR + 1, 2 * T::NR + 3] {
            for k in [1, 8, 17, 65] {
                for kc in [0, 32, 33] {
                    for trans in [false, true] {
                        let (alpha, beta) = PAIRS[index % PAIRS.len()];
                        index += 1;
                        let kc = if kc == 0 { k } else { kc };
                        let case = Case::<T>::new(m, n, k, kc, trans).scalars(alpha, beta);
                        let ops = Operands::new(&case);
                        let mut want = ops.c.clone();
                        reference_blocks(&case, &ops, &mut want);
                        let got = model(&case, &ops, oracle_few_rows::<T>);
                        assert_c(&case, &ops, &got, &want, "model vs reference");
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_model_matches_reference() {
    model_matches_reference::<f32>();
    model_matches_reference::<f64>();
}

/// The test oracle (`reference_blocks`) equals the blocked path of `gemm_planned` (in every
/// build: see `matches_neon` in the grid file), with the KC step its blocking uses.
fn reference_matches_neon<T: FrElem>() {
    let mut index = 0usize;
    for m in [1, 2, T::TILE_ROWS - 1, T::TILE_ROWS + 1, max_rows::<T>()] {
        for n in [1, T::NR - 1, 2 * T::NR + 1, 4 * T::NR + 3] {
            for (k, kc) in [(1, 8), (9, 8), (63, 32), (65, 64), (130, 64), (200, 48)] {
                for trans in [false, true] {
                    let (alpha, beta) = PAIRS[index % PAIRS.len()];
                    index += 1;
                    let case = Case::<T>::new(m, n, k, kc, trans).scalars(alpha, beta);
                    let ops = Operands::new(&case);
                    let (neon, used_kc) = neon_blocked(&case, &ops);
                    let case = Case {
                        kc: used_kc,
                        ..case
                    };
                    let mut want = ops.c.clone();
                    reference_blocks(&case, &ops, &mut want);
                    assert_c(&case, &ops, &neon, &want, "NEON vs reference");
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_reference_matches_neon() {
    reference_matches_neon::<f32>();
    reference_matches_neon::<f64>();
}

/// A stand-in that applies beta on every depth block (`mode_first` and
/// `beta_bits` throughout).
///
/// # Safety
///
/// As `oracle_few_rows`.
unsafe fn beta_every_block<T: SmeElement>(args: &FewRowsArgs<T>) {
    let wrong = FewRowsArgs {
        one_bits: args.beta_bits,
        mode_later: args.mode_first,
        ..*args
    };
    // SAFETY: forwarded.
    unsafe { oracle_few_rows(&wrong) }
}

/// A stand-in that ignores the KC partition (one depth block).
///
/// # Safety
///
/// As `oracle_few_rows`.
unsafe fn one_block<T: SmeElement>(args: &FewRowsArgs<T>) {
    let wrong = FewRowsArgs {
        kc: args.k,
        ..*args
    };
    // SAFETY: forwarded.
    unsafe { oracle_few_rows(&wrong) }
}

/// A stand-in that drops the last depth step.
///
/// # Safety
///
/// As `oracle_few_rows`.
unsafe fn short_depth<T: SmeElement>(args: &FewRowsArgs<T>) {
    let k = args.k.saturating_sub(1).max(1);
    let wrong = FewRowsArgs {
        k,
        kc: args.kc.min(k),
        ..*args
    };
    // SAFETY: forwarded; a shorter depth reads less.
    unsafe { oracle_few_rows(&wrong) }
}

/// A stand-in that computes the right block and then writes the element
/// right after the last row of column 0 (a guard of the check).
///
/// # Safety
///
/// As `oracle_few_rows`, and `ldc > m` (the check's C has that row).
unsafe fn writes_guard<T: SmeElement>(args: &FewRowsArgs<T>) {
    // SAFETY: forwarded; the check's C buffer has `ldc = m + 1`.
    unsafe {
        oracle_few_rows(args);
        *args.c.add(args.m) = T::ZERO;
    }
}

/// A stand-in that skips the last column.
///
/// # Safety
///
/// As `oracle_few_rows`.
unsafe fn drops_column<T: SmeElement>(args: &FewRowsArgs<T>) {
    let wrong = FewRowsArgs {
        n: args.n.saturating_sub(1).max(1),
        ..*args
    };
    // SAFETY: forwarded; fewer columns.
    unsafe { oracle_few_rows(&wrong) }
}

/// A stand-in that computes the right block and then writes the element
/// right before `A(0, 0)` (a guard of the check's A buffer).
///
/// # Safety
///
/// As `oracle_few_rows`, and the element before A must be writable memory of
/// the caller (the check's A starts at offset 1 of a buffer it owns).
unsafe fn writes_before_a<T: SmeElement>(args: &FewRowsArgs<T>) {
    // SAFETY: forwarded; the check's A buffer has one guard before A.
    unsafe {
        oracle_few_rows(args);
        *args.a.cast_mut().wrapping_sub(1) = T::ZERO;
    }
}

/// A stand-in that computes the right block and then writes the element
/// right after `op(B)(k - 1, n - 1)` (a padding guard of the check's B).
///
/// # Safety
///
/// As `oracle_few_rows`, and that element must be writable memory of the
/// caller (the check's stored B has a padding row after op(B)).
unsafe fn writes_after_b<T: SmeElement>(args: &FewRowsArgs<T>) {
    let (p, j) = (args.k - 1, args.n - 1);
    let last = if args.b_trans == 0 {
        p + j * args.ldb
    } else {
        j + p * args.ldb
    };
    // SAFETY: forwarded; the check's stored B has a guard row after op(B).
    unsafe {
        oracle_few_rows(args);
        *args.b.cast_mut().wrapping_add(last + 1) = T::ZERO;
    }
}

/// A stand-in that computes the right block and then writes the element
/// right before `C(0, 0)` (the check's front guard of C).
///
/// # Safety
///
/// As `oracle_few_rows`, and the element before C must be writable memory of
/// the caller (the check's C has guards before it).
unsafe fn writes_before_c<T: SmeElement>(args: &FewRowsArgs<T>) {
    // SAFETY: forwarded; the check's C buffer has guards before C.
    unsafe {
        oracle_few_rows(args);
        *args.c.wrapping_sub(1) = T::ZERO;
    }
}

fn self_check_rejects<T: FrElem>() {
    // SAFETY: plain-Rust kernels (no SME code); the check's blocks are valid.
    unsafe {
        assert!(
            self_check_with::<T>(oracle_few_rows::<T>),
            "the model itself"
        );
        for (name, wrong) in [
            (
                "beta every block",
                beta_every_block::<T> as FewRowsKernel<T>,
            ),
            ("one block", one_block::<T>),
            ("short depth", short_depth::<T>),
            ("writes a guard", writes_guard::<T>),
            ("drops a column", drops_column::<T>),
            ("writes before A", writes_before_a::<T>),
            ("writes after B", writes_after_b::<T>),
            ("writes before C", writes_before_c::<T>),
        ] {
            assert!(!self_check_with::<T>(wrong), "{name} passed the check");
        }
    }
}

#[test]
fn test_sme2_few_rows_self_check_rejects_wrong_stand_ins() {
    self_check_rejects::<f32>();
    self_check_rejects::<f64>();
}

// ---------------------------------------------------------------------------
// The registered templates: lint and structure
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_few_rows_registered_and_linted() {
    let names: Vec<&str> = SME_KERNELS.iter().map(|entry| entry.name).collect();
    assert_eq!(names, ["sme2_few_rows_f32", "sme2_few_rows_f64"]);
    for entry in SME_KERNELS {
        assert!(entry.file.ends_with("few_rows.rs"), "{}", entry.file);
        assert_eq!(entry.f64f64, entry.name.ends_with("f64"), "{}", entry.name);
        if let Err(errors) = lint_kernel(entry) {
            panic!("{}:\n{}", entry.name, report(&errors));
        }
        // The lint also proves exactly one streaming region per template
        // (one plain smstart before one plain smstop, `Rule::SmstartSmstop`),
        // so one kernel call is one region.
    }
}

/// What the structure checker needs to know about a kernel.
#[derive(Debug, Clone, Copy)]
pub(super) struct Shape {
    /// Element suffix: `s` or `d`.
    pub(super) e: char,
    /// Rows of a tile (16 / 8) = depth steps of a chunk.
    pub(super) tile: usize,
    /// C tiles at most (3 / 4).
    pub(super) tiles: usize,
    /// The transposer tile (3 / 4).
    pub(super) transposer: usize,
    /// Element size shift (2 / 3).
    pub(super) shift: usize,
}

/// The shapes of the two kernels, in registry order.
pub(super) const SHAPES: [Shape; 2] = [
    Shape {
        e: 's',
        tile: 16,
        tiles: 3,
        transposer: 3,
        shift: 2,
    },
    Shape {
        e: 'd',
        tile: 8,
        tiles: 4,
        transposer: 4,
        shift: 3,
    },
];

/// The body statements of a template (between `msr fpsr, xzr` and the exit
/// label), escapes resolved.
pub(super) fn body_lines(template: &str) -> Vec<String> {
    let lines: Vec<String> = template
        .lines()
        .map(|line| line.trim().replace("{{", "{").replace("}}", "}"))
        .collect();
    let start = lines
        .iter()
        .position(|line| line == "msr fpsr, xzr")
        .map_or(0, |at| at + 1);
    let end = lines
        .iter()
        .position(|line| line == "90:")
        .unwrap_or(lines.len());
    lines[start..end].to_vec()
}

/// The mnemonic and the operands of a statement.
fn split(line: &str) -> (&str, &str) {
    line.split_once(' ').unwrap_or((line, ""))
}

/// The base register of the memory operand of `line`, if it has one.
fn base_of(line: &str) -> Option<&str> {
    let (mnemonic, _) = split(line);
    if !(mnemonic.starts_with("ld") || mnemonic.starts_with("st") || mnemonic == "prfm") {
        return None;
    }
    let open = line.rfind('[')?;
    let inner = &line[open + 1..];
    let end = inner.find([',', ']'])?;
    Some(inner[..end].trim())
}

/// The token after the first operand group of a load or store: the
/// governing predicate (`p1/z`, `pn9/z`, `p2`).
fn governing(line: &str) -> Option<&str> {
    let (_, operands) = split(line);
    let after = operands.split_once("}, ")?.1;
    after.split(", ").next()
}

/// The label a line defines (`"20"` for `20:`).
fn label(line: &str) -> Option<&str> {
    line.strip_suffix(':')
        .filter(|name| !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_digit()))
}

/// The lines from label `from` up to (not including) the next label of `to`.
fn section<'l>(lines: &'l [String], from: &str, to: &[&str]) -> Result<&'l [String], String> {
    let start = lines
        .iter()
        .position(|line| label(line) == Some(from))
        .ok_or_else(|| format!("label {from} missing"))?;
    let end = lines[start + 1..]
        .iter()
        .position(|line| label(line).is_some_and(|name| to.contains(&name)))
        .map_or(lines.len(), |at| start + 1 + at);
    Ok(&lines[start..end])
}

/// The tile numbers of a `zero {…}` list.
fn zero_tiles(line: &str, e: char) -> Option<Vec<usize>> {
    let list = line.strip_prefix("zero {")?.strip_suffix('}')?;
    let suffix = format!(".{e}");
    list.split(", ")
        .map(|tile| {
            tile.strip_prefix("za")?
                .strip_suffix(suffix.as_str())?
                .parse()
                .ok()
        })
        .collect()
}

/// The ZA tile named first in `text` (`za<t>h.`, `za<t>v.`, `za<t>.`).
fn za_tile(text: &str) -> Option<usize> {
    let at = text.find("za")?;
    let digits: String = text[at + 2..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// The C tile of a `[x13…]` address: 0 for `[x13]`, `t` for
/// `[x13, x<13+t>, lsl #shift]`.
fn c_tile(line: &str, shift: usize) -> Result<usize, String> {
    let rest = line
        .split_once("[x13")
        .map(|(_, rest)| rest)
        .ok_or_else(|| format!("C address of `{line}`"))?;
    if rest == "]" {
        return Ok(0);
    }
    let reg: usize = rest
        .strip_prefix(", x")
        .and_then(|tail| tail.split(',').next())
        .and_then(|digits| digits.parse().ok())
        .ok_or_else(|| format!("C offset of `{line}`"))?;
    if reg < 14 || rest != format!(", x{reg}, lsl #{shift}]") {
        return Err(format!("C address of `{line}`"));
    }
    Ok(reg - 13)
}

/// The last line before `at` whose destination is `x16` / `w16`.
fn last_x16_writer(lines: &[String], at: usize) -> Option<&str> {
    lines[..at]
        .iter()
        .rev()
        .find(|line| {
            let (_, operands) = split(line);
            operands.starts_with("x16,") || operands.starts_with("w16,")
        })
        .map(String::as_str)
}

/// Proves the structure of one few-rows template (see the module docs):
///
/// 1. Predicates: `p0` = `ptrue`; `p1`, `pn9` = `whilelt` from 0 to `m`
///    (`x20`), `p<t+1>` from `tile · t` to `m`, each written once before the
///    first depth block; `p5` written only by `whilelt p5, x12, x11` right
///    after `mov w12, #0` (from 0 to the depth steps left); `p6` once, at the
///    head of the column panel, by `whilelt p6, x28, x21` (from `j0` to `n`);
///    no other predicate written.
/// 2. Memory: every access has base `x0` (`ldr` of a field), `x14` (A:
///    `p1/z` for one register, `pn9/z` for a group), `x16` (NoTrans B: a
///    horizontal slice of the transposer under `p5/z`, in a loop bounded by
///    `cmp w12, w8`), `x15` (Trans B: one register under `p6/z`) or `x13`
///    (C: tile `t` at `[x13]` / `[x13, x<13+t>, lsl #s]` under `p<t+1>`, ZA
///    slice stores of tile `t` or `z16` after the MOVA of tile `t`, loads of
///    `z17` for tile `t` in M1 / M3 only); every store is a C store.
/// 3. Tiles: variant `4(2L-1)` starts with `zero` of exactly `za0..za(L-1)`,
///    accumulates exactly those tiles, moves no C tile, and its inner labels
///    `2L?` are defined and targeted only inside it; in each epilogue tile
///    `t ≥ 1` is reached only after `cmp x20, #tile·t` / `b.ls` to the
///    column step, and the row offsets are `mov x<13+t>, #tile·t`.
/// 4. Depth blocks: `x25` (mode) is written in the body only by the switch
///    to `mode_later`, `z29` (beta) only from `beta` before the first block
///    and from `one` in that switch, which follows the close of the column
///    loop and ends with `cmp x24, x22` / `b.lo 70b`.
pub(super) fn check_structure(template: &str, shape: Shape) -> Result<(), String> {
    let lines = body_lines(template);
    let e = shape.e;
    let first_block = lines
        .iter()
        .position(|line| label(line) == Some("70"))
        .ok_or("label 70 missing")?;

    // 1. Predicates.
    let mut writers: Vec<(usize, usize)> = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        for used in predicate_uses(line) {
            if used.written {
                writers.push((used.register, at));
            }
        }
    }
    for &(register, at) in &writers {
        let line = lines[at].as_str();
        let previous = if at == 0 { "" } else { lines[at - 1].as_str() };
        let ok = match register {
            0 => line == format!("ptrue p0.{e}"),
            1 => {
                at < first_block
                    && line == format!("whilelt p1.{e}, x16, x20")
                    && previous == "mov x16, #0"
            }
            9 => {
                at < first_block
                    && line == format!("whilelt pn9.{e}, x16, x20, vlx4")
                    && last_x16_writer(&lines, at) == Some("mov x16, #0")
            }
            5 => line == format!("whilelt p5.{e}, x12, x11") && previous == "mov w12, #0",
            6 => line == format!("whilelt p6.{e}, x28, x21"),
            t if (2..=shape.tiles).contains(&t) => {
                at < first_block
                    && line == format!("whilelt p{t}.{e}, x16, x20")
                    && previous == format!("mov x16, #{}", shape.tile * (t - 1))
            }
            _ => false,
        };
        if !ok {
            return Err(format!("predicate p{register} written by `{line}`"));
        }
    }
    let count = |register: usize| writers.iter().filter(|(r, _)| *r == register).count();
    if count(5) == 0 {
        return Err("p5 is never written".into());
    }
    for register in [0, 1, 6, 9].into_iter().chain(2..=shape.tiles) {
        if count(register) != 1 {
            return Err(format!("p{register} written {} times", count(register)));
        }
    }
    let p6_at = writers
        .iter()
        .find(|(r, _)| *r == 6)
        .map_or(0, |(_, at)| *at);
    let panel = lines
        .iter()
        .position(|line| label(line) == Some("20"))
        .ok_or("label 20 missing")?;
    if !(panel < p6_at
        && lines[panel + 1..p6_at]
            .iter()
            .all(|line| label(line).is_none()))
    {
        return Err("p6 is not written at the head of the column panel".into());
    }

    // 2. Memory accesses.
    let mut tile_of_z16: Option<usize> = None;
    let mut mode: Option<usize> = None;
    let mut counts = [0usize; 4];
    let transposer = format!("{{za{}h.{e}[w12, 0]}}", shape.transposer);
    for (at, line) in lines.iter().enumerate() {
        if let Some(name) = label(line) {
            mode = match name {
                "51" | "52" | "53" | "54" => name.parse::<usize>().ok().map(|m| m - 51),
                "59" => None,
                _ => mode,
            };
            continue;
        }
        let (mnemonic, operands) = split(line);
        if mnemonic == "mova" && operands.starts_with(&format!("z16.{e}")) {
            tile_of_z16 = za_tile(operands);
        }
        let Some(base) = base_of(line) else {
            continue;
        };
        let load = mnemonic.starts_with("ld");
        let store = mnemonic.starts_with("st");
        match base {
            "x0" if mnemonic == "ldr" => {}
            "x14" if load => {
                counts[0] += 1;
                let group = operands
                    .split_once('}')
                    .is_some_and(|(dest, _)| dest.contains('-'));
                let want = if group { "pn9/z" } else { "p1/z" };
                if governing(line) != Some(want) {
                    return Err(format!("A load `{line}` not governed by {want}"));
                }
            }
            "x16" if load => {
                counts[1] += 1;
                if !operands.starts_with(&transposer) || governing(line) != Some("p5/z") {
                    return Err(format!(
                        "B column load `{line}`: not a transposer slice under p5/z"
                    ));
                }
                let next = &lines[at + 1..(at + 5).min(lines.len())];
                let bounded = next.iter().any(|l| l == "add w12, w12, #1")
                    && next.iter().any(|l| l == "cmp w12, w8");
                if !bounded {
                    return Err(format!("B column loop at `{line}` not bounded by w8"));
                }
            }
            "x15" if load => {
                counts[2] += 1;
                let one = operands.starts_with(&format!("{{z8.{e}}}"))
                    || operands.starts_with(&format!("{{z9.{e}}}"));
                if !one || governing(line) != Some("p6/z") {
                    return Err(format!("B row load `{line}`: not one register under p6/z"));
                }
            }
            "x13" if load || store => {
                counts[3] += 1;
                let Some(m) = mode else {
                    return Err(format!("C access `{line}` outside an epilogue"));
                };
                let t = c_tile(line, shape.shift)?;
                let want = format!("p{}", t + 1);
                if governing(line).map(|g| g.trim_end_matches("/z")) != Some(want.as_str()) {
                    return Err(format!(
                        "C access `{line}` (tile {t}) not governed by {want}"
                    ));
                }
                if load {
                    if !matches!(m, 1 | 3) {
                        return Err(format!("C read in mode M{m}: `{line}`"));
                    }
                    if !operands.starts_with(&format!("{{z17.{e}}}")) || tile_of_z16 != Some(t) {
                        return Err(format!(
                            "C load `{line}` for tile {t} after {tile_of_z16:?}"
                        ));
                    }
                } else {
                    let from_za = za_tile(operands);
                    let tile = if operands.starts_with(&format!("{{z16.{e}}}")) {
                        tile_of_z16
                    } else {
                        from_za
                    };
                    if tile != Some(t) || (m == 0) != from_za.is_some() {
                        return Err(format!("C store `{line}` (M{m}, tile {t}, from {tile:?})"));
                    }
                }
            }
            _ => return Err(format!("memory access with base {base}: `{line}`")),
        }
    }
    if counts.contains(&0) {
        return Err(format!(
            "missing accesses (A, B columns, B rows, C): {counts:?}"
        ));
    }

    // 3. Tiles of the variants and of the epilogues.
    let variant_labels: Vec<String> = (1..=shape.tiles)
        .map(|live| format!("4{}", 2 * live - 1))
        .collect();
    let mut stops: Vec<&str> = variant_labels.iter().map(String::as_str).collect();
    stops.push("50");
    for live in 1..=shape.tiles {
        let name = &variant_labels[live - 1];
        let start = lines
            .iter()
            .position(|line| label(line) == Some(name.as_str()))
            .ok_or_else(|| format!("variant {name} missing"))?;
        let body = section(&lines, name, &stops)?;
        let end = start + body.len();
        let want: Vec<usize> = (0..live).collect();
        if body.get(1).and_then(|line| zero_tiles(line, e)) != Some(want.clone()) {
            return Err(format!("variant {name} does not start by zeroing {want:?}"));
        }
        let mut accumulated = vec![false; live];
        for line in body {
            let (mnemonic, operands) = split(line);
            if mnemonic == "zero" && zero_tiles(line, e).as_ref() != Some(&want) {
                return Err(format!("variant {name}: `{line}`"));
            }
            if mnemonic == "fmopa" {
                match za_tile(operands) {
                    Some(t) if t < live => accumulated[t] = true,
                    t => return Err(format!("variant {name} accumulates {t:?}: `{line}`")),
                }
            }
            if mnemonic == "mova" && za_tile(operands).is_some_and(|t| t != shape.transposer) {
                return Err(format!("variant {name} moves a C tile: `{line}`"));
            }
        }
        if accumulated.contains(&false) {
            return Err(format!(
                "variant {name} leaves out a live tile: {accumulated:?}"
            ));
        }
        let inner = format!("2{live}");
        for (at, line) in lines.iter().enumerate() {
            let targets = line.split([' ', ',']).any(|word| {
                word.len() == 4
                    && word.starts_with(&inner)
                    && (word.ends_with('b') || word.ends_with('f'))
            });
            let defines = label(line).is_some_and(|l| l.len() == 3 && l.starts_with(&inner));
            if (targets || defines) && !(start..end).contains(&at) {
                return Err(format!(
                    "label 2{live}? used outside variant {name}: `{line}`"
                ));
            }
        }
    }
    let epilogue = section(&lines, "50", &["51"])?;
    for t in 1..shape.tiles {
        let want = format!("mov x{}, #{}", 13 + t, shape.tile * t);
        if !epilogue.contains(&want) {
            return Err(format!("epilogue row offset `{want}` missing"));
        }
    }
    for m in 0..4 {
        let step = format!("5{}", m + 5);
        let body = section(&lines, &format!("5{}", m + 1), &[step.as_str()])?;
        let mut last_tile = 0;
        for (at, line) in body.iter().enumerate() {
            if base_of(line) != Some("x13") {
                continue;
            }
            let t = c_tile(line, shape.shift)?;
            if t < last_tile {
                return Err(format!("M{m}: tile {t} after tile {last_tile}"));
            }
            if t > last_tile {
                let guard = format!("cmp x20, #{}", shape.tile * t);
                let branch = format!("b.ls {step}f");
                let seen = body[..at]
                    .windows(2)
                    .any(|pair| pair[0] == guard && pair[1] == branch);
                if t != last_tile + 1 || !seen {
                    return Err(format!("M{m}: tile {t} without `{guard}` / `{branch}`"));
                }
                last_tile = t;
            }
        }
        if last_tile + 1 != shape.tiles {
            return Err(format!("M{m} reaches {} tiles", last_tile + 1));
        }
    }

    // 4. Mode and beta of the depth blocks.
    let scalar = if e == 's' { "w16" } else { "x16" };
    let switch_lines = [
        "ldr x25, [x0, #{off_mode_later}]".to_string(),
        format!("ldr {scalar}, [x0, #{{off_one}}]"),
        format!("dup z29.{e}, {scalar}"),
        "cmp x24, x22".to_string(),
        "b.lo 70b".to_string(),
    ];
    let mode_writes = lines
        .iter()
        .filter(|line| {
            let (mnemonic, operands) = split(line);
            let reads_only = ["cmp", "cmn", "tst", "cbz", "cbnz", "tbz", "tbnz"]
                .contains(&mnemonic)
                || mnemonic.starts_with("st");
            !reads_only && (operands.starts_with("x25,") || operands.starts_with("w25,"))
        })
        .count();
    let switch = lines
        .iter()
        .position(|line| *line == switch_lines[0])
        .ok_or("mode switch missing")?;
    if mode_writes != 1 || lines[switch..] != switch_lines {
        return Err(format!("depth-block switch {:?}", &lines[switch..]));
    }
    let beta_writes: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("dup z29"))
        .map(|(at, _)| at)
        .collect();
    let from_beta = format!("ldr {scalar}, [x0, #{{off_beta}}]");
    if beta_writes.len() != 2
        || beta_writes[0] == 0
        || beta_writes[0] > first_block
        || lines[beta_writes[0] - 1] != from_beta
        || beta_writes[1] != switch + 2
    {
        return Err(format!("z29 written at {beta_writes:?}"));
    }
    let closes: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.ends_with(" 20b"))
        .map(|(at, _)| at)
        .collect();
    if closes.len() != 1 || closes[0] + 3 != switch || lines[closes[0]] != "b.lo 20b" {
        return Err(format!(
            "column loop closes at {closes:?}, switch at {switch}"
        ));
    }
    Ok(())
}

#[test]
fn test_sme2_few_rows_template_structure() {
    assert_eq!(SME_KERNELS.len(), SHAPES.len());
    for (entry, shape) in SME_KERNELS.iter().zip(SHAPES) {
        if let Err(error) = check_structure(entry.template, shape) {
            panic!("{}: {error}", entry.name);
        }
    }
}

/// The template with the one line `from` (escaped form, trimmed) replaced
/// by `to` (several lines allowed; empty deletes it). The line must occur
/// exactly `occurrences` times; all are replaced.
fn mutated(template: &str, from: &str, to: &str, occurrences: usize) -> String {
    let lines: Vec<&str> = template.lines().collect();
    let hits = lines.iter().filter(|line| line.trim() == from).count();
    assert_eq!(hits, occurrences, "`{from}` occurs {hits} times");
    lines
        .iter()
        .filter_map(|line| {
            if line.trim() == from {
                (!to.is_empty()).then_some(to)
            } else {
                Some(line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The template with only occurrence `nth` (0-based) of the line `from`
/// (escaped form, trimmed) replaced by `to`; panics when there is no such
/// occurrence.
fn mutated_nth(template: &str, from: &str, to: &str, nth: usize) -> String {
    let mut seen = 0usize;
    let mut hit = false;
    let lines: Vec<&str> = template
        .lines()
        .map(|line| {
            if line.trim() == from {
                seen += 1;
                if seen == nth + 1 {
                    hit = true;
                    return to;
                }
            }
            line
        })
        .collect();
    assert!(hit, "`{from}` occurs {seen} times, no occurrence {nth}");
    lines.join("\n")
}

#[test]
fn test_sme2_few_rows_structure_checker_rejects_mutants() {
    let f32_entry = &SME_KERNELS[0];
    let f64_entry = &SME_KERNELS[1];
    // (kernel, line, replacement, occurrences, what the mutant breaks)
    let mutants: [(usize, &str, &str, usize, &str); 18] = [
        (
            0,
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z0.s}}, p0/z, [x14]",
            5,
            "A row predicate",
        ),
        (
            1,
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]",
            "ld1d {{z0.d-z3.d}}, pn8/z, [x14]",
            10,
            "A counter",
        ),
        (
            0,
            "whilelt p5.s, x12, x11",
            "ptrue p5.s",
            3,
            "depth predicate",
        ),
        (
            1,
            "whilelt p5.d, x12, x11",
            "whilelt p5.d, x12, x26",
            4,
            "depth predicate bound",
        ),
        (
            0,
            "whilelt p6.s, x28, x21",
            "ptrue p6.s",
            1,
            "column predicate",
        ),
        (
            1,
            "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z8.d}}, p0/z, [x15]",
            8,
            "B row predicate",
        ),
        (
            0,
            "ld1w {{za3h.s[w12, 0]}}, p5/z, [x16]",
            "ld1w {{za3h.s[w12, 0]}}, p0/z, [x16]",
            3,
            "B column predicate",
        ),
        (0, "cmp w12, w8", "cmp w12, #16", 7, "column bound"),
        (
            0,
            "whilelt p2.s, x16, x20",
            "whilelt p2.s, x20, x16",
            1,
            "tile 1 predicate",
        ),
        (
            1,
            "mov x16, #16",
            "mov x16, #17",
            1,
            "tile 2 predicate start",
        ),
        (
            0,
            "zero {{za0.s}}",
            "zero {{za0.s, za1.s}}",
            1,
            "dead tile zeroed",
        ),
        (
            1,
            "zero {{za0.d, za1.d, za2.d}}",
            "zero {{za0.d, za1.d, za2.d, za3.d}}",
            1,
            "dead tile zeroed",
        ),
        (
            0,
            "st1w {{za1h.s[w12, 0]}}, p2, [x13, x14, lsl #2]",
            "st1w {{za1h.s[w12, 0]}}, p1, [x13, x14, lsl #2]",
            1,
            "C store predicate",
        ),
        (1, "cmp x20, #16", "nop", 5, "tile guard of the epilogue"),
        (
            0,
            "ldr x25, [x0, #{off_mode_later}]",
            "nop",
            1,
            "later mode",
        ),
        (1, "dup z29.d, x16", "nop", 2, "beta of the later blocks"),
        (0, "mov x14, #16", "mov x14, #15", 1, "row offset of tile 1"),
        (1, "b.lo 20b", "b.lo 20b\nb.lo 20b", 1, "column loop"),
    ];
    for (kernel, from, to, occurrences, what) in mutants {
        let (entry, shape) = if kernel == 0 {
            (f32_entry, SHAPES[0])
        } else {
            (f64_entry, SHAPES[1])
        };
        let text = mutated(entry.template, from, to, occurrences);
        assert!(
            check_structure(&text, shape).is_err(),
            "the checker accepts the mutant `{from}` -> `{to}` ({what}) of {}",
            entry.name
        );
    }
    // A C load in M0 (the first `mova` of M0 does not exist: insert a load).
    let text = mutated(
        f32_entry.template,
        "st1w {{za0h.s[w12, 0]}}, p1, [x13]",
        "ld1w {{z17.s}}, p1/z, [x13]\nst1w {{za0h.s[w12, 0]}}, p1, [x13]",
        1,
    );
    assert!(check_structure(&text, SHAPES[0]).is_err(), "C read in M0");
    // Single occurrences: one edited line among identical
    // ones must already fail the checker.
    let single: [(usize, &str, &str, usize, &str); 7] = [
        (
            1,
            "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z8.d}}, p0/z, [x15]",
            7,
            "the odd-depth Trans B load of the 4-tile variant unpredicated",
        ),
        (
            0,
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]",
            "ld1w {{z0.s-z3.s}}, pn8/z, [x14]",
            2,
            "one A load of the 3-tile partial chunk under an unwritten counter",
        ),
        (
            1,
            "st1d {{z16.d}}, p4, [x13, x16, lsl #3]",
            "st1d {{z16.d}}, p3, [x13, x16, lsl #3]",
            1,
            "the M2 store of tile 3 under the predicate of tile 2",
        ),
        (
            1,
            "cmp x20, #24",
            "cmp x20, #25",
            2,
            "the M1 guard of tile 3 one row late",
        ),
        (
            0,
            "cmp w12, w8",
            "cmp w12, #16",
            1,
            "the B column loop of the 2-tile variant not bounded by ncols",
        ),
        (
            1,
            "zero {{za0.d, za1.d, za2.d, za3.d}}",
            "zero {{za0.d, za1.d, za2.d}}",
            0,
            "a live tile of the 4-tile variant not zeroed",
        ),
        (
            0,
            "fmul z16.s, p1/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p1/z, [x13]\nfmul z16.s, p1/m, z16.s, z28.s",
            0,
            "a C read in M2",
        ),
    ];
    for (kernel, from, to, nth, what) in single {
        let (entry, shape) = if kernel == 0 {
            (f32_entry, SHAPES[0])
        } else {
            (f64_entry, SHAPES[1])
        };
        let text = mutated_nth(entry.template, from, to, nth);
        assert!(
            check_structure(&text, shape).is_err(),
            "the checker accepts the single mutant `{from}` -> `{to}` #{nth} ({what}) of {}",
            entry.name
        );
    }
    // An FMOPA into the transposer.
    let text = mutated(
        f64_entry.template,
        "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
        "fmopa za4.d, p0/m, p0/m, z8.d, z0.d",
        16,
    );
    assert!(check_structure(&text, SHAPES[1]).is_err(), "FMOPA into za4");
}
