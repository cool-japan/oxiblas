//! The harness of the SME2 GEMM driver tests (`tests_gemm.rs`,
//! `tests_stress.rs`, `tests_threads.rs`): the cases, their NEON reference
//! and the assertions.
//!
//! # How a case runs ([`run_case`])
//!
//! A [`Case`] describes one call: dimensions, transposes, alpha and beta,
//! how A and B are stored ([`Store`]: column-major or row-major with a
//! padded leading dimension, or fully strided), the padding of C, whether C
//! starts as NaN, the blocking source and the `Par`. [`run_case`] builds the
//! operands from a fixed xorshift sequence (inexact values in `[-1, 1)`, so
//! a different accumulation order shows in the bits), surrounds the C block
//! with a signalling-NaN border (rows `m..ldc` of every column, a guard
//! column before and after, one element before `C(0, 0)`; unused positions
//! of A and B hold the same sentinel), computes the NEON blocked result with
//! `gemm_planned` (the plan forced onto the blocked route, SME2 forced off,
//! the same blocking source) and calls `driver::try_gemm` with the fitted
//! blocking of the same plan. The [`Outcome`] keeps C before the call, after
//! it and the NEON result, the decision counter and the report of the
//! parallel split. [`run_case_around`] does the same, but hands the
//! `try_gemm` call to a closure of the test, which can change the state of
//! the calling thread for exactly that call (`tests_threads.rs`).
//!
//! - [`assert_same_bits`]: the call was accepted, decided exactly once,
//!   every C element equals the NEON one (`args::same_result`: bitwise,
//!   NaN payloads aside), the border is bitwise untouched, the thread is
//!   out of streaming mode with no ZA state afterwards, and every element is
//!   within a rounding bound of a naive `f64` evaluation
//!   ([`assert_near_f64`]: it catches a harness error that NEON and SME2
//!   would share).
//! - [`assert_matches_neon`]: that, the layout against the spec computed
//!   here ([`assert_spec_layout`]: the route, the split side, panel-aligned
//!   near-equal pieces, and per piece MC, the B cap, NC, the A mode and the
//!   buffer lengths from the spec's formulas, [`spec_part`]), and every
//!   block of every part ran on SME2 as the policy says ([`expected_part`];
//!   checked through the per-part records of the run report,
//!   [`assert_runs`]): the number of SME2 calls that compute C
//!   ([`expected_blocks`]), and the policy record of the part (where every
//!   block read A from, which packer packed B, prefetch, few rows) against
//!   the spec's own walk over the blocks ([`spec_policy`]). The proof that a
//!   kernel ran is not the driver's own count (it says only that the driver
//!   handed the block to the kernel) but `kernel_calls`: the entries that
//!   the kernels themselves counted on the thread that ran the part, over
//!   the part's run, which must equal the SME2 calls plus the ZA-transposer
//!   calls of the part. The driver's NEON fallbacks give the same bits, and
//!   the policy's choices never change a bit, so without the counts a block
//!   or a part that silently fell back, a kernel that never ran, or a wrong
//!   policy branch would pass.
//! - [`assert_kernel_calls_here`]: the same proof for a call made directly
//!   through `try_gemm`, with the exact count of the calling thread.
//! - [`assert_declined`]: the call declined, decided nothing, ran no part,
//!   and the whole C buffer (block and border) is bitwise what it was.
//! - [`check_spec`]: one of the two, as the seams file's decision rule
//!   says ([`spec_accepts`], written out here rather than taken from the
//!   driver's `classify` / `kc_accepted`, so a mutated rule cannot pass its
//!   own test).
//!
//! # The spec of the policy
//!
//! [`spec_route`], [`spec_part`] and [`spec_policy`] write the driver
//! policy out on their own, with the thresholds of the tuning in force on
//! the test's thread (`Tuning::of`, which the overrides of the tests
//! change): few rows for `m <= 48` (`f32`) / `32` (`f64`) when A is
//! column-contiguous or its copy fits the packing cache; else blocked with
//! MC = `balanced_block(rows, mc_target, 64)`, the B cap (2 MiB when the
//! part's rows are one MC block and its packed B of a depth block exceeds
//! 4 MiB, else 4 MiB), A per block in place when
//! `((depth - 1) · lda + rows) · elem <= 2 MiB`, otherwise online packing
//! (reused across jc when `round_up(part rows, 64) · kc · elem <= 4 MiB`
//! and it fits next to B), `Aᵀ` through the ZA transposer, B through the ZA
//! transposer for `op(B) = B`, prefetch for the blocked class. The small
//! class runs only with `Tuning::small_class`.

use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::same_result;
use super::driver::{AMode, BlockCount, Part, PolicyCount, Route, SmeGemm, Tuning, try_gemm};
use super::pack::Span;
use super::parallel::{RunReport, Split, take_last_run};
use super::test_support::{Need, decisions, force_off, kernel_entries_here, sme_state};
use crate::level3::autotune::balanced_block;
use crate::level3::gemm::{BlockingSource, GemmBlocking, call_blocking, gemm_planned};
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan, GemmRoute};
use crate::level3::trsm::Trans;

// ---------------------------------------------------------------------------
// Helpers (reused by the second set of driver tests)
// ---------------------------------------------------------------------------

/// An element type of the driver, with what the tests need.
pub(super) trait TestElem: SmeGemm + core::fmt::Debug {
    /// The hardware a test of this type needs.
    const NEED: Need;
    /// The border sentinel: a signalling NaN, so that any arithmetic that
    /// reaches a border element changes its bits.
    const SENTINEL: Self;
    /// Unit roundoff of the type (`2^-24` / `2^-53`).
    const UNIT_ROUNDOFF: f64;
    /// The smallest positive subnormal of the type, as `f64`.
    const TINY: f64;
    /// Binary exponent by which [`Special::Subnormal`] scales A and B down,
    /// so that every product lies in the subnormal range of the type.
    const SUBNORMAL_SHIFT: i32;
    /// Most rows of the few-rows route, written out (48 for `f32`, 32
    /// for `f64`; not read from `few_rows::max_rows`).
    const FEW_ROWS: usize;
    /// `x` rounded to this type.
    fn of(x: f64) -> Self;
    /// The value as `f64` (exact).
    fn to_f64(self) -> f64;
}

impl TestElem for f32 {
    const NEED: Need = Need::F32;
    const SENTINEL: Self = f32::from_bits(0x7fa0_0000);
    const UNIT_ROUNDOFF: f64 = 1.0 / (1_u64 << 24) as f64;
    const TINY: f64 = f32::from_bits(1) as f64;
    const SUBNORMAL_SHIFT: i32 = 70;
    const FEW_ROWS: usize = 48;
    fn of(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TestElem for f64 {
    const NEED: Need = Need::F64;
    const SENTINEL: Self = f64::from_bits(0x7ff4_0000_0000_0000);
    const UNIT_ROUNDOFF: f64 = 1.0 / (1_u64 << 53) as f64;
    const TINY: f64 = f64::from_bits(1);
    const SUBNORMAL_SHIFT: i32 = 530;
    const FEW_ROWS: usize = 32;
    fn of(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Special values a case plants in its operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Special {
    /// Finite data in `[-1, 1)` only.
    None,
    /// `+Inf` and a quiet NaN in A, `-Inf` in B, and `+Inf` and a NaN in C
    /// (when C is read): NaN positions must match NEON.
    NonFinite,
    /// A and B scaled by `2^-SUBNORMAL_SHIFT` (C by its square), so that
    /// the products, the sums and the results are subnormal.
    Subnormal,
}

/// A fixed xorshift sequence: deterministic data without a dependency.
pub(super) struct Rng(u64);

impl Rng {
    /// A sequence for `seed` (any value).
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// The next 64 random bits.
    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[-1, 1)` with 53 random bits (rounded to `T`).
    pub(super) fn value<T: TestElem>(&mut self) -> T {
        let unit = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        T::of(unit * 2.0 - 1.0)
    }
}

/// How an operand is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Store {
    /// Column-major, leading dimension `rows + pad` (unit column stride).
    ColMajor {
        /// Extra rows between columns.
        pad: usize,
    },
    /// Row-major, leading dimension `cols + pad` (unit row stride: the
    /// driver reads it through its transpose).
    RowMajor {
        /// Extra columns between rows.
        pad: usize,
    },
    /// Neither stride is 1: the driver must decline.
    Strided,
}

/// A stored matrix with sentinel gaps.
#[derive(Debug, Clone)]
pub(super) struct Operand<T> {
    /// The elements, gaps holding [`TestElem::SENTINEL`].
    pub(super) data: Vec<T>,
    /// Rows of the stored matrix.
    pub(super) rows: usize,
    /// Columns of the stored matrix.
    pub(super) cols: usize,
    /// Elements between columns.
    pub(super) row_stride: usize,
    /// Elements between rows.
    pub(super) col_stride: usize,
}

impl<T: TestElem> Operand<T> {
    /// A `rows × cols` matrix stored as `store`, with values from `rng`.
    pub(super) fn new(rows: usize, cols: usize, store: Store, rng: &mut Rng) -> Self {
        let (row_stride, col_stride) = match store {
            Store::ColMajor { pad } => (rows + pad, 1),
            Store::RowMajor { pad } => (1, cols + pad),
            Store::Strided => (2 * rows + 1, 2),
        };
        let len = if rows == 0 || cols == 0 {
            1
        } else {
            (rows - 1) * col_stride + (cols - 1) * row_stride + 1
        };
        let mut data = vec![T::SENTINEL; len];
        for j in 0..cols {
            for i in 0..rows {
                data[i * col_stride + j * row_stride] = rng.value();
            }
        }
        Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        }
    }

    /// Element `(i, p)` of `op(X)` for this stored `X`, as `f64`.
    pub(super) fn op_at(&self, trans: Trans, i: usize, p: usize) -> f64 {
        let (r, c) = match trans {
            Trans::NoTrans => (i, p),
            Trans::Trans | Trans::ConjTrans => (p, i),
        };
        self.data[r * self.col_stride + c * self.row_stride].to_f64()
    }

    /// Sets stored element `(r, c)`.
    fn set(&mut self, r: usize, c: usize, value: T) {
        self.data[r * self.col_stride + c * self.row_stride] = value;
    }

    /// Multiplies every stored element by `2^-shift`.
    fn scale_down(&mut self, shift: i32) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                let x = self.data[r * self.col_stride + c * self.row_stride].to_f64();
                self.set(r, c, T::of(x * 2_f64.powi(-shift)));
            }
        }
    }

    /// The view of the stored matrix.
    pub(super) fn view(&self) -> MatRef<'_, T> {
        MatRef::from_strided(
            &self.data,
            self.rows,
            self.cols,
            self.row_stride,
            self.col_stride,
        )
        .expect("operand view")
    }
}

/// C inside a sentinel border.
#[derive(Debug, Clone)]
pub(super) struct CBuf<T> {
    /// The whole buffer: border and block.
    pub(super) data: Vec<T>,
    /// Rows of the block.
    pub(super) m: usize,
    /// Columns of the block.
    pub(super) n: usize,
    /// Leading dimension (`m + pad`).
    pub(super) ldc: usize,
    /// Index of `C(0, 0)` in `data` (after the wide guard, a guard column
    /// and one element).
    pub(super) offset: usize,
}

impl<T: TestElem> CBuf<T> {
    /// An `m × n` block with leading dimension `m + pad`, holding values from
    /// `rng` (or quiet NaNs when `nan`), inside a sentinel border.
    pub(super) fn new(m: usize, n: usize, pad: usize, nan: bool, rng: &mut Rng) -> Self {
        Self::guarded(m, n, pad, nan, 0, rng)
    }

    /// [`CBuf::new`] with `guard` more sentinels before the guard column and
    /// after the last one (a wide guard: a store that runs a vector or a
    /// panel too far changes a sentinel instead of corrupting the heap).
    pub(super) fn guarded(
        m: usize,
        n: usize,
        pad: usize,
        nan: bool,
        guard: usize,
        rng: &mut Rng,
    ) -> Self {
        let ldc = (m + pad).max(1);
        let offset = guard + ldc + 1;
        let mut data = vec![T::SENTINEL; offset + (n + 1) * ldc + 1 + guard];
        for j in 0..n {
            for i in 0..m {
                data[offset + i + j * ldc] = if nan { T::of(f64::NAN) } else { rng.value() };
            }
        }
        Self {
            data,
            m,
            n,
            ldc,
            offset,
        }
    }

    /// The view of the block.
    pub(super) fn view(&mut self) -> MatMut<'_, T> {
        MatMut::from_strided(&mut self.data[self.offset..], self.m, self.n, self.ldc)
            .expect("C view")
    }

    /// Whether `data[index]` belongs to the block.
    pub(super) fn in_block(&self, index: usize) -> bool {
        index
            .checked_sub(self.offset)
            .is_some_and(|rel| rel < self.n * self.ldc && rel % self.ldc < self.m)
    }

    /// Element `(i, j)` of the block.
    pub(super) fn at(&self, i: usize, j: usize) -> T {
        self.data[self.offset + i + j * self.ldc]
    }

    /// Sets element `(i, j)` of the block.
    fn set(&mut self, i: usize, j: usize, value: T) {
        self.data[self.offset + i + j * self.ldc] = value;
    }
}

/// One `try_gemm` call and its NEON reference.
#[derive(Debug, Clone, Copy)]
pub(super) struct Case<T> {
    /// Transpose of A.
    pub(super) trans_a: Trans,
    /// Transpose of B.
    pub(super) trans_b: Trans,
    /// Rows of C.
    pub(super) m: usize,
    /// Columns of C.
    pub(super) n: usize,
    /// Depth.
    pub(super) k: usize,
    /// alpha.
    pub(super) alpha: T,
    /// beta.
    pub(super) beta: T,
    /// Storage of A.
    pub(super) a_store: Store,
    /// Storage of B.
    pub(super) b_store: Store,
    /// Rows of padding between the columns of C.
    pub(super) c_pad: usize,
    /// Whether C starts as NaN (for `beta == 0`).
    pub(super) c_nan: bool,
    /// Extra sentinels around C ([`CBuf::guarded`]).
    pub(super) c_guard: usize,
    /// Where the block sizes come from (both paths).
    pub(super) source: BlockingSource,
    /// Parallelism of the SME2 call (the reference runs `Par::Seq`).
    pub(super) par: Par,
    /// Special values planted in the operands.
    pub(super) special: Special,
    /// Seed of the data.
    pub(super) seed: u64,
}

impl<T: TestElem> Case<T> {
    /// `C = 0.75 · A · B - 0.5 · C`, column-major operands with padded
    /// leading dimensions, the default blocking, sequential.
    pub(super) fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            trans_a: Trans::NoTrans,
            trans_b: Trans::NoTrans,
            m,
            n,
            k,
            alpha: T::of(0.75),
            beta: T::of(-0.5),
            a_store: Store::ColMajor { pad: 3 },
            b_store: Store::ColMajor { pad: 1 },
            c_pad: 5,
            c_nan: false,
            c_guard: 0,
            source: BlockingSource::Default,
            par: Par::Seq,
            special: Special::None,
            seed: (m * 1_000_003 + n * 1009 + k) as u64,
        }
    }

    /// With `pad` rows of padding between the columns of C (`ldc = m + pad`).
    pub(super) fn c_pad(self, c_pad: usize) -> Self {
        Self { c_pad, ..self }
    }

    /// With `guard` extra sentinels around C.
    pub(super) fn c_guard(self, c_guard: usize) -> Self {
        Self { c_guard, ..self }
    }

    /// With special values.
    pub(super) fn special(self, special: Special) -> Self {
        Self { special, ..self }
    }

    /// With another data seed.
    pub(super) fn seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    /// With transposes.
    pub(super) fn trans(self, trans_a: Trans, trans_b: Trans) -> Self {
        Self {
            trans_a,
            trans_b,
            ..self
        }
    }

    /// With alpha and beta.
    pub(super) fn scalars(self, alpha: f64, beta: f64) -> Self {
        Self {
            alpha: T::of(alpha),
            beta: T::of(beta),
            ..self
        }
    }

    /// With the storage of A and B.
    pub(super) fn stores(self, a_store: Store, b_store: Store) -> Self {
        Self {
            a_store,
            b_store,
            ..self
        }
    }

    /// With an explicit blocking whose KC target is `kc`.
    pub(super) fn explicit_kc(self, kc: usize) -> Self {
        Self {
            source: BlockingSource::Explicit(GemmBlocking {
                mc: 256,
                kc,
                nc: 4096,
            }),
            ..self
        }
    }

    /// With C starting as NaN.
    pub(super) fn nan_c(self) -> Self {
        Self {
            c_nan: true,
            ..self
        }
    }

    /// With `par`.
    pub(super) fn par(self, par: Par) -> Self {
        Self { par, ..self }
    }
}

/// What one case did.
#[derive(Debug)]
pub(super) struct Outcome<T> {
    /// `try_gemm` returned `true`.
    pub(super) accepted: bool,
    /// SME2 decisions taken on this thread by the call.
    pub(super) decided: u64,
    /// The report of the parallel split (`None` when nothing ran).
    pub(super) report: Option<RunReport>,
    /// The depth of the call.
    pub(super) k: usize,
    /// The fitted KC of the call.
    pub(super) kc: usize,
    /// The case.
    pub(super) case: Case<T>,
    /// A as stored.
    pub(super) a: Operand<T>,
    /// B as stored.
    pub(super) b: Operand<T>,
    /// C before the call.
    pub(super) before: CBuf<T>,
    /// C after the SME2 call.
    pub(super) got: CBuf<T>,
    /// C after the NEON blocked reference.
    pub(super) want: CBuf<T>,
}

/// The stored shape of `op(X)` of shape `rows × cols`.
pub(super) fn stored_shape(trans: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match trans {
        Trans::NoTrans => (rows, cols),
        Trans::Trans | Trans::ConjTrans => (cols, rows),
    }
}

/// Runs `case` (see the module docs).
pub(super) fn run_case<T: TestElem>(case: &Case<T>) -> Outcome<T> {
    run_case_around(case, |call| call())
}

/// [`run_case`], with the `try_gemm` call made by `around`: it gets the call
/// as a closure and returns what the call returned. A test can so change the
/// state of the calling thread for exactly that call; the operands, the
/// NEON reference and the counters before and after the call are taken in
/// the thread's usual state.
pub(super) fn run_case_around<T: TestElem>(
    case: &Case<T>,
    around: impl FnOnce(&mut dyn FnMut() -> bool) -> bool,
) -> Outcome<T> {
    let mut rng = Rng::new(case.seed);
    let (ar, ac) = stored_shape(case.trans_a, case.m, case.k);
    let (br, bc) = stored_shape(case.trans_b, case.k, case.n);
    let mut a = Operand::<T>::new(ar, ac, case.a_store, &mut rng);
    let mut b = Operand::<T>::new(br, bc, case.b_store, &mut rng);
    let mut before = CBuf::<T>::guarded(
        case.m,
        case.n,
        case.c_pad,
        case.c_nan,
        case.c_guard,
        &mut rng,
    );
    plant(case, &mut a, &mut b, &mut before);

    let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
    let blocks = call_blocking(case.source, &plan, case.m, case.k, case.n);

    let mut want = before.clone();
    {
        let _neon = force_off();
        gemm_planned(
            case.trans_a,
            case.trans_b,
            case.alpha,
            a.view(),
            b.view(),
            case.beta,
            want.view(),
            Par::Seq,
            case.source,
            &plan,
        );
    }

    let mut got = before.clone();
    let _ = take_last_run();
    let decisions_before = decisions();
    let (a_view, b_view, mut c_view) = (a.view(), b.view(), got.view());
    let accepted = around(&mut || {
        try_gemm(
            case.trans_a,
            case.trans_b,
            case.alpha,
            &a_view,
            &b_view,
            case.beta,
            &mut c_view,
            &blocks,
            case.par,
        )
    });
    let decided = decisions() - decisions_before;
    Outcome {
        accepted,
        decided,
        report: take_last_run(),
        k: case.k,
        kc: blocks.sequential().kc.min(case.k),
        case: *case,
        a,
        b,
        before,
        got,
        want,
    }
}

/// Plants the special values of `case` (see [`Special`]).
fn plant<T: TestElem>(case: &Case<T>, a: &mut Operand<T>, b: &mut Operand<T>, c: &mut CBuf<T>) {
    let empty = a.rows == 0 || a.cols == 0 || b.rows == 0 || b.cols == 0 || c.m == 0 || c.n == 0;
    match case.special {
        Special::None => {}
        _ if empty => {}
        Special::NonFinite => {
            a.set(0, 0, T::of(f64::INFINITY));
            a.set(a.rows - 1, a.cols / 2, T::of(f64::NAN));
            b.set(b.rows / 2, b.cols - 1, T::of(f64::NEG_INFINITY));
            if !case.c_nan {
                c.set(c.m - 1, 0, T::of(f64::INFINITY));
                c.set(c.m / 2, c.n - 1, T::of(f64::NAN));
            }
        }
        Special::Subnormal => {
            a.scale_down(T::SUBNORMAL_SHIFT);
            b.scale_down(T::SUBNORMAL_SHIFT);
            if !case.c_nan {
                for j in 0..c.n {
                    for i in 0..c.m {
                        let x = c.at(i, j).to_f64() * 2_f64.powi(-2 * T::SUBNORMAL_SHIFT);
                        c.set(i, j, T::of(x));
                    }
                }
            }
        }
    }
}

/// Asserts that every C element of an accepted `out` is within a rounding
/// bound of a naive `f64` evaluation of `alpha · op(A) · op(B) + beta · C`
/// (C ignored when `beta == 0`). This guards the harness itself: an error
/// that NEON and SME2 would share (wrong operands, a wrong view) shows here.
/// Elements whose reference or data are not finite are left to the NEON
/// comparison; products above `2^22` multiply-adds check a sample.
#[track_caller]
pub(super) fn assert_near_f64<T: TestElem>(out: &Outcome<T>, label: &str) {
    let case = &out.case;
    let (m, n, k) = (case.m, case.n, case.k);
    let (alpha, beta) = (case.alpha.to_f64(), case.beta.to_f64());
    let step = (m * n * k).div_ceil(1 << 22).max(1);
    let kf = k as f64 + 3.0;
    for j in 0..n {
        for i in 0..m {
            if (i + j * m) % step != 0 {
                continue;
            }
            let (mut sum, mut mag) = (0.0_f64, 0.0_f64);
            for p in 0..k {
                let prod = out.a.op_at(case.trans_a, i, p) * out.b.op_at(case.trans_b, p, j);
                sum += prod;
                mag += prod.abs();
            }
            let c_term = if beta == 0.0 {
                0.0
            } else {
                beta * out.before.at(i, j).to_f64()
            };
            let want = alpha * sum + c_term;
            if !want.is_finite() || !mag.is_finite() {
                continue;
            }
            let got = out.got.at(i, j).to_f64();
            let bound = 2.0
                * (kf * T::UNIT_ROUNDOFF * alpha.abs() * mag
                    + 2.0 * T::UNIT_ROUNDOFF * c_term.abs()
                    + kf * T::TINY * (1.0 + alpha.abs()));
            assert!(
                (got - want).abs() <= bound,
                "{label}: C({i}, {j}) = {got:e}, f64 reference {want:e}, bound {bound:e}"
            );
        }
    }
}

/// Asserts that `out` was accepted and equals the NEON blocked result (see
/// the module docs).
#[track_caller]
pub(super) fn assert_same_bits<T: TestElem>(out: &Outcome<T>, label: &str) {
    assert!(out.accepted, "{label}: declined");
    assert_eq!(out.decided, 1, "{label}: decisions");
    assert!(out.report.is_some(), "{label}: no run report");
    assert_eq!(out.got.data.len(), out.want.data.len());
    for (index, ((got, want), before)) in out
        .got
        .data
        .iter()
        .zip(&out.want.data)
        .zip(&out.before.data)
        .enumerate()
    {
        if out.got.in_block(index) {
            let rel = index - out.got.offset;
            let (i, j) = (rel % out.got.ldc, rel / out.got.ldc);
            assert!(
                same_result(*got, *want),
                "{label}: C({i}, {j}) = {got:?} ({:#x}), NEON {want:?} ({:#x})",
                got.bits(),
                want.bits()
            );
        } else {
            assert_eq!(
                got.bits(),
                before.bits(),
                "{label}: border element {index} changed"
            );
            assert_eq!(
                want.bits(),
                before.bits(),
                "{label}: NEON changed border element {index}"
            );
        }
    }
    assert_eq!(
        sme_state(),
        Some((0, 0)),
        "{label}: SME state after the call"
    );
    assert_near_f64(out, label);
}

/// The number of SME2 calls that compute C in an SME2 `part` for depth `k`
/// in steps of `kc`: one per `(depth, column, row)` block of a blocked part
/// (with the part's MC and NC), one for a few-rows part.
pub(super) fn expected_blocks(part: &Part, k: usize, kc: usize) -> usize {
    match part.route {
        Route::FewRows { .. } => 1,
        Route::Blocked => {
            k.div_ceil(kc) * part.cols.len.div_ceil(part.nc) * part.rows.len.div_ceil(part.mc)
        }
    }
}

/// The record of a part of `out` that ran on SME2 with no NEON block: the
/// spec's number of SME2 calls ([`expected_blocks`] of the spec's part) and
/// the spec's policy record ([`spec_policy`]). For the closures of
/// [`assert_runs`].
pub(super) fn expected_part<T: TestElem>(out: &Outcome<T>, part: &Part) -> BlockCount {
    let spec =
        spec_part(out, part.rows, part.cols).expect("the spec declines a part that the driver ran");
    BlockCount {
        sme: expected_blocks(&spec, out.k, out.kc),
        neon: 0,
        policy: spec_policy(out, &spec),
        ..BlockCount::default()
    }
}

/// Asserts how every part of `out` ran: `expect(index, part)` gives whether
/// the part ran on the NEON sequential driver, and its block counts (`sme`
/// and `neon`; its `kernel_calls` and `policy` are ignored: the policy of
/// an SME2 part is always checked against [`spec_policy`]).
///
/// The driver's `sme` count says only that it handed a block to the kernel;
/// the proof that the kernels ran is `kernel_calls`, the kernel entries that
/// the kernels themselves counted on the thread that ran the part, over the
/// part's run. So every part must show `kernel_calls == sme + a_za + b_za`
/// (one entry per SME2 block and per ZA-transposer call, none for a NEON
/// block or a NEON part), `sme` must be the expected number, and the policy
/// record (counted from the argument blocks and around the packer calls)
/// must be the spec's walk over the part's blocks.
#[track_caller]
pub(super) fn assert_runs<T: TestElem>(
    out: &Outcome<T>,
    label: &str,
    expect: impl Fn(usize, &Part) -> (bool, BlockCount),
) {
    let report = out.report.as_ref().expect("run report");
    let parts = report.layout.parts.len();
    assert_eq!(report.neon.len(), parts, "{label}");
    assert_eq!(report.blocks.len(), parts, "{label}");
    assert_eq!(report.threads.len(), parts, "{label}");
    assert_eq!(report.workers.len(), parts, "{label}");
    assert_eq!(report.timed_out.len(), parts, "{label}");
    for (index, part) in report.layout.parts.iter().enumerate() {
        let (neon, blocks) = expect(index, part);
        let got = report.blocks[index];
        assert_eq!(report.neon[index], neon, "{label}: part {index} NEON");
        assert_eq!(
            (got.sme, got.neon),
            (blocks.sme, blocks.neon),
            "{label}: part {index} blocks (SME2, NEON)"
        );
        let want_policy = if neon {
            PolicyCount::default()
        } else {
            let spec = spec_part(out, part.rows, part.cols)
                .expect("the spec declines a part that the driver ran");
            spec_policy(out, &spec)
        };
        assert_eq!(got.policy, want_policy, "{label}: part {index} policy");
        assert_eq!(
            got.kernel_calls,
            (got.sme + got.policy.a_za + got.policy.b_za) as u64,
            "{label}: part {index}: kernel entries on its thread"
        );
    }
}

/// [`assert_same_bits`], and every block of every part ran on SME2 as the
/// policy says: no NEON part, no NEON block, exactly the expected SME2
/// calls and ZA-transposer calls counted by the kernels on the thread that
/// ran the part, and the spec's policy record ([`assert_runs`]). A NEON
/// block or part gives the same bits, and the driver's own block count
/// would not notice a kernel that never ran, so only the kernels' count can
/// tell.
#[track_caller]
pub(super) fn assert_matches_neon<T: TestElem>(out: &Outcome<T>, label: &str) {
    assert_same_bits(out, label);
    assert_spec_layout(out, label);
    assert_runs(out, label, |_, part| (false, expected_part(out, part)));
}

/// For an accepted call made directly through `try_gemm` on this thread,
/// with `entries` the value of `kernel_entries_here()` before it: takes the
/// run report and asserts that every part ran on SME2 with no NEON block,
/// that the kernels counted exactly one entry per SME2 call and per
/// ZA-transposer call on the part's thread, and that this thread's count
/// moved by exactly the entries of the parts that ran here. Returns the
/// number of kernel entries of the call.
#[track_caller]
pub(super) fn assert_kernel_calls_here(entries: u64, label: &str) -> u64 {
    let here = kernel_entries_here() - entries;
    let report = take_last_run().expect("run report");
    let me = std::thread::current().id();
    let mut total = 0;
    let mut on_this_thread = 0;
    for (index, blocks) in report.blocks.iter().enumerate() {
        assert!(!report.neon[index], "{label}: part {index} ran on NEON");
        assert_eq!(blocks.neon, 0, "{label}: part {index}: NEON blocks");
        assert!(blocks.sme > 0, "{label}: part {index}: no SME2 block");
        assert_eq!(
            blocks.kernel_calls,
            (blocks.sme + blocks.policy.a_za + blocks.policy.b_za) as u64,
            "{label}: part {index}: kernel entries on its thread"
        );
        total += blocks.kernel_calls;
        if report.threads[index] == me {
            on_this_thread += blocks.kernel_calls;
        }
    }
    assert_eq!(
        here, on_this_thread,
        "{label}: kernel entries on this thread"
    );
    total
}

/// How a stored operand normalises (`ColMajor::normalize`, from the
/// strides `Operand::new` gives `store`): `Some((flipped, ld))` with
/// `flipped` when the driver reads it through its transpose and `ld` the
/// leading dimension of the column-major matrix it reads; `None` when it
/// declines.
pub(super) fn spec_normalized(store: Store, rows: usize, cols: usize) -> Option<(bool, usize)> {
    let (row_stride, col_stride) = match store {
        Store::ColMajor { pad } => (rows + pad, 1),
        Store::RowMajor { pad } => (1, cols + pad),
        Store::Strided => (2 * rows + 1, 2),
    };
    if rows == 0 || cols == 0 {
        None
    } else if col_stride == 1 || rows == 1 {
        Some((false, row_stride))
    } else if row_stride == 1 || cols == 1 {
        Some((true, col_stride))
    } else {
        None
    }
}

/// Whether `op(X)` of the case's operand reads column-contiguously after
/// normalisation (`op(X) = X` of a column-major X), and the leading
/// dimension of the stored matrix the driver reads; `None` when it
/// declines.
fn spec_operand(store: Store, trans: Trans, rows: usize, cols: usize) -> Option<(bool, usize)> {
    let (stored_rows, stored_cols) = stored_shape(trans, rows, cols);
    let (flipped, ld) = spec_normalized(store, stored_rows, stored_cols)?;
    let transposed = trans != Trans::NoTrans;
    Some((transposed == flipped, ld))
}

/// The spec's route of `case` (few rows when `m <= FEW_ROWS` and A is
/// column-contiguous or its `m × k` copy fits the packing-cache limit of
/// `tuning`; blocked otherwise).
pub(super) fn spec_route<T: TestElem>(case: &Case<T>, tuning: &Tuning) -> Route {
    if case.m > T::FEW_ROWS {
        return Route::Blocked;
    }
    let contiguous = spec_operand(case.a_store, case.trans_a, case.m, case.k)
        .is_some_and(|(contiguous, _)| contiguous);
    if contiguous {
        Route::FewRows { copied: false }
    } else if case.m * case.k * core::mem::size_of::<T>() <= tuning.pack_limit {
        Route::FewRows { copied: true }
    } else {
        Route::Blocked
    }
}

/// Whether the case is in the blocked class (`m · n · k > 32³`): its kernel
/// calls carry `PREFETCH_C`.
fn spec_blocked_class<T: TestElem>(case: &Case<T>) -> bool {
    case.m.saturating_mul(case.n).saturating_mul(case.k) > 32 * 32 * 32
}

/// The spec's part `rows × cols` of `out` (the route, MC, the B cap, NC,
/// the A mode and the buffer lengths), with the tuning in force on this
/// thread; `None` when its buffers exceed the packing-cache limit (the
/// call declines).
pub(super) fn spec_part<T: TestElem>(out: &Outcome<T>, rows: Span, cols: Span) -> Option<Part> {
    spec_part_of(&out.case, out.kc, rows, cols, &Tuning::of::<T>())
}

/// [`spec_part`] for `case` with KC `kc` and `tuning`.
pub(super) fn spec_part_of<T: TestElem>(
    case: &Case<T>,
    kc: usize,
    rows: Span,
    cols: Span,
    tuning: &Tuning,
) -> Option<Part> {
    let route = spec_route(case, tuning);
    if route != Route::Blocked {
        return Some(Part {
            rows,
            cols,
            route,
            mc: rows.len,
            nc: cols.len,
            b_cap: 0,
            a_mode: AMode::InPlace,
            a_len: 0,
            b_len: 0,
        });
    }
    let elem = core::mem::size_of::<T>();
    let nr = T::NR;
    let mc = balanced_block(rows.len, tuning.mc_target, 64);
    let whole_b = cols.len.div_ceil(nr) * nr * kc * elem;
    let b_cap = if rows.len <= mc && whole_b > tuning.b_bytes {
        tuning.b_one_block_bytes.min(tuning.b_bytes)
    } else {
        tuning.b_bytes
    };
    let nc_target = (b_cap / (kc * elem) / nr * nr).max(nr);
    let nc = balanced_block(cols.len, nc_target, nr);
    let b_len = nc.div_ceil(nr) * nr * kc;
    let (contiguous, lda) = spec_operand(case.a_store, case.trans_a, case.m, case.k)?;
    let mc_len = mc.div_ceil(64) * 64 * kc;
    let fits = |a_len: usize| (a_len + b_len) * elem <= tuning.pack_limit;
    let (a_mode, a_len) = if !contiguous {
        (AMode::Za, mc_len)
    } else if ((kc - 1) * lda + mc) * elem <= tuning.in_place_bytes {
        (AMode::InPlace, 0)
    } else {
        let whole = rows.len.div_ceil(64) * 64 * kc;
        if whole * elem <= tuning.a_reuse_bytes && fits(whole) {
            (AMode::OnlineReuse, whole)
        } else {
            (AMode::OnlineRebuild, mc_len)
        }
    };
    fits(a_len).then_some(Part {
        rows,
        cols,
        route,
        mc,
        nc,
        b_cap,
        a_mode,
        a_len,
        b_len,
    })
}

/// The spec's policy record of the SME2 part `part` (a spec part) of
/// `out`: a walk over its blocks in the driver's order. Depth blocks that
/// the test tuning sends to the NEON macro-panel (`neon_depths`) still
/// pack B, but read no A and call no kernel.
pub(super) fn spec_policy<T: TestElem>(out: &Outcome<T>, part: &Part) -> PolicyCount {
    let case = &out.case;
    let tuning = Tuning::of::<T>();
    let mut policy = PolicyCount::default();
    if part.route != Route::Blocked {
        policy.few_rows = 1;
        return policy;
    }
    let elem = core::mem::size_of::<T>();
    let Some((a_contiguous, lda)) = spec_operand(case.a_store, case.trans_a, case.m, case.k) else {
        return policy;
    };
    let b_contiguous = spec_operand(case.b_store, case.trans_b, case.k, case.n)
        .is_some_and(|(contiguous, _)| contiguous);
    let prefetch = spec_blocked_class(case);
    let steps = |total: usize, step: usize| {
        (0..total)
            .step_by(step.max(1))
            .map(move |start| step.min(total - start))
    };
    for (depth_index, depth) in steps(out.k, out.kc).enumerate() {
        let forced = depth_index < 64 && tuning.neon_depths & (1 << depth_index) != 0;
        for (column_block, _) in steps(part.cols.len, part.nc).enumerate() {
            if b_contiguous {
                policy.b_za += 1;
            } else {
                policy.b_plain += 1;
            }
            for rows in steps(part.rows.len, part.mc) {
                if forced {
                    continue;
                }
                policy.prefetch += usize::from(prefetch);
                if !a_contiguous {
                    policy.a_za += 1;
                    policy.a_za_reads += 1;
                } else if ((depth - 1) * lda + rows) * elem <= tuning.in_place_bytes {
                    policy.a_in_place += 1;
                } else if part.a_mode == AMode::OnlineReuse {
                    if column_block == 0 {
                        policy.a_online_fill += 1;
                    } else {
                        policy.a_reused += 1;
                    }
                } else {
                    policy.a_online_rebuilt += 1;
                }
            }
        }
    }
    policy
}

/// Asserts the layout of `out` against the spec, independently of the
/// driver's own arithmetic (so the block counts of [`expected_blocks`],
/// which read the part's MC and NC, are not circular): the route; the
/// pieces tile C along the columns for few rows, else along the longer side
/// (rows when `m >= n`), in runs of whole 64-row / NR-column panels that
/// differ by at most one panel; and every piece equals the spec's part
/// ([`spec_part`]: MC, B cap, NC, A mode, buffer lengths) with the tuning
/// in force on this thread.
#[track_caller]
pub(super) fn assert_spec_layout<T: TestElem>(out: &Outcome<T>, label: &str) {
    let report = out.report.as_ref().expect("run report");
    let (m, n) = (out.case.m, out.case.n);
    let tuning = Tuning::of::<T>();
    let route = spec_route(&out.case, &tuning);
    assert_eq!(report.route, route, "{label}: route");
    let prefetch = if spec_blocked_class(&out.case) { 1 } else { 0 };
    assert_eq!(report.prefetch, prefetch, "{label}: prefetch switches");
    let rows_side = route == Route::Blocked && m >= n;
    assert_eq!(
        report.layout.split,
        if rows_side { Split::Rows } else { Split::Cols },
        "{label}: split side"
    );
    let (len, unit) = if rows_side { (m, 64) } else { (n, T::NR) };
    let mut next = 0;
    let mut panels = Vec::new();
    for part in &report.layout.parts {
        let (piece, other, other_len) = if rows_side {
            (part.rows, part.cols, n)
        } else {
            (part.cols, part.rows, m)
        };
        assert_eq!(piece.start, next, "{label}: pieces tile C");
        assert!(piece.len > 0, "{label}: empty piece");
        assert_eq!(piece.start % unit, 0, "{label}: piece not on a panel");
        assert_eq!((other.start, other.len), (0, other_len), "{label}");
        panels.push(piece.len.div_ceil(unit));
        next = piece.start + piece.len;
        let spec = spec_part_of(&out.case, out.kc, part.rows, part.cols, &tuning);
        assert_eq!(Some(*part), spec, "{label}: part against the spec");
    }
    assert_eq!(next, len, "{label}: pieces cover C");
    let (lo, hi) = (
        panels.iter().min().copied().unwrap_or(0),
        panels.iter().max().copied().unwrap_or(0),
    );
    assert!(hi - lo <= 1, "{label}: unbalanced pieces {panels:?}");
}

/// Whether a stored `rows × cols` operand in `store` can be normalised:
/// everything but a fully strided view, which still can when it is a
/// single row or column.
fn normalizable(store: Store, rows: usize, cols: usize) -> bool {
    store != Store::Strided || rows == 1 || cols == 1
}

/// The decision of the seams file (section 2) and of the policy,
/// written out from the spec rather than from `classify` / `kc_accepted`,
/// so a mutated rule cannot pass its own test: whether `case` with the
/// fitted KC `kc` is accepted, with the tuning in force on this thread. The
/// small class (`m · n · k <= 32³` and tall and wide enough) is accepted
/// only when the tuning's small class is on (it is off by default). A
/// sequential blocked call whose buffers do not fit the packing-cache limit
/// declines (the check takes C as one part, which is what a sequential
/// call runs; the tests that lower the limit are sequential).
pub(super) fn spec_accepts<T: TestElem>(case: &Case<T>, kc: usize) -> bool {
    let (m, n, k) = (case.m, case.n, case.k);
    if m == 0 || n == 0 || k == 0 || case.alpha.to_f64() == 0.0 {
        return false;
    }
    if m == 1 || n == 1 {
        return false;
    }
    let tuning = Tuning::of::<T>();
    let work = m.saturating_mul(n).saturating_mul(k);
    let small_ok = (m >= 24 && n >= 8 && k >= 8) || (m >= 16 && n >= 16 && k >= 64);
    if work <= 32 * 32 * 32 && !(small_ok && tuning.small_class) {
        return false;
    }
    if kc < 32 && kc < k {
        return false;
    }
    let (ar, ac) = stored_shape(case.trans_a, m, k);
    let (br, bc) = stored_shape(case.trans_b, k, n);
    if !(normalizable(case.a_store, ar, ac) && normalizable(case.b_store, br, bc)) {
        return false;
    }
    case.par != Par::Seq
        || spec_part_of(
            case,
            kc,
            Span { start: 0, len: m },
            Span { start: 0, len: n },
            &tuning,
        )
        .is_some()
}

/// Runs `case` and asserts the spec's verdict: accepted and equal to NEON
/// with every block on SME2, or declined with C untouched.
#[track_caller]
pub(super) fn check_spec<T: TestElem>(case: &Case<T>, label: &str) -> Outcome<T> {
    let out = run_case(case);
    let label = format!("{label} {case:?} kc {}", out.kc);
    if spec_accepts(case, out.kc) {
        assert_matches_neon(&out, &label);
    } else {
        assert_declined(&out, &label);
    }
    out
}

/// Asserts that `out` declined and left C bitwise untouched.
#[track_caller]
pub(super) fn assert_declined<T: TestElem>(out: &Outcome<T>, label: &str) {
    assert!(!out.accepted, "{label}: accepted");
    assert_eq!(out.decided, 0, "{label}: decisions");
    assert!(out.report.is_none(), "{label}: a part ran");
    for (index, (got, before)) in out.got.data.iter().zip(&out.before.data).enumerate() {
        assert_eq!(
            got.bits(),
            before.bits(),
            "{label}: element {index} of C changed"
        );
    }
}

/// The alpha/beta pairs of the four epilogue modes (all three branches of
/// M3), and whether C starts as NaN: M0 and M2 must not read it.
pub(super) const SCALARS: [(f64, f64, bool); 6] = [
    (1.0, 0.0, true),    // M0
    (1.0, 1.0, false),   // M1
    (0.5, -0.0, true),   // M2 (beta = -0 counts as zero)
    (-1.0, -0.5, false), // M3
    (1.0, -0.5, false),  // M3, alpha == 1
    (0.5, 1.0, false),   // M3, beta == 1
];

/// The three transposes.
pub(super) const TRANSPOSES: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
