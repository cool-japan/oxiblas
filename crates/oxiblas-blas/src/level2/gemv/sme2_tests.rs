//! Tests of the SME2 GEMV (`sme2.rs`), part 1: the shared harness, the
//! tests that run everywhere (the arithmetic of the definition against the
//! NEON kernels, the Rust finish, the decision as pure functions, the
//! declines, the self-check with plain-Rust stand-in kernels) and the
//! decline tests of the backend. `sme2_tests_templates.rs` holds the
//! structure tests of the registered templates, and `sme2_tests_grid.rs`
//! the tests that execute SME2 code over the size grid, special values,
//! views, parts and threads.
//!
//! - **Gates.** A test that calls the kernels directly starts with
//!   `test_support::hardware_or_skip`; a test that goes through
//!   `try_gemv`'s decision and expects it to accept starts with
//!   `test_support::backend_or_skip` (a build that caps SIMD below SME2
//!   turns the backend off by design). The `declines_without_backend` tests
//!   are ungated: where the backend is off they prove the declines, where
//!   it is on they print a `note:` line.
//! - **Proof that SME2 ran.** SME2 GEMV equals NEON bit for bit, so a
//!   result comparison alone passes when `try_gemv` silently runs NEON.
//!   Every accepted call is checked with exact kernel-entry counts on the
//!   thread that ran each part (the kernels count themselves; the per-part
//!   record of `sme2.rs` takes the delta on the part's own thread), and
//!   with the decision counter of the calling thread.
//! - **Borders.** Everything around A, `x` and `y` that must stay
//!   untouched holds a signalling NaN: a quiet NaN would survive an
//!   addition unchanged and hide an out-of-bounds read-modify-write.
//! - **Reads.** No result test can see a forbidden read (it feeds lanes
//!   that are never stored). The structure tests parse the registered
//!   templates and prove which predicate governs every load and store.

use core::fmt::Debug;

use oxiblas_core::parallel::Par;
use oxiblas_matrix::MatRef;

use super::GemvTrans;
use super::neon;
use super::sme2::{
    CHECK_NEGATIVE_ZERO_FROM, CHECK_NOTRANS, CHECK_TRANS, GemvArgs, GemvKernelFn, GemvRun,
    MIN_ROWS, MIN_WORK, PAR_MIN_BYTES, PartRecord, Piece, SmeGemv, TRANS_COLS_PER_CALL,
    TRANS_MIN_COLS, TRANS_SPLIT_COLS, crossover, part_count, reference_lanes, scalar_start,
    self_check_with, split, take_last_run, trans_finish, try_gemv, vector_tail,
};
use crate::level3::sme2::args::{SmeElement, same_result};
use crate::level3::sme2::detect::set_pretend_not_ready;
use crate::level3::sme2::test_support::{
    Need, backend_or_skip, backend_present, decisions, force_off, hardware_present,
    kernel_entries_here, note_declined, sme_state,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// The element types of the tests.
pub(super) trait GvElem: SmeGemv + Debug + PartialOrd {
    /// The hardware the type needs.
    const NEED: Need;
    /// The value of an `f64` literal (rounded for `f32`).
    fn of(value: f64) -> Self;
    /// The border sentinel: a signalling NaN with a recognisable payload.
    fn snan() -> Self;
    /// A quiet NaN.
    fn qnan() -> Self;
    /// `+∞`.
    fn inf() -> Self;
    /// The smallest positive subnormal.
    fn tiny() -> Self;
    /// The largest finite value.
    fn huge() -> Self;
}

impl GvElem for f32 {
    const NEED: Need = Need::F32;
    fn of(value: f64) -> Self {
        value as f32
    }
    fn snan() -> Self {
        f32::from_bits(0x7fa0_5a5a)
    }
    fn qnan() -> Self {
        f32::from_bits(0x7fc0_1234)
    }
    fn inf() -> Self {
        f32::INFINITY
    }
    fn tiny() -> Self {
        f32::from_bits(1)
    }
    fn huge() -> Self {
        f32::MAX
    }
}

impl GvElem for f64 {
    const NEED: Need = Need::F64;
    fn of(value: f64) -> Self {
        value
    }
    fn snan() -> Self {
        f64::from_bits(0x7ff4_0000_0000_5a5a)
    }
    fn qnan() -> Self {
        f64::from_bits(0x7ff8_0000_0000_1234)
    }
    fn inf() -> Self {
        f64::INFINITY
    }
    fn tiny() -> Self {
        f64::from_bits(1)
    }
    fn huge() -> Self {
        f64::MAX
    }
}

/// Deterministic xorshift generator.
pub(super) struct Rng(u64);

impl Rng {
    /// A generator for `seed`.
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

    /// Uniform in `0..bound` (`bound >= 1`).
    pub(super) fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// A full-mantissa value of mixed sign and magnitude (`2^-6 .. 2^6`),
    /// one in 32 an exact zero.
    pub(super) fn plain<T: GvElem>(&mut self) -> T {
        let bits = self.next_u64();
        if bits.is_multiple_of(32) {
            return T::ZERO;
        }
        let unit = (bits >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        let scale = f64::from_bits((1017 + (bits >> 3) % 13) << 52);
        T::of(unit * scale)
    }

    /// A plain value most of the time, otherwise a special one: NaN, ±∞,
    /// a subnormal, ±0, the largest finite value.
    pub(super) fn special<T: GvElem>(&mut self) -> T {
        match self.below(40) {
            0 => T::qnan(),
            1 => T::inf(),
            2 => T::ZERO * T::of(-1.0),
            3 => T::ZERO,
            4 => T::tiny(),
            5 => T::tiny() * T::of(-37.0),
            6 => T::huge(),
            7 => T::of(-1.0) * T::inf(),
            _ => self.plain(),
        }
    }

    /// A plain value nearly always: one in 400 a NaN with a random payload
    /// (quiet or signalling, either sign), one in 200 an infinity, then
    /// subnormals, signed zeros and the largest finite value. Most columns
    /// of a few hundred rows stay finite, so a wrong lane or row shows as a
    /// number, not only as a NaN position.
    pub(super) fn sparse<T: GvElem>(&mut self) -> T {
        match self.below(400) {
            0 => {
                let bits = self.next_u64();
                if core::mem::size_of::<T>() == 4 {
                    let payload = (bits & 0x003f_ffff) | 1;
                    let quiet = (bits >> 40) & 1;
                    T::from_bits64(
                        (((bits >> 41) & 1) << 31) | 0x7f80_0000 | (quiet << 22) | payload,
                    )
                } else {
                    let payload = (bits & 0x0007_ffff_ffff_ffff) | 1;
                    let quiet = (bits >> 60) & 1;
                    T::from_bits64(
                        (((bits >> 61) & 1) << 63)
                            | 0x7ff0_0000_0000_0000
                            | (quiet << 51)
                            | payload,
                    )
                }
            }
            1 => T::inf(),
            2 => T::of(-1.0) * T::inf(),
            3..=10 => T::tiny() * T::of((self.below(64) as f64) - 32.0),
            11..=16 => T::ZERO * T::of(if self.below(2) == 0 { 1.0 } else { -1.0 }),
            17 => T::huge(),
            _ => self.plain(),
        }
    }
}

/// How the values of A and `x` are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Data {
    /// [`Rng::plain`].
    Plain,
    /// [`Rng::special`].
    Special,
    /// [`Rng::sparse`].
    Sparse,
}

/// What `y` holds before the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum YFill {
    /// Plain values.
    Plain,
    /// A quiet NaN everywhere.
    Nan,
    /// `-0` everywhere.
    NegZero,
}

/// Where the operands sit in their buffers (all borders hold the
/// signalling NaN of [`GvElem::snan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Layout {
    /// `lda - m`: rows below the matrix in every column.
    pub(super) lda_pad: usize,
    /// Elements before `A(0, 0)` and after the last column.
    pub(super) a_border: usize,
    /// Elements before and after `x`.
    pub(super) x_border: usize,
    /// Elements before and after `y`.
    pub(super) y_border: usize,
}

impl Layout {
    /// Compact operands with a small guard around each.
    pub(super) const TIGHT: Self = Self {
        lda_pad: 0,
        a_border: 4,
        x_border: 4,
        y_border: 4,
    };
}

/// One GEMV call: A, `x` and `y` in guarded buffers.
#[derive(Debug, Clone)]
pub(super) struct Problem<T> {
    /// The transpose.
    pub(super) trans: GemvTrans,
    /// Rows of the stored A.
    pub(super) m: usize,
    /// Columns of the stored A.
    pub(super) n: usize,
    /// The column distance of A.
    pub(super) lda: usize,
    /// alpha.
    pub(super) alpha: T,
    /// The buffer of A; `A(0, 0)` at `a_off`.
    pub(super) a: Vec<T>,
    /// Offset of `A(0, 0)`.
    pub(super) a_off: usize,
    /// The buffer of `x`.
    pub(super) x: Vec<T>,
    /// Offset of `x[0]`.
    pub(super) x_off: usize,
    /// The buffer of `y`.
    pub(super) y: Vec<T>,
    /// Offset of `y[0]`.
    pub(super) y_off: usize,
}

impl<T: GvElem> Problem<T> {
    /// A problem of the stored shape `m × n`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        trans: GemvTrans,
        m: usize,
        n: usize,
        alpha: T,
        layout: Layout,
        data: Data,
        fill: YFill,
        rng: &mut Rng,
    ) -> Self {
        let lda = m + layout.lda_pad;
        let a_off = layout.a_border;
        let mut a = vec![T::snan(); a_off + lda * n + layout.a_border];
        for j in 0..n {
            for i in 0..m {
                a[a_off + i + j * lda] = match data {
                    Data::Plain => rng.plain(),
                    Data::Special => rng.special(),
                    Data::Sparse => rng.sparse(),
                };
            }
        }
        let transposed = trans != GemvTrans::NoTrans;
        let (x_len, y_len) = if transposed { (m, n) } else { (n, m) };
        let x_off = layout.x_border;
        let mut x = vec![T::snan(); x_len + 2 * x_off];
        for value in &mut x[x_off..x_off + x_len] {
            *value = match data {
                Data::Plain => rng.plain(),
                Data::Special => rng.special(),
                Data::Sparse => rng.sparse(),
            };
        }
        let y_off = layout.y_border;
        let mut y = vec![T::snan(); y_len + 2 * y_off];
        for value in &mut y[y_off..y_off + y_len] {
            *value = match fill {
                YFill::Plain => rng.plain(),
                YFill::Nan => T::qnan(),
                YFill::NegZero => T::ZERO * T::of(-1.0),
            };
        }
        Self {
            trans,
            m,
            n,
            lda,
            alpha,
            a,
            a_off,
            x,
            x_off,
            y,
            y_off,
        }
    }

    /// Whether the transpose is `Trans` / `ConjTrans`.
    pub(super) fn transposed(&self) -> bool {
        self.trans != GemvTrans::NoTrans
    }

    /// Lengths of `x` and `y`.
    pub(super) fn lens(&self) -> (usize, usize) {
        if self.transposed() {
            (self.m, self.n)
        } else {
            (self.n, self.m)
        }
    }

    /// The view of A.
    pub(super) fn view(&self) -> MatRef<'_, T> {
        MatRef::from_strided(&self.a[self.a_off..], self.m, self.n, self.lda, 1).expect("A view")
    }

    /// `x`.
    pub(super) fn xs(&self) -> &[T] {
        &self.x[self.x_off..self.x_off + self.lens().0]
    }

    /// The NEON result: the whole `y` buffer after the NEON kernel.
    pub(super) fn neon(&self) -> Vec<T> {
        let mut y = self.y.clone();
        let y_len = self.lens().1;
        let window = &mut y[self.y_off..self.y_off + y_len];
        if self.transposed() {
            T::neon_trans(self.alpha, self.view(), self.xs(), window);
        } else {
            T::neon_notrans(self.alpha, self.view(), self.xs(), window);
        }
        y
    }

    /// Kernel calls of an SME2 part of `len` rows or columns: one per
    /// `NoTrans` part, one per started 1024 columns for `Trans` (the spec,
    /// not the constant of `sme2.rs`).
    pub(super) fn calls_of(&self, len: usize) -> u64 {
        if self.transposed() {
            len.div_ceil(1024) as u64
        } else {
            1
        }
    }
}

/// What one `try_gemv` call did.
#[derive(Debug)]
pub(super) struct Outcome<T> {
    /// The return value.
    pub(super) accepted: bool,
    /// Decisions counted on the calling thread.
    pub(super) decided: u64,
    /// Kernel entries counted on the calling thread.
    pub(super) entries: u64,
    /// The run report (`None` when declined).
    pub(super) run: Option<GemvRun>,
    /// The `y` buffer after the call.
    pub(super) got: Vec<T>,
    /// The `y` buffer after the NEON kernel.
    pub(super) want: Vec<T>,
    /// The `y` buffer before the call.
    pub(super) before: Vec<T>,
    /// A and `x` came back bit for bit (borders included).
    pub(super) inputs_intact: bool,
    /// `sme_state()` after the call.
    pub(super) state: Option<(u64, u64)>,
    /// The thread that made the call.
    pub(super) caller: std::thread::ThreadId,
}

/// Bitwise equality of two buffers.
pub(super) fn identical<T: SmeElement>(got: &[T], want: &[T]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g.bits() == w.bits())
}

/// Runs `p` through `try_gemv` with `par` and records everything.
pub(super) fn run_problem<T: GvElem>(p: &Problem<T>, par: Par) -> Outcome<T> {
    let want = p.neon();
    let (a_before, x_before) = (p.a.clone(), p.x.clone());
    let before = p.y.clone();
    let mut got = p.y.clone();
    let y_len = p.lens().1;
    let _ = take_last_run();
    let decided = decisions();
    let entries = kernel_entries_here();
    let accepted = try_gemv(
        p.trans,
        p.alpha,
        p.view(),
        p.xs(),
        &mut got[p.y_off..p.y_off + y_len],
        par,
    );
    let entries = kernel_entries_here() - entries;
    let decided = decisions() - decided;
    let run = take_last_run();
    Outcome {
        accepted,
        decided,
        entries,
        run,
        got,
        want,
        before,
        inputs_intact: identical(&p.a, &a_before) && identical(&p.x, &x_before),
        state: sme_state(),
        caller: std::thread::current().id(),
    }
}

/// The accepted call gave the NEON bits in `y`, left every border element
/// and A and `x` bitwise untouched, took one decision and left the thread
/// out of streaming mode with no ZA state.
#[track_caller]
pub(super) fn assert_same_bits<T: GvElem>(p: &Problem<T>, out: &Outcome<T>, label: &str) {
    assert!(out.accepted, "{label}: declined");
    assert_eq!(out.decided, 1, "{label}: decisions");
    assert!(out.run.is_some(), "{label}: no run report");
    assert!(out.inputs_intact, "{label}: A or x changed");
    let y_len = p.lens().1;
    for (index, ((got, want), before)) in out.got.iter().zip(&out.want).zip(&out.before).enumerate()
    {
        if (p.y_off..p.y_off + y_len).contains(&index) {
            assert!(
                same_result(*got, *want),
                "{label}: y[{}] = {got:?} ({:#x}), NEON {want:?} ({:#x})",
                index - p.y_off,
                got.bits(),
                want.bits()
            );
        } else {
            assert_eq!(
                got.bits(),
                before.bits(),
                "{label}: border element {index} of y changed"
            );
        }
    }
    assert_eq!(out.state, Some((0, 0)), "{label}: SME state after the call");
}

/// The call was declined without touching anything: `false`, no decision,
/// no run report, no kernel entry on this thread, `y` bitwise untouched.
#[track_caller]
pub(super) fn assert_declined<T: GvElem>(out: &Outcome<T>, label: &str) {
    assert!(!out.accepted, "{label}: accepted");
    assert_eq!(out.decided, 0, "{label}: decisions");
    assert!(out.run.is_none(), "{label}: a part ran");
    assert_eq!(out.entries, 0, "{label}: a kernel ran");
    assert!(identical(&out.got, &out.before), "{label}: y changed");
    assert!(out.inputs_intact, "{label}: A or x changed");
}

/// The pieces of an axis of `len` cut into at most `parts` runs of whole
/// `unit`s, as the spec defines them (independently of `split`): the count
/// is `parts` clamped to `1..=ceil(len / unit)`, the runs are non-empty,
/// in order, tile `0..len`, start at multiples of `unit`, and their unit
/// counts differ by at most one.
#[track_caller]
pub(super) fn assert_spec_pieces(
    pieces: &[Piece],
    len: usize,
    parts: usize,
    unit: usize,
    label: &str,
) {
    let units = len.div_ceil(unit);
    assert_eq!(pieces.len(), parts.clamp(1, units), "{label}: part count");
    let mut next = 0;
    let mut counts = Vec::new();
    for piece in pieces {
        assert_eq!(piece.start, next, "{label}: pieces do not tile: {pieces:?}");
        assert!(piece.len > 0, "{label}: empty piece: {pieces:?}");
        assert_eq!(piece.start % unit, 0, "{label}: piece start: {pieces:?}");
        counts.push(piece.len.div_ceil(unit));
        next += piece.len;
    }
    assert_eq!(next, len, "{label}: pieces cover {next} of {len}");
    let (low, high) = (
        counts.iter().min().copied().unwrap_or(0),
        counts.iter().max().copied().unwrap_or(0),
    );
    assert!(high - low <= 1, "{label}: uneven pieces: {pieces:?}");
}

/// The run report of an accepted call, part by part: `parts` parts tiling
/// the split axis per the spec; part `index` ran on SME2 exactly when
/// `sme2(index, record)` says so, with the spec's number of kernel entries
/// on its own thread (none for a NEON part); and the calling thread's
/// kernel-entry delta is exactly the sum over the parts that ran on it.
#[track_caller]
pub(super) fn assert_parts<T: GvElem>(
    p: &Problem<T>,
    out: &Outcome<T>,
    parts: usize,
    sme2: impl Fn(usize, &PartRecord) -> bool,
    label: &str,
) {
    let run = out.run.as_ref().expect("run report");
    assert_eq!(run.transposed, p.transposed(), "{label}: transpose");
    let pieces: Vec<Piece> = run.parts.iter().map(|part| part.piece).collect();
    let (len, unit) = if p.transposed() {
        (p.n, 4)
    } else {
        (p.m, T::LANES)
    };
    assert_spec_pieces(&pieces, len, parts, unit, label);
    let me = out.caller;
    let mut here = 0;
    for (index, part) in run.parts.iter().enumerate() {
        let on = sme2(index, part);
        assert_eq!(part.sme2, on, "{label}: part {index} SME2 ({part:?})");
        let want = if on { p.calls_of(part.piece.len) } else { 0 };
        assert_eq!(
            part.kernel_calls, want,
            "{label}: part {index}: kernel entries on its thread ({part:?})"
        );
        assert!(!part.timed_out, "{label}: part {index} timed out");
        if part.thread == me {
            here += part.kernel_calls;
        }
    }
    assert_eq!(out.entries, here, "{label}: kernel entries on this thread");
}

/// [`assert_same_bits`] for a sequential call: one part, on this thread, on
/// SME2, with exactly the spec's kernel calls counted on this thread.
#[track_caller]
pub(super) fn assert_sequential<T: GvElem>(p: &Problem<T>, out: &Outcome<T>, label: &str) {
    assert_same_bits(p, out, label);
    assert_parts(p, out, 1, |_, _| true, label);
    let run = out.run.as_ref().expect("run report");
    assert_eq!(run.parts[0].thread, out.caller, "{label}: thread");
    let len = if p.transposed() { p.n } else { p.m };
    assert_eq!(out.entries, p.calls_of(len), "{label}: kernel calls");
}

// ---------------------------------------------------------------------------
// The definition against the NEON kernels (plain Rust, run everywhere)
// ---------------------------------------------------------------------------

/// The lane accumulation of the kernel for one column, written from the
/// text of the definition (the module docs of `sme2.rs`), independently
/// of `sme2.rs`: `V` chains over the full `V`-row steps, then the two vector
/// tail steps with the lane placement of the NEON kernel. Returns the lanes
/// and the first row of the scalar tail.
fn emulate_lanes<T: GvElem>(col: &[T], x: &[T]) -> (Vec<T>, usize) {
    let v = T::LANES;
    let m = col.len();
    let mut lanes = vec![T::ZERO; v];
    let body = m - m % v;
    let mut i0 = 0;
    while i0 < body {
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane = SmeElement::fused(col[i0 + l], x[i0 + l], *lane);
        }
        i0 += v;
    }
    let mut i = body;
    let (first, second) = (v / 2, v / 4);
    if m - i >= first {
        for (l, lane) in lanes[..first].iter_mut().enumerate() {
            *lane = SmeElement::fused(col[i + l], x[i + l], *lane);
        }
        i += first;
    }
    if m - i >= second {
        for (l, lane) in lanes[first..first + second].iter_mut().enumerate() {
            *lane = SmeElement::fused(col[i + l], x[i + l], *lane);
        }
        i += second;
    }
    (lanes, i)
}

/// `y += alpha · Aᵀ · x` through [`emulate_lanes`] and the real finish of
/// `sme2.rs` ([`trans_finish`], then `y + alpha · s`).
fn emulate_trans<T: GvElem>(alpha: T, a: MatRef<'_, T>, x: &[T], y: &mut [T]) {
    for (j, yv) in y.iter_mut().enumerate() {
        let col = a.col_as_slice(j).expect("column");
        let (lanes, start) = emulate_lanes(col, x);
        assert_eq!(
            start,
            scalar_start(col.len(), T::LANES),
            "scalar start, m = {}",
            col.len()
        );
        assert!(
            identical(&lanes, &reference_lanes(col, x, T::LANES)),
            "reference_lanes differs from the definition, m = {}",
            col.len()
        );
        *yv = *yv + alpha * trans_finish(&lanes, col, x);
    }
}

/// `y += alpha · A · x` in the order of the definition: per row,
/// `fma(A[i, j], alpha · x[j], y[i])` for `j` ascending.
fn emulate_notrans<T: GvElem>(alpha: T, a: MatRef<'_, T>, x: &[T], y: &mut [T]) {
    for j in 0..a.ncols() {
        let t = alpha * x[j];
        let col = a.col_as_slice(j).expect("column");
        for (yv, av) in y.iter_mut().zip(col) {
            *yv = SmeElement::fused(*av, t, *yv);
        }
    }
}

/// The emulations against the NEON kernels over a grid of `m` (every
/// residue of `m % 16`, small and large) and `n`, with plain and special
/// data, several alphas and `y` fills.
fn emulation_grid<T: GvElem>(label: &str) {
    let mut rng = Rng::new(17);
    let ms: Vec<usize> = (16..=48).chain(64..=81).chain(1000..=1017).collect();
    let alphas = [1.0, -1.0, 0.5, 1.0 / 3.0];
    let fills = [YFill::Plain, YFill::Nan, YFill::NegZero];
    let mut cases = 0;
    for (index, &m) in ms.iter().enumerate() {
        for n in [1, 2, 3, 4, 5, 7, 17] {
            for data in [Data::Plain, Data::Special, Data::Sparse] {
                for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
                    let alpha = T::of(alphas[(index + n) % 4]);
                    let fill = fills[(index + n + cases) % 3];
                    let p = Problem::new(trans, m, n, alpha, Layout::TIGHT, data, fill, &mut rng);
                    let want = p.neon();
                    let mut got = p.y.clone();
                    let y_len = p.lens().1;
                    let window = &mut got[p.y_off..p.y_off + y_len];
                    if p.transposed() {
                        emulate_trans(alpha, p.view(), p.xs(), window);
                    } else {
                        emulate_notrans(alpha, p.view(), p.xs(), window);
                    }
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            same_result(*g, *w),
                            "{label}: {trans:?} {m}x{n} {data:?} {fill:?}: element {i}: {g:?} vs NEON {w:?}"
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 1000, "{label}: {cases} cases");
}

#[test]
fn test_sme2_gemv_trans_finish_matches_neon_kernels() {
    emulation_grid::<f32>("test_sme2_gemv_trans_finish_matches_neon_kernels f32");
    emulation_grid::<f64>("test_sme2_gemv_trans_finish_matches_neon_kernels f64");
}

#[test]
fn test_sme2_gemv_vector_tail_cases() {
    // f32: lanes 0..8 iff r >= 8, lanes 8..12 for the next four rows,
    // based 8 rows below when the first step did not run.
    for m in 16..200 {
        let (r, body) = (m % 16, m - m % 16);
        let want = match r {
            12..=15 => Some((body, 0, 12)),
            8..=11 => Some((body, 0, 8)),
            4..=7 => Some((body - 8, 8, 12)),
            _ => None,
        };
        assert_eq!(vector_tail(m, 16), want, "f32 m = {m}");
        assert_eq!(scalar_start(m, 16), m - m % 4, "f32 m = {m}");
        let (r, body) = (m % 8, m - m % 8);
        let want = match r {
            6 | 7 => Some((body, 0, 6)),
            4 | 5 => Some((body, 0, 4)),
            2 | 3 => Some((body - 4, 4, 6)),
            _ => None,
        };
        assert_eq!(vector_tail(m, 8), want, "f64 m = {m}");
        assert_eq!(scalar_start(m, 8), m - m % 2, "f64 m = {m}");
    }
    // Rows the tail step reads are inside the column and above body.
    for m in 16..200 {
        for lanes in [8, 16] {
            if let Some((base, lo, hi)) = vector_tail(m, lanes) {
                assert!(base + lo >= m - m % lanes && base + hi <= m, "m = {m}");
            }
        }
    }
}

#[test]
fn test_sme2_gemv_reduce_order() {
    // Values whose sums depend on the order: the documented pairing.
    let l32: Vec<f32> = (0..16)
        .map(|l| 1.0 + (l as f32) * 1.0e-7 * if l % 3 == 0 { 1e7 } else { 1.0 })
        .collect();
    let v = |l: usize| (l32[l] + l32[4 + l]) + (l32[8 + l] + l32[12 + l]);
    assert_eq!(
        f32::reduce(&l32).to_bits(),
        ((v(0) + v(1)) + (v(2) + v(3))).to_bits()
    );
    let l64: Vec<f64> = (0..8)
        .map(|l| 1.0e16 * f64::from(l % 2) + f64::from(l))
        .collect();
    let w = |l: usize| (l64[l] + l64[2 + l]) + (l64[4 + l] + l64[6 + l]);
    assert_eq!(f64::reduce(&l64).to_bits(), (w(0) + w(1)).to_bits());
    // A NEON kernel on one column built so that only the documented order
    // gives its result.
    let m = 16;
    let col: Vec<f32> = vec![
        1.0e8, 1.0, -1.0e8, 1.0, 3.0, 5.0, 7.0, 9.0, 0.5, 0.25, 0.125, 1.0, 2.0, -2.0, 4.0, 1.0e-3,
    ];
    let x = vec![1.0f32; m];
    let a = MatRef::from_column_major(&col, m, 1).expect("view");
    let mut want = [0.0f32];
    neon::trans_f32(1.0, a, &x, &mut want);
    let (lanes, _) = emulate_lanes(&col, &x);
    assert_eq!(trans_finish(&lanes, &col, &x).to_bits(), want[0].to_bits());
}

// ---------------------------------------------------------------------------
// The decision as pure functions (run everywhere)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_gemv_crossover() {
    assert_eq!((MIN_ROWS, MIN_WORK, TRANS_MIN_COLS), (64, 128 * 128, 16));
    assert!(crossover(false, 64, 256));
    assert!(!crossover(false, 64, 255));
    assert!(!crossover(false, 63, 100_000));
    assert!(crossover(false, 16384, 1));
    assert!(!crossover(false, 16383, 1));
    assert!(crossover(true, 1024, 16));
    assert!(!crossover(true, 100_000, 15));
    assert!(!crossover(true, 63, 1000));
    assert!(crossover(true, 64, 256));
    assert!(!crossover(true, 64, 255));
    assert!(!crossover(false, 0, 0) && !crossover(true, 0, 0));
    // Saturating work.
    assert!(crossover(false, usize::MAX, 2) && crossover(true, usize::MAX, 16));
}

#[test]
fn test_sme2_gemv_part_count() {
    assert_eq!(PAR_MIN_BYTES, 8 << 20);
    assert_eq!(part_count(PAR_MIN_BYTES, Par::Seq, 4, PAR_MIN_BYTES), 1);
    assert_eq!(part_count(usize::MAX, Par::Seq, 4, 0), 1);
    #[cfg(feature = "parallel")]
    {
        assert_eq!(
            part_count(PAR_MIN_BYTES - 1, Par::RayonWith(8), 4, PAR_MIN_BYTES),
            1
        );
        assert_eq!(
            part_count(PAR_MIN_BYTES, Par::RayonWith(8), 1, PAR_MIN_BYTES),
            1
        );
        assert_eq!(
            part_count(PAR_MIN_BYTES, Par::RayonWith(8), 4, PAR_MIN_BYTES),
            4
        );
        assert_eq!(
            part_count(PAR_MIN_BYTES, Par::RayonWith(3), 4, PAR_MIN_BYTES),
            3
        );
        assert_eq!(
            part_count(PAR_MIN_BYTES, Par::RayonWith(1), 4, PAR_MIN_BYTES),
            1
        );
        assert_eq!(part_count(0, Par::RayonWith(2), 3, 0), 2);
        assert_eq!(
            part_count(PAR_MIN_BYTES, Par::Rayon, 2, PAR_MIN_BYTES),
            2.min(Par::Rayon.num_threads()).max(1)
        );
    }
}

#[test]
fn test_sme2_gemv_split() {
    assert_eq!(TRANS_SPLIT_COLS, 4);
    assert!(split(0, 3, 16).is_empty() && split(10, 3, 0).is_empty());
    for len in [1, 15, 16, 17, 63, 64, 65, 100, 1000, 1023, 1025, 4097] {
        for unit in [4, 8, 16] {
            for parts in [0, 1, 2, 3, 5, 7, 64, 10_000] {
                let label = format!("split({len}, {parts}, {unit})");
                assert_spec_pieces(&split(len, parts, unit), len, parts, unit, &label);
            }
        }
    }
    assert_eq!(
        split(100, 3, 16),
        vec![
            Piece { start: 0, len: 32 },
            Piece { start: 32, len: 32 },
            Piece { start: 64, len: 36 }
        ]
    );
}

// ---------------------------------------------------------------------------
// Declines
// ---------------------------------------------------------------------------

/// The shapes of the decline tests, accepted when the backend is on: both
/// transposes, below and above the 8 MiB split.
fn decline_shapes() -> [(GemvTrans, usize, usize); 4] {
    [
        (GemvTrans::NoTrans, 130, 140),
        (GemvTrans::Trans, 130, 140),
        (GemvTrans::NoTrans, 2048, 1100),
        (GemvTrans::ConjTrans, 1100, 2048),
    ]
}

/// Where the SME2 backend is off for `T` (no hardware, or a build that caps
/// SIMD below SME2), `try_gemv` declines calls it accepts when the backend
/// is on: NoTrans and Trans, below and above the parallel split, `y`
/// pre-filled with a signalling NaN and with plain values; each returns
/// `false`, takes no decision, runs no kernel on this thread and leaves `y`
/// bitwise untouched. Then the test writes the marker `sme2-gate: declined
/// <test>`. Where the backend is on it prints one `note:` line. It never
/// panics because of `OXIBLAS_TEST_REQUIRE_SME2` or
/// `OXIBLAS_TEST_EXPECT_NO_SME2`.
fn declines_without_backend<T: GvElem>(test: &str) {
    if backend_present(T::NEED) {
        println!(
            "note: {test}: the SME2 backend is on for {:?}; this test checks the declines where it is off",
            T::NEED
        );
        return;
    }
    let seen = if hardware_present(T::NEED) {
        "capped build"
    } else {
        "no hardware"
    };
    let _units = crate::level3::sme2::detect::override_units(2);
    let mut rng = Rng::new(41);
    let mut calls = 0;
    for (trans, m, n) in decline_shapes() {
        let transposed = trans != GemvTrans::NoTrans;
        assert!(crossover(transposed, m, n), "{test}: {trans:?} {m}x{n}");
        for signalling in [true, false] {
            let mut p = Problem::new(
                trans,
                m,
                n,
                T::of(0.75),
                Layout::TIGHT,
                Data::Plain,
                YFill::Plain,
                &mut rng,
            );
            if signalling {
                p.y.fill(T::snan());
            }
            #[cfg(feature = "parallel")]
            let pars = [Par::Seq, Par::RayonWith(2)];
            #[cfg(not(feature = "parallel"))]
            let pars = [Par::Seq];
            for par in pars {
                let label = format!("{test} ({seen}): {trans:?} {m}x{n} {par:?}");
                let out = run_problem(&p, par);
                assert_declined(&out, &label);
                calls += 1;
            }
        }
    }
    note_declined(test);
    println!("{test}: try_gemv declined all {calls} calls ({seen})");
}

#[test]
fn test_sme2_gemv_f32_declines_without_backend() {
    declines_without_backend::<f32>("test_sme2_gemv_f32_declines_without_backend");
}

#[test]
fn test_sme2_gemv_f64_declines_without_backend() {
    declines_without_backend::<f64>("test_sme2_gemv_f64_declines_without_backend");
}

/// Makes the calling thread pretend not to be ready while it lives.
pub(super) struct PretendHere {
    /// The setting to restore.
    previous: bool,
    /// Keeps the guard on its thread.
    _not_send: core::marker::PhantomData<*const ()>,
}

impl PretendHere {
    /// Starts pretending on the calling thread.
    pub(super) fn new() -> Self {
        Self {
            previous: set_pretend_not_ready(true),
            _not_send: core::marker::PhantomData,
        }
    }
}

impl Drop for PretendHere {
    fn drop(&mut self) {
        set_pretend_not_ready(self.previous);
    }
}

/// Every decline reason of the decision, each after a control call that
/// the backend accepts (so the decline cannot come from anything else):
/// `col_stride != 1`, `x` or `y` of the wrong length, `alpha == ±0`, below
/// the crossover on each of its three conditions, SME2 forced off, and a
/// calling thread that is not ready. Each declines without touching
/// anything.
fn declines_each_reason<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(5);
    for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
        let p = Problem::new(
            trans,
            130,
            140,
            T::of(0.5),
            Layout::TIGHT,
            Data::Plain,
            YFill::Plain,
            &mut rng,
        );
        let control = run_problem(&p, Par::Seq);
        assert_sequential(&p, &control, &format!("{test}: control {trans:?}"));
        // alpha == 0 and -0.
        for alpha in [T::ZERO, T::ZERO * T::of(-1.0)] {
            let mut q = p.clone();
            q.alpha = alpha;
            assert_declined(
                &run_problem(&q, Par::Seq),
                &format!("{test}: {trans:?} alpha {alpha:?}"),
            );
        }
        // Forced off on this thread.
        {
            let _off = force_off();
            assert_declined(
                &run_problem(&p, Par::Seq),
                &format!("{test}: {trans:?} forced off"),
            );
        }
        // The calling thread is not ready.
        {
            let _pretend = PretendHere::new();
            assert_declined(
                &run_problem(&p, Par::Seq),
                &format!("{test}: {trans:?} not ready"),
            );
        }
        // A view whose rows are not adjacent (col_stride 2).
        let (x_len, y_len) = p.lens();
        let wide: Vec<T> = (0..2 * 130 * 140).map(|_| rng.plain()).collect();
        let strided = MatRef::from_strided(&wide, 130, 140, 260, 2).expect("strided view");
        let xs = &p.x[p.x_off..p.x_off + x_len];
        let mut y = p.y[p.y_off..p.y_off + y_len].to_vec();
        let before = y.clone();
        let (decided, entries) = (decisions(), kernel_entries_here());
        assert!(
            !try_gemv(trans, p.alpha, strided, xs, &mut y, Par::Seq),
            "{test}: col_stride 2"
        );
        // Wrong lengths of x and y.
        assert!(
            !try_gemv(trans, p.alpha, p.view(), &xs[1..], &mut y, Par::Seq),
            "{test}: short x"
        );
        assert!(
            !try_gemv(trans, p.alpha, p.view(), xs, &mut y[1..], Par::Seq),
            "{test}: short y"
        );
        assert_eq!(
            (decisions(), kernel_entries_here()),
            (decided, entries),
            "{test}: {trans:?}"
        );
        assert!(identical(&y, &before), "{test}: {trans:?}: y changed");
        assert!(take_last_run().is_none(), "{test}: a part ran");
    }
    // Below the crossover, next to accepted shapes.
    for (trans, m, n, accepted) in [
        (GemvTrans::NoTrans, 64, 256, true),
        (GemvTrans::NoTrans, 64, 255, false),
        (GemvTrans::NoTrans, 63, 300, false),
        (GemvTrans::Trans, 1024, 16, true),
        (GemvTrans::Trans, 1100, 15, false),
        (GemvTrans::Trans, 63, 300, false),
        (GemvTrans::Trans, 64, 255, false),
    ] {
        let p = Problem::new(
            trans,
            m,
            n,
            T::of(-1.0),
            Layout::TIGHT,
            Data::Plain,
            YFill::Plain,
            &mut rng,
        );
        let out = run_problem(&p, Par::Seq);
        let label = format!("{test}: {trans:?} {m}x{n}");
        if accepted {
            assert_sequential(&p, &out, &label);
        } else {
            assert_declined(&out, &label);
        }
    }
}

#[test]
fn test_sme2_gemv_f32_declines_each_reason() {
    declines_each_reason::<f32>("test_sme2_gemv_f32_declines_each_reason");
}

#[test]
fn test_sme2_gemv_f64_declines_each_reason() {
    declines_each_reason::<f64>("test_sme2_gemv_f64_declines_each_reason");
}

/// Types other than `f32` / `f64` are never taken (runs everywhere: such a
/// call executes no SME2 code even with the backend on).
#[test]
fn test_sme2_gemv_other_types_decline() {
    use num_complex::Complex64;
    let (m, n) = (130, 140);
    let a: Vec<Complex64> = (0..m * n).map(|i| Complex64::new(i as f64, 1.0)).collect();
    let view = MatRef::from_column_major(&a, m, n).expect("view");
    for trans in [GemvTrans::NoTrans, GemvTrans::Trans, GemvTrans::ConjTrans] {
        let (x_len, y_len) = if trans == GemvTrans::NoTrans {
            (n, m)
        } else {
            (m, n)
        };
        let x = vec![Complex64::new(0.5, -1.0); x_len];
        let mut y = vec![Complex64::new(f64::NAN, 2.0); y_len];
        let (decided, entries) = (decisions(), kernel_entries_here());
        assert!(!try_gemv(
            trans,
            Complex64::new(1.0, 0.0),
            view,
            &x,
            &mut y,
            Par::Seq
        ));
        assert_eq!((decisions(), kernel_entries_here()), (decided, entries));
        assert!(
            y.iter().all(|v| v.re.is_nan() && v.im == 2.0),
            "{trans:?}: y changed"
        );
    }
}

// ---------------------------------------------------------------------------
// The self-check with plain-Rust stand-in kernels (run everywhere)
// ---------------------------------------------------------------------------

/// The ways a stand-in kernel can be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// None: the definition.
    None,
    /// NoTrans: skips the last column.
    LastColumn,
    /// NoTrans: writes one element past `y[m - 1]`.
    StorePastEnd,
    /// NoTrans: multiplies the sum by alpha instead of each `x`.
    AlphaAfter,
    /// Trans: puts every tail row in the first lanes.
    TailLanes,
    /// Trans: writes the first partial of one column too many.
    ExtraColumn,
    /// Trans: adds the rows in descending order within a lane.
    Reversed,
    /// Trans: adds `+0 · +0` in the dead lanes of the vector tail (turns a
    /// `-0` partial into `+0`; the SME2 kernel once did this).
    PositiveZeroDeadLanes,
}

/// The NoTrans stand-in with `fault`.
///
/// # Safety
///
/// `args` as the GEMV kernels' contract describes.
unsafe fn standin_notrans<T: GvElem>(args: &GemvArgs<T>, fault: Fault) {
    let alpha = T::from_bits64(args.alpha_bits);
    let cols = if fault == Fault::LastColumn {
        args.n - 1
    } else {
        args.n
    };
    for i in 0..args.m {
        // SAFETY: inside `y[0..m]`, `A(i, j)` and `x[0..n]` by the contract.
        unsafe {
            let y = args.out.add(i);
            let mut acc = if fault == Fault::AlphaAfter {
                T::ZERO
            } else {
                *y
            };
            for j in 0..cols {
                let t = if fault == Fault::AlphaAfter {
                    *args.x.add(j)
                } else {
                    alpha * *args.x.add(j)
                };
                acc = SmeElement::fused(*args.a.add(i + j * args.lda), t, acc);
            }
            *y = if fault == Fault::AlphaAfter {
                *y + alpha * acc
            } else {
                acc
            };
        }
    }
    if fault == Fault::StorePastEnd {
        // SAFETY: the self-check's buffers have a guard after `y` (this
        // stand-in exists to be caught writing into it).
        unsafe { *args.out.add(args.m) = T::ZERO };
    }
}

/// The Trans stand-in with `fault`.
///
/// # Safety
///
/// As for [`standin_notrans`].
unsafe fn standin_trans<T: GvElem>(args: &GemvArgs<T>, fault: Fault) {
    let lanes = T::LANES;
    // SAFETY: `x[0..m]` by the contract.
    let x = unsafe { core::slice::from_raw_parts(args.x, args.m) };
    for j in 0..args.n {
        // SAFETY: column `j < n` of A, rows `0..m`.
        let col = unsafe { core::slice::from_raw_parts(args.a.add(j * args.lda), args.m) };
        let body = args.m - args.m % lanes;
        let rows = col[..body].iter().zip(&x[..body]).enumerate();
        let acc = match fault {
            Fault::TailLanes => {
                let mut acc = vec![T::ZERO; lanes];
                for (i, (c, xv)) in rows {
                    acc[i % lanes] = SmeElement::fused(*c, *xv, acc[i % lanes]);
                }
                let tail = col[body..scalar_start(args.m, lanes)]
                    .iter()
                    .zip(&x[body..]);
                for (lane, (c, xv)) in acc.iter_mut().zip(tail) {
                    *lane = SmeElement::fused(*c, *xv, *lane);
                }
                acc
            }
            Fault::Reversed => {
                let mut acc = vec![T::ZERO; lanes];
                for (i, (c, xv)) in rows.rev() {
                    acc[i % lanes] = SmeElement::fused(*c, *xv, acc[i % lanes]);
                }
                if let Some((base, lo, hi)) = vector_tail(args.m, lanes) {
                    for (l, lane) in acc.iter_mut().enumerate().take(hi).skip(lo) {
                        *lane = SmeElement::fused(col[base + l], x[base + l], *lane);
                    }
                }
                acc
            }
            Fault::PositiveZeroDeadLanes => {
                let mut acc = reference_lanes(col, x, lanes);
                if let Some((_, lo, hi)) = vector_tail(args.m, lanes) {
                    for (l, lane) in acc.iter_mut().enumerate() {
                        if !(lo..hi).contains(&l) {
                            *lane = SmeElement::fused(T::ZERO, T::ZERO, *lane);
                        }
                    }
                }
                acc
            }
            _ => reference_lanes(col, x, lanes),
        };
        for (l, value) in acc.iter().enumerate() {
            // SAFETY: `out[0..n · lanes]` by the contract.
            unsafe { *args.out.add(j * lanes + l) = *value };
        }
    }
    if fault == Fault::ExtraColumn {
        // SAFETY: the self-check's partials buffer has a guard of 8
        // elements after `n · lanes` (this stand-in exists to be caught
        // writing the first lane of a column too many into it).
        unsafe { *args.out.add(args.n * lanes) = T::ZERO };
    }
}

macro_rules! standins {
    ($($name:ident: $kind:ident($fault:ident);)*) => {
        $(
            /// A stand-in kernel (see [`Fault`]).
            ///
            /// # Safety
            ///
            /// As for the GEMV kernels.
            pub(super) unsafe fn $name<T: GvElem>(args: &GemvArgs<T>) {
                // SAFETY: forwarded.
                unsafe { $kind(args, Fault::$fault) }
            }
        )*
    };
}

standins! {
    good_notrans: standin_notrans(None);
    last_column: standin_notrans(LastColumn);
    store_past_end: standin_notrans(StorePastEnd);
    alpha_after: standin_notrans(AlphaAfter);
    good_trans: standin_trans(None);
    tail_lanes: standin_trans(TailLanes);
    extra_column: standin_trans(ExtraColumn);
    reversed: standin_trans(Reversed);
    positive_zero_dead_lanes: standin_trans(PositiveZeroDeadLanes);
}

#[test]
fn test_sme2_gemv_self_check_with_stand_ins() {
    let good32: (GemvKernelFn<f32>, GemvKernelFn<f32>) = (good_notrans::<f32>, good_trans::<f32>);
    let good64: (GemvKernelFn<f64>, GemvKernelFn<f64>) = (good_notrans::<f64>, good_trans::<f64>);
    // SAFETY: the stand-ins are plain Rust that keep to the buffers the
    // self-check hands them (or write into its guards on purpose).
    unsafe {
        assert!(
            self_check_with(good32, Some(good64)),
            "definition stand-ins rejected"
        );
        assert!(self_check_with(good32, None));
        let wrong_notrans: [GemvKernelFn<f64>; 3] = [
            last_column::<f64>,
            store_past_end::<f64>,
            alpha_after::<f64>,
        ];
        for (index, wrong) in wrong_notrans.into_iter().enumerate() {
            assert!(
                !self_check_with(good32, Some((wrong, good64.1))),
                "f64 NoTrans fault {index}"
            );
        }
        let wrong_trans: [GemvKernelFn<f32>; 3] =
            [tail_lanes::<f32>, extra_column::<f32>, reversed::<f32>];
        for (index, wrong) in wrong_trans.into_iter().enumerate() {
            assert!(
                !self_check_with((good32.0, wrong), Some(good64)),
                "f32 Trans fault {index}"
            );
            assert!(
                !self_check_with((good32.0, wrong), None),
                "f32 Trans fault {index}"
            );
        }
        let wrong32: [GemvKernelFn<f32>; 3] = [
            last_column::<f32>,
            store_past_end::<f32>,
            alpha_after::<f32>,
        ];
        for (index, wrong) in wrong32.into_iter().enumerate() {
            assert!(
                !self_check_with((wrong, good32.1), None),
                "f32 NoTrans fault {index}"
            );
        }
        let wrong64: [GemvKernelFn<f64>; 3] =
            [tail_lanes::<f64>, extra_column::<f64>, reversed::<f64>];
        for (index, wrong) in wrong64.into_iter().enumerate() {
            assert!(
                !self_check_with(good32, Some((good64.0, wrong))),
                "f64 Trans fault {index}"
            );
            assert!(
                self_check_with(good32, None),
                "f64 kernels are not checked without f64"
            );
        }
    }
    // The shapes reach every vector-tail case of both types and both
    // NoTrans column remainders.
    for lanes in [8, 16] {
        let mut cases: Vec<Option<(usize, usize)>> = CHECK_TRANS
            .iter()
            .map(|&(m, _)| vector_tail(m, lanes).map(|(_, lo, hi)| (lo, hi)))
            .collect();
        cases.sort_unstable();
        cases.dedup();
        assert_eq!(cases.len(), 4, "lanes {lanes}: {cases:?}");
    }
    let mut rems: Vec<usize> = CHECK_NOTRANS.iter().map(|&(_, n)| n % 4).collect();
    rems.sort_unstable();
    rems.dedup();
    assert_eq!(rems, vec![0, 1, 2, 3]);
    assert!(CHECK_NOTRANS.iter().any(|&(m, _)| m > 1024));
    assert!(CHECK_TRANS.iter().any(|&(_, n)| n > 64));
    assert_eq!(TRANS_COLS_PER_CALL, 1024);
}

/// The self-check's negative-zero shapes catch a `Trans` kernel that adds
/// `+0 · +0` in the dead lanes of the vector tail (the defect pinned by the
/// `trans_negative_zero_lanes` tests), for both types: each such shape has
/// dead tail lanes for both lane counts, and the definition's partials there
/// are `-0` (the test thread runs with FPCR = 0).
#[test]
fn test_sme2_gemv_self_check_catches_positive_zero_dead_lanes() {
    let good32: (GemvKernelFn<f32>, GemvKernelFn<f32>) = (good_notrans::<f32>, good_trans::<f32>);
    let good64: (GemvKernelFn<f64>, GemvKernelFn<f64>) = (good_notrans::<f64>, good_trans::<f64>);
    let shapes = &CHECK_TRANS[CHECK_NEGATIVE_ZERO_FROM..];
    assert!(!shapes.is_empty());
    for lanes in [8, 16] {
        for &(m, _) in shapes {
            let tail = vector_tail(m, lanes);
            assert!(
                tail.is_some_and(|(_, lo, hi)| lo > 0 || hi < lanes),
                "m {m}, {lanes} lanes: no dead tail lane ({tail:?})"
            );
        }
    }
    // SAFETY: the stand-ins are plain Rust that keep to the buffers the
    // self-check hands them.
    unsafe {
        assert!(self_check_with(good32, Some(good64)), "definition rejected");
        assert!(
            !self_check_with((good32.0, positive_zero_dead_lanes::<f32>), None),
            "f32: +0 in the dead tail lanes accepted"
        );
        assert!(
            !self_check_with(good32, Some((good64.0, positive_zero_dead_lanes::<f64>))),
            "f64: +0 in the dead tail lanes accepted"
        );
    }
}
