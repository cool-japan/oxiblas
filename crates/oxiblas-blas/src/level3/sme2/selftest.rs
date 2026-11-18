//! First-use self-test of the SME2 macro-kernels.
//!
//! `self_test_with` runs one macro-kernel on a handful of odd blocks and a
//! systematic sweep of small ones, in every epilogue mode, without and with
//! online packing of A and the C prefetch, and compares every result (and
//! every packed copy of A) with the scalar oracle (`oracle.rs`), bit for bit
//! except NaN payloads in C. It exists because the kernels are verified
//! under QEMU only: if the first SME2 machine they meet (an Apple M4, say)
//! rounds, flushes subnormals, orders an epilogue or moves a multi-vector
//! store differently, the self-test fails and the backend must stay off for
//! the process, so GEMM keeps the NEON results.
//!
//! # What it checks
//!
//! The fixed blocks ([`CASES`]) cover several 64-row and `NR`-column panels,
//! partial tiles in both directions, full tiles only, a single element,
//! `kc = 1`, deep and odd `kc`, a `kc` that is a multiple of 8 (as the
//! caller's `KC` usually is), a last row panel of 17 to 32 rows (two active
//! 16-row tiles), three or more column panels with `kc > 1` for either panel
//! width, the plain packed A layout and the in-place A layout (column-major,
//! `a_depth = lda`, `a_panel = 64`) with more than one row panel and with
//! leading dimensions other than `m + 2`, a B panel stride larger than
//! `kc · NR` (the gap between panels holds a signalling NaN), and one block
//! of special values: subnormal inputs, products and results, signed zeros
//! and a few infinities.
//!
//! The sweep ([`sweep_cases`], generic over the tile height and the panel
//! width of the element type) takes every path class of the model in
//! `tests_selftest_paths.rs`: every row variant (1 to 4 live 16-row tiles for
//! `f32`, 1 to 8 live 8-row tiles for `f64`), each with `kc = 1`, an even and
//! an odd `kc` (two passes of the double-step loop, without and with the
//! remainder step), as the only row panel and as the last one after a full
//! panel, packed and in place, the last live tile full or holding one row,
//! each over a full and a partial column panel; and a full row panel between
//! two others (three row panels) with each depth, packed and in place. Its
//! in-place blocks set `PREFETCH_C`, its packed ones do not. The online
//! sweep ([`online_sweep_cases`]) runs the same blocks once more with online
//! packing, so the packing twin of every row variant (column panel 0) and
//! the plain variants on the buffer (column panel 1) meet the same classes;
//! the four [`online_cases`] add a packing call with one partial column
//! panel (`n ≤ NR` still fills the buffer) and the buffer read by full
//! column panels, each with and without `PREFETCH_C`. The tests compute the
//! classes of every block from this list and require all of them (the path
//! table).
//!
//! The tests also tie that model to the registered kernel text (the exact
//! text that is assembled; `derive_structure` in `tests_selftest_text.rs`
//! lists the features of the text it relies on). Read from the text: the
//! element size; the row variants and their packing twins (their zeroing,
//! depth loop and remainder) with their live tiles, depth steps per pass and
//! remainder steps, a twin's stores (all through the buffer register, which
//! advances one 64-row step per depth step); the pass count `kc >> s`; the
//! gate of the packing pass and both variant dispatches (which variant or
//! twin every row count `1..=64` of a row panel takes, and the thresholds);
//! the row-panel step; the switch of A to the buffer after column panel 0
//! (`a`, `a_depth`, `a_panel` and the gate, written nowhere else); the
//! prefetch block, evaluated for every row count, live column count and
//! both values of the bit (exactly one hint per live tile and live column
//! when the bit is set, none when it is clear); the column-panel step, the
//! clamp of the live columns and the C column advance (so `NR` comes from
//! the text); the epilogue dispatch (each of the four modes reaches its own
//! section, in order, and nothing else is tested there); the four epilogue
//! sections' tile chains and their column loops (each bounded by the live
//! columns); that no outer product and no conditional depth step lies
//! outside a variant's depth loop and remainder; and that the body has no
//! branch the model does not know (a new branch is a new path). Taken from
//! constants and checked against the text: the row panel of `MR = 64` rows,
//! the streaming vector of 64 bytes (SVL 512, so a tile holds 64 / element
//! size rows and a B row as many columns), the label conventions of the
//! kernels and the registers found through the prologue's loads of the
//! argument block. A change of the text in any of these fails a native test
//! until the model, the path computation and the sweeps follow it; a change
//! elsewhere (an instruction inside a section that the derivation does not
//! interpret, a load's operands, a store's offset) is not seen there.
//!
//! The model does not distinguish (so the self-test does not promise them):
//! a single pass of the depth loop (`kc` 2 or 3; [`CASES`] and the fourth
//! online case run one), a last live tile that is neither full nor one row,
//! more than two column panels and the position of the column panel (a
//! partial panel as the only one, a full panel as the last one: in the
//! fixed blocks only), the values of `ldc`, `lda` and `b_pad`, the data
//! (plain values everywhere but one special block of [`CASES`] and one of
//! [`online_cases`]), NaN inputs, and a non-default FPCR. It sees writes
//! outside the block and the buffer only where they land in the guards
//! described below, and a hint not at all (hints change nothing). The model
//! checks the hints in the text relative to the first one (their count per
//! live tile and column, the 64-byte tile step and the `ldc` column step),
//! not where the first one points: a prefetch block whose base is off (one
//! row panel, one column) passes the model and every result, and only the
//! kernels' interpreter tests see it. It cannot see a forbidden read of C (outside
//! the block, or in M0/M2), nor one of A rows `≥ m` without online packing
//! (such reads feed only lanes that are never stored); the template tests
//! of the kernels guard those. A packing pass that reads A rows `≥ m` and
//! stores them shows in the buffer's padding when A is in place (its spare
//! rows hold a signalling NaN).
//!
//! Each block runs with the six alpha/beta pairs of [`SCALARS`]:
//! M0, M1, M2 (with `beta = -0`) and M3 in its three branches (`alpha == 1`,
//! `beta == 1`, neither), all with inexact scalars. For every run it checks:
//!
//! - that A and B are unchanged, bit for bit (their padding included: the
//!   spare rows of in-place A and the gaps between B panels hold signalling
//!   NaNs);
//! - that the guard border around C still holds its sentinel bits exactly,
//!   in the kernel's C and in the oracle's (see [`c_layout`]). C has
//!   `ldc = m + 3` with `C(0, 0)` at index `ldc + 1`, and its buffer runs to
//!   the farther of two ends: the end of the column after the block (rows
//!   `0..ldc - 1` of column `n`, offset `(n + 1) · ldc - 1` from `C(0, 0)`),
//!   and one past the farthest element that a whole `64 × NR` C tile at any
//!   tile position of the block addresses, `C(R - 1, N - 1)` with
//!   `R = 64 · ceil(m / 64)` and `N = NR · ceil(n / NR)` (one past it is the
//!   offset `R + (N - 1) · ldc`). So it holds `ldc + 1 + max(…)` elements of
//!   those two offsets. The guard border
//!   is every element outside `C[0..m, 0..n]` (all but `m · n` of them): the
//!   whole column before the block (`ldc` elements, the last of which sits
//!   just before `C(0, 0)`) and the element before it (index 0), rows
//!   `m..ldc` of every column of the block, and everything after column
//!   `n - 1` to the end of the buffer. Because `ldc < 64 · ceil(m / 64)` in
//!   general, the rows of a whole tile below the block wrap into the
//!   following columns; they are all inside the buffer;
//! - so the self-test contains every store and every read-modify-write that
//!   stays within whole tiles: a kernel that stores all 64 rows of a partial
//!   last row panel, all `NR` columns of a partial last column panel, or
//!   both, writes inside the buffer and is reported at the first guard
//!   element it changed (never a panic or an abort). A store farther away
//!   (beyond the whole tiles of the block, or before the column in front of
//!   it) is outside what the self-test can contain: it lands outside the
//!   buffer, corrupts the heap and may end the process before a verdict.
//!   The kernels' template and QEMU tests guard against those;
//! - the sentinel is a signalling NaN: a read-modify-write of a guard
//!   element (one row or one column too many in M1 or M3, even with a zero
//!   accumulator) quiets it and changes its bits, where a quiet NaN would
//!   come back unchanged;
//! - every element of `C[0..m, 0..n]` against the oracle
//!   (`same_bits_or_nan`);
//! - that in M0 and M2 a C filled with NaN does not leak into the result;
//! - with online packing, the buffer ([`pack_layout`]): the kernel's and the
//!   oracle's start alike, `PACK_FRONT = 2 · MR` guard elements, the
//!   `P = 64 · ceil(m / 64) · kc` elements of the packed copy with stale
//!   contents ([`pack_stale`]: signalling NaNs and large negative numbers,
//!   none of them a value of a correct copy), then `max(N - 1, 1) · P +
//!   2 · MR` guard elements for the `N = ceil(n / NR)` column panels. The
//!   guards ([`pack_guard`], a signalling NaN of its own payload) must hold
//!   their bits exactly, in the kernel's buffer and in the oracle's; the
//!   packed copy must equal the oracle's bit for bit (no NaN allowance:
//!   packing only moves bits, `-0` against the `+0` padding is a
//!   difference), so an element the kernel does not write keeps its stale
//!   value and is reported;
//! - so the self-test contains every store of a faulty packing pass that
//!   stays within those guards: a store one vector or one four-vector group
//!   off at the last depth step (at most 64 elements after the buffer), the
//!   buffer pointer one depth step early or late (64 elements before or
//!   after), one depth step too far per pass of the double-step loop (about
//!   `P / 2` after), and the packing pass repeated on every later column
//!   panel with the pointer running on (`(N - 1) · P` after, and one more
//!   depth step with it): all are reported as a guard difference of the
//!   buffer, never a write outside the allocation
//!   (`test_sme2_selftest_online_containment` checks each class on every
//!   online block, and stand-ins of each are rejected). A store farther
//!   away, or a later column panel that reads A beyond the guards (a panel
//!   stride three or more times too large), is outside what the self-test
//!   can contain: on hardware the store corrupts the heap and the read may
//!   fault. The kernels' template and QEMU tests guard against those.
//!
//! Nothing in the self-test itself computes on these signalling NaNs, so they
//! cannot cause a false rejection: the fills are copied (`vec!`, `clone`)
//! and compared by their bits only, and the oracle (like a correct kernel)
//! reads neither A rows `≥ m`, nor B outside `kc · NR` of a panel, nor C
//! outside the block, nor the online buffer (it only writes the packed
//! copy). Only a kernel that breaks the contract computes on them.
//!
//! Cost (`test_sme2_selftest_cost_matches_the_docs` pins these figures):
//!
//! | | blocks | kernel calls | largest run | largest buffer | all runs together |
//! |---|---|---|---|---|---|
//! | `f32`, [`CASES`] only | 7 | 42 | 80 844 B | 26 308 B | 1 584 480 B |
//! | `f64`, [`CASES`] only | 7 | 42 | 143 744 B | 44 104 B | 2 698 416 B |
//! | `f32`, the whole list | 7 + 4 + 2 · 54 | 714 | 131 860 B | 26 308 B | 29 329 440 B |
//! | `f64`, the whole list | 7 + 4 + 2 · 102 | 1 290 | 207 464 B | 51 200 B | 61 938 096 B |
//!
//! So `run_self_test` makes 714 kernel calls for `f32` alone and
//! 714 + 1 290 = 2 004 with `f64` (84 with [`CASES`] only); with `f64` the
//! largest run is 207 464 B, the largest buffer 51 200 B, and all runs
//! together allocate 91 267 536 B. Each kernel call has one oracle call
//! beside it. A run holds A, B and their copies, C (with its guard border)
//! and its two copies, and with online packing the kernel's and the
//! oracle's buffer with their guards; the largest run is the third online
//! case, `84 × (2 · NR + 3) × 16` in place (three column panels, so the
//! guard after its buffer holds two more copies), for either type. The
//! largest single buffer is the C of the `130 × 33 × 1` block of [`CASES`]
//! for `f32` (its whole tiles reach `C(191, 47)`) and that online case's
//! buffer of 6 400 elements for `f64`. The buffers of one run are freed
//! before the next, so the peak is the largest run; the last column is the
//! sum of all allocations. Under QEMU (emulated, release build, one
//! single-threaded run on a shared host under load, so rough) the whole
//! `run_self_test` with both types took 284 ms and `f32` alone 178 ms
//! (before the online blocks: 159–175 ms and 90–105 ms). It has not been
//! measured on hardware, where emulation's cost per instruction is gone.
//!
//! The data are generated here (a fixed xorshift sequence, no clock, no
//! randomness source, no dependency), so the verdict is deterministic. The
//! test never panics: all indices are checked, and the first mismatch ends
//! it.
//!
//! # Reporting a mismatch
//!
//! The `bool` functions (`self_test_with`, `self_test_cases`,
//! `self_test_kernels`, `run_self_test`) give the verdict only. Beside
//! each, a `_detailed` function returns `Result<(), Mismatch>`: [`Mismatch`]
//! is a plain `Copy` value (no allocation) that names the element type, the
//! block (its index in the list and its shape, online packing and prefetch),
//! the scalar pair and what differed (the block refused, A or B changed, a
//! guard element of the kernel's or of the oracle's C or online buffer
//! changed, an element of the block, or an element of the packed copy by its
//! row and depth step, with the bits found and wanted). Its `Display` is one
//! line that a user can paste into an issue. The `bool` functions are
//! `is_ok()` of these, so both stop at the same first mismatch at the same
//! cost.
//!
//! The detection (`detect.rs`) runs [`run_self_test_detailed`] and keeps
//! the `Display` line of the first [`Mismatch`] as the line of its failed
//! verdict, which it writes to standard error once. [`Difference::Refused`],
//! [`Difference::OracleGuard`] and [`Difference::OracleBufferGuard`] are
//! defects of the self-test itself, not of the hardware; their lines say
//! so.
//!
//! Floating-point flags: the oracle and the kernel raise the usual cumulative
//! flags (inexact, underflow, invalid from `0 · ∞`) on the calling thread. The
//! self-test does not save FPSR itself; the detection runs it inside
//! `fpstate::with_default_fp_state` (FPCR = 0, the caller's FPCR and FPSR
//! restored afterwards).
//!
//! # The real kernels: `run_self_test`
//!
//! `run_self_test` binds the self-test to the real macro-kernels: it runs
//! `self_test_with(sme2_gemm_f32)`, then `self_test_with(sme2_gemm_f64)` when
//! its argument says that the CPU has FEAT_SME_F64F64, and returns `true` only
//! when every kernel it ran passed (the verdicts are combined by
//! `self_test_kernels`, which tests also run on plain-Rust stand-ins). It
//! makes `self_test_calls` kernel calls per type it runs: 714 for `f32`
//! and 1 290 for `f64`, 2 004 in total with `f64`. The detection
//! (`detect.rs`) calls [`run_self_test_detailed`] as the first of its four
//! first-use checks:
//!
//! - once per process, after the candidate (operating-system report and
//!   RDSVL) is present, on the first detection call (`detect::hw` or
//!   `detect::hw_now`) from a ready thread, under the detection's lock; a
//!   mismatch turns the whole backend off for the life of the process.
//! - It never calls `detect::hw()`, `detect::enabled()`, `detect::
//!   thread_ready()`, `sme2_active()` or any GEMM entry point, and creates
//!   no thread, so it cannot re-enter the detection. For the same reason the
//!   detection does not call `detect::thread_ready()` (which calls `hw_now()`)
//!   before it: it checks the thread state that function checks (SVCR = 0,
//!   TPIDR2_EL0 = 0, a 64-byte streaming vector length) with
//!   `probe_thread_state()` directly.
//! - A thread that is not in that state is not evidence of a broken kernel:
//!   the detection then does not run the self-test and records no mismatch
//!   (the verdict stays open for a later call from a ready thread).
//! - The self-test leaves its cumulative floating-point flags on the calling
//!   thread (see above); the detection restores the caller's FPCR and FPSR
//!   afterwards, so that the first query does not leave flags behind.

use core::fmt;

use super::args::{Epilogue, MR, MacroArgs, PREFETCH_C, PREFETCH_OFF, SmeElement};
use super::kernel_f32::sme2_gemm_f32;
use super::kernel_f64::sme2_gemm_f64;
use super::oracle::{oracle_block, same_bits_or_nan};

/// A macro-kernel, or anything with its signature (the oracle, a stand-in).
pub(super) type KernelFn<T> = unsafe fn(&MacroArgs<T>);

/// The online-packing and prefetch fields of a block's calls: online
/// packing of A into a
/// buffer (`a_pack`) and the C prefetch (`PREFETCH_C`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Online {
    /// `a_pack` null and `prefetch` 0: plain packed blocks.
    Off,
    /// `PREFETCH_C` only.
    Prefetch,
    /// Online packing only.
    Pack,
    /// Online packing and `PREFETCH_C`.
    PackPrefetch,
}

impl Online {
    /// Whether the calls pack A online (a non-null `a_pack`).
    pub(super) fn packs(self) -> bool {
        matches!(self, Self::Pack | Self::PackPrefetch)
    }

    /// The `prefetch` field of the calls.
    pub(super) fn prefetch(self) -> u64 {
        if matches!(self, Self::Prefetch | Self::PackPrefetch) {
            PREFETCH_C
        } else {
            PREFETCH_OFF
        }
    }
}

/// How the self-test lays out A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ALayout {
    /// The plain packed layout: `a_depth = 64`, `a_panel = 64 · kc`, rows
    /// `m..` of the last panel zero.
    Packed,
    /// Column-major in place: `a_depth = lda = m + lda_pad` ([`Case::lda_pad`]),
    /// `a_panel = 64`; the spare rows of every column hold a signalling NaN
    /// (a kernel must not use them).
    InPlace,
}

/// Which values the self-test generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Data {
    /// Full-mantissa values of mixed sign and magnitude, some exact zeros.
    Plain,
    /// Subnormal inputs and products, signed zeros, a few infinities.
    Special,
}

/// One block of the self-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Case {
    /// Rows.
    pub(super) m: usize,
    /// Columns.
    pub(super) n: usize,
    /// Depth.
    pub(super) kc: usize,
    /// Layout of A.
    pub(super) layout: ALayout,
    /// [`ALayout::InPlace`]: spare rows per column of A (`lda = m + lda_pad`).
    /// Must be 0 for [`ALayout::Packed`] (the run fails otherwise).
    pub(super) lda_pad: usize,
    /// Elements between consecutive B panels beyond `kc · NR`
    /// (`b_panel = kc · NR + b_pad`); the gaps hold a signalling NaN.
    pub(super) b_pad: usize,
    /// Values of A, B and C.
    pub(super) data: Data,
    /// Online packing and prefetch of the calls.
    pub(super) online: Online,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = match self.layout {
            ALayout::Packed => "packed",
            ALayout::InPlace => "in place",
        };
        let data = match self.data {
            Data::Plain => "plain",
            Data::Special => "special",
        };
        let online = match self.online {
            Online::Off => "",
            Online::Prefetch => ", prefetch C",
            Online::Pack => ", online packing",
            Online::PackPrefetch => ", online packing, prefetch C",
        };
        write!(
            f,
            "{}x{}x{}, A {layout}, lda pad {}, B pad {}, {data} data{online}",
            self.m, self.n, self.kc, self.lda_pad, self.b_pad
        )
    }
}

/// What the self-test found different in one run (see [`Mismatch`]).
///
/// A guard element is given three ways: its signed element offset from
/// `C(0, 0)`, the run's `ldc` (`m + 3`), and the coordinates that offset
/// splits into with the row in `0..ldc` (`offset = row + col · ldc`). The
/// column runs from -2 (only the first element of the buffer) to the last
/// column the buffer reaches (see [`c_layout`]). The offset is the
/// unambiguous reading: the coordinates alias, because a kernel that
/// addresses a row outside `0..ldc` reaches an element of another column.
/// One row above the block, `C(-1, j)`, is the element `C(m + 2, j - 1)`
/// (offset `j · ldc - 1`; for `j = 0` it is reported as `C(ldc - 1, -1)`),
/// and a row `r ≥ ldc` below the block, `C(r, j)`, is `C(r - ldc, j + 1)`
/// (a whole tile below a short block wraps into the following columns).
/// So `C(72, -1)` of a 70-row block may mean "one row above `C(0, 0)`", and
/// `C(0, n)` may mean "row `m + 3` of column `n - 1`"; the offset and `ldc`
/// tell which addresses a kernel could have meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Difference {
    /// The block could not be run (a zero dimension, a packed block with an
    /// lda pad, a size that overflows): a defect of the block list.
    Refused,
    /// The kernel changed A: the index of the first changed element in the
    /// self-test's A buffer (its padding included).
    AChanged {
        /// Index into A.
        index: usize,
    },
    /// The kernel changed B: the index of the first changed element in the
    /// self-test's B buffer (zero padding and gaps included).
    BChanged {
        /// Index into B.
        index: usize,
    },
    /// A guard element of the kernel's C no longer holds the sentinel.
    KernelGuard {
        /// Row relative to `C(0, 0)`, in `0..ldc` (see the type's docs).
        row: i32,
        /// Column relative to `C(0, 0)`.
        col: i32,
        /// Signed element offset from `C(0, 0)`: `row + col · ldc`.
        offset: i64,
        /// Leading dimension of the run's C (`m + 3`).
        ldc: u32,
        /// The bits found.
        found: u64,
    },
    /// A guard element of the oracle's C no longer holds the sentinel (a
    /// defect of the oracle, not of the kernel).
    OracleGuard {
        /// Row relative to `C(0, 0)`, in `0..ldc` (see the type's docs).
        row: i32,
        /// Column relative to `C(0, 0)`.
        col: i32,
        /// Signed element offset from `C(0, 0)`: `row + col · ldc`.
        offset: i64,
        /// Leading dimension of the run's C (`m + 3`).
        ldc: u32,
        /// The bits found.
        found: u64,
    },
    /// An element of the block differs from the oracle's.
    Element {
        /// Row of the block.
        row: u32,
        /// Column of the block.
        col: u32,
        /// The kernel's bits.
        found: u64,
        /// The oracle's bits.
        wanted: u64,
    },
    /// A guard element of the kernel's online A buffer (before the buffer,
    /// or after its first `len` elements) no longer holds its sentinel.
    BufferGuard {
        /// Signed element offset from the buffer's first element (negative
        /// before it, `len` or more after it).
        offset: i64,
        /// Elements the kernel may write: `ceil(m / 64) · 64 · kc`.
        len: u32,
        /// The bits found.
        found: u64,
    },
    /// A guard element of the oracle's online A buffer changed (a defect of
    /// the oracle, not of the kernel).
    OracleBufferGuard {
        /// Signed element offset from the buffer's first element.
        offset: i64,
        /// Elements the oracle may write.
        len: u32,
        /// The bits found.
        found: u64,
    },
    /// An element of the kernel's packed copy of A (online packing) differs
    /// from the oracle's, bit for bit: the element of row `row` (rows `≥ m`
    /// are the zero padding of the last row panel) and depth step `depth`.
    BufferElement {
        /// Row of A, `0..64 · ceil(m / 64)`.
        row: u32,
        /// Depth step, `0..kc`.
        depth: u32,
        /// The kernel's bits.
        found: u64,
        /// The oracle's bits.
        wanted: u64,
    },
}

/// The first mismatch of a self-test run: which run, and what differed.
///
/// A plain `Copy` value made without allocating; its `Display` is one line
/// for a bug report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Mismatch {
    /// Bits of the element type: 32 or 64.
    pub(super) element_bits: u32,
    /// Index of the block in the list that ran ([`self_test_blocks`] for
    /// `self_test_with`, the given list for `self_test_cases`).
    pub(super) block: usize,
    /// The block.
    pub(super) case: Case,
    /// Index of the scalar pair ([`SCALARS`], or the given pairs).
    pub(super) pair: usize,
    /// Alpha of the pair, as given (before rounding to `f32`).
    pub(super) alpha: f64,
    /// Beta of the pair, as given.
    pub(super) beta: f64,
    /// What differed.
    pub(super) difference: Difference,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hexadecimal digits of one element (with the `0x` prefix).
        let width = if self.element_bits == 32 { 10 } else { 18 };
        write!(
            f,
            "SME2 self-test mismatch, f{}: block {} ({}), pair {} (alpha {}, beta {}): ",
            self.element_bits, self.block, self.case, self.pair, self.alpha, self.beta
        )?;
        match self.difference {
            Difference::Refused => write!(f, "the block was refused (a self-test defect)"),
            Difference::AChanged { index } => write!(f, "the kernel changed A[{index}]"),
            Difference::BChanged { index } => write!(f, "the kernel changed B[{index}]"),
            Difference::KernelGuard {
                row,
                col,
                offset,
                ldc,
                found,
            } => write!(
                f,
                "the kernel wrote C({row}, {col}) outside the block (offset {offset} from \
                 C(0, 0), ldc {ldc}): {found:#0width$x}"
            ),
            Difference::OracleGuard {
                row,
                col,
                offset,
                ldc,
                found,
            } => write!(
                f,
                "the oracle wrote C({row}, {col}) outside the block (offset {offset} from \
                 C(0, 0), ldc {ldc}): {found:#0width$x} (an oracle defect)"
            ),
            Difference::Element {
                row,
                col,
                found,
                wanted,
            } => write!(
                f,
                "C({row}, {col}) is {found:#0width$x}, the oracle gives {wanted:#0width$x}"
            ),
            Difference::BufferGuard { offset, len, found } => write!(
                f,
                "the kernel wrote the online A buffer at offset {offset} (outside its {len} \
                 elements): {found:#0width$x}"
            ),
            Difference::OracleBufferGuard { offset, len, found } => write!(
                f,
                "the oracle wrote the online A buffer at offset {offset} (outside its {len} \
                 elements): {found:#0width$x} (an oracle defect)"
            ),
            Difference::BufferElement {
                row,
                depth,
                found,
                wanted,
            } => {
                let padding = if usize::try_from(row).is_ok_and(|row| row >= self.case.m) {
                    " (zero padding)"
                } else {
                    ""
                };
                write!(
                    f,
                    "the online A buffer holds {found:#0width$x} for row {row}{padding}, depth \
                     {depth}, the oracle {wanted:#0width$x}"
                )
            }
        }
    }
}

/// The blocks of `self_test_with` (see the module docs).
pub(super) const CASES: [Case; 7] = [
    // 2 row panels (64 + 6) and 2 (f32: 16 + 3) or 3 (f64: 8 + 8 + 3) column
    // panels: partial tiles in both directions; B panels 5 elements apart
    // beyond `kc · NR` (signalling NaN in the gaps).
    Case {
        m: 70,
        n: 19,
        kc: 13,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 5,
        data: Data::Plain,
        online: Online::Off,
    },
    // kc = 1; 3 row panels (64 + 64 + 2), 3 (f32) or 5 (f64) column panels.
    Case {
        m: 130,
        n: 33,
        kc: 1,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    },
    // Whole tiles only, A in place.
    Case {
        m: 64,
        n: 16,
        kc: 2,
        layout: ALayout::InPlace,
        lda_pad: 2,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    },
    // A single element.
    Case {
        m: 1,
        n: 1,
        kc: 3,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    },
    // Fewer rows than one tile, deep and odd kc, A in place.
    Case {
        m: 9,
        n: 7,
        kc: 67,
        layout: ALayout::InPlace,
        lda_pad: 2,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    },
    // Subnormals and special values, partial tiles in both directions.
    Case {
        m: 33,
        n: 17,
        kc: 5,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Special,
        online: Online::Off,
    },
    // A in place over 2 row panels (64 + 20: the last one has two active
    // 16-row tiles) with `lda = m + 3`; kc = 16 (a multiple of 4 and 8);
    // 3 (f32: 16 + 16 + 8) or 5 (f64) column panels with kc > 1; B panels
    // 3 elements apart beyond `kc · NR`.
    Case {
        m: 84,
        n: 40,
        kc: 16,
        layout: ALayout::InPlace,
        lda_pad: 3,
        b_pad: 3,
        data: Data::Plain,
        online: Online::Off,
    },
];

/// The alpha/beta pairs of `self_test_with`: M0, M1, M2 (`beta = -0`),
/// M3 with `alpha == 1`, M3 with `beta == 1`, M3 with neither.
pub(super) const SCALARS: [(f64, f64); 6] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (0.3, -0.0),
    (1.0, -0.7),
    (-1.3, 1.0),
    (0.3, -0.7),
];

/// The depths of the sweep: `kc = 1` (no pass of the double-step loop, the
/// remainder step only), an even depth (two passes, no remainder) and an odd
/// depth (two passes and the remainder). Two passes take the loop's back
/// branch and its pointer steps, which one pass would not.
pub(super) const SWEEP_DEPTHS: [usize; 3] = [1, 4, 5];

/// Blocks of the sweep per row variant: two row-panel positions (only panel,
/// last panel after a full one) × two A layouts × [`SWEEP_DEPTHS`].
pub(super) const SWEEP_BLOCKS_PER_VARIANT: usize = 2 * 2 * SWEEP_DEPTHS.len();

/// Blocks of the sweep with three row panels (a full row panel between two
/// others): two A layouts × [`SWEEP_DEPTHS`].
pub(super) const SWEEP_MIDDLE_BLOCKS: usize = 2 * SWEEP_DEPTHS.len();

/// The systematic sweep of `self_test_with` for a kernel with ZA tiles of
/// `tile_rows` rows and B panels of `nr` columns (`T::TILE_ROWS`, `T::NR`),
/// so that both element types are covered by construction.
///
/// For every row variant `L` (`1..=64 / tile_rows` live tiles in the last
/// row panel), every position of that panel (the only panel: `m ≤ 64`; the
/// last one after a full 64-row panel: `m = 64 + rows`), both A layouts and
/// every depth of [`SWEEP_DEPTHS`] it yields one block of `n = nr + c`
/// columns (`1 ≤ c < nr`: a full and a partial column panel). The last live
/// tile holds exactly one row (`rows = (L - 1) · tile_rows + 1`) for packed
/// A as the only panel and for in-place A as the last panel, and is full
/// (`rows = L · tile_rows`) otherwise, so each position and each layout meets
/// both. Then [`SWEEP_MIDDLE_BLOCKS`] blocks of three row panels
/// (`m = 128 + rows`, `129 ≤ m ≤ 192`), one per A layout and depth, whose
/// second row panel is full and neither the first nor the last. That middle
/// panel gets its A and C pointers from the first application of the
/// row-panel advance (as the last panel of a two-panel block does); what it
/// brings is the third panel behind it, which gets them from the second
/// application, with the same depth, A layout, epilogue and column panels.
/// So requiring the middle panel's classes requires the second application
/// in each of them. Their last panels vary in size (see [`middle_case`]).
/// With the six pairs of [`SCALARS`] every block runs in every epilogue
/// mode; the in-place blocks set `PREFETCH_C` ([`sweep_online`]). The
/// tests prove from this list (and [`online_sweep_cases`], the same blocks
/// with online packing) that every path class of their model of both
/// kernels is taken (see the module docs for what the model leaves out).
///
/// Empty when `tile_rows` is 0; a block that cannot be formed (`nr < 2`) is
/// left out.
pub(super) fn sweep_cases(tile_rows: usize, nr: usize) -> impl Iterator<Item = Case> {
    let variants = MR.checked_div(tile_rows).unwrap_or(0);
    let per_variant = variants.saturating_mul(SWEEP_BLOCKS_PER_VARIANT);
    let middle = if variants == 0 {
        0
    } else {
        SWEEP_MIDDLE_BLOCKS
    };
    (0..per_variant.saturating_add(middle)).filter_map(move |index| {
        match index.checked_sub(per_variant) {
            None => sweep_case(tile_rows, nr, index),
            Some(middle_index) => middle_case(tile_rows, nr, middle_index),
        }
    })
}

/// Block `index` of the per-variant part of [`sweep_cases`]: the depth
/// varies fastest, then the layout, then the position, then the row variant.
fn sweep_case(tile_rows: usize, nr: usize, index: usize) -> Option<Case> {
    let depth_index = index % SWEEP_DEPTHS.len();
    let kc = *SWEEP_DEPTHS.get(depth_index)?;
    let rest = index / SWEEP_DEPTHS.len();
    let in_place = rest % 2 == 1;
    let later = (rest / 2) % 2 == 1;
    let tiles = rest / 4 + 1;
    let rows = if in_place == later {
        (tiles - 1).checked_mul(tile_rows)?.checked_add(1)?
    } else {
        tiles.checked_mul(tile_rows)?
    };
    let m = if later { rows.checked_add(MR)? } else { rows };
    // A partial last column panel of 1..nr - 1 columns, varied over blocks.
    let widths = nr.checked_sub(1).filter(|&widths| widths > 0)?;
    let partial = (5 * tiles + depth_index) % widths + 1;
    let (layout, lda_pad) = if in_place {
        (ALayout::InPlace, 1 + tiles % 3)
    } else {
        (ALayout::Packed, 0)
    };
    Some(Case {
        m,
        n: nr.checked_add(partial)?,
        kc,
        layout,
        lda_pad,
        b_pad: depth_index,
        data: Data::Plain,
        online: sweep_online(in_place),
    })
}

/// Block `index` (`0..SWEEP_MIDDLE_BLOCKS`) of the three-row-panel part of
/// [`sweep_cases`]: the depth varies fastest, then the layout (packed first).
///
/// The last row panel holds `L_k = 1 + k · (V - 1) / 5` live tiles of the
/// `V = 64 / tile_rows` (spread over `1..=V`). Packed, its last tile holds
/// `1 + k` rows (one row, then more); in place it is full, then
/// `k - 3` rows short. So no block repeats the last-panel class of a
/// per-variant block (packed and full, or in place and one row), and every
/// block takes classes that no other block takes.
fn middle_case(tile_rows: usize, nr: usize, index: usize) -> Option<Case> {
    if index >= SWEEP_MIDDLE_BLOCKS {
        return None;
    }
    let depth_index = index % SWEEP_DEPTHS.len();
    let kc = *SWEEP_DEPTHS.get(depth_index)?;
    let in_place = index / SWEEP_DEPTHS.len() == 1;
    let variants = MR.checked_div(tile_rows)?;
    let tiles = 1 + index * variants.checked_sub(1)? / (SWEEP_MIDDLE_BLOCKS - 1);
    let rows = if in_place {
        tiles
            .checked_mul(tile_rows)?
            .saturating_sub(index - SWEEP_DEPTHS.len())
            .max(1)
    } else {
        (tiles - 1)
            .checked_mul(tile_rows)?
            .checked_add(1 + index)?
            .min(MR)
    };
    let widths = nr.checked_sub(1).filter(|&widths| widths > 0)?;
    let partial = (3 * index + 1) % widths + 1;
    let (layout, lda_pad) = if in_place {
        (ALayout::InPlace, 1 + index % 3)
    } else {
        (ALayout::Packed, 0)
    };
    Some(Case {
        m: MR.checked_mul(2)?.checked_add(rows)?,
        n: nr.checked_add(partial)?,
        kc,
        layout,
        lda_pad,
        b_pad: depth_index,
        data: Data::Plain,
        online: sweep_online(in_place),
    })
}

/// The calls of a block of [`sweep_cases`]: `PREFETCH_C` with A in place,
/// neither field with packed A (so the prefetch block and its skip both
/// meet every row variant, position and column panel).
fn sweep_online(in_place: bool) -> Online {
    if in_place {
        Online::Prefetch
    } else {
        Online::Off
    }
}

/// The online sweep: every block of [`sweep_cases`] once more, with online
/// packing (and `PREFETCH_C` where that block has it: A in place). Column
/// panel 0 of each runs the packing twin of its row variant, column panel 1
/// (partial) the plain variant on the buffer, so the twins and the reads of
/// the buffer meet every class the sweep gives the plain variants (but the
/// column panel: column panel 0 is full, the buffer's is partial).
pub(super) fn online_sweep_cases(tile_rows: usize, nr: usize) -> impl Iterator<Item = Case> {
    sweep_cases(tile_rows, nr).map(|case| Case {
        online: if case.online == Online::Prefetch {
            Online::PackPrefetch
        } else {
            Online::Pack
        },
        ..case
    })
}

/// The fixed blocks with online packing, for a kernel with B panels of `nr`
/// columns (`nr ≥ 4`): the column panels that the sweeps do not give the
/// online paths, each with and without `PREFETCH_C`.
///
/// 1. `70 × (nr - 3) × 13`, packed: one partial column panel (a call with
///    `n ≤ NR` still fills the buffer; the switch is followed by the end of
///    the call), two row panels, odd `kc`.
/// 2. `9 × (nr - 1) × 37`, in place, special values (the buffer copies
///    signed zeros, subnormals and infinities bit for bit), prefetch: one
///    partial column panel, a deep odd `kc`.
/// 3. `84 × (2 · nr + 3) × 16`, in place with `lda = m + 3`, B panels 3
///    elements apart: the buffer read by a full and a partial column panel,
///    `kc` a multiple of 8.
/// 4. `130 × (2 · nr + 1) × 3`, packed, prefetch: the same over three row
///    panels, one pass of the double-step loop and the remainder.
pub(super) fn online_cases(nr: usize) -> [Case; 4] {
    let block = |m, n, kc, layout, lda_pad, b_pad, data, online| Case {
        m,
        n,
        kc,
        layout,
        lda_pad,
        b_pad,
        data,
        online,
    };
    let (narrow, wide) = (nr.saturating_sub(3).max(1), nr.saturating_mul(2));
    [
        block(
            70,
            narrow,
            13,
            ALayout::Packed,
            0,
            0,
            Data::Plain,
            Online::Pack,
        ),
        block(
            9,
            nr.saturating_sub(1).max(1),
            37,
            ALayout::InPlace,
            2,
            0,
            Data::Special,
            Online::PackPrefetch,
        ),
        block(
            84,
            wide + 3,
            16,
            ALayout::InPlace,
            3,
            3,
            Data::Plain,
            Online::Pack,
        ),
        block(
            130,
            wide + 1,
            3,
            ALayout::Packed,
            0,
            0,
            Data::Plain,
            Online::PackPrefetch,
        ),
    ]
}

/// Every block `self_test_with` runs for a kernel with ZA tiles of
/// `tile_rows` rows and B panels of `nr` columns, in order: [`CASES`],
/// [`online_cases`], [`sweep_cases`], then [`online_sweep_cases`].
pub(super) fn self_test_blocks(tile_rows: usize, nr: usize) -> impl Iterator<Item = Case> {
    CASES
        .into_iter()
        .chain(online_cases(nr))
        .chain(sweep_cases(tile_rows, nr))
        .chain(online_sweep_cases(tile_rows, nr))
}

/// The kernel calls of one `self_test_with` run for `T` (when every run
/// passes): every block of [`self_test_blocks`] with every pair of
/// [`SCALARS`].
#[cfg(test)]
pub(super) fn self_test_calls<T: SmeElement>() -> usize {
    self_test_blocks(T::TILE_ROWS, T::NR).count() * SCALARS.len()
}

/// The bytes one run of `case` allocates for `T`: A, B and C with its guard
/// border, a copy of A and of B, two copies of C (the kernel's and the
/// oracle's), and with online packing the kernel's and the oracle's buffer
/// with their guards. `None` for a block that `self_test_with` refuses.
#[cfg(test)]
pub(super) fn run_bytes<T: SmeElement>(case: &Case) -> Option<usize> {
    let [a, b, c, pack] = run_buffers::<T>(case)?;
    let elements = a
        .checked_add(b)?
        .checked_add(pack)?
        .checked_mul(2)?
        .checked_add(c.checked_mul(3)?)?;
    elements.checked_mul(size_of::<T>())
}

/// The elements of the four buffers of one run of `case` for `T`: A (its
/// padding included), B (whole panels and their gaps), C (with its guard
/// border) and the online buffer with its guards (0 without online
/// packing). Each is allocated once more as a copy (C twice). `None` for a
/// block that `self_test_with` refuses.
#[cfg(test)]
pub(super) fn run_buffers<T: SmeElement>(case: &Case) -> Option<[usize; 4]> {
    let shape = Shape::of::<T>(case)?;
    Some([shape.a_len, shape.b_len, shape.c_len, shape.pack.alloc])
}

/// Guard elements before the online buffer: two depth steps (`2 · MR`).
pub(super) const PACK_FRONT: usize = 2 * MR;

/// The online buffer of one run of a block (see [`pack_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackLayout {
    /// Guard elements before the buffer ([`PACK_FRONT`]; 0 without online
    /// packing).
    pub(super) front: usize,
    /// Elements the kernel writes: `ceil(m / 64) · 64 · kc`.
    pub(super) len: usize,
    /// Elements allocated: the front guard, the buffer and the guard after
    /// it.
    pub(super) alloc: usize,
}

/// The online buffer of one run of `case` for `T` (all zero without online
/// packing; `None` for a size that overflows): [`PACK_FRONT`] guard
/// elements, the `P = ceil(m / 64) · 64 · kc` elements of the packed copy,
/// and `max(N - 1, 1) · P + 2 · MR` guard elements after it for the
/// `N = ceil(n / NR)` column panels (see the module docs).
pub(super) fn pack_layout<T: SmeElement>(case: &Case) -> Option<PackLayout> {
    if !case.online.packs() {
        return Some(PackLayout {
            front: 0,
            len: 0,
            alloc: 0,
        });
    }
    let len = case.m.div_ceil(MR).checked_mul(MR)?.checked_mul(case.kc)?;
    let later = case.n.div_ceil(T::NR).saturating_sub(1).max(1);
    let back = later.checked_mul(len)?.checked_add(2 * MR)?;
    let alloc = PACK_FRONT.checked_add(len)?.checked_add(back)?;
    Some(PackLayout {
        front: PACK_FRONT,
        len,
        alloc,
    })
}

/// The C buffer of one run of a block (see [`c_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CLayout {
    /// Leading dimension: `m + 3`.
    pub(super) ldc: usize,
    /// Index of `C(0, 0)`: `ldc + 1` (a guard column and one element before).
    pub(super) origin: usize,
    /// Elements of the buffer.
    pub(super) len: usize,
}

/// The C buffer of one run of `case` for `T` (`None` for a size that
/// overflows): `ldc = m + 3`, `C(0, 0)` at `ldc + 1`, and a length that
/// reaches the farther of the end of column `n` (rows `0..ldc - 1`) and the
/// last element a whole `64 × NR` tile at any tile position of the block
/// addresses (see the module docs).
pub(super) fn c_layout<T: SmeElement>(case: &Case) -> Option<CLayout> {
    let ldc = case.m.checked_add(3)?;
    let origin = ldc.checked_add(1)?;
    // Offset one past the last guard element of column `n`.
    let column_end = case.n.checked_add(1)?.checked_mul(ldc)?.checked_sub(1)?;
    // Offset one past `C(64 · ceil(m / 64) - 1, NR · ceil(n / NR) - 1)`.
    let tile_rows = case.m.div_ceil(MR).checked_mul(MR)?;
    let tile_columns = case.n.div_ceil(T::NR).checked_mul(T::NR)?;
    let tile_end = tile_columns
        .checked_sub(1)?
        .checked_mul(ldc)?
        .checked_add(tile_rows)?;
    let len = origin.checked_add(column_end.max(tile_end))?;
    Some(CLayout { ldc, origin, len })
}

/// Runs `kernel` on every block of [`self_test_blocks`] ([`CASES`], then the
/// sweep of [`sweep_cases`]) with every pair of [`SCALARS`] and returns
/// `true` only when every run matches the oracle (see the module docs for
/// what is compared). `is_ok()` of [`self_test_with_detailed`]. Never
/// panics.
///
/// # Safety
///
/// The caller guarantees that `kernel` may run on this thread now, for
/// arguments that satisfy the kernel contract of `args.rs`: for a real SME2
/// kernel, `detect::thread_ready()` returned `true` on this thread (and
/// FEAT_SME_F64F64 is present for an `f64` kernel). The self-test supplies
/// valid blocks itself.
#[cfg(test)]
pub(super) unsafe fn self_test_with<T: SmeElement>(kernel: KernelFn<T>) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { self_test_with_detailed(kernel) }.is_ok()
}

/// `self_test_with` with the first mismatch: `Err` describes it (the block
/// is an index into [`self_test_blocks`]). Never panics.
///
/// # Safety
///
/// As for `self_test_with`.
pub(super) unsafe fn self_test_with_detailed<T: SmeElement>(
    kernel: KernelFn<T>,
) -> Result<(), Mismatch> {
    // SAFETY: forwarded from this function's contract.
    unsafe { run_cases(kernel, self_test_blocks(T::TILE_ROWS, T::NR), &SCALARS) }
}

/// Runs the first-use self-test on the real SME2 macro-kernels:
/// [`sme2_gemm_f32`] always, and [`sme2_gemm_f64`] when `f64_present` says
/// that the CPU has FEAT_SME_F64F64. Returns `true` only when every kernel it
/// ran matched the oracle on every block (see the module docs); a mismatch
/// returns `false`. `is_ok()` of [`run_self_test_detailed`]. Never panics.
///
/// It runs `self_test_calls` kernel calls per type (714 for `f32`, 1 290
/// for `f64`), and it calls nothing that runs or reads the detection
/// (`detect::hw()`, `detect::enabled()`, `detect::thread_ready()`,
/// `sme2_active()`) and no GEMM entry point, so the detection may call it
/// while it holds the lock of its first-use check. The calling thread keeps
/// the floating-point flags that the runs raise.
///
/// # Safety
///
/// The safety contract is the one of the kernels. The caller guarantees that
///
/// - the CPU has SME2 with a 512-bit streaming vector length, and
///   FEAT_SME_F64F64 when `f64_present` is `true`;
/// - just before the call, the calling thread was in the state that
///   `detect::thread_ready()` checks: not in streaming mode and no ZA storage
///   on (SVCR = 0), no pending lazy ZA save (TPIDR2_EL0 = 0), a 64-byte
///   streaming vector length. Outside the detection that is
///   `detect::thread_ready()` returning `true`; inside it (where calling
///   `thread_ready()` would re-enter the first-use check) the caller reads
///   the state with `detect::probe_thread_state()`.
///
/// Every kernel call leaves the thread in that state again (the frame of
/// `asm.rs` exits streaming mode with ZA off and touches neither TPIDR2_EL0
/// nor the vector length), so the state checked once before this call holds
/// for every kernel call it makes.
///
/// A thread that is not in that state must not call this function. Such a
/// thread says nothing about the kernels: the caller declines the backend
/// without a verdict, or retries later, and does not record a mismatch.
#[cfg(test)]
pub(super) unsafe fn run_self_test(f64_present: bool) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { run_self_test_detailed(f64_present) }.is_ok()
}

/// `run_self_test` with the first mismatch: `Err` describes it (its
/// element type says which kernel failed). Never panics. The first of the
/// detection's four first-use checks (`detect::run_checks`), run with
/// FPCR = 0 and the caller's FPCR and FPSR restored afterwards.
///
/// # Safety
///
/// As for `run_self_test`.
pub(super) unsafe fn run_self_test_detailed(f64_present: bool) -> Result<(), Mismatch> {
    let f32_kernel: KernelFn<f32> = sme2_gemm_f32;
    let f64_kernel: Option<KernelFn<f64>> = if f64_present {
        Some(sme2_gemm_f64)
    } else {
        None
    };
    // SAFETY: forwarded from this function's contract: SME2 is present and
    // the thread is ready for the `f32` kernel, and `f64_kernel` is `Some`
    // only when the caller guarantees FEAT_SME_F64F64 as well.
    unsafe { self_test_kernels_detailed(f32_kernel, f64_kernel) }
}

/// The verdict of `run_self_test` for given kernels: `self_test_with` on
/// `f32_kernel`, then on `f64_kernel` when it is `Some`; `true` only when
/// every kernel that ran passed. Stops at the first failure. `is_ok()` of
/// [`self_test_kernels_detailed`]. Never panics.
///
/// # Safety
///
/// As for `self_test_with`, for `f32_kernel` and, when it is `Some`,
/// `f64_kernel`.
#[cfg(test)]
pub(super) unsafe fn self_test_kernels(
    f32_kernel: KernelFn<f32>,
    f64_kernel: Option<KernelFn<f64>>,
) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { self_test_kernels_detailed(f32_kernel, f64_kernel) }.is_ok()
}

/// `self_test_kernels` with the first mismatch. Never panics.
///
/// # Safety
///
/// As for `self_test_kernels`.
pub(super) unsafe fn self_test_kernels_detailed(
    f32_kernel: KernelFn<f32>,
    f64_kernel: Option<KernelFn<f64>>,
) -> Result<(), Mismatch> {
    // SAFETY: forwarded from this function's contract.
    unsafe { self_test_with_detailed(f32_kernel) }?;
    match f64_kernel {
        // SAFETY: forwarded from this function's contract.
        Some(kernel) => unsafe { self_test_with_detailed(kernel) },
        None => Ok(()),
    }
}

/// `self_test_with` over chosen blocks and scalar pairs (used by tests to
/// run kernels that support only some shapes or modes). `true` only when
/// every run matches; an empty list is vacuously `true`. `is_ok()` of
/// `self_test_cases_detailed`.
///
/// # Safety
///
/// As for `self_test_with`, for every block of `cases` and every pair of
/// `scalars`.
#[cfg(test)]
pub(super) unsafe fn self_test_cases<T: SmeElement>(
    kernel: KernelFn<T>,
    cases: &[Case],
    scalars: &[(f64, f64)],
) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { self_test_cases_detailed(kernel, cases, scalars) }.is_ok()
}

/// `self_test_cases` with the first mismatch (its block and pair are
/// indices into `cases` and `scalars`).
///
/// # Safety
///
/// As for `self_test_cases`.
#[cfg(test)]
pub(super) unsafe fn self_test_cases_detailed<T: SmeElement>(
    kernel: KernelFn<T>,
    cases: &[Case],
    scalars: &[(f64, f64)],
) -> Result<(), Mismatch> {
    // SAFETY: forwarded from this function's contract.
    unsafe { run_cases(kernel, cases.iter().copied(), scalars) }
}

/// Runs `kernel` on every block of `cases` with every pair of `scalars`, in
/// that order (block `b`, pair `s` gets the data seed `b · 256 + s`); `Ok`
/// only when every run matches. Stops at the first mismatch.
///
/// # Safety
///
/// As for `self_test_with`, for every block of `cases` and every pair of
/// `scalars`.
unsafe fn run_cases<T: SmeElement>(
    kernel: KernelFn<T>,
    cases: impl Iterator<Item = Case>,
    scalars: &[(f64, f64)],
) -> Result<(), Mismatch> {
    let element_bits = if size_of::<T>() == size_of::<f32>() {
        32
    } else {
        64
    };
    for (case_index, case) in cases.enumerate() {
        for (pair_index, &(alpha, beta)) in scalars.iter().enumerate() {
            let seed = ((case_index as u64) << 8) | pair_index as u64;
            // SAFETY: forwarded from this function's contract.
            let run = unsafe { run_case(kernel, &case, to_element(alpha), to_element(beta), seed) };
            if let Err(difference) = run {
                return Err(Mismatch {
                    element_bits,
                    block: case_index,
                    case,
                    pair: pair_index,
                    alpha,
                    beta,
                    difference,
                });
            }
        }
    }
    Ok(())
}

/// `v` as a `T` (`f32` rounds to nearest). `SmeElement` is implemented by
/// `f32` and `f64` only, told apart by their size.
fn to_element<T: SmeElement>(v: f64) -> T {
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(u64::from((v as f32).to_bits()))
    } else {
        T::from_bits64(v.to_bits())
    }
}

/// A quiet NaN with the given payload bits for `T` (never the default NaN).
fn nan_with_payload<T: SmeElement>(payload: u32) -> T {
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(u64::from(0x7fc0_0000 | (payload & 0x003f_ffff)))
    } else {
        T::from_bits64(0x7ff8_0000_0000_0000 | u64::from(payload))
    }
}

/// A signalling NaN for `T`: the quiet bit clear and `payload` (non-zero,
/// below `2^22`) in the low mantissa bits. Any arithmetic on it (`x + 0`,
/// `fma(x, b, 0)`) returns a quiet NaN, whose bits differ.
fn signalling_nan<T: SmeElement>(payload: u32) -> T {
    let payload = (payload & 0x003f_ffff).max(1);
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(u64::from(0x7f80_0000 | payload))
    } else {
        T::from_bits64(0x7ff0_0000_0000_0000 | u64::from(payload))
    }
}

/// The guard sentinel around C: a signalling NaN with a payload that no
/// arithmetic produces (`f32` `0x7f9e_1f7e`, `f64` `0x7ff4_0000_005e_1f7e`;
/// quiet bit clear). It is signalling because a kernel that addresses one
/// element too many in M1 (`acc + C`) or M3 (`fma(C, beta, acc · alpha)`)
/// loads the guard, computes on it and stores the result: a quiet NaN would
/// come back with the same bits and the guard would look untouched, while a
/// signalling one comes back quiet.
pub(super) fn sentinel<T: SmeElement>() -> T {
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(0x7f9e_1f7e)
    } else {
        T::from_bits64(0x7ff4_0000_005e_1f7e)
    }
}

/// The spare rows of in-place A (rows `m..lda` of every column, and the tail
/// of the buffer): a signalling NaN of its own payload. A is compared bit for
/// bit after the run; a kernel that wrote an A element back after computing
/// on it (`x + 0`) would leave a quiet NaN unchanged but quiets this one.
pub(super) fn a_spare<T: SmeElement>() -> T {
    signalling_nan(0x0a_0a0a)
}

/// The gap between B panels (`b_pad` elements after `kc · NR`): a signalling
/// NaN of its own payload, for the same reason as [`a_spare`].
pub(super) fn b_gap<T: SmeElement>() -> T {
    signalling_nan(0x0b_0b0b)
}

/// The fill of C in modes that must not read it: another unusual NaN (quiet:
/// any use of it shows as a NaN in the block, which the comparison sees).
fn c_fill<T: SmeElement>() -> T {
    nan_with_payload(0x0f_111c)
}

/// The guards around the online A buffer: a signalling NaN of its own
/// payload (a correct packed copy never holds a NaN).
pub(super) fn pack_guard<T: SmeElement>() -> T {
    signalling_nan(0x0d_0d0d)
}

/// The stale contents of element `k` of the online buffer before a call: a
/// signalling NaN for even `k`, `-(1024 + k mod 4096)` for odd `k`. Neither
/// is a value of the packed copy (the generated values of A are smaller in
/// magnitude, the padding is `+0`), so an element the kernel leaves alone
/// differs from the oracle's, and a later column panel that reads the
/// buffer before it was written computes on a NaN or on a wrong number.
pub(super) fn pack_stale<T: SmeElement>(k: usize) -> T {
    if k.is_multiple_of(2) {
        signalling_nan(0x0e_0e0e)
    } else {
        to_element(-((1024 + k % 4096) as f64))
    }
}

/// A fixed xorshift sequence of test values.
struct Values {
    /// The generator state (never zero).
    state: u64,
    /// Which values to make.
    data: Data,
    /// `true` for `f32`.
    single: bool,
}

impl Values {
    /// A sequence for `seed` (operand `which` of one run).
    fn new(seed: u64, which: u64, data: Data, single: bool) -> Self {
        let state = (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ which.wrapping_mul(0xd1b5_4a32_d192_ed03))
            | 1;
        Self {
            state,
            data,
            single,
        }
    }

    /// The next raw 64 bits.
    fn bits(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// The next value as `f64` (rounded to `f32` by [`Values::next`] when
    /// `single`).
    fn next_f64(&mut self) -> f64 {
        let bits = self.bits();
        // A full-mantissa magnitude in [1, 2), with the sign from bit 0.
        let unit = f64::from_bits(0x3ff0_0000_0000_0000 | (bits >> 12));
        let signed = if bits & 1 == 0 { unit } else { -unit };
        let pick = (bits >> 1) & 63;
        match self.data {
            Data::Plain => match pick % 8 {
                0 => 0.0,
                1 | 2 => signed * 0.001_9,
                3 => signed * 37.0,
                _ => signed * 0.6,
            },
            Data::Special => {
                // Half the depth exponent of the smallest normal: products of
                // two such values are subnormal or underflow.
                let (half, sub) = if self.single {
                    (-64, -133)
                } else {
                    (-512, -1_035)
                };
                match pick {
                    0 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    2..=5 => -0.0,
                    6..=9 => 0.0,
                    10..=19 => signed * pow2(sub),
                    20..=49 => signed * pow2(half),
                    _ => signed * 0.75,
                }
            }
        }
    }

    /// The next value as a `T`.
    fn next<T: SmeElement>(&mut self) -> T {
        to_element(self.next_f64())
    }
}

/// `2^e` for `-1074 ≤ e ≤ 1023`, built from its bits (exact, subnormal
/// powers included; `powi` would overflow an intermediate `2^-e`).
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1_u64 << (e + 1074).clamp(0, 51))
    }
}

/// The index of the first element whose bits differ (or the shorter length
/// when the lengths differ); `None` when both hold the same bits. One pass.
fn first_change<T: SmeElement>(got: &[T], want: &[T]) -> Option<usize> {
    if got.len() != want.len() {
        return Some(got.len().min(want.len()));
    }
    got.iter().zip(want).position(|(g, w)| g.bits() != w.bits())
}

/// The sizes and strides of the operands of one run.
struct Shape {
    /// `a_depth` of the case.
    a_depth: usize,
    /// `a_panel` of the case.
    a_panel: usize,
    /// Elements of A.
    a_len: usize,
    /// `b_panel` of the case (`kc · NR + b_pad`).
    b_panel: usize,
    /// Elements of B (`ceil(n / NR)` panels).
    b_len: usize,
    /// Leading dimension of C (`m + 3`).
    ldc: usize,
    /// Index of `C(0, 0)` in C (`ldc + 1`).
    origin: usize,
    /// Elements of C with its guard border ([`c_layout`]).
    c_len: usize,
    /// The online buffer ([`pack_layout`]).
    pack: PackLayout,
}

impl Shape {
    /// The shape of `case` for `T`; `None` for a packed case with an lda pad
    /// or a size that overflows.
    fn of<T: SmeElement>(case: &Case) -> Option<Self> {
        let Case {
            m,
            n,
            kc,
            layout,
            lda_pad,
            b_pad,
            ..
        } = *case;
        let (a_depth, a_panel, a_len) = match layout {
            ALayout::Packed => {
                if lda_pad != 0 {
                    return None;
                }
                let panel = MR.checked_mul(kc)?;
                (MR, panel, m.div_ceil(MR).checked_mul(panel)?)
            }
            // The allocation is sized as if the row panels were `lda · kc`
            // apart, plus 64: a stray read of rows `m..64` of the last column,
            // or of a kernel that steps row panels as in the packed layout
            // (by `a_depth · kc` or `64 · kc ≤ lda · kc` for `m ≥ 64`), stays
            // inside it and reads a NaN (the contract forbids both, but the
            // self-test must not crash on a kernel that breaks it).
            ALayout::InPlace => {
                let lda = m.checked_add(lda_pad)?;
                let panels = m.div_ceil(MR).checked_mul(lda)?.checked_mul(kc)?;
                (lda, MR, panels.checked_add(MR)?)
            }
        };
        let b_panel = T::NR.checked_mul(kc)?.checked_add(b_pad)?;
        let b_len = n.div_ceil(T::NR).checked_mul(b_panel)?;
        let CLayout { ldc, origin, len } = c_layout::<T>(case)?;
        Some(Self {
            a_depth,
            a_panel,
            a_len,
            b_panel,
            b_len,
            ldc,
            origin,
            c_len: len,
            pack: pack_layout::<T>(case)?,
        })
    }
}

/// The operands of one run, laid out as `args.rs` describes.
struct Operands<T> {
    /// A in the layout of the case.
    a: Vec<T>,
    /// `a_depth` of the case.
    a_depth: usize,
    /// `a_panel` of the case.
    a_panel: usize,
    /// B packed in `NR`-column panels, zero-padded, `b_panel` apart.
    b: Vec<T>,
    /// `b_panel` of the case (`kc · NR + b_pad`).
    b_panel: usize,
    /// C with its guard border.
    c: Vec<T>,
    /// Leading dimension of C (`m + 3`).
    ldc: usize,
    /// Index of `C(0, 0)` in `c` (`ldc + 1`).
    origin: usize,
    /// The online buffer before the call: guards, stale contents, guards
    /// (empty without online packing).
    pack: Vec<T>,
    /// Its layout.
    pack_layout: PackLayout,
}

impl<T: SmeElement> Operands<T> {
    /// Generates the operands of `case` for the mode `mode`.
    fn new(case: &Case, mode: Epilogue, seed: u64) -> Option<Self> {
        let Case {
            m,
            n,
            kc,
            layout,
            data,
            ..
        } = *case;
        let single = size_of::<T>() == size_of::<f32>();
        let nr = T::NR;
        let Shape {
            a_depth,
            a_panel,
            a_len,
            b_panel,
            b_len,
            ldc,
            origin,
            c_len,
            pack: pack_layout,
        } = Shape::of::<T>(case)?;

        let mut a = match layout {
            ALayout::Packed => vec![T::ZERO; a_len],
            ALayout::InPlace => vec![a_spare(); a_len],
        };
        let mut values = Values::new(seed, 1, data, single);
        for p in 0..kc {
            for i in 0..m {
                let index = (i / MR) * a_panel + p * a_depth + i % MR;
                *a.get_mut(index)? = values.next();
            }
        }

        // Each panel: `kc · NR` elements (columns `n..` of the last one zero),
        // then a gap of `b_pad` signalling NaNs.
        let b_used = nr.checked_mul(kc)?;
        let mut b = vec![T::ZERO; b_len];
        for (index, value) in b.iter_mut().enumerate() {
            if index % b_panel >= b_used {
                *value = b_gap();
            }
        }
        let mut values = Values::new(seed, 2, data, single);
        for p in 0..kc {
            for j in 0..n {
                let index = (j / nr) * b_panel + p * nr + j % nr;
                *b.get_mut(index)? = values.next();
            }
        }

        let mut c = vec![sentinel(); c_len];
        let mut values = Values::new(seed, 3, data, single);
        for j in 0..n {
            for i in 0..m {
                let value = if mode.reads_c() {
                    values.next()
                } else {
                    c_fill()
                };
                *c.get_mut(origin + i + j * ldc)? = value;
            }
        }

        let PackLayout { front, len, alloc } = pack_layout;
        let pack = (0..alloc)
            .map(|index| match index.checked_sub(front) {
                Some(k) if k < len => pack_stale(k),
                _ => pack_guard(),
            })
            .collect();
        Some(Self {
            a,
            a_depth,
            a_panel,
            b,
            b_panel,
            c,
            ldc,
            origin,
            pack,
            pack_layout,
        })
    }

    /// The argument block of this run, with C at `c` (a copy of `self.c`)
    /// and, with online packing, the buffer in `pack` (a copy of
    /// `self.pack`) after its front guard.
    ///
    /// The pointers into A and B come from `as_mut_ptr`, so a kernel that
    /// writes them (a contract violation) changes defined memory, and the
    /// self-test sees it.
    fn args(
        &mut self,
        case: &Case,
        c: &mut [T],
        pack: &mut [T],
        alpha: T,
        beta: T,
    ) -> Option<MacroArgs<T>> {
        let origin = c.get_mut(self.origin..)?.as_mut_ptr();
        let a_pack = if case.online.packs() {
            pack.get_mut(self.pack_layout.front..)?.as_mut_ptr()
        } else {
            core::ptr::null_mut()
        };
        Some(MacroArgs {
            a: self.a.as_mut_ptr().cast_const(),
            a_depth: self.a_depth,
            a_panel: self.a_panel,
            b: self.b.as_mut_ptr().cast_const(),
            b_panel: self.b_panel,
            c: origin,
            ldc: self.ldc,
            m: case.m,
            n: case.n,
            kc: case.kc,
            alpha_bits: alpha.bits(),
            beta_bits: beta.bits(),
            mode: Epilogue::select(alpha, beta) as u64,
            a_pack,
            prefetch: case.online.prefetch(),
        })
    }
}

/// One run: `kernel` and the oracle on the same block, then the comparisons
/// of the module docs, in this order: A, B, the guards of the online buffer
/// and the guard border of C (each the kernel's, then the oracle's, element
/// by element), the packed copy in the buffer, the block. `Err` names the
/// first difference, or [`Difference::Refused`] for an impossible size.
///
/// # Safety
///
/// As for `self_test_with`.
unsafe fn run_case<T: SmeElement>(
    kernel: KernelFn<T>,
    case: &Case,
    alpha: T,
    beta: T,
    seed: u64,
) -> Result<(), Difference> {
    if case.m == 0 || case.n == 0 || case.kc == 0 {
        return Err(Difference::Refused);
    }
    let mode = Epilogue::select(alpha, beta);
    let mut operands = Operands::<T>::new(case, mode, seed).ok_or(Difference::Refused)?;
    let (a_before, b_before) = (operands.a.clone(), operands.b.clone());
    let mut got = operands.c.clone();
    let mut want = operands.c.clone();
    let mut want_pack = core::mem::take(&mut operands.pack);
    let mut got_pack = want_pack.clone();
    let got_args = operands
        .args(case, &mut got, &mut got_pack, alpha, beta)
        .ok_or(Difference::Refused)?;
    // SAFETY: the block satisfies the kernel contract: A holds `A(i, p)` for
    // `i < m`, `p < kc` at the layout's offsets (in place: `i + p · lda` with
    // a spare tail), B holds `ceil(n / NR)` panels `b_panel = kc · NR + b_pad`
    // apart, each with `kc` zero-padded rows of `NR` elements. C has
    // `C(0, 0)` at `ldc + 1` with `ldc = m + 3` and reaches at least the end
    // of column `n` (`c_layout`), so `C(i, j)` for `i < m`, `j < n` is
    // inside `got`, which overlaps neither A nor B; `m, n, kc ≥ 1`; the mode is
    // `Epilogue::select(alpha, beta)`. `prefetch` is `PREFETCH_C` or 0, and
    // `a_pack` is null or points `PACK_FRONT` elements into `got_pack`,
    // which holds `ceil(m / 64) · 64 · kc` more elements after it
    // (`pack_layout`) and overlaps none of A, B, C or the argument block.
    // That the kernel may run on this thread is this function's contract.
    unsafe { kernel(&got_args) };
    let want_args = operands
        .args(case, &mut want, &mut want_pack, alpha, beta)
        .ok_or(Difference::Refused)?;
    // SAFETY: the same block with C in `want` and the buffer in `want_pack`;
    // the oracle needs no hardware.
    unsafe { oracle_block(&want_args) };

    if let Some(index) = first_change(&operands.a, &a_before) {
        return Err(Difference::AChanged { index });
    }
    if let Some(index) = first_change(&operands.b, &b_before) {
        return Err(Difference::BChanged { index });
    }
    let (origin, ldc, pack) = (operands.origin, operands.ldc, operands.pack_layout);
    compare_pack_guards(&got_pack, &want_pack, pack)?;
    compare_c_guards(&got, &want, origin, ldc, case.m, case.n)?;
    compare_pack(&got_pack, &want_pack, pack, case.kc)?;
    compare_c(&got, &want, origin, ldc, case.m, case.n)
}

/// The position of buffer element `index` relative to `C(0, 0)` at `origin`
/// with leading dimension `ldc ≥ 1`: (row in `0..ldc`, column, signed
/// offset), with `offset = row + column · ldc` (see [`Difference`] for the
/// aliasing of the coordinates). The self-test's blocks are far smaller
/// than `i32::MAX` elements; a larger value saturates.
fn guard_position(index: usize, origin: usize, ldc: usize) -> (i32, i32, i64) {
    let offset = i64::try_from(index)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(origin).unwrap_or(i64::MAX));
    let ldc = i64::try_from(ldc).unwrap_or(i64::MAX).max(1);
    let to_i32 = |value: i64| i32::try_from(value).unwrap_or(i32::MAX);
    (
        to_i32(offset.rem_euclid(ldc)),
        to_i32(offset.div_euclid(ldc)),
        offset,
    )
}

/// `value` as a `u32` for a report (saturating; the self-test's blocks are
/// far smaller).
fn narrow(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Every guard element of the online buffer (the [`PackLayout::front`]
/// elements before it and everything after its [`PackLayout::len`]
/// elements) must hold [`pack_guard`]'s exact bits, first in `got` (the
/// kernel's), then in `want` (the oracle's), element by element. `Err` names
/// the first difference by its signed offset from the buffer's first
/// element.
fn compare_pack_guards<T: SmeElement>(
    got: &[T],
    want: &[T],
    pack: PackLayout,
) -> Result<(), Difference> {
    if got.len() != want.len() || got.len() != pack.alloc {
        return Err(Difference::Refused);
    }
    let guard = pack_guard::<T>().bits();
    let len = narrow(pack.len);
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        let inside = index
            .checked_sub(pack.front)
            .is_some_and(|offset| offset < pack.len);
        if inside {
            continue;
        }
        let offset = i64::try_from(index)
            .unwrap_or(i64::MAX)
            .saturating_sub(i64::try_from(pack.front).unwrap_or(i64::MAX));
        if g.bits() != guard {
            return Err(Difference::BufferGuard {
                offset,
                len,
                found: g.bits(),
            });
        }
        if w.bits() != guard {
            return Err(Difference::OracleBufferGuard {
                offset,
                len,
                found: w.bits(),
            });
        }
    }
    Ok(())
}

/// The packed copy in the kernel's online buffer must equal the oracle's
/// bit for bit (no NaN allowance: packing only moves bits, and `+0` against
/// `-0` in the padding matters). `Err` names the first element that differs
/// by its row and depth step.
fn compare_pack<T: SmeElement>(
    got: &[T],
    want: &[T],
    pack: PackLayout,
    kc: usize,
) -> Result<(), Difference> {
    let span = pack.front..pack.front + pack.len;
    let (Some(got), Some(want)) = (got.get(span.clone()), want.get(span)) else {
        return Err(Difference::Refused);
    };
    let Some(index) = first_change(got, want) else {
        return Ok(());
    };
    let (Some(g), Some(w)) = (got.get(index), want.get(index)) else {
        return Err(Difference::Refused);
    };
    let panel = MR.saturating_mul(kc).max(1);
    Err(Difference::BufferElement {
        row: narrow((index / panel) * MR + index % MR),
        depth: narrow((index % panel) / MR),
        found: g.bits(),
        wanted: w.bits(),
    })
}

/// Every element outside the interior of C (`m × n` from `origin`, leading
/// dimension `ldc`) must hold the sentinel's exact bits, first in `got`
/// (the kernel's), then in `want` (the oracle's), element by element. `Err`
/// names the first difference.
fn compare_c_guards<T: SmeElement>(
    got: &[T],
    want: &[T],
    origin: usize,
    ldc: usize,
    m: usize,
    n: usize,
) -> Result<(), Difference> {
    if got.len() != want.len() || ldc == 0 || ldc < m {
        return Err(Difference::Refused);
    }
    let guard = sentinel::<T>().bits();
    let ldc_report = narrow(ldc);
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        let inside = index
            .checked_sub(origin)
            .is_some_and(|offset| offset % ldc < m && offset / ldc < n);
        if inside {
            continue;
        }
        if g.bits() != guard {
            let (row, col, offset) = guard_position(index, origin, ldc);
            return Err(Difference::KernelGuard {
                row,
                col,
                offset,
                ldc: ldc_report,
                found: g.bits(),
            });
        }
        if w.bits() != guard {
            let (row, col, offset) = guard_position(index, origin, ldc);
            return Err(Difference::OracleGuard {
                row,
                col,
                offset,
                ldc: ldc_report,
                found: w.bits(),
            });
        }
    }
    Ok(())
}

/// Every element of the interior of C (`m × n` from `origin`, leading
/// dimension `ldc`) must match with [`same_bits_or_nan`]. `Err` names the
/// first difference.
fn compare_c<T: SmeElement>(
    got: &[T],
    want: &[T],
    origin: usize,
    ldc: usize,
    m: usize,
    n: usize,
) -> Result<(), Difference> {
    for col in 0..n {
        for row in 0..m {
            let index = origin + row + col * ldc;
            let (Some(g), Some(w)) = (got.get(index), want.get(index)) else {
                return Err(Difference::Refused);
            };
            if !same_bits_or_nan(core::slice::from_ref(g), core::slice::from_ref(w)) {
                return Err(Difference::Element {
                    row: narrow(row),
                    col: narrow(col),
                    found: g.bits(),
                    wanted: w.bits(),
                });
            }
        }
    }
    Ok(())
}
