//! Tests of the ZA packers (`pack_za.rs`) that run the transposers on the
//! hardware: bitwise equality with the plain packers of `pack.rs`, bounds,
//! special values, flags and thread state, the entry points, adjacent
//! operands and the first-use check; and the cases and the shared runner of
//! every ZA packer test. The plain-Rust tests (registry, lint, the
//! structure of the transposer templates, the interpreter of the registered
//! text, delegation, the `None` cases, the first-use check with stand-ins)
//! are in `tests_pack_za_native.rs`.
//!
//! Every test that executes a transposer on the hardware starts with
//! `hardware_or_skip`: on a host without SME2 (the M3) it prints a skip
//! line, under QEMU with `-r` it must run. The expected buffer of every case
//! is the plain packer's (`pack::pack_a` / `pack::pack_b`), compared bit for
//! bit (`to_bits`, never `==`), so NaN payloads, signed zeros and subnormals
//! count.
//!
//! Each case builds a stored matrix around the block: the block starts at a
//! stored row and column `> 0` (or at `(0, 0)` with `ld` equal to the
//! block's rows), the elements outside it are signalling NaNs with distinct
//! payloads, and the elements inside it follow `pack_za::check_bits` (bits
//! that encode the position, mixed with signalling and quiet NaNs with
//! distinct payloads, `-0`, subnormals and infinities). The packed buffer is
//! pre-filled with stale `-0` and signalling NaNs (which must not survive,
//! in the padding or anywhere) and continues with a guard of signalling NaNs
//! that covers every address a whole tile group can reach, which must stay
//! bitwise untouched.

use std::cell::RefCell;

use oxiblas_core::memory::AlignedVec;
use oxiblas_matrix::MatRef;

use super::args::SmeElement;
use super::pack::{self, ColMajor, Span, a_pack_len, b_pack_len};
use super::pack_za::model::Window;
use super::pack_za::{
    GROUP_COLS, SmePack, ZaKernelFn, ZaPackArgs, check_bits, pack_a_za, pack_a_za_with, pack_b_za,
    pack_b_za_with, self_check, za_transpose_f32, za_transpose_f64,
};
use super::test_support::{
    FPSR_DZC, FPSR_IOC, FPSR_IXC, FPSR_QC, Need, fpsr_read, fpsr_write, hardware_or_skip,
    kernel_entries_here, sme_state,
};
use crate::level3::trsm::Trans;

/// `Err(format!(..))` unless the condition holds.
macro_rules! ensure {
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Err(format!($($why)+));
        }
    };
}
pub(super) use ensure;

// ---------------------------------------------------------------------------
// Cases and the shared runner
// ---------------------------------------------------------------------------

/// One packing request, in op coordinates, with the stored matrix around it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Case {
    /// `true`: A side (`pack_a_za`), `false`: B side (`pack_b_za`).
    a_side: bool,
    /// Whether the request goes through the transposer (`op(B) = B`,
    /// `op(A) = Aᵀ`) or is delegated (`op(B) = Bᵀ`, `op(A) = A`).
    za: bool,
    /// Depth of the block.
    depth: usize,
    /// Columns (B) or rows (A) of the block.
    width: usize,
    /// Stored row of the block's first element.
    r0: usize,
    /// Stored column of the block's first element.
    c0: usize,
    /// Stored rows after the block.
    tail: usize,
    /// Leading dimension minus the stored rows (`ld = rows + gap`).
    gap: usize,
}

impl Case {
    /// The B side through the transposer, the block in the middle of a
    /// larger matrix (`middle`) or the whole matrix with `ld = depth`.
    pub(super) fn b(depth: usize, width: usize, middle: bool) -> Self {
        Self::new(false, true, depth, width, middle)
    }

    /// The A side through the transposer.
    pub(super) fn a(depth: usize, width: usize, middle: bool) -> Self {
        Self::new(true, true, depth, width, middle)
    }

    pub(super) fn new(a_side: bool, za: bool, depth: usize, width: usize, middle: bool) -> Self {
        let (r0, c0, tail, gap) = if middle { (3, 2, 2, 3) } else { (0, 0, 0, 0) };
        Self {
            a_side,
            za,
            depth,
            width,
            r0,
            c0,
            tail,
            gap,
        }
    }

    /// Rows and columns of the stored block.
    fn stored_block(&self) -> (usize, usize) {
        if self.za {
            (self.depth, self.width)
        } else {
            (self.width, self.depth)
        }
    }

    /// Stored rows, leading dimension and stored columns of the matrix.
    fn stored_dims(&self) -> (usize, usize, usize) {
        let (rows, cols) = self.stored_block();
        let srows = self.r0 + rows + self.tail;
        (srows, srows + self.gap, self.c0 + cols + 1)
    }

    /// The transpose that makes `op(X)` of the stored matrix.
    fn trans(&self) -> Trans {
        match (self.a_side, self.za) {
            (false, true) | (true, false) => Trans::NoTrans,
            (false, false) | (true, true) => Trans::Trans,
        }
    }

    /// The depth span and the width span in op coordinates.
    fn spans(&self) -> (Span, Span) {
        let (depth0, width0) = if self.za {
            (self.r0, self.c0)
        } else {
            (self.c0, self.r0)
        };
        (
            Span {
                start: depth0,
                len: self.depth,
            },
            Span {
                start: width0,
                len: self.width,
            },
        )
    }

    /// The packed length of the block.
    fn len<T: SmeElement>(&self) -> usize {
        let len = if self.a_side {
            a_pack_len(self.width, self.depth)
        } else {
            b_pack_len::<T>(self.depth, self.width)
        };
        len.unwrap_or(usize::MAX)
    }

    /// Guard elements after the packed length: more than any tile group
    /// can reach (64 columns of `depth` rows, plus a packed row).
    fn guard(&self) -> usize {
        GROUP_COLS * (self.depth + 1)
    }
}

/// The stored matrix of a case: `check_bits` inside the block, signalling
/// NaNs with distinct payloads everywhere else (the gap rows included).
fn source<T: SmeElement>(case: &Case, bits: fn(usize, bool) -> u64) -> AlignedVec<T> {
    let (_, ld, scols) = case.stored_dims();
    let (rows, cols) = case.stored_block();
    let mut data = AlignedVec::<T>::with_capacity(ld * scols);
    for index in 0..ld * scols {
        let (r, c) = (index % ld, index / ld);
        let inside =
            (case.r0..case.r0 + rows).contains(&r) && (case.c0..case.c0 + cols).contains(&c);
        data.push(T::from_bits64(bits(index, inside)));
    }
    data
}

/// The stale value at `index` of a pre-filled buffer: alternately `-0` and
/// a signalling NaN.
pub(super) fn stale_bits<T: SmeElement>(index: usize) -> u64 {
    if index.is_multiple_of(2) {
        check_bits::<T>(5, true)
    } else {
        check_bits::<T>(index + 7, false)
    }
}

/// The guard value at `index` (beyond the packed length).
fn guard_bits<T: SmeElement>(index: usize) -> u64 {
    check_bits::<T>(index + 1_000_003, false)
}

/// A buffer of `len` elements pre-filled with [`stale_bits`].
fn stale_buffer<T: SmeElement>(len: usize) -> AlignedVec<T> {
    let mut buf = AlignedVec::<T>::with_capacity(len);
    (0..len).for_each(|i| buf.push(T::from_bits64(stale_bits::<T>(i))));
    buf
}

/// Whether the stale-filled `got` holds `want`'s first `packed` elements bit
/// for bit and is untouched after them (`packed = None`: untouched).
fn same_packed<T: SmeElement>(
    got: &AlignedVec<T>,
    want: &AlignedVec<T>,
    packed: Option<usize>,
) -> bool {
    let len = packed.unwrap_or(0);
    got.as_slice().iter().enumerate().all(|(i, g)| {
        let w = if i < len {
            want.as_slice().get(i).map(|w| w.bits())
        } else {
            Some(stale_bits::<T>(i))
        };
        w == Some(g.bits())
    })
}

thread_local! {
    /// The window of the current call (interpreter only).
    pub(super) static WINDOW: RefCell<Option<Window>> = const { RefCell::new(None) };
    /// What the interpreter (or a stand-in) reported during the current call.
    pub(super) static REPORT: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Notes a finding of a stand-in kernel for the current case.
pub(super) fn note(finding: String) {
    REPORT.with(|report| report.borrow_mut().push(finding));
}

/// Runs one case through `kernel` (the hardware transposer, the
/// interpreter, or a stand-in) and compares with the plain packer.
///
/// # Safety
///
/// `kernel` must be callable on this thread (the real kernels: SME2).
pub(super) unsafe fn run_case<T: SmePack>(
    case: &Case,
    kernel: ZaKernelFn<T>,
) -> Result<(), String> {
    let (srows, ld, scols) = case.stored_dims();
    let data = source::<T>(case, check_bits::<T>);
    let view = MatRef::from_strided(data.as_slice(), srows, scols, ld, 1)
        .ok_or_else(|| format!("{case:?}: no view"))?;
    let op = ColMajor::normalize(view, case.trans()).ok_or_else(|| format!("{case:?}: no op"))?;
    let (depth, width) = case.spans();
    let len = case.len::<T>();
    let mut want = AlignedVec::<T>::zeros(len);
    let plain = if case.a_side {
        pack::pack_a(&op, width, depth, &mut want).is_some()
    } else {
        pack::pack_b(&op, depth, width, &mut want).is_some()
    };
    ensure!(plain, "{case:?}: the plain packer declined");

    let total = len + case.guard();
    let mut got = AlignedVec::<T>::with_capacity(total);
    for index in 0..total {
        let bits = if index < len {
            stale_bits::<T>(index)
        } else {
            guard_bits::<T>(index)
        };
        got.push(T::from_bits64(bits));
    }
    let (rows, cols) = case.stored_block();
    let window = Window {
        src: view.ptr_at(case.r0, case.c0) as usize,
        ld,
        rows,
        cols,
        esize: core::mem::size_of::<T>(),
        dst: got.as_ptr() as usize,
        dst_len: len,
    };
    WINDOW.with(|slot| *slot.borrow_mut() = Some(window));
    REPORT.with(|report| report.borrow_mut().clear());
    // SAFETY: forwarded from this function's contract; the spans lie inside
    // op(X) and `got` is longer than the packed length.
    let packed = unsafe {
        if case.a_side {
            pack_a_za_with(kernel, &op, width, depth, &mut got).map(|p| p.data().len())
        } else {
            pack_b_za_with(kernel, &op, depth, width, &mut got).map(|p| p.data().len())
        }
    };
    WINDOW.with(|slot| *slot.borrow_mut() = None);
    let findings = REPORT.with(|report| core::mem::take(&mut *report.borrow_mut()));
    ensure!(findings.is_empty(), "{case:?}: {}", findings.join("; "));
    ensure!(
        packed == Some(len),
        "{case:?}: packed {packed:?}, want Some({len})"
    );
    for (index, (g, w)) in got.as_slice().iter().zip(want.as_slice()).enumerate() {
        ensure!(
            g.bits() == w.bits(),
            "{case:?}: element {index}: {:#x}, want {:#x}",
            g.bits(),
            w.bits()
        );
    }
    for (index, g) in got.as_slice().iter().enumerate().skip(len) {
        ensure!(
            g.bits() == guard_bits::<T>(index),
            "{case:?}: guard element {index} (packed length {len}) was written"
        );
    }
    Ok(())
}

/// Depths of the grid: around 1, both chunk heights (8 for `f64`, 16 for
/// `f32`: one below, at and one above each), the group of 64, two groups,
/// and one deep block. `test_sme2_pack_za_grid_reaches_every_edge` checks
/// that the last chunk of some depth stores 1, 2, `TR - 1` and `TR` rows.
pub(super) const DEPTHS: [usize; 15] = [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257];

/// Widths of the grid: `1 ..= NR + 1`, one below, at and one above every
/// multiple of `NR` (a tile, and a B panel) in the first and the second
/// group of 64, and 130 (three groups). So every tile-store threshold of
/// both kernels is met by some group (checked by
/// `test_sme2_pack_za_grid_reaches_every_edge`).
pub(super) fn widths<T: SmeElement>() -> Vec<usize> {
    let mut widths: Vec<usize> = (1..=T::NR + 1).collect();
    for base in [0, GROUP_COLS] {
        for k in 1..=GROUP_COLS / T::NR {
            let edge = base + k * T::NR;
            widths.extend([edge - 1, edge, edge + 1]);
        }
    }
    widths.push(2 * GROUP_COLS + 2);
    widths.sort_unstable();
    widths.dedup();
    widths
}

/// Every grid case of one side through the transposer, with the block in
/// the middle of a larger matrix and as the whole matrix (`ld = depth`).
pub(super) fn grid<T: SmeElement>(a_side: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    for depth in DEPTHS {
        for width in widths::<T>() {
            for middle in [false, true] {
                cases.push(Case::new(a_side, true, depth, width, middle));
            }
        }
    }
    cases
}

/// Runs every case and reports the failures (at most a few, with a count).
///
/// # Safety
///
/// As for [`run_case`].
pub(super) unsafe fn run_all<T: SmePack>(
    cases: &[Case],
    kernel: ZaKernelFn<T>,
) -> Result<(), String> {
    let failures: Vec<String> = cases
        .iter()
        // SAFETY: forwarded from this function's contract.
        .filter_map(|case| unsafe { run_case(case, kernel) }.err())
        .collect();
    ensure!(
        failures.is_empty(),
        "{} of {} cases failed, e.g.:\n{}",
        failures.len(),
        cases.len(),
        failures
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Hardware tests (SME2; under QEMU with -r they must run)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_pack_za_b_f32_grid() {
    if !hardware_or_skip("test_sme2_pack_za_b_f32_grid", Need::F32) {
        return;
    }
    // SAFETY: SME2 is present (the gate) and this thread is ready.
    if let Err(why) = unsafe { run_all::<f32>(&grid::<f32>(false), za_transpose_f32) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_a_f32_grid() {
    if !hardware_or_skip("test_sme2_pack_za_a_f32_grid", Need::F32) {
        return;
    }
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f32>(&grid::<f32>(true), za_transpose_f32) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_b_f64_grid() {
    if !hardware_or_skip("test_sme2_pack_za_b_f64_grid", Need::F64) {
        return;
    }
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f64>(&grid::<f64>(false), za_transpose_f64) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_a_f64_grid() {
    if !hardware_or_skip("test_sme2_pack_za_a_f64_grid", Need::F64) {
        return;
    }
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f64>(&grid::<f64>(true), za_transpose_f64) } {
        panic!("{why}");
    }
}

/// Bits of a block made only of special values: signalling and quiet NaNs
/// with distinct payloads and both signs, `±0`, subnormals, `±∞`, the
/// largest finite values.
fn special_bits<T: SmeElement>(index: usize, in_block: bool) -> u64 {
    if !in_block {
        return check_bits::<T>(index, false);
    }
    let i = index as u64;
    let wide = core::mem::size_of::<T>() == 8;
    let (sign, exp, quiet, frac) = if wide {
        (1 << 63, 0x7ff0_0000_0000_0000_u64, 1 << 51, (1 << 52) - 1)
    } else {
        (1 << 31, 0x7f80_0000_u64, 1 << 22, (1 << 23) - 1)
    };
    let payload = 1 + (i % (quiet - 1));
    let s = if index.is_multiple_of(2) { 0 } else { sign };
    match index % 7 {
        0 => s | exp | payload,
        1 => s | exp | quiet | payload,
        2 => s,
        3 => s | (payload & frac),
        4 => s | exp,
        5 => s | (exp - (frac + 1)) | frac,
        _ => exp | (i & (quiet - 1)) | 1,
    }
}

/// [`run_case`] with the special-value source (the plain packer is the
/// reference, so the comparison is bitwise as always).
///
/// # Safety
///
/// As for [`run_case`].
unsafe fn run_special<T: SmePack>(case: &Case) -> Result<(), String> {
    let (srows, ld, scols) = case.stored_dims();
    let data = source::<T>(case, special_bits::<T>);
    let view = MatRef::from_strided(data.as_slice(), srows, scols, ld, 1).ok_or("no view")?;
    let op = ColMajor::normalize(view, case.trans()).ok_or("no op")?;
    let (depth, width) = case.spans();
    let len = case.len::<T>();
    let mut want = AlignedVec::<T>::zeros(len);
    // Stale contents (`-0`, signalling NaNs): the padding must be written.
    let mut got = stale_buffer::<T>(len);
    // SAFETY: forwarded from this function's contract.
    let (plain, za) = unsafe {
        if case.a_side {
            (
                pack::pack_a(&op, width, depth, &mut want).is_some(),
                pack_a_za(&op, width, depth, &mut got).is_some(),
            )
        } else {
            (
                pack::pack_b(&op, depth, width, &mut want).is_some(),
                pack_b_za(&op, depth, width, &mut got).is_some(),
            )
        }
    };
    ensure!(plain && za, "{case:?}: declined");
    for (index, (g, w)) in got.as_slice().iter().zip(want.as_slice()).enumerate() {
        ensure!(
            g.bits() == w.bits(),
            "{case:?}: element {index}: {:#x}, want {:#x}",
            g.bits(),
            w.bits()
        );
    }
    // Every special value of the block arrived (not only the padding).
    let (rows, cols) = case.stored_block();
    let block: Vec<u64> = (0..cols)
        .flat_map(|c| (0..rows).map(move |r| (r + case.r0) + (c + case.c0) * ld))
        .map(|index| special_bits::<T>(index, true))
        .collect();
    let block: std::collections::HashSet<u64> = block.into_iter().collect();
    let foreign = got
        .as_slice()
        .iter()
        .filter(|v| v.bits() != 0 && !block.contains(&v.bits()))
        .count();
    ensure!(
        foreign == 0,
        "{case:?}: {foreign} values from outside the block reached the buffer"
    );
    Ok(())
}

#[test]
fn test_sme2_pack_za_special_values() {
    if !hardware_or_skip("test_sme2_pack_za_special_values", Need::F64) {
        return;
    }
    for (depth, width) in [(1, 1), (17, 19), (64, 64), (9, 130), (33, 65)] {
        for case in [Case::b(depth, width, true), Case::a(depth, width, true)] {
            // SAFETY: SME2 with f64 is present (the gate).
            unsafe {
                if let Err(why) = run_special::<f32>(&case) {
                    panic!("f32: {why}");
                }
                if let Err(why) = run_special::<f64>(&case) {
                    panic!("f64: {why}");
                }
            }
        }
    }
}

#[test]
fn test_sme2_pack_za_fpsr_and_thread_state() {
    if !hardware_or_skip("test_sme2_pack_za_fpsr_and_thread_state", Need::F64) {
        return;
    }
    let saved = fpsr_read();
    let cases = [Case::b(17, 35, true), Case::a(9, 70, true)];
    for initial in [0, FPSR_IXC | FPSR_DZC, FPSR_IOC | FPSR_QC] {
        for case in &cases {
            fpsr_write(initial);
            // SAFETY: SME2 with f64 is present (the gate).
            let results = unsafe {
                [
                    run_case::<f32>(case, za_transpose_f32),
                    run_case::<f64>(case, za_transpose_f64),
                ]
            };
            let after = fpsr_read();
            fpsr_write(saved);
            for result in results {
                if let Err(why) = result {
                    panic!("{why}");
                }
            }
            // The sources hold signalling NaNs: moved, never computed with.
            assert_eq!(
                after, initial,
                "{case:?}: FPSR moved from {initial:#x} to {after:#x}"
            );
            assert_eq!(
                sme_state(),
                Some((0, 0)),
                "{case:?}: the thread left streaming state behind"
            );
        }
    }
}

#[test]
fn test_sme2_pack_za_entry_points_and_kernel_counter() {
    if !hardware_or_skip(
        "test_sme2_pack_za_entry_points_and_kernel_counter",
        Need::F64,
    ) {
        return;
    }
    // The public entry points (not only the `_with` forms) give the plain
    // packers' buffers, and every ZA call enters a kernel once.
    let data: Vec<f64> = (0..50 * 40)
        .map(|i| f64::from_bits(check_bits::<f64>(i, true)))
        .collect();
    let Some(view) = MatRef::from_strided(&data, 50, 40, 50, 1) else {
        panic!("no view");
    };
    let (Some(b), Some(at)) = (
        ColMajor::normalize(view, Trans::NoTrans),
        ColMajor::normalize(view, Trans::Trans),
    ) else {
        panic!("no op");
    };
    let (depth, cols) = (Span { start: 3, len: 41 }, Span { start: 5, len: 29 });
    let mut got = stale_buffer::<f64>(4096);
    let mut want = AlignedVec::<f64>::zeros(4096);
    let before = kernel_entries_here();
    // SAFETY: SME2 with f64 is present (the gate).
    let za = unsafe {
        pack_b_za(&b, depth, cols, &mut got).map(|p| (p.depth(), p.cols(), p.data().len()))
    };
    let plain =
        pack::pack_b(&b, depth, cols, &mut want).map(|p| (p.depth(), p.cols(), p.data().len()));
    assert_eq!(za, plain);
    assert_eq!(
        kernel_entries_here(),
        before + 1,
        "pack_b_za must enter its kernel exactly once"
    );
    assert!(same_packed(&got, &want, za.map(|z| z.2)));
    let mut got = stale_buffer::<f64>(4096);
    let rows = Span { start: 1, len: 33 };
    let depth = Span { start: 2, len: 47 };
    let before = kernel_entries_here();
    // SAFETY: as above.
    let za = unsafe {
        pack_a_za(&at, rows, depth, &mut got).map(|p| (p.rows(), p.depth(), p.data().len()))
    };
    let plain =
        pack::pack_a(&at, rows, depth, &mut want).map(|p| (p.rows(), p.depth(), p.data().len()));
    assert_eq!(za, plain);
    assert_eq!(
        kernel_entries_here(),
        before + 1,
        "pack_a_za must enter its kernel exactly once"
    );
    assert!(same_packed(&got, &want, za.map(|z| z.2)));
    assert_eq!(sme_state(), Some((0, 0)));
    // The f32 side through the public entry point as well.
    let small: Vec<f32> = (0..20 * 70)
        .map(|i| f32::from_bits(check_bits::<f32>(i, true) as u32))
        .collect();
    let Some(view) = MatRef::from_strided(&small, 20, 70, 20, 1) else {
        panic!("no view");
    };
    let Some(b) = ColMajor::normalize(view, Trans::NoTrans) else {
        panic!("no op");
    };
    let mut got = stale_buffer::<f32>(4096);
    let mut want = AlignedVec::<f32>::zeros(4096);
    let (depth, cols) = (Span { start: 0, len: 20 }, Span { start: 0, len: 70 });
    // SAFETY: as above.
    let za = unsafe { pack_b_za(&b, depth, cols, &mut got).map(|p| p.data().len()) };
    assert_eq!(
        za,
        pack::pack_b(&b, depth, cols, &mut want).map(|p| p.data().len())
    );
    assert!(same_packed(&got, &want, za));
}

#[test]
fn test_sme2_pack_za_single_row_and_column_views() {
    if !hardware_or_skip("test_sme2_pack_za_single_row_and_column_views", Need::F32) {
        return;
    }
    // Normalised single rows and columns: the stored leading dimension can
    // be anything (here 1, 3 and 0 through a 1-row stored matrix).
    let data: Vec<f32> = (0..400)
        .map(|i| f32::from_bits(check_bits::<f32>(i, true) as u32))
        .collect();
    for (rows, cols, row_stride, col_stride) in
        [(1, 37, 3, 1), (1, 70, 1, 1), (29, 1, 1, 5), (1, 20, 0, 1)]
    {
        let Some(view) = MatRef::from_strided(&data, rows, cols, row_stride, col_stride) else {
            panic!("no view {rows}x{cols}");
        };
        for trans in [Trans::NoTrans, Trans::Trans] {
            let Some(op) = ColMajor::normalize(view, trans) else {
                panic!("no op");
            };
            let (depth, width) = (
                Span {
                    start: 0,
                    len: op.rows(),
                },
                Span {
                    start: 0,
                    len: op.cols(),
                },
            );
            let mut got = stale_buffer::<f32>(8192);
            let mut want = AlignedVec::<f32>::zeros(8192);
            // SAFETY: SME2 is present (the gate).
            let za = unsafe { pack_b_za(&op, depth, width, &mut got).map(|p| p.data().len()) };
            assert_eq!(
                za,
                pack::pack_b(&op, depth, width, &mut want).map(|p| p.data().len())
            );
            assert!(same_packed(&got, &want, za), "B {rows}x{cols} {trans:?}");
            let mut got = stale_buffer::<f32>(8192);
            let (rows_span, depth_span) = (
                Span {
                    start: 0,
                    len: op.rows(),
                },
                Span {
                    start: 0,
                    len: op.cols(),
                },
            );
            // SAFETY: as above.
            let za =
                unsafe { pack_a_za(&op, rows_span, depth_span, &mut got).map(|p| p.data().len()) };
            assert_eq!(
                za,
                pack::pack_a(&op, rows_span, depth_span, &mut want).map(|p| p.data().len())
            );
            assert!(same_packed(&got, &want, za), "A {rows}x{cols} {trans:?}");
        }
    }
}

#[test]
fn test_sme2_pack_za_self_check_passes() {
    if !hardware_or_skip("test_sme2_pack_za_self_check_passes", Need::F64) {
        return;
    }
    let saved = fpsr_read();
    fpsr_write(0);
    // SAFETY: SME2 with f64 is present (the gate).
    let (both, f32_only) = unsafe { (self_check(true), self_check(false)) };
    let after = fpsr_read();
    fpsr_write(saved);
    assert!(
        both && f32_only,
        "the ZA transposers fail their first-use check"
    );
    assert_eq!(after, 0, "the first-use check raised floating-point flags");
    assert_eq!(sme_state(), Some((0, 0)));
}

/// One block of [`test_sme2_pack_za_adjacent_operands`]: the stored matrix
/// (block at stored `(1, 1)`, `ld = depth + 1 + ld_pad`) and the packed
/// buffer in ONE allocation, `src_first` deciding their order, with
/// signalling-NaN guards of a whole group's reach before, between and after
/// them. The transposer is called directly (the entry points always write
/// from the start of their own buffer, so only this shows a write BEFORE
/// the buffer); afterwards the buffer must be the plain packer's and every
/// other element of the allocation, the source included, unchanged.
///
/// # Safety
///
/// SME2 for `T` is present and this thread is ready.
unsafe fn run_adjacent<T: SmePack>(
    a_side: bool,
    (depth, width, ld_pad): (usize, usize, usize),
    src_first: bool,
) -> Result<(), String> {
    let case = format!("{a_side} {depth}x{width} pad {ld_pad} src_first {src_first}");
    let (ld, scols) = (depth + 1 + ld_pad, width + 2);
    let (src_len, guard) = (ld * scols, GROUP_COLS * (depth + 1));
    let len = if a_side {
        a_pack_len(width, depth)
    } else {
        b_pack_len::<T>(depth, width)
    };
    let len = len.ok_or("length")?;
    let (src_at, dst_at) = if src_first {
        (guard, 2 * guard + src_len)
    } else {
        (2 * guard + len, guard)
    };
    let total = 3 * guard + src_len + len;
    let block = |i: usize| {
        let (r, c) = ((i - src_at) % ld, (i - src_at) / ld);
        (1..=depth).contains(&r) && (1..=width).contains(&c)
    };
    let orig: Vec<u64> = (0..total)
        .map(|i| match i {
            _ if (dst_at..dst_at + len).contains(&i) => stale_bits::<T>(i),
            _ if (src_at..src_at + src_len).contains(&i) => check_bits::<T>(i, block(i)),
            _ => guard_bits::<T>(i),
        })
        .collect();
    let mut all = AlignedVec::<T>::with_capacity(total);
    orig.iter().for_each(|bits| all.push(T::from_bits64(*bits)));
    let base = all.as_mut_ptr();
    let mut want = AlignedVec::<T>::zeros(len);
    // SAFETY: `src_at .. src_at + src_len` lies inside `all` and is
    // disjoint from the destination range that the kernel writes through
    // `base`; nothing else touches it while the slice lives.
    let src = unsafe { core::slice::from_raw_parts(base.add(src_at).cast_const(), src_len) };
    let view = MatRef::from_strided(src, ld, scols, ld, 1).ok_or("no view")?;
    let trans = if a_side { Trans::Trans } else { Trans::NoTrans };
    let op = ColMajor::normalize(view, trans).ok_or("no op")?;
    let (d, w) = (
        Span {
            start: 1,
            len: depth,
        },
        Span {
            start: 1,
            len: width,
        },
    );
    let plain = if a_side {
        pack::pack_a(&op, w, d, &mut want).is_some()
    } else {
        pack::pack_b(&op, d, w, &mut want).is_some()
    };
    ensure!(plain, "{case}: the plain packer declined");
    let (step, tile_step) = if a_side {
        (GROUP_COLS, T::TILE_ROWS)
    } else {
        (T::NR, T::NR * depth)
    };
    let args = ZaPackArgs {
        src: view.ptr_at(1, 1),
        ld,
        // SAFETY: `dst_at + len <= total`: inside `all`.
        dst: unsafe { base.add(dst_at) },
        depth,
        cols: width,
        width: step,
        tile_step,
    };
    // SAFETY: the block lies inside the source range, `dst` has `len`
    // writable elements disjoint from it; SME2 is the caller's.
    unsafe { T::TRANSPOSE(&args) };
    for (i, g) in all.as_slice().iter().enumerate() {
        let expect = if (dst_at..dst_at + len).contains(&i) {
            want.as_slice().get(i - dst_at).map(|w| w.bits())
        } else {
            orig.get(i).copied()
        };
        ensure!(
            expect == Some(g.bits()),
            "{case}: element {i} (source at {src_at}, buffer at {dst_at}, length {len}): {:#x}, want {expect:x?}",
            g.bits()
        );
    }
    Ok(())
}

#[test]
fn test_sme2_pack_za_adjacent_operands() {
    if !hardware_or_skip("test_sme2_pack_za_adjacent_operands", Need::F64) {
        return;
    }
    let shapes = [
        (1, 1, 0),
        (17, 35, 3),
        (9, 24, 0),
        (8, 57, 1),
        (16, 64, 0),
        (65, 129, 2),
        (129, 65, 5),
        (513, 300, 7),
    ];
    for src_first in [true, false] {
        for a_side in [false, true] {
            for shape in shapes {
                // SAFETY: SME2 with f64 is present (the gate).
                let results = unsafe {
                    [
                        run_adjacent::<f32>(a_side, shape, src_first),
                        run_adjacent::<f64>(a_side, shape, src_first),
                    ]
                };
                for result in results {
                    if let Err(why) = result {
                        panic!("{why}");
                    }
                }
            }
        }
    }
    assert_eq!(sme_state(), Some((0, 0)));
}
