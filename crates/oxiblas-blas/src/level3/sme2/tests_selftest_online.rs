//! Tests of the online cases of the first-use self-test (`selftest.rs`):
//! the blocks with online packing of A and `PREFETCH_C`.
//!
//! - Plain-Rust stand-ins whose packed copy is wrong in one known way (the
//!   buffer skipped, the zero padding left stale or written as `-0` or from
//!   the source's rows `≥ m`, a row off) and the fault classes of the
//!   containment limit (a store one vector after the buffer, the buffer
//!   pointer one depth step late or early, one depth step too far per pass,
//!   the packing pass repeated on every later column panel, the pass left
//!   out for `n ≤ NR`): each is rejected, and reported where its defect is.
//! - The containment limit itself, block by block: every fault class above
//!   stays inside the guards of every online block of the list, so a kernel
//!   with such a fault is reported, never a write outside the allocation.
//! - The claims of the online cases, the report of a buffer difference, and
//!   the real macro-kernels on the online blocks (skipped without SME2).
//!
//! The path table (`tests_selftest_paths.rs`) proves that the online blocks
//! take every path class of the packing twins, the buffer and the prefetch
//! block.

use super::args::{MR, MacroArgs, SmeElement};
use super::oracle::oracle_block;
use super::selftest::{
    ALayout, CASES, Case, Data, Difference, KernelFn, Mismatch, Online, PACK_FRONT, PackLayout,
    SCALARS, a_spare, online_cases, online_sweep_cases, pack_guard, pack_layout, pack_stale,
    run_buffers, self_test_blocks, self_test_cases, self_test_cases_detailed, self_test_with,
    self_test_with_detailed,
};
use super::test_support::{
    Need, fpsr_read, fpsr_write, hardware_or_skip, kernel_entries_here, kernel_ready, sme_state,
};
use super::tests_oracle::TestFloat;

/// Elements of one streaming vector: the rows of one ZA tile.
fn vector<T: SmeElement>() -> usize {
    T::TILE_ROWS
}

/// The packed copy of A of `args` (what the oracle writes into the buffer).
///
/// # Safety
///
/// A must hold `A(i, p)` for `i < m`, `p < kc` at the layout of `args.rs`.
unsafe fn packed_copy<T: SmeElement>(args: &MacroArgs<T>) -> Vec<T> {
    let panel = MR * args.kc;
    let mut copy = vec![T::ZERO; args.m.div_ceil(MR) * panel];
    for i in 0..args.m {
        for p in 0..args.kc {
            // SAFETY: `i < m`, `p < kc`: `A(i, p)` (caller's contract).
            let value = unsafe {
                *args
                    .a
                    .add((i / MR) * args.a_panel + p * args.a_depth + i % MR)
            };
            if let Some(slot) = copy.get_mut((i / MR) * panel + p * MR + i % MR) {
                *slot = value;
            }
        }
    }
    copy
}

/// Writes `values` to the buffer of `args` from element `start` on (a
/// negative start writes before the buffer).
///
/// # Safety
///
/// Every element written must lie inside the self-test's allocation of the
/// buffer (its guards included): `test_sme2_selftest_online_containment`
/// proves that for every fault class the stand-ins below model, on every
/// online block of the list.
unsafe fn write_at<T: SmeElement>(args: &MacroArgs<T>, start: isize, values: &[T]) {
    for (offset, &value) in values.iter().enumerate() {
        // SAFETY: inside the allocation by this function's contract.
        unsafe { *args.a_pack.offset(start + offset as isize) = value };
    }
}

/// The oracle without the buffer (C right), then `write` on the buffer with
/// the right packed copy (nothing when the call does not pack).
///
/// # Safety
///
/// As for `oracle_block`, and `write` keeps to the contract of [`write_at`].
unsafe fn with_buffer<T: SmeElement>(
    args: &MacroArgs<T>,
    write: impl FnOnce(&MacroArgs<T>, Vec<T>),
) {
    let unpacked = MacroArgs {
        a_pack: core::ptr::null_mut(),
        ..*args
    };
    // SAFETY: forwarded; the oracle writes no buffer.
    unsafe { oracle_block(&unpacked) };
    if !args.a_pack.is_null() {
        // SAFETY: A as the self-test lays it out.
        let copy = unsafe { packed_copy(args) };
        write(args, copy);
    }
}

/// Stand-in: right in C, but writes nothing into the buffer.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_pack_skips<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; nothing is written to the buffer.
    unsafe { with_buffer(args, |_, _| {}) };
}

/// The rows of element `e` of a packed copy of depth `kc`.
fn row_of(e: usize, kc: usize) -> usize {
    (e / (MR * kc)) * MR + e % MR
}

/// Stand-in: the packed copy without its zero padding (rows `≥ m` of the
/// last panel keep their stale contents).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_pack_padding_stale<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the buffer.
    unsafe {
        with_buffer(args, |args, copy| {
            for (e, &value) in copy.iter().enumerate() {
                if row_of(e, args.kc) < args.m {
                    write_at(args, e as isize, &[value]);
                }
            }
        });
    }
}

/// Stand-in: the zero padding written as `-0`.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_pack_padding_negative<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the buffer.
    unsafe {
        with_buffer(args, |args, mut copy| {
            for (e, value) in copy.iter_mut().enumerate() {
                if row_of(e, args.kc) >= args.m {
                    *value = T::from_bits64(if T::NR == 16 {
                        0x8000_0000
                    } else {
                        0x8000_0000_0000_0000
                    });
                }
            }
            write_at(args, 0, &copy);
        });
    }
}

/// Stand-in: the padding rows loaded from the source like live rows (a
/// packing pass that reads A rows `≥ m`): the zero padding of packed A,
/// the spare rows of in-place A.
///
/// # Safety
///
/// As for `oracle_block`, and A's allocation holds those rows (the
/// self-test's A does, see `Operands::new`).
unsafe fn wrong_pack_reads_padding<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the buffer.
    unsafe {
        with_buffer(args, |args, mut copy| {
            let panel = MR * args.kc;
            for (e, value) in copy.iter_mut().enumerate() {
                let (i, p) = (row_of(e, args.kc), (e % panel) / MR);
                if i >= args.m {
                    // SAFETY: row `i < 64 · ceil(m / 64)` of the last panel:
                    // inside the self-test's A (caller's contract).
                    *value = *args
                        .a
                        .add((i / MR) * args.a_panel + p * args.a_depth + i % MR);
                }
            }
            write_at(args, 0, &copy);
        });
    }
}

/// Stand-in: every row of the copy one row up (row `i` holds row `i + 1`
/// of its panel, the last row of a panel zero).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_pack_row_off<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the buffer.
    unsafe {
        with_buffer(args, |args, copy| {
            let shifted: Vec<T> = (0..copy.len())
                .map(|e| match e % MR {
                    63 => T::ZERO,
                    _ => copy.get(e + 1).copied().unwrap_or(T::ZERO),
                })
                .collect();
            write_at(args, 0, &shifted);
        });
    }
}

/// Stand-in (fault class: a store one vector off): the right copy, and its
/// last vector stored once more one vector after the end.
///
/// # Safety
///
/// As for `oracle_block`; the store lies in the guard after the buffer.
unsafe fn wrong_pack_vector_after<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the allocation.
    unsafe {
        with_buffer(args, |args, copy| {
            write_at(args, 0, &copy);
            let tail = copy.len().saturating_sub(vector::<T>());
            write_at(
                args,
                copy.len() as isize,
                copy.get(tail..).unwrap_or_default(),
            );
        });
    }
}

/// Stand-in (fault class: the buffer pointer one depth step late): the copy
/// written `MR` elements after the buffer's start.
///
/// # Safety
///
/// As for `oracle_block`; the writes reach `MR` elements into the guard.
unsafe fn wrong_pack_step_late<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the allocation.
    unsafe { with_buffer(args, |args, copy| write_at(args, MR as isize, &copy)) };
}

/// Stand-in (fault class: the buffer pointer one depth step early): the
/// copy written `MR` elements before the buffer's start.
///
/// # Safety
///
/// As for `oracle_block`; the writes reach `MR` elements into the guard
/// before the buffer.
unsafe fn wrong_pack_step_early<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the allocation.
    unsafe { with_buffer(args, |args, copy| write_at(args, -(MR as isize), &copy)) };
}

/// The offset of depth step `p` of row panel `panel` when the buffer
/// pointer moves one depth step too far per pass of the double-step loop
/// (three steps per pass of two, one per remainder step).
fn stride_offset(panel: usize, p: usize, kc: usize) -> usize {
    let per_panel = (kc / 2) * 3 + kc % 2;
    (panel * per_panel + (p / 2) * 3 + p % 2) * MR
}

/// Stand-in (fault class: one depth step too far per pass): each depth step
/// of the copy at [`stride_offset`].
///
/// # Safety
///
/// As for `oracle_block`; the writes reach about half the buffer's length
/// into the guard after it.
unsafe fn wrong_pack_stride<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the allocation.
    unsafe {
        with_buffer(args, |args, copy| {
            for (step, values) in copy.chunks(MR).enumerate() {
                let (panel, p) = (step / args.kc, step % args.kc);
                write_at(args, stride_offset(panel, p, args.kc) as isize, values);
            }
        });
    }
}

/// Stand-in (fault class: the packing pass repeated on every later column
/// panel, the buffer pointer running on): the copy written once per column
/// panel, back to back.
///
/// # Safety
///
/// As for `oracle_block`; the writes reach `(N - 1) · P` elements into the
/// guard after the buffer.
unsafe fn wrong_pack_every_panel<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the allocation.
    unsafe {
        with_buffer(args, |args, copy| {
            for panel in 0..args.n.div_ceil(T::NR) {
                write_at(args, (panel * copy.len()) as isize, &copy);
            }
        });
    }
}

/// Stand-in: packs only when a later column panel reads the buffer
/// (`n > NR`), against the contract.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_pack_needs_later_panel<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded; the writes stay inside the buffer.
    unsafe {
        with_buffer(args, |args, copy| {
            if args.n > T::NR {
                write_at(args, 0, &copy);
            }
        });
    }
}

/// Every buffer stand-in for one type, with its name.
fn wrong_packers<T: TestFloat>() -> [(&'static str, KernelFn<T>); 11] {
    [
        ("skips the buffer", wrong_pack_skips::<T>),
        ("padding stale", wrong_pack_padding_stale::<T>),
        ("padding -0", wrong_pack_padding_negative::<T>),
        ("padding read from A", wrong_pack_reads_padding::<T>),
        ("one row off", wrong_pack_row_off::<T>),
        ("one vector after", wrong_pack_vector_after::<T>),
        ("one step late", wrong_pack_step_late::<T>),
        ("one step early", wrong_pack_step_early::<T>),
        ("one step too far per pass", wrong_pack_stride::<T>),
        ("every column panel", wrong_pack_every_panel::<T>),
        ("only with a later panel", wrong_pack_needs_later_panel::<T>),
    ]
}

/// The first mismatch of `kernel` over the whole list.
fn report<T: SmeElement>(kernel: KernelFn<T>) -> Mismatch {
    // SAFETY: the stand-ins are plain Rust and stay inside the self-test's
    // allocations (`test_sme2_selftest_online_containment`).
    match unsafe { self_test_with_detailed(kernel) } {
        Ok(()) => panic!("{} stand-in accepted", T::NR),
        Err(mismatch) => {
            println!("{mismatch}");
            mismatch
        }
    }
}

/// Every buffer stand-in is rejected, at the block, pair and place of its
/// defect: the first online block of the list (`70 × (NR - 3) × 13`, packed,
/// block 7, pair 0) for all but two: the copy repeated per column panel is
/// right where there is one column panel and first fails at the first block
/// with three (block 9, `84 × (2 · NR + 3) × 16`), and the padding read from
/// A is right where A is packed (its padding is `+0`) and first fails at the
/// first in-place online block (block 8, `9 × (NR - 1) × 37`, whose spare
/// rows hold `a_spare`).
fn rejects_and_reports<T: TestFloat>() {
    for (name, kernel) in wrong_packers::<T>() {
        // SAFETY: plain-Rust stand-ins inside the self-test's allocations.
        let verdict = unsafe { self_test_with(kernel) };
        assert!(!verdict, "{} {name} accepted", T::NR);
    }
    let cases = online_cases(T::NR);
    let first = cases[0];
    let len = first.m.div_ceil(MR) * MR * first.kc;
    let at = |block: usize, difference: Difference| (block, 0, difference);
    let got = |mismatch: Mismatch| (mismatch.block, mismatch.pair, mismatch.difference);
    let stale = |e: usize| pack_stale::<T>(e).bits();
    // The buffer untouched: its first element still stale.
    let skipped = report(wrong_pack_skips::<T>);
    let Difference::BufferElement {
        row: 0,
        depth: 0,
        found,
        ..
    } = skipped.difference
    else {
        panic!("{}: {skipped:?}", T::NR);
    };
    assert_eq!(
        (skipped.block, skipped.pair, found),
        (CASES.len(), 0, stale(0))
    );
    // The padding: row 70 (the first padding row of the second panel),
    // depth 0, element `64 · 13 + 6` of the buffer.
    let padding = MR * first.kc + first.m % MR;
    assert_eq!(
        got(report(wrong_pack_padding_stale::<T>)),
        at(
            CASES.len(),
            Difference::BufferElement {
                row: first.m as u32,
                depth: 0,
                found: stale(padding),
                wanted: 0,
            }
        )
    );
    let negative = report(wrong_pack_padding_negative::<T>);
    assert!(
        matches!(
            negative.difference,
            Difference::BufferElement {
                row: 70,
                depth: 0,
                wanted: 0,
                ..
            }
        ),
        "{negative:?}"
    );
    assert!(
        negative
            .to_string()
            .contains("row 70 (zero padding), depth 0"),
        "{negative}"
    );
    let read = report(wrong_pack_reads_padding::<T>);
    let in_place = cases[1];
    assert_eq!(
        got(read),
        at(
            CASES.len() + 1,
            Difference::BufferElement {
                row: in_place.m as u32,
                depth: 0,
                found: a_spare::<T>().bits(),
                wanted: 0,
            }
        )
    );
    let row_off = report(wrong_pack_row_off::<T>);
    assert!(
        matches!(row_off.difference, Difference::BufferElement { .. })
            && (row_off.block, row_off.pair) == (CASES.len(), 0),
        "{row_off:?}"
    );
    // The fault classes of the containment limit: guard differences.
    let guard_at = |mismatch: &Mismatch| match mismatch.difference {
        Difference::BufferGuard { offset, len, .. } => Some((mismatch.block, offset, len)),
        _ => None,
    };
    let len32 = len as u32;
    let after = report(wrong_pack_vector_after::<T>);
    assert_eq!(
        guard_at(&after),
        Some((CASES.len(), len as i64, len32)),
        "{after:?}"
    );
    let Difference::BufferGuard { found, .. } = after.difference else {
        panic!("{after:?}");
    };
    // The last vector of the copy: padding rows of the last step, zero.
    assert_eq!(found, 0, "{}", T::NR);
    assert!(
        after.to_string().contains(&format!(
            "the kernel wrote the online A buffer at offset {len} (outside its {len} elements)"
        )),
        "{after}"
    );
    let late = report(wrong_pack_step_late::<T>);
    assert_eq!(
        guard_at(&late),
        Some((CASES.len(), len as i64, len32)),
        "{late:?}"
    );
    let early = report(wrong_pack_step_early::<T>);
    assert_eq!(
        guard_at(&early),
        Some((CASES.len(), -(MR as i64), len32)),
        "{early:?}"
    );
    let stride = report(wrong_pack_stride::<T>);
    assert_eq!(
        guard_at(&stride),
        Some((CASES.len(), len as i64, len32)),
        "{stride:?}"
    );
    let third = cases[2];
    let third_len = third.m.div_ceil(MR) * MR * third.kc;
    let every = report(wrong_pack_every_panel::<T>);
    assert_eq!(
        guard_at(&every),
        Some((CASES.len() + 2, third_len as i64, third_len as u32)),
        "{every:?}"
    );
    let later = report(wrong_pack_needs_later_panel::<T>);
    assert!(
        matches!(
            later.difference,
            Difference::BufferElement {
                row: 0,
                depth: 0,
                ..
            }
        ) && later.block == CASES.len(),
        "{later:?}"
    );
}

#[test]
fn test_sme2_selftest_online_rejects_wrong_packers() {
    rejects_and_reports::<f32>();
    rejects_and_reports::<f64>();
}

/// Each stand-in is right where its defect cannot show, so the verdicts
/// above come from the buffer's comparisons: without online packing all are
/// right; the padding stand-ins are right where `m` is a multiple of 64; the
/// repeated copy is right with one column panel; the copy that needs a later
/// panel is right with two.
fn right_where_they_are_right<T: TestFloat>() {
    let block = |m: usize, n: usize, online: Online| Case {
        m,
        n,
        kc: 5,
        layout: ALayout::InPlace,
        lda_pad: 2,
        b_pad: 1,
        data: Data::Plain,
        online,
    };
    let unpacked = block(70, 2 * T::NR + 1, Online::Prefetch);
    for (name, kernel) in wrong_packers::<T>() {
        // SAFETY: plain-Rust stand-ins inside the self-test's allocations.
        let verdict = unsafe { self_test_cases(kernel, &[unpacked], &SCALARS) };
        assert!(verdict, "{} {name} without online packing", T::NR);
    }
    let checks: [(KernelFn<T>, Case); 5] = [
        (
            wrong_pack_padding_stale::<T>,
            block(MR, T::NR + 1, Online::Pack),
        ),
        (
            wrong_pack_padding_negative::<T>,
            block(2 * MR, T::NR + 1, Online::Pack),
        ),
        (
            wrong_pack_reads_padding::<T>,
            block(MR, 3, Online::PackPrefetch),
        ),
        (wrong_pack_every_panel::<T>, block(70, T::NR, Online::Pack)),
        (
            wrong_pack_needs_later_panel::<T>,
            block(70, T::NR + 1, Online::Pack),
        ),
    ];
    for (index, (kernel, case)) in checks.into_iter().enumerate() {
        // SAFETY: as above.
        let verdict = unsafe { self_test_cases(kernel, &[case], &SCALARS) };
        assert!(verdict, "{} check {index} on {case}", T::NR);
    }
}

#[test]
fn test_sme2_selftest_online_stand_ins_right_where_they_are_right() {
    right_where_they_are_right::<f32>();
    right_where_they_are_right::<f64>();
}

/// The reach of one fault class on a block: the range of buffer offsets it
/// writes (relative to the buffer's first element), for a buffer of `len`
/// elements in `panels` column panels and `kc` depth steps, with `vector`
/// elements per vector.
fn fault_reach(
    len: usize,
    panels: usize,
    kc: usize,
    vector: usize,
) -> [(&'static str, i64, i64); 7] {
    let len = len as i64;
    let mr = MR as i64;
    let rows = len / (mr * kc as i64).max(1);
    let stride_end = stride_offset(rows.max(1) as usize - 1, kc - 1, kc) as i64 + mr;
    [
        ("a store one vector after", 0, len + vector as i64),
        (
            "a four-vector group one group off",
            0,
            len + 4 * vector as i64,
        ),
        ("a store one depth step off", 0, len + mr),
        ("the pointer one depth step late", mr, len + mr),
        ("the pointer one depth step early", -mr, len - mr),
        ("one depth step too far per pass", 0, stride_end),
        (
            "the packing pass on every later column panel, one step off",
            0,
            panels as i64 * len + mr,
        ),
    ]
}

/// The containment limit of the online buffer, for one type: every online
/// block of the list has the documented guards ([`PACK_FRONT`] before,
/// `max(N - 1, 1) · P + 2 · MR` after), and every fault class of
/// [`fault_reach`] stays inside them. Returns the largest buffer allocation
/// (elements) and the farthest reach past the end of a buffer as a share of
/// the guard after it.
fn containment<T: SmeElement>() -> (usize, f64) {
    let mut largest = 0;
    let mut tightest = 0.0_f64;
    for case in self_test_blocks(T::TILE_ROWS, T::NR) {
        let Some(PackLayout { front, len, alloc }) = pack_layout::<T>(&case) else {
            panic!("{}: no buffer layout for {case}", T::NR);
        };
        assert_eq!(
            run_buffers::<T>(&case).map(|[_, _, _, pack]| pack),
            Some(alloc),
            "{case}"
        );
        if !case.online.packs() {
            assert_eq!((front, len, alloc), (0, 0, 0), "{case}");
            continue;
        }
        let panels = case.n.div_ceil(T::NR);
        assert_eq!(front, PACK_FRONT, "{case}");
        assert_eq!(front, 2 * MR, "{case}");
        assert_eq!(len, case.m.div_ceil(MR) * MR * case.kc, "{case}");
        let back = alloc - front - len;
        assert_eq!(
            back,
            panels.saturating_sub(1).max(1) * len + 2 * MR,
            "{case}"
        );
        largest = largest.max(alloc);
        for (fault, start, end) in fault_reach(len, panels, case.kc, T::TILE_ROWS) {
            assert!(
                start >= -(front as i64),
                "{}: {fault} on {case}: {start}",
                T::NR
            );
            assert!(
                end <= (len + back) as i64,
                "{}: {fault} on {case}: {end}",
                T::NR
            );
            tightest = tightest.max((end - len as i64) as f64 / back as f64);
        }
    }
    (largest, tightest)
}

#[test]
fn test_sme2_selftest_online_containment() {
    let (f32_largest, f32_tightest) = containment::<f32>();
    let (f64_largest, f64_tightest) = containment::<f64>();
    println!(
        "largest buffer allocation: f32 {f32_largest} elements, f64 {f64_largest}; farthest \
         fault reach after the end, share of the guard: f32 {f32_tightest:.3}, f64 \
         {f64_tightest:.3}"
    );
    // The guards are larger than the farthest fault class needs.
    assert!(f32_tightest < 1.0 && f64_tightest < 1.0);
    // The guards hold the sentinel, the buffer the stale contents: neither
    // is a value of a correct packed copy (finite values of A and +0), and
    // the stale contents mix signalling NaNs and finite values.
    fn fills<T: TestFloat>() {
        let guard = pack_guard::<T>();
        assert!(
            guard.nan() && guard.bits() != a_spare::<T>().bits(),
            "{}",
            T::NR
        );
        let stale: Vec<T> = (0..8).map(pack_stale::<T>).collect();
        assert!(stale.iter().step_by(2).all(|value| value.nan()));
        assert!(
            stale
                .iter()
                .skip(1)
                .step_by(2)
                .all(|value| !value.nan() && value.bits() != T::ZERO.bits())
        );
    }
    fills::<f32>();
    fills::<f64>();
}

/// The claims of [`online_cases`] for a panel width `nr`.
fn online_claims(nr: usize) {
    let cases = online_cases(nr);
    let panels = |case: &Case| case.n.div_ceil(nr);
    assert!(cases.iter().all(|case| case.online.packs()));
    // One partial column panel (n ≤ NR still fills the buffer), with and
    // without prefetch.
    for (index, prefetch) in [(0, false), (1, true)] {
        let case = cases[index];
        assert!(case.n < nr, "{case}");
        assert_eq!(case.online.prefetch() != 0, prefetch, "{case}");
    }
    // Three column panels (the buffer read by a full and a partial panel),
    // with and without prefetch.
    for (index, prefetch) in [(2, false), (3, true)] {
        let case = cases[index];
        assert!(panels(&case) == 3 && !case.n.is_multiple_of(nr), "{case}");
        assert_eq!(case.online.prefetch() != 0, prefetch, "{case}");
    }
    // Special values, a deep odd kc, kc a multiple of 8, three row panels,
    // in place and packed.
    assert!(cases.iter().any(|case| case.data == Data::Special));
    assert!(cases.iter().any(|case| case.kc > 32 && case.kc % 2 == 1));
    assert!(
        cases
            .iter()
            .any(|case| case.kc.is_multiple_of(8) && case.m > MR)
    );
    assert!(cases.iter().any(|case| case.m > 2 * MR));
    assert!(
        cases
            .iter()
            .any(|case| case.layout == ALayout::InPlace && case.m > MR)
    );
    assert!(cases.iter().any(|case| case.layout == ALayout::Packed));
    assert!(
        cases
            .iter()
            .all(|case| case.layout == ALayout::InPlace || case.lda_pad == 0)
    );
}

#[test]
fn test_sme2_selftest_online_cases_cover_the_claims() {
    online_claims(16);
    online_claims(8);
    // The online sweep is the plain sweep with online packing, prefetch
    // where the plain sweep has it (A in place).
    for (tile_rows, nr) in [(16, 16), (8, 8)] {
        for (online, plain) in
            online_sweep_cases(tile_rows, nr).zip(super::selftest::sweep_cases(tile_rows, nr))
        {
            assert_eq!(
                Case {
                    online: plain.online,
                    ..online
                },
                plain
            );
            assert!(online.online.packs());
            assert_eq!(online.online.prefetch(), plain.online.prefetch());
            assert_eq!(
                plain.online.prefetch() != 0,
                plain.layout == ALayout::InPlace
            );
        }
    }
    // The fixed blocks of the plain paths keep both fields off.
    assert!(CASES.iter().all(|case| case.online == Online::Off));
}

#[test]
fn test_sme2_selftest_online_report_is_one_line() {
    let case = online_cases(16)[1];
    assert!(
        case.to_string()
            .ends_with("special data, online packing, prefetch C"),
        "{case}"
    );
    for difference in [
        Difference::BufferGuard {
            offset: -3,
            len: 2368,
            found: 0x7f8d_0d0d,
        },
        Difference::OracleBufferGuard {
            offset: 2368,
            len: 2368,
            found: 0,
        },
        Difference::BufferElement {
            row: 9,
            depth: 36,
            found: 0x7f8e_0e0e,
            wanted: 0,
        },
    ] {
        let mismatch = Mismatch {
            element_bits: 32,
            block: 8,
            case,
            pair: 2,
            alpha: 0.3,
            beta: -0.0,
            difference,
        };
        let line = mismatch.to_string();
        println!("{line}");
        assert!(!line.contains('\n'), "{line}");
        assert!(
            line.starts_with("SME2 self-test mismatch, f32: block 8 (9x15x37, A in place")
                && line.contains("online packing, prefetch C")
                && line.contains("pair 2 (alpha 0.3, beta -0)"),
            "{line}"
        );
        match difference {
            Difference::BufferGuard { .. } => assert!(
                line.ends_with("the kernel wrote the online A buffer at offset -3 (outside its 2368 elements): 0x7f8d0d0d"),
                "{line}"
            ),
            Difference::OracleBufferGuard { .. } => {
                assert!(line.ends_with("0x00000000 (an oracle defect)"), "{line}");
            }
            _ => assert!(
                line.ends_with(
                    "the online A buffer holds 0x7f8e0e0e for row 9 (zero padding), depth 36, \
                     the oracle 0x00000000"
                ),
                "{line}"
            ),
        }
    }
}

/// The real kernel of `T` on the online blocks only (the online cases and
/// the online sweep): every run matches, one kernel call per block and
/// pair, counted on this thread. Restores FPSR.
///
/// # Safety
///
/// `hardware_or_skip` returned true for the kernel's type.
unsafe fn real_online<T: SmeElement>(test: &str, kernel: KernelFn<T>) {
    assert!(kernel_ready(), "{test}: thread not ready");
    let blocks: Vec<Case> = online_cases(T::NR)
        .into_iter()
        .chain(online_sweep_cases(T::TILE_ROWS, T::NR))
        .collect();
    let fpsr = fpsr_read();
    let entries = kernel_entries_here();
    // SAFETY: SME2 (with FEAT_SME_F64F64 for `f64`) is present by this
    // function's contract, and `kernel_ready()` was true just above.
    let verdict = unsafe { self_test_cases_detailed(kernel, &blocks, &SCALARS) };
    let state = sme_state();
    let grown = kernel_entries_here() - entries;
    fpsr_write(fpsr);
    if let Err(mismatch) = verdict {
        panic!("{test}: the kernel differs from the oracle: {mismatch}");
    }
    assert_eq!(state, Some((0, 0)), "{test}: SME state left");
    assert_eq!(
        grown,
        (blocks.len() * SCALARS.len()) as u64,
        "{test}: kernel calls"
    );
}

#[test]
fn test_sme2_selftest_real_online_f32() {
    let test = "test_sme2_selftest_real_online_f32";
    if hardware_or_skip(test, Need::F32) {
        // SAFETY: `hardware_or_skip` returned true for `f32`.
        unsafe { real_online::<f32>(test, super::kernel_f32::sme2_gemm_f32) };
    }
}

#[test]
fn test_sme2_selftest_real_online_f64() {
    let test = "test_sme2_selftest_real_online_f64";
    if hardware_or_skip(test, Need::F64) {
        // SAFETY: `hardware_or_skip` returned true for `f64`
        // (FEAT_SME_F64F64).
        unsafe { real_online::<f64>(test, super::kernel_f64::sme2_gemm_f64) };
    }
}
