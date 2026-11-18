//! The online parts of the path model's text derivation
//! (`tests_selftest_text.rs`): the prefetch block of the row-panel head, the
//! switch of A to the online buffer after column panel 0, and the accounting
//! of every branch of a kernel's body.
//!
//! - The prefetch block is evaluated, not matched: [`prefetch_problems`]
//!   runs the general-purpose part of the sections from the row-panel head
//!   to the depth setup for every row count `1..=64` of a last row panel,
//!   the row counts [`FULL_PANEL_ROWS`] of a row panel with more rows after
//!   it, every count of live columns `1..=NR` and both values of bit 0 of
//!   `prefetch`, and requires exactly one hint per 64 bytes of each live
//!   column's live rows (one per live tile, in order; every tile of a panel
//!   with more than 64 rows left) with the bit set, none with it clear, and
//!   the row count unchanged when the block hands over to the dispatch. So
//!   the chain that stops after the last live tile, the column loop and the
//!   skip are all read from the text, and an edit of any of them that
//!   changes a hint fails here. The hints are compared relative to the first
//!   one (the base register's value is unknown here): a block whose first
//!   hint is not at `C(i0, j0)` (a base one row panel or one column off)
//!   passes; the kernels' interpreter tests check the base.
//! - [`switch_problems`] reads the switch: after the row-panel loop of a
//!   column panel, `cbz <a_pack>` skips to the column-panel advance, else A
//!   becomes the buffer (`a := a_pack`, `a_depth := 64` elements, `a_panel
//!   := 64 · kc` elements, in bytes) and `a_pack := 0`, so only column
//!   panel 0 packs. It also requires that nothing else writes these
//!   registers, and that the column-panel head and the row-panel advance use
//!   them.
//! - [`unknown_branch`] names the first branch of the body that no part of
//!   the derivation interpreted: a branch is a path, so a kernel edit that
//!   adds one fails the model until the derivation (and with it the path
//!   table and the self-test) knows it. A conditional select is a path as
//!   much as a branch is; only the clamp of the live columns is interpreted.
//!   The branches of the prefetch block and of the epilogues count as
//!   interpreted only in the forms of a tile chain and a column loop
//!   ([`account_prefetch`], [`account_epilogues`]): forward, a compare of the
//!   row count with a tile threshold (a positive multiple of the tile height
//!   below 64), or of `m` with a tile's first row (`i0` plus such a
//!   threshold); back, one of the kernels' two column loops ([`column_loop`]:
//!   a counter from 0 up to the live columns, or down from them to 0). Every
//!   row count from 64 on decides such a compare as 64 does, and a column
//!   loop runs once per live column whatever the row count, so a panel with
//!   more rows left (the self-test's blocks reach 192) takes the paths of a
//!   full one; a compare with any other number (`cmp <rows>, #200`) would be
//!   a path that only longer blocks take, which no self-test block reaches.
//!
//! Everything here is plain Rust and runs on every host.

use std::collections::BTreeSet;

use super::args::MR;
use super::tests_selftest_text::{
    Section, branch_section, branch_target, condition, instruction, operands_of, register_number,
    same_register, section_name, writes_first_operand,
};

/// Bytes of one streaming vector (SVL 512): the rows of one ZA tile in one
/// C column, so one prefetch hint per live tile.
const VECTOR_BYTES: u64 = 64;

/// The leading dimension of C (bytes) the evaluation uses: far larger than a
/// row panel's 64 rows, so a hint's column and row offset never alias.
const LDC_BYTES: u64 = 1 << 20;

/// The first row of the row panel in the evaluation (a second row panel, so
/// that a block that confused `i0` with 0 shows).
const I0: u64 = 64;

/// Statements after which an evaluation counts as one that never ends.
const STEP_LIMIT: usize = 100_000;

/// Row counts of a row panel with more rows after it (`m - i0 > 64`) that
/// [`prefetch_problems`] evaluates besides `1..=64`: just above a panel, the
/// two- and three-panel counts the self-test's blocks reach (up to 192), just
/// beyond them, and beyond every immediate a compare can encode (12 bits,
/// or 12 bits shifted by 12), so that a compare of the row count with any
/// immediate meets both of its outcomes among the evaluated counts.
const FULL_PANEL_ROWS: [u64; 8] = [65, 128, 129, 192, 193, 201, 4096, 1 << 24];

/// Whether `k` rows is a tile threshold of the row count: a positive
/// multiple of `tile_rows` below `MR`. Every row count from `MR` on decides
/// a compare with it as `MR` does.
fn tile_threshold(k: usize, tile_rows: usize) -> bool {
    k > 0 && k < MR && k.is_multiple_of(tile_rows.max(1))
}

/// Whether `operand` is an immediate `#k` with `k` a [`tile_threshold`].
fn tile_threshold_immediate(operand: &str, tile_rows: usize) -> bool {
    operand
        .strip_prefix('#')
        .and_then(|digits| digits.parse::<usize>().ok())
        .is_some_and(|k| tile_threshold(k, tile_rows))
}

/// A general-purpose register's value in the evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    /// A known number.
    Known(u64),
    /// The value register `base` held when the evaluation began, plus
    /// `offset` (an address the evaluation does not know).
    Entry(u32, i64),
    /// Anything else.
    Unknown,
}

impl Value {
    /// `self + other`.
    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Known(a), Self::Known(b)) => Self::Known(a.wrapping_add(b)),
            (Self::Entry(base, offset), Self::Known(k))
            | (Self::Known(k), Self::Entry(base, offset)) => {
                Self::Entry(base, offset.wrapping_add(k as i64))
            }
            _ => Self::Unknown,
        }
    }

    /// `self - other`.
    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Self::Known(a), Self::Known(b)) => Self::Known(a.wrapping_sub(b)),
            (Self::Entry(base, offset), Self::Known(k)) => {
                Self::Entry(base, offset.wrapping_sub(k as i64))
            }
            (Self::Entry(a, x), Self::Entry(b, y)) if a == b => {
                Self::Known(x.wrapping_sub(y) as u64)
            }
            _ => Self::Unknown,
        }
    }
}

/// The registers an evaluation of the prefetch block starts from: the row
/// count, `m`, `i0`, the live columns, `prefetch` and `ldc` (in bytes, as
/// the prologue leaves it). Every other register holds its unknown entry
/// value.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefetchSeeds<'r> {
    /// The row count of the row panel (`m - i0`).
    pub(super) rows: &'r str,
    /// `m`.
    pub(super) m: &'r str,
    /// `i0`.
    pub(super) i0: &'r str,
    /// The live columns of the column panel.
    pub(super) columns: &'r str,
    /// `prefetch`.
    pub(super) prefetch: &'r str,
    /// `ldc` in bytes.
    pub(super) ldc: &'r str,
}

/// The general-purpose registers of an evaluation, with the flags of the
/// last compare.
struct Machine {
    /// `x0`–`x30`.
    registers: [Value; 31],
    /// The operands of the last compare (`cmp`, `subs`): both known, or
    /// `None`; and whether it compared `w` registers.
    flags: Option<(u64, u64, bool)>,
}

impl Machine {
    /// The value of a register operand (`x<n>`, `w<n>`), or of an immediate
    /// (`#k`).
    fn read(&self, operand: &str) -> Result<Value, String> {
        if let Some(digits) = operand.strip_prefix('#') {
            return digits
                .parse()
                .map(Value::Known)
                .map_err(|_| format!("immediate `{operand}`"));
        }
        let number = register_number(operand).ok_or(format!("operand `{operand}`"))?;
        let value = *self
            .registers
            .get(number as usize)
            .ok_or(format!("register `{operand}`"))?;
        Ok(match (value, operand.starts_with('w')) {
            (Value::Known(k), true) => Value::Known(k & u64::from(u32::MAX)),
            (Value::Known(_), false) => value,
            (_, true) => Value::Unknown,
            (other, false) => other,
        })
    }

    /// Writes `value` to the register operand `to` (a `w` register keeps
    /// the low 32 bits).
    fn write(&mut self, to: &str, value: Value) -> Result<(), String> {
        let number = register_number(to).ok_or(format!("destination `{to}`"))? as usize;
        let value = match (value, to.starts_with('w')) {
            (Value::Known(k), true) => Value::Known(k & u64::from(u32::MAX)),
            (_, true) => Value::Unknown,
            (other, false) => other,
        };
        let slot = self
            .registers
            .get_mut(number)
            .ok_or(format!("destination `{to}`"))?;
        *slot = value;
        Ok(())
    }

    /// Whether `cond` holds after the last compare.
    fn holds(&self, cond: &str) -> Option<bool> {
        let (lhs, rhs, word) = self.flags?;
        if word {
            // Signed conditions on `w` registers compare the low 32 bits.
            let widen = |value: u64| (value as u32 as i32) as i64 as u64;
            let signed = matches!(cond, "lt" | "ge" | "gt" | "le");
            if signed {
                return condition(cond, widen(lhs), widen(rhs));
            }
        }
        condition(cond, lhs, rhs)
    }
}

/// The value of `<shifted>` in `add <d>, <s>, <t>, lsl #k`.
fn shifted(machine: &Machine, operand: &str, shift: Option<&str>) -> Result<Value, String> {
    let value = machine.read(operand)?;
    let Some(shift) = shift else {
        return Ok(value);
    };
    let amount: u32 = shift
        .strip_prefix("lsl #")
        .and_then(|digits| digits.parse().ok())
        .ok_or(format!("shift `{shift}`"))?;
    Ok(match value {
        Value::Known(k) => Value::Known(k.checked_shl(amount).unwrap_or(0)),
        _ => Value::Unknown,
    })
}

/// The base register value and the immediate of a `prfm <op>, [<base>]` or
/// `[<base>, #k]`.
fn hint_address(machine: &Machine, line: &str) -> Result<Value, String> {
    let address = line
        .split_once('[')
        .and_then(|(_, rest)| rest.strip_suffix(']'))
        .ok_or(format!("address of `{line}`"))?;
    let mut parts = address.split(',').map(str::trim);
    let base = machine.read(parts.next().unwrap_or(""))?;
    let offset = match parts.next() {
        Some(immediate) => machine.read(immediate)?,
        None => Value::Known(0),
    };
    if parts.next().is_some() {
        return Err(format!("address of `{line}`"));
    }
    Ok(base.add(offset))
}

/// Runs the sections `head..setup` from the start of `head` for one row
/// panel (`rows` live rows, `columns` live columns, `prefetch = bits`) and
/// returns the hints in issue order (as values) and the row count when
/// control reaches `setup`. `Err` names a statement the evaluation does not
/// know, a branch it cannot decide, a branch out of the block, or a block
/// that does not end.
fn run_block(
    sections: &[Section<'_>],
    head: usize,
    setup: usize,
    seeds: PrefetchSeeds<'_>,
    (rows, columns, bits): (u64, u64, u64),
) -> Result<(Vec<Value>, Value), String> {
    let mut machine = Machine {
        registers: core::array::from_fn(|number| Value::Entry(number as u32, 0)),
        flags: None,
    };
    for (register, value) in [
        (seeds.m, I0 + rows),
        (seeds.i0, I0),
        (seeds.columns, columns),
        (seeds.prefetch, bits),
        (seeds.ldc, LDC_BYTES),
    ] {
        machine.write(register, Value::Known(value))?;
    }
    let mut hints = Vec::new();
    let (mut at, mut index, mut steps) = (head, 0, 0);
    while at != setup {
        if !(head..setup).contains(&at) {
            return Err(format!(
                "the prefetch block branches to section {}",
                section_name(sections, at)
            ));
        }
        steps += 1;
        if steps > STEP_LIMIT {
            return Err("the prefetch block does not end".to_string());
        }
        let Some(&line) = sections.get(at).and_then(|(_, lines)| lines.get(index)) else {
            (at, index) = (at + 1, 0);
            continue;
        };
        index += 1;
        let (mnemonic, operands) = instruction(line);
        let jump = |taken: bool| -> Result<Option<usize>, String> {
            if !taken {
                return Ok(None);
            }
            branch_section(sections, at, line)
                .map(Some)
                .ok_or(format!("`{line}` names no section"))
        };
        let target = match (mnemonic, operands.as_slice()) {
            ("prfm", _) => {
                hints.push(hint_address(&machine, line)?);
                None
            }
            ("whilelt", _) => None,
            ("mov", [to, from]) => {
                let value = machine.read(from)?;
                machine.write(to, value)?;
                None
            }
            ("add", [to, from, other, rest @ ..]) if rest.len() <= 1 => {
                let value =
                    machine
                        .read(from)?
                        .add(shifted(&machine, other, rest.first().copied())?);
                machine.write(to, value)?;
                None
            }
            ("sub" | "subs", [to, from, other]) => {
                let (left, right) = (machine.read(from)?, machine.read(other)?);
                if mnemonic == "subs" {
                    machine.flags = known_pair(left, right, from.starts_with('w'));
                }
                machine.write(to, left.sub(right))?;
                None
            }
            ("cmp", [left, right]) => {
                let (lhs, rhs) = (machine.read(left)?, machine.read(right)?);
                machine.flags = known_pair(lhs, rhs, left.starts_with('w'));
                None
            }
            ("b", [_]) => jump(true)?,
            (_, [_]) if mnemonic.starts_with("b.") => {
                let cond = mnemonic.get(2..).unwrap_or("");
                let taken = machine
                    .holds(cond)
                    .ok_or(format!("`{line}` tests flags the block does not know"))?;
                jump(taken)?
            }
            ("cbz" | "cbnz", [tested, _]) => {
                let Value::Known(value) = machine.read(tested)? else {
                    return Err(format!("`{line}` tests a value the block does not know"));
                };
                jump((value == 0) == (mnemonic == "cbz"))?
            }
            ("tbz" | "tbnz", [tested, bit, _]) => {
                let (Value::Known(value), Value::Known(bit)) =
                    (machine.read(tested)?, machine.read(bit)?)
                else {
                    return Err(format!("`{line}` tests a value the block does not know"));
                };
                let clear = value.checked_shr(bit as u32).unwrap_or(0) & 1 == 0;
                jump(clear == (mnemonic == "tbz"))?
            }
            _ => return Err(format!("`{line}` in the prefetch block")),
        };
        if let Some(next) = target {
            (at, index) = (next, 0);
        }
    }
    let rows_after = machine.read(seeds.rows)?;
    Ok((hints, rows_after))
}

/// Both operands of a compare when they are known numbers (or the same
/// unknown base, whose offsets compare).
fn known_pair(lhs: Value, rhs: Value, word: bool) -> Option<(u64, u64, bool)> {
    match (lhs, rhs) {
        (Value::Known(a), Value::Known(b)) => Some((a, b, word)),
        (Value::Entry(a, x), Value::Entry(b, y)) if a == b => Some((x as u64, y as u64, word)),
        _ => None,
    }
}

/// The prefetch block of a kernel (the sections `head..setup`: the
/// row-panel head up to the depth setup, which the skip `tbz <prefetch>,
/// #0, …` reaches) against the contract, for every row count `1..=64` and
/// the row counts [`FULL_PANEL_ROWS`] of a panel with more rows after it,
/// every count of live columns `1..=nr` and `prefetch` 0 and `PREFETCH_C`:
/// with the bit set the hints are, relative to the first, exactly
/// `c · ldc + 64 · t` bytes for every live column `c` and every live tile
/// `t < ceil(min(rows, 64) / tile_rows)`, each once; with it clear there is
/// none; and the row count is unchanged at the depth setup. Returns the
/// problems (empty when the block agrees everywhere; the first combination
/// that differs per kind of difference).
pub(super) fn prefetch_problems(
    sections: &[Section<'_>],
    (head, setup): (usize, usize),
    seeds: PrefetchSeeds<'_>,
    nr: usize,
    tile_rows: usize,
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut note = |problem: String| {
        if !problems.contains(&problem) && problems.len() < 4 {
            problems.push(problem);
        }
    };
    for rows in (1..=MR as u64).chain(FULL_PANEL_ROWS) {
        let tiles = rows.min(MR as u64).div_ceil(tile_rows.max(1) as u64);
        for columns in 1..=nr as u64 {
            for bits in [0, 1] {
                let (hints, rows_after) =
                    match run_block(sections, head, setup, seeds, (rows, columns, bits)) {
                        Ok(result) => result,
                        Err(problem) => {
                            note(problem);
                            continue;
                        }
                    };
                if rows_after != Value::Known(rows) {
                    note(format!(
                        "the prefetch block leaves the row count {rows} as {rows_after:?}"
                    ));
                }
                let mut want: Vec<i64> = Vec::new();
                if bits == 1 {
                    for column in 0..columns {
                        for tile in 0..tiles {
                            want.push((column * LDC_BYTES + tile * VECTOR_BYTES) as i64);
                        }
                    }
                }
                let got = relative(&hints);
                let mut sorted = got.clone().unwrap_or_default();
                sorted.sort_unstable();
                if got.is_none() || sorted != want {
                    note(format!(
                        "with {rows} rows, {columns} columns and prefetch {bits} the prefetch \
                         block hints at {} (bytes from the first hint, column stride ldc), the \
                         live part is {} hints",
                        describe(got.as_deref()),
                        want.len()
                    ));
                }
            }
        }
    }
    problems
}

/// The hints relative to the first one, when all share one base.
fn relative(hints: &[Value]) -> Option<Vec<i64>> {
    let Some(Value::Entry(base, first)) = hints.first().copied() else {
        return hints.is_empty().then(Vec::new);
    };
    hints
        .iter()
        .map(|hint| match *hint {
            Value::Entry(other, offset) if other == base => Some(offset - first),
            _ => None,
        })
        .collect()
}

/// The hints of [`relative`] as `[column:offset, …]` for a message.
fn describe(hints: Option<&[i64]>) -> String {
    let Some(hints) = hints else {
        return "addresses of different bases".to_string();
    };
    let shown: Vec<String> = hints
        .iter()
        .map(|&hint| {
            let column = hint.div_euclid(LDC_BYTES as i64);
            format!("{column}:{}", hint.rem_euclid(LDC_BYTES as i64))
        })
        .collect();
    format!("[{}]", shown.join(", "))
}

/// The registers the switch to the buffer and its checks are about.
#[derive(Clone, Copy, Debug)]
pub(super) struct SwitchRegisters<'r> {
    /// `a` (A of the call, then the buffer).
    pub(super) a: &'r str,
    /// `a_depth` in bytes.
    pub(super) a_depth: &'r str,
    /// `a_panel` in bytes.
    pub(super) a_panel: &'r str,
    /// `a_pack` (the gate).
    pub(super) a_pack: &'r str,
    /// `kc`.
    pub(super) kc: &'r str,
}

/// The switch of A to the buffer (see the module docs) in the section
/// `row_advance` after its branch back to the row-panel head, with the
/// column-panel advance right after it (`column_advance`) and the
/// column-panel head `column_head`, for elements of `element_bytes`; and
/// the writes of its registers in the body (`sections`) and the scaling of
/// the strides in the prologue (`head`). Returns the problems (empty when
/// the switch is exactly the contract's); the switch's `cbz` goes into
/// `accounted`.
pub(super) fn switch_problems(
    sections: &[Section<'_>],
    head: &[&str],
    (column_head, row_advance, column_advance): (usize, usize, usize),
    registers: SwitchRegisters<'_>,
    element_bytes: usize,
    accounted: &mut BTreeSet<(usize, usize)>,
) -> Vec<String> {
    let mut problems = Vec::new();
    let SwitchRegisters {
        a,
        a_depth,
        a_panel,
        a_pack,
        kc,
    } = registers;
    let step_bytes = (MR * element_bytes) as u64;
    let lines = sections
        .get(row_advance)
        .map(|(_, lines)| lines.as_slice())
        .unwrap_or_default();
    // The switch follows the branch back to the row-panel head.
    let back = lines
        .iter()
        .position(|line| branch_target(line).is_some_and(|(_, forward)| !forward));
    let tail = back
        .and_then(|back| lines.get(back + 1..))
        .unwrap_or_default();
    let gate = tail
        .first()
        .and_then(|line| match operands_of(line, "cbz")?[..] {
            [tested, _] if same_register(tested, a_pack) => Some(*line),
            _ => None,
        });
    let Some(gate) = gate else {
        problems.push(format!(
            "no switch to the buffer: the row-panel advance does not continue with `cbz \
             {a_pack}, …` after its loop"
        ));
        return problems;
    };
    if let Some(back) = back {
        accounted.insert((row_advance, back + 1));
    }
    if branch_section(sections, row_advance, gate) != Some(column_advance) {
        problems.push(format!(
            "the switch's `{gate}` does not skip to the column-panel advance"
        ));
    }
    let want: BTreeSet<String> = [
        format!("mov {a}, {a_pack}"),
        format!("mov {a_depth}, #{step_bytes}"),
        format!("lsl {a_panel}, {kc}, #{}", step_bytes.trailing_zeros()),
        format!("mov {a_pack}, #0"),
    ]
    .into_iter()
    .collect();
    let got: BTreeSet<String> = tail
        .get(1..)
        .unwrap_or_default()
        .iter()
        .map(|line| (*line).to_string())
        .collect();
    if got != want || tail.len() != want.len() + 1 {
        let missing: Vec<&String> = want.difference(&got).collect();
        let extra: Vec<&String> = got.difference(&want).collect();
        problems.push(format!(
            "the switch to the buffer is {:?}: missing {missing:?}, not the contract's {extra:?}",
            tail.get(1..).unwrap_or_default()
        ));
    }
    // The prologue gives the strides in bytes; nothing but the prologue and
    // the switch writes the registers of A and the gate.
    let scale = element_bytes.trailing_zeros();
    for register in [a_depth, a_panel] {
        let scaled = format!("lsl {register}, {register}, #{scale}");
        if !head.contains(&scaled.as_str()) {
            problems.push(format!(
                "the prologue does not scale {register} to bytes (`{scaled}`)"
            ));
        }
    }
    for (at, (_, lines)) in sections.iter().enumerate() {
        for (index, line) in lines.iter().enumerate() {
            let in_switch = at == row_advance && back.is_some_and(|back| index > back + 1);
            let (mnemonic, operands) = instruction(line);
            let written = writes_first_operand(mnemonic)
                .then(|| operands.first().copied())
                .flatten();
            let writes = |register: &str| written.is_some_and(|to| same_register(to, register));
            if !in_switch && [a, a_depth, a_panel, a_pack].into_iter().any(writes) {
                problems.push(format!(
                    "`{line}` (section {}) writes a register of A or the gate outside the switch",
                    section_name(sections, at)
                ));
            }
        }
    }
    // The column-panel head starts every column panel at `a` (the buffer
    // after the switch), and the row-panel advance steps by `a_panel`.
    let lines_of = |at: usize| {
        sections
            .get(at)
            .map(|(_, lines)| lines.as_slice())
            .unwrap_or_default()
    };
    let starts_at_a = lines_of(column_head).iter().any(|line| {
        operands_of(line, "mov")
            .is_some_and(|operands| matches!(operands[..], [_, from] if same_register(from, a)))
    });
    if !starts_at_a {
        problems.push(format!(
            "the column-panel head does not start the row panels' A at {a}"
        ));
    }
    let steps_by_panel = lines_of(row_advance).iter().any(|line| {
        operands_of(line, "add").is_some_and(|operands| {
            matches!(operands[..], [to, again, by] if to == again && same_register(by, a_panel))
        })
    });
    if !steps_by_panel {
        problems.push(format!(
            "the row-panel advance does not step A by {a_panel}"
        ));
    }
    problems
}

/// Whether `mnemonic` makes a path: a branch, or a conditional select or
/// compare (a branch without a jump: what it computes depends on the flags).
fn makes_path(mnemonic: &str) -> bool {
    mnemonic == "b"
        || mnemonic.starts_with("b.")
        || matches!(
            mnemonic,
            "cbz" | "cbnz" | "tbz" | "tbnz" | "br" | "ret" | "bl" | "blr"
        )
        || matches!(
            mnemonic,
            "csel"
                | "csinc"
                | "csinv"
                | "csneg"
                | "cset"
                | "csetm"
                | "cinc"
                | "cinv"
                | "cneg"
                | "ccmp"
                | "ccmn"
        )
}

/// The first branch or conditional select of the body (`body` sections)
/// that is not in `accounted`: (section, line). A conditional select is a
/// path as much as a branch is (its value depends on the flags); the
/// derivation accounts the clamp of the live columns only.
pub(super) fn unknown_branch(
    sections: &[Section<'_>],
    accounted: &BTreeSet<(usize, usize)>,
) -> Option<(String, String)> {
    sections.iter().enumerate().find_map(|(at, (_, lines))| {
        lines.iter().enumerate().find_map(|(index, line)| {
            let (mnemonic, _) = instruction(line);
            (makes_path(mnemonic) && !accounted.contains(&(at, index)))
                .then(|| (section_name(sections, at), (*line).to_string()))
        })
    })
}

/// The mnemonic and operands of the last instruction before line `index` of
/// `lines` that sets the flags (the ones a conditional branch there tests,
/// when they are set in its own section).
fn last_flags<'l>(lines: &[&'l str], index: usize) -> Option<(&'l str, Vec<&'l str>)> {
    lines
        .get(..index)
        .unwrap_or_default()
        .iter()
        .rev()
        .map(|line| instruction(line))
        .find(|(mnemonic, _)| matches!(*mnemonic, "cmp" | "subs" | "cmn" | "adds" | "tst" | "ands"))
}

/// The registers (and the tile height) that the tile chains and the column
/// loops of a kernel's prefetch block and epilogues are about.
#[derive(Clone, Copy, Debug)]
pub(super) struct LoopRegisters<'r> {
    /// The row count of the row panel (`m - i0`).
    pub(super) rows: &'r str,
    /// `m`.
    pub(super) m: &'r str,
    /// `i0`.
    pub(super) i0: &'r str,
    /// The live columns of the column panel.
    pub(super) columns: &'r str,
    /// Rows of one ZA tile.
    pub(super) tile_rows: usize,
}

/// Whether `line` writes the general-purpose register `register`.
fn writes_register(line: &str, register: &str) -> bool {
    let (mnemonic, operands) = instruction(line);
    writes_first_operand(mnemonic)
        && operands
            .first()
            .is_some_and(|first| same_register(first, register))
}

/// Whether the conditional branch back at line `index` of section `at` to
/// section `to` closes a column loop over the live columns `columns` in one
/// of the two forms of the kernels: `cmp <r>, <columns>` and `b.lo`, with
/// `add <r>, <r>, #1` the one write of `<r>` in the loop (an up-counter); or
/// `subs <r>, <r>, #1` and `b.ne`, with no other write of `<r>` in the loop
/// and `mov <r>, <columns>` the last write of `<r>` before it in text order
/// (a down-counter from the live columns). So the loop runs once per live
/// column, whatever the row count (a back branch on anything else, such as
/// the row count against the columns plus a constant, is a path of its own).
fn column_loop(
    sections: &[Section<'_>],
    (to, at, index): (usize, usize, usize),
    columns: &str,
) -> bool {
    let Some((_, lines)) = sections.get(at) else {
        return false;
    };
    let Some(&branch) = lines.get(index) else {
        return false;
    };
    let (condition, _) = instruction(branch);
    // The lines of the loop: from the start of section `to` up to the branch.
    let body: Vec<&str> = sections
        .get(to..=at)
        .unwrap_or_default()
        .iter()
        .zip(to..)
        .flat_map(|((_, lines), section)| {
            let end = if section == at { index } else { lines.len() };
            lines.get(..end).unwrap_or_default().iter().copied()
        })
        .collect();
    let writes_in_loop = |register: &str| -> Vec<&str> {
        body.iter()
            .copied()
            .filter(|line| writes_register(line, register))
            .collect()
    };
    match last_flags(lines, index) {
        Some(("cmp", operands)) => match operands[..] {
            [counter, bound] if condition == "b.lo" && same_register(bound, columns) => {
                matches!(writes_in_loop(counter)[..], [step]
                    if matches!(operands_of(step, "add").as_deref(), Some([a, b, "#1"])
                        if same_register(a, counter) && same_register(b, counter)))
            }
            _ => false,
        },
        Some(("subs", operands)) => match operands[..] {
            [counter, again, "#1"] if condition == "b.ne" && same_register(counter, again) => {
                let only_the_count = writes_in_loop(counter).len() == 1;
                let started = sections
                    .get(..to)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .flat_map(|(_, lines)| lines.iter().rev())
                    .find(|line| writes_register(line, counter))
                    .and_then(|line| operands_of(line, "mov"))
                    .is_some_and(|operands| {
                        matches!(operands[..], [_, from] if same_register(from, columns))
                    });
                only_the_count && started
            }
            _ => false,
        },
        _ => false,
    }
}

/// Accounts the branches of the prefetch block (the sections `head..setup`,
/// whose skip the caller accounts), with the row count, the live columns and
/// the tile height of `registers`: a conditional branch forward inside the
/// block or to `setup` after `cmp <rows>, #k` with `k` a tile threshold (the
/// end of the tile chain; a row count from 64 on decides it as 64 does), and
/// a conditional branch back to a section of the block that closes a column
/// loop ([`column_loop`]; its count is evaluated as well). Anything else, a
/// compare of the row count with another number, a branch on another
/// register or on another bit of `prefetch` among it, is left for
/// [`unknown_branch`].
pub(super) fn account_prefetch(
    sections: &[Section<'_>],
    (head, setup): (usize, usize),
    registers: LoopRegisters<'_>,
    accounted: &mut BTreeSet<(usize, usize)>,
) {
    let LoopRegisters {
        rows,
        columns,
        tile_rows,
        ..
    } = registers;
    for at in head..setup {
        let Some((_, lines)) = sections.get(at) else {
            continue;
        };
        for (index, line) in lines.iter().enumerate() {
            let (mnemonic, _) = instruction(line);
            let (Some((_, forward)), Some(to)) =
                (branch_target(line), branch_section(sections, at, line))
            else {
                continue;
            };
            if !mnemonic.starts_with("b.") {
                continue;
            }
            let fine = if forward {
                let chain_end = last_flags(lines, index).is_some_and(|(compare, operands)| {
                    compare == "cmp"
                        && matches!(operands[..], [lhs, rhs]
                            if same_register(lhs, rows) && tile_threshold_immediate(rhs, tile_rows))
                });
                chain_end && (head..=setup).contains(&to)
            } else {
                (head..setup).contains(&to) && column_loop(sections, (to, at, index), columns)
            };
            if fine {
                accounted.insert((at, index));
            }
        }
    }
}

/// Accounts the branches of the epilogue region `region` (the sections
/// after the epilogue dispatch up to the row-panel advance `row_advance`),
/// whose epilogues start at `starts`, with the row count, `m`, `i0`, the
/// live columns and the tile height of `registers`: a branch back to a
/// section of its own epilogue that closes a column loop ([`column_loop`];
/// `compare_epilogues` requires one bound per loop as well), a branch
/// forward to a section of its own epilogue or to the row-panel advance after
/// the end of a tile chain, and `b` to the row-panel advance. The end of a
/// tile chain is `cmp <rows>, #k` with `k` a tile threshold, or
/// `cmp <r>, <m>` where `r` holds `i0 + k` (a tile's first row, from `mov
/// <r>, <i0>` and `add <r>, <r>, #step` earlier in the epilogue, in text
/// order) with `k` a tile threshold: a row count from 64 on decides both as
/// 64 does. Anything else is left for [`unknown_branch`].
pub(super) fn account_epilogues(
    sections: &[Section<'_>],
    (region, starts): (core::ops::Range<usize>, &[usize]),
    registers: LoopRegisters<'_>,
    row_advance: usize,
    accounted: &mut BTreeSet<(usize, usize)>,
) {
    let LoopRegisters {
        rows,
        m,
        i0,
        columns,
        tile_rows,
    } = registers;
    let span = |at: usize| {
        let start = starts.iter().rev().find(|&&start| start <= at).copied()?;
        let end = starts
            .iter()
            .copied()
            .find(|&next| next > start)
            .unwrap_or(region.end);
        Some(start..end)
    };
    // `i0 + offset` held by each general-purpose register, in text order
    // from the start of the current epilogue (`None`: anything else).
    let fresh = || {
        let mut rows_from_i0: [Option<usize>; 31] = [None; 31];
        if let Some(slot) = register_number(i0).and_then(|n| rows_from_i0.get_mut(n as usize)) {
            *slot = Some(0);
        }
        rows_from_i0
    };
    let mut rows_from_i0 = fresh();
    for at in region.clone() {
        let Some(own) = span(at) else {
            continue;
        };
        if own.start == at {
            rows_from_i0 = fresh();
        }
        let Some((_, lines)) = sections.get(at) else {
            continue;
        };
        for (index, line) in lines.iter().enumerate() {
            let (mnemonic, operands) = instruction(line);
            let offset_of = |register: &str| {
                register_number(register)
                    .and_then(|n| rows_from_i0.get(n as usize).copied().flatten())
            };
            if let (Some((_, forward)), Some(to)) =
                (branch_target(line), branch_section(sections, at, line))
            {
                let conditional = mnemonic.starts_with("b.");
                let fine = if mnemonic == "b" {
                    forward && to == row_advance
                } else if !conditional {
                    false
                } else if !forward {
                    own.contains(&to) && column_loop(sections, (to, at, index), columns)
                } else {
                    // The row count against a tile threshold, or a tile's
                    // first row against m (the same register may be both:
                    // the f64 kernel counts the tiles' rows in its row-count
                    // register).
                    let chain_end = last_flags(lines, index).is_some_and(|(compare, operands)| {
                        compare == "cmp"
                            && matches!(operands[..], [lhs, rhs]
                                if (same_register(lhs, rows)
                                    && tile_threshold_immediate(rhs, tile_rows))
                                    || (same_register(rhs, m)
                                        && offset_of(lhs)
                                            .is_some_and(|k| tile_threshold(k, tile_rows))))
                    });
                    chain_end && (own.contains(&to) || to == row_advance)
                };
                if fine {
                    accounted.insert((at, index));
                }
            }
            // Follow `i0 + offset` through the line's write, if any.
            if !writes_first_operand(mnemonic) {
                continue;
            }
            let Some(to) = operands.first().and_then(|to| register_number(to)) else {
                continue;
            };
            let value = match (mnemonic, &operands[..]) {
                ("mov", [_, from]) => offset_of(from),
                ("add", [_, from, step]) => offset_of(from).and_then(|offset| {
                    let step: usize = step.strip_prefix('#')?.parse().ok()?;
                    offset.checked_add(step)
                }),
                _ => None,
            };
            if let Some(slot) = rows_from_i0.get_mut(to as usize) {
                *slot = value;
            }
        }
    }
}
