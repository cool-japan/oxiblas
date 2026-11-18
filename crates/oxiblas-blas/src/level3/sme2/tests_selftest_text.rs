//! The model of the self-test's path table against the kernel text: the
//! [`KernelPaths`] descriptions of `tests_selftest_paths.rs` are tied to the
//! macro-kernels here.
//!
//! [`derive_structure`] reads the structure from the registered template
//! (the exact text that is assembled; its docs list every feature of the
//! text it reads and what it takes from constants instead) and
//! [`compare_with_model`] requires it to equal the description, so a change
//! of the text in what it reads fails here until the description, the
//! table's `paths_of_call` and the sweep follow it. A change inside a
//! section that the derivation does not interpret (the operands of a load,
//! say) is not seen here; the kernels' own tests and the self-test under
//! QEMU see what it computes.
//!
//! Everything here is plain Rust and runs on every host.

use std::collections::BTreeSet;

use super::args::{MR, SmeElement};
use super::tests_selftest_paths::{F32_PATHS, F64_PATHS, KernelPaths, kernel_template};
use super::tests_selftest_text_online::{
    LoopRegisters, PrefetchSeeds, SwitchRegisters, account_epilogues, account_prefetch,
    prefetch_problems, switch_problems, unknown_branch,
};

// ---------------------------------------------------------------------------
// The model against the kernel text
// ---------------------------------------------------------------------------

/// Bytes of one streaming vector (SVL 512, the only length the backend
/// accepts): one A vector per tile and one B row per depth step.
const VECTOR_BYTES: usize = 64;

/// One row variant as the kernel text shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VariantText {
    /// The label of its zeroing (`41`, ...).
    label: String,
    /// Live tiles: the ZA tiles its depth loop accumulates (`za0..`).
    tiles: usize,
    /// Depth steps per pass of its depth loop.
    steps_per_pass: usize,
    /// Depth steps of its remainder.
    remainder_steps: usize,
}

/// The epilogue of one mode as the kernel text shows it: the sections from
/// the label that the epilogue dispatch reaches for the mode up to the next
/// mode's entry or the row-panel advance.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EpilogueText {
    /// The label the dispatch reaches.
    entry: String,
    /// The ZA tiles its lines name (`za<t>h`), in order.
    chain: Vec<usize>,
    /// Its column loops: branches back to a label inside it.
    column_loops: usize,
    /// Its reads of the live-column count (the clamped register, as `x` or
    /// `w`): the bound of a column loop.
    column_bounds: usize,
}

/// A variant dispatch as the kernel text shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DispatchText {
    /// The immediates it compares the row count with, in order.
    thresholds: Vec<usize>,
    /// For each row count `1..=64` of a row panel, the live tiles of the
    /// variant of its kind it reaches (`None`: a label that starts no such
    /// variant, or a line it cannot decide).
    dispatch: Vec<Option<usize>>,
    /// Its lines that it cannot decide on the row count alone.
    problems: Vec<String>,
}

/// The online packing pass as the kernel text shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PackingText {
    /// The packing twins, in text order.
    twins: Vec<VariantText>,
    /// The packing dispatch (the depth setup after the gate).
    dispatch: DispatchText,
    /// Problems of the variants' stores and buffer advances, the switch and
    /// the registers it writes, in derivation order.
    problems: Vec<String>,
}

/// The prefetch block as the kernel text shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PrefetchText {
    /// Where its hints differ from the live part of C, or what its
    /// evaluation cannot decide (empty when they agree everywhere).
    problems: Vec<String>,
}

/// The structure of a macro-kernel as its template text shows it (see
/// [`derive_structure`]). [`compare_with_model`] judges it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TextStructure {
    /// Bytes of one element (the `.s` or `.d` of the outer products).
    element_bytes: usize,
    /// ZA tiles of that size that the outer products use.
    tiles: usize,
    /// Rows of one tile: elements of one 64-byte vector (one A vector per
    /// tile and depth step).
    tile_rows: usize,
    /// Columns of one column panel: the step of the column-panel loop.
    nr: usize,
    /// The clamp of the live columns of a panel: `min(clamp, n - j0)`.
    column_clamp: usize,
    /// Columns the C pointer advances per column panel.
    c_column_step: usize,
    /// Rows the row-panel loop advances.
    row_step: usize,
    /// The row variants of the plain depth loops, in text order.
    variants: Vec<VariantText>,
    /// The register that holds the row count of a row panel (`m - i0`).
    rows_register: String,
    /// The immediates the variant dispatch compares the row count with, in
    /// order.
    row_thresholds: Vec<usize>,
    /// For each row count `1..=64` of a row panel, the live tiles of the
    /// variant the dispatch reaches (`None`: a label that starts no variant,
    /// or a line the dispatch cannot decide).
    row_dispatch: Vec<Option<usize>>,
    /// Lines of the variant dispatch that it cannot decide on the row count
    /// alone (a test of another register, a write of the row count).
    row_dispatch_problems: Vec<String>,
    /// The online packing pass, when the depth setup has the gate.
    packing: Option<PackingText>,
    /// The prefetch block, when the row-panel head skips it on bit 0 of
    /// `prefetch`.
    prefetch: Option<PrefetchText>,
    /// Problems of stores and buffer advances in a text without the gate.
    online_problems: Vec<String>,
    /// How many different sections the variants' remainders leave to, by
    /// their bit test of `kc` and at their end (one: the epilogue dispatch).
    remainder_exits: usize,
    /// The register that holds the mode.
    mode_register: String,
    /// Lines of the epilogue dispatch that it cannot decide on the mode
    /// alone.
    mode_dispatch_problems: Vec<String>,
    /// The labels of the sections after the epilogue dispatch, up to the
    /// row-panel advance, that are a mode's entry or are entered from
    /// outside the epilogue they lie in, in text order.
    epilogue_sections: Vec<String>,
    /// The register that holds the live columns of a panel.
    columns_register: String,
    /// The epilogues of modes M0 to M3 (`None` for a mode the dispatch
    /// sends nowhere it can name).
    epilogues: Vec<Option<EpilogueText>>,
    /// Sections with outer products outside every variant's depth loop and
    /// remainder: (label, outer products).
    stray_products: Vec<(String, usize)>,
    /// Bit tests of `kc` outside the first line of a variant's remainder:
    /// (label, line).
    stray_steps: Vec<(String, String)>,
    /// The first branch that no part of the derivation interprets:
    /// (section, line).
    unknown_branch: Option<(String, String)>,
}

/// The register of `ldr <reg>, [x0, #{<field>}]` among `lines`.
fn field_register<'t>(lines: &[&'t str], field: &str) -> Result<&'t str, String> {
    let suffix = format!(", [x0, #{{{field}}}]");
    lines
        .iter()
        .copied()
        .find_map(|line: &'t str| line.strip_prefix("ldr ")?.strip_suffix(suffix.as_str()))
        .ok_or_else(|| format!("no load of {field} in the prologue"))
}

/// The ZA tile number and element letter of a `za<t>.<s|d>` or
/// `za<t>h.<s|d>[...]` operand at the start of `operand`.
fn za_tile(operand: &str) -> Option<(usize, char)> {
    let rest = operand.strip_prefix("za")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let tile = digits.parse().ok()?;
    let rest = rest.get(digits.len()..)?;
    let rest = rest.strip_prefix('h').unwrap_or(rest);
    let letter = rest.strip_prefix('.')?.chars().next()?;
    Some((tile, letter))
}

/// The ZA tiles and element letters an instruction names (`zero`, `fmopa`,
/// `mova`, a slice store), in order.
fn za_tiles(line: &str) -> Vec<(usize, char)> {
    line.split([' ', ',', '{', '}'])
        .filter_map(za_tile)
        .collect()
}

/// The tile of one outer-product instruction (`fmopa za<t>.<x>, ...`).
fn fmopa_tile(line: &str) -> Option<(usize, char)> {
    za_tile(line.strip_prefix("fmopa ")?)
}

/// The ZA tiles a `zero {...}` names, for `tiles` tiles of the element size
/// (`{za}` names all of them).
fn zeroed_tiles(line: &str, tiles: usize) -> Option<BTreeSet<usize>> {
    let list = line.strip_prefix("zero {")?.strip_suffix('}')?;
    if list.trim() == "za" {
        return Some((0..tiles).collect());
    }
    Some(za_tiles(list).into_iter().map(|(tile, _)| tile).collect())
}

/// The base register of a load or store `... [base]` / `[base, ...]`.
fn base_register(line: &str) -> Option<&str> {
    let address = line.split_once('[')?.1;
    address.split([',', ']']).next().map(str::trim)
}

/// Whether `line` is a numeric label (`42:`); its number.
fn label_of(line: &str) -> Option<&str> {
    line.trim()
        .strip_suffix(':')
        .filter(|label| !label.is_empty() && label.chars().all(|c| c.is_ascii_digit()))
}

/// A section of a body: its label (none for the lines before the first
/// label) and its lines up to the next label.
pub(super) type Section<'t> = (Option<&'t str>, Vec<&'t str>);

/// The sections of a body, cut at its numeric label lines.
fn sections<'t>(body: &[&'t str]) -> Vec<Section<'t>> {
    let mut sections: Vec<Section<'t>> = vec![(None, Vec::new())];
    for &line in body {
        match (label_of(line), sections.last_mut()) {
            (Some(label), _) => sections.push((Some(label), Vec::new())),
            (None, Some((_, lines))) => lines.push(line),
            (None, None) => {}
        }
    }
    sections
}

/// The label of section `at` for a message.
pub(super) fn section_name(sections: &[Section<'_>], at: usize) -> String {
    match sections.get(at) {
        Some((Some(label), _)) => (*label).to_string(),
        _ => format!("(section {at}, unlabelled)"),
    }
}

/// The outer products among `lines`.
fn products(lines: &[&str]) -> usize {
    lines
        .iter()
        .filter(|line| fmopa_tile(line).is_some())
        .count()
}

/// The mnemonic and the comma-separated operands of an instruction (enough
/// for the plain forms the derivation interprets: `cmp`, `add`, `sub`,
/// `mov`, `csel`, branches).
pub(super) fn instruction(line: &str) -> (&str, Vec<&str>) {
    let (mnemonic, rest) = line.split_once(' ').unwrap_or((line, ""));
    let operands = rest
        .split(',')
        .map(str::trim)
        .filter(|operand| !operand.is_empty())
        .collect();
    (mnemonic, operands)
}

/// The operands of `line` when its mnemonic is `mnemonic`.
pub(super) fn operands_of<'l>(line: &'l str, mnemonic: &str) -> Option<Vec<&'l str>> {
    let (found, operands) = instruction(line);
    (found == mnemonic).then_some(operands)
}

/// The number of an `x<n>` or `w<n>` register operand.
pub(super) fn register_number(operand: &str) -> Option<u32> {
    let digits = operand
        .strip_prefix('x')
        .or_else(|| operand.strip_prefix('w'))?;
    digits.parse().ok()
}

/// Whether two operands name the same general-purpose register (as `x<n>`
/// or `w<n>`).
pub(super) fn same_register(a: &str, b: &str) -> bool {
    register_number(a).is_some_and(|number| register_number(b) == Some(number))
}

/// Whether an instruction with this mnemonic writes its first operand (not a
/// compare, a branch, a store or `zero`).
pub(super) fn writes_first_operand(mnemonic: &str) -> bool {
    !(matches!(mnemonic, "cmp" | "cmn" | "tst" | "b" | "zero" | "prfm")
        || mnemonic.starts_with("b.")
        || mnemonic.starts_with("cb")
        || mnemonic.starts_with("tb")
        || mnemonic.starts_with("st"))
}

/// Whether `line` reads `register` (as `x<n>` or `w<n>`): any register
/// token but the destination of an instruction that writes one.
fn reads_register(line: &str, register: &str) -> bool {
    let (mnemonic, _) = instruction(line);
    let skip = usize::from(writes_first_operand(mnemonic));
    line.split([' ', ',', '[', ']', '{', '}'])
        .filter(|token| !token.is_empty())
        .skip(1 + skip)
        .any(|token| same_register(token, register))
}

/// The target of a branch (`b`, `b.<cond>`, `cbz`, `cbnz`, `tbz`, `tbnz`):
/// its label and whether the reference is forward (`<n>f`).
pub(super) fn branch_target(line: &str) -> Option<(&str, bool)> {
    let (mnemonic, operands) = instruction(line);
    let branch = mnemonic == "b"
        || mnemonic.starts_with("b.")
        || matches!(mnemonic, "cbz" | "cbnz" | "tbz" | "tbnz");
    if !branch {
        return None;
    }
    let target = *operands.last()?;
    let digits = |label: &str| label.chars().all(|c| c.is_ascii_digit()) && !label.is_empty();
    match (target.strip_suffix('f'), target.strip_suffix('b')) {
        (Some(label), _) if digits(label) => Some((label, true)),
        (_, Some(label)) if digits(label) => Some((label, false)),
        _ => None,
    }
}

/// The section that a reference to `label` from section `from` names: the
/// next section with that label for a forward reference, the last one at or
/// before `from` for a backward one (numeric labels may repeat).
fn resolve(sections: &[Section<'_>], from: usize, label: &str, forward: bool) -> Option<usize> {
    let named = |(at, (name, _)): (usize, &Section<'_>)| (*name == Some(label)).then_some(at);
    if forward {
        sections.iter().enumerate().skip(from + 1).find_map(named)
    } else {
        sections
            .iter()
            .enumerate()
            .take(from + 1)
            .rev()
            .find_map(named)
    }
}

/// The section a branch in section `from` goes to.
pub(super) fn branch_section(sections: &[Section<'_>], from: usize, line: &str) -> Option<usize> {
    let (label, forward) = branch_target(line)?;
    resolve(sections, from, label, forward)
}

/// Whether the condition `cond` holds after `cmp lhs, rhs`.
pub(super) fn condition(cond: &str, lhs: u64, rhs: u64) -> Option<bool> {
    let (signed_lhs, signed_rhs) = (lhs as i64, rhs as i64);
    Some(match cond {
        "eq" => lhs == rhs,
        "ne" => lhs != rhs,
        "lo" | "cc" => lhs < rhs,
        "hs" | "cs" => lhs >= rhs,
        "hi" => lhs > rhs,
        "ls" => lhs <= rhs,
        "lt" => signed_lhs < signed_rhs,
        "ge" => signed_lhs >= signed_rhs,
        "gt" => signed_lhs > signed_rhs,
        "le" => signed_lhs <= signed_rhs,
        _ => return None,
    })
}

/// Where the compares and branches of `lines` (in section `at`) send the
/// value `value` of `register`: the section reached (`None` for a label that
/// does not resolve), or the next section when no branch is taken. `Err`
/// names a line that the chain cannot decide on `register` alone: a test of
/// another register, a write of `register`, an unknown condition.
fn dispatch_target(
    sections: &[Section<'_>],
    at: usize,
    lines: &[&str],
    register: &str,
    value: u64,
) -> Result<Option<usize>, String> {
    let undecided = |line: &str| Err(line.to_string());
    let mut flags: Option<(u64, u64)> = None;
    for &line in lines {
        let (mnemonic, operands) = instruction(line);
        let tested = operands.first().copied().unwrap_or("");
        match mnemonic {
            "cmp" => {
                let immediate = operands
                    .get(1)
                    .and_then(|operand| operand.strip_prefix('#'))
                    .and_then(|digits| digits.parse::<u64>().ok());
                match immediate {
                    Some(immediate) if same_register(tested, register) => {
                        flags = Some((value, immediate));
                    }
                    _ => return undecided(line),
                }
            }
            "cbz" | "cbnz" => {
                if !same_register(tested, register) {
                    return undecided(line);
                }
                if (value == 0) == (mnemonic == "cbz") {
                    return Ok(branch_section(sections, at, line));
                }
            }
            "b" => return Ok(branch_section(sections, at, line)),
            _ if mnemonic.starts_with("b.") => {
                let cond = mnemonic.get(2..).unwrap_or("");
                match flags.and_then(|(lhs, rhs)| condition(cond, lhs, rhs)) {
                    Some(true) => return Ok(branch_section(sections, at, line)),
                    Some(false) => {}
                    None => return undecided(line),
                }
            }
            _ if mnemonic.starts_with("tb") || mnemonic.starts_with("cb") => {
                return undecided(line);
            }
            _ => {
                if writes_first_operand(mnemonic) && same_register(tested, register) {
                    return undecided(line);
                }
            }
        }
    }
    Ok((at + 1 < sections.len()).then_some(at + 1))
}

/// The structure of the macro-kernel whose registered template is
/// `template` (or an edited copy of it); [`compare_with_model`] judges it.
///
/// It relies on these features of the text, which both kernels have (the
/// loop-nest comment of `kernel_f32.rs`; the generator
/// `tools/gen_sme2_kernel_f64.py` and the loop-nest comment of
/// `kernel_f64.rs`):
///
/// - the body lies between `msr fpsr, xzr` and the exit label `90:` of the
///   frame, and the prologue loads the fields with `ldr <reg>, [x0,
///   #{off_<field>}]`: `kc`, `a_depth`, `m`, `n`, `ldc` and `mode` are known
///   by those registers (an `x` register and its `w` view are the same), and
///   `a`, `a_panel`, `a_pack` and `prefetch` when the text has the online
///   paths;
/// - the body is cut into sections at its numeric labels; a branch names a
///   section by `<label>f` (the next one) or `<label>b` (the last one before);
/// - a row variant is a section that starts with `zero {...}`, followed by
///   the section of its depth loop (which ends with a branch back to its own
///   label) and the section of its remainder (which starts with
///   `tbz <kc>, #0, ...`: the step is taken for an odd `kc` only). The depth
///   steps of one pass of a loop are its outer products (`fmopa`) per
///   accumulated tile, and the same number must show three more ways: the B
///   pointer (the base of the loads governed by `p0/z`) advances by one
///   vector per step, A advances by `a_depth` once per step, and the pass
///   count is `kc >> s` for `2^s` steps (`lsr <counter>, <kc>, #s`). Its
///   zeroing section skips the loop with `cbz <counter>, <its remainder>`;
/// - a variant whose sections store through the buffer register (the
///   register the body start sets with `mov <buf>, <a_pack>`) is a packing
///   twin: every store of its sections goes through that register, which
///   advances by 64 rows of elements per depth step (per pass and in the
///   remainder). Every other variant stores nothing and leaves it alone;
/// - outer products and bit tests of `kc` (`tbz`, `tbnz`) appear nowhere but
///   in the variants' depth loops and remainders (the first line);
/// - the row-panel head is the section that computes the row count with
///   `sub <rows>, <m>, <i0>`; the row-panel advance is the section that
///   branches back to it, and steps `<i0>` with `add <i0>, <i0>, #64`;
/// - with online packing, the depth setup (the section before the first
///   variant) holds the gate `cbz <a_pack>, <plain dispatch>`: the rest of
///   that section is the packing dispatch (falling through to the first
///   twin), the section it names the plain dispatch. Without it the plain
///   dispatch is the section before the first variant (after the row count
///   when that is the row-panel head). Each dispatch compares only the row
///   count with immediates (`cmp <rows>, #k`) and branches (`b.<cond>`, `b`,
///   or falls through); the derivation runs both for every row count
///   `1..=64` and records the variant reached (only twins count for the
///   packing dispatch, only plain variants for the plain one);
/// - the prefetch block: `tbz <prefetch>, #0, <depth setup>` in the
///   row-panel head after the row count, and the sections up to the depth
///   setup, evaluated by `tests_selftest_text_online.rs` for every row
///   count, live column count and both values of the bit;
/// - the switch to the buffer at the end of the row-panel advance and the
///   registers it writes (`tests_selftest_text_online.rs`);
/// - the column-panel loop ends with `cmp <j0>, <n>` and a branch back to
///   the column-panel head, and steps `add <j0>, <j0>, #NR` (`NR` is read
///   here); its head computes the live columns with `sub <cols>, <n>, <j0>`
///   and clamps them with `mov <t>, #K`, `csel <cols>, <cols>, <t>, lo`; C
///   advances by `add <c>, <c>, <ldc>, lsl #s` (`2^s` columns);
/// - every variant's remainder leaves (its `tbz`) to the same section, the
///   epilogue dispatch, which tests only the mode register (`cmp`, `cbz`,
///   `cbnz` and branches); the derivation runs it for the modes 0 to 3.
///   The sections from there up to the row-panel advance belong to the
///   epilogue whose entry precedes them; a section entered from outside its
///   epilogue counts as an epilogue section of its own;
/// - an epilogue's tile chain is the order in which its lines name the ZA
///   tiles (`za<t>h`), its column loops are its branches back to a label
///   inside it, and each of them is bounded by a read of the live columns;
///   its other branches go forward inside it or to the row-panel advance,
///   after a compare of the row count or of `m`;
/// - every branch of the body is one of those named above.
///
/// Taken from constants: `MR = 64` rows per row panel and the 64-byte
/// vector (SVL 512), so a tile holds `64 / element size` rows and a B row as
/// many columns; [`compare_with_model`] checks the text against both.
///
/// `Err` names the first feature that is missing or that cannot be read.
fn derive_structure(template: &str) -> Result<TextStructure, String> {
    let text = template.replace("{{", "{").replace("}}", "}");
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let start = lines
        .iter()
        .position(|line| *line == "msr fpsr, xzr")
        .ok_or("no body start (`msr fpsr, xzr`) in the text")?;
    let stop = lines
        .iter()
        .position(|line| *line == "90:")
        .ok_or("no exit label `90:` in the text")?;
    let head = lines.get(..start).ok_or("no prologue")?;
    let body = lines.get(start + 1..stop).ok_or("no body")?;
    let kc = field_register(head, "off_kc")?;
    let a_depth = field_register(head, "off_a_depth")?;
    let m_register = field_register(head, "off_m")?;
    let n_register = field_register(head, "off_n")?;
    let ldc_register = field_register(head, "off_ldc")?;
    let mode_register = field_register(head, "off_mode")?;
    // The fields of the online paths (a text without them has none).
    let a_pack_register = field_register(head, "off_a_pack").ok();
    let prefetch_register = field_register(head, "off_prefetch").ok();

    // The element size and the tiles of the outer products.
    let all: Vec<(usize, char)> = body.iter().filter_map(|line| fmopa_tile(line)).collect();
    let letters: Vec<char> = all
        .iter()
        .map(|&(_, letter)| letter)
        .collect::<BTreeSet<char>>()
        .into_iter()
        .collect();
    let element_bytes = match letters[..] {
        ['s'] => 4,
        ['d'] => 8,
        _ => return Err(format!("outer products of element sizes {letters:?}")),
    };
    let tiles = all
        .iter()
        .map(|&(tile, _)| tile + 1)
        .max()
        .ok_or("no outer product")?;
    let step_bytes = MR * element_bytes;

    // `kc >> s`: the pass count of every depth loop, in `<counter>`.
    let shifts: Vec<(&str, u32)> = body
        .iter()
        .filter_map(|line| match operands_of(line, "lsr")?[..] {
            [counter, from, shift] if from == kc => {
                Some((counter, shift.strip_prefix('#')?.parse().ok()?))
            }
            _ => None,
        })
        .collect();
    let [(counter, shift)] = shifts[..] else {
        return Err(format!("{} `lsr` of kc ({kc}), want one", shifts.len()));
    };
    let shifted_steps = 1_usize
        .checked_shl(shift)
        .ok_or("a pass count shift out of range")?;

    let sections = sections(body);
    // Branch lines that a part of the structure interprets: (section, line).
    let mut accounted: BTreeSet<(usize, usize)> = BTreeSet::new();
    let lines_of = |at: usize| {
        sections
            .get(at)
            .map(|(_, lines)| lines.as_slice())
            .unwrap_or_default()
    };
    // The buffer register: `mov <buf>, <a_pack>` before the first label.
    let buffer = a_pack_register.and_then(|a_pack| {
        lines_of(0)
            .iter()
            .find_map(|line| match operands_of(line, "mov")?[..] {
                [to, from] if same_register(from, a_pack) => Some(to),
                _ => None,
            })
    });
    // A text that tests `a_pack` must take its buffer register from it.
    if let (None, Some(a_pack)) = (buffer, a_pack_register) {
        let tests = body.iter().find(|line| {
            matches!(operands_of(line, "cbz").as_deref(), Some([tested, _]) if same_register(tested, a_pack))
        });
        if let Some(line) = tests {
            return Err(format!(
                "the body tests a_pack (`{line}`) but does not start a buffer register from it \
                 (`mov <buf>, {a_pack}` before the first label)"
            ));
        }
    }
    let mut online_problems: Vec<String> = Vec::new();

    let mut variants = Vec::new();
    let mut twins = Vec::new();
    // The section of each variant's zeroing, whether it is a twin and its
    // live tiles; the section of each remainder.
    let mut variant_at: Vec<(usize, bool, usize)> = Vec::new();
    let mut remainder_at = Vec::new();
    for (at, (label, section)) in sections.iter().enumerate() {
        let Some(zero) = section.first().filter(|line| line.starts_with("zero {")) else {
            continue;
        };
        let label = label.unwrap_or("?");
        let name = format!("the variant at label {label}");
        let zeroed = zeroed_tiles(zero, tiles).ok_or(format!("{name}: unreadable {zero:?}"))?;
        let (Some((Some(loop_label), body_lines)), Some((_, remainder))) =
            (sections.get(at + 1), sections.get(at + 2))
        else {
            return Err(format!(
                "{name}: no depth loop and remainder after the zeroing"
            ));
        };
        let back = format!("{loop_label}b");
        let closes = body_lines.last().is_some_and(|line| {
            line.ends_with(back.as_str())
                && (line.starts_with("b.") || line.starts_with("cbnz ") || line.starts_with("cbz "))
        });
        if !closes {
            return Err(format!(
                "{name}: the section after the zeroing ({loop_label}) is not a loop"
            ));
        }
        let accumulated: BTreeSet<usize> = body_lines
            .iter()
            .filter_map(|line| fmopa_tile(line))
            .map(|(tile, _)| tile)
            .collect();
        let live = accumulated.len();
        if live == 0 || accumulated != (0..live).collect() || zeroed != accumulated {
            return Err(format!(
                "{name}: zeroes tiles {zeroed:?} and accumulates {accumulated:?}"
            ));
        }
        let in_loop = products(body_lines);
        if !in_loop.is_multiple_of(live) {
            return Err(format!("{name}: {in_loop} outer products for {live} tiles"));
        }
        let steps = in_loop / live;
        // B: the base of the all-true loads, and its step per pass.
        let b_bases: Vec<&str> = body_lines
            .iter()
            .filter(|line| line.starts_with("ld1") && line.contains(" p0/z, "))
            .filter_map(|line| base_register(line))
            .collect::<BTreeSet<&str>>()
            .into_iter()
            .collect();
        let [b] = b_bases[..] else {
            return Err(format!("{name}: B loads from {b_bases:?}"));
        };
        let b_step_prefix = format!("add {b}, {b}, #");
        let b_step: usize = body_lines
            .iter()
            .filter_map(|line| {
                line.strip_prefix(b_step_prefix.as_str())?
                    .parse::<usize>()
                    .ok()
            })
            .sum();
        if b_step != steps * VECTOR_BYTES {
            return Err(format!(
                "{name}: the B pointer advances {b_step} bytes per pass ({} depth rows), \
                 the outer products make {steps} depth steps",
                b_step / VECTOR_BYTES
            ));
        }
        let a_suffix = format!(", {a_depth}");
        let a_steps = body_lines
            .iter()
            .filter(|line| line.starts_with("add ") && line.ends_with(a_suffix.as_str()))
            .count();
        if a_steps != steps {
            return Err(format!(
                "{name}: A advances by a_depth {a_steps} times per pass, \
                 the outer products make {steps} depth steps"
            ));
        }
        if steps != shifted_steps {
            return Err(format!(
                "{name}: {steps} depth steps per pass, but the pass count is kc >> {shift}"
            ));
        }
        let conditional = format!("tbz {kc}, #0, ");
        if !remainder
            .first()
            .is_some_and(|line| line.starts_with(conditional.as_str()))
        {
            return Err(format!(
                "{name}: the remainder does not start with `{conditional}...` (odd kc only)"
            ));
        }
        let rest = products(remainder);
        if !rest.is_multiple_of(live) {
            return Err(format!(
                "{name}: {rest} remainder outer products for {live} tiles"
            ));
        }
        let rest_steps = rest / live;
        // Its branches: the loop skip, the loop's closing branch, the
        // remainder's bit test and closing `b` (where it leaves to is
        // checked with the epilogue dispatch).
        for (index, line) in section.iter().enumerate() {
            let skips = matches!(operands_of(line, "cbz").as_deref(), Some([tested, _]) if *tested == counter);
            if skips && branch_section(&sections, at, line) == Some(at + 2) {
                accounted.insert((at, index));
            }
        }
        accounted.insert((at + 1, body_lines.len() - 1));
        accounted.insert((at + 2, 0));
        if remainder
            .last()
            .is_some_and(|line| instruction(line).0 == "b")
        {
            accounted.insert((at + 2, remainder.len() - 1));
        }
        // Stores and buffer advances: a twin stores only through the
        // buffer register and advances it by its depth steps; a plain
        // variant does neither.
        let own = [
            section.as_slice(),
            body_lines.as_slice(),
            remainder.as_slice(),
        ];
        let stores: Vec<(&str, Option<&str>)> = own
            .iter()
            .flat_map(|lines| lines.iter())
            .filter(|line| line.starts_with("st"))
            .map(|line| (*line, base_register(line)))
            .collect();
        let twin = buffer.is_some_and(|buf| stores.iter().any(|&(_, base)| base == Some(buf)));
        let kind = if twin { "packing twin" } else { "variant" };
        match (twin, buffer) {
            (true, Some(buf)) => {
                if let Some((line, _)) = stores.iter().find(|&&(_, base)| base != Some(buf)) {
                    online_problems.push(format!(
                        "the packing twin at label {label} stores `{line}`, not through the \
                         buffer register {buf}"
                    ));
                }
            }
            _ => {
                if let Some((line, _)) = stores.first() {
                    online_problems.push(format!("the variant at label {label} stores `{line}`"));
                }
            }
        }
        if let Some(buf) = buffer {
            let advance = |lines: &[&str]| -> usize {
                lines
                    .iter()
                    .filter_map(|line| match operands_of(line, "add")?[..] {
                        [to, from, step] if to == buf && from == buf => {
                            step.strip_prefix('#')?.parse::<usize>().ok()
                        }
                        _ => None,
                    })
                    .sum()
            };
            let got = (advance(body_lines), advance(remainder));
            let want = if twin {
                (steps * step_bytes, rest_steps * step_bytes)
            } else {
                (0, 0)
            };
            if got != want {
                online_problems.push(format!(
                    "the {kind} at label {label} advances the buffer register {buf} by {} bytes \
                     per pass and {} in its remainder, want {} and {} ({steps} and {rest_steps} \
                     depth steps of {MR} rows)",
                    got.0, got.1, want.0, want.1
                ));
            }
        }
        let variant = VariantText {
            label: label.to_string(),
            tiles: live,
            steps_per_pass: steps,
            remainder_steps: rest_steps,
        };
        if twin {
            twins.push(variant);
        } else {
            variants.push(variant);
        }
        variant_at.push((at, twin, live));
        remainder_at.push(at + 2);
    }
    let first_variant = variant_at
        .first()
        .map(|&(at, _, _)| at)
        .ok_or("no row variant (no section starts with `zero {...}`)")?;

    // Outer products and bit tests of kc outside the variants.
    let in_variant = |at: usize| {
        variant_at
            .iter()
            .any(|&(zero, _, _)| at == zero + 1 || at == zero + 2)
    };
    let mut stray_products = Vec::new();
    let mut stray_steps = Vec::new();
    for (at, (_, section)) in sections.iter().enumerate() {
        let count = products(section);
        if count > 0 && !in_variant(at) {
            stray_products.push((section_name(&sections, at), count));
        }
        for (index, line) in section.iter().enumerate() {
            let (mnemonic, operands) = instruction(line);
            let on_kc = matches!(mnemonic, "tbz" | "tbnz")
                && operands.first().is_some_and(|reg| same_register(reg, kc));
            let remainder_head = index == 0 && remainder_at.contains(&at);
            if on_kc && !remainder_head {
                stray_steps.push((section_name(&sections, at), (*line).to_string()));
            }
        }
    }

    // The row-panel head: the section that computes the row count.
    let (row_head, rows_line, rows_register, i0_register) = sections
        .iter()
        .enumerate()
        .find_map(|(at, (_, lines))| {
            lines
                .iter()
                .enumerate()
                .find_map(|(index, line)| match operands_of(line, "sub")?[..] {
                    [rows, m, i0] if same_register(m, m_register) => Some((at, index, rows, i0)),
                    _ => None,
                })
        })
        .ok_or("no row count `sub <rows>, <m>, <i0>` in the body")?;

    // The dispatches: with a gate `cbz <a_pack>, …` in the depth setup (the
    // section before the first variant), its rest is the packing dispatch
    // and the section it names the plain one.
    let setup = first_variant
        .checked_sub(1)
        .ok_or("no section before the first variant")?;
    let gate = a_pack_register.and_then(|a_pack| {
        lines_of(setup).iter().position(|line| {
            matches!(operands_of(line, "cbz").as_deref(), Some([tested, _]) if same_register(tested, a_pack))
        })
    });
    let (plain_at, plain_lines) = match gate {
        Some(index) => {
            accounted.insert((setup, index));
            let line = lines_of(setup).get(index).copied().unwrap_or_default();
            let target = branch_section(&sections, setup, line)
                .ok_or(format!("the gate `{line}` names no section"))?;
            (target, lines_of(target))
        }
        None if setup == row_head => (
            setup,
            lines_of(setup).get(rows_line + 1..).unwrap_or_default(),
        ),
        None => (setup, lines_of(setup)),
    };
    let packing_lines = gate.map(|index| lines_of(setup).get(index + 1..).unwrap_or_default());
    // The live tiles of the variant (of the dispatch's kind) at section `at`.
    let tiles_at = |at: usize, twin: bool| {
        variant_at
            .iter()
            .find_map(|&(zero, is_twin, tiles)| (zero == at && is_twin == twin).then_some(tiles))
    };
    let run_dispatch = |at: usize, decide: &[&str], twin: bool| {
        let mut dispatch = Vec::new();
        let mut problems: Vec<String> = Vec::new();
        for rows in 1..=MR as u64 {
            match dispatch_target(&sections, at, decide, rows_register, rows) {
                Ok(target) => dispatch.push(target.and_then(|to| tiles_at(to, twin))),
                Err(line) => {
                    dispatch.push(None);
                    if !problems.contains(&line) {
                        problems.push(line);
                    }
                }
            }
        }
        let thresholds: Vec<usize> = decide
            .iter()
            .filter_map(|line| match operands_of(line, "cmp")?[..] {
                [register, immediate] if same_register(register, rows_register) => {
                    immediate.strip_prefix('#')?.parse().ok()
                }
                _ => None,
            })
            .collect();
        DispatchText {
            thresholds,
            dispatch,
            problems,
        }
    };
    let plain_dispatch = run_dispatch(plain_at, plain_lines, false);
    let packing_dispatch = packing_lines.map(|lines| run_dispatch(setup, lines, true));
    // Every branch of the dispatches is the dispatch's.
    let first_plain = if plain_at == row_head {
        rows_line + 1
    } else {
        0
    };
    for index in first_plain..lines_of(plain_at).len() {
        accounted.insert((plain_at, index));
    }
    if let Some(gate) = gate {
        for index in gate + 1..lines_of(setup).len() {
            accounted.insert((setup, index));
        }
        // The lines of the depth setup before the gate keep the row count.
        if let Some(line) = lines_of(setup)
            .get(..gate)
            .unwrap_or_default()
            .iter()
            .find(|line| {
                let (mnemonic, operands) = instruction(line);
                writes_first_operand(mnemonic)
                    && operands
                        .first()
                        .is_some_and(|to| same_register(to, rows_register))
            })
        {
            online_problems.push(format!("the depth setup writes the row count: `{line}`"));
        }
    }

    // The row-panel advance: the section that branches back to the row-panel
    // head, and its step of i0.
    let (row_advance, row_back) = (row_head + 1..sections.len())
        .find_map(|at| {
            lines_of(at)
                .iter()
                .position(|line| branch_section(&sections, at, line) == Some(row_head))
                .map(|index| (at, index))
        })
        .ok_or(format!(
            "no branch back to the row-panel label {}",
            section_name(&sections, row_head)
        ))?;
    accounted.insert((row_advance, row_back));
    let row_step = immediate_steps(&sections, row_advance..row_advance + 1, i0_register);

    // The column-panel loop: j0 from its closing compare, its step, the clamp
    // of the live columns and the C advance.
    let (column_advance, j0_register) = sections
        .iter()
        .enumerate()
        .rev()
        .find_map(|(at, (_, lines))| {
            lines
                .iter()
                .rev()
                .find_map(|line| match operands_of(line, "cmp")?[..] {
                    [j0, n] if same_register(n, n_register) => Some((at, j0)),
                    _ => None,
                })
        })
        .ok_or("no column-panel compare `cmp <j0>, <n>`")?;
    let column_back = lines_of(column_advance)
        .iter()
        .position(|line| branch_target(line).is_some_and(|(_, forward)| !forward));
    let column_head = column_back
        .and_then(|index| {
            let line = lines_of(column_advance).get(index)?;
            branch_section(&sections, column_advance, line)
        })
        .ok_or("no branch back to the column-panel head")?;
    if let Some(index) = column_back {
        accounted.insert((column_advance, index));
    }
    let nr = immediate_steps(&sections, 0..sections.len(), j0_register);
    let (columns_at, columns_register) = sections
        .iter()
        .enumerate()
        .find_map(|(at, (_, lines))| {
            lines
                .iter()
                .find_map(|line| match operands_of(line, "sub")?[..] {
                    [cols, n, j0]
                        if same_register(n, n_register) && same_register(j0, j0_register) =>
                    {
                        Some((at, cols))
                    }
                    _ => None,
                })
        })
        .ok_or("no live-column count `sub <cols>, <n>, <j0>`")?;
    let column_clamp = column_clamp(&sections, columns_at, columns_register)
        .ok_or("no clamp `mov <t>, #K` and `csel <cols>, <cols>, <t>, lo` of the live columns")?;
    // The clamp's conditional select is the one the model interprets (any
    // other is a path of its own: `unknown_branch`).
    if let Some((_, lines)) = sections.get(columns_at) {
        for (index, line) in lines.iter().enumerate() {
            let clamp = operands_of(line, "csel").is_some_and(|operands| {
                matches!(operands[..], [to, from, _, "lo"]
                    if same_register(to, columns_register) && same_register(from, columns_register))
            });
            if clamp {
                accounted.insert((columns_at, index));
            }
        }
    }
    let loop_registers = LoopRegisters {
        rows: rows_register,
        m: m_register,
        i0: i0_register,
        columns: columns_register,
        tile_rows: VECTOR_BYTES / element_bytes,
    };
    let c_column_step: usize = body
        .iter()
        .filter_map(|line| match operands_of(line, "add")?[..] {
            [c, c_again, ldc, shift] if c == c_again && same_register(ldc, ldc_register) => {
                1_usize.checked_shl(shift.strip_prefix("lsl #")?.parse().ok()?)
            }
            _ => None,
        })
        .sum();

    // The prefetch block: the skip in the row-panel head after the row
    // count, to the depth setup, and the sections in between.
    let prefetch = prefetch_register.and_then(|register| {
        let skip = lines_of(row_head)
            .iter()
            .enumerate()
            .skip(rows_line + 1)
            .find_map(|(index, line)| match operands_of(line, "tbz")?[..] {
                [tested, "#0", _] if same_register(tested, register) => Some((index, *line)),
                _ => None,
            })?;
        let mut problems = Vec::new();
        if branch_section(&sections, row_head, skip.1) != Some(setup) {
            problems.push(format!(
                "the prefetch skip `{}` does not go to the depth setup",
                skip.1
            ));
        }
        accounted.insert((row_head, skip.0));
        account_prefetch(&sections, (row_head, setup), loop_registers, &mut accounted);
        let seeds = PrefetchSeeds {
            rows: rows_register,
            m: m_register,
            i0: i0_register,
            columns: columns_register,
            prefetch: register,
            ldc: ldc_register,
        };
        problems.extend(prefetch_problems(
            &sections,
            (row_head, setup),
            seeds,
            nr,
            VECTOR_BYTES / element_bytes,
        ));
        Some(PrefetchText { problems })
    });

    // The epilogue dispatch: where the remainders leave to, by their bit
    // test of kc and at their end (a closing `b`, or falling through).
    let exits: BTreeSet<Option<usize>> = remainder_at
        .iter()
        .flat_map(|&at| {
            let lines = lines_of(at);
            let skip = lines
                .first()
                .and_then(|line| branch_section(&sections, at, line));
            let end = match lines.last() {
                Some(line) if instruction(line).0 == "b" => branch_section(&sections, at, line),
                _ => (at + 1 < sections.len()).then_some(at + 1),
            };
            [skip, end]
        })
        .collect();
    let epilogue_at = remainder_at
        .first()
        .and_then(|&at| {
            let line = sections.get(at)?.1.first()?;
            branch_section(&sections, at, line)
        })
        .ok_or("the first remainder leaves to no label")?;
    let epilogue_lines = lines_of(epilogue_at);
    for index in 0..epilogue_lines.len() {
        accounted.insert((epilogue_at, index));
    }
    let mut mode_dispatch_problems: Vec<String> = Vec::new();
    let mut entries = Vec::new();
    for mode in 0..4 {
        match dispatch_target(&sections, epilogue_at, epilogue_lines, mode_register, mode) {
            Ok(target) => entries.push(target),
            Err(line) => {
                entries.push(None);
                if !mode_dispatch_problems.contains(&line) {
                    mode_dispatch_problems.push(line);
                }
            }
        }
    }
    let region = epilogue_at + 1..row_advance;
    let (epilogue_sections, epilogues, starts) =
        epilogue_spans(&sections, region.clone(), &entries, columns_register);
    account_epilogues(
        &sections,
        (region, &starts),
        loop_registers,
        row_advance,
        &mut accounted,
    );

    // The online packing pass: the dispatch, the switch, the buffer register.
    let packing = packing_dispatch.map(|dispatch| {
        let mut problems = core::mem::take(&mut online_problems);
        if let (Some(a_pack), Ok(a), Ok(a_panel)) = (
            a_pack_register,
            field_register(head, "off_a"),
            field_register(head, "off_a_panel"),
        ) {
            let registers = SwitchRegisters {
                a,
                a_depth,
                a_panel,
                a_pack,
                kc,
            };
            problems.extend(switch_problems(
                &sections,
                head,
                (column_head, row_advance, column_advance),
                registers,
                element_bytes,
                &mut accounted,
            ));
        } else {
            problems.push("no load of a, a_panel or a_pack for the switch".to_string());
        }
        if let Some(buf) = buffer {
            // The buffer register: set once at the body start, then only the
            // twins' advances (checked with each variant) write it.
            let set = a_pack_register.map(|a_pack| format!("mov {buf}, {a_pack}"));
            for (at, (_, lines)) in sections.iter().enumerate() {
                let in_twin = variant_at
                    .iter()
                    .any(|&(zero, twin, _)| twin && (zero..zero + 3).contains(&at));
                for line in lines {
                    let (mnemonic, operands) = instruction(line);
                    let writes = writes_first_operand(mnemonic)
                        && operands.first().is_some_and(|to| same_register(to, buf));
                    let advance = matches!(
                        operands_of(line, "add").as_deref(),
                        Some([to, from, _]) if *to == buf && *from == buf
                    );
                    let start = at == 0 && set.as_deref() == Some(*line);
                    if writes && !start && !(in_twin && advance) {
                        problems.push(format!(
                            "`{line}` (section {}) writes the buffer register {buf}",
                            section_name(&sections, at)
                        ));
                    }
                }
            }
        } else {
            problems
                .push("no buffer register (`mov <buf>, <a_pack>` at the body start)".to_string());
        }
        PackingText {
            twins: core::mem::take(&mut twins),
            dispatch,
            problems,
        }
    });
    if packing.is_none() {
        // Without a gate, a twin or a buffer is a problem of the plain
        // structure.
        if let Some(twin) = twins.first() {
            online_problems.push(format!(
                "the packing twin at label {} is reached by no packing dispatch",
                twin.label
            ));
        }
    }
    let unknown = unknown_branch(&sections, &accounted);

    Ok(TextStructure {
        element_bytes,
        tiles,
        tile_rows: VECTOR_BYTES / element_bytes,
        nr,
        column_clamp,
        c_column_step,
        row_step,
        variants,
        rows_register: rows_register.to_string(),
        row_thresholds: plain_dispatch.thresholds,
        row_dispatch: plain_dispatch.dispatch,
        row_dispatch_problems: plain_dispatch.problems,
        packing,
        prefetch,
        online_problems,
        remainder_exits: exits.len(),
        mode_register: mode_register.to_string(),
        mode_dispatch_problems,
        epilogue_sections,
        columns_register: columns_register.to_string(),
        epilogues,
        stray_products,
        stray_steps,
        unknown_branch: unknown,
    })
}

/// The sum of the immediates `k` of `add <register>, <register>, #k` in the
/// sections `range`.
fn immediate_steps(
    sections: &[Section<'_>],
    range: core::ops::Range<usize>,
    register: &str,
) -> usize {
    sections
        .get(range)
        .unwrap_or_default()
        .iter()
        .flat_map(|(_, lines)| lines.iter())
        .filter_map(|line| match operands_of(line, "add")?[..] {
            [to, from, step] if same_register(to, register) && same_register(from, register) => {
                step.strip_prefix('#')?.parse::<usize>().ok()
            }
            _ => None,
        })
        .sum()
}

/// The clamp `K` of the live columns in section `at`: `csel <cols>, <cols>,
/// <t>, lo` after `mov <t>, #K`.
fn column_clamp(sections: &[Section<'_>], at: usize, columns: &str) -> Option<usize> {
    let lines = &sections.get(at)?.1;
    let (csel, limit) =
        lines
            .iter()
            .enumerate()
            .find_map(|(at, line)| match operands_of(line, "csel")?[..] {
                [to, from, limit, "lo"]
                    if same_register(to, columns) && same_register(from, columns) =>
                {
                    Some((at, limit))
                }
                _ => None,
            })?;
    lines
        .get(..csel)?
        .iter()
        .rev()
        .find_map(|line| match operands_of(line, "mov")?[..] {
            [to, immediate] if same_register(to, limit) => {
                immediate.strip_prefix('#')?.parse().ok()
            }
            _ => None,
        })
}

/// The epilogue sections and the four epilogues in the sections `region`
/// (after the epilogue dispatch, up to the row-panel advance), given the
/// section each mode enters (`entries`), and the sections where the
/// epilogues start.
fn epilogue_spans(
    sections: &[Section<'_>],
    region: core::ops::Range<usize>,
    entries: &[Option<usize>],
    columns: &str,
) -> (Vec<String>, Vec<Option<EpilogueText>>, Vec<usize>) {
    // Every branch in the body, by the section it goes to: (from, to).
    let branches: Vec<(usize, usize)> = sections
        .iter()
        .enumerate()
        .flat_map(|(from, (_, lines))| {
            lines
                .iter()
                .filter_map(move |line| Some((from, branch_section(sections, from, line)?)))
        })
        .collect();
    let mut starts: Vec<usize> = entries
        .iter()
        .flatten()
        .copied()
        .filter(|at| region.contains(at))
        .collect();
    starts.sort_unstable();
    starts.dedup();
    let span_of = |at: usize| {
        region
            .contains(&at)
            .then(|| starts.iter().rev().find(|&&start| start <= at).copied())
            .flatten()
    };
    let mut epilogue_sections = Vec::new();
    for at in region.clone() {
        let entered: Vec<usize> = branches
            .iter()
            .filter(|&&(_, to)| to == at)
            .map(|&(from, _)| from)
            .collect();
        let local = span_of(at).is_some()
            && !entered.is_empty()
            && entered.iter().all(|&from| span_of(from) == span_of(at));
        if starts.contains(&at) || !local {
            epilogue_sections.push(section_name(sections, at));
        }
    }
    let epilogues = entries
        .iter()
        .map(|entry| {
            let start = (*entry)?;
            let mut text = EpilogueText {
                entry: section_name(sections, start),
                chain: Vec::new(),
                column_loops: 0,
                column_bounds: 0,
            };
            if !region.contains(&start) {
                return Some(text);
            }
            let end = starts
                .iter()
                .copied()
                .find(|&next| next > start)
                .unwrap_or(region.end);
            for at in start..end {
                let Some((_, lines)) = sections.get(at) else {
                    continue;
                };
                for line in lines {
                    text.chain
                        .extend(za_tiles(line).into_iter().map(|(tile, _)| tile));
                    let back = branch_target(line).is_some_and(|(_, forward)| !forward);
                    let inside = branch_section(sections, at, line)
                        .is_some_and(|to| (start..end).contains(&to));
                    if back && inside {
                        text.column_loops += 1;
                    }
                    if reads_register(line, columns) {
                        text.column_bounds += 1;
                    }
                }
            }
            Some(text)
        })
        .collect();
    (epilogue_sections, epilogues, starts)
}

/// `Ok` when the structure of `text` equals the model `desc`, else `Err`
/// naming the first difference: the tile height, the panel width (the
/// column step, the clamp of the live columns, the C advance, one B row),
/// the row-panel step and tiles that fill it; `desc.variants` row variants
/// of `1..=variants` live tiles, each with a depth loop of `desc.unroll`
/// steps per pass and a remainder of `desc.unroll - 1` steps (one
/// conditional step for odd `kc`, which matches the model's `kc % unroll`
/// only for `unroll == 2`); no outer product and no bit test of `kc`
/// elsewhere; a variant dispatch that decides on the row count alone, with
/// one threshold per boundary between variants, each a whole number of
/// tiles, and that sends every row count `1..=64` to the variant of
/// `ceil(rows / tile_rows)` tiles; one epilogue dispatch that decides on
/// the mode alone and enters four distinct epilogues, M0 to M3 in text
/// order, with nothing else entered between them; each epilogue's tile
/// chain names every tile once, in order, and each of its column loops is
/// bounded by the live columns.
fn compare_with_model(text: &TextStructure, desc: &KernelPaths) -> Result<(), String> {
    let name = desc.name;
    if text.tile_rows != desc.tile_rows {
        return Err(format!(
            "{name}: tile height {} in the text (one vector), {} in the model",
            text.tile_rows, desc.tile_rows
        ));
    }
    if text.nr != desc.nr {
        return Err(format!(
            "{name}: column panels of {} columns in the text (the step of j0), {} in the model",
            text.nr, desc.nr
        ));
    }
    if text.column_clamp != text.nr {
        return Err(format!(
            "{name}: the live columns are clamped to {}, the column panels step {}",
            text.column_clamp, text.nr
        ));
    }
    if text.c_column_step != text.nr {
        return Err(format!(
            "{name}: C advances {} columns per column panel, the column panels step {}",
            text.c_column_step, text.nr
        ));
    }
    if text.nr != text.tile_rows {
        return Err(format!(
            "{name}: column panels of {} columns, but a B row (one vector) holds {}",
            text.nr, text.tile_rows
        ));
    }
    if text.row_step != MR {
        return Err(format!(
            "{name}: row panels step {} rows in the text, {MR} in the model",
            text.row_step
        ));
    }
    if text.tiles * text.tile_rows != MR {
        return Err(format!(
            "{name}: {} tiles of {} rows do not fill a {MR}-row panel",
            text.tiles, text.tile_rows
        ));
    }
    if text.variants.len() != desc.variants {
        return Err(format!(
            "{name}: {} row variants in the text, {} in the model",
            text.variants.len(),
            desc.variants
        ));
    }
    let mut counts: Vec<usize> = text.variants.iter().map(|variant| variant.tiles).collect();
    counts.sort_unstable();
    if counts != (1..=desc.variants).collect::<Vec<_>>() {
        return Err(format!(
            "{name}: row variants of {counts:?} live tiles in the text, 1..={} in the model",
            desc.variants
        ));
    }
    if desc.unroll != 2 {
        return Err(format!(
            "{name}: the model unrolls by {}, but a remainder of one conditional step \
             covers only an unrolling by 2",
            desc.unroll
        ));
    }
    for variant in &text.variants {
        if variant.steps_per_pass != desc.unroll {
            return Err(format!(
                "{name}: the variant of {} tiles (label {}) makes {} depth steps per pass, \
                 the model {}",
                variant.tiles, variant.label, variant.steps_per_pass, desc.unroll
            ));
        }
        if variant.remainder_steps != desc.unroll - 1 {
            return Err(format!(
                "{name}: the remainder of the variant of {} tiles (label {}) makes {} \
                 depth steps, the model {}",
                variant.tiles,
                variant.label,
                variant.remainder_steps,
                desc.unroll - 1
            ));
        }
    }
    if let Some((label, count)) = text.stray_products.first() {
        return Err(format!(
            "{name}: section {label} has {count} outer products outside the variants' depth \
             loops and remainders"
        ));
    }
    if let Some((label, line)) = text.stray_steps.first() {
        return Err(format!(
            "{name}: `{line}` (section {label}) is a depth step conditional on kc outside a \
             variant's remainder"
        ));
    }
    compare_dispatch(
        (desc, "variant", "variant"),
        &text.rows_register,
        (
            &text.row_thresholds,
            &text.row_dispatch,
            &text.row_dispatch_problems,
        ),
    )?;
    compare_epilogues(text, desc)?;
    compare_online(text, desc)?;
    if let Some((section, line)) = &text.unknown_branch {
        return Err(format!(
            "{name}: `{line}` (section {section}) is a branch the model does not know: a path \
             the path table and the self-test do not take"
        ));
    }
    Ok(())
}

/// A variant dispatch of [`compare_with_model`] (`which`: "variant" for the
/// plain one, "packing" for the packing one; `kind`: what it reaches):
/// decided on the row count `rows` alone, one threshold per boundary between
/// variants, each a whole number of tiles, and every row count `1..=64` to
/// the variant of its kind with `ceil(rows / tile_rows)` tiles.
fn compare_dispatch(
    (desc, which, kind): (&KernelPaths, &str, &str),
    rows: &str,
    (thresholds, dispatch, problems): (&[usize], &[Option<usize>], &[String]),
) -> Result<(), String> {
    let name = desc.name;
    if let Some(line) = problems.first() {
        return Err(format!(
            "{name}: the {which} dispatch cannot decide `{line}` on the row count ({rows}) alone"
        ));
    }
    if thresholds.len() + 1 != desc.variants {
        return Err(format!(
            "{name}: the {which} dispatch compares the row count with {thresholds:?}: {} \
             thresholds for {} variants, want one per boundary between variants",
            thresholds.len(),
            desc.variants
        ));
    }
    if let Some(threshold) = thresholds
        .iter()
        .find(|threshold| !threshold.is_multiple_of(desc.tile_rows))
    {
        return Err(format!(
            "{name}: the {which} dispatch compares the row count with {thresholds:?}: \
             {threshold} is not a multiple of the tile height {}",
            desc.tile_rows
        ));
    }
    let want = |rows: usize| rows.div_ceil(desc.tile_rows);
    let got: Vec<(usize, Option<usize>)> = (1..=MR).zip(dispatch.iter().copied()).collect();
    if got.len() != MR {
        return Err(format!(
            "{name}: the {which} dispatch ran for {} row counts",
            got.len()
        ));
    }
    if let Some(&(first, reached)) = got.iter().find(|&&(rows, tiles)| tiles != Some(want(rows))) {
        let last = got
            .iter()
            .skip(first - 1)
            .take_while(|&&(rows, tiles)| tiles == reached && want(rows) == want(first))
            .last()
            .map_or(first, |&(rows, _)| rows);
        let reached = match reached {
            Some(tiles) => format!("the {kind} of {tiles} tiles"),
            None => format!("no {kind}"),
        };
        return Err(format!(
            "{name}: the {which} dispatch sends rows {first}..={last} of a row panel to \
             {reached}, the model to the {kind} of {} tiles",
            want(first)
        ));
    }
    Ok(())
}

/// The online paths of [`compare_with_model`]: problems of a text without
/// the gate; with `desc.packing` the packing pass (the gate, `variants`
/// packing twins of `1..=variants` live tiles with the depth loops of the
/// plain variants, their stores and buffer advances, the packing dispatch,
/// the switch and the registers it writes), else none; with
/// `desc.prefetch` the prefetch block, hint for hint, else none.
fn compare_online(text: &TextStructure, desc: &KernelPaths) -> Result<(), String> {
    let name = desc.name;
    if let Some(problem) = text.online_problems.first() {
        return Err(format!("{name}: {problem}"));
    }
    match (&text.packing, desc.packing) {
        (None, false) => {}
        (None, true) => {
            return Err(format!(
                "{name}: no packing pass in the text (no gate `cbz <a_pack>, …` in the depth \
                 setup), the model packs A online"
            ));
        }
        (Some(_), false) => {
            return Err(format!(
                "{name}: the text packs A online, the model does not"
            ));
        }
        (Some(packing), true) => {
            if packing.twins.len() != desc.variants {
                return Err(format!(
                    "{name}: {} packing twins in the text, {} in the model",
                    packing.twins.len(),
                    desc.variants
                ));
            }
            let mut counts: Vec<usize> = packing.twins.iter().map(|twin| twin.tiles).collect();
            counts.sort_unstable();
            if counts != (1..=desc.variants).collect::<Vec<_>>() {
                return Err(format!(
                    "{name}: packing twins of {counts:?} live tiles in the text, 1..={} in the \
                     model",
                    desc.variants
                ));
            }
            for twin in &packing.twins {
                if twin.steps_per_pass != desc.unroll || twin.remainder_steps + 1 != desc.unroll {
                    return Err(format!(
                        "{name}: the packing twin of {} tiles (label {}) makes {} depth steps \
                         per pass and {} in its remainder, the model {} and {}",
                        twin.tiles,
                        twin.label,
                        twin.steps_per_pass,
                        twin.remainder_steps,
                        desc.unroll,
                        desc.unroll - 1
                    ));
                }
            }
            if let Some(problem) = packing.problems.first() {
                return Err(format!("{name}: {problem}"));
            }
            let dispatch = &packing.dispatch;
            compare_dispatch(
                (desc, "packing", "packing twin"),
                &text.rows_register,
                (&dispatch.thresholds, &dispatch.dispatch, &dispatch.problems),
            )?;
        }
    }
    match (&text.prefetch, desc.prefetch) {
        (None, true) => Err(format!(
            "{name}: no prefetch block in the text (no skip `tbz <prefetch>, #0, …` in the \
             row-panel head), the model prefetches C"
        )),
        (Some(_), false) => Err(format!("{name}: the text prefetches C, the model does not")),
        (Some(prefetch), true) => match prefetch.problems.first() {
            Some(problem) => Err(format!("{name}: {problem}")),
            None => Ok(()),
        },
        (None, false) => Ok(()),
    }
}

/// The epilogue dispatch and the epilogues of [`compare_with_model`].
fn compare_epilogues(text: &TextStructure, desc: &KernelPaths) -> Result<(), String> {
    let name = desc.name;
    if text.remainder_exits != 1 {
        return Err(format!(
            "{name}: the variants' remainders leave to {} different sections, want one \
             epilogue dispatch",
            text.remainder_exits
        ));
    }
    if let Some(line) = text.mode_dispatch_problems.first() {
        return Err(format!(
            "{name}: the epilogue dispatch tests `{line}`, not only the mode ({})",
            text.mode_register
        ));
    }
    let mut entries = Vec::new();
    for (mode, epilogue) in text.epilogues.iter().enumerate() {
        let Some(epilogue) = epilogue else {
            return Err(format!(
                "{name}: the epilogue dispatch sends M{mode} to no label"
            ));
        };
        if let Some(other) = entries.iter().position(|entry| *entry == epilogue.entry) {
            return Err(format!(
                "{name}: M{other} and M{mode} enter the same epilogue (label {})",
                epilogue.entry
            ));
        }
        entries.push(epilogue.entry.clone());
    }
    if entries.len() != 4 || text.epilogue_sections != entries {
        return Err(format!(
            "{name}: the sections after the epilogue dispatch are entered at {:?}, the four \
             modes at {entries:?} (M0 to M3): want exactly the four, in order",
            text.epilogue_sections
        ));
    }
    let full: Vec<usize> = (0..desc.variants).collect();
    for (mode, epilogue) in text.epilogues.iter().flatten().enumerate() {
        if epilogue.chain != full {
            return Err(format!(
                "{name}: the M{mode} epilogue's tile chain is {:?}, the model {full:?}",
                epilogue.chain
            ));
        }
        if epilogue.column_loops == 0 || epilogue.column_bounds != epilogue.column_loops {
            return Err(format!(
                "{name}: the M{mode} epilogue has {} column loop(s) and reads the live columns \
                 ({}) {} times, want one bound per loop",
                epilogue.column_loops, text.columns_register, epilogue.column_bounds
            ));
        }
    }
    Ok(())
}

/// [`derive_structure`] of `template` compared with `desc` (the edited
/// templates of `tests_selftest.rs` use it too).
pub(super) fn model_matches_text(desc: &KernelPaths, template: &str) -> Result<(), String> {
    let text = derive_structure(template).map_err(|error| format!("{}: {error}", desc.name))?;
    compare_with_model(&text, desc)
}

#[test]
fn test_sme2_selftest_paths_model_matches_text() {
    for (desc, template) in [
        (&F32_PATHS, kernel_template::<f32>()),
        (&F64_PATHS, kernel_template::<f64>()),
    ] {
        let text = match derive_structure(template) {
            Ok(text) => text,
            Err(error) => panic!("{}: {error}", desc.name),
        };
        let shape: Vec<(usize, usize, usize)> = text
            .variants
            .iter()
            .map(|variant| {
                (
                    variant.tiles,
                    variant.steps_per_pass,
                    variant.remainder_steps,
                )
            })
            .collect();
        let chains: Vec<(&str, &[usize], usize, usize)> = text
            .epilogues
            .iter()
            .flatten()
            .map(|epilogue| {
                (
                    epilogue.entry.as_str(),
                    epilogue.chain.as_slice(),
                    epilogue.column_loops,
                    epilogue.column_bounds,
                )
            })
            .collect();
        println!(
            "{}: {} tiles of {} rows, nr {} (clamp {}, C step {}), row step {}, variants \
             (tiles, steps per pass, remainder) {shape:?}, thresholds {:?}, epilogues \
             (entry, chain, loops, bounds) {chains:?}",
            desc.name,
            text.tiles,
            text.tile_rows,
            text.nr,
            text.column_clamp,
            text.c_column_step,
            text.row_step,
            text.row_thresholds
        );
        assert_eq!(text.element_bytes * desc.nr, VECTOR_BYTES, "{}", desc.name);
        assert_eq!(text.row_dispatch.len(), MR, "{}", desc.name);
        assert_eq!(
            text.epilogue_sections,
            ["51", "52", "53", "54"],
            "{}",
            desc.name
        );
        if let Err(error) = compare_with_model(&text, desc) {
            panic!("{error}");
        }
    }
    // A model with another row variant or unrolling differs from the text
    // (the edited texts are in `tests_selftest.rs`).
    for (desc, template) in [
        (&F32_PATHS, kernel_template::<f32>()),
        (&F64_PATHS, kernel_template::<f64>()),
    ] {
        let more = KernelPaths {
            variants: desc.variants + 1,
            ..*desc
        };
        assert!(
            model_matches_text(&more, template).is_err(),
            "{}",
            desc.name
        );
        let by_4 = KernelPaths { unroll: 4, ..*desc };
        assert!(
            model_matches_text(&by_4, template).is_err(),
            "{}",
            desc.name
        );
    }
    // The model's literals agree with the element types as well.
    assert_eq!(
        (F32_PATHS.tile_rows, F32_PATHS.nr),
        (f32::TILE_ROWS, f32::NR)
    );
    assert_eq!(
        (F64_PATHS.tile_rows, F64_PATHS.nr),
        (f64::TILE_ROWS, f64::NR)
    );
}
