//! Tests of the `f64` macro-kernel `sme2_gemm_f64` (`kernel_f64.rs`).
//!
//! Every test that executes the kernel starts with `hardware_or_skip`: on a
//! host without SME2 (the Apple M3) it prints a skip line, under QEMU with the
//! runner's `-r` it must run. Results are compared with `args::reference`
//! (the contract's executable definition) through `args::same_result`
//! (bitwise, NaN payloads aside).
//!
//! - **Registration and structure** (run everywhere): the kernel is the one
//!   registered entry of its module, passes the template lint, and each
//!   depth-loop variant zeroes and accumulates exactly its live tiles. The
//!   registered template text also carries the memory rules: the A-row
//!   predicates `p2` / `pn9` / `pn10` are set only by the row-panel setup
//!   and bounded by `m` (whatever the form of a write: a braced destination
//!   list, a register range or the `p9` / `p10` names of the counters all
//!   count), no predicate appears elsewhere outside a governing position,
//!   every A load uses the predicate of its rows, M0 and M2 load nothing,
//!   M1 and M3 load only the C column under `p1`, and every store is under
//!   `p1`. Mutated templates that break the predicate rules are rejected by
//!   the same checker.
//! - **Shapes**: the grid of m around 8 / 64, n around 8, kc around the
//!   unroll and 64, every (alpha, beta) pair of `tests_contract.rs` (all four
//!   modes; M3 also with betas that are not powers of two, so that the single
//!   rounding `fma(C, beta, t)` is told apart from a product rounded before
//!   the sum), and every (m, n) in 1..=20 × 1..=20 with kc ∈ {1, 5}; n
//!   around 64 (eight and nine column panels) on a reduced grid, every mode
//!   and layout.
//! - **Layouts**: plain packing, A read in place (`a_panel = 64`,
//!   `a_depth = lda > m`) and stretched strides (`a_depth > 64`, gaps between
//!   A and B panels). A and B are allocated exactly as large as the contract
//!   lets the kernel read, and every element it must not read (rows ≥ m,
//!   gaps) holds a NaN. The NaN gaps catch a read that reaches a live lane
//!   (a wrong stride or offset). A read that reaches only dead lanes (A rows
//!   ≥ m under a too-wide predicate, C outside the block or C in M0 / M2)
//!   never changes a stored element, so the result tests cannot see it; the
//!   template-structure assertions above cover those rules.
//! - **Borders**: C has `ldc > m` and a guard column on each side, all filled
//!   with a sentinel NaN that must stay bitwise untouched; in M0 and M2 the
//!   block itself is NaN and must not leak into the result.
//! - **Special values** in A, B, C, alpha and beta.
//! - **FPSR**: flags set before a call survive it; the call adds exactly the
//!   flags of its live SVE epilogue lanes (none in M0).
//! - **Thread state**: ready before, SVCR = TPIDR2_EL0 = 0 after, kernel
//!   entries counted, and 120 calls interleaved with NEON and scalar `f64`
//!   work give the same bits every time.

use oxiblas_matrix::MatMut;

use super::args::{Epilogue, MR, MacroArgs, reference, same_result};
use super::detect;
use super::kernel_f64::{SME_KERNELS, sme2_gemm_f64};
use super::lint::{lint_kernel, lint_streaming_template, report};
use super::test_support::{
    FPSR_DZC, FPSR_IDC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_QC, FPSR_UFC, Need, fpsr_read,
    fpsr_write, hardware_or_skip, kernel_entries, sme_state,
};
use super::tests_kernel_f32::{PredicateUse, predicate_uses, template_with};
use crate::level3::gemm::macro_panel_multiply_with;
use crate::level3::gemm_kernel::GemmKernel;

/// Columns of one packed B panel.
const NR: usize = 8;
/// Fills C outside the block, including the guard columns. A signalling
/// NaN: any arithmetic on it returns a quiet NaN (other bits), so even a
/// write of `acc + C` or `t + C * beta` back to a guard element shows.
const C_SENTINEL: u64 = 0x7ff0_0000_dead_beef;
/// Fills every A and B position the kernel must not read (a signalling NaN,
/// for the same reason).
const GAP: u64 = 0x7ff0_0000_0bad_0a0a;
/// Fills the C block when the mode does not read C.
const UNREAD_C: u64 = 0x7ff8_0000_0000_c0de;

/// The (alpha, beta) pairs of `tests_contract.rs`: M0 twice, M1, M2 twice and
/// M3 five times. The last two M3 pairs (alpha != 1 and alpha == 1) have a
/// beta that is not a power of two: `C * beta` is inexact, so a kernel that
/// rounds it before adding `t` (FMUL then FADD) differs from the single
/// rounding `fma(C, beta, t)` of rule R3.
const PAIRS: [(f64, f64); 10] = [
    (1.0, 0.0),
    (1.0, -0.0),
    (1.0, 1.0),
    (-1.5, 0.0),
    (0.3, 0.0),
    (-1.5, 0.25),
    (1.0, -0.5),
    (2.0, 1.0),
    (0.3, 0.7),
    (1.0, 0.7),
];

/// Row counts of the shape grid: around one tile (8), the variants and 64.
const GRID_M: [usize; 20] = [
    1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 70, 127, 128, 129,
];
/// Column counts of the shape grid: around one panel (8) and several panels.
const GRID_N: [usize; 7] = [1, 7, 8, 9, 15, 17, 23];
/// Depths of the shape grid: the unroll by 2 with and without remainder.
const GRID_KC: [usize; 6] = [1, 2, 3, 7, 64, 65];

/// How A and B are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// The plain packed layouts of `MacroArgs::packed`.
    Packed,
    /// A read in place: column-major with `lda = m + 5`, `a_panel = 64`.
    InPlace,
    /// Packed panels with stretched strides and NaN gaps between them.
    Strided,
}

impl Layout {
    /// `(a_depth, a_panel, b_panel)` for a block of `m` rows and depth `kc`.
    fn strides(self, m: usize, kc: usize) -> (usize, usize, usize) {
        match self {
            Self::Packed => (MR, MR * kc, kc * NR),
            Self::InPlace => (m + 5, MR, kc * NR),
            Self::Strided => (MR + 3, (MR + 3) * kc + 5, kc * NR + 7),
        }
    }
}

/// Deterministic test values.
struct Values {
    /// Xorshift state.
    state: u64,
}

impl Values {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
        }
    }

    fn bits(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Mixed signs and magnitudes, a few exact zeros.
    fn mixed(&mut self) -> f64 {
        let bits = self.bits();
        let unit = (bits >> 11) as f64 / (1_u64 << 53) as f64;
        let scale = [1.0, 1.0e-3, 37.0, 0.0][(bits & 3) as usize];
        (unit * 2.0 - 1.0) * scale
    }

    /// A small positive integer in `1..=limit`.
    fn small(&mut self, limit: u64) -> f64 {
        (self.bits() % limit + 1) as f64
    }

    /// One of `choices`.
    fn pick(&mut self, choices: &[f64]) -> f64 {
        choices[(self.bits() % choices.len() as u64) as usize]
    }
}

/// The A and B blocks of one call, in one [`Layout`].
struct Operands {
    m: usize,
    n: usize,
    kc: usize,
    a_depth: usize,
    a_panel: usize,
    b_panel: usize,
    /// Exactly up to the last element the kernel may read.
    a: Vec<f64>,
    /// Exactly up to the end of the last panel.
    b: Vec<f64>,
}

impl Operands {
    /// Fills A(i, p) and B(p, j) with `a(i, p)` / `b(p, j)`; every other
    /// element is [`GAP`], except the zero padding of the last B panel.
    fn new(
        (m, n, kc): (usize, usize, usize),
        layout: Layout,
        mut a: impl FnMut(usize, usize) -> f64,
        mut b: impl FnMut(usize, usize) -> f64,
    ) -> Self {
        let (a_depth, a_panel, b_panel) = layout.strides(m, kc);
        let a_at = |i: usize, p: usize| (i / MR) * a_panel + p * a_depth + i % MR;
        let a_len = (0..m)
            .flat_map(|i| (0..kc).map(move |p| (i, p)))
            .map(|(i, p)| a_at(i, p))
            .max()
            .map_or(0, |last| last + 1);
        let mut a_data = vec![f64::from_bits(GAP); a_len];
        for p in 0..kc {
            for i in 0..m {
                a_data[a_at(i, p)] = a(i, p);
            }
        }
        let panels = n.div_ceil(NR);
        let mut b_data = vec![f64::from_bits(GAP); (panels - 1) * b_panel + kc * NR];
        for p in 0..kc {
            for j in 0..panels * NR {
                let value = if j < n { b(p, j) } else { 0.0 };
                b_data[(j / NR) * b_panel + p * NR + j % NR] = value;
            }
        }
        Self {
            m,
            n,
            kc,
            a_depth,
            a_panel,
            b_panel,
            a: a_data,
            b: b_data,
        }
    }

    /// Random mixed values in A and B.
    fn mixed(shape: (usize, usize, usize), layout: Layout, values: &mut Values) -> Self {
        let mut a_values = Values::new(values.bits());
        let mut b_values = Values::new(values.bits());
        Self::new(
            shape,
            layout,
            |_, _| a_values.mixed(),
            |_, _| b_values.mixed(),
        )
    }

    /// The argument block for C and alpha / beta.
    fn args(&self, c: &mut CBlock, alpha: f64, beta: f64) -> MacroArgs<f64> {
        let ldc = c.ldc;
        let packed = MacroArgs::packed(
            self.a.as_ptr(),
            self.b.as_ptr(),
            c.data[ldc..].as_mut_ptr(),
            ldc,
            self.m,
            self.n,
            self.kc,
            alpha,
            beta,
        );
        MacroArgs {
            a_depth: self.a_depth,
            a_panel: self.a_panel,
            b_panel: self.b_panel,
            ..packed
        }
    }
}

/// C with `ldc = m + 3` and one guard column before and after the block.
#[derive(Clone)]
struct CBlock {
    m: usize,
    n: usize,
    ldc: usize,
    /// `ldc * (n + 2)` elements; the block starts at column 1.
    data: Vec<f64>,
}

impl CBlock {
    /// [`C_SENTINEL`] everywhere, `c(i, j)` inside the block.
    fn new(m: usize, n: usize, mut c: impl FnMut(usize, usize) -> f64) -> Self {
        let ldc = m + 3;
        let mut data = vec![f64::from_bits(C_SENTINEL); ldc * (n + 2)];
        for j in 0..n {
            for i in 0..m {
                data[i + (j + 1) * ldc] = c(i, j);
            }
        }
        Self { m, n, ldc, data }
    }

    /// Random mixed values when the mode reads C, [`UNREAD_C`] otherwise.
    fn for_mode(m: usize, n: usize, mode: Epilogue, values: &mut Values) -> Self {
        if mode.reads_c() {
            Self::new(m, n, |_, _| values.mixed())
        } else {
            Self::new(m, n, |_, _| f64::from_bits(UNREAD_C))
        }
    }

    /// Asserts the block equals `want` (`same_result`) and everything else is
    /// still [`C_SENTINEL`].
    #[track_caller]
    fn check(&self, want: &Self, label: &str) {
        for (index, (got, want)) in self.data.iter().zip(&want.data).enumerate() {
            let (i, column) = (index % self.ldc, index / self.ldc);
            if i < self.m && (1..=self.n).contains(&column) {
                assert!(
                    same_result(*got, *want),
                    "{label}: C({i}, {}) = {got:e} ({:#x}), want {want:e} ({:#x})",
                    column - 1,
                    got.to_bits(),
                    want.to_bits()
                );
            } else {
                assert_eq!(
                    got.to_bits(),
                    C_SENTINEL,
                    "{label}: element ({i}, {}) outside the block was written",
                    column as isize - 1
                );
            }
        }
    }
}

/// Runs the kernel on `c` and returns the contract's value of the same call.
fn run(ops: &Operands, c: &mut CBlock, alpha: f64, beta: f64) -> CBlock {
    let mut want = c.clone();
    let want_args = ops.args(&mut want, alpha, beta);
    // SAFETY: `ops` holds every A and B element of the block's layout and
    // `want` every C element of the block (`ldc ≥ m`).
    unsafe { reference(&want_args) };
    let args = ops.args(c, alpha, beta);
    // SAFETY: every test that calls `run` passed `hardware_or_skip(.., F64)`,
    // so SME2 with FEAT_SME_F64F64 is present, and the thread is not in
    // streaming mode; m, n, kc ≥ 1, `mode` comes from `Epilogue::select`, and
    // the buffers cover the block as above (C is its own allocation).
    unsafe { sme2_gemm_f64(&args) };
    want
}

/// One call with mixed values, checked against the contract.
#[track_caller]
fn check_case(
    (m, n, kc): (usize, usize, usize),
    layout: Layout,
    (alpha, beta): (f64, f64),
    values: &mut Values,
) {
    let mode = Epilogue::select(alpha, beta);
    let ops = Operands::mixed((m, n, kc), layout, values);
    let mut c = CBlock::for_mode(m, n, mode, values);
    let want = run(&ops, &mut c, alpha, beta);
    c.check(
        &want,
        &format!("{m}x{n}x{kc} {layout:?} {mode:?} alpha {alpha} beta {beta}"),
    );
}

/// The shape grid for the given pairs and layouts.
fn grid(name: &str, pairs: &[(f64, f64)], layouts: &[Layout]) {
    let mut values = Values::new(0x5eed_0f64);
    let mut calls = 0_usize;
    for &m in &GRID_M {
        for &n in &GRID_N {
            for &kc in &GRID_KC {
                for &layout in layouts {
                    for &pair in pairs {
                        check_case((m, n, kc), layout, pair, &mut values);
                        calls += 1;
                    }
                }
            }
        }
    }
    println!("{name}: {calls} calls");
}

// ---------------------------------------------------------------------------
// Registration and structure (no SME2 needed)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_kernel_f64_registered_and_linted() {
    let names: Vec<&str> = SME_KERNELS.iter().map(|entry| entry.name).collect();
    assert_eq!(names, ["sme2_gemm_f64"]);
    let entry = &SME_KERNELS[0];
    assert!(entry.f64f64, "the f64 kernel enables sme-f64f64");
    assert!(entry.file.ends_with("kernel_f64.rs"), "{}", entry.file);
    if let Err(errors) = lint_kernel(entry) {
        panic!("{}", report(&errors));
    }
    // The Wave-4 fields are never read.
    for unused in ["a_pack", "prefetch"] {
        assert!(
            entry.consts.iter().all(|name| !name.contains(unused)),
            "{unused} is declared: {:?}",
            entry.consts
        );
    }
    // The driver and the self-test call it through this type.
    let _: unsafe fn(&MacroArgs<f64>) = sme2_gemm_f64;
}

/// The ZA tile numbers (`za{t}.d` / `za{t}h.d`) a template line names.
fn tiles_in(line: &str) -> Vec<usize> {
    line.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|token| token.strip_prefix("za"))
        .filter_map(|rest| rest.trim_end_matches(['h', 'v']).parse().ok())
        .collect()
}

#[test]
fn test_sme2_kernel_f64_variants_touch_only_live_tiles() {
    let template = SME_KERNELS[0].template;
    // Variant `4L` (L live tiles) runs from its label to the next variant or
    // to the epilogue dispatch (`50:`).
    let mut variant: Option<usize> = None;
    let mut seen: Vec<(bool, Vec<usize>)> = vec![(false, Vec::new()); 9];
    let mut epilogue_tiles = Vec::new();
    let mut in_epilogue = false;
    for line in template.lines().map(str::trim) {
        if let Some(label) = line.strip_suffix(':') {
            match label.parse::<usize>() {
                Ok(live @ 41..=48) => variant = Some(live - 40),
                Ok(50) => {
                    variant = None;
                    in_epilogue = true;
                }
                Ok(59) => in_epilogue = false,
                _ => {}
            }
            continue;
        }
        let Some(live) = variant else {
            if in_epilogue {
                epilogue_tiles.extend(tiles_in(line));
            }
            continue;
        };
        let tiles = tiles_in(line);
        if line.starts_with("zero") {
            assert!(!seen[live].0, "variant {live} zeroes twice");
            assert_eq!(
                tiles,
                (0..live).collect::<Vec<_>>(),
                "zero of variant {live}"
            );
            seen[live].0 = true;
        }
        if line.starts_with("fmopa") {
            assert!(
                tiles.iter().all(|&t| t < live),
                "variant {live} accumulates a dead tile: {line}"
            );
            seen[live].1.extend(tiles);
        }
    }
    for (live, (zeroed, accumulated)) in seen.iter().enumerate().skip(1) {
        assert!(zeroed, "variant {live} has no zero");
        // Unrolled pair (2 FMOPAs per tile) plus the remainder step (1).
        for tile in 0..live {
            let count = accumulated.iter().filter(|&&t| t == tile).count();
            assert_eq!(count, 3, "variant {live}, tile {tile}: {count} FMOPAs");
        }
    }
    // Each of the four epilogues walks the eight tiles in order.
    let expected: Vec<usize> = (0..4).flat_map(|_| 0..8).collect();
    assert_eq!(epilogue_tiles, expected);

    // The memory rules of the contract (section 4) that no result test can
    // see: a read of an A row >= m, of C outside the block, or of C in M0 /
    // M2 only feeds lanes that are never stored. They are protected here, on
    // the registered template text.
    if let Err(error) = check_row_predicates(template) {
        panic!("sme2_gemm_f64: {error}");
    }
    let sections = sections(template);
    check_a_loads(&sections);
    check_epilogue_memory(&sections);
}

/// Labels that start a section of the loop nest: the column panel (20), the
/// row panel (30), the depth-loop variants (41..48), the epilogue dispatch
/// (50), the four epilogues (51..54), the panel advance (59) and the frame's
/// exit (90). The inner labels (6L, 7L, 56) stay inside their section.
const SECTION_LABELS: [usize; 17] = [
    20, 30, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 59, 90,
];

/// The template's instruction lines, grouped by the section label that
/// precedes them (`0` for the lines before the first one: the frame entry
/// and the prologue).
fn sections(template: &str) -> Vec<(usize, Vec<&str>)> {
    let mut sections: Vec<(usize, Vec<&str>)> = vec![(0, Vec::new())];
    for line in template.lines().map(str::trim) {
        if let Some(label) = line.strip_suffix(':') {
            if let Ok(label) = label.parse::<usize>()
                && SECTION_LABELS.contains(&label)
            {
                assert!(
                    sections.iter().all(|(seen, _)| *seen != label),
                    "section label {label} defined twice"
                );
                sections.push((label, Vec::new()));
            }
            continue;
        }
        if let Some((_, lines)) = sections.last_mut() {
            lines.push(line);
        }
    }
    let labels: Vec<usize> = sections.iter().map(|(label, _)| *label).collect();
    let mut expected = vec![0];
    expected.extend(SECTION_LABELS);
    assert_eq!(labels, expected, "section labels of the template");
    sections
}

/// The lines of the section that starts at `label`.
fn section<'a>(sections: &'a [(usize, Vec<&'a str>)], label: usize) -> &'a [&'a str] {
    match sections.iter().find(|(seen, _)| *seen == label) {
        Some((_, lines)) => lines,
        None => panic!("no section {label}"),
    }
}

/// The mnemonic of an instruction line.
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

/// Whether the line is a memory read (`ld1d`, `ldr`, `ldnt1d`, ...).
fn is_load(line: &str) -> bool {
    mnemonic(line).starts_with("ld")
}

/// Whether the line is a memory write (`st1d`, `str`, `stp`, ...).
fn is_store(line: &str) -> bool {
    mnemonic(line).starts_with("st")
}

/// A contiguous vector load or store `op {{zS.d[-zE.d]}}, pred, [addr]`,
/// split into its first register, its width, the predicate operand and the
/// address (without the brackets).
fn vector_access(line: &str) -> Option<(usize, usize, &str, &str)> {
    let (_, rest) = line.split_once("{{")?;
    let (group, rest) = rest.split_once("}}, ")?;
    let (predicate, address) = rest.split_once(", [")?;
    let address = address.strip_suffix(']')?;
    let register =
        |text: &str| -> Option<usize> { text.strip_prefix('z')?.strip_suffix(".d")?.parse().ok() };
    let (first, last) = match group.split_once('-') {
        Some((first, last)) => (register(first)?, register(last)?),
        None => (register(group)?, register(group)?),
    };
    (last >= first).then_some((first, last - first + 1, predicate, address))
}

/// The A-row predicate setup of the row panel (section 30), in this order.
const ROW_PANEL_SETUP: [&str; 4] = [
    "whilelt p2.d, x10, x20",
    "whilelt pn9.d, x10, x20, vlx4",
    "add x16, x10, #32",
    "whilelt pn10.d, x16, x20, vlx4",
];

/// The predicates of the A rows: `p2` (rows `i0..+7`), `pn9` (`i0..+31`)
/// and `pn10` (`i0+32..+63`), all bounded by `m`, set once per row panel by
/// exactly [`ROW_PANEL_SETUP`] and nowhere else; the epilogue's `p1` is only
/// ever `whilelt p1.d, x13, x20` (rows of the tile being stored below `m`);
/// `p0` is only `ptrue p0.d`. A write in any form counts: a braced
/// destination list (`whilelt {{p2.d, p3.d}}, …`, `pext {{…}}, pn…[i]`), a
/// register range, or the other name of a register (`p9` is `pn9`). No
/// predicate appears outside a governing position (`pK/z`, `pK/m`, a
/// store's `pK`) except in those defining lines. Checked on a template text:
/// the registered one, or a mutated copy of it.
fn check_row_predicates(template: &str) -> Result<(), String> {
    let sections = sections(template);
    let row_panel = section(&sections, 30);
    let start = row_panel
        .iter()
        .position(|line| *line == ROW_PANEL_SETUP[0])
        .ok_or_else(|| {
            format!(
                "the row panel does not set p2 with `{}`",
                ROW_PANEL_SETUP[0]
            )
        })?;
    let setup = row_panel.get(start..start + ROW_PANEL_SETUP.len());
    if setup != Some(&ROW_PANEL_SETUP[..]) {
        return Err(format!("the row-panel predicate setup: {setup:?}"));
    }
    let mut a_predicates = Vec::new();
    let mut p0_writes = 0_usize;
    for (label, lines) in &sections {
        for line in lines {
            for PredicateUse {
                register,
                written,
                governing,
            } in predicate_uses(line)
            {
                if !matches!(register, 0 | 1 | 2 | 9 | 10) {
                    return Err(format!(
                        "section {label}: p{register} is not a predicate of this kernel: {line:?}"
                    ));
                }
                if !written {
                    if governing {
                        continue;
                    }
                    return Err(format!(
                        "section {label}: p{register} outside a governing position (pK/z, \
                         pK/m, a store's pK) in a line that does not define it: {line:?}"
                    ));
                }
                let defining = match register {
                    0 => *line == "ptrue p0.d",
                    1 => (51..=54).contains(label) && *line == "whilelt p1.d, x13, x20",
                    _ => *label == 30 && ROW_PANEL_SETUP.contains(line),
                };
                if !defining {
                    return Err(format!(
                        "section {label}: p{register} written by {line:?}, which is not \
                         its defining line"
                    ));
                }
                match register {
                    0 => p0_writes += 1,
                    1 => {}
                    _ => a_predicates.push((*label, *line)),
                }
            }
        }
    }
    let expected: Vec<(usize, &str)> = [ROW_PANEL_SETUP[0], ROW_PANEL_SETUP[1], ROW_PANEL_SETUP[3]]
        .into_iter()
        .map(|line| (30, line))
        .collect();
    if a_predicates != expected {
        return Err(format!("writers of p2, pn9 and pn10: {a_predicates:?}"));
    }
    if p0_writes != 1 {
        return Err(format!("`ptrue p0.d` occurs {p0_writes} times"));
    }
    Ok(())
}

/// A mutated template the predicate checker must reject.
struct Mutant {
    /// What the mutation does.
    label: &'static str,
    /// Lines inserted after an anchor line of the registered template.
    edits: &'static [(&'static str, &'static [&'static str])],
    /// The statement the checker must name (as the registry writes it).
    culprit: &'static str,
}

#[test]
fn test_sme2_kernel_f64_predicate_checker_rejects_mutants() {
    // Wrong templates that the lint accepts and that a first-operand-only
    // parse does not see: each widens or moves an A-row predicate, so A
    // loads may read rows >= m (which no result test can observe).
    const MUTANTS: [Mutant; 6] = [
        Mutant {
            label: "p2 from row 0 through the pair form (second row panel reads rows >= m)",
            edits: &[("zero {{za0.d}}", &["whilelt {{p2.d, p3.d}}, xzr, x20"])],
            culprit: "whilelt {{p2.d, p3.d}}, xzr, x20",
        },
        Mutant {
            label: "the pair form as a register range",
            edits: &[("zero {{za0.d}}", &["whilelt {{p2.d-p3.d}}, xzr, x20"])],
            culprit: "whilelt {{p2.d-p3.d}}, xzr, x20",
        },
        Mutant {
            label: "pext of the upper counter into p2 (rows i0+32.. for rows i0..)",
            edits: &[(ROW_PANEL_SETUP[3], &["pext {{p2.d, p3.d}}, pn10[0]"])],
            culprit: "pext {{p2.d, p3.d}}, pn10[0]",
        },
        Mutant {
            label: "pn9 overwritten through its other name p9",
            edits: &[(ROW_PANEL_SETUP[3], &["mov p9.b, p0.b"])],
            culprit: "mov p9.b, p0.b",
        },
        Mutant {
            label: "pn10 copied through its other name p10 (sel)",
            edits: &[(ROW_PANEL_SETUP[3], &["sel p10.b, p0, p0.b, p10.b"])],
            culprit: "sel p10.b, p0, p0.b, p10.b",
        },
        Mutant {
            label: "a second pn9 with other operands, in a variant",
            edits: &[("zero {{za0.d, za1.d}}", &["whilelt pn9.d, xzr, x20, vlx4"])],
            culprit: "whilelt pn9.d, xzr, x20, vlx4",
        },
    ];
    let registered = SME_KERNELS[0].template;
    for mutant in &MUTANTS {
        let template = template_with(registered, mutant.edits);
        assert_ne!(template, registered, "{}", mutant.label);
        if let Err(errors) = lint_streaming_template(&template) {
            panic!(
                "{}: the lint rejects the mutant:\n{}",
                mutant.label,
                report(&errors)
            );
        }
        match check_row_predicates(&template) {
            Ok(()) => panic!("{}: the checker accepts the mutant", mutant.label),
            Err(error) => assert!(
                error.contains(&format!("{:?}", mutant.culprit)),
                "{}: rejected for another reason: {error}",
                mutant.label
            ),
        }
    }
}

/// Every A load of variant `4L` (label 41..48, its loop 6L and its remainder
/// 7L) uses the predicate of its rows: the first group `[x14]` (rows
/// `i0..+31`) `p2/z` for one live tile and `pn9/z` otherwise, the second
/// group `[x14, #4, mul vl]` (rows `i0+32..+63`) `pn10/z`, and only when
/// rows past `i0+31` are live (L >= 5). The group widths are those of the
/// generator: 1, 2, 4, 4, 4, ... for the first group and 2 (L = 5, 6) or 4
/// (L = 7, 8) for the second.
fn check_a_loads(sections: &[(usize, Vec<&str>)]) {
    for live in 1..=8_usize {
        let label = 40 + live;
        let (mut first_group, mut second_group) = (0, 0);
        for line in section(sections, label) {
            assert!(!is_store(line), "variant {live} stores: {line}");
            if !is_load(line) {
                continue;
            }
            let Some((register, width, predicate, address)) = vector_access(line) else {
                panic!("variant {live}: unexpected load `{line}`");
            };
            match address {
                // B rows: the packed panel is zero-padded, loaded all-true.
                "x15" | "x15, #1, mul vl" => {
                    assert_eq!(
                        (width, predicate),
                        (1, "p0/z"),
                        "variant {live}: B load `{line}`"
                    );
                }
                "x14" => {
                    let want = match live {
                        1 => (1, "p2/z"),
                        2 => (2, "pn9/z"),
                        _ => (4, "pn9/z"),
                    };
                    assert_eq!(
                        (width, predicate),
                        want,
                        "variant {live}: first A group `{line}`"
                    );
                    assert!(
                        register == 0 || register == 16,
                        "variant {live}: first A group in z{register}: `{line}`"
                    );
                    first_group += 1;
                }
                "x14, #4, mul vl" => {
                    assert!(
                        live >= 5,
                        "variant {live} reads A rows i0+32.. although none is live: `{line}`"
                    );
                    let want = if live <= 6 { 2 } else { 4 };
                    assert_eq!(
                        (width, predicate),
                        (want, "pn10/z"),
                        "variant {live}: second A group `{line}`"
                    );
                    assert!(
                        register == 4 || register == 20,
                        "variant {live}: second A group in z{register}: `{line}`"
                    );
                    second_group += 1;
                }
                _ => panic!("variant {live}: load from an unexpected address `{line}`"),
            }
        }
        // Two depth steps in the loop and one in the remainder.
        assert_eq!(first_group, 3, "variant {live}: loads of A rows i0..+31");
        let want = if live >= 5 { 3 } else { 0 };
        assert_eq!(
            second_group, want,
            "variant {live}: loads of A rows i0+32.."
        );
    }
}

/// C is read and written only under the epilogue's row predicate `p1`
/// (rows of the tile below `m`), at the slice address `x15`; M0 (51) and M2
/// (53) contain no load at all; M1 (52) and M3 (54) load only the C column
/// of the slice. No other section of the body reads or writes memory, except
/// the prologue's reads of the argument block.
fn check_epilogue_memory(sections: &[(usize, Vec<&str>)]) {
    const C_LOAD: &str = "ld1d {{z17.d}}, p1/z, [x15]";
    for (label, lines) in sections {
        let loads: Vec<&str> = lines.iter().copied().filter(|line| is_load(line)).collect();
        let stores: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|line| is_store(line))
            .collect();
        match label {
            0 => {
                assert!(
                    loads
                        .iter()
                        .all(|line| line.ends_with(']') && line.contains(", [x0, #")),
                    "the prologue reads only the argument block: {loads:?}"
                );
                assert!(stores.is_empty(), "the prologue stores: {stores:?}");
            }
            // Checked by `check_a_loads`.
            41..=48 => {}
            51..=54 => {
                let mode = label - 51;
                if mode % 2 == 0 {
                    assert!(loads.is_empty(), "M{mode} reads memory: {loads:?}");
                } else {
                    assert!(
                        loads.iter().all(|line| *line == C_LOAD),
                        "M{mode}: every load must be `{C_LOAD}`: {loads:?}"
                    );
                    assert_eq!(loads.len(), 8, "M{mode}: one C load per tile");
                }
                // A ZA slice store `st1d {{za{t}h.d[w12, 0]}}, p1, [x15]` or
                // a vector store `st1d {{z16.d}}, p1, [x15]`.
                for line in &stores {
                    assert!(
                        mnemonic(line) == "st1d" && line.ends_with("}}, p1, [x15]"),
                        "M{mode}: every store must be under p1 at [x15]: `{line}`"
                    );
                }
                assert_eq!(stores.len(), 8, "M{mode}: one store per tile");
            }
            _ => {
                assert!(loads.is_empty(), "section {label} reads memory: {loads:?}");
                assert!(
                    stores.is_empty(),
                    "section {label} writes memory: {stores:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shapes, layouts, modes
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_kernel_f64_grid_store_packed_and_in_place() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_grid_store_packed_and_in_place",
        Need::F64,
    ) {
        return;
    }
    grid("M0", &PAIRS[0..2], &[Layout::Packed, Layout::InPlace]);
}

#[test]
fn test_sme2_kernel_f64_grid_add_packed_and_in_place() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_grid_add_packed_and_in_place",
        Need::F64,
    ) {
        return;
    }
    grid("M1", &PAIRS[2..3], &[Layout::Packed, Layout::InPlace]);
}

#[test]
fn test_sme2_kernel_f64_grid_scale_packed_and_in_place() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_grid_scale_packed_and_in_place",
        Need::F64,
    ) {
        return;
    }
    grid("M2", &PAIRS[3..5], &[Layout::Packed, Layout::InPlace]);
}

#[test]
fn test_sme2_kernel_f64_grid_combine_packed() {
    if !hardware_or_skip("test_sme2_kernel_f64_grid_combine_packed", Need::F64) {
        return;
    }
    grid("M3", &PAIRS[5..10], &[Layout::Packed]);
}

#[test]
fn test_sme2_kernel_f64_grid_combine_in_place() {
    if !hardware_or_skip("test_sme2_kernel_f64_grid_combine_in_place", Need::F64) {
        return;
    }
    grid("M3", &PAIRS[5..10], &[Layout::InPlace]);
}

#[test]
fn test_sme2_kernel_f64_grid_strided_layout() {
    if !hardware_or_skip("test_sme2_kernel_f64_grid_strided_layout", Need::F64) {
        return;
    }
    // One pair per mode on stretched strides.
    grid(
        "strided",
        &[PAIRS[0], PAIRS[2], PAIRS[4], PAIRS[5]],
        &[Layout::Strided],
    );
}

#[test]
fn test_sme2_kernel_f64_small_shapes_every_mode() {
    if !hardware_or_skip("test_sme2_kernel_f64_small_shapes_every_mode", Need::F64) {
        return;
    }
    let mut values = Values::new(0x0005_ca1e);
    for m in 1..=20 {
        for n in 1..=20 {
            for kc in [1, 5] {
                for layout in [Layout::Packed, Layout::InPlace, Layout::Strided] {
                    for pair in PAIRS {
                        check_case((m, n, kc), layout, pair, &mut values);
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f64_columns_around_64_every_mode() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_columns_around_64_every_mode",
        Need::F64,
    ) {
        return;
    }
    // n = 63, 64, 65: eight column panels (the last one partial or full) and
    // nine (one column in the last), after seven or eight advances of the B
    // panel and C column pointers; a few m around one tile and 64, kc with
    // and without the remainder step.
    let mut values = Values::new(0x0064_c015);
    let mut calls = 0_usize;
    for m in [1, 9, 64, 65] {
        for n in [63, 64, 65] {
            for kc in [1, 2, 65] {
                for layout in [Layout::Packed, Layout::InPlace, Layout::Strided] {
                    for pair in PAIRS {
                        check_case((m, n, kc), layout, pair, &mut values);
                        calls += 1;
                    }
                }
            }
        }
    }
    println!("n around 64: {calls} calls");
}

// ---------------------------------------------------------------------------
// Special values
// ---------------------------------------------------------------------------

/// Operand values: signed zeros, infinities, NaN, subnormals, values whose
/// products overflow or underflow, and ordinary numbers.
const SPECIALS: [f64; 16] = [
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1.0e-310,
    -2.5e-315,
    f64::MIN_POSITIVE,
    1.0e-160,
    1.0e300,
    -1.0e300,
    f64::MAX,
    1.5,
    -3.0,
    0.1,
    7.0,
];

/// Alpha and beta values, special ones included (`0`, `-0`, `1` select the
/// cheap modes).
const SCALARS: [f64; 12] = [
    1.0,
    0.0,
    -0.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1.0e-310,
    1.0e300,
    -0.75,
    f64::MIN_POSITIVE,
    0.1,
    -1.0,
];

#[test]
fn test_sme2_kernel_f64_special_values() {
    if !hardware_or_skip("test_sme2_kernel_f64_special_values", Need::F64) {
        return;
    }
    let mut values = Values::new(0x0005_bec1);
    for (m, n, kc) in [(3, 2, 2), (9, 9, 3), (65, 17, 2), (70, 9, 5)] {
        for layout in [Layout::Packed, Layout::InPlace] {
            for alpha in SCALARS {
                for beta in SCALARS {
                    let mode = Epilogue::select(alpha, beta);
                    // Half special, half ordinary, so that both finite and
                    // non-finite accumulators occur.
                    let mut a_values = Values::new(values.bits());
                    let mut b_values = Values::new(values.bits());
                    let special = |v: &mut Values| {
                        if v.bits() & 1 == 0 {
                            v.pick(&SPECIALS)
                        } else {
                            v.mixed()
                        }
                    };
                    let ops = Operands::new(
                        (m, n, kc),
                        layout,
                        |_, _| special(&mut a_values),
                        |_, _| special(&mut b_values),
                    );
                    let mut c = if mode.reads_c() {
                        CBlock::new(m, n, |_, _| special(&mut values))
                    } else {
                        CBlock::new(m, n, |_, _| f64::from_bits(UNREAD_C))
                    };
                    let want = run(&ops, &mut c, alpha, beta);
                    c.check(
                        &want,
                        &format!("special {m}x{n}x{kc} {layout:?} alpha {alpha} beta {beta}"),
                    );
                }
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f64_nan_c_does_not_leak_without_beta() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_nan_c_does_not_leak_without_beta",
        Need::F64,
    ) {
        return;
    }
    // beta = ±0: C is never read, so neither NaN nor infinity in C may reach
    // the result (the reference does not read it either).
    let mut values = Values::new(0x0000_0bad);
    for (m, n, kc) in [(1, 1, 1), (64, 8, 2), (129, 23, 3), (17, 9, 1)] {
        for (alpha, beta) in [(1.0, 0.0), (1.0, -0.0), (-2.0, 0.0), (0.5, -0.0)] {
            for poison in [f64::NAN, f64::INFINITY, f64::from_bits(UNREAD_C)] {
                let ops = Operands::mixed((m, n, kc), Layout::Packed, &mut values);
                let mut c = CBlock::new(m, n, |_, _| poison);
                let want = run(&ops, &mut c, alpha, beta);
                c.check(
                    &want,
                    &format!("poison {poison} {m}x{n}x{kc} alpha {alpha}"),
                );
                for j in 0..n {
                    for i in 0..m {
                        let got = c.data[i + (j + 1) * c.ldc];
                        assert!(got.is_finite(), "C({i}, {j}) = {got} leaked from C");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FPSR
// ---------------------------------------------------------------------------

/// Flags the caller may have set before a call.
const EARLIER: [u64; 6] = [
    0,
    FPSR_DZC,
    FPSR_DZC | FPSR_OFC,
    FPSR_IOC | FPSR_UFC | FPSR_IDC,
    FPSR_IXC,
    FPSR_QC,
];

/// FPSR after one call that starts from `before`, and the call's C.
fn fpsr_across(ops: &Operands, c: &mut CBlock, alpha: f64, beta: f64, before: u64) -> u64 {
    let args = std::hint::black_box(ops.args(c, alpha, beta));
    assert!(detect::thread_ready());
    let saved = fpsr_read();
    fpsr_write(before);
    // SAFETY: as in `run` (the caller passed the hardware gate).
    unsafe { sme2_gemm_f64(&args) };
    let after = fpsr_read();
    fpsr_write(saved);
    after
}

/// One FPSR case: operands, C and the flags the call must add.
struct FlagCase {
    label: &'static str,
    alpha: f64,
    beta: f64,
    /// A(i, p) and B(p, j).
    a: fn(&mut Values) -> f64,
    /// C(i, j) (ignored when the mode does not read C).
    c: fn(&mut Values) -> f64,
    /// The flags the call raises.
    raised: u64,
}

#[test]
fn test_sme2_kernel_f64_fpsr_flags_per_mode() {
    if !hardware_or_skip("test_sme2_kernel_f64_fpsr_flags_per_mode", Need::F64) {
        return;
    }
    let third = std::hint::black_box(1.0_f64) / std::hint::black_box(3.0_f64);
    let cases = [
        // M0: FMOPA and the ZA slice stores raise nothing, even for inexact
        // or overflowing accumulations.
        FlagCase {
            label: "M0 exact",
            alpha: 1.0,
            beta: 0.0,
            a: |v| v.small(4),
            c: |_| 0.0,
            raised: 0,
        },
        FlagCase {
            label: "M0 inexact accumulation",
            alpha: 1.0,
            beta: 0.0,
            a: |v| v.mixed() + 0.1,
            c: |_| 0.0,
            raised: 0,
        },
        FlagCase {
            label: "M0 overflowing accumulation",
            alpha: 1.0,
            beta: 0.0,
            a: |v| v.small(4) * 1.0e200,
            c: |_| 0.0,
            raised: 0,
        },
        // M1: the SVE FADD of acc + C.
        FlagCase {
            label: "M1 exact",
            alpha: 1.0,
            beta: 1.0,
            a: |v| v.small(4),
            c: |v| v.small(9),
            raised: 0,
        },
        FlagCase {
            label: "M1 inexact",
            alpha: 1.0,
            beta: 1.0,
            a: |v| v.small(4),
            c: |v| v.small(9) + 0.1,
            raised: FPSR_IXC,
        },
        FlagCase {
            label: "M1 infinity minus infinity",
            alpha: 1.0,
            beta: 1.0,
            a: |v| v.small(4) * 1.0e200,
            c: |_| f64::NEG_INFINITY,
            raised: FPSR_IOC,
        },
        // M2: the SVE FMUL by alpha.
        FlagCase {
            label: "M2 exact",
            alpha: 0.5,
            beta: 0.0,
            a: |v| v.small(4),
            c: |_| 0.0,
            raised: 0,
        },
        FlagCase {
            label: "M2 inexact",
            alpha: third,
            beta: 0.0,
            a: |v| v.small(4),
            c: |_| 0.0,
            raised: FPSR_IXC,
        },
        FlagCase {
            label: "M2 overflow",
            alpha: f64::MAX / 2.0,
            beta: 0.0,
            a: |v| v.small(4),
            c: |_| 0.0,
            raised: FPSR_OFC | FPSR_IXC,
        },
        // Exact on the live lanes; an unpredicated FMUL would compute
        // 0 * inf on the dead lanes and raise IOC.
        FlagCase {
            label: "M2 infinite alpha",
            alpha: f64::INFINITY,
            beta: 0.0,
            a: |v| v.small(4),
            c: |_| 0.0,
            raised: 0,
        },
        // M3: the SVE FMUL by alpha and the SVE FMLA t + C * beta.
        FlagCase {
            label: "M3 exact",
            alpha: 0.5,
            beta: 0.25,
            a: |v| v.small(4),
            c: |v| v.small(9),
            raised: 0,
        },
        FlagCase {
            label: "M3 inexact",
            alpha: third,
            beta: 0.25,
            a: |v| v.small(4),
            c: |v| v.small(9),
            raised: FPSR_IXC,
        },
        // Exact on the live lanes; the dead lanes of the zeroing C load hold
        // 0, so an unpredicated FMLA would raise IOC from 0 * inf.
        FlagCase {
            label: "M3 infinite beta",
            alpha: 1.0,
            beta: f64::INFINITY,
            a: |v| v.small(4),
            c: |v| v.small(9),
            raised: 0,
        },
    ];
    // Partial tiles and a partial column panel, and several panels.
    for (m, n, kc) in [(13, 11, 3), (70, 9, 5)] {
        for case in &cases {
            for before in EARLIER {
                let mut values = Values::new(before + m as u64);
                let mut a_values = Values::new(values.bits());
                let mut b_values = Values::new(values.bits());
                let ops = Operands::new(
                    (m, n, kc),
                    Layout::Packed,
                    |_, _| (case.a)(&mut a_values),
                    |_, _| (case.a)(&mut b_values),
                );
                let mode = Epilogue::select(case.alpha, case.beta);
                let mut c = if mode.reads_c() {
                    CBlock::new(m, n, |_, _| (case.c)(&mut values))
                } else {
                    CBlock::new(m, n, |_, _| f64::from_bits(UNREAD_C))
                };
                let mut want = c.clone();
                let want_args = ops.args(&mut want, case.alpha, case.beta);
                let after = fpsr_across(&ops, &mut c, case.alpha, case.beta, before);
                assert_eq!(
                    after,
                    before | case.raised,
                    "{} {m}x{n}x{kc}: FPSR {before:#x} -> {after:#x}, want {:#x}",
                    case.label,
                    before | case.raised
                );
                // SAFETY: `ops` and `want` cover the block (see `run`).
                unsafe { reference(&want_args) };
                c.check(&want, case.label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thread state and interleaving with NEON / scalar work
// ---------------------------------------------------------------------------

/// Packs column-major `a` (`m × kc`) into row panels of `mr` (NEON layout).
fn pack_rows(a: &[f64], m: usize, kc: usize, mr: usize) -> Vec<f64> {
    let mut packed = vec![0.0; m.div_ceil(mr) * mr * kc];
    for i in 0..m {
        for p in 0..kc {
            packed[(i / mr) * mr * kc + p * mr + i % mr] = a[i + p * m];
        }
    }
    packed
}

/// Packs row-major `b` (`kc × n`) into column panels of `nr` (NEON layout).
fn pack_cols(b: &[f64], kc: usize, n: usize, nr: usize) -> Vec<f64> {
    let mut packed = vec![0.0; n.div_ceil(nr) * nr * kc];
    for p in 0..kc {
        for j in 0..n {
            packed[(j / nr) * nr * kc + p * nr + j % nr] = b[p * n + j];
        }
    }
    packed
}

/// The NEON blocked product `C = alpha * A * B + beta * C` of the same
/// operands (`ldc = m`).
fn neon_product(
    a: &[f64],
    b: &[f64],
    (m, n, kc): (usize, usize, usize),
    (alpha, beta): (f64, f64),
    c: &mut [f64],
) {
    let kernel = f64::micro_kernel_dispatch();
    let (mr, nr) = (kernel.shape.mr, kernel.shape.nr);
    let packed_a = pack_rows(a, m, kc, mr);
    let packed_b = pack_cols(b, kc, n, nr);
    let Some(view) = MatMut::from_strided(c, m, n, m) else {
        panic!("C view {m}x{n}");
    };
    macro_panel_multiply_with(alpha, &packed_a, m, kc, &packed_b, n, beta, view, &kernel);
}

#[test]
fn test_sme2_kernel_f64_thread_state_and_interleaving() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_thread_state_and_interleaving",
        Need::F64,
    ) {
        return;
    }
    let (m, n, kc) = (70, 11, 9);
    let mut values = Values::new(0x0000_7a1e);
    let a: Vec<f64> = (0..m * kc).map(|_| values.mixed()).collect();
    let b: Vec<f64> = (0..kc * n).map(|_| values.mixed()).collect();
    let c0: Vec<f64> = (0..m * n).map(|_| values.mixed()).collect();
    // The SME2 operands: A read in place (lda = m + 5), B packed.
    let ops = Operands::new(
        (m, n, kc),
        Layout::InPlace,
        |i, p| a[i + p * m],
        |p, j| b[p * n + j],
    );
    let (alpha, beta) = (0.75, -0.5);
    let fresh_c = || CBlock::new(m, n, |i, j| c0[i + j * m]);

    let mut first: Option<Vec<u64>> = None;
    let mut neon_first: Option<Vec<u64>> = None;
    let mut scalar = std::hint::black_box(1.0_f64);
    for round in 0..120_u32 {
        let mut c = fresh_c();
        assert!(detect::thread_ready(), "round {round}: thread not ready");
        let entries = kernel_entries();
        let want = run(&ops, &mut c, alpha, beta);
        assert!(
            kernel_entries() > entries,
            "round {round}: entry not counted"
        );
        assert_eq!(
            sme_state(),
            Some((0, 0)),
            "round {round}: SVCR / TPIDR2_EL0"
        );
        c.check(&want, &format!("round {round}"));
        let bits: Vec<u64> = c.data.iter().map(|v| v.to_bits()).collect();
        match &first {
            None => first = Some(bits),
            Some(first) => assert_eq!(first, &bits, "round {round}: SME2 result changed"),
        }

        // NEON work on the same thread, then scalar f64 work kept live
        // across the next kernel call. The NEON blocked path computes the
        // same call (same alpha, beta and initial C), and the contract makes
        // it bit-equal to the SME2 block: compare the two backends directly.
        let mut neon_c = c0.clone();
        neon_product(&a, &b, (m, n, kc), (alpha, beta), &mut neon_c);
        for j in 0..n {
            for i in 0..m {
                let (sme2, neon) = (c.data[i + (j + 1) * c.ldc], neon_c[i + j * m]);
                assert!(
                    same_result(sme2, neon),
                    "round {round}: C({i}, {j}) SME2 {sme2:e} ({:#x}), NEON {neon:e} ({:#x})",
                    sme2.to_bits(),
                    neon.to_bits()
                );
            }
        }
        let neon_bits: Vec<u64> = neon_c.iter().map(|v| v.to_bits()).collect();
        match &neon_first {
            None => neon_first = Some(neon_bits),
            Some(first) => assert_eq!(first, &neon_bits, "round {round}: NEON result changed"),
        }
        scalar = std::hint::black_box(scalar.mul_add(1.000_001, f64::from(round)));
    }
    assert!(scalar.is_finite() && scalar > 1.0);
    assert!(detect::thread_ready());
}
