//! Plain-Rust tests of the ZA packers (`pack_za.rs`), run on every host:
//! the registry and the lint, the static structure of the transposer
//! templates, the interpreter of the registered text on the grids and on
//! mutated templates, the delegated orientations, the `None` cases, the
//! block lengths, one transposer call per request, and the first-use check
//! with stand-ins. The cases, the grids and the shared runner are those of
//! `tests_pack_za.rs`.
//!
//! A read outside the block cannot be seen in any result (such a lane is
//! either a dead depth lane of the last chunk, whose vertical slices are
//! never stored, or a dead column, whose slice is never stored either), so
//! two native guards look at the registered template text itself:
//!
//! - `test_sme2_pack_za_template_load_predicates` proves statically that
//!   every source load is governed by `psel(whilelt(p0, depth),
//!   whilelt(c0, cols)[column])` with the column index and the column
//!   pointer advanced together; `..._checker_rejects_mutants` shows that the
//!   checker rejects wrong templates the lint accepts.
//! - An interpreter of the registered text (`pack_za::model`) executes the
//!   templates natively, on real memory, with every access checked against
//!   the block (reads) and the packed length (writes) before it happens.
//!   It runs the whole grid on any host, and shows that mutated templates
//!   read outside the block.
//!
//! The interpreter models the instructions the transposers use (and fails
//! on any other), with the SVL-512 shapes of the hardware; QEMU (and the
//! first-use check on real hardware) remain the judges of the architecture.

use std::cell::RefCell;

use oxiblas_core::memory::AlignedVec;
use oxiblas_matrix::MatRef;

use super::args::SmeElement;
use super::lint::{lint_kernel, report};
use super::pack::{
    self, ColMajor, PackedA, PackedB, Span, a_block_len, a_pack_len, b_block_len, b_pack_len,
};
use super::pack_za::model::{Program, Window, operands, run_template, statements};
use super::pack_za::{
    CHECK_BLOCKS, GROUP_COLS, SME_KERNELS, SmePack, ZaKernelFn, ZaPackArgs, check_bits,
    pack_a_za_with, pack_b_za_with, self_check_with, za_transpose_f32, za_transpose_f64,
};
use super::tests_pack_za::{
    Case, DEPTHS, REPORT, WINDOW, ensure, grid, note, run_all, run_case, stale_bits, widths,
};
use crate::level3::trsm::Trans;

/// Interprets the registered transposer `name` on `args`, against the
/// window of the current case.
fn interpret<T>(name: &str, args: &ZaPackArgs<T>) -> Result<(), String> {
    let entry = SME_KERNELS
        .iter()
        .find(|entry| entry.name == name)
        .ok_or("no such kernel")?;
    interpret_template(entry.template, args)
}

/// Interprets a template (registered or mutated) on `args`.
fn interpret_template<T>(template: &str, args: &ZaPackArgs<T>) -> Result<(), String> {
    let window = WINDOW
        .with(|slot| *slot.borrow())
        .ok_or("no window for this call")?;
    run_template(template, args, &window)
}

/// The `f32` transposer, interpreted (a stand-in kernel).
fn interpreted_f32(args: &ZaPackArgs<f32>) {
    if let Err(why) = interpret("za_transpose_f32", args) {
        note(why);
    }
}

/// The `f64` transposer, interpreted (a stand-in kernel).
fn interpreted_f64(args: &ZaPackArgs<f64>) {
    if let Err(why) = interpret("za_transpose_f64", args) {
        note(why);
    }
}

thread_local! {
    /// The template the mutant stand-ins interpret.
    static MUTANT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// The current mutant (`f32`), interpreted.
fn interpreted_mutant_f32(args: &ZaPackArgs<f32>) {
    let template = MUTANT.with(|m| m.borrow().clone());
    if let Err(why) = interpret_template(&template, args) {
        note(why);
    }
}

/// The current mutant (`f64`), interpreted.
fn interpreted_mutant_f64(args: &ZaPackArgs<f64>) {
    let template = MUTANT.with(|m| m.borrow().clone());
    if let Err(why) = interpret_template(&template, args) {
        note(why);
    }
}

// ---------------------------------------------------------------------------
// Static structure of the templates
// ---------------------------------------------------------------------------

/// The registered entry of a transposer.
fn entry(name: &str) -> &'static super::asm::KernelEntry {
    SME_KERNELS
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("{name} is not registered"))
}

/// The registers (and predicate registers) a statement names, as
/// normalised tokens (`w13` → `x13`, `pn5` → `p5`).
fn tokens(line: &str) -> Vec<String> {
    line.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| {
            if let Some(n) = t.strip_prefix('w').filter(|n| n.parse::<u8>().is_ok()) {
                format!("x{n}")
            } else if let Some(n) = t.strip_prefix("pn").filter(|n| n.parse::<u8>().is_ok()) {
                format!("p{n}")
            } else {
                t.to_owned()
            }
        })
        .collect()
}

/// The registers a statement writes: its first operand, every register of
/// a braced destination list (`{p4.s, p5.s}`, `{p4.s-p5.s}` counts both
/// ends and everything between), normalised as in [`tokens`]. Stores,
/// compares and branches write no register.
fn written(line: &str) -> Vec<String> {
    let (mnemonic, rest) = line.split_once(' ').unwrap_or((line, ""));
    if mnemonic.starts_with("st") || mnemonic.starts_with('b') || mnemonic == "cmp" {
        return Vec::new();
    }
    let parts = operands(rest);
    let Some(first) = parts.first() else {
        return Vec::new();
    };
    let mut regs = tokens(first);
    if first.contains('-') {
        let ps: Vec<u8> = regs
            .iter()
            .filter_map(|t| t.strip_prefix('p')?.parse().ok())
            .collect();
        if let [lo, hi] = ps[..] {
            regs = (lo..=hi).map(|n| format!("p{n}")).collect();
        }
    }
    regs.retain(|t| (t.starts_with('x') || t.starts_with('p')) && t[1..].parse::<u8>().is_ok());
    regs
}

/// Checks that every source load of a transposer template is governed by
/// `psel(depth predicate, column predicate[column])`, see the module docs.
/// `esize` is the element size, `chunk` the depth steps of a chunk.
fn check_load_predicates(template: &str, esize: usize, chunk: usize) -> Result<(), String> {
    let lines = statements(template, &[]);
    let start = lines
        .iter()
        .position(|line| line == "msr fpsr, xzr")
        .ok_or("no body start")?;
    let stop = lines
        .iter()
        .position(|line| line == "90:")
        .ok_or("no exit label")?;
    let suffix = if esize == 4 { "s" } else { "d" };
    let shift = esize.trailing_zeros();
    // Four `.s` tiles or eight `.d` tiles.
    let tiles = esize;
    // Who may write the registers of the proof, anywhere in the template.
    let rules: [(&str, Vec<String>); 12] = [
        ("x1", vec!["ldr x1, [x0, #{off_src}]".into()]),
        (
            "x2",
            vec![
                "ldr x2, [x0, #{off_ld}]".into(),
                format!("lsl x2, x2, #{shift}"),
            ],
        ),
        ("x4", vec!["ldr x4, [x0, #{off_depth}]".into()]),
        ("x5", vec!["ldr x5, [x0, #{off_cols}]".into()]),
        ("x9", vec!["mov x9, #0".into(), "add x9, x9, #64".into()]),
        (
            "x10",
            vec!["mov x10, x1".into(), "add x10, x10, x2, lsl #6".into()],
        ),
        ("x13", vec!["mov w13, #0".into(), "add w13, w13, #1".into()]),
        (
            "x22",
            vec!["mov x22, #0".into(), format!("add x22, x22, #{chunk}")],
        ),
        (
            "x24",
            vec!["mov x24, x10".into(), "add x24, x24, x2".into()],
        ),
        ("p1", vec![format!("whilelt p1.{suffix}, x22, x4")]),
        ("p2", vec!["whilelt p2.b, x9, x5".into()]),
        ("p5", vec!["psel p5, p1, p2.b[w13, 0]".into()]),
    ];
    for line in &lines {
        for reg in written(line) {
            if let Some((_, allowed)) = rules.iter().find(|(r, _)| *r == reg) {
                ensure!(allowed.contains(line), "{line}");
            }
        }
    }
    // `p1`, `p2` and `p5` appear nowhere else (no copy, no other source).
    let load = |line: &str| line.starts_with("ld1") && line.contains("{za");
    for line in &lines {
        let names: Vec<String> = tokens(line);
        for p in ["p1", "p2", "p5"] {
            if names.iter().any(|t| t == p) {
                let allowed = rules.iter().any(|(_, a)| a.contains(line))
                    || (p == "p5" && load(line) && line.contains(", p5/z, "));
                ensure!(allowed, "{line}");
            }
        }
    }
    let count = |want: &str| lines.iter().filter(|line| *line == want).count();
    ensure!(
        count(&format!("whilelt p1.{suffix}, x22, x4")) == 1 && count("whilelt p2.b, x9, x5") == 1,
        "the depth and column predicates are not written exactly once each"
    );
    // Every load: `p5/z`, column pointer `x24`, depth index `x22`, right
    // after its `psel`, and followed by the joint advance of the column
    // pointer and the column index before the next load.
    let body = &lines[start + 1..stop];
    let mut seen = vec![false; tiles];
    let mut loads = 0;
    for (at, line) in body.iter().enumerate() {
        if !line.starts_with("ld") {
            continue;
        }
        loads += 1;
        let tile = (0..tiles).find(|t| {
            *line
                == format!(
                    "ld1{} {{za{t}h.{suffix}[w12, 0]}}, p5/z, [x24, x22, lsl #{shift}]",
                    if esize == 4 { "w" } else { "d" }
                )
        });
        let Some(tile) = tile else {
            return Err(line.clone());
        };
        ensure!(!seen[tile], "{line}");
        seen[tile] = true;
        ensure!(
            at > 0 && body[at - 1] == "psel p5, p1, p2.b[w13, 0]",
            "{line}"
        );
        ensure!(
            body.get(at + 1).map(String::as_str) == Some("add x24, x24, x2")
                && body.get(at + 2).map(String::as_str) == Some("add w13, w13, #1"),
            "{line}"
        );
    }
    ensure!(
        loads == tiles && seen.iter().all(|s| *s),
        "{loads} loads, want one per tile"
    );
    // Every advance of the column pointer is paired with one of the column
    // index, and both restart together.
    for (at, line) in body.iter().enumerate() {
        if line == "add x24, x24, x2" {
            ensure!(
                body.get(at + 1).map(String::as_str) == Some("add w13, w13, #1"),
                "{line}"
            );
        }
        if line == "add w13, w13, #1" {
            ensure!(at > 0 && body[at - 1] == "add x24, x24, x2", "{line}");
        }
        if line == "mov x24, x10" {
            ensure!(
                body.get(at + 1).map(String::as_str) == Some("mov w13, #0"),
                "{line}"
            );
        }
        if line == "mov w13, #0" {
            ensure!(at > 0 && body[at - 1] == "mov x24, x10", "{line}");
        }
        if line == "add x9, x9, #64" {
            ensure!(
                body.get(at + 1).map(String::as_str) == Some("add x10, x10, x2, lsl #6"),
                "{line}"
            );
        }
    }
    Ok(())
}

/// A template with `edits` applied: each `(anchor, replacement)` replaces
/// the first statement equal to `anchor` (registry escapes as written) by
/// the replacement lines.
fn mutate(template: &str, edits: &[(&str, &[&str])]) -> Result<String, String> {
    let mut lines: Vec<String> = template.lines().map(str::to_owned).collect();
    for (anchor, replacement) in edits {
        let at = lines
            .iter()
            .position(|line| line.trim() == *anchor)
            .ok_or_else(|| format!("anchor {anchor:?} not found"))?;
        lines.splice(at..=at, replacement.iter().map(|l| (*l).to_owned()));
    }
    Ok(lines.join("\n"))
}

/// One mutated `f32` template.
struct Mutant {
    /// What the mutation does.
    label: &'static str,
    /// Statement replacements.
    edits: &'static [(&'static str, &'static [&'static str])],
    /// Whether the static load checker must reject it.
    static_rejects: bool,
    /// Whether the interpreted grid must see a read outside the block.
    reads_outside: bool,
}

/// Mutations of `za_transpose_f32` that the lint accepts and no result test
/// can see (each only reads outside the block), and two more for contrast.
const MUTANTS: [Mutant; 8] = [
    Mutant {
        label: "the psel before tile 0's loads is missing (p5 left from the previous chunk)",
        edits: &[("psel p5, p1, p2.b[w13, 0]", &[])],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "tile 1 loads with the all-true predicate",
        edits: &[(
            "ld1w {{za1h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            &["ld1w {{za1h.s[w12, 0]}}, p0/z, [x24, x22, lsl #2]"],
        )],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "psel picks the all-true predicate instead of the depth predicate",
        edits: &[("psel p5, p1, p2.b[w13, 0]", &["psel p5, p0, p2.b[w13, 0]"])],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "the column predicate bound is cols + 1",
        edits: &[(
            "whilelt p2.b, x9, x5",
            &["add x16, x5, #1", "whilelt p2.b, x9, x16"],
        )],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "the depth predicate runs one chunk lane further",
        edits: &[(
            "whilelt p1.s, x22, x4",
            &["add x16, x4, #1", "whilelt p1.s, x22, x16"],
        )],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "the column index skips ahead of the column pointer",
        edits: &[("mov w13, #0", &["mov w13, #1"])],
        static_rejects: true,
        reads_outside: true,
    },
    Mutant {
        label: "p5 is overwritten through a pair destination before tile 3's loads (lanes up to the stored columns)",
        edits: &[(
            "ld1w {{za3h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            &[
                "whilelt {{p4.s, p5.s}}, x9, x14",
                "ld1w {{za3h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            ],
        )],
        static_rejects: true,
        reads_outside: false,
    },
    Mutant {
        label: "contrast: an extra harmless scratch move",
        edits: &[("mov x16, #64", &["mov x16, #64", "mov x15, x16"])],
        static_rejects: false,
        reads_outside: false,
    },
];

/// The registry template of a mutated `za_transpose_f32`.
fn mutant_template(mutant: &Mutant) -> Result<String, String> {
    mutate(entry("za_transpose_f32").template, mutant.edits)
}

// ---------------------------------------------------------------------------
// Native tests: registry, lint, structure, interpreter
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_pack_za_registry_and_lint() {
    let names: Vec<&str> = SME_KERNELS.iter().map(|entry| entry.name).collect();
    assert_eq!(names, ["za_transpose_f32", "za_transpose_f64"]);
    for entry in SME_KERNELS {
        assert!(entry.file.ends_with("pack_za.rs"), "{}", entry.file);
        assert!(
            !entry.f64f64,
            "{}: tile-slice moves need no sme-f64f64",
            entry.name
        );
        if let Err(errors) = lint_kernel(entry) {
            panic!("{}:\n{}", entry.name, report(&errors));
        }
    }
    // The kernels behind the trait are the registered ones.
    let f32_kernel: ZaKernelFn<f32> = za_transpose_f32;
    let f64_kernel: ZaKernelFn<f64> = za_transpose_f64;
    assert_eq!(<f32 as SmePack>::TRANSPOSE as usize, f32_kernel as usize);
    assert_eq!(<f64 as SmePack>::TRANSPOSE as usize, f64_kernel as usize);
}

#[test]
fn test_sme2_pack_za_templates_only_move_bits() {
    // The transposers must not execute floating-point arithmetic (FPSR
    // unchanged, payloads unchanged): the body uses only these mnemonics.
    const MOVES_ONLY: [&str; 18] = [
        "ptrue", "mov", "whilelt", "sub", "cmp", "csel", "psel", "ld1w", "ld1d", "st1w", "st1d",
        "add", "b.lo", "b.ls", "ldr", "lsl", "neg", "and",
    ];
    for entry in SME_KERNELS {
        let lines = statements(entry.template, &[]);
        let Some(stop) = lines.iter().position(|l| l == "90:") else {
            panic!("{}: no frame", entry.name);
        };
        for line in &lines[..stop] {
            let frame = line.starts_with('.')
                || line.ends_with(':')
                || ((line.starts_with("mrs")
                    || line.starts_with("msr")
                    || line.starts_with("orr"))
                    && (line.contains("fpsr") || line.contains("x17")))
                || (line.starts_with("sm") && !line.contains(' '));
            if frame {
                continue;
            }
            let mnemonic = line.split(' ').next().unwrap_or("");
            assert!(
                MOVES_ONLY.contains(&mnemonic),
                "{}: {line:?} is not a pure data move or integer step",
                entry.name
            );
        }
        // Every decoded statement is known to the interpreter, too.
        if let Err(why) = Program::new(entry.template) {
            panic!("{}: {why}", entry.name);
        }
    }
}

#[test]
fn test_sme2_pack_za_template_load_predicates() {
    // A STRUCTURAL check of the registered templates, on purpose: a load of
    // a source element outside the block cannot be observed in any result
    // (it only feeds a dead depth lane or a dead column, whose slices are
    // never stored), and the heap past a test buffer does not fault. Only
    // the predicate of each load keeps the kernel inside the block, so this
    // test asserts the property on the text that is assembled: every source
    // load is governed by `p5/z`; `p5` is written only by `psel p5, p1,
    // p2.b[w13, 0]` right before the load; `p1 = whilelt(p0, depth)` and
    // `p2 = whilelt(c0, cols)` (one lane per group column) are written once
    // each and nowhere else; the load reads column pointer `x24` at depth
    // `x22`, and `x24` and the column index `w13` start and advance together
    // (`x24 = src + (c0 + w13) · ld`).
    for (name, esize, chunk) in [("za_transpose_f32", 4, 16), ("za_transpose_f64", 8, 8)] {
        if let Err(culprit) = check_load_predicates(entry(name).template, esize, chunk) {
            panic!("{name}: {culprit}");
        }
    }
}

#[test]
fn test_sme2_pack_za_template_checker_rejects_mutants() {
    // Wrong templates that the lint accepts and that no result test can
    // see: the structural checker names a culprit for each, and the
    // interpreted grid sees each one read outside the block.
    let cases: Vec<Case> = [(17, 3), (16, 16), (1, 1), (33, 20), (65, 64), (7, 70)]
        .into_iter()
        .flat_map(|(d, w)| [Case::b(d, w, true), Case::a(d, w, true)])
        .collect();
    for mutant in &MUTANTS {
        let template = match mutant_template(mutant) {
            Ok(template) => template,
            Err(why) => panic!("{}: {why}", mutant.label),
        };
        let lint = lint_kernel(&super::asm::KernelEntry {
            template: Box::leak(template.clone().into_boxed_str()),
            ..*entry("za_transpose_f32")
        });
        assert!(
            lint.is_ok(),
            "{}: the lint is not the guard here",
            mutant.label
        );
        let checked = check_load_predicates(&template, 4, 16);
        assert_eq!(
            checked.is_err(),
            mutant.static_rejects,
            "{}: {checked:?}",
            mutant.label
        );
        MUTANT.with(|m| *m.borrow_mut() = template);
        let failures: Vec<String> = cases
            .iter()
            // SAFETY: the mutant runs in the interpreter, which checks every
            // access before it happens.
            .filter_map(|case| unsafe { run_case::<f32>(case, interpreted_mutant_f32) }.err())
            .collect();
        let outside = failures
            .iter()
            .any(|f| f.contains("read outside the block"));
        assert_eq!(
            outside, mutant.reads_outside,
            "{}: {failures:?}",
            mutant.label
        );
    }
}

#[test]
fn test_sme2_pack_za_interpreted_b_f32_grid() {
    // SAFETY: the interpreter checks every access before it happens.
    if let Err(why) = unsafe { run_all::<f32>(&grid::<f32>(false), interpreted_f32) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_interpreted_a_f32_grid() {
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f32>(&grid::<f32>(true), interpreted_f32) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_interpreted_b_f64_grid() {
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f64>(&grid::<f64>(false), interpreted_f64) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_interpreted_a_f64_grid() {
    // SAFETY: as above.
    if let Err(why) = unsafe { run_all::<f64>(&grid::<f64>(true), interpreted_f64) } {
        panic!("{why}");
    }
}

#[test]
fn test_sme2_pack_za_interpreted_f64_mutants_read_outside() {
    // The same holes in the f64 template: the tile-7 load without its
    // predicate, and a column bound one too far.
    let f64_entry = entry("za_transpose_f64");
    let mutants: [(&str, &[&str]); 2] = [
        (
            "ld1d {{za7h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            &["ld1d {{za7h.d[w12, 0]}}, p0/z, [x24, x22, lsl #3]"],
        ),
        (
            "whilelt p2.b, x9, x5",
            &["add x16, x5, #1", "whilelt p2.b, x9, x16"],
        ),
    ];
    let cases = [
        Case::b(9, 3, true),
        Case::a(5, 61, true),
        Case::b(3, 60, true),
    ];
    for (anchor, replacement) in mutants {
        let template = match mutate(f64_entry.template, &[(anchor, replacement)]) {
            Ok(template) => template,
            Err(why) => panic!("{why}"),
        };
        assert!(check_load_predicates(&template, 8, 8).is_err(), "{anchor}");
        MUTANT.with(|m| *m.borrow_mut() = template);
        let failures: Vec<String> = cases
            .iter()
            // SAFETY: interpreted, every access checked.
            .filter_map(|case| unsafe { run_case::<f64>(case, interpreted_mutant_f64) }.err())
            .collect();
        assert!(
            failures
                .iter()
                .any(|f| f.contains("read outside the block")),
            "{anchor}: {failures:?}"
        );
    }
}

/// Statement replacements of a mutant (see [`mutate`]).
type Edits = &'static [(&'static str, &'static [&'static str])];

/// Store-side mutants that the load checker cannot see (kernel, what the
/// mutation does, the edits, what the interpreted cases must report): a
/// write outside the packed prefix, or a wrong packed element.
const STORE_MUTANTS: [(&str, &str, Edits, &str); 6] = [
    (
        "za_transpose_f32",
        "every chunk stores one packed row more",
        &[(
            "csel x25, x25, x16, lo",
            &["csel x25, x25, x16, lo", "add x25, x25, #1"],
        )],
        "write outside the packed buffer",
    ),
    (
        "za_transpose_f64",
        "every chunk stores one packed row more",
        &[(
            "csel x25, x25, x16, lo",
            &["csel x25, x25, x16, lo", "add x25, x25, #1"],
        )],
        "write outside the packed buffer",
    ),
    (
        "za_transpose_f32",
        "a stray store one vector before the buffer",
        &[(
            "mov x27, x23",
            &[
                "sub x26, x3, #64",
                "st1w {{za0v.s[w12, 0]}}, p0, [x26]",
                "mov x27, x23",
            ],
        )],
        "write outside the packed buffer",
    ),
    (
        "za_transpose_f64",
        "tile 2 is stored only above 24 stored columns (B groups of 17..24 columns)",
        &[("cmp x21, #16", &["cmp x21, #24"])],
        "element",
    ),
    (
        "za_transpose_f32",
        "tile 2 stores slice q + 1 as packed row p0 + q",
        &[(
            "st1w {{za2v.s[w12, 0]}}, p0, [x26]",
            &["st1w {{za2v.s[w12, 1]}}, p0, [x26]"],
        )],
        "element",
    ),
    (
        "za_transpose_f64",
        "the group destination step is 32 · depth",
        &[("lsl x20, x4, #9", &["lsl x20, x4, #8"])],
        "element",
    ),
];

/// The interpreted failures of the current mutant over `cases`.
fn mutant_failures<T: SmePack>(stand_in: ZaKernelFn<T>, cases: &[Case]) -> Vec<String> {
    cases
        .iter()
        // SAFETY: interpreted, every access checked before it happens.
        .filter_map(|case| unsafe { run_case::<T>(case, stand_in) }.err())
        .collect()
}

#[test]
fn test_sme2_pack_za_interpreted_store_mutants_fail() {
    // Writes beyond the packed prefix (also BEFORE it, which no guard after
    // the buffer can see) and misplaced stores: the interpreter must report
    // each, and the load checker must not be what catches them.
    let cases: Vec<Case> = [(17, 20), (9, 70), (3, 129), (8, 24), (1, 1), (33, 17)]
        .into_iter()
        .flat_map(|(d, w)| [Case::b(d, w, true), Case::a(d, w, false)])
        .collect();
    for (name, label, edits, expect) in STORE_MUTANTS {
        let template = mutate(entry(name).template, edits).unwrap_or_else(|why| panic!("{why}"));
        let (esize, chunk) = if name.ends_with("f32") {
            (4, 16)
        } else {
            (8, 8)
        };
        assert!(
            check_load_predicates(&template, esize, chunk).is_ok(),
            "{name}: {label}: the load checker is not the guard here"
        );
        MUTANT.with(|m| *m.borrow_mut() = template);
        let failures = if esize == 4 {
            mutant_failures::<f32>(interpreted_mutant_f32, &cases)
        } else {
            mutant_failures::<f64>(interpreted_mutant_f64, &cases)
        };
        assert!(
            failures.iter().any(|f| f.contains(expect)),
            "{name}: {label}: want {expect:?}, got {failures:?}"
        );
    }
}

#[test]
fn test_sme2_pack_za_grid_reaches_every_edge() {
    // The grids are only as good as the edges they meet: every number of
    // stored columns a B group can have (each tile-store threshold), and
    // last chunks of 1, 2, TR - 1 and TR rows for both chunk heights.
    fn sweep<T: SmeElement>() {
        let (nr, tr) = (T::NR, T::TILE_ROWS);
        let mut stored = std::collections::BTreeSet::new();
        for width in widths::<T>() {
            for c0 in (0..width).step_by(GROUP_COLS) {
                stored.insert((width.div_ceil(nr) * nr - c0).min(GROUP_COLS));
            }
        }
        let every: std::collections::BTreeSet<usize> =
            (1..=GROUP_COLS / nr).map(|k| k * nr).collect();
        assert_eq!(stored, every, "stored columns of a B group (NR = {nr})");
        let last_rows: Vec<usize> = DEPTHS.iter().map(|d| d - (d - 1) / tr * tr).collect();
        for rows in [1, 2, tr - 1, tr] {
            assert!(last_rows.contains(&rows), "no last chunk of {rows} rows");
        }
        assert!(widths::<T>().iter().any(|w| *w > 2 * GROUP_COLS));
        assert!(DEPTHS.iter().any(|d| *d > 2 * GROUP_COLS));
    }
    sweep::<f32>();
    sweep::<f64>();
}

#[test]
fn test_sme2_pack_za_model_skips_only_the_frame() {
    // The interpreter treats exactly the frame's FPSR statements as no-ops:
    // another statement that names FPSR or x17 is not skipped (it fails to
    // decode), so a mutation hidden in such a line cannot pass unseen.
    for name in ["za_transpose_f32", "za_transpose_f64"] {
        let template = entry(name).template;
        assert!(Program::new(template).is_ok(), "{name}");
        let lines = statements(template, &[]);
        for frame in super::pack_za::model::FRAME {
            assert_eq!(lines.iter().filter(|l| *l == frame).count(), 1, "{frame}");
        }
        for extra in ["orr x24, x24, x17", "msr fpsr, x24", "mrs x24, fpsr"] {
            let mutant = mutate(template, &[("mov x24, x10", &["mov x24, x10", extra])])
                .unwrap_or_else(|why| panic!("{why}"));
            assert!(
                Program::new(&mutant).is_err(),
                "{name}: {extra} was skipped"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Native tests: delegation, None cases, the pack.rs additions, self_check
// ---------------------------------------------------------------------------

thread_local! {
    /// Calls of [`forbidden`] on this thread.
    static FORBIDDEN_CALLS: RefCell<usize> = const { RefCell::new(0) };
}

/// A stand-in that must never be called (delegated requests, `None` cases).
fn forbidden<T>(_: &ZaPackArgs<T>) {
    FORBIDDEN_CALLS.with(|calls| *calls.borrow_mut() += 1);
}

fn forbidden_calls() -> usize {
    FORBIDDEN_CALLS.with(|calls| *calls.borrow())
}

#[test]
fn test_sme2_pack_za_delegated_orientations_match_plain() {
    // `op(B) = Bᵀ` and `op(A) = A` are contiguous copies: the entry points
    // hand them to the plain packers and never call a transposer.
    fn sweep<T: SmePack>() {
        let before = forbidden_calls();
        for depth in [1, 2, 7, 16, 17, 65, 129] {
            for width in [1, 2, T::NR - 1, T::NR, T::NR + 1, 63, 64, 65, 130] {
                for middle in [false, true] {
                    for a_side in [false, true] {
                        let case = Case::new(a_side, false, depth, width, middle);
                        // SAFETY: the stand-in does nothing (and is not called).
                        if let Err(why) = unsafe { run_case::<T>(&case, forbidden::<T>) } {
                            panic!("{why}");
                        }
                    }
                }
            }
        }
        assert_eq!(
            forbidden_calls(),
            before,
            "a delegated request reached the transposer"
        );
    }
    sweep::<f32>();
    sweep::<f64>();
}

#[test]
fn test_sme2_pack_za_none_cases_leave_the_buffer_untouched() {
    fn sweep<T: SmePack>() {
        let before = forbidden_calls();
        let data: Vec<T> = (0..40 * 30)
            .map(|i| T::from_bits64(check_bits::<T>(i, true)))
            .collect();
        let Some(view) = MatRef::from_strided(&data, 40, 30, 40, 1) else {
            panic!("no view");
        };
        let span = |start, len| Span { start, len };
        for trans in [Trans::NoTrans, Trans::Trans] {
            let Some(op) = ColMajor::normalize(view, trans) else {
                panic!("no op");
            };
            let (rows, cols) = (op.rows(), op.cols());
            // (first span, second span, buffer length) of requests to decline.
            let requests = [
                (span(0, 0), span(0, 3), 1000),
                (span(0, 3), span(0, 0), 1000),
                (span(rows, 1), span(0, 3), 1000),
                (span(0, 3), span(cols - 1, 2), 1000),
                (span(usize::MAX, 2), span(0, 3), 1000),
                (span(1, rows), span(0, 3), 100_000),
            ];
            for (first, second, buf_len) in requests {
                for a_side in [false, true] {
                    let fill = |i: usize| T::from_bits64(stale_bits::<T>(i));
                    let mut buf = AlignedVec::<T>::with_capacity(buf_len);
                    (0..buf_len).for_each(|i| buf.push(fill(i)));
                    let mut plain_buf = AlignedVec::<T>::with_capacity(buf_len);
                    (0..buf_len).for_each(|i| plain_buf.push(fill(i)));
                    // SAFETY: the stand-in does nothing; declined requests
                    // never reach it.
                    let (za, plain) = unsafe {
                        if a_side {
                            (
                                pack_a_za_with(forbidden::<T>, &op, first, second, &mut buf)
                                    .is_some(),
                                pack::pack_a(&op, first, second, &mut plain_buf).is_some(),
                            )
                        } else {
                            (
                                pack_b_za_with(forbidden::<T>, &op, first, second, &mut buf)
                                    .is_some(),
                                pack::pack_b(&op, first, second, &mut plain_buf).is_some(),
                            )
                        }
                    };
                    assert!(!plain && !za, "{trans:?} {a_side} {first:?} {second:?}");
                    assert!(
                        buf.as_slice()
                            .iter()
                            .enumerate()
                            .all(|(i, v)| v.bits() == fill(i).bits())
                    );
                }
            }
            // A buffer one element short.
            for a_side in [false, true] {
                let (first, second) = (span(1, 17), span(2, 5));
                let len = if a_side {
                    a_pack_len(17, 5)
                } else {
                    b_pack_len::<T>(17, 5)
                };
                let Some(len) = len else { panic!("len") };
                let mut buf = AlignedVec::<T>::with_capacity(len - 1);
                (0..len - 1).for_each(|i| buf.push(T::from_bits64(stale_bits::<T>(i))));
                // SAFETY: as above.
                let za = unsafe {
                    if a_side {
                        pack_a_za_with(forbidden::<T>, &op, first, second, &mut buf).is_some()
                    } else {
                        pack_b_za_with(forbidden::<T>, &op, first, second, &mut buf).is_some()
                    }
                };
                assert!(!za, "{trans:?} {a_side}: a short buffer was accepted");
                assert!(
                    buf.as_slice()
                        .iter()
                        .enumerate()
                        .all(|(i, v)| v.bits() == stale_bits::<T>(i)),
                    "{trans:?} {a_side}: a declined request wrote the short buffer"
                );
            }
        }
        assert_eq!(
            forbidden_calls(),
            before,
            "a declined request reached the transposer"
        );
    }
    sweep::<f32>();
    sweep::<f64>();
}

#[test]
fn test_sme2_pack_za_block_len_matches_the_plain_packers() {
    // The validators added to `pack.rs` decline exactly what the plain
    // packers decline, and give the length they pack.
    let data: Vec<f32> = (0..20 * 9).map(|i| i as f32).collect();
    let Some(view) = MatRef::from_strided(&data, 20, 9, 20, 1) else {
        panic!("no view");
    };
    for trans in [Trans::NoTrans, Trans::Trans] {
        let Some(op) = ColMajor::normalize(view, trans) else {
            panic!("no op");
        };
        for s1 in 0..=op.rows() + 1 {
            for l1 in [0, 1, 2, 7, 20, usize::MAX] {
                for (s2, l2) in [(0, 1), (0, op.cols()), (op.cols(), 1), (1, 3)] {
                    let (x, y) = (Span { start: s1, len: l1 }, Span { start: s2, len: l2 });
                    for buf_len in [0, 15, 16, 64 * 9, 4000] {
                        let mut buf = AlignedVec::<f32>::zeros(buf_len);
                        let a = pack::pack_a(&op, x, y, &mut buf).map(|p| p.data().len());
                        assert_eq!(
                            a_block_len(&op, x, y, buf_len),
                            a,
                            "A {trans:?} {x:?} {y:?} {buf_len}"
                        );
                        let yx = (Span { start: s2, len: l2 }, Span { start: s1, len: l1 });
                        let b = pack::pack_b(&op, yx.0, yx.1, &mut buf).map(|p| p.data().len());
                        assert_eq!(
                            b_block_len(&op, yx.0, yx.1, buf_len),
                            b,
                            "B {trans:?} {yx:?} {buf_len}"
                        );
                    }
                }
            }
        }
    }
    // The constructors accept exactly whole packed blocks.
    let buf = vec![0.0_f32; 64 * 3];
    assert!(PackedA::from_packed(&buf, 1, 3).is_some());
    assert!(PackedA::from_packed(&buf, 64, 3).is_some());
    assert!(PackedA::from_packed(&buf, 65, 3).is_none());
    assert!(PackedA::from_packed(&buf[..64 * 3 - 1], 64, 3).is_none());
    assert!(PackedA::from_packed(&buf[..0], 0, 3).is_none());
    assert!(PackedB::from_packed(&buf[..16 * 12], 12, 16).is_some());
    assert!(PackedB::from_packed(&buf[..16 * 12], 12, 1).is_some());
    assert!(PackedB::from_packed(&buf[..16 * 12], 12, 17).is_none());
    assert!(PackedB::from_packed(&buf[..0], 12, 0).is_none());
    let wide = vec![0.0_f64; 8 * 5];
    assert!(PackedB::from_packed(&wide, 5, 8).is_some());
    assert!(PackedB::from_packed(&wide, 5, 9).is_none());
}

/// The primitive in plain Rust, as a stand-in kernel: the value the
/// transposers must produce (module docs of `pack_za.rs`).
///
/// # Safety
///
/// `args` describes a readable block and a writable buffer as the kernels'
/// contract says.
unsafe fn rust_transpose<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    let (d, w, width) = (args.depth, args.cols, args.width);
    for c in 0..w.div_ceil(width) * width {
        for p in 0..d {
            let at = (c / width) * width * d + p * width + c % width;
            // SAFETY: `c < w`, `p < d` address the block; `at` is below
            // `ceil(w / width) · width · d`, the buffer's length.
            unsafe {
                let value = if c < w {
                    *args.src.add(p + c * args.ld)
                } else {
                    T::ZERO
                };
                *args.dst.add(at) = value;
            }
        }
    }
}

/// [`rust_transpose`] without the zero padding (stale contents survive).
///
/// # Safety
///
/// As for [`rust_transpose`].
unsafe fn rust_no_padding<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    let (d, w, width) = (args.depth, args.cols, args.width);
    for c in 0..w {
        for p in 0..d {
            // SAFETY: as in `rust_transpose`.
            unsafe {
                *args
                    .dst
                    .add((c / width) * width * d + p * width + c % width) =
                    *args.src.add(p + c * args.ld)
            };
        }
    }
}

/// [`rust_transpose`], then one more element written after the buffer.
///
/// # Safety
///
/// As for [`rust_transpose`], and the buffer must hold one element more.
unsafe fn rust_one_past<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    // SAFETY: as in `rust_transpose`; the self-check's buffers carry a guard
    // after the packed length, so the extra element is inside them.
    unsafe {
        rust_transpose(args);
        let len = args.cols.div_ceil(args.width) * args.width * args.depth;
        *args.dst.add(len) = T::ZERO;
    }
}

/// [`rust_transpose`] that drops the last depth step (the stale row stays).
///
/// # Safety
///
/// As for [`rust_transpose`].
unsafe fn rust_short_depth<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    let shorter = ZaPackArgs {
        depth: args.depth - 1,
        ..*args
    };
    if shorter.depth > 0 {
        // SAFETY: a sub-block of the caller's block; the shorter layout lies
        // inside the caller's buffer.
        unsafe { rust_transpose(&shorter) };
    }
}

/// [`rust_transpose`] with the first two columns swapped.
///
/// # Safety
///
/// As for [`rust_transpose`].
unsafe fn rust_swapped<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    // SAFETY: as in `rust_transpose`.
    unsafe {
        rust_transpose(args);
        if args.cols >= 2 {
            for p in 0..args.depth {
                let row = args.dst.add(p * args.width);
                core::ptr::swap(row, row.add(1));
            }
        }
    }
}

thread_local! {
    /// Calls of [`counted`] on this thread.
    static COUNTED: RefCell<usize> = const { RefCell::new(0) };
}

/// [`rust_transpose`], counted.
///
/// # Safety
///
/// As for [`rust_transpose`].
unsafe fn counted<T: Copy + SmeElement>(args: &ZaPackArgs<T>) {
    COUNTED.with(|calls| *calls.borrow_mut() += 1);
    // SAFETY: forwarded from this function's contract.
    unsafe { rust_transpose(args) }
}

#[test]
fn test_sme2_pack_za_transposer_runs_once_per_request() {
    // Each request of a ZA orientation enters the transposer exactly once
    // (one streaming region per pack call, not one per panel, and never
    // skipped in favour of the plain packer). Counted per thread, unlike
    // the process-wide kernel counter.
    fn sweep<T: SmePack>() {
        for case in [
            Case::b(17, 35, true),
            Case::a(9, 70, false),
            Case::b(1, 1, false),
            Case::a(65, 129, true),
            Case::b(257, 130, true),
        ] {
            let before = COUNTED.with(|calls| *calls.borrow());
            // SAFETY: a plain Rust stand-in under the kernels' contract.
            if let Err(why) = unsafe { run_case::<T>(&case, counted::<T>) } {
                panic!("{why}");
            }
            let calls = COUNTED.with(|calls| *calls.borrow()) - before;
            assert_eq!(calls, 1, "{case:?}: {calls} transposer calls");
        }
    }
    sweep::<f32>();
    sweep::<f64>();
}

#[test]
fn test_sme2_pack_za_self_check_stand_ins() {
    // The first-use check passes for the primitive in plain Rust and for the
    // interpreted templates, and fails for each deliberately wrong stand-in.
    fn sweep<T: SmePack>(interpreted: ZaKernelFn<T>) {
        // SAFETY: plain Rust stand-ins on the check's own buffers.
        unsafe {
            assert!(self_check_with::<T>(rust_transpose::<T>));
            assert!(!self_check_with::<T>(rust_no_padding::<T>));
            assert!(!self_check_with::<T>(rust_one_past::<T>));
            assert!(!self_check_with::<T>(rust_short_depth::<T>));
            assert!(!self_check_with::<T>(rust_swapped::<T>));
            assert!(!self_check_with::<T>(forbidden::<T>));
        }
        // The interpreter checks its accesses against a window: give it one
        // that admits the check's matrices and buffers (any heap address);
        // the comparison with the plain packer is the check itself.
        WINDOW.with(|slot| {
            *slot.borrow_mut() = Some(Window {
                src: 0,
                ld: usize::MAX,
                rows: usize::MAX,
                cols: 1,
                esize: core::mem::size_of::<T>(),
                dst: 0,
                dst_len: usize::MAX / 8,
            });
        });
        REPORT.with(|report| report.borrow_mut().clear());
        // SAFETY: interpreted.
        let passed = unsafe { self_check_with::<T>(interpreted) };
        WINDOW.with(|slot| *slot.borrow_mut() = None);
        let findings = REPORT.with(|report| core::mem::take(&mut *report.borrow_mut()));
        assert!(passed && findings.is_empty(), "{findings:?}");
    }
    sweep::<f32>(interpreted_f32);
    sweep::<f64>(interpreted_f64);
    // The fixed blocks cover full panels and groups, partial panels, depth
    // tails of both chunk heights, one element and several groups.
    for a_side in [false, true] {
        let blocks: Vec<_> = CHECK_BLOCKS.iter().filter(|b| b.a_side == a_side).collect();
        assert!(
            blocks
                .iter()
                .any(|b| b.depth.is_multiple_of(16) && b.width.is_multiple_of(64))
        );
        assert!(
            blocks
                .iter()
                .any(|b| !b.depth.is_multiple_of(8) && !b.width.is_multiple_of(16))
        );
        assert!(blocks.iter().any(|b| b.depth == 1 && b.width == 1));
        assert!(blocks.iter().any(|b| b.width > 128));
        assert!(blocks.iter().any(|b| b.ld_pad > 0));
        assert!(
            blocks
                .iter()
                .any(|b| b.depth > 2 * b.width && !b.depth.is_multiple_of(8))
        );
    }
}
