//! Tests of the FPSR frame rules, the kernel-entry checks (declared `const`
//! placeholders, extensions) and the FPSR accessor rule of `lint.rs`. Every
//! rule is shown rejecting a template that breaks it.

use super::tests_rules::{GOOD_STREAMING, assert_violates, violations, with_body_line};
use super::*;
use crate::level3::sme2::test_support::FPSR_TEMPLATES;

/// A registry entry for a template, with the given `const` operands.
fn entry(template: &str, consts: &'static [&'static str], f64f64: bool) -> KernelEntry {
    KernelEntry {
        name: "test_kernel",
        file: "crates/oxiblas-blas/src/level3/sme2/tests_lint_frame.rs",
        template: Box::leak(template.to_owned().into_boxed_str()),
        consts,
        f64f64,
    }
}

/// The rules a kernel entry violates.
fn kernel_violations(entry: &KernelEntry) -> Vec<Rule> {
    match lint_kernel(entry) {
        Ok(()) => Vec::new(),
        Err(errors) => errors.iter().map(|e| e.rule).collect(),
    }
}

/// The placeholders of [`GOOD_STREAMING`].
const GOOD_CONSTS: &[&str] = &["off_a", "off_b", "off_k", "off_alpha", "off_rows"];

/// The frame lines in the order the macro emits them.
const FRAME_LINES: [&str; 5] = [
    "mrs x17, fpsr",
    "msr fpsr, xzr",
    "mrs x16, fpsr",
    "orr x17, x17, x16",
    "msr fpsr, x17",
];

#[test]
fn test_sme2_lint_frame_of_the_macro_is_accepted() {
    assert_eq!(violations(GOOD_STREAMING), []);
    assert_eq!(
        kernel_violations(&entry(GOOD_STREAMING, GOOD_CONSTS, true)),
        []
    );
    // The frame as `asm.rs` documents it.
    let lines: Vec<&str> = GOOD_STREAMING.lines().collect();
    assert_eq!(
        lines[..3],
        [
            ".arch_extension sme2",
            ".arch_extension sme-f64f64",
            FRAME_LINES[0]
        ]
    );
    let start = lines.iter().position(|l| *l == "smstart").unwrap_or(0);
    assert_eq!(lines[start + 1], FRAME_LINES[1]);
    assert_eq!(
        lines[lines.len() - 5..],
        [
            "90:",
            FRAME_LINES[2],
            "smstop",
            FRAME_LINES[3],
            FRAME_LINES[4]
        ]
    );
    for line in FRAME_LINES {
        let count = lines.iter().filter(|l| **l == line).count();
        assert_eq!(count, 1, "{line}");
    }
}

#[test]
fn test_sme2_lint_frame_lines_are_required() {
    for line in FRAME_LINES {
        let without = GOOD_STREAMING.replacen(&format!("{line}\n"), "", 1);
        let without = if without == GOOD_STREAMING {
            // The last line has no line break after it.
            GOOD_STREAMING.replacen(&format!("\n{line}"), "", 1)
        } else {
            without
        };
        assert_ne!(without, GOOD_STREAMING, "{line} not found");
        assert_violates(&without, Rule::FpsrFrame);
    }
}

#[test]
fn test_sme2_lint_frame_lines_must_be_in_place() {
    // The save after a prologue load instead of first.
    let late_save = GOOD_STREAMING.replacen("mrs x17, fpsr\n", "", 1).replacen(
        "prfm pldl2keep, [x2]\n",
        "prfm pldl2keep, [x2]\nmrs x17, fpsr\n",
        1,
    );
    assert_violates(&late_save, Rule::FpsrFrame);
    // The clear after another body instruction instead of right after smstart.
    let late_clear = GOOD_STREAMING.replacen("msr fpsr, xzr\n", "", 1).replacen(
        "ptrue p0.s\n",
        "ptrue p0.s\nmsr fpsr, xzr\n",
        1,
    );
    assert_violates(&late_clear, Rule::FpsrFrame);
    assert_violates(&late_clear, Rule::RejectedMnemonic);
    // The read before the exit label instead of right before smstop.
    let early_read = GOOD_STREAMING.replacen("90:\nmrs x16, fpsr\n", "mrs x16, fpsr\n90:\n", 1);
    assert_violates(&early_read, Rule::FpsrFrame);
    // The two lines after smstop swapped.
    let swapped = GOOD_STREAMING.replacen(
        "orr x17, x17, x16\nmsr fpsr, x17",
        "msr fpsr, x17\norr x17, x17, x16",
        1,
    );
    assert_violates(&swapped, Rule::FpsrFrame);
    // Other operands in the frame's place.
    for (from, to) in [
        ("mrs x17, fpsr", "mrs x17, fpcr"),
        ("msr fpsr, xzr", "msr fpsr, x1"),
        ("mrs x16, fpsr", "mrs x15, fpsr"),
        ("orr x17, x17, x16", "orr x17, x17, x15"),
    ] {
        assert_violates(&GOOD_STREAMING.replacen(from, to, 1), Rule::FpsrFrame);
    }
    let wrong_restore = GOOD_STREAMING.replacen("msr fpsr, x17", "msr fpcr, x17", 1);
    assert_violates(&wrong_restore, Rule::FpsrFrame);
}

#[test]
fn test_sme2_lint_rejects_x17_outside_the_frame() {
    for line in [
        "mov x17, #0",
        "add x1, x1, x17",
        "ldr w17, [x1]",
        "str x17, [x1]",
        "cbz x17, 90f",
    ] {
        assert_violates(&with_body_line(line), Rule::X17);
    }
    let prologue = GOOD_STREAMING.replacen("smstart\n", "mov x17, x1\nsmstart\n", 1);
    assert_violates(&prologue, Rule::X17);
    // x16 is free inside the body.
    assert_eq!(violations(&with_body_line("add x16, x16, #1")), []);
}

#[test]
fn test_sme2_lint_only_the_frame_follows_smstop() {
    for (line, rule) in [
        ("nop", Rule::AfterSmstop),
        ("mov x1, x2", Rule::AfterSmstop),
        ("msr fpsr, x17", Rule::RejectedMnemonic),
        ("20:", Rule::AfterSmstop),
    ] {
        let appended = format!("{GOOD_STREAMING}\n{line}");
        assert_violates(&appended, rule);
        let between = GOOD_STREAMING.replacen("smstop\n", &format!("smstop\n{line}\n"), 1);
        assert_violates(&between, Rule::FpsrFrame);
    }
}

#[test]
fn test_sme2_lint_rejects_other_fpsr_and_system_register_access() {
    for line in [
        "msr fpsr, xzr",
        "mrs x16, fpsr",
        "mrs x1, fpsr",
        "msr fpcr, x1",
        "mrs x1, fpcr",
        "msr svcr, x1",
    ] {
        assert_violates(&with_body_line(line), Rule::RejectedMnemonic);
    }
}

#[test]
fn test_sme2_lint_exit_label_belongs_to_the_frame() {
    // A second `90:` in the body.
    assert_violates(&with_body_line("90:"), Rule::ExitLabel);
    // No exit label at all (the body's `cbz x5, 90f` then fails too).
    let without = GOOD_STREAMING.replacen("90:\n", "", 1);
    assert_violates(&without, Rule::ExitLabel);
    assert_violates(&without, Rule::BranchTarget);
}

#[test]
fn test_sme2_lint_kernel_placeholders_must_be_declared_consts() {
    let good = entry(GOOD_STREAMING, GOOD_CONSTS, true);
    assert_eq!(kernel_violations(&good), []);
    // One placeholder fewer declared: `{off_rows}` is not a const operand.
    let missing = entry(
        GOOD_STREAMING,
        &["off_a", "off_b", "off_k", "off_alpha"],
        true,
    );
    assert_eq!(kernel_violations(&missing), [Rule::UndeclaredPlaceholder]);
    // Operands of a forbidden class: a `reg`-style name, a positional
    // operand, a modifier on a const, a vector-register modifier.
    for line in [
        "add x1, x1, {tmp}",
        "add x1, x1, {0}",
        "add x1, x1, {}",
        "ldr x1, [x0, #{off_a:w}]",
        "mov {off_a:x}, x1",
    ] {
        let template = with_body_line(line);
        let found = kernel_violations(&entry(&template, GOOD_CONSTS, true));
        assert!(
            found.contains(&Rule::UndeclaredPlaceholder),
            "{line}: {found:?}"
        );
    }
    let vector = with_body_line("fmov {v:d}, x1");
    let found = kernel_violations(&entry(&vector, GOOD_CONSTS, true));
    assert!(found.contains(&Rule::UndeclaredPlaceholder), "{found:?}");
    assert!(found.contains(&Rule::PlaceholderModifier), "{found:?}");
    // Literal braces are not placeholders.
    assert_eq!(
        placeholders("zero {{za}}\nldr x1, [x0, #{off_a}]"),
        [(2, "off_a".to_owned())]
    );
}

#[test]
fn test_sme2_lint_kernel_placeholders_are_bare_immediates() {
    // A const glued to a register prefix prints a register name: `x{n}`
    // with n = 17 is `x17`, `v{n}.16b` an AdvSIMD register, `z{n}.d` inside
    // a memory operand a gather.
    for line in [
        "mov x{off_a}, #0",
        "mov w{off_a}, #0",
        "mov v{off_a}.16b, v{off_a}.16b",
        "ldr q{off_a}, [x1]",
        "ld1d {{z0.d}}, p0/z, [x1, z{off_a}.d, lsl #3]",
        "add x1, x1, #{off_a}{off_b}",
        "add x1, x1, #{off_a}0",
    ] {
        let template = with_body_line(line);
        let found = kernel_violations(&entry(&template, GOOD_CONSTS, true));
        assert!(found.contains(&Rule::GluedPlaceholder), "{line}: {found:?}");
        // The same rule holds for every template kind.
        assert!(
            violations(&template).contains(&Rule::GluedPlaceholder),
            "{line}"
        );
    }
    // A kernel's consts are immediates: `#{off}` / `#-{off}` only.
    for line in ["add x1, x1, {off_a}", "ldr x1, [x0, {off_a}]"] {
        let template = with_body_line(line);
        let found = kernel_violations(&entry(&template, GOOD_CONSTS, true));
        assert!(
            found.contains(&Rule::PlaceholderNotImmediate),
            "{line}: {found:?}"
        );
    }
    for line in [
        "ldr x1, [x0, # {off_a}]",
        "add x1, x1, #-{off_a}",
        "sub x1, x1, #{off_b}",
    ] {
        let template = with_body_line(line);
        let found = kernel_violations(&entry(&template, GOOD_CONSTS, true));
        assert_eq!(found, [], "{line}");
    }
    // Register placeholders of the probe and the FPSR accessors are fine.
    for template in FPSR_TEMPLATES {
        assert_eq!(lint_fpsr_template(template), Ok(()), "{template}");
    }
}

#[test]
fn test_sme2_lint_kernel_rejects_hidden_statements() {
    // `\r` separates statements for the assembler on every target and `%%`
    // on Mach-O: a statement hidden behind either must fail the kernel lint.
    for line in [
        "nop\rsmstop",
        "nop\r.inst 0xd503467f",
        "nop\rret",
        "ptrue p0.s %% smstop",
        "nop %% .word 0xd503467f",
    ] {
        let template = with_body_line(line);
        let found = kernel_violations(&entry(&template, GOOD_CONSTS, true));
        assert!(
            found.contains(&Rule::ForbiddenCharacter),
            "{line:?}: {found:?}"
        );
    }
    let prologue = GOOD_STREAMING.replacen("prfm pldl2keep, [x2]", "nop\rsmstart", 1);
    let found = kernel_violations(&entry(&prologue, GOOD_CONSTS, true));
    assert!(found.contains(&Rule::ForbiddenCharacter), "{found:?}");
}

#[test]
fn test_sme2_lint_kernel_never_stores_to_the_argument_block() {
    for line in [
        "str x1, [x0, #8]",
        "stp x1, x2, [x0]",
        "stur w1, [x0, #-4]",
        "st1w {{z0.s}}, p0, [x0]",
        "str za[w12, 0], [x0]",
    ] {
        assert_violates(&with_body_line(line), Rule::StoreToArgs);
    }
    // Loads from the block and stores elsewhere are fine.
    for line in [
        "ldr x1, [x0, #8]",
        "str x1, [x6, #8]",
        "st1w {{z0.s}}, p0, [x6, x0, lsl #2]",
    ] {
        let found = violations(&with_body_line(line));
        assert!(!found.contains(&Rule::StoreToArgs), "{line}: {found:?}");
    }
}

#[test]
fn test_sme2_lint_kernel_extensions_match_the_entry() {
    let f64_template = GOOD_STREAMING;
    let f32_template = GOOD_STREAMING.replacen(".arch_extension sme-f64f64\n", "", 1);
    assert_eq!(
        kernel_violations(&entry(&f32_template, GOOD_CONSTS, false)),
        []
    );
    assert_eq!(
        kernel_violations(&entry(f64_template, GOOD_CONSTS, false)),
        [Rule::ExtensionMismatch]
    );
    assert_eq!(
        kernel_violations(&entry(&f32_template, GOOD_CONSTS, true)),
        [Rule::ExtensionMismatch]
    );
}

#[test]
fn test_sme2_lint_fpsr_accessors() {
    for template in FPSR_TEMPLATES {
        if let Err(errors) = lint_fpsr_template(template) {
            panic!("{template}: {}", report(&errors));
        }
    }
    let fpsr_violations = |template: &str| -> Vec<Rule> {
        match lint_fpsr_template(template) {
            Ok(()) => Vec::new(),
            Err(errors) => errors.iter().map(|e| e.rule).collect(),
        }
    };
    for template in [
        "mrs {x}, svcr",
        "msr fpcr, {x}",
        "msr fpsr, xzr",
        "mrs {x}, fpsr\nmsr fpsr, {x}",
        ".arch_extension sme\nmrs {x}, fpsr",
        "smstart",
        "",
    ] {
        let found = fpsr_violations(template);
        assert!(found.contains(&Rule::FpsrAccess), "{template:?}: {found:?}");
    }
}
