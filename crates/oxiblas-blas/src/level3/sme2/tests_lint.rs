//! Tests of the template rules of `lint.rs`: the allowlist and reject list,
//! registers, the region structure, labels, portability and the probe. The
//! FPSR frame, the kernel-entry checks and the FPSR accessors are tested in
//! `tests_lint_frame.rs`, the registry walk and the source scan in
//! `tests_registry.rs`.

use super::*;
use crate::level3::sme2::asm::sme_template;

/// A template in the shape of the planned macro-kernels, using every
/// instruction family the kernels need, wrapped in the frame of `asm.rs` by
/// the macro rule the kernels use (so these tests check that very frame).
pub(super) const GOOD_STREAMING: &str = sme_template!(
    [f64f64],
    [
        "ldr x1, [x0, #{off_a}]",
        "ldr x2, [x0, #{off_b}]",
        "ldr x3, [x0, #{off_k}]",
        "ldr w20, [x0, #{off_alpha}]",
        "ldp x4, x5, [x0, #{off_rows}]",
        "prfm pldl2keep, [x2]"
    ],
    [
        "ptrue p0.s",
        "ptrue pn8.s",
        "zero {{za}}",
        "dup z28.s, w20",
        "mov w8, #0",
        "mov w12, #0",
        "20:",
        "ld1w {{z0.s-z3.s}}, pn8/z, [x1]",
        "ld1w {{z8.s}}, p0/z, [x2]",
        "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
        "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
        "addvl x1, x1, #4",
        "addvl x2, x2, #1",
        "subs x3, x3, #1",
        "b.ne 20b",
        "whilelt pn9.s, x4, x5, vlx4",
        "whilelt p1.s, x4, x5",
        "cbz x5, 90f",
        "50: mova {{z16.s-z19.s}}, za0h.s[w12, 0:3]",
        "fmul z16.s, z16.s, z28.s",
        "ld1w {{z20.s}}, p1/z, [x6, x9, lsl #2]",
        "fmla z16.s, p1/m, z20.s, z29.s",
        "st1w {{z16.s-z19.s}}, pn9, [x6]",
        "fadd za.s[w8, 0, vgx4], {{z16.s-z19.s}}",
        "fmla za.s[w8, 7, vgx4], {{z0.s-z3.s}}, z4.s[3]",
        "ld1rqw {{z4.s}}, p0/z, [x7]",
        "dup z5.q, z4.q[0]",
        "mov x16, x9",
        "st1w {{za0h.s[w12, 0]}}, p1, [x6, x16, lsl #2]",
        "ldr za[w12, 0], [x6]",
        "add w12, w12, #4",
        "cmp w12, #16",
        "b.lo 50b"
    ]
);

/// The rules a streaming template violates.
pub(super) fn violations(template: &str) -> Vec<Rule> {
    match lint_streaming_template(template) {
        Ok(()) => Vec::new(),
        Err(errors) => errors.iter().map(|e| e.rule).collect(),
    }
}

/// Asserts that a streaming template violates `rule`.
#[track_caller]
pub(super) fn assert_violates(template: &str, rule: Rule) {
    let found = violations(template);
    assert!(
        found.contains(&rule),
        "expected {rule:?}, found {found:?} in\n{template}"
    );
}

/// The good template with `line` added at the end of the streaming body.
pub(super) fn with_body_line(line: &str) -> String {
    GOOD_STREAMING.replace("90:\n", &format!("{line}\n90:\n"))
}

/// A minimal probe-shaped template around one extra line.
fn probe_with(line: &str) -> String {
    format!(".arch_extension sme\nmrs {{svcr}}, svcr\n{line}\nrdsvl {{svl}}, #1")
}

#[test]
fn test_sme2_lint_accepts_kernel_shaped_template() {
    if let Err(errors) = lint_streaming_template(GOOD_STREAMING) {
        panic!("{}", report(&errors));
    }
}

#[test]
fn test_sme2_lint_accepts_spike_verified_instructions() {
    // Each form assembled with rustc 1.89.0 and 1.95.0 for
    // aarch64-apple-darwin and aarch64-unknown-linux-gnu (toolchain spike).
    let forms = [
        "fmopa za3.s, p0/m, p1/m, z0.s, z1.s",
        "fmopa za7.d, p0/m, p0/m, z0.d, z1.d",
        "fmla za.s[w8, 0, vgx4], {{z0.s-z3.s}}, z15.s[0]",
        "fmla za.s[w8, 0, vgx4], {{z0.s-z3.s}}, z15.s",
        "fmla za.d[w8, 7, vgx4], {{z0.d-z3.d}}, z4.d[1]",
        "fadd za.d[w8, 0, vgx4], {{z16.d-z19.d}}",
        "mova za0h.s[w12, 0:3], {{z0.s-z3.s}}",
        "mova {{z0.s-z3.s}}, za3v.s[w13, 0:3]",
        "mova za.s[w8, 0, vgx4], {{z0.s-z3.s}}",
        "mova {{z4.d-z7.d}}, za.d[w8, 7, vgx4]",
        "ld1rqd {{z0.d}}, p0/z, [x0]",
        "ld1rw {{z0.s}}, p0/z, [x0]",
        "dup z28.d, x20",
        "whilelt pn9.d, x1, x2, vlx4",
        "ld1w {{z0.s-z3.s}}, pn9/z, [x0, x1, lsl #2]",
        "ld1w {{z0.s, z4.s, z8.s, z12.s}}, pn8/z, [x0]",
        "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn8/z, [x0]",
        "st1d {{za0h.d[w12, 0]}}, p1, [x6]",
        "ld1w {{za0h.s[w12, 0]}}, p1/z, [x0, x9, lsl #2]",
        "zero {{za0.s, za1.s}}",
        "zero {{za0.d, za7.d}}",
        "fmul z16.s, z16.s, z7.s[0]",
        "fadd z16.s, z16.s, z17.s",
        "cntw x1",
        "incw x1",
        "cntp x1, pn9.s, vlx4",
        "pext p1.s, pn9[0]",
        "str za[w12, 15], [x6, #15, mul vl]",
        "movprfx z16, z17",
        "sel z20.s, p0, z21.s, z22.s",
        "fmad z16.s, p0/m, z17.s, z18.s",
        "rdsvl x10, #1",
    ];
    for form in forms {
        if let Err(errors) = lint_streaming_template(&with_body_line(form)) {
            panic!("{form}: {}", report(&errors));
        }
    }
}

#[test]
fn test_sme2_lint_allowlist_and_reject_list_are_disjoint() {
    for family in FAMILIES {
        for mnemonic in *family {
            assert_eq!(
                rejection(mnemonic),
                None,
                "`{mnemonic}` is both allowed and rejected"
            );
        }
    }
    // Every reject-list entry is actually rejected by the checker.
    for (_, names) in REJECTED {
        for mnemonic in *names {
            assert_violates(&with_body_line(mnemonic), Rule::RejectedMnemonic);
        }
    }
    for (_, prefixes) in REJECTED_PREFIXES {
        for prefix in *prefixes {
            let mnemonic = format!("{prefix}w");
            assert_violates(&with_body_line(&mnemonic), Rule::RejectedMnemonic);
        }
    }
}

#[test]
fn test_sme2_lint_normalises_braces_and_placeholders() {
    let mut found = Findings::default();
    let normalised = normalise(
        "zero {{za}}\nldr x1, [x0, #{off}]\nmov {t:w}, #1\nadd {0}, {}, {t:x}\nld1w {{z0.s-z3.s}}, pn8/z, [{p}]",
        &mut found,
    );
    assert!(found.0.is_empty(), "{}", report(&found.0));
    assert_eq!(
        normalised,
        "zero {za}\nldr x1, [x0, #operand]\nmov operand, #1\nadd operand, operand, operand\nld1w {z0.s-z3.s}, pn8/z, [operand]"
    );

    // FP/SIMD modifiers are rejected; `w` and `x` are fine.
    for modifier in ["b", "h", "s", "d", "q", "v"] {
        assert_violates(
            &with_body_line(&format!("mov {{tmp:{modifier}}}, x1")),
            Rule::PlaceholderModifier,
        );
    }
    // A register list whose braces were not doubled, a lone `}` and an
    // unclosed `{`.
    assert_violates(
        &with_body_line("ld1w {z0.s-z3.s}, pn8/z, [x1]"),
        Rule::MalformedBraces,
    );
    assert_violates(&with_body_line("add x1, x1, #1 }"), Rule::MalformedBraces);
    assert_violates(&with_body_line("add {x1, x1, #1"), Rule::MalformedBraces);
}

#[test]
fn test_sme2_lint_rejects_advsimd_and_ffr_registers() {
    for line in [
        "fmla v0.4s, v1.4s, v2.4s",
        "ldr q0, [x1]",
        "str d8, [x1]",
        "ldr s3, [x1]",
        "ldr h1, [x2]",
        "ldr b2, [x2]",
        "mov v1.16b, v2.16b",
    ] {
        assert_violates(&with_body_line(line), Rule::AdvSimdOrFfrRegister);
    }
    // Scalar FP mnemonics are not on the allowlist either.
    assert_violates(&with_body_line("fmov d0, x1"), Rule::UnknownMnemonic);
    assert_violates(
        &with_body_line("ld1w {{z0.s}}, p0/z, [x1]\nwrffr p0.b"),
        Rule::RejectedMnemonic,
    );
    assert_violates(&with_body_line("mov p1.b, ffr"), Rule::AdvSimdOrFfrRegister);
}

#[test]
fn test_sme2_lint_rejects_gathers_scatters_and_illegal_sve() {
    for line in [
        "ld1w {{z0.s}}, p0/z, [x1, z1.s, uxtw #2]",
        "ld1d {{z0.d}}, p0/z, [z1.d]",
        "st1w {{z0.s}}, p0, [x1, z2.s, sxtw]",
        "ldnt1w {{z0.s}}, p0/z, [z1.s, x2]",
    ] {
        assert_violates(&with_body_line(line), Rule::VectorAddressing);
    }
    for line in [
        "ldff1w {{z0.s}}, p0/z, [x1]",
        "ldnf1w {{z0.s}}, p0/z, [x1]",
        "ld1row {{z0.s}}, p0/z, [x1]",
        "rdffr p0.b",
        "setffr",
        "fadda s0, p0, s0, z1.s",
        "fexpa z0.s, z1.s",
        "ftmad z0.s, z0.s, z1.s, #0",
        "ftssel z0.s, z1.s, z2.s",
        "compact z0.s, p0, z1.s",
        "adr z0.s, [z1.s, z2.s]",
        "histcnt z0.s, p0/z, z1.s, z2.s",
        "histseg z0.b, z1.b, z2.b",
        "match p0.b, p1/z, z0.b, z1.b",
        "nmatch p0.b, p1/z, z0.b, z1.b",
        "aese z0.b, z0.b, z1.b",
        "sm4e z0.s, z0.s, z1.s",
        "rax1 z0.d, z1.d, z2.d",
        "pmullb z0.q, z1.d, z2.d",
    ] {
        assert_violates(&with_body_line(line), Rule::RejectedMnemonic);
    }
    // Not listed anywhere: unknown until someone adds it with a reason.
    assert_violates(
        &with_body_line("fmax z0.s, p0/m, z0.s, z1.s"),
        Rule::UnknownMnemonic,
    );
}

#[test]
fn test_sme2_lint_rejects_calls_returns_and_system_registers() {
    for line in [
        "bl 90f",
        "blr x1",
        "br x1",
        "ret",
        "retaa",
        "svc #0",
        "brk #0",
        "msr svcr, x1",
        "mrs x1, svcr",
    ] {
        assert_violates(&with_body_line(line), Rule::RejectedMnemonic);
    }
}

#[test]
fn test_sme2_lint_rejects_reserved_registers() {
    for line in [
        "mov x18, #0",
        "mov w19, #0",
        "ldr x1, [x29]",
        "mov x30, x1",
        "mov lr, x1",
        "mov fp, x1",
        "add sp, sp, #16",
        "ldr x1, [sp, #8]",
    ] {
        assert_violates(&with_body_line(line), Rule::ReservedRegister);
    }
}

#[test]
fn test_sme2_lint_requires_one_streaming_region() {
    let no_start = GOOD_STREAMING.replace("smstart\n", "");
    assert_violates(&no_start, Rule::SmstartSmstop);
    let no_stop = GOOD_STREAMING.replace("\nsmstop\n", "\nnop\n");
    assert_violates(&no_stop, Rule::SmstartSmstop);
    assert_violates(&with_body_line("smstop\nsmstart"), Rule::SmstartSmstop);
    let partial = GOOD_STREAMING.replace("smstart\n", "smstart sm\n");
    assert_violates(&partial, Rule::SmstartSmstop);
    let partial_stop = GOOD_STREAMING.replace("\nsmstop\n", "\nsmstop za\n");
    assert_violates(&partial_stop, Rule::SmstartSmstop);
    let reversed = concat!(".arch_extension sme2\n", "smstop\n", "nop\n", "smstart");
    assert_violates(reversed, Rule::SmstartSmstop);

    // Nothing but the two frame lines may follow smstop.
    assert_violates(&format!("{GOOD_STREAMING}\nnop"), Rule::AfterSmstop);
    assert_violates(&format!("{GOOD_STREAMING}\n30:"), Rule::AfterSmstop);
}

#[test]
fn test_sme2_lint_rejects_non_base_prologue() {
    for line in [
        "ptrue p0.s",
        "ldr z0, [x0]",
        "dup z28.s, w20",
        "zero {{za}}",
        "cbz x3, 90f",
        "b 90f",
        "mova za0h.s[w12, 0:3], {{z0.s-z3.s}}",
    ] {
        let template = GOOD_STREAMING.replace("smstart\n", &format!("{line}\nsmstart\n"));
        assert_violates(&template, Rule::Prologue);
    }
    let label = GOOD_STREAMING.replace("smstart\n", "30:\nsmstart\n");
    assert_violates(&label, Rule::LabelOutsideRegion);
}

#[test]
fn test_sme2_lint_checks_labels_and_branch_targets() {
    // Labels made only of 0 and 1, as definitions and as targets.
    let binary = GOOD_STREAMING
        .replace("20:\n", "10:\n")
        .replace("b.ne 20b", "b.ne 10b");
    assert_violates(&binary, Rule::BinaryLookingLabel);
    assert_violates(&with_body_line("101:"), Rule::BinaryLookingLabel);
    // Named labels and targets.
    assert_violates(&with_body_line("loop:"), Rule::NamedLabel);
    assert_violates(&with_body_line("b some_symbol"), Rule::BranchTarget);
    // Targets that do not resolve in their direction.
    assert_violates(&with_body_line("b.ne 30b"), Rule::BranchTarget);
    assert_violates(&with_body_line("b 20f"), Rule::BranchTarget);
    assert_violates(&with_body_line("cbnz x1, 50f"), Rule::BranchTarget);
    // Unknown condition codes.
    assert_violates(&with_body_line("b.xx 20b"), Rule::UnknownMnemonic);
    // SVE condition aliases are fine.
    assert!(violations(&with_body_line("b.first 20b")).is_empty());
    assert!(violations(&with_body_line("b.none 90f")).is_empty());
}

#[test]
fn test_sme2_lint_rejects_separators_comments_and_directives() {
    assert_violates(
        &with_body_line("add x1, x1, #1; add x2, x2, #1"),
        Rule::Semicolon,
    );
    assert_violates(&with_body_line("add x1, x1, #1 // note"), Rule::Comment);
    assert_violates(&with_body_line("/* note */ nop"), Rule::Comment);
    assert_violates(&with_body_line("# note"), Rule::Comment);
    for directive in [
        ".inst 0xd503477f",
        ".word 0",
        ".byte 0",
        ".arch_extension sve2",
    ] {
        let template = GOOD_STREAMING.replace(
            ".arch_extension sme2\n",
            &format!(".arch_extension sme2\n{directive}\n"),
        );
        assert_violates(&template, Rule::ForbiddenDirective);
    }
    // Directives must lead, and a streaming template needs sme2.
    assert_violates(
        &with_body_line(".arch_extension sme2"),
        Rule::DirectiveNotLeading,
    );
    let without = GOOD_STREAMING.replace(".arch_extension sme2\n", "");
    assert_violates(&without, Rule::MissingArchExtension);
    let sme_only = GOOD_STREAMING.replace(".arch_extension sme2\n", ".arch_extension sme\n");
    assert_violates(&sme_only, Rule::MissingArchExtension);
}

#[test]
fn test_sme2_lint_probe_template() {
    // The real probe of detect.rs.
    if let Err(errors) = lint_probe_template(crate::level3::sme2::detect::PROBE_TEMPLATE) {
        panic!("{}", report(&errors));
    }
    let probe_violations = |template: &str| -> Vec<Rule> {
        match lint_probe_template(template) {
            Ok(()) => Vec::new(),
            Err(errors) => errors.iter().map(|e| e.rule).collect(),
        }
    };
    assert!(probe_violations(&probe_with("mrs {tpidr2}, tpidr2_el0")).is_empty());
    for line in [
        "smstart",
        "smstop",
        "mrs x1, tpidr_el0",
        "msr svcr, x1",
        "fmov d0, x1",
        "zero {{za}}",
        "20:",
    ] {
        let found = probe_violations(&probe_with(line));
        assert!(
            found.contains(&Rule::ProbeInstruction),
            "{line}: expected ProbeInstruction, found {found:?}"
        );
    }
    let found = probe_violations("mrs {svcr}, svcr\nrdsvl {svl}, #1");
    assert!(found.contains(&Rule::MissingArchExtension), "{found:?}");
    let found = probe_violations(&probe_with("mrs {v:d}, svcr"));
    assert!(found.contains(&Rule::PlaceholderModifier), "{found:?}");
}

#[test]
fn test_sme2_lint_reports_lines_and_text() {
    let template = with_body_line("ret");
    let errors = match lint_streaming_template(&template) {
        Ok(()) => Vec::new(),
        Err(errors) => errors,
    };
    let line = template
        .lines()
        .position(|l| l == "ret")
        .map(|index| index + 1);
    assert_eq!(errors.len(), 1, "{}", report(&errors));
    assert_eq!(Some(errors[0].line), line);
    assert!(errors[0].text.contains("`ret`"), "{}", errors[0].text);
    assert!(report(&errors).contains("RejectedMnemonic"));
}

#[test]
fn test_sme2_lint_rejects_characters_outside_the_template_alphabet() {
    for line in [
        "nop\rsmstop",
        "nop\tnop",
        "nop\u{b}smstop",
        "nop\u{c}smstop",
        "nop\u{0}",
        "nop %% smstop",
        "mov x1, #1 @ x",
        "nop\u{2028}smstop",
        "add x1, x1, #1 $",
        "ldr x1, =0",
        "mov x1, x2 | x3",
    ] {
        assert_violates(&with_body_line(line), Rule::ForbiddenCharacter);
    }
    // The alphabet covers every form the backend writes.
    assert_eq!(violations(GOOD_STREAMING), []);
    assert_eq!(
        violations(&with_body_line("str za[w12, 15], [x6, #15, mul vl]")),
        []
    );
}
