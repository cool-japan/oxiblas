//! Site tests for the assembly of `detect.rs` (the register probe) and
//! `fpstate.rs` (the FPCR and FPSR accessors), and the template tests of
//! the FPCR / FPSR accessors.
//!
//! `detect.rs` is in `lint_scan::STREAMING_TEXT_FILES`, so the source scan
//! accepts any streaming text in it, and the template lint sees the probe
//! only through `detect::PROBE_TEMPLATE`: a literal template written out at
//! the site would escape both. The same holds for `fpstate.rs` and its
//! template constants (the scan only counts its sites). So each site must
//! take its text from its template macro and nothing else: the first
//! argument of the invocation is the macro call, no string literal appears
//! anywhere in the arguments, and every later argument is an operand or an
//! option, not another template piece.

use super::fpstate::{FPCR_TEMPLATES, FPSR_TEMPLATES};
use super::lint::{Rule, lint_fpsr_template, report};
use super::lint_scan::{lex, macro_calls};

/// The top-level arguments of every `asm!` invocation of `source` (lexed:
/// comments removed, string literals empty, so a literal reads as `""`).
fn asm_arguments(source: &str) -> Vec<Vec<String>> {
    let lexed = lex(source);
    let code = lexed.code.as_str();
    macro_calls(code, "asm")
        .into_iter()
        .map(|after_bang| {
            let rest = code[after_bang..].trim_start();
            let Some(inner) = rest.strip_prefix('(') else {
                return vec![format!("<not a parenthesised invocation: {rest:.20}>")];
            };
            let mut args = Vec::new();
            let mut depth = 0_usize;
            let mut current = String::new();
            for ch in inner.chars() {
                match ch {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' if depth == 0 => break,
                    ')' | ']' | '}' => depth -= 1,
                    ',' if depth == 0 => {
                        args.push(current.trim().to_owned());
                        current.clear();
                        continue;
                    }
                    _ => {}
                }
                current.push(ch);
            }
            if !current.trim().is_empty() {
                args.push(current.trim().to_owned());
            }
            args
        })
        .collect()
}

/// Why the arguments of one invocation do not take exactly `template` (one
/// of the allowed template macro calls), or `None` when they do.
fn site_problem(args: &[String], templates: &[&str]) -> Option<String> {
    let first = args.first().map_or("", String::as_str);
    if !templates.contains(&first) {
        return Some(format!(
            "the template is {first:?}, not one of {templates:?}"
        ));
    }
    if args.iter().any(|arg| arg.contains("\"\"")) {
        return Some(format!("a string literal among the arguments: {args:?}"));
    }
    // Every later argument is a named operand (`name = …`) or an option.
    let piece = args[1..].iter().find(|arg| {
        let named = arg
            .split_once('=')
            .is_some_and(|(name, _)| name.trim().chars().all(|c| c.is_alphanumeric() || c == '_'));
        !(named || arg.starts_with("options(") || arg.starts_with("clobber_abi("))
    });
    piece.map(|piece| format!("another template piece: {piece:?}"))
}

/// The templates each site of `source` takes, sorted, after checking every
/// site with [`site_problem`] (panics with the problem).
fn site_templates(file: &str, source: &str, templates: &[&str]) -> Vec<String> {
    let mut seen = Vec::new();
    for args in asm_arguments(source) {
        if let Some(problem) = site_problem(&args, templates) {
            panic!("{file}: an asm! site does not take exactly its template: {problem}");
        }
        seen.push(args[0].clone());
    }
    seen.sort_unstable();
    seen
}

#[test]
fn test_sme2_detect_probe_site_uses_the_template() {
    let templates = ["probe_template!()"];
    assert_eq!(
        site_templates("detect.rs", include_str!("detect.rs"), &templates),
        ["probe_template!()"],
        "detect.rs: the register probe is its only assembly site"
    );
    // The check catches a literal template, a literal piece after the
    // macro, a template built with `concat!`, and a second macro piece.
    let site = |args: &str| format!("fn f() {{ unsafe {{ core::arch::asm!({args}) }} }}\n");
    for (broken, needle) in [
        (
            "\"mrs {svcr}, svcr\", svcr = out(reg) svcr",
            "the template is",
        ),
        (
            "probe_template!(), \"nop\", svcr = out(reg) svcr",
            "string literal",
        ),
        (
            "concat!(\"rdsvl {svl}, #1\"), svl = out(reg) svl",
            "the template is",
        ),
        (
            "probe_template!(), other_template!(), svl = out(reg) svl",
            "another template piece",
        ),
    ] {
        let args = asm_arguments(&site(broken));
        assert_eq!(args.len(), 1, "{broken}");
        let problem = site_problem(&args[0], &templates);
        assert!(
            problem.as_deref().is_some_and(|p| p.contains(needle)),
            "{broken}: {problem:?}"
        );
    }
    let good = asm_arguments(&site(
        "probe_template!(), svcr = out(reg) svcr, options(nomem, nostack)",
    ));
    assert_eq!(site_problem(&good[0], &templates), None);
}

#[test]
fn test_sme2_fpstate_sites_use_the_templates() {
    let templates = [
        "fpcr_read_template!()",
        "fpcr_write_template!()",
        "fpsr_read_template!()",
        "fpsr_write_template!()",
    ];
    assert_eq!(
        site_templates("fpstate.rs", include_str!("fpstate.rs"), &templates),
        templates,
        "fpstate.rs: each template macro at exactly one site"
    );
}

#[test]
fn test_sme2_fpstate_templates_are_linted() {
    // The exact text of the four templates.
    assert_eq!(FPSR_TEMPLATES, ["mrs {fpsr}, fpsr", "msr fpsr, {fpsr}"]);
    assert_eq!(FPCR_TEMPLATES, ["mrs {fpcr}, fpcr", "msr fpcr, {fpcr}"]);
    for template in FPSR_TEMPLATES {
        if let Err(errors) = lint_fpsr_template(template) {
            panic!("{template}: {}", report(&errors));
        }
    }
    // The lint has no kind for FPCR: with the register renamed to FPSR an
    // FPCR template passes the FPSR kind, so every rule a non-streaming
    // accessor must keep (alphabet, braces and placeholders, one statement,
    // no directive, registers) holds, and as it stands the one finding left
    // is the register on the instruction line.
    for template in FPCR_TEMPLATES {
        let renamed = template.replace("fpcr", "fpsr");
        assert_eq!(lint_fpsr_template(&renamed), Ok(()), "{renamed}");
        let errors = match lint_fpsr_template(template) {
            Ok(()) => panic!("the FPSR kind accepted an FPCR template: {template:?}"),
            Err(errors) => errors,
        };
        assert_eq!(errors.len(), 1, "{template:?}:\n{}", report(&errors));
        assert_eq!(errors[0].rule, Rule::FpsrAccess, "{}", report(&errors));
        assert_eq!(errors[0].line, 1, "{}", report(&errors));
    }
    // The renamed check is a real one: a second statement, a directive or
    // another register adds findings.
    for broken in [
        "mrs {fpcr}, fpcr\nmsr fpcr, {fpcr}",
        ".arch_extension sve2\nmrs {fpcr}, fpcr",
        "mrs {fpcr}, fpcr; nop",
    ] {
        let renamed = broken.replace("fpcr", "fpsr");
        let errors = lint_fpsr_template(&renamed).err().unwrap_or_default();
        assert!(!errors.is_empty(), "{renamed:?}");
    }
}
