//! Source scan: every streaming region in the sources is in the registry.
//!
//! The template lint only checks what it is given, so a streaming region it
//! never sees would escape it. This scan closes that gap structurally, over
//! every `.rs` file of the crate's `src/` (subdirectories included), with
//! comments and string literals separated from the code by a small Rust
//! lexer ([`lex`]):
//!
//! - Raw `asm!` may appear only at the sites of [`ASM_SITES`], counted per
//!   file across the whole crate: the definition of `sme_kernels!`, the
//!   register probe, the two test-support FPSR accessors, and the two
//!   `prfm` prefetches of the NEON GEMM. An invocation is the identifier
//!   `asm` followed by `!`, whatever whitespace separates them (`asm !(…)`
//!   and `asm\n!(…)` count), so a `macro_rules!` wrapper around `asm!`
//!   defined anywhere in the crate is a site of its own. `global_asm!` and
//!   `naked_asm!` may not appear at all, nor may `include!` (which could
//!   splice in text the scan never reads), nor a path of the form
//!   `arch::$name` (an assembly macro named through a metavariable).
//! - No `use` may import `asm!`, `global_asm!` or `naked_asm!` from
//!   `core::arch` / `std::arch` (a renamed import such as `use
//!   core::arch::asm as emit;` would hide an invocation from the count).
//! - Every `#[path = "…"]` stays inside `src/` (no absolute path, no `..`),
//!   so every module file is one the scan reads.
//! - Every `streaming fn NAME` (the kernel syntax of `sme_kernels!`, which is
//!   not valid Rust anywhere else) must have exactly one registry entry with
//!   that name and that file, and every registry entry must be found in the
//!   sources. Kernel names are literal: `streaming fn $name` (a name passed
//!   through a metavariable of a wrapper macro) is allowed only in the
//!   definition of `sme_kernels!` itself. A file holds at most one
//!   `sme_kernels!` invocation.
//! - A string literal that mentions streaming-mode code (`smstart`, `smstop`,
//!   an SME `.arch_extension`, or a raw encoding: `.inst`, `.word`, `.long`,
//!   `.4byte`) may appear only inside a file's `sme_kernels!` invocation, or
//!   anywhere in the files of [`STREAMING_TEXT_FILES`] (the macro, the probe,
//!   the lint and its tests).
//!
//! The sources come from the source tree when it is readable, and otherwise
//! from copies of the files under [`EMBEDDED_DIRS`] embedded with
//! `include_str!` ([`EMBEDDED_SOURCES`]), for a test binary that runs where
//! the tree is not (a container, another machine). Where the tree is
//! readable, a file under those directories (at any depth) that is missing
//! from the embedded list fails the scan, so the list cannot fall behind.

use std::path::{Path, PathBuf};

use super::asm::KernelEntry;

#[cfg(test)]
#[path = "tests_registry.rs"]
mod tests;

/// Directories whose sources (at any depth) are embedded for hosts without
/// the source tree, relative to the crate's `src/`: where the SME2 kernels
/// live (`level2/gemv/` gets `sme2.rs` in Wave 5).
pub(super) const EMBEDDED_DIRS: [&str; 2] = ["level3/sme2", "level2/gemv"];

/// Files whose string literals may mention streaming-mode code anywhere:
/// the macro that writes the frame, the probe, the lint and its tests.
pub(super) const STREAMING_TEXT_FILES: &[&str] = &[
    "level3/sme2/asm.rs",
    "level3/sme2/detect.rs",
    "level3/sme2/lint.rs",
    "level3/sme2/lint_scan.rs",
    "level3/sme2/tests_lint.rs",
    "level3/sme2/tests_lint_frame.rs",
    "level3/sme2/tests_registry.rs",
];

/// The file that defines `sme_kernels!`, the only place where a kernel name
/// may come from a metavariable.
const MACRO_FILE: &str = "level3/sme2/asm.rs";

/// Files of the crate that may invoke `asm!` directly, and how often.
pub(super) const ASM_SITES: &[(&str, usize)] = &[
    // `prfm` prefetch hints of the NEON GEMM path (not streaming code).
    ("level3/gemm.rs", 1),
    ("level3/gemm_kernel/mod.rs", 1),
    // The `sme_kernels!` definition: every streaming region.
    ("level3/sme2/asm.rs", 1),
    // The register probe (SVCR, TPIDR2_EL0, RDSVL), outside streaming mode.
    ("level3/sme2/detect.rs", 1),
    // The FPSR read and write of the tests, outside streaming mode.
    ("level3/sme2/test_support.rs", 2),
];

/// Copies of every scanned file, for hosts without the source tree.
pub(super) const EMBEDDED_SOURCES: &[(&str, &str)] = &[
    ("level3/sme2/args.rs", include_str!("args.rs")),
    ("level3/sme2/asm.rs", include_str!("asm.rs")),
    ("level3/sme2/detect.rs", include_str!("detect.rs")),
    ("level3/sme2/driver.rs", include_str!("driver.rs")),
    ("level3/sme2/kernel_f32.rs", include_str!("kernel_f32.rs")),
    ("level3/sme2/kernel_f64.rs", include_str!("kernel_f64.rs")),
    ("level3/sme2/lint.rs", include_str!("lint.rs")),
    ("level3/sme2/lint_scan.rs", include_str!("lint_scan.rs")),
    ("level3/sme2/mod.rs", include_str!("mod.rs")),
    ("level3/sme2/oracle.rs", include_str!("oracle.rs")),
    ("level3/sme2/pack.rs", include_str!("pack.rs")),
    ("level3/sme2/parallel.rs", include_str!("parallel.rs")),
    ("level3/sme2/selftest.rs", include_str!("selftest.rs")),
    (
        "level3/sme2/test_support.rs",
        include_str!("test_support.rs"),
    ),
    (
        "level3/sme2/tests_contract.rs",
        include_str!("tests_contract.rs"),
    ),
    (
        "level3/sme2/tests_detect.rs",
        include_str!("tests_detect.rs"),
    ),
    ("level3/sme2/tests_gemm.rs", include_str!("tests_gemm.rs")),
    (
        "level3/sme2/tests_kernel_f32.rs",
        include_str!("tests_kernel_f32.rs"),
    ),
    (
        "level3/sme2/tests_kernel_f64.rs",
        include_str!("tests_kernel_f64.rs"),
    ),
    ("level3/sme2/tests_lint.rs", include_str!("tests_lint.rs")),
    (
        "level3/sme2/tests_lint_frame.rs",
        include_str!("tests_lint_frame.rs"),
    ),
    (
        "level3/sme2/tests_oracle.rs",
        include_str!("tests_oracle.rs"),
    ),
    (
        "level3/sme2/tests_registry.rs",
        include_str!("tests_registry.rs"),
    ),
    (
        "level3/sme2/tests_selftest.rs",
        include_str!("tests_selftest.rs"),
    ),
    ("level3/sme2/tests_smoke.rs", include_str!("tests_smoke.rs")),
    (
        "level3/sme2/tests_stress.rs",
        include_str!("tests_stress.rs"),
    ),
    (
        "level2/gemv/kernel_tests.rs",
        include_str!("../../level2/gemv/kernel_tests.rs"),
    ),
    (
        "level2/gemv/mod.rs",
        include_str!("../../level2/gemv/mod.rs"),
    ),
    (
        "level2/gemv/neon.rs",
        include_str!("../../level2/gemv/neon.rs"),
    ),
    (
        "level2/gemv/tests.rs",
        include_str!("../../level2/gemv/tests.rs"),
    ),
];

/// A source with its comments and string literals separated from its code.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Lexed {
    /// The code: comments become spaces, string and character literals
    /// become empty literals; line breaks are kept.
    pub(super) code: String,
    /// The contents of every string literal, raw strings included.
    pub(super) strings: Vec<String>,
    /// Where each of [`Lexed::strings`] sits in [`Lexed::code`] (the offset
    /// of its empty literal).
    pub(super) string_offsets: Vec<usize>,
}

/// Separates code from comments and literals: line comments, nested block
/// comments, string literals with escapes, byte strings, raw strings with any
/// number of `#`, character literals, and lifetimes (which stay code).
pub(super) fn lex(source: &str) -> Lexed {
    let chars: Vec<char> = source.chars().collect();
    let mut lexed = Lexed::default();
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev_ident = i > 0 && ident(chars[i - 1]);
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let mut depth = 0_usize;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    lexed.code.push(if chars[i] == '\n' { '\n' } else { ' ' });
                    i += 1;
                }
            }
            continue;
        }
        // Raw strings: `r"…"`, `r#"…"#`, `br"…"` (not part of an identifier
        // such as `for"`, except for the `b` prefix).
        let raw_start = c == 'r'
            && (!prev_ident || (i > 0 && chars[i - 1] == 'b' && !(i > 1 && ident(chars[i - 2]))));
        if raw_start {
            let mut hashes = 0;
            while chars.get(i + 1 + hashes) == Some(&'#') {
                hashes += 1;
            }
            if chars.get(i + 1 + hashes) == Some(&'"') {
                let mut j = i + 2 + hashes;
                let mut text = String::new();
                while j < chars.len() {
                    if chars[j] == '"' && (0..hashes).all(|h| chars.get(j + 1 + h) == Some(&'#')) {
                        break;
                    }
                    text.push(chars[j]);
                    j += 1;
                }
                lexed.string_offsets.push(lexed.code.len());
                lexed.code.push_str("\"\"");
                lexed.code.extend(text.chars().filter(|&ch| ch == '\n'));
                lexed.strings.push(text);
                i = j + 1 + hashes;
                continue;
            }
        }
        if c == '"' {
            let mut j = i + 1;
            let mut text = String::new();
            while j < chars.len() && chars[j] != '"' {
                if chars[j] == '\\' {
                    text.push(chars[j]);
                    j += 1;
                }
                if let Some(&ch) = chars.get(j) {
                    text.push(ch);
                }
                j += 1;
            }
            lexed.string_offsets.push(lexed.code.len());
            lexed.code.push_str("\"\"");
            lexed.code.extend(text.chars().filter(|&ch| ch == '\n'));
            lexed.strings.push(text);
            i = j + 1;
            continue;
        }
        if c == '\'' {
            // `'\n'`, `'\''`, `'\u{1f600}'`: an escape runs to the next quote.
            if next == Some('\\') {
                let mut j = i + 3;
                while j < chars.len() && chars[j] != '\'' {
                    j += 1;
                }
                lexed.code.push_str("' '");
                i = j + 1;
                continue;
            }
            // `'x'` is a character; `'a` (no closing quote) a lifetime.
            if chars.get(i + 2) == Some(&'\'') {
                lexed.code.push_str("' '");
                i += 3;
                continue;
            }
        }
        lexed.code.push(c);
        i += 1;
    }
    lexed
}

/// What the scan finds in one source.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Found {
    /// `asm!` invocations (not `sme_asm!`-style names that end in `asm`).
    pub(super) asm: usize,
    /// `global_asm!` and `naked_asm!` invocations.
    pub(super) module_asm: usize,
    /// `sme_kernels!` invocations.
    pub(super) invocations: usize,
    /// Names after `streaming fn`.
    pub(super) kernels: Vec<String>,
    /// `streaming fn $…`: a kernel name taken from a metavariable.
    pub(super) generated_names: usize,
    /// `use` declarations that import `asm!`, `global_asm!` or `naked_asm!`.
    pub(super) asm_imports: usize,
    /// `include!` invocations.
    pub(super) includes: usize,
    /// `arch::$…` paths: a macro of `core::arch` named by a metavariable.
    pub(super) generated_arch_paths: usize,
    /// `#[path = "…"]` values that leave the directory tree (absolute, or
    /// with a `..` component).
    pub(super) escaping_paths: Vec<String>,
    /// String literals that mention streaming-mode code outside the file's
    /// `sme_kernels!` invocation.
    pub(super) streaming_text: usize,
}

/// Whether `c` can be part of an identifier.
fn ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte offsets in `code` of the identifier `name` used as a macro: the
/// whole identifier, then optional whitespace, then `!` (not `!=`). Returns
/// the offset just after the `!`.
pub(super) fn macro_calls(code: &str, name: &str) -> Vec<usize> {
    let mut calls = Vec::new();
    for (at, _) in code.match_indices(name) {
        if code[..at].chars().last().is_some_and(ident_char) {
            continue;
        }
        let after = &code[at + name.len()..];
        if after.starts_with(ident_char) {
            continue;
        }
        let rest = after.trim_start();
        if rest.starts_with('!') && !rest.starts_with("!=") {
            calls.push(code.len() - rest.len() + 1);
        }
    }
    calls
}

/// The byte range of the delimited group (`{…}`, `(…)` or `[…]`) that starts
/// at the first non-whitespace character from `from`, or `None`.
fn delimited_group(code: &str, from: usize) -> Option<(usize, usize)> {
    let rest = code.get(from..)?;
    let open_at = from + (rest.len() - rest.trim_start().len());
    let open = code[open_at..].chars().next()?;
    let close = match open {
        '{' => '}',
        '(' => ')',
        '[' => ']',
        _ => return None,
    };
    let mut depth = 0_usize;
    for (offset, ch) in code[open_at..].char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some((open_at, open_at + offset));
            }
        }
    }
    None
}

/// Whether a `#[path = "…"]` value leaves the directory tree.
fn escapes_tree(value: &str) -> bool {
    let value = value.replace('\\', "/");
    value.starts_with('/') || value.contains(':') || value.split('/').any(|part| part == "..")
}

/// The values of the `#[path = "…"]` attributes of lexed code.
fn path_attributes(lexed: &Lexed) -> Vec<String> {
    let code = lexed.code.as_str();
    let mut values = Vec::new();
    for (at, _) in code.match_indices("path") {
        let before = code[..at].trim_end();
        let attribute =
            before.ends_with('[') && before[..before.len() - 1].trim_end().ends_with('#');
        let after = code[at + 4..].trim_start();
        if !attribute || !after.starts_with('=') {
            continue;
        }
        let value_at = code.len() - after[1..].trim_start().len();
        if let Some(index) = lexed.string_offsets.iter().position(|&o| o == value_at) {
            values.push(lexed.strings[index].clone());
        }
    }
    values
}

/// Scans the code of one source (see [`lex`]).
pub(super) fn scan(source: &str) -> Found {
    let lexed = lex(source);
    let code = lexed.code.as_str();
    let mut found = Found {
        asm: macro_calls(code, "asm").len(),
        module_asm: macro_calls(code, "global_asm").len() + macro_calls(code, "naked_asm").len(),
        includes: macro_calls(code, "include").len(),
        ..Found::default()
    };
    let invocations = macro_calls(code, "sme_kernels");
    found.invocations = invocations.len();
    let spans: Vec<(usize, usize)> = invocations
        .iter()
        .filter_map(|&after| delimited_group(code, after))
        .collect();
    let words: Vec<&str> = code
        .split(|c: char| !(ident_char(c) || c == '$'))
        .filter(|word| !word.is_empty())
        .collect();
    for window in words.windows(3) {
        if let [first, second, name] = window
            && *first == "streaming"
            && *second == "fn"
        {
            if name.starts_with('$') {
                found.generated_names += 1;
            } else if name.starts_with(|c: char| c.is_alphabetic() || c == '_') {
                found.kernels.push((*name).to_owned());
            }
        }
    }
    for (at, _) in code.match_indices("arch") {
        let whole = !code[..at].chars().last().is_some_and(ident_char);
        let rest = code[at + 4..].trim_start();
        if whole
            && let Some(path) = rest.strip_prefix("::")
            && path.trim_start().starts_with('$')
        {
            found.generated_arch_paths += 1;
        }
    }
    found.asm_imports = asm_imports(code);
    found.escaping_paths = path_attributes(&lexed)
        .into_iter()
        .filter(|value| escapes_tree(value))
        .collect();
    found.streaming_text = lexed
        .strings
        .iter()
        .zip(&lexed.string_offsets)
        .filter(|&(text, &offset)| {
            mentions_streaming(text) && !spans.iter().any(|&(a, b)| a < offset && offset < b)
        })
        .count();
    found
}

/// `use` declarations of lexed code that import an assembly macro from
/// `core::arch` / `std::arch` (directly, renamed or in a group).
pub(super) fn asm_imports(code: &str) -> usize {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut count = 0;
    let mut rest = code;
    while let Some(at) = rest.find("use") {
        let starts_word = !rest[..at].chars().last().is_some_and(ident);
        let after = &rest[at + 3..];
        let ends_word = after.starts_with(|c: char| c.is_whitespace() || c == '{' || c == ':');
        if starts_word && ends_word {
            let end = after.find(';').unwrap_or(after.len());
            let words: Vec<&str> = after[..end]
                .split(|c: char| !ident(c))
                .filter(|word| !word.is_empty())
                .collect();
            let names_arch = words.contains(&"arch");
            let names_asm = words
                .iter()
                .any(|word| matches!(*word, "asm" | "global_asm" | "naked_asm"));
            if names_arch && names_asm {
                count += 1;
            }
            rest = &after[end..];
        } else {
            rest = after;
        }
    }
    count
}

/// The registry file of an entry relative to `src/` (`file!()` is relative
/// to the workspace root and may use `\` on some hosts).
pub(super) fn entry_path(entry: &KernelEntry) -> String {
    let file = entry.file.replace('\\', "/");
    match file.rsplit_once("src/") {
        Some((_, rest)) => rest.to_owned(),
        None => file,
    }
}

/// Every problem of a set of sources against a registry. Empty means: each
/// streaming region is registered exactly once and each raw `asm!` is an
/// allowed site.
pub(super) fn check_sources(sources: &[(String, String)], registry: &[KernelEntry]) -> Vec<String> {
    let mut problems = Vec::new();
    let mut in_sources = Vec::new();
    for (path, text) in sources {
        let found = scan(text);
        let allowed = ASM_SITES
            .iter()
            .find(|(site, _)| site == path)
            .map_or(0, |(_, count)| *count);
        if found.asm != allowed {
            problems.push(format!(
                "{path}: {} raw asm! sites, {allowed} allowed (streaming regions use sme_kernels!)",
                found.asm
            ));
        }
        if found.module_asm != 0 {
            problems.push(format!("{path}: global_asm!/naked_asm! is not allowed"));
        }
        if found.invocations > 1 {
            problems.push(format!(
                "{path}: {} sme_kernels! invocations, at most one",
                found.invocations
            ));
        }
        if !found.kernels.is_empty() && found.invocations == 0 {
            problems.push(format!("{path}: `streaming fn` outside sme_kernels!"));
        }
        if found.generated_names != 0 && path != MACRO_FILE {
            problems.push(format!(
                "{path}: `streaming fn $…`: kernel names must be literal (no wrapper macros), \
                 so that the scan can match them with the registry"
            ));
        }
        if found.asm_imports != 0 {
            problems.push(format!(
                "{path}: `use` of an assembly macro from core::arch / std::arch \
                 (write core::arch::asm! in full at an ASM_SITES entry)"
            ));
        }
        if found.includes != 0 {
            problems.push(format!(
                "{path}: include! splices in text the scan does not read"
            ));
        }
        if found.generated_arch_paths != 0 {
            problems.push(format!(
                "{path}: `arch::$…` names an assembly macro through a metavariable"
            ));
        }
        for value in &found.escaping_paths {
            problems.push(format!(
                "{path}: #[path = {value:?}] leaves src/ (the scan reads only src/)"
            ));
        }
        if found.streaming_text != 0 && !STREAMING_TEXT_FILES.contains(&path.as_str()) {
            problems.push(format!(
                "{path}: {} string literal(s) with streaming-mode code outside the \
                 file's sme_kernels! invocation",
                found.streaming_text
            ));
        }
        for name in found.kernels {
            let registered = registry
                .iter()
                .filter(|entry| entry.name == name && entry_path(entry) == *path)
                .count();
            if registered != 1 {
                problems.push(format!(
                    "{path}: streaming region `{name}` has {registered} registry entries, needs 1 \
                     (add the module's SME_KERNELS to the registry in tests_registry.rs)"
                ));
            }
            in_sources.push((path.clone(), name));
        }
    }
    for entry in registry {
        let key = (entry_path(entry), entry.name.to_owned());
        if !in_sources.contains(&key) {
            problems.push(format!(
                "registered `{}` ({}) is not in the scanned sources",
                key.1, key.0
            ));
        }
    }
    for (index, entry) in registry.iter().enumerate() {
        if registry[..index]
            .iter()
            .any(|other| other.name == entry.name)
        {
            problems.push(format!("`{}` is registered twice", entry.name));
        }
    }
    problems
}

/// The crate's `src/` directory.
fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Whether a `src/`-relative path lies under one of [`EMBEDDED_DIRS`]
/// (compared by path component: `level3/sme2_extra.rs` does not).
pub(super) fn in_embedded_dir(path: &str) -> bool {
    EMBEDDED_DIRS.iter().any(|dir| {
        path.strip_prefix(dir)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    })
}

/// Every `.rs` file under `src/` (at any depth) as `(path relative to src/,
/// text)`, sorted, or `None` when the tree is not readable.
fn tree_sources() -> Option<Vec<(String, String)>> {
    let root = src_dir();
    std::fs::read_dir(&root).ok()?;
    let mut pending = vec![root.clone()];
    let mut sources = Vec::new();
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("cannot list {}: {err}", dir.display()));
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(&root)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            if relative.ends_with(".rs") {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("cannot read {relative}: {err}"));
                sources.push((relative, text));
            }
        }
    }
    sources.sort();
    Some(sources)
}

/// The scanned sources as `(path relative to src/, text)`, and whether they
/// were read from the source tree (every `.rs` file of `src/`) or are the
/// embedded copies (the files under [`EMBEDDED_DIRS`]).
pub(super) fn sources() -> (Vec<(String, String)>, bool) {
    match tree_sources() {
        Some(sources) => (sources, true),
        None => (embedded_sources(), false),
    }
}

/// The embedded copies as owned `(path, text)` pairs.
pub(super) fn embedded_sources() -> Vec<(String, String)> {
    EMBEDDED_SOURCES
        .iter()
        .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
        .collect()
}

/// How the files of the tree under [`EMBEDDED_DIRS`] and the embedded list
/// `listed` differ; empty when they match exactly.
pub(super) fn embedded_mismatch(tree: &[&str], listed: &[&str]) -> Vec<String> {
    let mut problems = Vec::new();
    for path in tree.iter().filter(|path| in_embedded_dir(path)) {
        if !listed.contains(path) {
            problems.push(format!("{path} is missing from EMBEDDED_SOURCES"));
        }
    }
    for path in listed {
        if !tree.contains(path) {
            problems.push(format!("EMBEDDED_SOURCES lists {path}, which is gone"));
        }
        if !in_embedded_dir(path) {
            problems.push(format!(
                "EMBEDDED_SOURCES lists {path}, outside EMBEDDED_DIRS"
            ));
        }
    }
    problems
}

/// Whether string-literal text looks like streaming-mode assembly: an SME
/// mode switch or extension, or a raw instruction encoding.
pub(super) fn mentions_streaming(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    let raw_encoding = [".inst", ".word", ".long", ".4byte"]
        .iter()
        .any(|directive| {
            text.match_indices(directive).any(|(at, _)| {
                !text[at + directive.len()..]
                    .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
            })
        });
    raw_encoding
        || ["smstart", "smstop", ".arch_extension sme"]
            .iter()
            .any(|needle| text.contains(needle))
}
