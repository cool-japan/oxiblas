//! Static lint for the inline-assembly templates of the SME2 backend.
//!
//! An instruction that is illegal in streaming mode assembles without
//! complaint and only fails at run time, with SIGILL, on SME hardware. QEMU
//! cannot model the Apple M4 (SME without non-streaming SVE), so this lint is
//! the primary defence. It checks the template text and runs on any host:
//!
//! - **Alphabet.** A template holds only ASCII letters, digits, spaces, line
//!   feeds and `_ . , # : [ ] { } - + / !` (plus `;` and `*`, which the
//!   portability rules report). The assembler splits statements at `\r` on
//!   every target and at `%%` on Mach-O, which a line-based lint cannot see,
//!   so those and every other character are rejected before parsing.
//! - **Normalisation.** `{{` and `}}` become literal braces. Placeholders
//!   (`{name}`, `{0}`, `{}`, and `{name:w}` / `{name:x}`) become a neutral
//!   operand. Other modifiers (`:b`, `:h`, `:s`, `:d`, `:q`, `:v`) are
//!   rejected because they print FP/SIMD register names. A placeholder may
//!   not touch an identifier character or another placeholder (`x{n}`,
//!   `v{n}.16b`, `z{n}.d`, `{a}{b}`): the substituted value would build a
//!   register name.
//! - **Allowlist.** Every mnemonic must belong to one of the [`FAMILIES`]
//!   (base A64, SVE predicates, counters, moves, floating point and memory,
//!   and SME); each family documents why it is legal in streaming mode.
//! - **Reject list.** [`REJECTED`] and [`REJECTED_PREFIXES`] name instructions
//!   that are illegal in streaming mode without FEAT_SME_FA64, or that the
//!   backend forbids (calls, returns, system-register access).
//! - **Registers.** Templates may not use Advanced SIMD or scalar FP registers
//!   (`v0`, `q0`, `d0`, `s0`, `h0`, `b0`), `ffr`, vector-addressed memory
//!   operands (gathers and scatters), or any of [`RESERVED_REGISTERS`].
//! - **The frame of a streaming template** (`asm.rs`). Leading
//!   `.arch_extension sme2` (directives do not carry over between `asm!`
//!   blocks); `mrs x17, fpsr` as the first instruction; a straight-line
//!   base-A64 prologue; exactly one plain `smstart` followed at once by
//!   `msr fpsr, xzr`; the body; the exit label `90:`, `mrs x16, fpsr` and
//!   exactly one plain `smstop`; then exactly `orr x17, x17, x16` and
//!   `msr fpsr, x17`, and nothing else. These are the only `mrs` / `msr`
//!   allowed, and `x17` / `w17` appears nowhere else. Labels are numeric,
//!   never made only of the digits 0 and 1, defined inside the streaming
//!   region, and `90` belongs to the frame. Every `Nb` / `Nf` branch target
//!   resolves in its direction.
//! - **Portability.** A template has no `;`, which is a comment on Mach-O but
//!   a separator on ELF: `add …; add …` silently loses the second instruction
//!   on macOS. It has no comments at all, and no directive other than
//!   `.arch_extension` (so no `.inst` or `.word`).
//! - **Registered kernels** ([`lint_kernel`]). Every placeholder is one of the
//!   kernel's declared `const` operands, written bare (`{off_a}`): the
//!   `sme_kernels!` grammar admits no other operand, and this check makes the
//!   registry entry prove it. Each is written as an immediate (`#{off}` or
//!   `#-{off}`). `sme-f64f64` is enabled exactly when the entry says so.
//! - **Read-only argument block.** No store uses `x0` as its base.
//!
//! The registry walk (`tests_registry.rs`) calls [`lint_kernel`] for every
//! registered kernel; the register probe in `detect.rs` is checked with
//! [`lint_probe_template`] and the test-support FPSR accessors with
//! [`lint_fpsr_template`]. [`report`] formats violations for assertion
//! messages.

use super::asm::KernelEntry;

#[cfg(test)]
#[path = "tests_lint.rs"]
mod tests_rules;

#[cfg(test)]
#[path = "tests_lint_frame.rs"]
mod tests_frame;

/// A rule enforced by the lint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rule {
    /// A `{` or `}` that is neither doubled nor part of a placeholder.
    MalformedBraces,
    /// A placeholder modifier other than `w` or `x`.
    PlaceholderModifier,
    /// A `;` statement separator (a comment on Mach-O).
    Semicolon,
    /// A comment inside the template.
    Comment,
    /// A directive other than `.arch_extension sme`, `sme2` or `sme-f64f64`.
    ForbiddenDirective,
    /// A directive after the first instruction or label.
    DirectiveNotLeading,
    /// The `.arch_extension` that the template kind needs is missing.
    MissingArchExtension,
    /// A label that is not a plain number.
    NamedLabel,
    /// A label made only of the digits 0 and 1.
    BinaryLookingLabel,
    /// A branch target that is not `Nb` / `Nf`, or that does not resolve.
    BranchTarget,
    /// A mnemonic on the reject list.
    RejectedMnemonic,
    /// A mnemonic in no allowlist family.
    UnknownMnemonic,
    /// An Advanced SIMD or scalar FP register, or `ffr`.
    AdvSimdOrFfrRegister,
    /// A vector-addressed memory operand (gather or scatter).
    VectorAddressing,
    /// A register the backend reserves.
    ReservedRegister,
    /// Not exactly one plain `smstart` before exactly one plain `smstop`.
    SmstartSmstop,
    /// Something after `smstop` other than the two frame lines.
    AfterSmstop,
    /// A prologue statement that is not straight-line base A64.
    Prologue,
    /// A label outside the streaming region.
    LabelOutsideRegion,
    /// A probe template statement other than its register reads.
    ProbeInstruction,
    /// A line of the FPSR frame is missing or misplaced.
    FpsrFrame,
    /// `x17` / `w17` outside the FPSR frame.
    X17,
    /// The exit label `90:` is missing before `mrs x16, fpsr`, or defined twice.
    ExitLabel,
    /// A placeholder that is not a bare declared `const` operand.
    UndeclaredPlaceholder,
    /// `.arch_extension sme-f64f64` does not match the registry entry.
    ExtensionMismatch,
    /// An FPSR accessor template other than one `mrs` / `msr` of FPSR.
    FpsrAccess,
    /// A character outside the template alphabet (control characters such
    /// as `\r`, `%`, non-ASCII, ...).
    ForbiddenCharacter,
    /// A placeholder glued to an identifier character or to another
    /// placeholder (`x{n}`, `z{n}.d`, `{a}{b}`).
    GluedPlaceholder,
    /// A placeholder of a registered kernel that is not an immediate
    /// (`#{off}` or `#-{off}`).
    PlaceholderNotImmediate,
    /// A store through `x0`, the argument block.
    StoreToArgs,
}

impl Rule {
    /// What the rule requires, used in violation messages.
    fn why(self) -> &'static str {
        match self {
            Self::MalformedBraces => "literal braces must be doubled (`{{`, `}}`)",
            Self::PlaceholderModifier => "only the `w` and `x` placeholder modifiers are allowed",
            Self::Semicolon => "`;` is a comment on Mach-O and a separator on ELF",
            Self::Comment => "comments belong in the Rust source, not in the template",
            Self::ForbiddenDirective => "only `.arch_extension sme | sme2 | sme-f64f64` is allowed",
            Self::DirectiveNotLeading => "directives come before the first statement",
            Self::MissingArchExtension => {
                "a streaming template needs `.arch_extension sme2`, the probe `.arch_extension sme`"
            }
            Self::NamedLabel => "use numeric local labels",
            Self::BinaryLookingLabel => "labels made only of 0 and 1 read as binary literals",
            Self::BranchTarget => {
                "targets are numeric labels defined in that direction (`20b`, `90f`)"
            }
            Self::RejectedMnemonic => "this instruction is not allowed",
            Self::UnknownMnemonic => {
                "in no allowlist family; add it with a reason if streaming-legal"
            }
            Self::AdvSimdOrFfrRegister => "Advanced SIMD, scalar FP and FFR registers are illegal",
            Self::VectorAddressing => "gathers and scatters are illegal in streaming mode",
            Self::ReservedRegister => "x18, x19, x29 / fp, x30 / lr and sp are reserved",
            Self::SmstartSmstop => "one plain smstart, then one plain smstop (SM and ZA together)",
            Self::AfterSmstop => "only `orr x17, x17, x16` and `msr fpsr, x17` follow smstop",
            Self::Prologue => {
                "the prologue runs outside streaming mode: straight-line base A64 only"
            }
            Self::LabelOutsideRegion => "labels belong inside the streaming region",
            Self::ProbeInstruction => "the probe only reads SVCR, TPIDR2_EL0 and RDSVL",
            Self::FpsrFrame => {
                "the FPSR frame of asm.rs is required: mrs x17 first, msr fpsr, xzr right after \
                 smstart, mrs x16 right before smstop, then orr and msr fpsr, x17"
            }
            Self::X17 => "x17 is reserved for the FPSR frame",
            Self::ExitLabel => "`90:` is the frame's exit label, right before `mrs x16, fpsr`",
            Self::UndeclaredPlaceholder => {
                "only bare placeholders of declared const operands (`{off_a}`) are allowed"
            }
            Self::ExtensionMismatch => "`.arch_extension sme-f64f64` must match the entry",
            Self::FpsrAccess => "an FPSR accessor is one `mrs {x}, fpsr` or `msr fpsr, {x}`",
            Self::ForbiddenCharacter => {
                "a template uses only ASCII letters, digits, spaces, line feeds and \
                 `_ . , # : [ ] { } - + / !` (`\\r` and `%%` separate statements)"
            }
            Self::GluedPlaceholder => {
                "a placeholder stands alone: an identifier character or placeholder right \
                 next to it would build a register name (`x{n}` prints `x17`)"
            }
            Self::PlaceholderNotImmediate => {
                "a kernel's const operands are immediates: write `#{off}` (or `#-{off}`)"
            }
            Self::StoreToArgs => "the argument block is read-only: never store through `x0`",
        }
    }
}

/// One violation found by the lint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LintError {
    /// The rule that failed.
    pub(super) rule: Rule,
    /// 1-based line of the template.
    pub(super) line: usize,
    /// The offending statement after normalisation, with an explanation.
    pub(super) text: String,
}

/// Formats violations one per line, for assertion messages.
pub(super) fn report(errors: &[LintError]) -> String {
    errors
        .iter()
        .map(|e| format!("line {}: {:?}: {}", e.line, e.rule, e.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Checks a template that contains one streaming region.
pub(super) fn lint_streaming_template(template: &str) -> Result<(), Vec<LintError>> {
    lint(template, Kind::Streaming)
}

/// Checks the non-streaming register probe: no `smstart` / `smstop`, only
/// `mrs` of SVCR or TPIDR2_EL0 and `rdsvl`.
pub(super) fn lint_probe_template(template: &str) -> Result<(), Vec<LintError>> {
    lint(template, Kind::Probe)
}

/// Checks a non-streaming FPSR accessor: one `mrs {x}, fpsr` or one
/// `msr fpsr, {x}`, nothing else.
pub(super) fn lint_fpsr_template(template: &str) -> Result<(), Vec<LintError>> {
    lint(template, Kind::FpsrAccess)
}

/// Checks a registered kernel: its template as a streaming template, its
/// placeholders against its declared `const` operands, and its extensions.
pub(super) fn lint_kernel(entry: &KernelEntry) -> Result<(), Vec<LintError>> {
    let mut errors = match lint_streaming_template(entry.template) {
        Ok(()) => Vec::new(),
        Err(errors) => errors,
    };
    for (line, inner, immediate) in placeholder_sites(entry.template) {
        if !entry.consts.contains(&inner.as_str()) {
            let rule = Rule::UndeclaredPlaceholder;
            let text = format!("`{{{inner}}}` in `{}`: {}", entry.name, rule.why());
            errors.push(LintError { rule, line, text });
        }
        if !immediate {
            let rule = Rule::PlaceholderNotImmediate;
            let text = format!("`{{{inner}}}` in `{}`: {}", entry.name, rule.why());
            errors.push(LintError { rule, line, text });
        }
    }
    let enables_f64 = entry.template.lines().any(|line| {
        line.split_whitespace()
            .eq([".arch_extension", "sme-f64f64"])
    });
    if enables_f64 != entry.f64f64 {
        let rule = Rule::ExtensionMismatch;
        let text = format!(
            "`{}`: the entry says f64f64 = {}, the template says {enables_f64}",
            entry.name, entry.f64f64
        );
        errors.push(LintError {
            rule,
            line: 1,
            text,
        });
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Instruction families
// ---------------------------------------------------------------------------

/// Base A64 instructions on general-purpose registers: integer arithmetic and
/// logic, moves, compares, conditional selects, general-purpose loads and
/// stores, branches to local labels, `nop`, and the `prfm` prefetch hint.
///
/// They are legal in streaming mode because the streaming-mode restrictions
/// of FEAT_SME without FEAT_SME_FA64 cover only Advanced SIMD, scalar
/// floating-point and some SVE instructions. Instructions that use only
/// general-purpose registers execute normally, and `prfm` is a hint that
/// never faults. These are also the only instructions allowed in the
/// non-streaming prologue before `smstart`.
///
/// `ldr`, `str` and `mov` are listed for their general-purpose forms. The
/// register rule catches their FP/SIMD forms, and the prologue rule keeps
/// their SVE and ZA forms out of the prologue.
const BASE_A64: &[&str] = &[
    "add", "adds", "sub", "subs", "neg", "cmp", "cmn", "tst", "mov", "movz", "movk", "movn", "and",
    "ands", "orr", "eor", "bic", "lsl", "lsr", "asr", "mul", "madd", "msub", "umulh", "udiv",
    "csel", "csinc", "cset", "cinc", "ccmp", "ubfx", "ubfiz", "uxtw", "sxtw", "ldr", "ldp", "ldur",
    "ldrsw", "str", "stp", "stur", "b", "cbz", "cbnz", "tbz", "tbnz", "nop", "prfm",
];

// SVE in streaming mode. The Z and P registers take the streaming vector
// length, and SVE data-processing, predicate and contiguous load/store
// instructions execute normally. Without FEAT_SME_FA64, FEAT_SME makes only a
// specific set illegal: gathers and scatters; first-fault and non-fault loads
// and FFR access; FADDA, FEXPA, FTMAD, FTSSEL, FTSMUL, COMPACT, ADR, HISTCNT,
// HISTSEG, MATCH, NMATCH and LD1RO*; and the SVE2 crypto and 128-bit
// polynomial instructions. Those are in `REJECTED` / `REJECTED_PREFIXES` or
// caught by the vector-addressing rule; the families below hold none of them.

/// SVE predicate instructions, including the SME2 predicate-as-counter forms
/// (`ptrue pn8.s`, `whilelt pn9.s, …, vlx4`, `pext`, `cntp …, vlx4`).
///
/// Streaming-legal: predicate generation is outside the FA64-only set, and
/// the counter forms exist only for multi-vector streaming code.
const SVE_PREDICATES: &[&str] = &[
    "ptrue", "pfalse", "whilelt", "whilelo", "whilele", "whilels", "pext", "cntp",
];

/// SVE vector-length counters and address arithmetic.
///
/// Streaming-legal: in streaming mode they count in units of the streaming
/// vector length, which is what the kernels step by.
const SVE_COUNTERS: &[&str] = &[
    "cntb", "cnth", "cntw", "cntd", "incb", "inch", "incw", "incd", "decb", "dech", "decw", "decd",
    "addvl", "addpl", "rdvl",
];

/// SVE data movement within Z registers (`dup z28.s, w20`, `dup z5.q,
/// z4.q[0]`, `movprfx`, `sel`).
///
/// Streaming-legal: SVE permutes and moves other than COMPACT are outside the
/// FA64-only set.
const SVE_MOVES: &[&str] = &["dup", "movprfx", "sel"];

/// SVE floating-point arithmetic (`fmul`, `fmla`, …). FMLA, FMLS, FADD and
/// FSUB also have SME2 forms that accumulate into groups of ZA vectors.
///
/// Streaming-legal: only FADDA, FEXPA, FTMAD, FTSSEL and FTSMUL among the SVE
/// floating-point instructions are FA64-only.
const SVE_FLOAT: &[&str] = &[
    "fmul", "fadd", "fsub", "fmla", "fmls", "fmad", "fmsb", "fneg", "fabs",
];

/// SVE contiguous loads and stores, including the replicating loads (`ld1rw`,
/// `ld1rqw`) and the SME2 multi-vector forms (`ld1w {z0.s-z3.s}, pn8/z, …`).
/// `ld1w` / `st1w` / `ld1d` / `st1d` also have SME forms that move ZA tile
/// slices.
///
/// Streaming-legal as long as the address is a scalar base plus scalar or
/// immediate offset; vector-addressed forms are gathers and scatters, which
/// the vector-addressing rule rejects.
const SVE_MEMORY: &[&str] = &[
    "ld1w", "ld1d", "st1w", "st1d", "ld1rw", "ld1rd", "ld1rqw", "ld1rqd", "ldnt1w", "ldnt1d",
    "stnt1w", "stnt1d",
];

/// SME and SME2 instructions.
///
/// They read or write ZA or need streaming mode, which a streaming region
/// provides: plain `smstart` turns on both PSTATE.SM and PSTATE.ZA. `ldr` /
/// `str` (ZA array vectors), `ld1w` / `st1w` (ZA tile slices) and `fmla` /
/// `fadd` (ZA array groups) also have ZA forms; their mnemonics are listed
/// with their base and SVE families.
const SME: &[&str] = &[
    "zero", "fmopa", "fmops", "mova", "rdsvl", "addsvl", "addspl",
];

/// Every allowlist family of a streaming template.
const FAMILIES: &[&[&str]] = &[
    BASE_A64,
    SVE_PREDICATES,
    SVE_COUNTERS,
    SVE_MOVES,
    SVE_FLOAT,
    SVE_MEMORY,
    SME,
];

/// Mnemonics that are never allowed, grouped by reason.
const REJECTED: &[(&str, &[&str])] = &[
    (
        "FFR access is illegal in streaming mode",
        &["rdffr", "rdffrs", "setffr", "wrffr"],
    ),
    (
        "illegal in streaming mode without FEAT_SME_FA64",
        &[
            "fadda", "fexpa", "ftmad", "ftssel", "ftsmul", "compact", "histcnt", "histseg",
            "match", "nmatch",
        ],
    ),
    ("the SVE vector ADR is illegal in streaming mode", &["adr"]),
    (
        "crypto and 128-bit polynomial instructions are illegal in streaming mode",
        &["pmullb", "pmullt", "rax1"],
    ),
    (
        "calls are not allowed inside an asm template",
        &["bl", "blr", "blraa", "blraaz", "blrab", "blrabz"],
    ),
    (
        "indirect branches and returns leave the template",
        &[
            "br", "braa", "braaz", "brab", "brabz", "ret", "retaa", "retab", "eret",
        ],
    ),
    (
        "exception-generating instructions are not allowed",
        &["svc", "hvc", "smc", "brk", "hlt", "udf"],
    ),
    (
        "only smstart / smstop may change PSTATE.SM and PSTATE.ZA",
        &["msr"],
    ),
    (
        "system-register reads belong to the probe, not to streaming templates",
        &["mrs"],
    ),
];

/// Mnemonic prefixes that are never allowed, grouped by reason.
const REJECTED_PREFIXES: &[(&str, &[&str])] = &[
    (
        "first-fault and non-fault loads use the FFR, which is illegal in streaming mode",
        &["ldff1", "ldnf1"],
    ),
    (
        "LD1RO* (256-bit replicate) is illegal in streaming mode",
        &["ld1ro"],
    ),
    (
        "crypto instructions are illegal in streaming mode",
        &["aes", "sha", "sm3", "sm4"],
    ),
];

/// Registers the backend never touches: `x18` is the platform register on
/// Apple and Windows targets, `x19` is LLVM's base pointer, `x29` / `fp` the
/// frame pointer, `x30` / `lr` the link register, and the templates run with
/// `nostack`, so `sp` is off limits as well.
const RESERVED_REGISTERS: &[&str] = &[
    "x18", "w18", "x19", "w19", "x29", "w29", "fp", "x30", "w30", "lr", "sp", "wsp",
];

/// Branch conditions: the base A64 condition codes and the SVE aliases that
/// WHILE* and PTEST results are tested with.
const CONDITIONS: &[&str] = &[
    "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
    "none", "any", "nlast", "last", "first", "nfrst", "pmore", "plast", "tcont", "tstop",
];

/// Extensions a template may enable.
const ARCH_EXTENSIONS: &[&str] = &["sme", "sme2", "sme-f64f64"];

/// What a placeholder becomes after normalisation: a token that matches no
/// register rule.
const OPERAND: &str = "operand";

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// The two kinds of template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// One streaming region: prologue, `smstart`, body, `smstop`.
    Streaming,
    /// The non-streaming register probe.
    Probe,
    /// A non-streaming FPSR accessor of the test support.
    FpsrAccess,
}

/// A parsed statement.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    /// A label definition (`20:`).
    Label(String),
    /// A directive with its arguments (`.arch_extension sme2`).
    Directive(String),
    /// An instruction.
    Instr {
        /// Lower-case mnemonic, including a `.cond` suffix.
        mnemonic: String,
        /// Everything after the mnemonic.
        operands: String,
    },
}

/// A statement with its position.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Stmt {
    /// 1-based template line.
    line: usize,
    /// The normalised statement text.
    text: String,
    /// What the statement is.
    item: Item,
}

/// The violations found in one template.
#[derive(Debug, Default)]
struct Findings(Vec<LintError>);

impl Findings {
    /// Records that `stmt` violates `rule`, explained by [`Rule::why`].
    fn flag(&mut self, rule: Rule, stmt: &Stmt) {
        self.flag_because(rule, stmt, rule.why());
    }

    /// Records that `stmt` violates `rule`, with a specific explanation.
    fn flag_because(&mut self, rule: Rule, stmt: &Stmt, why: &str) {
        self.note(rule, stmt.line, format!("`{}`: {why}", stmt.text));
    }

    /// Records a violation of `rule` on template line `line`.
    fn note(&mut self, rule: Rule, line: usize, text: String) {
        self.0.push(LintError { rule, line, text });
    }
}

/// Replaces `{{` / `}}` with literal braces and placeholders with
/// [`OPERAND`], reporting malformed braces and FP/SIMD modifiers.
fn normalise(template: &str, found: &mut Findings) -> String {
    let chars: Vec<char> = template.chars().collect();
    let mut out = String::with_capacity(template.len());
    let mut line = 1;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !template_char(c) {
            let text = format!("{c:?} in line {line}");
            found.note(Rule::ForbiddenCharacter, line, text);
        }
        match (c, chars.get(i + 1).copied()) {
            ('{', Some('{')) | ('}', Some('}')) => {
                out.push(c);
                i += 2;
                continue;
            }
            ('{', _) => {
                let end = chars[i + 1..]
                    .iter()
                    .position(|&ch| matches!(ch, '{' | '}' | '\n'))
                    .map(|offset| i + 1 + offset)
                    .filter(|&end| chars[end] == '}');
                let Some(end) = end else {
                    found.note(Rule::MalformedBraces, line, "unclosed `{`".to_owned());
                    i += 1;
                    continue;
                };
                let inner: String = chars[i + 1..end].iter().collect();
                let (name, modifier) = inner.split_once(':').unwrap_or((&inner, ""));
                if !name
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                {
                    let text = format!("`{{{inner}}}` is not a placeholder");
                    found.note(Rule::MalformedBraces, line, text);
                } else if !matches!(modifier, "" | "w" | "x") {
                    let text = format!("`{{{inner}}}` prints an FP/SIMD register");
                    found.note(Rule::PlaceholderModifier, line, text);
                }
                let glued_before = out.chars().last().is_some_and(identifier_char);
                let glued_after = match (chars.get(end + 1), chars.get(end + 2)) {
                    (Some(&'{'), Some(&'{')) => false,
                    (Some(&'{'), _) => true,
                    (Some(&next), _) => identifier_char(next),
                    (None, _) => false,
                };
                if glued_before || glued_after {
                    let text = format!("`{{{inner}}}` touches an identifier or placeholder");
                    found.note(Rule::GluedPlaceholder, line, text);
                }
                out.push_str(OPERAND);
                i = end + 1;
                continue;
            }
            ('}', _) => found.note(Rule::MalformedBraces, line, "unmatched `}`".to_owned()),
            ('\n', _) => {
                line += 1;
                out.push(c);
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Whether `c` may appear in a template at all.
///
/// The alphabet of the instructions the backend writes. Everything else is
/// rejected before parsing, because the assembler gives some characters a
/// meaning the line-based lint cannot see: `\r` separates statements on
/// every target and `%%` on Mach-O (so `nop\rsmstop` would hide a `smstop`),
/// other control characters and non-ASCII text have no use. `;` and `*` are
/// part of the alphabet only so that the `;` and comment rules report them.
fn template_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, ' ' | '\n' | '_' | '.' | ',' | '#' | ':')
        || matches!(c, '[' | ']' | '{' | '}' | '-' | '+' | '/' | '!' | ';' | '*')
}

/// A character that can continue an identifier or a register name.
fn identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The placeholders of a template as `(line, text between the braces)`,
/// skipping `{{` / `}}`. Malformed braces are left to [`normalise`].
fn placeholders(template: &str) -> Vec<(usize, String)> {
    placeholder_sites(template)
        .into_iter()
        .map(|(line, inner, _)| (line, inner))
        .collect()
}

/// [`placeholders`] with, for each, whether it is written as an immediate:
/// preceded (spaces aside) by `#` or `#-`.
fn placeholder_sites(template: &str) -> Vec<(usize, String, bool)> {
    let mut found = Vec::new();
    for (index, line) in template.lines().enumerate() {
        let mut rest = line;
        while let Some(at) = rest.find(['{', '}']) {
            let tail = &rest[at..];
            if tail.starts_with("{{") || tail.starts_with("}}") {
                rest = &tail[2..];
                continue;
            }
            if let Some(stripped) = tail.strip_prefix('{')
                && let Some(end) = stripped.find('}')
                && !stripped[..end].contains('{')
            {
                let before = line[..line.len() - rest.len() + at].trim_end();
                let immediate = before.ends_with('#') || before.ends_with("#-");
                found.push((index + 1, stripped[..end].to_owned(), immediate));
                rest = &stripped[end + 1..];
            } else {
                rest = &tail[1..];
            }
        }
    }
    found
}

/// Splits a normalised template into statements, reporting separators,
/// comments and named labels.
fn parse(normalised: &str, found: &mut Findings) -> Vec<Stmt> {
    let mut stmts = Vec::new();
    for (index, raw) in normalised.lines().enumerate() {
        let line = index + 1;
        let mut rest = raw.trim().to_ascii_lowercase();
        if rest.is_empty() {
            continue;
        }
        let whole = Stmt {
            line,
            text: rest.clone(),
            item: Item::Directive(String::new()),
        };
        if rest.contains(';') {
            found.flag(Rule::Semicolon, &whole);
        }
        if ["//", "/*", "*/"].iter().any(|c| rest.contains(c)) || rest.starts_with('#') {
            found.flag(Rule::Comment, &whole);
        }
        // A label may share its line with an instruction.
        while let Some((head, tail)) = rest.split_once(':') {
            let head = head.trim();
            if head.is_empty() || head.contains(char::is_whitespace) || head.contains('[') {
                break;
            }
            let label = Stmt {
                line,
                text: format!("{head}:"),
                item: Item::Label(head.to_owned()),
            };
            if !head.chars().all(|ch| ch.is_ascii_digit()) {
                found.flag(Rule::NamedLabel, &label);
            }
            stmts.push(label);
            rest = tail.trim().to_owned();
        }
        if rest.is_empty() {
            continue;
        }
        let item = if rest.starts_with('.') {
            Item::Directive(rest.clone())
        } else {
            let (mnemonic, operands) = rest
                .split_once(char::is_whitespace)
                .unwrap_or((rest.as_str(), ""));
            Item::Instr {
                mnemonic: mnemonic.to_owned(),
                operands: operands.trim().to_owned(),
            }
        };
        stmts.push(Stmt {
            line,
            text: rest,
            item,
        });
    }
    stmts
}

/// Splits operands into identifier-like tokens (`z0.s` gives `z0` and `s`).
fn tokens(operands: &str) -> impl Iterator<Item = &str> {
    operands
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| !token.is_empty())
}

/// One of `prefixes` followed by one or more decimal digits.
fn is_numbered(token: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| {
        token
            .strip_prefix(prefix)
            .is_some_and(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
    })
}

/// An Advanced SIMD or scalar FP register name.
fn is_advsimd_or_fp_register(token: &str) -> bool {
    is_numbered(token, &["v", "q", "d", "s", "h", "b"])
}

/// An SVE or SME register name: `z0`, `p0`, `pn8`, `za`, `za0`, `za0h`,
/// `za3v`, `zt0`.
fn is_sve_or_sme_register(token: &str) -> bool {
    if is_numbered(token, &["z", "p", "pn", "zt"]) || token == "za" {
        return true;
    }
    token.strip_prefix("za").is_some_and(|tile| {
        let digits = tile.trim_end_matches(['h', 'v']);
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
    })
}

/// The memory operands of an instruction: top-level `[ … ]` groups that
/// start an operand, as opposed to selectors such as `za.s[w8, 0]` or
/// `z4.s[0]`.
fn memory_operands(operands: &str) -> Vec<&str> {
    let mut groups = Vec::new();
    let mut depth = 0_usize;
    let mut start = None;
    for (i, ch) in operands.char_indices() {
        if ch == '[' {
            let previous = operands[..i].trim_end().chars().last();
            if depth == 0 && matches!(previous, None | Some(',')) {
                start = Some(i + 1);
            }
            depth += 1;
        } else if ch == ']' {
            depth = depth.saturating_sub(1);
            if depth == 0
                && let Some(begin) = start.take()
            {
                groups.push(&operands[begin..i]);
            }
        }
    }
    groups
}

/// Whether a mnemonic is a branch to a label.
fn is_branch(mnemonic: &str) -> bool {
    matches!(mnemonic, "b" | "cbz" | "cbnz" | "tbz" | "tbnz") || mnemonic.starts_with("b.")
}

/// Why a mnemonic is rejected, if it is.
fn rejection(mnemonic: &str) -> Option<&'static str> {
    REJECTED
        .iter()
        .find(|(_, names)| names.contains(&mnemonic))
        .or_else(|| {
            REJECTED_PREFIXES
                .iter()
                .find(|(_, prefixes)| prefixes.iter().any(|prefix| mnemonic.starts_with(prefix)))
        })
        .map(|(why, _)| *why)
}

/// Whether a label name consists only of the digits 0 and 1, which the
/// assembler can read as a binary literal.
fn binary_looking(label: &str) -> bool {
    !label.is_empty() && label.chars().all(|c| c == '0' || c == '1')
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/// Runs every check for a template of the given kind.
fn lint(template: &str, kind: Kind) -> Result<(), Vec<LintError>> {
    let mut found = Findings::default();
    let normalised = normalise(template, &mut found);
    let stmts = parse(&normalised, &mut found);
    check_directives(&stmts, kind, &mut found);
    for stmt in &stmts {
        if let Item::Instr { operands, .. } = &stmt.item {
            check_operands(stmt, operands, &mut found);
        }
    }
    match kind {
        Kind::Streaming => check_streaming(&stmts, &mut found),
        Kind::Probe => check_probe(&stmts, &mut found),
        Kind::FpsrAccess => check_fpsr_access(&stmts, &mut found),
    }
    if found.0.is_empty() {
        Ok(())
    } else {
        Err(found.0)
    }
}

/// Directives: only leading `.arch_extension` with a known extension, and the
/// one the template kind needs must be there.
fn check_directives(stmts: &[Stmt], kind: Kind, found: &mut Findings) {
    let mut seen_code = false;
    let mut extensions = Vec::new();
    for stmt in stmts {
        let Item::Directive(text) = &stmt.item else {
            seen_code = true;
            continue;
        };
        if seen_code {
            found.flag(Rule::DirectiveNotLeading, stmt);
        }
        match text.split_whitespace().collect::<Vec<_>>().as_slice() {
            [".arch_extension", name] if ARCH_EXTENSIONS.contains(name) => {
                extensions.push((*name).to_owned());
            }
            _ => found.flag(Rule::ForbiddenDirective, stmt),
        }
    }
    let has = |name: &str| extensions.iter().any(|e| e == name);
    let present = match kind {
        Kind::Streaming => has("sme2"),
        Kind::Probe => has("sme") || has("sme2"),
        // FPSR is base A64: an accessor needs no extension and enables none.
        Kind::FpsrAccess => {
            for stmt in stmts {
                if matches!(stmt.item, Item::Directive(_)) {
                    found.flag(Rule::FpsrAccess, stmt);
                }
            }
            true
        }
    };
    if !present {
        let rule = Rule::MissingArchExtension;
        found.note(rule, 1, rule.why().to_owned());
    }
}

/// Register and addressing rules, for every instruction.
fn check_operands(stmt: &Stmt, operands: &str, found: &mut Findings) {
    for token in tokens(operands) {
        if is_advsimd_or_fp_register(token) || token == "ffr" {
            found.flag(Rule::AdvSimdOrFfrRegister, stmt);
        }
        if RESERVED_REGISTERS.contains(&token) {
            found.flag(Rule::ReservedRegister, stmt);
        }
    }
    let vector_addressed = |memory: &str| tokens(memory).any(|token| is_numbered(token, &["z"]));
    if memory_operands(operands).into_iter().any(vector_addressed) {
        found.flag(Rule::VectorAddressing, stmt);
    }
}

/// Mnemonic allowlist and reject list for a streaming template.
fn check_mnemonic(stmt: &Stmt, mnemonic: &str, found: &mut Findings) {
    let allowed = match mnemonic.strip_prefix("b.") {
        Some(condition) => CONDITIONS.contains(&condition),
        None => FAMILIES.iter().any(|family| family.contains(&mnemonic)),
    };
    if let Some(why) = rejection(mnemonic) {
        found.flag_because(Rule::RejectedMnemonic, stmt, why);
    } else if !allowed {
        found.flag(Rule::UnknownMnemonic, stmt);
    }
}

/// The FPSR frame lines of a streaming template (`asm.rs`), as
/// `(mnemonic, operands without spaces)`.
const FRAME_SAVE: (&str, &str) = ("mrs", "x17,fpsr");
/// Clears FPSR right after `smstart`.
const FRAME_CLEAR: (&str, &str) = ("msr", "fpsr,xzr");
/// Reads the kernel's flags right before `smstop`.
const FRAME_READ: (&str, &str) = ("mrs", "x16,fpsr");
/// Merges the caller's and the kernel's flags after `smstop`.
const FRAME_MERGE: (&str, &str) = ("orr", "x17,x17,x16");
/// Writes the merged flags back: the last statement.
const FRAME_RESTORE: (&str, &str) = ("msr", "fpsr,x17");
/// The frame's exit label, right before [`FRAME_READ`].
const EXIT_LABEL: &str = "90";

/// Whether `stmt` is the instruction `line`.
fn is_line(stmt: Option<&Stmt>, line: (&str, &str)) -> bool {
    stmt.is_some_and(|stmt| match &stmt.item {
        Item::Instr { mnemonic, operands } => {
            let compact: String = operands.chars().filter(|c| !c.is_whitespace()).collect();
            mnemonic == line.0 && compact == line.1
        }
        _ => false,
    })
}

/// Structure of a streaming template: the region and the FPSR frame.
fn check_streaming(stmts: &[Stmt], found: &mut Findings) {
    let mut starts = Vec::new();
    let mut stops = Vec::new();
    for (index, stmt) in stmts.iter().enumerate() {
        let Item::Instr { mnemonic, operands } = &stmt.item else {
            continue;
        };
        match mnemonic.as_str() {
            "smstart" | "smstop" if !operands.is_empty() => {
                // `smstart sm` / `smstop za`: the region needs SM and ZA together.
                found.flag(Rule::SmstartSmstop, stmt);
            }
            "smstart" => starts.push(index),
            "smstop" => stops.push(index),
            _ => {}
        }
    }
    let region = match (starts.as_slice(), stops.as_slice()) {
        ([start], [stop]) if start < stop => Some((*start, *stop)),
        _ => {
            let (n_start, n_stop) = (starts.len(), stops.len());
            let text = format!("{n_start} plain smstart, {n_stop} plain smstop, or wrong order");
            found.note(Rule::SmstartSmstop, 1, text);
            None
        }
    };
    let frame = region.map_or_else(Vec::new, |(start, stop)| {
        check_frame(stmts, start, stop, found)
    });
    for (index, stmt) in stmts.iter().enumerate() {
        let Item::Instr { mnemonic, operands } = &stmt.item else {
            continue;
        };
        if frame.contains(&index) || matches!(mnemonic.as_str(), "smstart" | "smstop") {
            continue;
        }
        check_mnemonic(stmt, mnemonic, found);
        if tokens(operands).any(|token| token == "x17" || token == "w17") {
            found.flag(Rule::X17, stmt);
        }
        let through_args = |memory: &str| tokens(memory).next() == Some("x0");
        if mnemonic.starts_with("st") && memory_operands(operands).into_iter().any(through_args) {
            found.flag(Rule::StoreToArgs, stmt);
        }
    }
    let Some((start, stop)) = region else {
        return;
    };
    for (index, stmt) in stmts[..start].iter().enumerate() {
        match &stmt.item {
            Item::Label(_) => found.flag(Rule::LabelOutsideRegion, stmt),
            Item::Instr { mnemonic, operands } if !frame.contains(&index) => {
                let base = BASE_A64.contains(&mnemonic.as_str()) && !is_branch(mnemonic);
                if !base || tokens(operands).any(is_sve_or_sme_register) {
                    found.flag(Rule::Prologue, stmt);
                }
            }
            _ => {}
        }
    }
    for (offset, stmt) in stmts[stop + 1..].iter().enumerate() {
        let frame_line = frame.contains(&(stop + 1 + offset));
        if !frame_line && !matches!(stmt.item, Item::Directive(_)) {
            found.flag(Rule::AfterSmstop, stmt);
        }
    }
    check_labels(stmts, start, stop, found);
}

/// The FPSR frame around the region `start..=stop`: reports what is missing
/// or misplaced and returns the indices of the frame lines that are in place
/// (the only `mrs` / `msr` and `x17` a template may contain).
fn check_frame(stmts: &[Stmt], start: usize, stop: usize, found: &mut Findings) -> Vec<usize> {
    let mut frame = Vec::new();
    let mut expect =
        |index: Option<usize>, line: (&str, &str), place: &str, found: &mut Findings| match index
            .filter(|&index| is_line(stmts.get(index), line))
        {
            Some(index) => frame.push(index),
            None => {
                let rule = Rule::FpsrFrame;
                let at = index
                    .and_then(|index| stmts.get(index))
                    .map_or(1, |stmt| stmt.line);
                let text = format!("`{} {}` must be {place}", line.0, line.1);
                found.note(rule, at, text);
            }
        };
    let first_instruction = stmts
        .iter()
        .position(|stmt| matches!(stmt.item, Item::Instr { .. }));
    expect(
        first_instruction,
        FRAME_SAVE,
        "the first instruction",
        found,
    );
    expect(
        Some(start + 1),
        FRAME_CLEAR,
        "the statement right after smstart",
        found,
    );
    expect(
        stop.checked_sub(1),
        FRAME_READ,
        "the statement right before smstop",
        found,
    );
    expect(
        Some(stop + 1),
        FRAME_MERGE,
        "the statement right after smstop",
        found,
    );
    expect(Some(stop + 2), FRAME_RESTORE, "the last statement", found);

    let exit = stop.checked_sub(2).and_then(|index| stmts.get(index));
    let exit_in_place =
        exit.is_some_and(|stmt| matches!(&stmt.item, Item::Label(l) if l == EXIT_LABEL));
    if !exit_in_place {
        let rule = Rule::ExitLabel;
        found.note(
            rule,
            exit.map_or(1, |stmt| stmt.line),
            rule.why().to_owned(),
        );
    }
    let exits = stmts
        .iter()
        .filter(|stmt| matches!(&stmt.item, Item::Label(l) if l == EXIT_LABEL))
        .collect::<Vec<_>>();
    if exits.len() > 1 {
        for stmt in exits {
            found.flag(Rule::ExitLabel, stmt);
        }
    }
    frame
}

/// Label definitions and branch targets of a streaming template.
fn check_labels(stmts: &[Stmt], start: usize, stop: usize, found: &mut Findings) {
    let defined_in = |name: &str, range: std::ops::Range<usize>| {
        stmts[range]
            .iter()
            .any(|stmt| matches!(&stmt.item, Item::Label(label) if label == name))
    };
    for (index, stmt) in stmts.iter().enumerate() {
        match &stmt.item {
            Item::Label(label) => {
                if binary_looking(label) {
                    found.flag(Rule::BinaryLookingLabel, stmt);
                }
                if index < start || index > stop {
                    found.flag(Rule::LabelOutsideRegion, stmt);
                }
            }
            Item::Instr { mnemonic, operands } if is_branch(mnemonic) => {
                let target = operands.rsplit(',').next().unwrap_or("").trim();
                let cut = target.char_indices().last().map_or(0, |(at, _)| at);
                let (name, direction) = target.split_at(cut);
                let numeric = !name.is_empty() && name.chars().all(|c| c.is_ascii_digit());
                let resolves = match direction {
                    "b" => defined_in(name, 0..index),
                    "f" => defined_in(name, index + 1..stmts.len()),
                    _ => false,
                };
                if !numeric || !resolves {
                    found.flag(Rule::BranchTarget, stmt);
                } else if binary_looking(name) {
                    found.flag(Rule::BinaryLookingLabel, stmt);
                }
            }
            _ => {}
        }
    }
}

/// An FPSR accessor is exactly one `mrs {x}, fpsr` or `msr fpsr, {x}`.
fn check_fpsr_access(stmts: &[Stmt], found: &mut Findings) {
    let instructions = stmts
        .iter()
        .filter(|stmt| !matches!(stmt.item, Item::Directive(_)))
        .collect::<Vec<_>>();
    if instructions.len() != 1 {
        let text = format!("{} statements instead of one", instructions.len());
        found.note(Rule::FpsrAccess, 1, text);
    }
    let read = ("mrs", "operand,fpsr");
    let write = ("msr", "fpsr,operand");
    for stmt in instructions {
        if !is_line(Some(stmt), read) && !is_line(Some(stmt), write) {
            found.flag(Rule::FpsrAccess, stmt);
        }
    }
}

/// The probe may only read SVCR / TPIDR2_EL0 and the streaming vector length.
fn check_probe(stmts: &[Stmt], found: &mut Findings) {
    for stmt in stmts {
        let allowed = match &stmt.item {
            Item::Directive(_) => true,
            Item::Label(_) => false,
            Item::Instr { mnemonic, operands } => match mnemonic.as_str() {
                "mrs" => matches!(
                    operands.rsplit(',').next().map(str::trim),
                    Some("svcr" | "tpidr2_el0")
                ),
                "rdsvl" => true,
                _ => false,
            },
        };
        if !allowed {
            found.flag(Rule::ProbeInstruction, stmt);
        }
    }
}
