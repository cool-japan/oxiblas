//! Tests of the SME2 GEMV (`sme2.rs`), part 3: the structure of the
//! registered templates (run everywhere). No result test can see a
//! forbidden read (it feeds lanes that are never stored), so these tests
//! parse the registered templates and prove which predicate governs every
//! load and store. `sme2_tests.rs` holds the shared harness and the other
//! tests that run everywhere.

use super::sme2::SME_KERNELS;

// ---------------------------------------------------------------------------
// Structure of the registered templates (run everywhere)
// ---------------------------------------------------------------------------

/// One statement of a template body.
#[derive(Debug, Clone)]
struct Stmt {
    /// A label definition (`"20"`), or `None` for an instruction.
    label: Option<String>,
    /// Lower-case mnemonic (empty for a label).
    mnemonic: String,
    /// Operands split at top-level commas.
    ops: Vec<String>,
    /// The text.
    text: String,
}

/// Splits operands at commas outside `[ ]` and `{ }`.
fn split_operands(text: &str) -> Vec<String> {
    let mut ops = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in text.chars() {
        match ch {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ => {}
        }
        if ch == ',' && depth == 0 {
            ops.push(current.trim().to_owned());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    if !current.trim().is_empty() {
        ops.push(current.trim().to_owned());
    }
    ops
}

/// The statements of a template (braces unescaped), and the index of
/// the streaming-mode entry and of the frame's exit label.
fn statements(template: &str) -> (Vec<Stmt>, usize, usize) {
    let text = template.replace("{{", "{").replace("}}", "}");
    let stmts: Vec<Stmt> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('.'))
        .map(|line| {
            if let Some(label) = line.strip_suffix(':') {
                Stmt {
                    label: Some(label.to_owned()),
                    mnemonic: String::new(),
                    ops: Vec::new(),
                    text: line.to_owned(),
                }
            } else {
                let (mnemonic, rest) = line.split_once(' ').unwrap_or((line, ""));
                Stmt {
                    label: None,
                    mnemonic: mnemonic.to_ascii_lowercase(),
                    ops: split_operands(rest),
                    text: line.to_owned(),
                }
            }
        })
        .collect();
    // The frame's first body line follows the streaming-mode entry (a
    // test file may not spell the entry itself; see the source scan).
    let start = stmts
        .iter()
        .position(|s| s.text == "msr fpsr, xzr")
        .and_then(|clear| clear.checked_sub(1))
        .expect("streaming entry");
    let exit = stmts
        .iter()
        .position(|s| s.label.as_deref() == Some("90"))
        .expect("exit label");
    (stmts, start, exit)
}

/// The register name of an operand (`p1.s` gives `p1`, `pn9/z` gives
/// `pn9`, `x5` gives `x5`).
fn reg(op: &str) -> &str {
    op.split(['.', '/']).next().unwrap_or(op).trim()
}

/// Whether `stmt` writes the general-purpose register `x` (or its `w`
/// half).
fn writes_gpr(stmt: &Stmt, x: &str) -> bool {
    const WRITERS: [&str; 10] = [
        "add", "sub", "mov", "lsl", "lsr", "and", "csel", "mul", "ldr", "orr",
    ];
    let w = x.replacen('x', "w", 1);
    WRITERS.contains(&stmt.mnemonic.as_str())
        && stmt.ops.first().is_some_and(|op| op == x || *op == w)
}

/// Whether `stmt` writes the predicate register `p`.
fn writes_pred(stmt: &Stmt, p: &str) -> bool {
    const WRITERS: [&str; 6] = ["whilelt", "and", "bic", "ptrue", "pfalse", "eor"];
    WRITERS.contains(&stmt.mnemonic.as_str()) && stmt.ops.first().is_some_and(|op| reg(op) == p)
}

/// A memory access of a template body.
#[derive(Debug)]
struct Access {
    /// Statement index.
    at: usize,
    /// The governing predicate.
    pred: String,
    /// Base register.
    base: String,
    /// Index register.
    index: String,
    /// Shift of the index.
    shift: u32,
}

/// The SVE / SME memory accesses of a body (`ld1*`, `st1*`).
fn accesses(stmts: &[Stmt], start: usize, exit: usize, label: &str) -> Vec<Access> {
    let mut found = Vec::new();
    for (at, stmt) in stmts.iter().enumerate().take(exit).skip(start) {
        if !(stmt.mnemonic.starts_with("ld1") || stmt.mnemonic.starts_with("st1")) {
            assert!(
                !(stmt.mnemonic.starts_with("ld")
                    || stmt.mnemonic.starts_with("st")
                    || stmt.mnemonic.starts_with("prf")),
                "{label}: unexpected memory instruction `{}`",
                stmt.text
            );
            continue;
        }
        assert_eq!(stmt.ops.len(), 3, "{label}: `{}`", stmt.text);
        let pred = reg(&stmt.ops[1]).to_owned();
        if stmt.mnemonic.starts_with("ld1") {
            assert!(
                stmt.ops[1].ends_with("/z"),
                "{label}: load not zeroing: `{}`",
                stmt.text
            );
        }
        let address = stmt.ops[2].trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = address.split(',').map(str::trim).collect();
        assert_eq!(parts.len(), 3, "{label}: address of `{}`", stmt.text);
        let shift = parts[2]
            .strip_prefix("lsl #")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("{label}: shift of `{}`", stmt.text));
        found.push(Access {
            at,
            pred,
            base: parts[0].to_owned(),
            index: parts[1].to_owned(),
            shift,
        });
    }
    found
}

/// Proves that every access of `entry` is bounded: its predicate is (a
/// self-narrowing `and` of) `whilelt P, I, L` with `I` the access's index
/// register and `L` one of the limits that `limits` allows for the base
/// register, defined in the same straight-line run (no label between) with
/// `I` unchanged in between; the shift matches the element size. So the
/// access touches only elements `base[I + l]` with `I + l < L`. Returns
/// the bases seen.
fn assert_accesses_bounded(
    name: &str,
    limits: &dyn Fn(&str) -> Vec<&'static str>,
    shift: u32,
) -> Vec<String> {
    let entry = SME_KERNELS.iter().find(|e| e.name == name).expect("kernel");
    let (stmts, start, exit) = statements(entry.template);
    let mut bases = Vec::new();
    for access in accesses(&stmts, start, exit, name) {
        let text = &stmts[access.at].text;
        assert_eq!(access.shift, shift, "{name}: shift of `{text}`");
        let allowed = limits(&access.base);
        assert!(!allowed.is_empty(), "{name}: unknown base in `{text}`");
        // The definition of the predicate, through self-narrowing ANDs.
        let mut at = access.at;
        let def = loop {
            let found = (start..at)
                .rev()
                .find(|&k| writes_pred(&stmts[k], &access.pred));
            let k =
                found.unwrap_or_else(|| panic!("{name}: `{text}`: {} never defined", access.pred));
            for between in &stmts[k + 1..access.at] {
                assert!(
                    between.label.is_none(),
                    "{name}: `{text}`: label between the predicate and its use"
                );
            }
            let s = &stmts[k];
            if s.mnemonic == "and" {
                let me = format!("{}.b", access.pred);
                assert!(
                    s.ops.len() == 4 && s.ops[0] == me && s.ops[2] == me && s.ops[3] == me,
                    "{name}: `{}` is not a narrowing of {}",
                    s.text,
                    access.pred
                );
                at = k;
                continue;
            }
            break k;
        };
        let s = &stmts[def];
        assert_eq!(
            s.mnemonic, "whilelt",
            "{name}: `{text}`: predicate from `{}`",
            s.text
        );
        assert!(s.ops.len() >= 3, "{name}: `{}`", s.text);
        // The predicate counts elements of the access's size (a `.b` or
        // `.h` counter would make more lanes of a `.s` / `.d` access live).
        let size = if shift == 2 { ".s" } else { ".d" };
        assert!(
            s.ops[0].ends_with(size),
            "{name}: `{text}`: `{}` does not count {size} elements",
            s.text
        );
        assert_eq!(
            s.ops[1], access.index,
            "{name}: `{text}`: index differs from `{}`",
            s.text
        );
        assert!(
            allowed.contains(&s.ops[2].as_str()),
            "{name}: `{text}`: limit {} of `{}` not allowed for base {}",
            s.ops[2],
            s.text,
            access.base
        );
        for between in &stmts[def + 1..access.at] {
            assert!(
                !writes_gpr(between, &access.index),
                "{name}: `{text}`: index changed by `{}`",
                between.text
            );
        }
        bases.push(access.base.clone());
    }
    bases.sort();
    bases.dedup();
    bases
}

/// The body statements of `name`.
fn body_of(name: &str) -> Vec<Stmt> {
    let entry = SME_KERNELS.iter().find(|e| e.name == name).expect("kernel");
    let (stmts, start, exit) = statements(entry.template);
    stmts[start + 1..exit].to_vec()
}

/// The prologue loads of `name`: register -> constant.
fn prologue_loads(name: &str) -> Vec<(String, String)> {
    let entry = SME_KERNELS.iter().find(|e| e.name == name).expect("kernel");
    let (stmts, start, _) = statements(entry.template);
    stmts[..start]
        .iter()
        .filter(|s| s.mnemonic == "ldr")
        .map(|s| (s.ops[0].clone(), s.ops[1].clone()))
        .collect()
}

#[test]
fn test_sme2_gemv_kernels_registered() {
    let names: Vec<(&str, bool)> = SME_KERNELS.iter().map(|e| (e.name, e.f64f64)).collect();
    assert_eq!(
        names,
        vec![
            ("sme2_gemv_notrans_f32", false),
            ("sme2_gemv_notrans_f64", true),
            ("sme2_gemv_trans_f32", false),
            ("sme2_gemv_trans_f64", true),
        ]
    );
    for entry in SME_KERNELS {
        assert!(entry.file.ends_with("gemv/sme2.rs"), "{}", entry.file);
        let loads = prologue_loads(entry.name);
        let has = |reg: &str, konst: &str| loads.iter().any(|(r, c)| r == reg && c.contains(konst));
        assert!(
            has("x1", "off_a") && has("x2", "off_lda") && has("x3", "off_x"),
            "{}",
            entry.name
        );
        assert!(
            has("x4", "off_out") && has("x5", "off_m") && has("x6", "off_n"),
            "{}",
            entry.name
        );
        // m and n are never written in the body.
        for stmt in body_of(entry.name) {
            assert!(
                !writes_gpr(&stmt, "x5") && !writes_gpr(&stmt, "x6"),
                "{}: `{}` writes m or n",
                entry.name,
                stmt.text
            );
        }
    }
}

#[test]
fn test_sme2_gemv_template_notrans_accesses_bounded() {
    // A columns and y against m (x5), x against n (x6).
    let limits = |base: &str| -> Vec<&'static str> {
        match base {
            "x4" | "x14" | "x20" | "x21" | "x22" => vec!["x5"],
            "x3" => vec!["x6"],
            _ => Vec::new(),
        }
    };
    let bases = assert_accesses_bounded("sme2_gemv_notrans_f32", &limits, 2);
    assert_eq!(bases, ["x14", "x20", "x21", "x22", "x3", "x4"]);
    let bases = assert_accesses_bounded("sme2_gemv_notrans_f64", &limits, 3);
    assert_eq!(bases, ["x14", "x20", "x3", "x4"]);
    // The column pointers start at A(0, ·): x14 = a, then lda apart.
    for (name, cols) in [("sme2_gemv_notrans_f32", 4), ("sme2_gemv_notrans_f64", 2)] {
        let body = body_of(name);
        let texts: Vec<&str> = body.iter().map(|s| s.text.as_str()).collect();
        assert!(texts.contains(&"mov x14, x1"), "{name}");
        assert!(texts.contains(&"add x20, x14, x2"), "{name}");
        if cols == 4 {
            assert!(
                texts.contains(&"add x21, x20, x2") && texts.contains(&"add x22, x21, x2"),
                "{name}"
            );
            assert!(texts.contains(&"add x14, x14, x2, lsl #2"), "{name}");
        } else {
            assert!(texts.contains(&"add x14, x14, x2, lsl #1"), "{name}");
        }
    }
}

#[test]
fn test_sme2_gemv_template_notrans_column_order() {
    // Every FMLA uses the vectors loaded just before from the column whose
    // lane it names, in the order 0, 1, 2, 3 (columns ascending), and the
    // indexed vector z8 is always alpha · x (fmul by z28 after each ld1rq).
    for (name, lanes, ptrs) in [
        (
            "sme2_gemv_notrans_f32",
            4usize,
            ["x14", "x20", "x21", "x22"],
        ),
        (
            "sme2_gemv_notrans_f64",
            2usize,
            ["x14", "x20", "x21", "x22"],
        ),
    ] {
        let body = body_of(name);
        let mut previous_lane: Option<usize> = None;
        let mut fmla = 0;
        for (k, stmt) in body.iter().enumerate() {
            if stmt.mnemonic.starts_with("ld1rq") {
                let next = &body[k + 1];
                assert_eq!(next.mnemonic, "fmul", "{name}: after `{}`", stmt.text);
                assert_eq!(reg(&next.ops[0]), "z8", "{name}");
                assert!(
                    next.ops.iter().skip(1).any(|op| reg(op) == "z28"),
                    "{name}: `{}`",
                    next.text
                );
                previous_lane = None;
            }
            if stmt.mnemonic == "fmla" {
                fmla += 1;
                let lane: usize = stmt.ops[2]
                    .split_once('[')
                    .and_then(|(_, rest)| rest.trim_end_matches(']').parse().ok())
                    .expect("lane");
                assert!(lane < lanes, "{name}: `{}`", stmt.text);
                assert!(stmt.ops[2].starts_with("z8."), "{name}: `{}`", stmt.text);
                let load = &body[k - 1];
                assert!(
                    load.mnemonic.starts_with("ld1"),
                    "{name}: before `{}`",
                    stmt.text
                );
                assert_eq!(
                    load.ops[0], stmt.ops[1],
                    "{name}: `{}` after `{}`",
                    stmt.text, load.text
                );
                assert!(
                    load.ops[2].starts_with(&format!("[{},", ptrs[lane])),
                    "{name}: lane {lane} from `{}`",
                    load.text
                );
                match previous_lane {
                    None => assert_eq!(lane, 0, "{name}: first lane"),
                    Some(p) => assert_eq!(
                        lane,
                        (p + 1) % lanes.max(1),
                        "{name}: lane order at `{}`",
                        stmt.text
                    ),
                }
                previous_lane = if lane + 1 == lanes { None } else { Some(lane) };
                assert!(
                    stmt.ops[0].starts_with("za.") && stmt.ops[0].contains("w8, 0, vgx4"),
                    "{name}"
                );
            }
            if stmt.label.is_some() && stmt.label.as_deref() != Some("42") {
                previous_lane = None;
            }
        }
        assert_eq!(fmla, if lanes == 4 { 7 } else { 3 }, "{name}: FMLA count");
        // alpha's broadcast comes from the alpha field.
        let loads = prologue_loads(name);
        let alpha_reg = loads
            .iter()
            .find(|(_, c)| c.contains("off_alpha"))
            .expect("alpha load")
            .0
            .clone();
        assert!(
            body.iter()
                .any(|s| s.mnemonic == "dup" && reg(&s.ops[0]) == "z28" && s.ops[1] == alpha_reg),
            "{name}"
        );
    }
}

#[test]
fn test_sme2_gemv_template_trans_accesses_bounded() {
    // x against m; each column pointer against its own limit register;
    // the partials against x7 (16 n or 8 n).
    let limits = |base: &str| -> Vec<&'static str> {
        match base {
            "x3" => vec!["x5"],
            "x11" => vec!["x12"],
            "x24" => vec!["x13"],
            "x25" => vec!["x14"],
            "x26" => vec!["x15"],
            "x4" => vec!["x7"],
            _ => Vec::new(),
        }
    };
    for (name, shift, lanes_shift) in [("sme2_gemv_trans_f32", 2, 4), ("sme2_gemv_trans_f64", 3, 3)]
    {
        let bases = assert_accesses_bounded(name, &limits, shift);
        assert_eq!(bases, ["x11", "x24", "x25", "x26", "x3", "x4"], "{name}");
        let body = body_of(name);
        let texts: Vec<&str> = body.iter().map(|s| s.text.as_str()).collect();
        // The limits are m for a live column (4g + k < nb), else 0, and
        // nothing else writes them.
        let group = texts
            .iter()
            .position(|t| *t == "lsl x16, x8, #2")
            .expect("group start");
        let want = [
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "csel x12, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x13, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x14, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x15, x5, xzr, lo",
            "add x24, x11, x2",
            "add x25, x24, x2",
            "add x26, x25, x2",
        ];
        assert_eq!(
            &texts[group..group + want.len()],
            want,
            "{name}: group prologue"
        );
        for limit in ["x12", "x13", "x14", "x15"] {
            let writers = body.iter().filter(|s| writes_gpr(s, limit)).count();
            assert_eq!(writers, 1, "{name}: {limit} written elsewhere");
        }
        // nb = min(64, n - jb), the block's columns start at A(0, jb), the
        // group pointer advances by four columns, and the partials limit
        // is n · lanes, written once, before the first store.
        for line in [
            "sub x10, x6, x9",
            "mov x16, #64",
            "cmp x10, x16",
            "csel x10, x10, x16, lo",
            "mul x11, x9, x2",
            "add x11, x1, x11",
            "add x11, x11, x2, lsl #2",
            "add x9, x9, #64",
        ] {
            assert_eq!(
                texts.iter().filter(|t| **t == line).count(),
                1,
                "{name}: `{line}`"
            );
        }
        let end = format!("lsl x7, x6, #{lanes_shift}");
        let end_at = texts
            .iter()
            .position(|t| *t == end)
            .expect("partials limit");
        let x7_writes: Vec<usize> = body
            .iter()
            .enumerate()
            .filter(|(_, s)| writes_gpr(s, "x7"))
            .map(|(k, _)| k)
            .collect();
        assert_eq!(
            *x7_writes.last().expect("x7"),
            end_at,
            "{name}: x7 after the limit"
        );
        let first_store = body
            .iter()
            .position(|s| s.mnemonic.starts_with("st1"))
            .expect("store");
        assert!(
            end_at < first_store && end_at < texts.iter().position(|t| *t == "20:").expect("20"),
            "{name}"
        );
        // x10 (nb) and the group counter are not written in the row loops.
        for stmt in body.iter().filter(|s| writes_gpr(s, "x10")) {
            assert!(
                stmt.text.contains("x6, x9") || stmt.text.contains("x10, x10, x16"),
                "{name}: `{}`",
                stmt.text
            );
        }
    }
}

#[test]
fn test_sme2_gemv_template_trans_same_predicate_for_a_and_x() {
    // In every row step the x load and the A loads take predicates built
    // from the same index register, and in the tail both are narrowed by
    // p6 (the lane predicate): a dead lane is zero on both sides of every
    // multiply-add. Every FMLA multiplies four A vectors by the x vector of
    // the same rows: z8 with the first group, z9..z11 with the others.
    for name in ["sme2_gemv_trans_f32", "sme2_gemv_trans_f64"] {
        let body = body_of(name);
        let mut tail_ands = 0;
        for (k, stmt) in body.iter().enumerate() {
            if stmt.mnemonic == "and" && stmt.ops.len() == 4 {
                assert_eq!(reg(&stmt.ops[1]), "p6", "{name}: `{}`", stmt.text);
                tail_ands += 1;
            }
            if stmt.mnemonic == "fmla" {
                let zm = reg(&stmt.ops[2]);
                let first = stmt.ops[1]
                    .trim_start_matches('{')
                    .split('.')
                    .next()
                    .unwrap_or("");
                let want = match first {
                    "z16" => "z8",
                    "z20" => "z9",
                    "z24" => "z10",
                    "z28" => "z11",
                    other => panic!("{name}: group {other}"),
                };
                assert_eq!(zm, want, "{name}: `{}`", stmt.text);
                // The x load of this step uses the same index register as
                // the A loads of the step.
                let step: Vec<&Stmt> = body[..k]
                    .iter()
                    .rev()
                    .take_while(|s| s.label.is_none())
                    .filter(|s| s.mnemonic.starts_with("ld1"))
                    .collect();
                let index_of = |s: &Stmt| s.ops[2].split(',').nth(1).map(|i| i.trim().to_owned());
                let indices: Vec<_> = step.iter().map(|s| index_of(s)).collect();
                assert!(
                    !indices.is_empty() && indices.windows(2).all(|w| w[0] == w[1]),
                    "{name}: `{}`",
                    stmt.text
                );
                assert!(
                    step.iter().any(|s| s.ops[2].starts_with("[x3,")),
                    "{name}: no x load before `{}`",
                    stmt.text
                );
            }
        }
        assert_eq!(tail_ands, 5, "{name}: tail narrowings");
    }
}

#[test]
fn test_sme2_gemv_template_trans_dead_tail_lanes_add_negative_zero() {
    // A dead lane of the vector-tail step must add an exact -0, the only
    // value that leaves every lane unchanged: a lane can be -0 (products
    // that underflow to -0), and +0 · +0 would turn it into +0 where NEON
    // never touches the lane. So after the zeroing x load of the tail, the
    // lanes outside its predicate are negated (+0 to -0), with the
    // complement built from that predicate after its last definition.
    for (name, size) in [("sme2_gemv_trans_f32", "s"), ("sme2_gemv_trans_f64", "d")] {
        let body = body_of(name);
        let load = body
            .iter()
            .position(|s| s.mnemonic.starts_with("ld1") && s.ops[2].starts_with("[x3, x21,"))
            .expect("tail x load");
        let z = format!("z8.{size}");
        assert_eq!(body[load].ops[0], format!("{{{z}}}"), "{name}");
        let pred = reg(&body[load].ops[1]).to_owned();
        let fmla = (load..body.len())
            .find(|&k| body[k].mnemonic == "fmla")
            .expect("tail fmla");
        assert!(body[load..fmla].iter().all(|s| s.label.is_none()), "{name}");
        let negs: Vec<usize> = (load + 1..fmla)
            .filter(|&k| body[k].mnemonic == "fneg")
            .collect();
        assert_eq!(negs.len(), 1, "{name}: dead-lane negation");
        let neg = &body[negs[0]];
        assert!(
            neg.ops.len() == 3 && neg.ops[0] == z && neg.ops[2] == z,
            "{name}: `{}`",
            neg.text
        );
        let dead = neg.ops[1].strip_suffix("/m").expect("merging").to_owned();
        for (k, s) in body.iter().enumerate().take(fmla).skip(load + 1) {
            assert!(
                k == negs[0]
                    || reg(s.ops.first().map_or("", |op| op.trim_start_matches('{'))) != "z8",
                "{name}: `{}`",
                s.text
            );
        }
        let pred_def = (0..load)
            .rev()
            .find(|&k| writes_pred(&body[k], &pred))
            .expect("pred");
        let def = (0..negs[0])
            .rev()
            .find(|&k| writes_pred(&body[k], &dead))
            .expect("dead");
        assert!(
            def > pred_def && body[def..negs[0]].iter().all(|s| s.label.is_none()),
            "{name}"
        );
        assert_eq!(
            body[def].text,
            format!("eor {dead}.b, p0/z, {pred}.b, p0.b"),
            "{name}"
        );
        let p0: Vec<&str> = body
            .iter()
            .filter(|s| writes_pred(s, "p0"))
            .map(|s| s.text.as_str())
            .collect();
        assert_eq!(p0, [format!("ptrue p0.{size}")], "{name}");
    }
}
