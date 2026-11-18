//! The kernel texts that the path model of the self-test must reject: the
//! registered templates of both macro-kernels, edited one structural feature
//! at a time, must fail `model_matches_text` (`tests_selftest_text.rs`) with
//! the message of that feature, and harmless reorderings must not.
//!
//! The edits of the plain paths (among them the audit's E2 to E11) and those
//! of the online paths: the switch to the buffer, the packing
//! twins, their dispatch and stores, the prefetch block (the mutants of the
//! kernel lanes that no result can see among them), and branches the model
//! does not know. Everything here is plain Rust and runs on every host.

use super::tests_selftest_paths::{F32_PATHS, F64_PATHS, kernel_template};
use super::tests_selftest_text::model_matches_text;

/// Whether a template line is a numeric label (`42:`).
fn is_label(line: &str) -> bool {
    line.trim()
        .strip_suffix(':')
        .is_some_and(|label| !label.is_empty() && label.chars().all(|c| c.is_ascii_digit()))
}

/// The index of the first line at or after `from` that is `needle`
/// (trimmed).
fn line_at(lines: &[String], needle: &str, from: usize) -> usize {
    let found = lines
        .iter()
        .skip(from)
        .position(|line| line.trim() == needle);
    let Some(offset) = found else {
        panic!("no line {needle:?} at or after line {from}");
    };
    from + offset
}

/// The lines of the section of the label line `label`: from it up to the
/// next label line.
fn section_span(lines: &[String], label: &str) -> core::ops::Range<usize> {
    let start = line_at(lines, label, 0);
    let end = lines
        .iter()
        .skip(start + 1)
        .position(|line| is_label(line))
        .map_or(lines.len(), |offset| start + 1 + offset);
    start..end
}

/// `new` inserted before line `at`.
fn insert_at(lines: &mut Vec<String>, at: usize, new: &[&str]) {
    lines.splice(at..at, new.iter().map(|line| (*line).to_string()));
}

/// The first line `old` at or after the label line `label`, replaced by
/// `new`.
fn replace_after(lines: &mut [String], label: &str, old: &str, new: &str) {
    let at = line_at(lines, old, line_at(lines, label, 0));
    if let Some(line) = lines.get_mut(at) {
        *line = new.to_string();
    }
}

/// The lines from the label line `from` up to the label line `to`, taken
/// out.
fn cut(lines: &mut Vec<String>, from: &str, to: &str) -> Vec<String> {
    let start = line_at(lines, from, 0);
    let end = line_at(lines, to, start);
    lines.drain(start..end).collect()
}

/// The lines from the label line `from` up to the label line `to`, twice.
fn twice(lines: &mut Vec<String>, from: &str, to: &str) {
    let start = line_at(lines, from, 0);
    let end = line_at(lines, to, start);
    let copy: Vec<String> = lines[start..end].to_vec();
    lines.splice(start..start, copy);
}

/// The lines of the section `label` that contain `needle`, removed.
fn drop_in_section(lines: &mut Vec<String>, label: &str, needle: &str) {
    let span = section_span(lines, label);
    let mut at = 0;
    lines.retain(|line| {
        let keep = !(span.contains(&at) && line.contains(needle));
        at += 1;
        keep
    });
}

/// The steps of the depth loop `label` written out twice (its last two
/// lines are the counter update and the branch back).
fn pass_twice(lines: &mut Vec<String>, label: &str) {
    let span = section_span(lines, label);
    let steps: Vec<String> = lines[span.start + 1..span.end - 2].to_vec();
    lines.splice(span.end - 2..span.end - 2, steps);
}

/// An edit of a kernel template: its name, the element type (`true` for
/// `f64`), the edit, and a piece of the message the comparison with the
/// model must give.
type TextEdit = (&'static str, bool, TextChange, &'static str);

/// A change of a template's lines.
type TextChange = fn(&mut Vec<String>);

/// One [`TextEdit`] (a function argument, so a closure becomes a `fn`).
fn edit(name: &'static str, f64: bool, needle: &'static str, change: TextChange) -> TextEdit {
    (name, f64, change, needle)
}

/// The lines between the label lines `from` and `to` that contain `needle`,
/// removed.
fn drop_between(lines: &mut Vec<String>, from: &str, to: &str, needle: &str) {
    let span = line_at(lines, from, 0)..line_at(lines, to, 0);
    let mut at = 0;
    lines.retain(|line| {
        let keep = !(span.contains(&at) && line.contains(needle));
        at += 1;
        keep
    });
}

/// The `f32` epilogue dispatch with a fifth mode (label 49) that M3 enters,
/// its section placed before the label line `before`.
fn fifth_mode_f32(lines: &mut Vec<String>, before: &str) {
    let at = line_at(lines, "b 54f", 0);
    lines[at] = "b.eq 54f".to_string();
    insert_at(lines, at + 1, &["b 49f"]);
    let at = line_at(lines, before, 0);
    insert_at(
        lines,
        at,
        &["49:", "st1w {{za0h.s[w12, 0]}}, p1, [x13]", "b 59f"],
    );
}

/// Every depth loop of the `f32` text unrolled by three, the pass count by
/// `udiv` (the audit's E3b).
fn unrolled_by_3_f32(lines: &mut Vec<String>) {
    for (label, tiles) in [("42:", 1), ("44:", 2), ("46:", 3), ("48:", 4)] {
        let at = line_at(lines, "add x15, x15, #128", line_at(lines, label, 0));
        lines[at] = "add x15, x15, #192".to_string();
        let mut extra = vec!["ld1w {{z10.s}}, p0/z, [x15, #2, mul vl]".to_string()];
        for tile in 0..tiles {
            extra.push(format!("fmopa za{tile}.s, p0/m, p0/m, z10.s, z{tile}.s"));
        }
        extra.push("add x14, x14, x2".to_string());
        lines.splice(at..at, extra);
    }
    let at = line_at(lines, "lsr x13, x22, #1", 0);
    lines[at] = "mov x4, #3".to_string();
    insert_at(lines, at + 1, &["udiv x13, x22, x4"]);
}

/// Structural edits of both templates (`false`: `f32`, `true`: `f64`),
/// among them the audit's E2 to E11 (E2b, E5b, E6, E10 and E11 were silent
/// before) and its K29 as text.
#[rustfmt::skip]
fn text_edits() -> Vec<TextEdit> {
    vec![
        edit("a variant removed", false, "row variants", |l| {
            cut(l, "43:", "45:");
        }),
        edit("a variant removed", true, "row variants", |l| {
            cut(l, "42:", "43:");
        }),
        edit("a variant twice", false, "row variants", |l| twice(l, "43:", "45:")),
        edit("a variant twice", true, "row variants", |l| twice(l, "42:", "43:")),
        edit("a B step doubled", false, "B pointer advances 256 bytes",
            |l| replace_after(l, "44:", "add x15, x15, #128", "add x15, x15, #256")),
        edit("a B step doubled", true, "B pointer advances 256 bytes",
            |l| replace_after(l, "63:", "add x15, x15, #128", "add x15, x15, #256")),
        edit("a pass written twice", false, "4 depth steps per pass", |l| pass_twice(l, "44:")),
        edit("a pass written twice", true, "4 depth steps per pass", |l| pass_twice(l, "63:")),
        edit("a remainder step removed", false, "remainder",
            |l| drop_in_section(l, "63:", "fmopa ")),
        edit("a remainder step removed", true, "remainder",
            |l| drop_in_section(l, "74:", "fmopa ")),
        edit("M1's chain one tile short", false, "the M1 epilogue's tile chain is [0, 1, 2]",
            |l| drop_between(l, "52:", "53:", "za3h")),
        edit("M1's chain one tile short", true,
            "the M1 epilogue's tile chain is [0, 1, 2, 3, 4, 5, 6]",
            |l| drop_between(l, "52:", "53:", "za7h")),
        // E2: a second remainder step in the remainder section.
        edit("E2", false, "(label 43) makes 2 depth steps", |l| {
            let at = line_at(l, "b 50f", line_at(l, "62:", 0));
            insert_at(l, at, &[
                "ld1w {{z8.s}}, p0/z, [x15, #1, mul vl]",
                "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
                "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            ]);
        }),
        // E2b: a second remainder step in a labelled section of its own.
        edit("E2b", false, "section 65 has 2 outer products outside", |l| {
            let at = line_at(l, "b 50f", line_at(l, "62:", 0));
            insert_at(l, at, &[
                "add x15, x15, #64", "65:", "tbz x22, #1, 50f", "ld1w {{z8.s}}, p0/z, [x15]",
                "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            ]);
        }),
        edit("E2b", true, "section 79 has 2 outer products outside", |l| {
            let at = line_at(l, "b 50f", line_at(l, "72:", 0));
            insert_at(l, at, &[
                "79:", "tbz x22, #1, 50f", "ld1d {{z8.d}}, p0/z, [x15, #1, mul vl]",
                "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            ]);
        }),
        // E3: one loop unrolled by three; E3b: all of them, with `udiv`.
        edit("E3", false, "3 depth steps per pass, but the pass count is kc >> 1", |l| {
            let at = line_at(l, "add x15, x15, #128", line_at(l, "42:", 0));
            l[at] = "add x15, x15, #192".to_string();
            insert_at(l, at, &[
                "ld1w {{z10.s}}, p0/z, [x15, #2, mul vl]", "ld1w {{z11.s}}, p1/z, [x14]",
                "fmopa za0.s, p0/m, p0/m, z10.s, z11.s", "add x14, x14, x2",
            ]);
        }),
        edit("E3b", false, "0 `lsr` of kc", unrolled_by_3_f32),
        // E4: the M1 epilogue visits tile 1 twice.
        edit("E4", false, "the M1 epilogue's tile chain is [0, 1, 1, 2, 3]", |l| {
            let at = line_at(l, "mova z16.s, p0/m, za1h.s[w12, 0]", line_at(l, "52:", 0));
            let copy: Vec<String> = l[at..at + 4].to_vec();
            l.splice(at + 4..at + 4, copy);
        }),
        // E5a / E5b: a fifth epilogue mode after M3, or before M0.
        edit("E5a", false, "the four modes at [\"51\", \"52\", \"53\", \"49\"]",
            |l| fifth_mode_f32(l, "59:")),
        edit("E5b", false, "entered at [\"49\", \"51\", \"52\", \"53\", \"54\"]",
            |l| fifth_mode_f32(l, "51:")),
        edit("E5b", true, "entered at [\"49\", \"51\", \"52\", \"53\", \"54\"]", |l| {
            let at = line_at(l, "b.eq 54f", 0);
            insert_at(l, at + 1, &["cmp x25, #4", "b.hs 49f"]);
            let at = line_at(l, "51:", 0);
            insert_at(l, at, &["49:", "st1d {{za0h.d[w12, 0]}}, p1, [x15]", "b 59f"]);
        }),
        // E6: column panels of another width.
        edit("E6", false, "column panels of 32 columns in the text", |l| {
            replace_after(l, "20:", "mov x16, #16", "mov x16, #32");
            replace_after(l, "59:", "add x26, x26, #16", "add x26, x26, #32");
        }),
        edit("E6", true, "column panels of 16 columns in the text",
            |l| replace_after(l, "59:", "add x26, x26, #8", "add x26, x26, #16")),
        // E10: a variant threshold moved off the tile height, or onto another.
        // E10 / E10b on the plain dispatch (label 40; the row-panel head 30
        // is followed by the prefetch chain, which compares the row count
        // as well: see `online_text_edits`).
        edit("E10", false, "40 is not a multiple of the tile height 16",
            |l| replace_after(l, "40:", "cmp x23, #48", "cmp x23, #40")),
        edit("E10", true, "60 is not a multiple of the tile height 8",
            |l| replace_after(l, "40:", "cmp x13, #56", "cmp x13, #60")),
        edit("E10b", false,
            "sends rows 33..=48 of a row panel to the variant of 4 tiles, the model to the \
             variant of 3",
            |l| replace_after(l, "40:", "cmp x23, #48", "cmp x23, #32")),
        edit("E10b", true,
            "sends rows 33..=40 of a row panel to the variant of 6 tiles, the model to the \
             variant of 5",
            |l| replace_after(l, "40:", "cmp x13, #40", "cmp x13, #32")),
        // E11: an edge path in the epilogue dispatch.
        edit("E11", false, "the epilogue dispatch tests `cmp x24, #16`, not only the mode (x25)",
            |l| insert_at(l, line_at(l, "50:", 0) + 1, &["cmp x24, #16", "b.lo 59f"])),
        edit("E11", true, "the epilogue dispatch tests `cmp x9, #8`, not only the mode (x25)",
            |l| insert_at(l, line_at(l, "50:", 0) + 1, &["cmp x9, #8", "b.lo 59f"])),
        // K29 as text: a column loop that ignores the live columns.
        edit("K29", false,
            "the M0 epilogue has 1 column loop(s) and reads the live columns (x24) 0 times",
            |l| replace_after(l, "51:", "cmp w12, w24", "cmp w12, #16")),
        edit("K29", true,
            "the M0 epilogue has 8 column loop(s) and reads the live columns (x9) 7 times",
            |l| replace_after(l, "51:", "mov x16, x9", "mov x16, #8")),
        edit("row panels of 32 rows", false, "row panels step 32 rows",
            |l| replace_after(l, "59:", "add x9, x9, #64", "add x9, x9, #32")),
        edit("row panels of 32 rows", true, "row panels step 32 rows",
            |l| replace_after(l, "59:", "add x10, x10, #64", "add x10, x10, #32")),
        edit("C steps 8 columns", false, "C advances 8 columns per column panel",
            |l| replace_after(l, "59:", "add x28, x28, x7, lsl #4", "add x28, x28, x7, lsl #3")),
        edit("the live columns clamped to 32", false, "the live columns are clamped to 32",
            |l| replace_after(l, "20:", "mov x16, #16", "mov x16, #32")),
        edit("a bit test of kc outside a remainder", false,
            "`tbnz x22, #1, 50f` (section 41) is a depth step conditional on kc",
            |l| insert_at(l, line_at(l, "cbz x13, 61f", 0) + 1, &["tbnz x22, #1, 50f"])),
        // A variant moved behind the last one: the last remainder now falls
        // into it instead of the epilogue dispatch.
        edit("a variant moved to the end", false, "leave to 2 different sections", |l| {
            let moved = cut(l, "43:", "45:");
            let at = line_at(l, "50:", 0);
            l.splice(at..at, moved);
        }),
    ]
}

/// `template` with `change` applied.
fn edited(template: &str, change: TextChange) -> String {
    let mut lines: Vec<String> = template.lines().map(str::to_string).collect();
    change(&mut lines);
    lines.join("\n")
}

#[test]
fn test_sme2_selftest_paths_text_changes_are_seen() {
    let templates = [
        (&F32_PATHS, kernel_template::<f32>()),
        (&F64_PATHS, kernel_template::<f64>()),
    ];
    for (desc, template) in templates {
        assert_eq!(model_matches_text(desc, template), Ok(()));
    }
    let edits = text_edits();
    for (name, f64, change, needle) in &edits {
        let (desc, template) = templates[usize::from(*f64)];
        let text = edited(template, *change);
        assert_ne!(text, template, "{name}: nothing edited");
        match model_matches_text(desc, &text) {
            Ok(()) => panic!("{name} ({}) is not seen", desc_name(*f64)),
            Err(error) => {
                println!("{name}: {error}");
                assert!(error.contains(needle), "{name}: {error}");
            }
        }
    }
    // Every silent edit of the audit is here for each type it applies to.
    for (name, types) in [("E2b", 2), ("E5b", 2), ("E6", 2), ("E10", 2), ("E11", 2)] {
        let count = edits.iter().filter(|edit| edit.0 == name).count();
        assert_eq!(count, types, "{name}");
    }
    // A harmless reordering is not a change of structure: two variants'
    // sections swapped (each ends with a branch to the epilogue dispatch).
    let swaps: [(bool, TextChange); 2] = [
        (false, |l| {
            let moved = cut(l, "43:", "45:");
            let at = line_at(l, "47:", 0);
            l.splice(at..at, moved);
        }),
        (true, |l| {
            let moved = cut(l, "42:", "43:");
            let at = line_at(l, "44:", 0);
            l.splice(at..at, moved);
        }),
    ];
    for (f64, change) in swaps {
        let (desc, template) = templates[usize::from(f64)];
        let text = edited(template, change);
        assert_ne!(text, template);
        assert_eq!(
            model_matches_text(desc, &text),
            Ok(()),
            "{}",
            desc_name(f64)
        );
    }
}

/// The element type's name for a message.
fn desc_name(f64: bool) -> &'static str {
    if f64 { "f64" } else { "f32" }
}

/// The index of the one line that is `needle` (trimmed).
fn unique(lines: &[String], needle: &str) -> usize {
    let found: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim() == needle)
        .map(|(at, _)| at)
        .collect();
    let [at] = found[..] else {
        panic!("{needle:?} occurs {} times", found.len());
    };
    at
}

/// The one line `old` removed.
fn drop_line(lines: &mut Vec<String>, old: &str) {
    let at = unique(lines, old);
    lines.remove(at);
}

/// The one line `old` replaced by `new`.
fn replace_line(lines: &mut [String], old: &str, new: &str) {
    let at = unique(lines, old);
    if let Some(line) = lines.get_mut(at) {
        *line = new.to_string();
    }
}

/// `new` inserted after the first line `after`.
fn insert_after(lines: &mut Vec<String>, after: &str, new: &[&str]) {
    let at = line_at(lines, after, 0);
    insert_at(lines, at + 1, new);
}

/// Structural edits of the online paths of both templates: the kernel
/// lanes' mutants that no result can see (K5 / F5 the switch to the buffer
/// left out, K7 / F7, K8 / F8, K11 / F11 and F16 in the prefetch block, K9 /
/// F9 a twin that zeroes dead tiles), a packing store into C's register,
/// the other mutants of the switch and the buffer advance, the packing
/// dispatch, and branches the model does not know.
#[rustfmt::skip]
fn online_text_edits() -> Vec<TextEdit> {
    vec![
        // The switch to the buffer after column panel 0.
        edit("K5", false, "the switch to the buffer is", |l| {
            for old in ["mov x1, x8", "mov x2, #256", "lsl x3, x22, #8"] {
                drop_line(l, old);
            }
        }),
        edit("F5", true, "the switch to the buffer is", |l| {
            for old in ["mov x1, x8", "mov x2, #512", "lsl x3, x22, #9"] {
                drop_line(l, old);
            }
        }),
        edit("K2", false, "missing [\"mov x8, #0\"]", |l| drop_line(l, "mov x8, #0")),
        edit("F2", true, "missing [\"mov x8, #0\"]", |l| drop_line(l, "mov x8, #0")),
        edit("K3", false, "missing [\"mov x2, #256\"]",
            |l| replace_line(l, "mov x2, #256", "mov x2, #128")),
        edit("F3", true, "missing [\"mov x2, #512\"]",
            |l| replace_line(l, "mov x2, #512", "mov x2, #256")),
        edit("K13", false, "missing [\"lsl x3, x22, #8\"]",
            |l| replace_line(l, "lsl x3, x22, #8", "lsl x3, x22, #7")),
        edit("F13", true, "missing [\"lsl x3, x22, #9\"]",
            |l| replace_line(l, "lsl x3, x22, #9", "lsl x3, x22, #8")),
        edit("the gate set again", false, "writes a register of A or the gate outside the switch",
            |l| insert_after(l, "20:", &["mov x8, x6"])),
        // The packing twins: a store into C's register, the buffer advance,
        // the zeroing of dead tiles.
        edit("a packing store into C's register", false,
            "stores `st1w {z0.s}, p0, [x11]`, not through the buffer register x6",
            |l| replace_after(l, "72:", "st1w {{z0.s}}, p0, [x6]", "st1w {{z0.s}}, p0, [x11]")),
        edit("a packing store into C's register", true,
            "stores `st1d {z0.d-z3.d}, pn8, [x28]`, not through the buffer register x6",
            |l| replace_after(l, "161:", "st1d {{z0.d-z3.d}}, pn8, [x6]",
                "st1d {{z0.d-z3.d}}, pn8, [x28]")),
        edit("K10", false, "by 512 bytes per pass and 0 in its remainder",
            |l| drop_in_section(l, "68:", "add x6, x6, #256")),
        edit("F10", true, "by 1024 bytes per pass and 0 in its remainder",
            |l| drop_in_section(l, "178:", "add x6, x6, #512")),
        edit("F15", true, "by 512 bytes per pass and 512 in its remainder",
            |l| replace_after(l, "165:", "add x6, x6, #1024", "add x6, x6, #512")),
        edit("the buffer reset per column panel", false, "writes the buffer register x6",
            |l| insert_after(l, "20:", &["mov x6, x1"])),
        edit("the buffer one depth step late", false,
            "does not start a buffer register from it",
            |l| replace_line(l, "mov x6, x8", "add x6, x8, #256")),
        edit("the buffer one depth step late", true,
            "does not start a buffer register from it",
            |l| replace_line(l, "mov x6, x8", "add x6, x8, #512")),
        edit("K9", false, "zeroes tiles", |l| replace_after(l, "71:", "zero {{za0.s}}", "zero {{za}}")),
        edit("F9", true, "zeroes tiles", |l| replace_after(l, "141:", "zero {{za0.d}}", "zero {{za}}")),
        // The packing dispatch.
        edit("a twin threshold", false,
            "the packing dispatch compares the row count with [40, 32, 16]: 40 is not a multiple",
            |l| replace_after(l, "39:", "cmp x23, #48", "cmp x23, #40")),
        edit("a twin threshold", true,
            "the packing dispatch compares the row count with [8, 16, 24, 32, 40, 48, 60]: 60 is \
             not a multiple",
            |l| replace_after(l, "39:", "cmp x13, #56", "cmp x13, #60")),
        edit("a plain variant in the packing pass", false,
            "the packing dispatch sends rows 49..=64 of a row panel to no packing twin",
            |l| replace_after(l, "39:", "b.hi 77f", "b.hi 47f")),
        // Without the gate the packing dispatch becomes the plain one and
        // reaches the twins.
        edit("no gate", true,
            "the variant dispatch sends rows 1..=8 of a row panel to no variant",
            |l| drop_line(l, "cbz x8, 40f")),
        // The prefetch block.
        edit("K8", false, "no prefetch block in the text", |l| drop_line(l, "tbz x4, #0, 39f")),
        edit("F8", true, "no prefetch block in the text", |l| drop_line(l, "tbz x4, #0, 39f")),
        edit("K7", false, "the prefetch block hints at",
            |l| replace_after(l, "33:", "b.lo 32b", "b.ls 32b")),
        edit("F7", true, "the prefetch block hints at",
            |l| replace_after(l, "30:", "mov x16, x9", "add x16, x9, #1")),
        edit("K11", false, "the prefetch block hints at",
            |l| replace_after(l, "32:", "cmp x23, #16", "cmp x23, #8")),
        edit("F11", true, "the prefetch block hints at",
            |l| replace_after(l, "32:", "cmp x13, #16", "cmp x13, #8")),
        edit("a chain threshold", false, "the prefetch block hints at",
            |l| replace_after(l, "32:", "cmp x23, #48", "cmp x23, #40")),
        edit("F16", true, "the prefetch block hints at",
            |l| replace_line(l, "prfm pldl2keep, [x14, #448]", "prfm pldl2keep, [x14, #512]")),
        edit("the row count clobbered", false, "leaves the row count",
            |l| insert_after(l, "33:", &["mov x23, #64"])),
        // Branches the model does not know: paths the self-test does not take.
        edit("a branch in a twin", false, "is a branch the model does not know",
            |l| insert_after(l, "72:", &["cbz x1, 50f"])),
        edit("a branch in a twin", true, "is a branch the model does not know",
            |l| insert_after(l, "161:", &["cbz x1, 50f"])),
        edit("a reserved prefetch bit", false, "is a branch the model does not know",
            |l| insert_after(l, "tbz x4, #0, 39f", &["tbnz x4, #1, 39f"])),
        edit("a reserved prefetch bit", true, "is a branch the model does not know",
            |l| insert_after(l, "tbz x4, #0, 39f", &["tbnz x4, #1, 39f"])),
        edit("a branch in an epilogue", true, "is a branch the model does not know",
            |l| insert_after(l, "52:", &["cbz x22, 59f"])),
        edit("a branch in the column-panel head", false, "is a branch the model does not know",
            |l| insert_after(l, "20:", &["cbz x21, 29f"])),
        // Paths beyond the row counts the self-test reaches (its blocks have
        // at most 192 rows, so a row panel has at most 192 rows left): a
        // compare of the row count with a number that is no tile threshold,
        // a back branch that is not a column loop of the kernels' two forms,
        // a tile chain's compare with m on a row that is not a tile's first
        // row, a conditional select. Each changes nothing in the self-test's
        // blocks, and the model accepted each before its accounting knew
        // only those forms.
        edit("the prefetch skipped beyond 200 rows", false, "with 201 rows",
            |l| insert_after(l, "32:", &["cmp x23, #200", "b.hi 39f"])),
        edit("the prefetch skipped beyond 200 rows", true, "with 201 rows",
            |l| insert_after(l, "32:", &["cmp x13, #200", "b.hi 39f"])),
        edit("a prefetch column restarted beyond 200 rows", false, "does not end",
            |l| insert_after(l, "prfm pldl2keep, [x13, #192]", &["cmp x23, #200", "b.hi 32b"])),
        edit("a prefetch column restarted beyond 200 rows", true, "does not end",
            |l| insert_after(l, "prfm pldl2keep, [x14, #448]", &["cmp x13, #200", "b.hi 32b"])),
        edit("A clobbered in the prefetch block beyond 200 rows", false,
            "is a branch the model does not know",
            |l| insert_after(l, "prfm pldl2keep, [x13, #192]",
                &["cmp x23, #200", "b.ls 33f", "mov x10, x1"])),
        edit("A clobbered in the prefetch block beyond 200 rows", true,
            "is a branch the model does not know",
            |l| insert_after(l, "prfm pldl2keep, [x14, #448]",
                &["cmp x13, #200", "b.ls 33f", "mov x11, x1"])),
        edit("a back branch in the prefetch block", false, "is a branch the model does not know",
            |l| insert_after(l, "prfm pldl2keep, [x13, #192]", &["cmp x13, x13", "b.ne 32b"])),
        edit("a back branch in the prefetch block", true, "is a branch the model does not know",
            |l| insert_after(l, "prfm pldl2keep, [x14, #448]", &["cmp x14, x14", "b.ne 32b"])),
        edit("an epilogue skipped beyond 200 rows", false, "is a branch the model does not know",
            |l| insert_after(l, "51:", &["cmp x23, #200", "b.hi 55f"])),
        edit("an epilogue skipped beyond 200 rows", true, "is a branch the model does not know",
            |l| insert_after(l, "51:", &["cmp x13, #200", "b.hi 59f"])),
        edit("an epilogue skipped when i0 + 200 < m", true, "is a branch the model does not know",
            |l| insert_after(l, "51:", &["add x16, x10, #200", "cmp x16, x20", "b.lo 59f"])),
        edit("an epilogue loop that runs on beyond 200 rows", false,
            "is a branch the model does not know",
            |l| insert_after(l, "51:", &["add x0, x24, #200", "cmp x23, x0", "b.hi 51b"])),
        edit("an epilogue loop that runs on beyond 200 rows", true,
            "is a branch the model does not know",
            |l| insert_after(l, "51:", &["add x16, x9, #200", "cmp x13, x16", "b.hi 51b"])),
        edit("an epilogue loop on the row count less 200 against the live columns", false,
            "is a branch the model does not know",
            |l| insert_after(l, "51:", &["sub x0, x23, #200", "cmp x0, x24", "b.gt 51b"])),
        edit("C's column chosen by a conditional select beyond 200 rows", false,
            "is a branch the model does not know",
            |l| insert_after(l, "51:", &["cmp x23, #200", "csel x13, x13, x11, ls"])),
    ]
}

#[test]
fn test_sme2_selftest_paths_online_text_changes_are_seen() {
    let templates = [
        (&F32_PATHS, kernel_template::<f32>()),
        (&F64_PATHS, kernel_template::<f64>()),
    ];
    let edits = online_text_edits();
    for (name, f64, change, needle) in &edits {
        let (desc, template) = templates[usize::from(*f64)];
        let text = edited(template, *change);
        assert_ne!(text, template, "{name}: nothing edited");
        match model_matches_text(desc, &text) {
            Ok(()) => panic!("{name} ({}) is not seen", desc_name(*f64)),
            Err(error) => {
                println!("{name}: {error}");
                assert!(error.contains(needle), "{name}: {error}");
            }
        }
    }
    // The kernel lanes' mutants that no result can see, for each type.
    for (f32_name, f64_name) in [
        ("K5", "F5"),
        ("K7", "F7"),
        ("K8", "F8"),
        ("K9", "F9"),
        ("K11", "F11"),
    ] {
        for (name, f64) in [(f32_name, false), (f64_name, true)] {
            assert!(
                edits.iter().any(|edit| edit.0 == name && edit.1 == f64),
                "{name}"
            );
        }
    }
    assert!(edits.iter().any(|edit| edit.0 == "F16"));
    // A harmless reordering of the packing twins is not a change: a twin's
    // sections moved behind another (each ends with a branch to the
    // epilogue dispatch).
    let swaps: [(bool, TextChange); 2] = [
        (false, |l| {
            let moved = cut(l, "73:", "75:");
            let at = line_at(l, "77:", 0);
            l.splice(at..at, moved);
        }),
        (true, |l| {
            let moved = cut(l, "142:", "143:");
            let at = line_at(l, "144:", 0);
            l.splice(at..at, moved);
        }),
    ];
    for (f64, change) in swaps {
        let (desc, template) = templates[usize::from(f64)];
        let text = edited(template, change);
        assert_ne!(text, template);
        assert_eq!(
            model_matches_text(desc, &text),
            Ok(()),
            "{}",
            desc_name(f64)
        );
    }
}
