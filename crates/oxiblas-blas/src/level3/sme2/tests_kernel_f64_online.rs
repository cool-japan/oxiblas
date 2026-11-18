//! Tests of online packing of A and of the C prefetch in the `f64`
//! macro-kernel `sme2_gemm_f64` (`kernel_f64.rs`). The contract
//! is the section "Online packing of A and prefetching" of `args.rs`;
//! `args::reference` defines C and `pack::pack_a` the buffer.
//!
//! # The interpreter (native, every host)
//!
//! What online packing and prefetching must not do is mostly invisible in
//! results (module docs of `test_interpreter_f64.rs`), so the first tests
//! here run the registered template in that interpreter over a grid of
//! blocks, every combination of online packing and prefetch, and require
//! every access, every buffer element, every hint and every tile to keep the
//! contract; mutated templates prove that each rule can fail
//! (`test_sme2_kernel_f64_online_interpreter_rejects_mutants`). The
//! line-based structure checks of `tests_kernel_f64.rs` cover the A loads
//! and the buffer stores of both passes as well.
//!
//! # Result tests (SME2 hardware, or QEMU with `-r`)
//!
//! Each case calls the kernel four times on the same operands: with and
//! without online packing, each with and without `PREFETCH_C`. C must be
//! `args::reference` (bitwise except NaN payloads) and, in all four calls,
//! the same bits; FPSR after the call the same in all four; the buffer,
//! filled with stale values first and between signalling-NaN guards (see
//! [`buffer_tail`]), must hold exactly `pack::pack_a`'s block of A (NaN
//! payloads and signed zeros included) with the guards untouched. A, B and
//! C sit between guards as well, and rows `≥ m` of A (the padding of the
//! packed source, the gaps of in-place A) are signalling NaNs. Every call
//! counts exactly one kernel entry on this thread and leaves SVCR and
//! TPIDR2_EL0 zero. The comparisons are with `reference` and between the
//! four calls only, never with the dispatched micro-kernel (which is the
//! portable one in builds that cap the SIMD tier).

use oxiblas_core::memory::AlignedVec;
use oxiblas_matrix::MatRef;

use super::args::{Epilogue, MR, MacroArgs, PREFETCH_C, PREFETCH_OFF, reference, same_result};
use super::kernel_f64::{SME_KERNELS, sme2_gemm_f64};
use super::lint::lint_streaming_template;
use super::pack::{ColMajor, Span, a_pack_len, pack_a};
use super::test_interpreter_f64::{
    Case, GUARD, LANES, NR, Program, Shape, decode, in_place_lda, interpret,
};
use super::test_support::{
    FPSR_DZC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_UFC, Need, fpsr_read, fpsr_write,
    hardware_or_skip, kernel_entries_here, kernel_ready, sme_state,
};
use crate::level3::trsm::Trans;

// ---------------------------------------------------------------------------
// The interpreter tests
// ---------------------------------------------------------------------------

/// Online packing and prefetch: the four combinations of each test case.
const SWITCHES: [(bool, u64); 4] = [
    (false, PREFETCH_OFF),
    (false, PREFETCH_C),
    (true, PREFETCH_OFF),
    (true, PREFETCH_C),
];

/// The interpreter's grid. Rows: every row variant (1 to 8 live tiles) at
/// its first and last row, two to four row panels; columns: one panel
/// (full or partial) and several with a partial last one; odd and even
/// depths, with and without whole double steps. Per shape one A layout
/// (packed, in place with a small or a large `lda`) and one mode, in
/// rotation, each with the four [`SWITCHES`].
fn interpreter_cases() -> Vec<Case> {
    const MS: [usize; 22] = [
        1, 2, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 63, 64, 65, 72, 129, 193,
    ];
    const NS: [usize; 8] = [1, 2, 7, 8, 9, 16, 17, 33];
    const KCS: [usize; 5] = [1, 2, 3, 4, 7];
    let mut cases = Vec::new();
    let mut index = 0_usize;
    for m in MS {
        for n in NS {
            for kc in KCS {
                let lda = match index % 3 {
                    0 => None,
                    1 => Some(in_place_lda(m, 1)),
                    _ => Some(in_place_lda(m, 70)),
                };
                let shape = Shape {
                    m,
                    n,
                    kc,
                    lda,
                    b_gap: [0, 5, 8][index % 3],
                    ldc_pad: [0, 3, 1][(index / 3) % 3],
                };
                for (pack, prefetch) in SWITCHES {
                    cases.push(Case {
                        shape,
                        mode: (index % 4) as u64,
                        pack,
                        prefetch,
                    });
                }
                index += 1;
            }
        }
    }
    cases
}

/// The registered template, decoded.
fn registered() -> Program {
    match Program::load(SME_KERNELS[0].template) {
        Ok(program) => program,
        Err(error) => panic!("sme2_gemm_f64: {error}"),
    }
}

#[test]
fn test_sme2_kernel_f64_online_interpreter_registered_template() {
    // The kernel as assembled keeps every rule of the module docs of
    // `test_interpreter_f64.rs` in every case of the grid, and each case
    // exercises what it should: tiles, hints exactly when PREFETCH_C is set,
    // whole buffer vectors (eight per depth step) exactly when it packs.
    let program = registered();
    let (mut hinted, mut packed) = (0_usize, 0_usize);
    for case in interpreter_cases() {
        let report = match interpret(&program, case) {
            Ok(report) => report,
            Err(error) => panic!("sme2_gemm_f64, {case:?}: {error}"),
        };
        let s = case.shape;
        assert_eq!(
            report.tiles,
            s.m.div_ceil(MR) * s.n.div_ceil(NR),
            "{case:?}"
        );
        assert_eq!(report.hints > 0, case.prefetch != 0, "{case:?}");
        let want_vectors = if case.pack { s.pack_len() / LANES } else { 0 };
        assert_eq!(report.buffer_vectors, want_vectors, "{case:?}");
        hinted += report.hints;
        packed += report.buffer_vectors;
    }
    assert!(hinted > 0 && packed > 0);
}

/// One edit of a mutant: after the one line equal to `anchor`, the first
/// line equal to `old` is replaced by `new` (no lines: removed). Lines are
/// compared trimmed, with the registry's `{{` / `}}` escapes.
struct Edit {
    /// A line that occurs once (a label).
    anchor: &'static str,
    /// The line to replace.
    old: &'static str,
    /// Its replacement.
    new: &'static [&'static str],
}

/// `template` with `edits` applied in order.
fn mutated(template: &str, edits: &[Edit]) -> String {
    let mut lines: Vec<String> = template.lines().map(str::to_string).collect();
    for edit in edits {
        let anchors: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim() == edit.anchor)
            .map(|(at, _)| at)
            .collect();
        let [anchor] = anchors[..] else {
            panic!("anchor {:?} occurs {} times", edit.anchor, anchors.len());
        };
        let Some(at) = (anchor + 1..lines.len()).find(|at| lines[*at].trim() == edit.old) else {
            panic!("no {:?} after {:?}", edit.old, edit.anchor);
        };
        lines.splice(at..=at, edit.new.iter().map(|line| (*line).to_string()));
    }
    lines.join("\n")
}

/// A wrong template that the interpreter must reject.
struct Mutant {
    /// What is wrong.
    label: &'static str,
    /// How it is made from the registered template.
    edits: &'static [Edit],
    /// The violation the interpreter must report (one of these message
    /// fragments).
    expect: &'static [&'static str],
}

/// The wrong templates of [`test_sme2_kernel_f64_online_interpreter_rejects_mutants`].
const MUTANTS: [Mutant; 19] = [
    Mutant {
        label: "the switch to the buffer left out (later panels read the source again)",
        edits: &[
            Edit {
                anchor: "59:",
                old: "mov x1, x8",
                new: &[],
            },
            Edit {
                anchor: "59:",
                old: "mov x2, #512",
                new: &[],
            },
            Edit {
                anchor: "59:",
                old: "lsl x3, x22, #9",
                new: &[],
            },
        ],
        expect: &["times from the source"],
    },
    Mutant {
        label: "the packing pass also on the later column panels",
        edits: &[Edit {
            anchor: "59:",
            old: "mov x8, #0",
            new: &[],
        }],
        expect: &["guard or gap element of Buffer", "outside every operand"],
    },
    Mutant {
        label: "the depth stride of the buffer wrong for the later panels",
        edits: &[Edit {
            anchor: "59:",
            old: "mov x2, #512",
            new: &["mov x2, #256"],
        }],
        expect: &["accumulator", "buffer row", "times from the source", "read"],
    },
    Mutant {
        label: "the panel stride of the buffer wrong for the later panels",
        edits: &[Edit {
            anchor: "59:",
            old: "lsl x3, x22, #9",
            new: &["lsl x3, x22, #8"],
        }],
        expect: &["accumulator", "buffer row", "before it was written"],
    },
    Mutant {
        label: "the store of rows i0+32.. (no live row there) left out of a double step",
        edits: &[Edit {
            anchor: "162:",
            old: "st1d {{z4.d-z7.d}}, pn8, [x6, #4, mul vl]",
            new: &[],
        }],
        expect: &["written 0 times"],
    },
    Mutant {
        label: "a packing store group one group off",
        edits: &[Edit {
            anchor: "178:",
            old: "st1d {{z4.d-z7.d}}, pn8, [x6, #4, mul vl]",
            new: &["st1d {{z4.d-z7.d}}, pn8, [x6, #8, mul vl]"],
        }],
        expect: &["written", "guard or gap element of Buffer", "accumulator"],
    },
    Mutant {
        label: "the A vectors of the second step stored for the first",
        edits: &[Edit {
            anchor: "166:",
            old: "st1d {{z0.d-z3.d}}, pn8, [x6]",
            new: &["st1d {{z16.d-z19.d}}, pn8, [x6]"],
        }],
        expect: &["accumulator", "holds something else"],
    },
    Mutant {
        label: "a row predicate removed from an A load of the packing pass",
        edits: &[Edit {
            anchor: "164:",
            old: "ld1d {{z0.d-z3.d}}, pn9/z, [x14]",
            new: &["ld1d {{z0.d-z3.d}}, pn8/z, [x14]"],
        }],
        expect: &["guard or gap element of A"],
    },
    Mutant {
        label: "the upper row predicate of the packing pass from the lower rows",
        edits: &[Edit {
            anchor: "176:",
            old: "ld1d {{z4.d-z7.d}}, pn10/z, [x14, #4, mul vl]",
            new: &["ld1d {{z4.d-z7.d}}, pn9/z, [x14, #4, mul vl]"],
        }],
        expect: &["guard or gap element of A"],
    },
    Mutant {
        label: "the remainder of the eight-tile packing pass does not advance the buffer",
        edits: &[Edit {
            anchor: "178:",
            old: "add x6, x6, #512",
            new: &[],
        }],
        expect: &["written", "accumulator", "holds something else"],
    },
    Mutant {
        label: "the packing double step advances the buffer by one step only",
        edits: &[Edit {
            anchor: "165:",
            old: "add x6, x6, #1024",
            new: &["add x6, x6, #512"],
        }],
        expect: &["written", "accumulator", "holds something else"],
    },
    Mutant {
        label: "a packing store under the row counter (the padding left stale)",
        edits: &[Edit {
            anchor: "162:",
            old: "st1d {{z0.d-z3.d}}, pn8, [x6]",
            new: &["st1d {{z0.d-z3.d}}, pn9, [x6]"],
        }],
        expect: &["not whole vectors"],
    },
    Mutant {
        label: "a packing variant zeroes a tile without a live row",
        edits: &[Edit {
            anchor: "141:",
            old: "zero {{za0.d}}",
            new: &["zero {{za}}"],
        }],
        expect: &["zeroed 1 times, want 0"],
    },
    Mutant {
        label: "a packing variant accumulates a tile without a live row",
        edits: &[Edit {
            anchor: "162:",
            old: "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            new: &[
                "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
                "fmopa za2.d, p0/m, p0/m, z8.d, z2.d",
            ],
        }],
        expect: &["outer products"],
    },
    Mutant {
        label: "the packing variants for every panel (no plain dispatch)",
        edits: &[Edit {
            anchor: "39:",
            old: "cbz x8, 40f",
            new: &[],
        }],
        expect: &["outside every operand", "guard or gap element of Buffer"],
    },
    Mutant {
        label: "the prefetch loop one column too far",
        edits: &[Edit {
            anchor: "30:",
            old: "mov x16, x9",
            new: &["add x16, x9, #1"],
        }],
        expect: &["prefetch hint"],
    },
    Mutant {
        label: "the prefetch loop run with the bit clear",
        edits: &[Edit {
            anchor: "30:",
            old: "tbz x4, #0, 39f",
            new: &[],
        }],
        expect: &["prefetch hints"],
    },
    Mutant {
        label: "a prefetch hint below the live rows",
        edits: &[Edit {
            anchor: "32:",
            old: "cmp x13, #16",
            new: &["cmp x13, #8"],
        }],
        expect: &["prefetch hint"],
    },
    Mutant {
        label: "a prefetch hint inside the depth loop",
        edits: &[Edit {
            anchor: "143:",
            old: "cbz x16, 173f",
            new: &["cbz x16, 173f", "prfm pldl2keep, [x28]"],
        }],
        expect: &["inside the depth loop"],
    },
];

/// The block skipped with the bit set: no hints at all.
const SKIPPED_WITH_BIT_SET: Mutant = Mutant {
    label: "the prefetch block skipped with the bit set (no hints)",
    edits: &[Edit {
        anchor: "30:",
        old: "tbz x4, #0, 39f",
        new: &["tbnz x4, #0, 39f"],
    }],
    expect: &["prefetch hints"],
};

#[test]
fn test_sme2_kernel_f64_online_interpreter_rejects_mutants() {
    // Wrong templates that the template lint accepts and that, except for
    // the stride and store mutants, no result test can see: each must be
    // rejected by the interpreter, for its own reason, in at least one case
    // of a grid with one and several row and column panels, every row
    // variant, both A layouts and all four switches.
    let mut cases = Vec::new();
    for (index, m) in [5, 12, 17, 31, 40, 48, 49, 57, 64, 65, 100, 129]
        .into_iter()
        .enumerate()
    {
        for n in [1, 9, 17] {
            for kc in [1, 2, 3] {
                for lda in [None, Some(in_place_lda(m, 3))] {
                    for (pack, prefetch) in SWITCHES {
                        cases.push(Case {
                            shape: Shape {
                                lda,
                                ..Shape::plain(m, n, kc)
                            },
                            mode: ((index + n + kc) % 4) as u64,
                            pack,
                            prefetch,
                        });
                    }
                }
            }
        }
    }
    let registered_text = SME_KERNELS[0].template;
    let original = registered();
    for case in &cases {
        if let Err(error) = interpret(&original, *case) {
            panic!("the registered template fails the mutant grid: {case:?}: {error}");
        }
    }
    for mutant in MUTANTS.iter().chain([&SKIPPED_WITH_BIT_SET]) {
        let text = mutated(registered_text, mutant.edits);
        assert_ne!(text, registered_text, "{}", mutant.label);
        assert!(
            lint_streaming_template(&text).is_ok(),
            "{}: the lint rejects the mutant, so it proves nothing about the interpreter",
            mutant.label
        );
        let program = match Program::load(&text) {
            Ok(program) => program,
            Err(error) => panic!("{}: not decoded: {error}", mutant.label),
        };
        let errors: Vec<String> = cases
            .iter()
            .filter_map(|case| interpret(&program, *case).err())
            .collect();
        assert!(
            errors
                .iter()
                .any(|error| mutant.expect.iter().any(|want| error.contains(want))),
            "{}: not rejected for its own reason ({:?}); errors: {:?}",
            mutant.label,
            mutant.expect,
            errors.first()
        );
    }
}

#[test]
fn test_sme2_kernel_f64_online_interpreter_decodes_exactly() {
    // The decoder of the interpreter fails closed: forms the kernel does not
    // use, and forms the encodings do not have, are errors, not silently
    // different statements.
    let resolve = |_: &str| -> Result<usize, String> { Ok(0) };
    for line in [
        "ld1w {z0.s}, p1/z, [x14]",
        "ld1d {z1.d-z4.d}, pn9/z, [x14]",
        "ld1d {z0.d-z2.d}, pn9/z, [x14]",
        "ld1d {z1.d-z2.d}, pn9/z, [x14]",
        "ld1d {z2.d-z5.d}, pn9/z, [x14]",
        "ld1d {z0.d, z1.d}, pn9/z, [x14]",
        "ld1d {z0.d-z3.d}, pn9/z, [x14, #2, mul vl]",
        "ld1d {z0.d-z3.d}, pn9/z, [x14, #32, mul vl]",
        "ld1d {z0.d-z1.d}, pn9/z, [x14, #16, mul vl]",
        "ld1d {z0.d-z3.d}, p1/z, [x14]",
        "ld1d {z0.d-z3.d}, pn7/z, [x14]",
        "ld1d {z0.d}, pn9/z, [x14]",
        "ld1d {z0.d-z3.d}, pn9/z, [x14, x2, lsl #3]",
        "ld1d {z0.d}, p1/z, [x14, #8, mul vl]",
        "ld1d {z0.d}, p1/z, [x14, #8]",
        "ld1d {z0.d}, p1/z, [x14, x2, lsl #2]",
        "ld1d {z0.d}, p8/z, [x14]",
        "st1d {z0.d-z3.d}, pn9/z, [x6]",
        "st1d {z0.d-z3.d}, p0, [x6]",
        "st1d {z0.d-z3.d}, pn8, [x6, #2, mul vl]",
        "st1d {za0v.d[w12, 0]}, p1, [x15]",
        "st1d {za0h.d[w12, 2]}, p1, [x15]",
        "st1d {za0h.d[w8, 0]}, p1, [x15]",
        "whilelt pn9.d, x10, x20, vlx2",
        "whilelt pn9.d, x10, x20",
        "whilelt p9.d, x10, x20, vlx4",
        "whilelt pn9.s, x10, x20, vlx4",
        "whilelt p1.d, w13, w20",
        "ptrue p0.s",
        "ptrue pn7.d",
        "prfm pstl1keep, [x14]",
        "prfm pldl2keep, [x14, #1, mul vl]",
        "fmopa za8.d, p0/m, p0/m, z8.d, z0.d",
        "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
        "fmopa za0.d, pn8/m, p0/m, z8.d, z0.d",
        "mova z16.d, p1/m, za0v.d[w12, 0]",
        "ldr x1, [x2, #8]",
        "ldr w20, [x0, #8]",
        "dup z28.d, w23",
        "dup z18.d, #0",
        "madd x1, x2, x3, x4",
        "adds x16, x16, #1",
        "fadd z16.d, p1/m, z17.d, z18.d",
    ] {
        assert!(decode(line, resolve).is_err(), "{line:?} decoded");
    }
    let decoded = |line: &str| match decode(line, resolve) {
        Ok(ins) => format!("{ins:?}"),
        Err(error) => panic!("{line:?}: {error}"),
    };
    assert_eq!(
        decoded("st1d {z20.d-z23.d}, pn8, [x6, #12, mul vl]"),
        "Store { zs: 20, count: 4, pg: 8, at: Vl(Gp { n: 6, w: false }, 12) }"
    );
    assert_eq!(
        decoded("ld1d {z4.d-z5.d}, pn10/z, [x14, #4, mul vl]"),
        "Load { zd: 4, count: 2, pg: 10, at: Vl(Gp { n: 14, w: false }, 4) }"
    );
    assert_eq!(
        decoded("ld1d {z0.d}, p2/z, [x14]"),
        "Load { zd: 0, count: 1, pg: 2, at: Plain(Gp { n: 14, w: false }, 0) }"
    );
    assert_eq!(
        decoded("whilelt pn9.d, x10, x20, vlx4"),
        "Whilelt { pd: 9, rn: Gp { n: 10, w: false }, rm: Gp { n: 20, w: false }, counter: \
         true }"
    );
    assert_eq!(decoded("ptrue pn8.d"), "Ptrue { pd: 8, counter: true }");
    assert_eq!(
        decoded("subs x16, x16, #1"),
        "Arith { rd: Gp { n: 16, w: false }, rn: Gp { n: 16, w: false }, src: Imm(1), \
         subtract: true, set_flags: true }"
    );
    assert_eq!(
        decoded("prfm pldl2keep, [x14, #448]"),
        "Prefetch { at: Plain(Gp { n: 14, w: false }, 448) }"
    );
    assert_eq!(
        decoded("st1d {za7h.d[w12, 0]}, p1, [x15]"),
        "StoreSlice { tile: 7, ws: Gp { n: 12, w: true }, off: 0, pg: 1, at: Plain(Gp { n: 15, \
         w: false }, 0) }"
    );
}

// ---------------------------------------------------------------------------
// Result tests
// ---------------------------------------------------------------------------

/// The alpha/beta pairs: M0, M1, M2, M3, M3 with `alpha = 1`, M3 with
/// `beta = 1` (the last three take the three branches of M3).
const PAIRS: [(f64, f64); 6] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (-1.5, 0.0),
    (0.3, 0.7),
    (1.0, -0.5),
    (2.0, 1.0),
];

/// The value outside every operand (guards, gaps, rows `m..ldc` of C, rows
/// `≥ m` of A): a signalling NaN, which no arithmetic returns, so a store
/// there or a border element written back through FADD or FMLA changes its
/// bits.
fn sentinel() -> f64 {
    f64::from_bits(0x7ff0_0000_dead_beef)
}

/// The value inside C when the mode does not read it: must not leak.
fn unread() -> f64 {
    f64::from_bits(0x7ff0_0000_0000_0bad)
}

/// A stale buffer element before the call: a finite number or a
/// signalling NaN with a payload of its own, never a zero, so a missing
/// store of the padding or of a value shows.
fn stale(index: usize) -> f64 {
    if index.is_multiple_of(3) {
        1.0e7 + index as f64
    } else {
        f64::from_bits(0x7ff0_0000_0004_0000 | (index as u64 & 0x3fff))
    }
}

/// Which operand a value is generated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operand {
    /// `A(i, p)`.
    A,
    /// `B(p, j)`.
    B,
    /// `C(i, j)`.
    C,
}

/// Deterministic test values (xorshift).
struct Gen(u64);

impl Gen {
    /// A generator for `seed`.
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// The next 64 random bits.
    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Mixed signs and magnitudes, a few exact zeros.
    fn normal(&mut self) -> f64 {
        let bits = self.bits();
        let unit = (bits >> 11) as f64 / (1_u64 << 53) as f64;
        let scale = [1.0, 1.0e-3, 37.0, 0.0][(bits & 3) as usize];
        (unit * 2.0 - 1.0) * scale
    }

    /// One of `values`.
    fn pick(&mut self, values: &[f64]) -> f64 {
        values[(self.bits() >> 33) as usize % values.len()]
    }
}

/// The bits of a buffer, for exact comparisons.
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The operands of one kernel call in owned buffers, each with [`GUARD`]
/// sentinels before and after it.
struct Operands {
    /// Geometry.
    shape: Shape,
    /// alpha.
    alpha: f64,
    /// beta.
    beta: f64,
    /// A in its layout (gaps hold the sentinel).
    a: Vec<f64>,
    /// Packed B (zero padding columns, sentinel gaps between panels).
    b: Vec<f64>,
    /// C before the call.
    c: Vec<f64>,
    /// `pack::pack_a`'s block of A: what the buffer must hold.
    packed: Vec<f64>,
}

/// What one kernel call left.
struct Outcome {
    /// The C buffer.
    c: Vec<f64>,
    /// The packing buffer (guards included), when the call packed.
    buffer: Option<Vec<f64>>,
    /// FPSR after the call.
    fpsr: u64,
}

impl Operands {
    /// Builds the operands; `value` gives `A(i, p)`, `B(p, j)`, `C(i, j)`.
    fn new(
        shape: Shape,
        alpha: f64,
        beta: f64,
        value: &mut dyn FnMut(Operand, usize, usize) -> f64,
    ) -> Self {
        let Shape { m, n, kc, .. } = shape;
        assert!(m >= 1 && n >= 1 && kc >= 1, "{shape:?}");
        let mut a = vec![sentinel(); GUARD + shape.a_len() + GUARD];
        let mut logical = vec![0.0_f64; m * kc];
        for p in 0..kc {
            for i in 0..m {
                let v = value(Operand::A, i, p);
                a[GUARD + shape.a_index(i, p)] = v;
                logical[i + p * m] = v;
            }
        }
        let mut b = vec![sentinel(); GUARD + shape.b_len() + GUARD];
        for panel in 0..n.div_ceil(NR) {
            for p in 0..kc {
                for col in 0..NR {
                    let j = panel * NR + col;
                    b[GUARD + panel * shape.b_panel() + p * NR + col] =
                        if j < n { value(Operand::B, p, j) } else { 0.0 };
                }
            }
        }
        let ldc = shape.ldc();
        let reads_c = Epilogue::select(alpha, beta).reads_c();
        let mut c = vec![sentinel(); GUARD + ldc * n + GUARD];
        for j in 0..n {
            for i in 0..m {
                c[GUARD + i + j * ldc] = if reads_c {
                    value(Operand::C, i, j)
                } else {
                    unread()
                };
            }
        }
        let packed = pack_logical(&logical, m, kc);
        Self {
            shape,
            alpha,
            beta,
            a,
            b,
            c,
            packed,
        }
    }

    /// Normal random values for every operand.
    fn random(shape: Shape, alpha: f64, beta: f64, seed: u64) -> Self {
        let mut generator = Gen::new(seed);
        Self::new(shape, alpha, beta, &mut |_, _, _| generator.normal())
    }

    /// The argument block (online packing and prefetch off) with C in `c`,
    /// a copy of [`Operands::c`].
    fn args(&self, c: &mut [f64]) -> MacroArgs<f64> {
        let s = self.shape;
        let plain = MacroArgs::packed(
            self.a[GUARD..].as_ptr(),
            self.b[GUARD..].as_ptr(),
            c[GUARD..].as_mut_ptr(),
            s.ldc(),
            s.m,
            s.n,
            s.kc,
            self.alpha,
            self.beta,
        );
        let (a_depth, a_panel) = s.a_strides();
        MacroArgs {
            a_depth,
            a_panel,
            b_panel: s.b_panel(),
            ..plain
        }
    }

    /// The expected C buffer: `reference` (without online packing) on a
    /// copy of the initial C.
    fn expected(&self) -> Vec<f64> {
        let mut want = self.c.clone();
        let args = self.args(&mut want);
        // SAFETY: `new` sized A, B and C so that every address the layout
        // names for `i < m`, `p < kc`, `j < ceil(n / 8) · 8` lies inside
        // `self.a`, `self.b` and `want`; online packing is off.
        unsafe { reference(&args) };
        want
    }

    /// One kernel call on copies of C and of a stale buffer, starting with
    /// FPSR = `fpsr`; checks the thread state and the kernel entry around
    /// it.
    #[track_caller]
    fn call(&self, pack: bool, prefetch: u64, fpsr: u64) -> Outcome {
        let mut c = self.c.clone();
        let len = self.shape.pack_len();
        let mut buffer: Vec<f64> = (0..GUARD + len + buffer_tail(&self.shape))
            .map(|k| {
                if (GUARD..GUARD + len).contains(&k) {
                    stale(k)
                } else {
                    sentinel()
                }
            })
            .collect();
        let a_pack = if pack {
            buffer[GUARD..].as_mut_ptr()
        } else {
            core::ptr::null_mut()
        };
        let args = self
            .args(&mut c)
            .with_online_pack(a_pack)
            .with_prefetch(prefetch);
        assert!(kernel_ready(), "thread not ready before the kernel");
        let entries = kernel_entries_here();
        let saved = fpsr_read();
        fpsr_write(fpsr);
        // SAFETY: every test that calls `call` passed
        // `hardware_or_skip(.., F64)`, so SME2 with FEAT_SME_F64F64 is
        // present, and the thread is ready (checked above). `args` names A,
        // B and C inside `self.a`, `self.b` and `c` for every address of the
        // layout (`new` sized them), `m, n, kc ≥ 1`, `ldc ≥ m`, the mode is
        // `Epilogue::select`'s, and with online packing `a_pack` points at
        // `pack_len()` writable elements of `buffer` (between guards) that
        // overlap no operand.
        unsafe { sme2_gemm_f64(&args) };
        let after = fpsr_read();
        fpsr_write(saved);
        assert_eq!(
            kernel_entries_here(),
            entries + 1,
            "the kernel entry was not counted on this thread"
        );
        assert_eq!(
            sme_state(),
            Some((0, 0)),
            "SVCR / TPIDR2_EL0 after the kernel"
        );
        Outcome {
            c,
            buffer: pack.then_some(buffer),
            fpsr: after,
        }
    }

    /// C against the expected buffer: `same_result` inside the block, the
    /// sentinel bit for bit everywhere else (guards included).
    #[track_caller]
    fn check_c(&self, got: &[f64], want: &[f64], label: &str) {
        let s = self.shape;
        let ldc = s.ldc();
        assert_eq!(got.len(), want.len());
        for (index, (got, want)) in got.iter().zip(want).enumerate() {
            let at = index
                .checked_sub(GUARD)
                .filter(|e| e % ldc < s.m && e / ldc < s.n);
            let ok = match at {
                Some(_) => same_result(*got, *want),
                None => got.to_bits() == sentinel().to_bits(),
            };
            assert!(
                ok,
                "{label}: {s:?} alpha {} beta {}: C element {index} (block element {at:?}) = \
                 {got} ({:#018x}), want {want} ({:#018x})",
                self.alpha,
                self.beta,
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    /// The buffer: `pack::pack_a`'s block bit for bit, guards untouched.
    #[track_caller]
    fn check_buffer(&self, buffer: &[f64], label: &str) {
        let s = self.shape;
        let len = self.packed.len();
        assert_eq!(buffer.len(), GUARD + len + buffer_tail(&s), "{label}");
        for (k, got) in buffer.iter().enumerate() {
            let (want, what) = match k.checked_sub(GUARD) {
                Some(e) if e < len => (self.packed[e], format!("A{:?}", s.packed_at(e))),
                _ => (sentinel(), "a guard".to_string()),
            };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}: {s:?}: buffer element {k} ({what}) = {got}, want {want}"
            );
        }
    }

    /// The four calls of [`SWITCHES`] from FPSR = `fpsr`: C is `reference`
    /// in each and the same bits in all four, the buffer is `pack_a`'s
    /// block in both packing calls, FPSR the same after all four.
    #[track_caller]
    fn verify_from(&self, fpsr: u64, label: &str) -> u64 {
        let want = self.expected();
        let outcomes: Vec<Outcome> = SWITCHES
            .iter()
            .map(|&(pack, prefetch)| self.call(pack, prefetch, fpsr))
            .collect();
        for (outcome, (pack, prefetch)) in outcomes.iter().zip(SWITCHES) {
            let what = format!("{label} (online packing {pack}, prefetch {prefetch})");
            self.check_c(&outcome.c, &want, &what);
            match (&outcome.buffer, pack) {
                (Some(buffer), true) => self.check_buffer(buffer, &what),
                (None, false) => {}
                _ => panic!("{what}: buffer kept for the wrong call"),
            }
        }
        let first = &outcomes[0];
        for (outcome, (pack, prefetch)) in outcomes.iter().zip(SWITCHES).skip(1) {
            assert!(
                bits(&outcome.c) == bits(&first.c),
                "{label}: {:?}: C with online packing {pack}, prefetch {prefetch} differs \
                 from the plain call",
                self.shape
            );
            assert_eq!(
                outcome.fpsr, first.fpsr,
                "{label}: {:?}: FPSR with online packing {pack}, prefetch {prefetch}",
                self.shape
            );
        }
        first.fpsr
    }

    /// [`Operands::verify_from`] with FPSR = 0.
    #[track_caller]
    fn verify(&self, label: &str) {
        self.verify_from(0, label);
    }
}

/// Guard elements after the buffer: [`GUARD`], and room for one more
/// packed block per later column panel, so that even a kernel that packs
/// again in every column panel (it continues past the end) writes only
/// into guards that the test checks, never into the heap.
fn buffer_tail(shape: &Shape) -> usize {
    shape.pack_len() * (shape.n.div_ceil(NR) - 1) + GUARD
}

/// `pack::pack_a`'s block of a column-major `m × kc` A (leading dimension
/// `m`).
fn pack_logical(logical: &[f64], m: usize, kc: usize) -> Vec<f64> {
    let Some(view) = MatRef::from_strided(logical, m, kc, m, 1) else {
        panic!("view {m}x{kc}");
    };
    let Some(op) = ColMajor::normalize(view, Trans::NoTrans) else {
        panic!("operand declined");
    };
    let Some(len) = a_pack_len(m, kc) else {
        panic!("length");
    };
    let mut buf = AlignedVec::filled(len, sentinel());
    let Some(block) = pack_a(
        &op,
        Span { start: 0, len: m },
        Span { start: 0, len: kc },
        &mut buf,
    ) else {
        panic!("pack_a declined {m}x{kc}");
    };
    block.data().to_vec()
}

/// Sweeps `ms × ns × kcs` in both A layouts with random values, rotating
/// the [`PAIRS`] (every mode in both layouts) and the strides.
fn sweep(label: &str, ms: &[usize], ns: &[usize], kcs: &[usize]) {
    let mut index = 0_usize;
    for &m in ms {
        for &n in ns {
            for &kc in kcs {
                for (layout, lda) in [None, Some(in_place_lda(m, [1, 7, 70][index % 3]))]
                    .into_iter()
                    .enumerate()
                {
                    let (alpha, beta) = PAIRS[(index + layout) % PAIRS.len()];
                    let shape = Shape {
                        m,
                        n,
                        kc,
                        lda,
                        b_gap: [0, 5, 8][(index + layout) % 3],
                        ldc_pad: [0, 3, 1][index % 3],
                    };
                    let seed = (m * 1_000_003 + n * 1009 + kc * 17 + index * 2 + layout) as u64;
                    Operands::random(shape, alpha, beta, seed).verify(label);
                }
                index += 1;
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f64_online_harness_agrees() {
    // The harness itself (any host): `pack::pack_a`'s block of the logical
    // A equals the packed copy that `args::reference` writes from the
    // kernel's view of A (in both layouts), and C does not depend on it.
    for (m, n, kc) in [(1, 1, 1), (9, 17, 3), (64, 8, 2), (70, 5, 4), (129, 23, 5)] {
        for lda in [None, Some(in_place_lda(m, 9))] {
            let shape = Shape {
                lda,
                ldc_pad: 3,
                b_gap: 5,
                ..Shape::plain(m, n, kc)
            };
            let operands = Operands::random(shape, 0.3, 0.7, (m + n + kc) as u64);
            let want = operands.expected();
            let mut c = operands.c.clone();
            let len = shape.pack_len();
            let mut copy = vec![sentinel(); len + GUARD];
            let args = operands.args(&mut c).with_online_pack(copy.as_mut_ptr());
            // SAFETY: as in `Operands::expected`, and the copy holds
            // `pack_len()` elements (plus a guard) that overlap no operand.
            unsafe { reference(&args) };
            assert_eq!(bits(&copy[..len]), bits(&operands.packed), "{shape:?}");
            assert!(
                copy[len..]
                    .iter()
                    .all(|x| x.to_bits() == sentinel().to_bits())
            );
            assert_eq!(bits(&c), bits(&want), "{shape:?}");
            assert_eq!(operands.packed.len(), len);
        }
    }
}

#[test]
fn test_sme2_kernel_f64_online_row_variants() {
    if !hardware_or_skip("test_sme2_kernel_f64_online_row_variants", Need::F64) {
        return;
    }
    // Every row variant (1 to 8 live tiles) at its first and last row in
    // one row panel; one column panel (n ≤ 8: the buffer is still filled)
    // and several; odd and even depths with and without whole double steps.
    sweep(
        "row variants",
        &[
            1, 2, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 63, 64,
        ],
        &[1, 7, 8, 9, 17],
        &[1, 2, 3, 8],
    );
}

#[test]
fn test_sme2_kernel_f64_online_row_panels() {
    if !hardware_or_skip("test_sme2_kernel_f64_online_row_panels", Need::F64) {
        return;
    }
    // Two to four row panels (the buffer's panel stride and the source's
    // differ) with one and several column panels (the later ones read the
    // buffer), depths up to 65.
    sweep(
        "row panels",
        &[65, 72, 120, 128, 129, 193],
        &[1, 8, 9, 23, 65],
        &[1, 2, 7, 64, 65],
    );
}

#[test]
fn test_sme2_kernel_f64_online_large_strides() {
    if !hardware_or_skip("test_sme2_kernel_f64_online_large_strides", Need::F64) {
        return;
    }
    // A read in place far apart in depth, wide gaps between B panels, a
    // tall C: every source stride differs from the buffer's.
    for (index, (m, n, kc)) in [
        (1, 9, 2),
        (40, 17, 9),
        (64, 8, 4),
        (129, 23, 3),
        (70, 65, 64),
    ]
    .into_iter()
    .enumerate()
    {
        for (alpha, beta) in PAIRS {
            let shape = Shape {
                m,
                n,
                kc,
                lda: Some(in_place_lda(m, 997)),
                b_gap: 131,
                ldc_pad: 37,
            };
            Operands::random(shape, alpha, beta, (index * 31 + m) as u64).verify("strides");
        }
    }
}

#[test]
fn test_sme2_kernel_f64_online_special_values() {
    if !hardware_or_skip("test_sme2_kernel_f64_online_special_values", Need::F64) {
        return;
    }
    // Signalling and quiet NaNs with payloads, signed zeros, infinities,
    // subnormals: the buffer holds A's bits exactly. Dense specials (most
    // results NaN) and sparse ones (finite results with a few NaN rows and
    // columns), so that neither hides the other's errors.
    const SPECIALS: [f64; 12] = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0123),
        f64::from_bits(0xfff8_0000_0000_0456),
        1.0e-310,
        -1.0e-310,
        f64::MIN_POSITIVE,
        1.0e300,
        -3.0,
    ];
    for (index, (m, n, kc)) in [
        (3, 2, 2),
        (17, 9, 3),
        (49, 17, 5),
        (70, 20, 7),
        (129, 23, 4),
    ]
    .into_iter()
    .enumerate()
    {
        for (pair, (alpha, beta)) in PAIRS.into_iter().enumerate() {
            for (dense, lda) in [(true, None), (false, Some(in_place_lda(m, 5)))] {
                let shape = Shape {
                    lda,
                    ldc_pad: 1,
                    b_gap: 2,
                    ..Shape::plain(m, n, kc)
                };
                let mut generator = Gen::new((index * 100 + pair * 10 + usize::from(dense)) as u64);
                let operands = Operands::new(shape, alpha, beta, &mut |_, _, _| {
                    let special = if dense {
                        !generator.bits().is_multiple_of(4)
                    } else {
                        generator.bits().is_multiple_of(29)
                    };
                    if special {
                        generator.pick(&SPECIALS)
                    } else {
                        generator.normal()
                    }
                });
                operands.verify(if dense {
                    "dense special values"
                } else {
                    "sparse special values"
                });
            }
        }
    }
}

/// Small integers: every product, sum and multiple by 0.5 or 2 is exact.
fn small_int(operand: Operand, x: usize, y: usize) -> f64 {
    let base = match operand {
        Operand::A => 1 + (x * 3 + y) % 5,
        Operand::B => 1 + (x + 2 * y) % 4,
        Operand::C => 1 + (x + y) % 7,
    };
    base as f64
}

#[test]
fn test_sme2_kernel_f64_online_fpsr_unchanged() {
    if !hardware_or_skip("test_sme2_kernel_f64_online_fpsr_unchanged", Need::F64) {
        return;
    }
    // Online packing and prefetching raise no flag of their own: from any
    // earlier flags, all four calls leave the same FPSR, which keeps the
    // earlier flags and adds what the epilogue raises.
    type Values = fn(Operand, usize, usize) -> f64;
    let cases: [(&str, f64, f64, Values, u64); 6] = [
        ("M0 exact", 1.0, 0.0, small_int, 0),
        (
            "M1 inexact",
            1.0,
            1.0,
            |operand, x, y| match operand {
                Operand::C => 1.0 / 1_152_921_504_606_846_976.0,
                _ => small_int(operand, x, y),
            },
            FPSR_IXC,
        ),
        (
            "M1 inf + -inf",
            1.0,
            1.0,
            |operand, _, _| match operand {
                Operand::A => f64::INFINITY,
                Operand::B => 2.0,
                Operand::C => f64::NEG_INFINITY,
            },
            FPSR_IOC,
        ),
        ("M2 inexact", 1.0 / 3.0, 0.0, small_int, FPSR_IXC),
        (
            "M2 overflow",
            1.0e308,
            0.0,
            |operand, x, y| small_int(operand, x, y) * 4.0,
            FPSR_OFC | FPSR_IXC,
        ),
        ("M3 inexact", 1.0 / 3.0, 0.5, small_int, FPSR_IXC),
    ];
    for (m, n, kc) in [(5, 3, 2), (37, 11, 3), (70, 17, 1)] {
        for (label, alpha, beta, value, added) in cases {
            for lda in [None, Some(in_place_lda(m, 2))] {
                let shape = Shape {
                    lda,
                    ..Shape::plain(m, n, kc)
                };
                let operands = Operands::new(shape, alpha, beta, &mut |o, x, y| value(o, x, y));
                for before in [0, FPSR_DZC, FPSR_IOC | FPSR_UFC, FPSR_IXC, FPSR_OFC] {
                    let after = operands.verify_from(before, label);
                    assert_eq!(
                        after,
                        before | added,
                        "{label}: {shape:?}: FPSR {before:#x} became {after:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f64_online_repeated_calls_counted() {
    if !hardware_or_skip(
        "test_sme2_kernel_f64_online_repeated_calls_counted",
        Need::F64,
    ) {
        return;
    }
    // Calls with and without online packing and prefetch, interleaved on
    // one thread: one kernel entry each, counted on this thread; the same
    // bits every time; the thread ready after each call.
    let shapes = [
        Shape {
            lda: Some(in_place_lda(70, 3)),
            ldc_pad: 3,
            b_gap: 4,
            ..Shape::plain(70, 17, 7)
        },
        Shape::plain(9, 23, 5),
    ];
    let operands = [
        Operands::random(shapes[0], -1.5, 0.25, 1),
        Operands::random(shapes[1], 1.0, 0.0, 2),
    ];
    let first: Vec<Outcome> = operands
        .iter()
        .map(|o| o.call(true, PREFETCH_C, 0))
        .collect();
    let entries = kernel_entries_here();
    let rounds = 64_usize;
    for round in 0..rounds {
        let which = round % operands.len();
        let (pack, prefetch) = SWITCHES[(round / 2) % SWITCHES.len()];
        let outcome = operands[which].call(pack, prefetch, 0);
        assert!(
            bits(&outcome.c) == bits(&first[which].c),
            "round {round}: other bits"
        );
        if let (Some(buffer), Some(want)) = (&outcome.buffer, &first[which].buffer) {
            assert!(bits(buffer) == bits(want), "round {round}: other buffer");
        }
        assert!(kernel_ready(), "round {round}: thread not ready");
    }
    assert_eq!(kernel_entries_here(), entries + rounds as u64);
    for (outcome, operand) in first.iter().zip(&operands) {
        operand.check_c(&outcome.c, &operand.expected(), "first call");
        match &outcome.buffer {
            Some(buffer) => operand.check_buffer(buffer, "first call"),
            None => panic!("the first call packed"),
        }
    }
}
