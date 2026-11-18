"""Generates the template of the SME2 f64 macro-kernel `sme2_gemm_f64`.

The kernel lives in crates/oxiblas-blas/src/level3/sme2/kernel_f64.rs, inside
its module's one `sme_kernels!` invocation. Every prologue and body entry of
that macro must be ONE string literal written out in the source (the macro's
grammar takes literals only, and the template lint checks exactly that text),
and the ZA tile number of `zero`, `fmopa`, `mova` and the slice `st1d` is an
immediate. So the eight depth-loop variants (one per number of live tiles),
their eight packing twins (online packing of A) and the four epilogue modes
(each unrolled over the eight tiles) are written out in full. This script
produces those lines; it is development tooling only: the build never runs
it, and its output is committed.

What the generated body does (the kernel's doc comment in kernel_f64.rs has
the loop nest, the register map and the store table):

- Plain depth loops 4L (L = 1..8 live tiles; double-step loop 6L, odd-kc
  remainder 7L) read A with the predicates of its rows (`p2`, `pn9`,
  `pn10`), exactly as a plain packed block.
- Online packing (`a_pack` non-null, held in x8 until column panel 0 is
  done): the row panels of column panel 0 run the packing twins 14L (loop
  16L, remainder 17L). Each depth step loads all 64 rows of A as eight
  vectors under `pn9` / `pn10` (rows >= m and tiles without a live row read
  nothing and are +0), multiplies the live tiles exactly like 4L, and stores
  the eight vectors as two whole four-vector groups (all-true `pn8`) at the
  buffer pointer x6, which then moves on by one depth step (512 bytes). The
  variant is chosen once per row panel; the depth loops never test whether
  they pack. After column panel 0 (label 29) A is the buffer: a_depth = 64,
  a_panel = 64 * kc.
- Prefetch (`prefetch` bit 0, PREFETCH_C): before the depth loop of each
  tile, a loop over the live columns (labels 32, 33) issues one
  `prfm pldl2keep` per 64 bytes (one 8-row ZA tile) of the live rows; with
  the bit clear `tbz` skips the block (label 39).

It owns the lines from the `// @generated begin` marker to the
`// @generated end` marker of kernel_f64.rs (the `consts`, `prologue` and
`body` lists of the kernel); everything else in the file (module docs, the
kernel's doc comment) is written by hand.

Usage, from the workspace root:

    python3 -I crates/oxiblas-blas/tools/gen_sme2_kernel_f64.py           # rewrite the region
    python3 -I crates/oxiblas-blas/tools/gen_sme2_kernel_f64.py --check   # verify it

`--check` exits with status 0 when the region in kernel_f64.rs is exactly
this script's output, and with status 1 (printing a unified diff) otherwise.
The emitted lines are already in the layout rustfmt leaves alone (rustfmt
does not reformat the inside of a brace-delimited macro invocation).
"""
import difflib
import os
import sys

KERNEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "src", "level3", "sme2", "kernel_f64.rs",
)
BEGIN = "        // @generated begin: tools/gen_sme2_kernel_f64.py (edit the generator, not these lines)"
END = "        // @generated end"

# Indentation of the list entries inside the kernel definition.
LIST = " " * 8
ITEM = " " * 12

# Argument-block fields read by the prologue: (const name, args.rs const).
CONSTS = [
    ("off_a", "OFF_A"),
    ("off_a_depth", "OFF_A_DEPTH"),
    ("off_a_panel", "OFF_A_PANEL"),
    ("off_b", "OFF_B"),
    ("off_b_panel", "OFF_B_PANEL"),
    ("off_c", "OFF_C"),
    ("off_ldc", "OFF_LDC"),
    ("off_m", "OFF_M"),
    ("off_n", "OFF_N"),
    ("off_kc", "OFF_KC"),
    ("off_alpha", "OFF_ALPHA"),
    ("off_beta", "OFF_BETA"),
    ("off_mode", "OFF_MODE"),
    ("off_a_pack", "OFF_A_PACK"),
    ("off_prefetch", "OFF_PREFETCH"),
]

# The prologue: loads from the argument block (x0), then the strides in bytes.
# B and C go straight to the registers the body walks them with (x27, x28).
PROLOGUE_LOADS = [
    ("x1", "off_a"),
    ("x2", "off_a_depth"),
    ("x3", "off_a_panel"),
    ("x27", "off_b"),
    ("x5", "off_b_panel"),
    ("x28", "off_c"),
    ("x7", "off_ldc"),
    ("x8", "off_a_pack"),
    ("x4", "off_prefetch"),
    ("x20", "off_m"),
    ("x21", "off_n"),
    ("x22", "off_kc"),
    ("x23", "off_alpha"),
    ("x24", "off_beta"),
    ("x25", "off_mode"),
]
PROLOGUE_BYTES = ["x2", "x3", "x5", "x7"]

TILES = 8          # .d ZA tiles, 8 rows each
TILE_BYTES = 64    # one tile column slice: 8 doubles
NR = 8             # columns of one packed B panel
STEP_BYTES = TILES * TILE_BYTES   # one packed depth step of A: 64 doubles


class Body:
    """Collects body lines: instructions (string literals) and comments."""

    def __init__(self):
        self.lines = []

    def comment(self, text):
        self.lines.append(f"{ITEM}// {text}")

    def blank(self):
        self.lines.append("")

    def __call__(self, *statements):
        for statement in statements:
            assert '"' not in statement and ";" not in statement, statement
            self.lines.append(f'{ITEM}"{statement}",')


def a_loads(live, set_two):
    """The A loads of one depth step for `live` tiles (set one: z0-z7,
    set two: z16-z23). Rows i >= m are inactive lanes (/z, never read)."""
    base = 16 if set_two else 0
    z = lambda k: f"z{base + k}.d"
    if live == 1:
        return [f"ld1d {{{{{z(0)}}}}}, p2/z, [x14]"]
    if live == 2:
        return [f"ld1d {{{{{z(0)}-{z(1)}}}}}, pn9/z, [x14]"]
    loads = [f"ld1d {{{{{z(0)}-{z(3)}}}}}, pn9/z, [x14]"]
    if live in (5, 6):
        loads.append(f"ld1d {{{{{z(4)}-{z(5)}}}}}, pn10/z, [x14, #4, mul vl]")
    elif live in (7, 8):
        loads.append(f"ld1d {{{{{z(4)}-{z(7)}}}}}, pn10/z, [x14, #4, mul vl]")
    return loads


def fmopas(live, b_reg, set_two):
    base = 16 if set_two else 0
    return [
        f"fmopa za{t}.d, p0/m, p0/m, {b_reg}.d, z{base + t}.d" for t in range(live)
    ]


def depth_variant(w, live):
    tiles = ", ".join(f"za{t}.d" for t in range(live))
    w.comment(f"{live} live tile{'s' if live > 1 else ''} "
              f"({8 * (live - 1) + 1} to {8 * live} rows): zero the live tiles, then the depth loop.")
    w(f"4{live}:", f"zero {{{{{tiles}}}}}", f"cbz x16, 7{live}f")
    w(f"6{live}:")
    w(*a_loads(live, False))
    w("ld1d {{z8.d}}, p0/z, [x15]", "add x14, x14, x2")
    w(*a_loads(live, True))
    w("ld1d {{z9.d}}, p0/z, [x15, #1, mul vl]", "add x14, x14, x2", "add x15, x15, #128")
    w(*fmopas(live, "z8", False))
    w(*fmopas(live, "z9", True))
    w("subs x16, x16, #1", f"b.ne 6{live}b")
    w(f"7{live}:", "tbz x22, #0, 50f")
    w(*a_loads(live, False))
    w("ld1d {{z8.d}}, p0/z, [x15]")
    w(*fmopas(live, "z8", False))
    w("b 50f")


def step_loads(set_two):
    """The A loads of one depth step of the packing pass: all 64 rows as two
    groups of four vectors (rows i0..+31 under pn9, i0+32..+63 under pn10).
    Rows >= m are inactive lanes: never read, +0 in the register, so the
    eight vectors are the step of the plain packed layout."""
    base = 16 if set_two else 0
    return [
        f"ld1d {{{{z{base}.d-z{base + 3}.d}}}}, pn9/z, [x14]",
        f"ld1d {{{{z{base + 4}.d-z{base + 7}.d}}}}, pn10/z, [x14, #4, mul vl]",
    ]


def step_stores(set_two, first):
    """The buffer stores of one depth step: the eight A vectors of the step as
    two whole groups (all-true pn8) at vector offsets `first` and `first + 4`
    from the buffer pointer x6."""
    base = 16 if set_two else 0
    offset = lambda k: "[x6]" if k == 0 else f"[x6, #{k}, mul vl]"
    return [
        f"st1d {{{{z{base}.d-z{base + 3}.d}}}}, pn8, {offset(first)}",
        f"st1d {{{{z{base + 4}.d-z{base + 7}.d}}}}, pn8, {offset(first + 4)}",
    ]


def packing_variant(w, live):
    """The packing twin 14L of variant 4L: the same outer products into the
    same live tiles, from loads of whole depth steps that it also stores."""
    tiles = ", ".join(f"za{t}.d" for t in range(live))
    w.comment(f"Packing twin of 4{live} ({live} live tile{'s' if live > 1 else ''}): "
              f"whole depth steps of A to the buffer at x6.")
    w(f"14{live}:", f"zero {{{{{tiles}}}}}", f"cbz x16, 17{live}f")
    w(f"16{live}:")
    w(*step_loads(False))
    w("ld1d {{z8.d}}, p0/z, [x15]", "add x14, x14, x2")
    w(*step_loads(True))
    w("ld1d {{z9.d}}, p0/z, [x15, #1, mul vl]", "add x14, x14, x2", "add x15, x15, #128")
    w(*fmopas(live, "z8", False))
    w(*fmopas(live, "z9", True))
    w(*step_stores(False, 0))
    w(*step_stores(True, TILES))
    w(f"add x6, x6, #{2 * STEP_BYTES}", "subs x16, x16, #1", f"b.ne 16{live}b")
    w(f"17{live}:", "tbz x22, #0, 50f")
    w(*step_loads(False))
    w("ld1d {{z8.d}}, p0/z, [x15]")
    w(*fmopas(live, "z8", False))
    w(*step_stores(False, 0))
    w(f"add x6, x6, #{STEP_BYTES}", "b 50f")


def dispatch(w, prefix):
    """The choice of the variant `prefix`L by the live rows x13 = m - i0."""
    for live in range(1, TILES):
        w(f"cmp x13, #{8 * live}", f"b.ls {prefix}{live}f")
    w(f"b {prefix}{TILES}f")


def prefetch(w):
    """The C prefetch of the tile at (i0, j0): for each live column, one hint
    at C(i0 + 8t, j0 + c) for every live tile t (64 bytes each)."""
    w("tbz x4, #0, 39f", "add x14, x28, x10, lsl #3", "mov x16, x9")
    w("32:", "prfm pldl2keep, [x14]")
    for t in range(1, TILES):
        w(f"cmp x13, #{8 * t}", "b.ls 33f", f"prfm pldl2keep, [x14, #{TILE_BYTES * t}]")
    w("33:", "add x14, x14, x7", "subs x16, x16, #1", "b.ne 32b")


# The work on one column slice of tile t in each mode, on rows predicated by
# p1, at the C address x15 (column j0 + w12). z28 = alpha, z29 = beta.
MODES = [
    ("M0: C = acc. Direct ZA slice stores; C is not read.",
     lambda t: [f"st1d {{{{za{t}h.d[w12, 0]}}}}, p1, [x15]"]),
    ("M1: C = acc + C (SVE FADD, accumulator first).",
     lambda t: [
         f"mova z16.d, p1/m, za{t}h.d[w12, 0]",
         "ld1d {{z17.d}}, p1/z, [x15]",
         "fadd z16.d, p1/m, z16.d, z17.d",
         "st1d {{z16.d}}, p1, [x15]",
     ]),
    ("M2: C = acc * alpha (SVE FMUL); C is not read.",
     lambda t: [
         f"mova z16.d, p1/m, za{t}h.d[w12, 0]",
         "fmul z16.d, p1/m, z16.d, z28.d",
         "st1d {{z16.d}}, p1, [x15]",
     ]),
    ("M3: t = acc * alpha (SVE FMUL), C = t + C * beta (SVE FMLA, one rounding).",
     lambda t: [
         f"mova z16.d, p1/m, za{t}h.d[w12, 0]",
         "fmul z16.d, p1/m, z16.d, z28.d",
         "ld1d {{z17.d}}, p1/z, [x15]",
         "fmla z16.d, p1/m, z17.d, z29.d",
         "st1d {{z16.d}}, p1, [x15]",
     ]),
]


def epilogue(w, mode):
    doc, slice_ops = MODES[mode]
    w.comment(doc)
    w.comment("Tile by tile while its first row is below m; columns 0..ncols of each.")
    w(f"5{mode + 1}:", "mov x13, x10", "add x14, x28, x10, lsl #3", "whilelt p1.d, x13, x20")
    for t in range(TILES):
        w("mov w12, #0", "mov x15, x14", "mov x16, x9")
        w("56:")
        w(*slice_ops(t))
        w("add x15, x15, x7", "add w12, w12, #1", "subs x16, x16, #1", "b.ne 56b")
        if t + 1 < TILES:
            w("add x13, x13, #8", "cmp x13, x20", "b.hs 59f")
            w(f"add x14, x14, #{TILE_BYTES}", "whilelt p1.d, x13, x20")
    if mode + 1 < len(MODES):
        w("b 59f")


def generate():
    out = [BEGIN]
    out.append(f"{LIST}consts: [")
    out += [f"{ITEM}{name} = {value}," for name, value in CONSTS]
    out.append(f"{LIST}],")
    out.append(f"{LIST}prologue: [")
    out += [f'{ITEM}"ldr {reg}, [x0, #{{{name}}}]",' for reg, name in PROLOGUE_LOADS]
    out += [f'{ITEM}"lsl {reg}, {reg}, #3",' for reg in PROLOGUE_BYTES]
    out.append(f"{LIST}],")
    out.append(f"{LIST}body: [")

    w = Body()
    w.comment("p0 all-true, pn8 all-true counter; z28 alpha, z29 beta; x26 j0, x27 B panel,")
    w.comment("x28 C column j0; x8 a_pack until column panel 0 is done, x6 where it is written.")
    w("ptrue p0.d", "ptrue pn8.d", "dup z28.d, x23", "dup z29.d, x24")
    w("mov x26, #0", "mov x6, x8")
    w.comment("Column panel: x9 = ncols = min(8, n - j0); row panels from i0 = 0.")
    w("20:", "sub x9, x21, x26", "mov x16, #8", "cmp x9, x16", "csel x9, x9, x16, lo")
    w("mov x10, #0", "mov x11, x1")
    w.comment("Row panel: x13 = m - i0; p2 (rows i0..+7), pn9 (i0..+31) and pn10")
    w.comment("(i0+32..+63) select the A rows below m.")
    w("30:", "sub x13, x20, x10", "whilelt p2.d, x10, x20", "whilelt pn9.d, x10, x20, vlx4")
    w("add x16, x10, #32", "whilelt pn10.d, x16, x20, vlx4")
    w.comment("Prefetch (bit 0 of prefetch): x14 C(i0, j0 + c), x16 columns left.")
    prefetch(w)
    w.comment("x14 A, x15 B, x16 kc / 2. Live tiles L = ceil(min(64, m - i0) / 8) select")
    w.comment("the packing twin 14L while online packing is pending (x8 != 0), else 4L.")
    w("39:", "mov x14, x11", "mov x15, x27", "lsr x16, x22, #1", "cbz x8, 40f")
    dispatch(w, "14")
    for live in range(1, TILES + 1):
        packing_variant(w, live)
    w.comment("The plain depth loops.")
    w("40:")
    dispatch(w, "4")
    for live in range(1, TILES + 1):
        depth_variant(w, live)
    w.comment("Epilogue of the 64 x 8 tile at (i0, j0), by mode.")
    w("50:", "cmp x25, #1", "b.eq 52f", "cmp x25, #2", "b.eq 53f", "cmp x25, #3", "b.eq 54f")
    for mode in range(len(MODES)):
        epilogue(w, mode)
    w.comment("Next row panel, then next column panel; the end falls through to 90.")
    w("59:", "add x10, x10, #64", "add x11, x11, x3", "cmp x10, x20", "b.lo 30b")
    w.comment("After column panel 0 with online packing, A is the buffer (plain packed).")
    w("cbz x8, 29f", "mov x1, x8", f"mov x2, #{STEP_BYTES}")
    w(f"lsl x3, x22, #{STEP_BYTES.bit_length() - 1}", "mov x8, #0")
    w("29:", "add x26, x26, #8", "add x27, x27, x5", "add x28, x28, x7, lsl #3")
    w("cmp x26, x21", "b.lo 20b")
    out += w.lines
    out.append(f"{LIST}],")
    out.append(END)
    return out


def region(lines):
    try:
        begin = lines.index(BEGIN)
        end = lines.index(END, begin)
    except ValueError:
        sys.exit(f"{KERNEL}: the @generated markers are missing")
    return begin, end


def main(argv):
    check = argv[1:] == ["--check"]
    if argv[1:] and not check:
        sys.exit(__doc__)
    with open(KERNEL, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    begin, end = region(lines)
    current = lines[begin:end + 1]
    wanted = generate()
    if check:
        if current == wanted:
            print("kernel_f64.rs: the generated region is current")
            return 0
        sys.stdout.writelines(
            line + "\n"
            for line in difflib.unified_diff(
                current, wanted, "kernel_f64.rs", "generator", lineterm=""
            )
        )
        return 1
    lines[begin:end + 1] = wanted
    with open(KERNEL, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
