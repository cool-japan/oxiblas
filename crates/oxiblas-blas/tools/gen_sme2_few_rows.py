"""Generates the templates of the SME2 few-rows kernels `sme2_few_rows_f32`
and `sme2_few_rows_f64`.

The kernels live in crates/oxiblas-blas/src/level3/sme2/few_rows.rs, inside
the module's one `sme_kernels!` invocation. Every prologue and body entry of
that macro must be ONE string literal written out in the source (the macro's
grammar takes literals only, and the template lint checks exactly that text),
and ZA tile numbers are immediates. So the depth-loop variants (one per
number of live C tiles, each with a NoTrans and a Trans path for B) and the
four epilogue modes (each unrolled over the C tiles) are written out in full.
This script produces those lines; it is development tooling only: the build
never runs it, and its output is committed.

It owns, for each kernel, the lines from its `// @generated begin` marker to
the following `// @generated end` marker (the `consts`, `prologue` and `body`
lists); everything else in few_rows.rs (module docs, the kernels' doc
comments, the Rust glue) is written by hand.

Usage, from the workspace root:

    python3 -I crates/oxiblas-blas/tools/gen_sme2_few_rows.py           # rewrite the regions
    python3 -I crates/oxiblas-blas/tools/gen_sme2_few_rows.py --check   # verify them

`--check` exits with status 0 when both regions in few_rows.rs are exactly
this script's output, and with status 1 (printing a unified diff) otherwise.
The emitted lines are already in the layout rustfmt leaves alone (rustfmt
does not reformat the inside of a brace-delimited macro invocation).

Register map and loop nest: see the doc comments of the kernels in
few_rows.rs (they describe exactly what `kernel_lines` emits).
"""
import difflib
import os
import sys

SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "src", "level3", "sme2", "few_rows.rs",
)
END = "        // @generated end"

# Indentation of the list entries inside a kernel definition.
LIST = " " * 8
ITEM = " " * 12
# Longest source line of a group of literals (rustfmt's max_width).
WIDTH = 100

# Argument-block fields: (const name, few_rows.rs const).
CONSTS = [
    ("off_a", "OFF_A"),
    ("off_lda", "OFF_LDA"),
    ("off_b", "OFF_B"),
    ("off_ldb", "OFF_LDB"),
    ("off_b_trans", "OFF_B_TRANS"),
    ("off_c", "OFF_C"),
    ("off_ldc", "OFF_LDC"),
    ("off_m", "OFF_M"),
    ("off_n", "OFF_N"),
    ("off_k", "OFF_K"),
    ("off_kc", "OFF_KC"),
    ("off_alpha", "OFF_ALPHA"),
    ("off_beta", "OFF_BETA"),
    ("off_one", "OFF_ONE"),
    ("off_mode_first", "OFF_MODE_FIRST"),
    ("off_mode_later", "OFF_MODE_LATER"),
]

# The prologue: loads from the argument block (x0), then strides in bytes.
PROLOGUE_LOADS = [
    ("x1", "off_a"),
    ("x2", "off_lda"),
    ("x3", "off_b"),
    ("x4", "off_ldb"),
    ("x5", "off_b_trans"),
    ("x6", "off_c"),
    ("x7", "off_ldc"),
    ("x20", "off_m"),
    ("x21", "off_n"),
    ("x22", "off_k"),
    ("x23", "off_kc"),
    ("x25", "off_mode_first"),
]
PROLOGUE_BYTES = ["x2", "x4", "x7"]


class Type:
    """What differs between the two kernels."""

    def __init__(self, name, e, shift, tile, nr, tiles, transposer, ld, st, scalar):
        self.name = name              # kernel name
        self.e = e                    # element suffix: "s" or "d"
        self.shift = shift            # log2 of the element size in bytes
        self.tile = tile              # rows of one ZA tile (= columns of a panel = depth chunk)
        self.nr = nr                  # columns of one column panel
        self.tiles = tiles            # C tiles at most (max rows / tile rows)
        self.transposer = transposer  # the ZA tile that transposes NoTrans B
        self.ld = ld                  # contiguous load mnemonic
        self.st = st                  # contiguous store mnemonic
        self.scalar = scalar          # register that carries a bit pattern for dup

    @property
    def chunk(self):
        """Depth steps of one transposed chunk: the transposer's slice length."""
        return self.tile

    @property
    def nr_shift(self):
        return {16: 4, 8: 3}[self.nr]


F32 = Type("sme2_few_rows_f32", "s", 2, 16, 16, 3, 3, "ld1w", "st1w", "w16")
F64 = Type("sme2_few_rows_f64", "d", 3, 8, 8, 4, 4, "ld1d", "st1d", "x16")


def begin_marker(ty):
    return (f"        // @generated begin: tools/gen_sme2_few_rows.py, {ty.name} "
            "(edit the generator, not these lines)")


class Body:
    """Collects body lines: instructions (string literals) and comments."""

    def __init__(self):
        self.lines = []

    def comment(self, text):
        self.lines.append(f"{ITEM}// {text}")

    def __call__(self, *statements):
        """One call is one group of statements: its string literals share
        source lines, packed up to WIDTH columns (the template still has one
        statement per line: the macro joins the literals with line feeds)."""
        line = ""
        for statement in statements:
            assert '"' not in statement and ";" not in statement, statement
            literal = f'"{statement}",'
            if line and len(ITEM) + len(line) + 1 + len(literal) > WIDTH:
                self.lines.append(ITEM + line)
                line = ""
            line = f"{line} {literal}" if line else literal
        if line:
            self.lines.append(ITEM + line)


def a_load(ty, live, set_two):
    """The A load of one depth step for `live` tiles into set one (z0-z3)
    or set two (z4-z7), from x14. Rows i >= m are inactive lanes (/z, never
    read): p1 covers rows 0..tile-1, pn9 rows 0..4*tile-1, both < m."""
    base = 4 if set_two else 0
    e = ty.e
    if live == 1:
        return f"{ty.ld} {{{{z{base}.{e}}}}}, p1/z, [x14]"
    if live == 2:
        return f"{ty.ld} {{{{z{base}.{e}-z{base + 1}.{e}}}}}, pn9/z, [x14]"
    return f"{ty.ld} {{{{z{base}.{e}-z{base + 3}.{e}}}}}, pn9/z, [x14]"


def fmopas(ty, live, b_reg, set_two):
    base = 4 if set_two else 0
    e = ty.e
    return [f"fmopa za{t}.{e}, p0/m, p0/m, {b_reg}.{e}, z{base + t}.{e}" for t in range(live)]


def variant(ty, w, live):
    """The depth loop of one (depth block, column panel) for `live` C tiles."""
    e = ty.e
    za = f"za{ty.transposer}"
    tiles = ", ".join(f"za{t}.{e}" for t in range(live))
    lo, hi = ty.tile * (live - 1) + 1, ty.tile * live
    w.comment(f"{live} live C tile{'s' if live > 1 else ''} ({lo} to {hi} rows): zero them, then the depth loop.")
    w(f"4{2 * live - 1}:", f"zero {{{{{tiles}}}}}", f"cbnz x5, 2{live}6f")
    w.comment(f"NoTrans B. Chunk of min({ty.chunk}, x11) depth steps: column c of the panel")
    w.comment(f"(c < ncols) into horizontal slice c of {za}.{e}, lanes of the live depth (p5).")
    w(f"2{live}2:", "mov w12, #0", f"whilelt p5.{e}, x12, x11", "mov x16, x15")
    w(f"2{live}3:", f"{ty.ld} {{{{{za}h.{e}[w12, 0]}}}}, p5/z, [x16]", "add x16, x16, x4")
    w("add w12, w12, #1", "cmp w12, w8", f"b.lo 2{live}3b")
    w.comment(f"Vertical slice q of {za}.{e} is the B row of depth step q of the chunk.")
    w("mov w13, #0", f"cmp x11, #{ty.chunk}", f"b.lo 2{live}5f")
    w.comment(f"A whole chunk: four B rows per MOVA, {ty.chunk // 4} groups.")
    w(f"2{live}4:", f"mova {{{{z8.{e}-z11.{e}}}}}, {za}v.{e}[w13, 0:3]")
    w(a_load(ty, live, False), "add x14, x14, x2", a_load(ty, live, True), "add x14, x14, x2")
    w(*fmopas(ty, live, "z8", False))
    w(*fmopas(ty, live, "z9", True))
    w(a_load(ty, live, False), "add x14, x14, x2", a_load(ty, live, True), "add x14, x14, x2")
    w(*fmopas(ty, live, "z10", False))
    w(*fmopas(ty, live, "z11", True))
    w("add w13, w13, #4", f"cmp w13, #{ty.chunk}", f"b.lo 2{live}4b")
    w("add x15, x15, #64", f"subs x11, x11, #{ty.chunk}", f"b.ne 2{live}2b", "b 50f")
    w.comment("The last, partial chunk of the block: one depth step per B row.")
    w(f"2{live}5:", f"mova z8.{e}, p0/m, {za}v.{e}[w13, 0]")
    w(a_load(ty, live, False), "add x14, x14, x2")
    w(*fmopas(ty, live, "z8", False))
    w("add w13, w13, #1", "cmp w13, w11", f"b.lo 2{live}5b", "b 50f")
    w.comment("Trans B: the B row of a depth step is contiguous; columns < n (p6). Two")
    w.comment("steps per iteration, then one more when the block's depth is odd.")
    w(f"2{live}6:", "lsr x16, x11, #1", f"cbz x16, 2{live}7f")
    w(f"2{live}8:", f"{ty.ld} {{{{z8.{e}}}}}, p6/z, [x15]", "add x15, x15, x4")
    w(a_load(ty, live, False), "add x14, x14, x2")
    w(f"{ty.ld} {{{{z9.{e}}}}}, p6/z, [x15]", "add x15, x15, x4")
    w(a_load(ty, live, True), "add x14, x14, x2")
    w(*fmopas(ty, live, "z8", False))
    w(*fmopas(ty, live, "z9", True))
    w("subs x16, x16, #1", f"b.ne 2{live}8b")
    w(f"2{live}7:", "tbz x11, #0, 50f", f"{ty.ld} {{{{z8.{e}}}}}, p6/z, [x15]")
    w(a_load(ty, live, False))
    w(*fmopas(ty, live, "z8", False))
    w("b 50f")


def address(ty, t):
    """C(tile * t, j0 + c): x13 plus the row offset of tile t (x14, x15, x16)."""
    return "[x13]" if t == 0 else f"[x13, x{13 + t}, lsl #{ty.shift}]"


def mode_ops(ty, mode, t):
    """The work on one column slice of C tile t, rows predicated by p<t+1>.
    z28 = alpha, z29 = beta of the block (beta, or one after the first)."""
    e, p, a = ty.e, f"p{t + 1}", address(ty, t)
    slice_ = f"za{t}h.{e}[w12, 0]"
    if mode == 0:
        return [f"{ty.st} {{{{{slice_}}}}}, {p}, {a}"]
    ops = [f"mova z16.{e}, p0/m, {slice_}"]
    if mode == 1:
        ops += [f"{ty.ld} {{{{z17.{e}}}}}, {p}/z, {a}", f"fadd z16.{e}, {p}/m, z16.{e}, z17.{e}"]
    elif mode == 2:
        ops += [f"fmul z16.{e}, {p}/m, z16.{e}, z28.{e}"]
    else:
        ops += [
            f"fmul z16.{e}, {p}/m, z16.{e}, z28.{e}",
            f"{ty.ld} {{{{z17.{e}}}}}, {p}/z, {a}",
            f"fmla z16.{e}, {p}/m, z17.{e}, z29.{e}",
        ]
    return ops + [f"{ty.st} {{{{z16.{e}}}}}, {p}, {a}"]


MODE_DOCS = [
    "M0: C = acc. Direct ZA slice stores; C is not read.",
    "M1: C = acc + C (SVE FADD, accumulator first).",
    "M2: C = acc * alpha (SVE FMUL); C is not read.",
    "M3: t = acc * alpha (SVE FMUL), C = t + C * beta (SVE FMLA, one rounding).",
]


def epilogue(ty, w):
    w.comment(f"Epilogue of the {ty.tile * ty.tiles} x {ty.nr} C tile of the column panel, by mode:")
    w.comment("columns 0..ncols (w12), each tile while its first row is below m.")
    w("50:", "mov w12, #0", "mov x13, x9")
    w(*[f"mov x{13 + t}, #{ty.tile * t}" for t in range(1, ty.tiles)])
    w("cbz x25, 51f", "cmp x25, #2", "b.lo 52f", "b.eq 53f", "b 54f")
    for mode in range(4):
        w.comment(MODE_DOCS[mode])
        w(f"5{mode + 1}:")
        for t in range(ty.tiles):
            w(*mode_ops(ty, mode, t))
            if t + 1 < ty.tiles:
                w(f"cmp x20, #{ty.tile * (t + 1)}", f"b.ls 5{mode + 5}f")
        w(f"5{mode + 5}:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", f"b.lo 5{mode + 1}b")
        if mode < 3:
            w("b 59f")


def kernel_lines(ty):
    e = ty.e
    out = [begin_marker(ty)]
    out.append(f"{LIST}consts: [")
    out += [f"{ITEM}{name} = {value}," for name, value in CONSTS]
    out.append(f"{LIST}],")
    out.append(f"{LIST}prologue: [")
    out += [f'{ITEM}"ldr {reg}, [x0, #{{{name}}}]",' for reg, name in PROLOGUE_LOADS]
    out += [f'{ITEM}"lsl {reg}, {reg}, #{ty.shift}",' for reg in PROLOGUE_BYTES]
    out.append(f"{LIST}],")
    out.append(f"{LIST}body: [")

    w = Body()
    w.comment("p0 all-true; z28 alpha, z29 beta (the first depth block's).")
    w(f"ptrue p0.{e}")
    w(f"ldr {ty.scalar}, [x0, #{{off_alpha}}]", f"dup z28.{e}, {ty.scalar}")
    w(f"ldr {ty.scalar}, [x0, #{{off_beta}}]", f"dup z29.{e}, {ty.scalar}")
    w.comment(f"Rows below m: p1 (tile 0), pn9 (tiles 0-3, for the A loads), p2..p{ty.tiles} (tiles 1..).")
    w("mov x16, #0", f"whilelt p1.{e}, x16, x20", f"whilelt pn9.{e}, x16, x20, vlx4")
    for t in range(1, ty.tiles):
        w(f"mov x16, #{ty.tile * t}", f"whilelt p{t + 1}.{e}, x16, x20")
    w.comment("x24 = p0 (first depth step of the block), x27 = A(0, p0).")
    w("mov x24, #0", "mov x27, x1")
    w.comment("Depth block: x26 = d = min(kc, k - p0); x28 = j0, x9 = C(0, j0),")
    w.comment("x10 = B(p0, j0) of op(B): b + p0 (NoTrans, column-major) or b + p0 * ldb (Trans).")
    w("70:", "sub x26, x22, x24", "cmp x26, x23", "csel x26, x26, x23, lo")
    w("mov x28, #0", "mov x9, x6", "cbnz x5, 71f", f"add x10, x3, x24, lsl #{ty.shift}", "b 20f")
    w("71:", "madd x10, x24, x4, x3")
    w.comment(f"Column panel: x8 = ncols = min({ty.nr}, n - j0), p6 = columns below n; x14 = A,")
    w.comment("x15 = B at depth p0 and column j0, x11 = depth steps left in the block.")
    w("20:", "sub x8, x21, x28", f"mov x16, #{ty.nr}", "cmp x8, x16", "csel x8, x8, x16, lo")
    w(f"whilelt p6.{e}, x28, x21", "mov x14, x27", "mov x15, x10", "mov x11, x26")
    w.comment(f"Live C tiles L = ceil(m / {ty.tile}) select the variant 4(2L-1).")
    for live in range(ty.tiles, 1, -1):
        w(f"cmp x20, #{ty.tile * (live - 1)}", f"b.hi 4{2 * live - 1}f")
    for live in range(1, ty.tiles + 1):
        variant(ty, w, live)
    epilogue(ty, w)
    w.comment("Next column panel; then the next depth block, whose epilogue mode is the later")
    w.comment("blocks' and whose beta is one. The end falls through to the exit label 90.")
    w("59:", f"add x28, x28, #{ty.nr}", f"add x9, x9, x7, lsl #{ty.nr_shift}")
    w("cbnz x5, 66f", f"add x10, x10, x4, lsl #{ty.nr_shift}", "b 67f")
    w("66:", "add x10, x10, #64")
    w("67:", "cmp x28, x21", "b.lo 20b")
    w("add x24, x24, x23", "madd x27, x23, x2, x27")
    w("ldr x25, [x0, #{off_mode_later}]")
    w(f"ldr {ty.scalar}, [x0, #{{off_one}}]", f"dup z29.{e}, {ty.scalar}")
    w("cmp x24, x22", "b.lo 70b")
    out += w.lines
    out.append(f"{LIST}],")
    out.append(END)
    return out


def region(lines, ty):
    marker = begin_marker(ty)
    try:
        begin = lines.index(marker)
        end = lines.index(END, begin)
    except ValueError:
        sys.exit(f"{SOURCE}: the @generated markers of {ty.name} are missing")
    return begin, end


def main(argv):
    check = argv[1:] == ["--check"]
    if argv[1:] and not check:
        sys.exit(__doc__)
    with open(SOURCE, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    status = 0
    for ty in (F32, F64):
        begin, end = region(lines, ty)
        current = lines[begin:end + 1]
        wanted = kernel_lines(ty)
        if check:
            if current != wanted:
                status = 1
                sys.stdout.writelines(
                    line + "\n"
                    for line in difflib.unified_diff(
                        current, wanted, f"few_rows.rs ({ty.name})", "generator", lineterm=""
                    )
                )
            continue
        lines[begin:end + 1] = wanted
    if check:
        if status == 0:
            print("few_rows.rs: the generated regions are current")
        return status
    with open(SOURCE, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
