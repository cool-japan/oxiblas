//! The `f32` macro-kernel of the SME2 backend: [`sme2_gemm_f32`], one
//! streaming region that computes a whole packed block `C[0..m, 0..n]` to the
//! contract of `args.rs`, in C tiles of 64 × 16 held in the four `.s` ZA
//! tiles.
//!
//! # Shape of the computation
//!
//! With a 512-bit streaming vector length a `.s` ZA tile is 16 × 16. The
//! kernel walks column panels of 16 columns and, inside each, row panels of
//! 64 rows; the 64 × 16 C tile of one (row panel, column panel) pair lives in
//! `za0`–`za3` (tile `t` holds rows `i0 + 16t .. i0 + 16t + 15`). FMOPA takes
//! the packed B row as `Zn` and the A vector as `Zm`, so horizontal slice `c`
//! of a tile is a contiguous piece of column `j0 + c` of C and is stored with
//! one predicated vector store.
//!
//! - **Row variants.** A row panel with 1–16, 17–32, 33–48 or 49–64 live rows
//!   runs a depth loop over 1, 2, 3 or 4 tiles: a tile without a live row is
//!   neither zeroed, nor accumulated, nor stored. The per-tile row predicates
//!   (`whilelt`) zero the A lanes of rows `≥ m` without reading them and
//!   limit every C access to rows `< m`.
//! - **Column edge.** Packed B is zero-padded to whole panels, so the B side
//!   of FMOPA runs all-true; the epilogue visits only the live columns of a
//!   partial last panel.
//! - **Depth loop.** Unrolled by two (two B rows, two sets of A vectors) with
//!   a one-step remainder for odd `kc`; the used tiles are zeroed before it
//!   (rule R2), so every C element is `fma(A(i, p), B(p, j), acc)` over `p`
//!   ascending from `+0`, as the contract defines.
//! - **Epilogue** (rule R3, mode chosen in Rust): M0 stores the ZA slices
//!   directly; M1, M2 and M3 move a slice to a Z register (MOVA) and use
//!   predicated SVE arithmetic: `acc + C` (FADD, accumulator first), `acc ·
//!   alpha` (FMUL), and `acc · alpha + C · beta` (FMUL, then FMLA). C is read
//!   only in M1 and M3.
//!
//! # Floating-point flags
//!
//! FMOPA raises no flags (the Arm ARM's `FPMulAdd_ZA`), so the depth loop
//! never adds any. The epilogue's SVE arithmetic does, on live lanes only
//! (it is predicated by the row predicate): M0 raises nothing, M1 what its
//! FADD raises, M2 what its FMUL raises, M3 what its FMUL and FMLA raise.
//! The frame of `asm.rs` adds them to the caller's flags.
//!
//! The tests live in `tests_kernel_f32.rs`.

use super::args::{
    MacroArgs, OFF_A, OFF_A_DEPTH, OFF_A_PANEL, OFF_ALPHA, OFF_B, OFF_B_PANEL, OFF_BETA, OFF_C,
    OFF_KC, OFF_LDC, OFF_M, OFF_MODE, OFF_N,
};
use super::asm::sme_kernels;

sme_kernels! {
    /// Computes the whole block `C[0..m, 0..n]` of one `f32` macro-kernel
    /// call in one streaming region (contract: `args.rs`; layout of the
    /// computation: the module docs).
    ///
    /// Loop nest (labels in brackets):
    ///
    /// ```text
    /// for j0 in (0..n).step_by(16)                     [20] column panel
    ///     ncols = min(16, n - j0)
    ///     for i0 in (0..m).step_by(64)                 [30] row panel
    ///         p1..p4 = whilelt(i0 + 16t, m), rows = m - i0
    ///         T = 1, 2, 3 or 4 tiles for rows ≤ 16, ≤ 32, ≤ 48, > 48
    ///         variant T                                [41 43 45 47]
    ///             zero za0..za(T-1)
    ///             kc / 2 double steps                  [42 44 46 48]
    ///             one more step when kc is odd         [61 62 63 64]
    ///         epilogue of mode M0..M3                  [50] dispatch
    ///             for c in 0..ncols                    [51 52 53 54]
    ///                 for t in 0..T: slice c of za<t>, predicate p<t+1>
    ///                 next column                      [55 56 57 58]
    ///         next row panel                           [59]
    /// ```
    ///
    /// Register map:
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1` | `a` (A block) |
    /// | `x2` | `a_depth` in bytes (A step between depth `p` and `p + 1`) |
    /// | `x3` | `a_panel` in bytes (A step between 64-row panels) |
    /// | `x4` | `b` (packed B block) |
    /// | `x5` | `b_panel` in bytes (B step between 16-column panels) |
    /// | `x6` | `c` (`C(0, 0)`) |
    /// | `x7` | `ldc` in bytes |
    /// | `x9` | `i0`, first row of the row panel |
    /// | `x10` | A of the row panel: `a + (i0 / 64) · a_panel` |
    /// | `x11` | `C(i0, j0)` |
    /// | `w12` | `c`, slice index = column within the panel |
    /// | `x13` | depth double steps left; in the epilogue `C(i0, j0 + c)` |
    /// | `x14` | A at depth `p`; in the epilogue 16 (tile 1 row offset) |
    /// | `x15` | B row at depth `p`; in the epilogue 32 (tile 2 row offset) |
    /// | `x16` | A at depth `p + 1`, scratch; in the epilogue 48 (tile 3 row offset) |
    /// | `x20`, `x21`, `x22` | `m`, `n`, `kc` |
    /// | `w23` / `x23` | alpha bits in the prologue; then `rows = m - i0` |
    /// | `w24` / `x24` | beta bits in the prologue; then `ncols` |
    /// | `x25` | `mode` (0 M0, 1 M1, 2 M2, 3 M3) |
    /// | `x26` | `j0`, first column of the column panel |
    /// | `x27` | B of the column panel: `b + (j0 / 16) · b_panel` |
    /// | `x28` | `C(0, j0)` |
    /// | `z0`–`z3`, `z4`–`z7` | A vectors of tiles 0–3 at depth `p`, `p + 1` |
    /// | `z8`, `z9` | B rows at depth `p`, `p + 1` |
    /// | `z16`, `z17` | epilogue: the moved ZA slice, the loaded C |
    /// | `z28`, `z29` | alpha, beta broadcast |
    /// | `p0` | all-true |
    /// | `p1`–`p4` | rows `< m` of tiles 0–3 of the row panel |
    ///
    /// Epilogue of tile `t`, column `c` (`pt` = `p<t+1>`, `addr` =
    /// `C(i0 + 16t, j0 + c)`):
    ///
    /// ```text
    /// M0: st1w {za<t>h.s[w12, 0]}, pt, [addr]
    /// M1: mova z16.s, p0/m, za<t>h.s[w12, 0]; ld1w {z17.s}, pt/z, [addr]
    ///     fadd z16.s, pt/m, z16.s, z17.s;     st1w {z16.s}, pt, [addr]
    /// M2: mova z16.s, p0/m, za<t>h.s[w12, 0]; fmul z16.s, pt/m, z16.s, z28.s
    ///     st1w {z16.s}, pt, [addr]
    /// M3: mova z16.s, p0/m, za<t>h.s[w12, 0]; fmul z16.s, pt/m, z16.s, z28.s
    ///     ld1w {z17.s}, pt/z, [addr];         fmla z16.s, pt/m, z17.s, z29.s
    ///     st1w {z16.s}, pt, [addr]
    /// ```
    ///
    /// M3 multiplies by alpha and uses FMLA (`t + C · beta`) unconditionally:
    /// `acc · 1` is exact and `fma(C, 1, t)` equals `t + C` (rule R3).
    ///
    /// The kernel reads `a_depth`, `a_panel` and `b_panel` from the block
    /// and uses exactly those strides, so A read in place (`a_panel = 64`,
    /// `a_depth = lda`) works as well as the plain packing. It reads no A row
    /// `≥ m`, no B beyond panel `ceil(n / 16) - 1`, and no C outside
    /// `C[0..m, 0..n]` (none at all in M0 and M2); it writes only
    /// `C[0..m, 0..n]`. It never writes the argument block, never touches
    /// `a_pack` or `prefetch`, and leaves through the frame's exit label.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread:
    ///   SME2 with a 512-bit streaming vector length, not in streaming mode,
    ///   no live ZA state and no pending lazy save.
    /// - `m ≥ 1`, `n ≥ 1`, `kc ≥ 1`, `ldc ≥ m`, and `mode` must match
    ///   `alpha_bits` / `beta_bits` as `Epilogue::select` gives it.
    /// - Every address the layout of `args.rs` names must be valid: A for
    ///   `i < m`, `p < kc`; B for `p < kc`, `j < ceil(n / 16) · 16`; C for
    ///   `i < m`, `j < n` (read and written). C must not overlap A, B or the
    ///   argument block.
    pub(super) streaming fn sme2_gemm_f32(args: &MacroArgs<f32>) {
        extensions: [],
        consts: [
            off_a = OFF_A,
            off_a_depth = OFF_A_DEPTH,
            off_a_panel = OFF_A_PANEL,
            off_b = OFF_B,
            off_b_panel = OFF_B_PANEL,
            off_c = OFF_C,
            off_ldc = OFF_LDC,
            off_m = OFF_M,
            off_n = OFF_N,
            off_kc = OFF_KC,
            off_alpha = OFF_ALPHA,
            off_beta = OFF_BETA,
            off_mode = OFF_MODE,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_a_depth}]",
            "ldr x3, [x0, #{off_a_panel}]",
            "ldr x4, [x0, #{off_b}]",
            "ldr x5, [x0, #{off_b_panel}]",
            "ldr x6, [x0, #{off_c}]",
            "ldr x7, [x0, #{off_ldc}]",
            "ldr x20, [x0, #{off_m}]",
            "ldr x21, [x0, #{off_n}]",
            "ldr x22, [x0, #{off_kc}]",
            "ldr w23, [x0, #{off_alpha}]",
            "ldr w24, [x0, #{off_beta}]",
            "ldr x25, [x0, #{off_mode}]",
            "lsl x2, x2, #2",
            "lsl x3, x3, #2",
            "lsl x5, x5, #2",
            "lsl x7, x7, #2",
        ],
        body: [
            "ptrue p0.s",
            "dup z28.s, w23",
            "dup z29.s, w24",
            "mov x26, #0",
            "mov x27, x4",
            "mov x28, x6",
            "20:",
            "sub x24, x21, x26",
            "mov x16, #16",
            "cmp x24, x16",
            "csel x24, x24, x16, lo",
            "mov x9, #0",
            "mov x10, x1",
            "mov x11, x28",
            "30:",
            "whilelt p1.s, x9, x20",
            "add x16, x9, #16",
            "whilelt p2.s, x16, x20",
            "add x16, x9, #32",
            "whilelt p3.s, x16, x20",
            "add x16, x9, #48",
            "whilelt p4.s, x16, x20",
            "sub x23, x20, x9",
            "mov x14, x10",
            "mov x15, x27",
            "lsr x13, x22, #1",
            "cmp x23, #48",
            "b.hi 47f",
            "cmp x23, #32",
            "b.hi 45f",
            "cmp x23, #16",
            "b.hi 43f",
            "41:",
            "zero {{za0.s}}",
            "cbz x13, 61f",
            "42:",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z9.s}}, p0/z, [x15, #1, mul vl]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "add x16, x14, x2",
            "ld1w {{z4.s}}, p1/z, [x16]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "add x14, x16, x2",
            "add x15, x15, #128",
            "sub x13, x13, #1",
            "cbnz x13, 42b",
            "61:",
            "tbz x22, #0, 50f",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "b 50f",
            "43:",
            "zero {{za0.s, za1.s}}",
            "cbz x13, 62f",
            "44:",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z9.s}}, p0/z, [x15, #1, mul vl]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "add x16, x14, x2",
            "ld1w {{z4.s}}, p1/z, [x16]",
            "ld1w {{z5.s}}, p2/z, [x16, #1, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "add x14, x16, x2",
            "add x15, x15, #128",
            "sub x13, x13, #1",
            "cbnz x13, 44b",
            "62:",
            "tbz x22, #0, 50f",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "b 50f",
            "45:",
            "zero {{za0.s, za1.s, za2.s}}",
            "cbz x13, 63f",
            "46:",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z9.s}}, p0/z, [x15, #1, mul vl]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "ld1w {{z2.s}}, p3/z, [x14, #2, mul vl]",
            "add x16, x14, x2",
            "ld1w {{z4.s}}, p1/z, [x16]",
            "ld1w {{z5.s}}, p2/z, [x16, #1, mul vl]",
            "ld1w {{z6.s}}, p3/z, [x16, #2, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "fmopa za2.s, p0/m, p0/m, z9.s, z6.s",
            "add x14, x16, x2",
            "add x15, x15, #128",
            "sub x13, x13, #1",
            "cbnz x13, 46b",
            "63:",
            "tbz x22, #0, 50f",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "ld1w {{z2.s}}, p3/z, [x14, #2, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "b 50f",
            "47:",
            "zero {{za}}",
            "cbz x13, 64f",
            "48:",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z9.s}}, p0/z, [x15, #1, mul vl]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "ld1w {{z2.s}}, p3/z, [x14, #2, mul vl]",
            "ld1w {{z3.s}}, p4/z, [x14, #3, mul vl]",
            "add x16, x14, x2",
            "ld1w {{z4.s}}, p1/z, [x16]",
            "ld1w {{z5.s}}, p2/z, [x16, #1, mul vl]",
            "ld1w {{z6.s}}, p3/z, [x16, #2, mul vl]",
            "ld1w {{z7.s}}, p4/z, [x16, #3, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z8.s, z3.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "fmopa za2.s, p0/m, p0/m, z9.s, z6.s",
            "fmopa za3.s, p0/m, p0/m, z9.s, z7.s",
            "add x14, x16, x2",
            "add x15, x15, #128",
            "sub x13, x13, #1",
            "cbnz x13, 48b",
            "64:",
            "tbz x22, #0, 50f",
            "ld1w {{z8.s}}, p0/z, [x15]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "ld1w {{z1.s}}, p2/z, [x14, #1, mul vl]",
            "ld1w {{z2.s}}, p3/z, [x14, #2, mul vl]",
            "ld1w {{z3.s}}, p4/z, [x14, #3, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z8.s, z3.s",
            "50:",
            "mov w12, #0",
            "mov x13, x11",
            "mov x14, #16",
            "mov x15, #32",
            "mov x16, #48",
            "cbz x25, 51f",
            "cmp x25, #2",
            "b.lo 52f",
            "b.eq 53f",
            "b 54f",
            "51:",
            "st1w {{za0h.s[w12, 0]}}, p1, [x13]",
            "cmp x23, #16",
            "b.ls 55f",
            "st1w {{za1h.s[w12, 0]}}, p2, [x13, x14, lsl #2]",
            "cmp x23, #32",
            "b.ls 55f",
            "st1w {{za2h.s[w12, 0]}}, p3, [x13, x15, lsl #2]",
            "cmp x23, #48",
            "b.ls 55f",
            "st1w {{za3h.s[w12, 0]}}, p4, [x13, x16, lsl #2]",
            "55:",
            "add x13, x13, x7",
            "add w12, w12, #1",
            "cmp w12, w24",
            "b.lo 51b",
            "b 59f",
            "52:",
            "mova z16.s, p0/m, za0h.s[w12, 0]",
            "ld1w {{z17.s}}, p1/z, [x13]",
            "fadd z16.s, p1/m, z16.s, z17.s",
            "st1w {{z16.s}}, p1, [x13]",
            "cmp x23, #16",
            "b.ls 56f",
            "mova z16.s, p0/m, za1h.s[w12, 0]",
            "ld1w {{z17.s}}, p2/z, [x13, x14, lsl #2]",
            "fadd z16.s, p2/m, z16.s, z17.s",
            "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x23, #32",
            "b.ls 56f",
            "mova z16.s, p0/m, za2h.s[w12, 0]",
            "ld1w {{z17.s}}, p3/z, [x13, x15, lsl #2]",
            "fadd z16.s, p3/m, z16.s, z17.s",
            "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "cmp x23, #48",
            "b.ls 56f",
            "mova z16.s, p0/m, za3h.s[w12, 0]",
            "ld1w {{z17.s}}, p4/z, [x13, x16, lsl #2]",
            "fadd z16.s, p4/m, z16.s, z17.s",
            "st1w {{z16.s}}, p4, [x13, x16, lsl #2]",
            "56:",
            "add x13, x13, x7",
            "add w12, w12, #1",
            "cmp w12, w24",
            "b.lo 52b",
            "b 59f",
            "53:",
            "mova z16.s, p0/m, za0h.s[w12, 0]",
            "fmul z16.s, p1/m, z16.s, z28.s",
            "st1w {{z16.s}}, p1, [x13]",
            "cmp x23, #16",
            "b.ls 57f",
            "mova z16.s, p0/m, za1h.s[w12, 0]",
            "fmul z16.s, p2/m, z16.s, z28.s",
            "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x23, #32",
            "b.ls 57f",
            "mova z16.s, p0/m, za2h.s[w12, 0]",
            "fmul z16.s, p3/m, z16.s, z28.s",
            "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "cmp x23, #48",
            "b.ls 57f",
            "mova z16.s, p0/m, za3h.s[w12, 0]",
            "fmul z16.s, p4/m, z16.s, z28.s",
            "st1w {{z16.s}}, p4, [x13, x16, lsl #2]",
            "57:",
            "add x13, x13, x7",
            "add w12, w12, #1",
            "cmp w12, w24",
            "b.lo 53b",
            "b 59f",
            "54:",
            "mova z16.s, p0/m, za0h.s[w12, 0]",
            "fmul z16.s, p1/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p1/z, [x13]",
            "fmla z16.s, p1/m, z17.s, z29.s",
            "st1w {{z16.s}}, p1, [x13]",
            "cmp x23, #16",
            "b.ls 58f",
            "mova z16.s, p0/m, za1h.s[w12, 0]",
            "fmul z16.s, p2/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p2/z, [x13, x14, lsl #2]",
            "fmla z16.s, p2/m, z17.s, z29.s",
            "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x23, #32",
            "b.ls 58f",
            "mova z16.s, p0/m, za2h.s[w12, 0]",
            "fmul z16.s, p3/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p3/z, [x13, x15, lsl #2]",
            "fmla z16.s, p3/m, z17.s, z29.s",
            "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "cmp x23, #48",
            "b.ls 58f",
            "mova z16.s, p0/m, za3h.s[w12, 0]",
            "fmul z16.s, p4/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p4/z, [x13, x16, lsl #2]",
            "fmla z16.s, p4/m, z17.s, z29.s",
            "st1w {{z16.s}}, p4, [x13, x16, lsl #2]",
            "58:",
            "add x13, x13, x7",
            "add w12, w12, #1",
            "cmp w12, w24",
            "b.lo 54b",
            "59:",
            "add x9, x9, #64",
            "add x10, x10, x3",
            "add x11, x11, #256",
            "cmp x9, x20",
            "b.lo 30b",
            "add x26, x26, #16",
            "add x27, x27, x5",
            "add x28, x28, x7, lsl #4",
            "cmp x26, x21",
            "b.lo 20b",
        ],
    }
}
