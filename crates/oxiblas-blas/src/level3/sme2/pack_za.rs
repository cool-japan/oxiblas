//! Packing through ZA for the SME2 path.
//!
//! The plain packers of `pack.rs` gather strided elements for two operand
//! orientations: `op(B) = B` (a packed row of B crosses the stored columns)
//! and `op(A) = Aᵀ` (a packed depth step of A crosses the stored columns).
//! Both are the same operation, the transposition of a column-major block
//! into zero-padded row-major rows, which a ZA tile does sixteen (`f32`) or
//! eight (`f64`) rows at a time: the columns are loaded into the horizontal
//! slices of a tile and the vertical slices are stored as packed rows.
//!
//! # The primitive
//!
//! For a stored, column-major block `src` (element `(p, c)` at
//! `src[p + c · ld]`) of `d` rows and `w` columns, and a panel width `W`:
//!
//! ```text
//! dst[(c / W) · W · d + p · W + c % W] = src[p + c · ld]   for p < d, c < w
//! dst[(c / W) · W · d + p · W + c % W] = +0                for p < d, w ≤ c < ceil(w / W) · W
//! ```
//!
//! - `op(B) = B` ([`pack_b_za`]): the stored matrix is B (`k × n`), the
//!   block is `B[depth, cols]`, `W = NR` (16 for `f32`, 8 for `f64`): the
//!   result is the plain packed B block of `pack::pack_b`.
//! - `op(A) = Aᵀ` ([`pack_a_za`]): the stored matrix is `k × m`, the block is
//!   its `[depth, rows]`, `W = 64`: packed row `p` of panel `i / 64` is depth
//!   step `p` of the 64 rows of op(A), the plain packed A block of
//!   `pack::pack_a`.
//!
//! The other two orientations (`op(B) = Bᵀ`, `op(A) = A`) are contiguous
//! copies; the entry points hand them to the plain packers unchanged.
//!
//! # Layout in ZA
//!
//! With a 512-bit streaming vector length a `.s` tile is 16 × 16 and a `.d`
//! tile 8 × 8. The kernel walks GROUPS of 64 consecutive block columns (the
//! width of one A panel, four B panels for `f32`, eight for `f64`), and in
//! each group CHUNKS of `TR` depth steps (`TR` = 16 for `f32`, 8 for
//! `f64`). Inside a chunk, tile `t` (`za0`–`za3` `.s`, `za0`–`za7` `.d`)
//! holds the group's columns `TR · t .. TR · t + TR - 1`:
//!
//! - **Loads.** Horizontal slice `s` of tile `t` receives the `TR` depth
//!   steps `p0 .. p0 + TR - 1` of group column `c = TR · t + s` (one
//!   contiguous vector of the source column). The load is governed by
//!   `psel(depth, live[c])`: the depth predicate `whilelt(p0, d)` when column
//!   `c` of the group is a column of the block (`whilelt(c0, w)` over the 64
//!   byte lanes of a predicate, one lane per group column), all-false
//!   otherwise. So no element outside the block is ever read, and the slices
//!   of a dead column or a dead depth lane are zero (`/z`).
//! - **Stores.** Vertical slice `q` of tile `t` is depth step `p0 + q` of the
//!   tile's `TR` columns, i.e. a run of `TR` consecutive elements of packed
//!   row `p0 + q`. It is stored whole (all-true predicate: live columns and
//!   the zero padding of a partial panel together) for `q < min(TR, d -
//!   p0)`. For B a group spans several panels, and only the tiles of panels
//!   that exist are stored (`ceil(live / NR)` tiles); for A every tile of the
//!   group belongs to the one panel and is stored.
//!
//! The destination steps are passed in the argument block: `width` (the
//! packed row, `W`), `tile_step` (from one tile's columns to the next: `TR`
//! inside an A panel, `NR · d` from one B panel to the next); a group always
//! advances `64 · d` elements.
//!
//! The transposers only move bits: no floating-point instruction appears in
//! them (the template tests check this), so NaN payloads, signed zeros and
//! subnormals reach the buffer unchanged and FPSR is the caller's after a
//! call (the frame of `asm.rs` adds no flags when the body raises none).
//!
//! The tests live in `tests_pack_za.rs`. Test builds also hold `model`,
//! an interpreter of the registered templates that the tests use to run
//! them natively with every memory access checked (it lives here, next to
//! the templates it models, because the test file is near its size limit).

use core::mem::offset_of;

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

use super::args::{MR, SmeElement};
use super::asm::sme_kernels;
use super::pack::{
    self, ColMajor, PackedA, PackedB, Span, a_block_len, a_pack_len, b_block_len, b_pack_len,
};
use crate::level3::trsm::Trans;

/// Columns of one group of the transposers: one A panel (`MR` = 64).
pub(super) const GROUP_COLS: usize = MR;

/// The argument block of one transposer call (all sizes in ELEMENTS).
///
/// Every field is 8 bytes wide, so the offsets are the same for `f32` and
/// `f64` (the `OFF_*` constants are taken from `ZaPackArgs<f32>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct ZaPackArgs<T> {
    /// Element `(0, 0)` of the stored, column-major block.
    pub(super) src: *const T,
    /// Leading dimension of the stored block (column to column).
    pub(super) ld: usize,
    /// The packed buffer (`ceil(cols / width) · width · depth` elements).
    pub(super) dst: *mut T,
    /// Rows of the stored block = depth of the packed block (`≥ 1`).
    pub(super) depth: usize,
    /// Columns of the stored block (`≥ 1`).
    pub(super) cols: usize,
    /// Packed row width `W`: `T::NR` (B) or 64 (A).
    pub(super) width: usize,
    /// Destination step from one tile's columns to the next tile's:
    /// `T::TILE_ROWS` when `width = 64`, `width · depth` when
    /// `width = T::TILE_ROWS`.
    pub(super) tile_step: usize,
}

/// Offset of [`ZaPackArgs::src`].
pub(super) const OFF_SRC: usize = offset_of!(ZaPackArgs<f32>, src);
/// Offset of [`ZaPackArgs::ld`].
pub(super) const OFF_LD: usize = offset_of!(ZaPackArgs<f32>, ld);
/// Offset of [`ZaPackArgs::dst`].
pub(super) const OFF_DST: usize = offset_of!(ZaPackArgs<f32>, dst);
/// Offset of [`ZaPackArgs::depth`].
pub(super) const OFF_DEPTH: usize = offset_of!(ZaPackArgs<f32>, depth);
/// Offset of [`ZaPackArgs::cols`].
pub(super) const OFF_COLS: usize = offset_of!(ZaPackArgs<f32>, cols);
/// Offset of [`ZaPackArgs::width`].
pub(super) const OFF_WIDTH: usize = offset_of!(ZaPackArgs<f32>, width);
/// Offset of [`ZaPackArgs::tile_step`].
pub(super) const OFF_TILE_STEP: usize = offset_of!(ZaPackArgs<f32>, tile_step);

// The kernels read the fields of either element type at these offsets.
const _: () = {
    assert!(OFF_SRC == offset_of!(ZaPackArgs<f64>, src));
    assert!(OFF_LD == offset_of!(ZaPackArgs<f64>, ld));
    assert!(OFF_DST == offset_of!(ZaPackArgs<f64>, dst));
    assert!(OFF_DEPTH == offset_of!(ZaPackArgs<f64>, depth));
    assert!(OFF_COLS == offset_of!(ZaPackArgs<f64>, cols));
    assert!(OFF_WIDTH == offset_of!(ZaPackArgs<f64>, width));
    assert!(OFF_TILE_STEP == offset_of!(ZaPackArgs<f64>, tile_step));
    assert!(core::mem::size_of::<ZaPackArgs<f32>>() == 7 * 8);
};

/// A transposer kernel (the real ones, or a stand-in of the tests).
pub(super) type ZaKernelFn<T> = unsafe fn(&ZaPackArgs<T>);

/// The element types the ZA packers serve, with their transposer kernel.
pub(super) trait SmePack: SmeElement + Field + bytemuck::Zeroable {
    /// The transposer of this element type.
    const TRANSPOSE: ZaKernelFn<Self>;
}

impl SmePack for f32 {
    const TRANSPOSE: ZaKernelFn<Self> = za_transpose_f32;
}

impl SmePack for f64 {
    const TRANSPOSE: ZaKernelFn<Self> = za_transpose_f64;
}

sme_kernels! {
    /// Transposes one column-major `f32` block into zero-padded row-major
    /// panels of `width` columns, in one streaming region (the primitive and
    /// the ZA layout: the module docs).
    ///
    /// Loop nest (labels in brackets):
    ///
    /// ```text
    /// for c0 in (0..cols).step_by(64)                  [20] group
    ///     p2.b  = whilelt(c0, cols)                    live group columns
    ///     store = min(64, ceil(cols / width) · width - c0)
    ///     for p0 in (0..depth).step_by(16)             [30] chunk
    ///         p1.s = whilelt(p0, depth)                live depth lanes
    ///         for t in 0..4, s in 0..16                [41 42 43 44]
    ///             p5 = psel(p1, p2.b[16t + s])
    ///             za<t>h.s[s] = load(p5/z, column c0 + 16t + s at p0)
    ///         rows = min(16, depth - p0)
    ///         for t in 0..4 while 16t < store          [51 52 53 54]
    ///             for q in 0..rows: store za<t>v.s[q] to packed row p0 + q
    ///                                                  [59] next chunk
    /// ```
    ///
    /// Register map:
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1` | `src` |
    /// | `x2` | `ld` in bytes |
    /// | `x3` | `dst` |
    /// | `x4` | `depth` |
    /// | `x5` | `cols` |
    /// | `x6` | `width` in bytes (packed row step) |
    /// | `x7` | `tile_step` in bytes |
    /// | `x8` | `width` |
    /// | `x9` | `c0`, first column of the group |
    /// | `x10` | source column `c0`: `src + c0 · ld` |
    /// | `x11` | destination of the group: `dst + c0 · depth` |
    /// | `w12` | tile slice: `s` (loads), `q` (stores) |
    /// | `w13` | column within the group, `16t + s` (the `psel` index) |
    /// | `x14` | `ceil(cols / width) · width`, the end of the stored columns |
    /// | `x20` | `64 · depth` in bytes: the destination step of a group |
    /// | `x15`, `x16` | scratch |
    /// | `x21` | columns of the group that are stored (live and padding) |
    /// | `x22` | `p0`, first depth step of the chunk (the load index) |
    /// | `x23` | packed row `p0` of the group |
    /// | `x24` | source column `c0 + 16t + s` |
    /// | `x25` | `rows = min(16, depth - p0)` |
    /// | `x26` | packed row `p0 + q` of tile `t`'s columns |
    /// | `x27` | packed row `p0` of tile `t`'s columns |
    /// | `p0` | all-true (stores) |
    /// | `p1` | live depth lanes of the chunk |
    /// | `p2` | live columns of the group (`.b`, one lane per column) |
    /// | `p5` | governs each load: `p1` for a live column, else all-false |
    ///
    /// It reads only `src[p + c · ld]` for `p < depth`, `c < cols` and writes
    /// only `dst[0 .. ceil(cols / width) · width · depth]`, every element of
    /// it once per covering tile (padding as `+0`). It never writes the
    /// argument block, executes no floating-point instruction and leaves
    /// through the frame's exit label.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread:
    ///   SME2 with a 512-bit streaming vector length, not in streaming mode,
    ///   no live ZA state and no pending lazy save.
    /// - `depth ≥ 1`, `cols ≥ 1`, `width` is 16 or 64 and `tile_step` is
    ///   `16` when `width = 64`, `16 · depth` when `width = 16`.
    /// - `src[p + c · ld]` must be readable for `p < depth`, `c < cols`, and
    ///   `dst` writable for `ceil(cols / width) · width · depth` elements
    ///   that overlap neither the source nor the argument block.
    pub(super) streaming fn za_transpose_f32(args: &ZaPackArgs<f32>) {
        extensions: [],
        consts: [
            off_src = OFF_SRC,
            off_ld = OFF_LD,
            off_dst = OFF_DST,
            off_depth = OFF_DEPTH,
            off_cols = OFF_COLS,
            off_width = OFF_WIDTH,
            off_tile_step = OFF_TILE_STEP,
        ],
        prologue: [
            "ldr x1, [x0, #{off_src}]",
            "ldr x2, [x0, #{off_ld}]",
            "ldr x3, [x0, #{off_dst}]",
            "ldr x4, [x0, #{off_depth}]",
            "ldr x5, [x0, #{off_cols}]",
            "ldr x8, [x0, #{off_width}]",
            "ldr x7, [x0, #{off_tile_step}]",
            "lsl x2, x2, #2",
            "lsl x6, x8, #2",
            "lsl x7, x7, #2",
            "lsl x20, x4, #8",
            "add x14, x5, x8",
            "sub x14, x14, #1",
            "neg x15, x8",
            "and x14, x14, x15",
        ],
        body: [
            "ptrue p0.s",
            "mov x9, #0",
            "mov x10, x1",
            "mov x11, x3",
            "20:",
            "whilelt p2.b, x9, x5",
            "sub x21, x14, x9",
            "mov x16, #64",
            "cmp x21, x16",
            "csel x21, x21, x16, lo",
            "mov x22, #0",
            "mov x23, x11",
            "30:",
            "whilelt p1.s, x22, x4",
            "mov x24, x10",
            "mov w13, #0",
            "mov w12, #0",
            "41:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1w {{za0h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lo 41b",
            "mov w12, #0",
            "42:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1w {{za1h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lo 42b",
            "mov w12, #0",
            "43:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1w {{za2h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lo 43b",
            "mov w12, #0",
            "44:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1w {{za3h.s[w12, 0]}}, p5/z, [x24, x22, lsl #2]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lo 44b",
            "sub x25, x4, x22",
            "mov x16, #16",
            "cmp x25, x16",
            "csel x25, x25, x16, lo",
            "mov x27, x23",
            "mov x26, x27",
            "mov w12, #0",
            "51:",
            "st1w {{za0v.s[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 51b",
            "cmp x21, #16",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "52:",
            "st1w {{za1v.s[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 52b",
            "cmp x21, #32",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "53:",
            "st1w {{za2v.s[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 53b",
            "cmp x21, #48",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "54:",
            "st1w {{za3v.s[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 54b",
            "59:",
            "add x23, x23, x6, lsl #4",
            "add x22, x22, #16",
            "cmp x22, x4",
            "b.lo 30b",
            "add x9, x9, #64",
            "add x10, x10, x2, lsl #6",
            "add x11, x11, x20",
            "cmp x9, x5",
            "b.lo 20b",
        ],
    }

    /// Transposes one column-major `f64` block into zero-padded row-major
    /// panels of `width` columns, in one streaming region (the primitive and
    /// the ZA layout: the module docs). The structure is that of
    /// `za_transpose_f32` with 8 × 8 `.d` tiles: eight tiles of eight
    /// columns per 64-column group, chunks of eight depth steps.
    ///
    /// Loop nest (labels in brackets):
    ///
    /// ```text
    /// for c0 in (0..cols).step_by(64)                  [20] group
    ///     p2.b  = whilelt(c0, cols)                    live group columns
    ///     store = min(64, ceil(cols / width) · width - c0)
    ///     for p0 in (0..depth).step_by(8)              [30] chunk
    ///         p1.d = whilelt(p0, depth)                live depth lanes
    ///         for t in 0..8, s in 0..8                 [41 .. 48]
    ///             p5 = psel(p1, p2.b[8t + s])
    ///             za<t>h.d[s] = load(p5/z, column c0 + 8t + s at p0)
    ///         rows = min(8, depth - p0)
    ///         for t in 0..8 while 8t < store           [51 .. 58]
    ///             for q in 0..rows: store za<t>v.d[q] to packed row p0 + q
    ///                                                  [59] next chunk
    /// ```
    ///
    /// Register map: as `za_transpose_f32`, with byte scales of 8 (`x2`,
    /// `x6`, `x7`; `x20` = `64 · depth · 8`), `w13 = 8t + s`, and
    /// `x25 = min(8, depth - p0)`.
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1`–`x8` | `src`, `ld` (bytes), `dst`, `depth`, `cols`, `width` (bytes), `tile_step` (bytes), `width` |
    /// | `x9`, `x10`, `x11` | `c0`, source column `c0`, destination of the group |
    /// | `w12`, `w13` | tile slice; column within the group |
    /// | `x14`, `x20` | `ceil(cols / width) · width`; the group step `64 · depth · 8` in bytes |
    /// | `x15`, `x16` | scratch |
    /// | `x21`, `x22`, `x23` | stored columns of the group, `p0`, packed row `p0` |
    /// | `x24`–`x27` | source column, rows, packed row of tile `t` at `p0 + q`, at `p0` |
    /// | `p0`, `p1`, `p2`, `p5` | all-true, live depth, live columns (`.b`), load governor |
    ///
    /// It reads only `src[p + c · ld]` for `p < depth`, `c < cols` and writes
    /// only `dst[0 .. ceil(cols / width) · width · depth]`. It never writes
    /// the argument block, executes no floating-point instruction and leaves
    /// through the frame's exit label. (Tile-slice moves are base SME: no
    /// `sme-f64f64` is needed.)
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread.
    /// - `depth ≥ 1`, `cols ≥ 1`, `width` is 8 or 64 and `tile_step` is `8`
    ///   when `width = 64`, `8 · depth` when `width = 8`.
    /// - `src[p + c · ld]` must be readable for `p < depth`, `c < cols`, and
    ///   `dst` writable for `ceil(cols / width) · width · depth` elements
    ///   that overlap neither the source nor the argument block.
    pub(super) streaming fn za_transpose_f64(args: &ZaPackArgs<f64>) {
        extensions: [],
        consts: [
            off_src = OFF_SRC,
            off_ld = OFF_LD,
            off_dst = OFF_DST,
            off_depth = OFF_DEPTH,
            off_cols = OFF_COLS,
            off_width = OFF_WIDTH,
            off_tile_step = OFF_TILE_STEP,
        ],
        prologue: [
            "ldr x1, [x0, #{off_src}]",
            "ldr x2, [x0, #{off_ld}]",
            "ldr x3, [x0, #{off_dst}]",
            "ldr x4, [x0, #{off_depth}]",
            "ldr x5, [x0, #{off_cols}]",
            "ldr x8, [x0, #{off_width}]",
            "ldr x7, [x0, #{off_tile_step}]",
            "lsl x2, x2, #3",
            "lsl x6, x8, #3",
            "lsl x7, x7, #3",
            "lsl x20, x4, #9",
            "add x14, x5, x8",
            "sub x14, x14, #1",
            "neg x15, x8",
            "and x14, x14, x15",
        ],
        body: [
            "ptrue p0.d",
            "mov x9, #0",
            "mov x10, x1",
            "mov x11, x3",
            "20:",
            "whilelt p2.b, x9, x5",
            "sub x21, x14, x9",
            "mov x16, #64",
            "cmp x21, x16",
            "csel x21, x21, x16, lo",
            "mov x22, #0",
            "mov x23, x11",
            "30:",
            "whilelt p1.d, x22, x4",
            "mov x24, x10",
            "mov w13, #0",
            "mov w12, #0",
            "41:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za0h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 41b",
            "mov w12, #0",
            "42:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za1h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 42b",
            "mov w12, #0",
            "43:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za2h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 43b",
            "mov w12, #0",
            "44:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za3h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 44b",
            "mov w12, #0",
            "45:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za4h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 45b",
            "mov w12, #0",
            "46:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za5h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 46b",
            "mov w12, #0",
            "47:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za6h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 47b",
            "mov w12, #0",
            "48:",
            "psel p5, p1, p2.b[w13, 0]",
            "ld1d {{za7h.d[w12, 0]}}, p5/z, [x24, x22, lsl #3]",
            "add x24, x24, x2",
            "add w13, w13, #1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.lo 48b",
            "sub x25, x4, x22",
            "mov x16, #8",
            "cmp x25, x16",
            "csel x25, x25, x16, lo",
            "mov x27, x23",
            "mov x26, x27",
            "mov w12, #0",
            "51:",
            "st1d {{za0v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 51b",
            "cmp x21, #8",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "52:",
            "st1d {{za1v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 52b",
            "cmp x21, #16",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "53:",
            "st1d {{za2v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 53b",
            "cmp x21, #24",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "54:",
            "st1d {{za3v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 54b",
            "cmp x21, #32",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "55:",
            "st1d {{za4v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 55b",
            "cmp x21, #40",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "56:",
            "st1d {{za5v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 56b",
            "cmp x21, #48",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "57:",
            "st1d {{za6v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 57b",
            "cmp x21, #56",
            "b.ls 59f",
            "add x27, x27, x7",
            "mov x26, x27",
            "mov w12, #0",
            "58:",
            "st1d {{za7v.d[w12, 0]}}, p0, [x26]",
            "add x26, x26, x6",
            "add w12, w12, #1",
            "cmp w12, w25",
            "b.lo 58b",
            "59:",
            "add x23, x23, x6, lsl #3",
            "add x22, x22, #8",
            "cmp x22, x4",
            "b.lo 30b",
            "add x9, x9, #64",
            "add x10, x10, x2, lsl #6",
            "add x11, x11, x20",
            "cmp x9, x5",
            "b.lo 20b",
        ],
    }
}

/// The argument block that transposes the stored block of `stored` at
/// `(row0, col0)` (`depth` rows, `cols` columns) into `dst` in panels of
/// `width` columns. `width` is `T::TILE_ROWS` or [`GROUP_COLS`]; the caller
/// checked the block against the view and the buffer length.
fn transpose_args<T: SmeElement + Field>(
    stored: &MatRef<'_, T>,
    row0: usize,
    col0: usize,
    depth: usize,
    cols: usize,
    width: usize,
    dst: *mut T,
) -> ZaPackArgs<T> {
    let tile_step = if width == GROUP_COLS {
        T::TILE_ROWS
    } else {
        width * depth
    };
    ZaPackArgs {
        src: stored.ptr_at(row0, col0),
        ld: stored.row_stride(),
        dst,
        depth,
        cols,
        width,
        tile_step,
    }
}

/// [`pack_b_za`] with a given transposer (the real kernel or a stand-in).
///
/// # Safety
///
/// `kernel` must be callable under the caller's conditions (for the real
/// kernels: as for [`pack_b_za`]).
pub(super) unsafe fn pack_b_za_with<'p, T: SmePack>(
    kernel: ZaKernelFn<T>,
    b: &ColMajor<'_, T>,
    depth: Span,
    cols: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedB<'p, T>> {
    if b.trans() != Trans::NoTrans {
        return pack::pack_b(b, depth, cols, buf);
    }
    let len = b_block_len(b, depth, cols, buf.len())?;
    // The transposer puts one panel of B into one ZA tile, so the packed
    // width of B must be the tile height.
    const { assert!(T::NR == T::TILE_ROWS) };
    // `op(B) = B`: op(B)[depth.start + p, cols.start + c] is element
    // (depth.start + p, cols.start + c) of the stored matrix.
    let args = transpose_args(
        b.stored(),
        depth.start,
        cols.start,
        depth.len,
        cols.len,
        T::NR,
        buf.as_mut_ptr(),
    );
    // SAFETY: `b_block_len` checked that both spans are non-empty and inside
    // op(B) = the stored matrix, so `src[p + c · ld]` is an element of the
    // view for `p < depth.len`, `c < cols.len`; and that `buf` holds at least
    // `len = ceil(cols / NR) · NR · depth` elements, exactly what the kernel
    // writes with `width = NR` and `tile_step = NR · depth`. `buf` is
    // borrowed exclusively, so it overlaps neither the view nor `args`. The
    // SME2 conditions are the caller's.
    unsafe { kernel(&args) };
    PackedB::from_packed(buf.as_slice().get(..len)?, depth.len, cols.len)
}

/// [`pack_a_za`] with a given transposer (the real kernel or a stand-in).
///
/// # Safety
///
/// `kernel` must be callable under the caller's conditions (for the real
/// kernels: as for [`pack_a_za`]).
pub(super) unsafe fn pack_a_za_with<'p, T: SmePack>(
    kernel: ZaKernelFn<T>,
    a: &ColMajor<'_, T>,
    rows: Span,
    depth: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedA<'p, T>> {
    if a.trans() != Trans::Trans {
        return pack::pack_a(a, rows, depth, buf);
    }
    let len = a_block_len(a, rows, depth, buf.len())?;
    // `op(A) = Aᵀ`: op(A)[rows.start + i, depth.start + p] is element
    // (depth.start + p, rows.start + i) of the stored matrix, so the stored
    // block has `depth.len` rows and `rows.len` columns.
    let args = transpose_args(
        a.stored(),
        depth.start,
        rows.start,
        depth.len,
        rows.len,
        GROUP_COLS,
        buf.as_mut_ptr(),
    );
    // SAFETY: `a_block_len` checked that both spans are non-empty and inside
    // op(A), so `src[p + c · ld]` is an element of the stored view for
    // `p < depth.len`, `c < rows.len`; and that `buf` holds at least
    // `len = ceil(rows / 64) · 64 · depth` elements, exactly what the kernel
    // writes with `width = 64`. `buf` is borrowed exclusively. The SME2
    // conditions are the caller's.
    unsafe { kernel(&args) };
    PackedA::from_packed(buf.as_slice().get(..len)?, rows.len, depth.len)
}

/// `pack::pack_b`'s result (the same `Option`, the same buffer contents bit
/// for bit) through the ZA transposer when `b.trans()` is `NoTrans`;
/// delegates to `pack::pack_b` otherwise.
///
/// `None`, with `buf` untouched, exactly when `pack::pack_b` returns `None`.
/// Only the first `b_pack_len(depth.len, cols.len)` elements of `buf` are
/// written; nothing outside the block of B is read; FPSR is unchanged; the
/// thread leaves with SVCR = 0 and TPIDR2_EL0 = 0.
///
/// # Safety
///
/// `detect::thread_ready()` returned `true` on this thread and
/// `detect::enabled::<T>()` holds (the kernels' contract). Everything else
/// is checked.
pub(super) unsafe fn pack_b_za<'p, T: SmePack>(
    b: &ColMajor<'_, T>,
    depth: Span,
    cols: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedB<'p, T>> {
    // SAFETY: the real transposer, under this function's contract.
    unsafe { pack_b_za_with(T::TRANSPOSE, b, depth, cols, buf) }
}

/// `pack::pack_a`'s result through the ZA transposer when `a.trans()` is
/// `Trans`; delegates otherwise.
///
/// `None`, with `buf` untouched, exactly when `pack::pack_a` returns `None`.
/// Only the first `a_pack_len(rows.len, depth.len)` elements of `buf` are
/// written; nothing outside the block of A is read; FPSR is unchanged; the
/// thread leaves with SVCR = 0 and TPIDR2_EL0 = 0.
///
/// # Safety
///
/// As for [`pack_b_za`].
pub(super) unsafe fn pack_a_za<'p, T: SmePack>(
    a: &ColMajor<'_, T>,
    rows: Span,
    depth: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedA<'p, T>> {
    // SAFETY: the real transposer, under this function's contract.
    unsafe { pack_a_za_with(T::TRANSPOSE, a, rows, depth, buf) }
}

// ---------------------------------------------------------------------------
// First-use check
// ---------------------------------------------------------------------------

/// One fixed block of the first-use check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CheckBlock {
    /// `true`: `op(A) = Aᵀ` through [`pack_a_za`]; `false`: `op(B) = B`
    /// through [`pack_b_za`].
    pub(super) a_side: bool,
    /// Depth of the block.
    pub(super) depth: usize,
    /// Columns (B) or rows (A) of the block.
    pub(super) width: usize,
    /// Extra rows of the stored matrix beyond the rows a chunk can address
    /// (the block starts at stored row 1; `ld = 1 + roundup(depth, 16) +
    /// ld_pad`).
    pub(super) ld_pad: usize,
}

/// The blocks of the first-use check: full panels and groups, partial
/// panels, depth tails (`depth % 16` and `% 8` non-zero), a single element,
/// several groups, blocks deeper than wide (so that a depth predicate bound
/// by the wrong count loses elements), and leading dimensions larger than
/// the block; each block starts inside a larger stored matrix.
pub(super) const CHECK_BLOCKS: [CheckBlock; 12] = [
    CheckBlock {
        a_side: false,
        depth: 64,
        width: 64,
        ld_pad: 0,
    },
    CheckBlock {
        a_side: false,
        depth: 17,
        width: 35,
        ld_pad: 3,
    },
    CheckBlock {
        a_side: false,
        depth: 1,
        width: 1,
        ld_pad: 0,
    },
    CheckBlock {
        a_side: false,
        depth: 33,
        width: 130,
        ld_pad: 5,
    },
    CheckBlock {
        a_side: false,
        depth: 9,
        width: 9,
        ld_pad: 1,
    },
    CheckBlock {
        a_side: true,
        depth: 16,
        width: 64,
        ld_pad: 0,
    },
    CheckBlock {
        a_side: true,
        depth: 17,
        width: 70,
        ld_pad: 2,
    },
    CheckBlock {
        a_side: true,
        depth: 1,
        width: 1,
        ld_pad: 0,
    },
    CheckBlock {
        a_side: true,
        depth: 23,
        width: 129,
        ld_pad: 7,
    },
    CheckBlock {
        a_side: true,
        depth: 8,
        width: 9,
        ld_pad: 1,
    },
    CheckBlock {
        a_side: false,
        depth: 41,
        width: 3,
        ld_pad: 2,
    },
    CheckBlock {
        a_side: true,
        depth: 41,
        width: 5,
        ld_pad: 0,
    },
];

/// Stored columns before the block (the block starts at stored column 2).
const CHECK_COL0: usize = 2;

/// The largest depth chunk of the transposers (`f32`: 16 steps).
const CHECK_CHUNK: usize = 16;

/// The bits of element `index` of a check matrix: inside the block a
/// pattern with special values (signalling and quiet NaNs with distinct
/// payloads, `-0`, subnormals, infinities) and position-encoding normal
/// numbers; outside it a signalling NaN with a distinct payload.
pub(super) fn check_bits<T: SmeElement>(index: usize, in_block: bool) -> u64 {
    let wide = core::mem::size_of::<T>() == 8;
    let i = index as u64;
    if !in_block {
        return if wide {
            0x7ff0_0000_0000_0000 | (1 << 50) | (i & ((1 << 50) - 1))
        } else {
            0x7f80_0000 | (1 << 21) | (i & ((1 << 21) - 1))
        };
    }
    match (index % 11, wide) {
        (0, false) => 0x7f80_0000 | (1 << 20) | (i & 0xf_ffff),
        (0, true) => 0x7ff0_0000_0000_0000 | (1 << 49) | (i & ((1 << 49) - 1)),
        (3, false) => 0xffc0_0000 | (i & 0x3f_ffff),
        (3, true) => 0xfff8_0000_0000_0000 | (i & ((1 << 51) - 1)),
        (5, false) => 0x8000_0000,
        (5, true) => 0x8000_0000_0000_0000,
        (7, false) => 1 + (i & 0x7f_fffe),
        (7, true) => 1 + (i & ((1 << 52) - 2)),
        (9, false) => 0x7f80_0000 | (i & 0x8000_0000),
        (9, true) => 0x7ff0_0000_0000_0000 | ((i & 1) << 63),
        (_, false) => 0x4000_0000 | (i & 0x3f_ffff),
        (_, true) => 0x4000_0000_0000_0000 | (i & ((1 << 52) - 1)),
    }
}

/// Runs one check block through `kernel` and through the plain packer and
/// compares the buffers bit for bit (the prefix and the guard after it).
///
/// The check exists for hardware whose behaviour may deviate from the
/// architecture as QEMU models it, so it must report a deviation instead of
/// faulting: the stored matrix extends over every address the transposer
/// forms (each group's 64 columns, each chunk's whole depth), filled with
/// signalling NaNs outside the block, and the buffer carries a guard after
/// the packed length over everything a group can store (`64 · (depth + 1)`
/// elements). A predicate that let a dead lane or a dead column through
/// then shows up as wrong bits.
///
/// # Safety
///
/// As for [`self_check_with`].
unsafe fn check_block<T: SmePack>(kernel: ZaKernelFn<T>, block: &CheckBlock) -> bool {
    let srows = 1 + block.depth.next_multiple_of(CHECK_CHUNK) + block.ld_pad;
    let scols = CHECK_COL0 + block.width.next_multiple_of(GROUP_COLS) + 1;
    let Some(guard_len) = block
        .depth
        .checked_add(1)
        .and_then(|d| d.checked_mul(GROUP_COLS))
    else {
        return false;
    };
    let Some(total) = srows.checked_mul(scols) else {
        return false;
    };
    let mut data = AlignedVec::<T>::zeros(total);
    for (index, slot) in data.as_mut_slice().iter_mut().enumerate() {
        let (r, c) = (index % srows, index / srows);
        let in_block =
            (1..=block.depth).contains(&r) && (CHECK_COL0..CHECK_COL0 + block.width).contains(&c);
        *slot = T::from_bits64(check_bits::<T>(index, in_block));
    }
    let Some(view) = MatRef::from_strided(data.as_slice(), srows, scols, srows, 1) else {
        return false;
    };
    let trans = if block.a_side {
        Trans::Trans
    } else {
        Trans::NoTrans
    };
    let Some(op) = ColMajor::normalize(view, trans) else {
        return false;
    };
    let depth = Span {
        start: 1,
        len: block.depth,
    };
    let span = Span {
        start: CHECK_COL0,
        len: block.width,
    };
    let len = if block.a_side {
        a_pack_len(block.width, block.depth)
    } else {
        b_pack_len::<T>(block.depth, block.width)
    };
    let Some(len) = len else {
        return false;
    };
    let mut want = AlignedVec::<T>::zeros(len);
    let Some(total_len) = len.checked_add(guard_len) else {
        return false;
    };
    let mut got = AlignedVec::<T>::zeros(total_len);
    for (index, slot) in got.as_mut_slice().iter_mut().enumerate() {
        // Stale contents: `-0` and signalling NaNs, which must not survive
        // in the padding or anywhere else in the prefix.
        // (`check_bits(5, true)` is `-0`; the out-of-block pattern is a
        // signalling NaN.)
        let stale = if index.is_multiple_of(2) {
            T::from_bits64(check_bits::<T>(5, true))
        } else {
            T::from_bits64(check_bits::<T>(index + 1, false))
        };
        *slot = if index < len {
            stale
        } else {
            T::from_bits64(check_bits::<T>(index, false))
        };
    }
    let plain = if block.a_side {
        pack::pack_a(&op, span, depth, &mut want).is_some()
    } else {
        pack::pack_b(&op, depth, span, &mut want).is_some()
    };
    // SAFETY: forwarded from this function's contract; the block lies inside
    // the view and `got` is longer than the packed length.
    let packed = unsafe {
        if block.a_side {
            pack_a_za_with(kernel, &op, span, depth, &mut got).is_some()
        } else {
            pack_b_za_with(kernel, &op, depth, span, &mut got).is_some()
        }
    };
    if !plain || !packed {
        return false;
    }
    let prefix = got
        .as_slice()
        .iter()
        .zip(want.as_slice())
        .all(|(g, w)| g.bits() == w.bits());
    let guard = got
        .as_slice()
        .iter()
        .enumerate()
        .skip(len)
        .all(|(index, g)| g.bits() == check_bits::<T>(index, false));
    prefix && guard
}

/// The first-use check of one transposer: every [`CHECK_BLOCKS`] entry
/// packed through `kernel` equals the plain packer's buffer bit for bit,
/// and the guard after it is untouched. `true` = all equal.
///
/// # Safety
///
/// `kernel` must be callable on this thread with any valid argument block
/// (for the real kernels: `detect::thread_ready()` returned `true` and
/// `detect::enabled::<T>()` holds).
pub(super) unsafe fn self_check_with<T: SmePack>(kernel: ZaKernelFn<T>) -> bool {
    CHECK_BLOCKS.iter().all(|block| {
        // SAFETY: forwarded from this function's contract.
        unsafe { check_block(kernel, block) }
    })
}

/// First-use check of the ZA transposers: runs every transposer kernel of
/// the present types on a few fixed blocks (full panels, a partial panel,
/// depth tails) and compares with plain Rust; `true` = all equal. The
/// detection runs it (`detect::run_checks`, second of the four checks) on
/// the first detection call (`detect::hw` or `detect::hw_now`) from a ready
/// thread while the verdict is open, with FPCR = 0 and the
/// caller's FPCR and FPSR restored afterwards. It calls nothing that runs
/// or reads the detection, creates no thread and never panics.
///
/// Cost: one kernel call per block of [`CHECK_BLOCKS`], 12 for `f32` and 12
/// more for `f64`.
///
/// # Safety
///
/// The CPU has SME2 with a 512-bit streaming vector length (and
/// FEAT_SME_F64F64 when `f64_present`), and the calling thread was just
/// found in the state that `detect::thread_ready()` checks. Outside the
/// detection that is `detect::thread_ready()` returning `true` and
/// `detect::enabled::<f32>()` holding (and `detect::enabled::<f64>()` when
/// `f64_present`); the detection reads the state with
/// `detect::probe_thread_state()`.
pub(super) unsafe fn self_check(f64_present: bool) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        self_check_with::<f32>(za_transpose_f32)
            && (!f64_present || self_check_with::<f64>(za_transpose_f64))
    }
}

// ---------------------------------------------------------------------------
// An executable model of the transposer templates (test builds)
// ---------------------------------------------------------------------------

/// An interpreter of the registered transposer templates, for the tests
/// (`tests_pack_za.rs`): it executes the template text natively, on real
/// memory, and checks every access against a [`model::Window`] (the block
/// for reads, the packed prefix for writes) BEFORE it happens. It models
/// exactly the statements the transposers use, with the SVL-512 shapes
/// (tile-slice selects modulo the tile height, `psel` indices modulo the
/// element count, `whilelt` on signed operands), and fails on any other
/// statement. A read outside the block is invisible to every result test;
/// this model makes it visible on any host. QEMU and the first-use check on
/// real hardware remain the judges of the architecture.
#[cfg(test)]
pub(crate) mod model {
    use super::{
        OFF_COLS, OFF_DEPTH, OFF_DST, OFF_LD, OFF_SRC, OFF_TILE_STEP, OFF_WIDTH, ZaPackArgs,
    };

    /// `Err(format!(..))` unless the condition holds.
    macro_rules! ensure {
        ($cond:expr, $($why:tt)+) => {
            if !$cond {
                return Err(format!($($why)+));
            }
        };
    }

    /// What the interpreter may touch during one call.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Window {
        /// Address of the stored block's element `(0, 0)`.
        pub(crate) src: usize,
        /// Leading dimension in elements.
        pub(crate) ld: usize,
        /// Rows and columns of the stored block.
        pub(crate) rows: usize,
        pub(crate) cols: usize,
        /// Element size in bytes.
        pub(crate) esize: usize,
        /// Address and length (elements) of the packed prefix.
        pub(crate) dst: usize,
        pub(crate) dst_len: usize,
    }

    impl Window {
        /// Whether the element at `addr` is an element of the stored block.
        pub(crate) fn readable(&self, addr: usize) -> bool {
            let Some(offset) = addr.checked_sub(self.src) else {
                return false;
            };
            if offset % self.esize != 0 {
                return false;
            }
            let o = offset / self.esize;
            if self.ld >= self.rows && self.ld > 0 {
                return o / self.ld < self.cols && o % self.ld < self.rows;
            }
            (0..self.cols).any(|c| o.checked_sub(c * self.ld).is_some_and(|p| p < self.rows))
        }

        /// Whether the element at `addr` is inside the packed prefix.
        pub(crate) fn writable(&self, addr: usize) -> bool {
            addr.checked_sub(self.dst).is_some_and(|offset| {
                offset % self.esize == 0 && offset / self.esize < self.dst_len
            })
        }
    }

    /// A condition of a branch or a `csel` (unsigned comparisons only).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Cond {
        Lo,
        Hs,
        Ls,
        Hi,
        Eq,
        Ne,
    }

    /// The second source of an ALU operation: an immediate or a shifted register.
    #[derive(Debug, Clone, Copy)]
    enum Src {
        Imm(u64),
        Reg(usize, u32),
    }

    /// Add, subtract or bitwise and.
    #[derive(Debug, Clone, Copy)]
    enum AluOp {
        Add,
        Sub,
        And,
    }

    /// One decoded statement.
    #[derive(Debug, Clone, Copy)]
    enum Op {
        /// The frame's FPSR lines and the mode switches.
        Nop,
        Ldr {
            d: usize,
            off: u64,
        },
        Mov {
            d: usize,
            src: Src,
            w: bool,
        },
        Alu {
            op: AluOp,
            d: usize,
            n: usize,
            src: Src,
            w: bool,
        },
        Lsl {
            d: usize,
            n: usize,
            s: u32,
        },
        Neg {
            d: usize,
            n: usize,
        },
        Cmp {
            n: usize,
            src: Src,
            w: bool,
        },
        Csel {
            d: usize,
            n: usize,
            m: usize,
            cond: Cond,
        },
        Branch {
            cond: Option<Cond>,
            target: usize,
        },
        Ptrue {
            d: usize,
            esize: usize,
        },
        Whilelt {
            d: usize,
            n: usize,
            m: usize,
            esize: usize,
        },
        Psel {
            d: usize,
            n: usize,
            m: usize,
            esize: usize,
            v: usize,
            imm: u64,
        },
        ZeroZa,
        Slice(SliceOp),
    }

    /// A ZA tile-slice load or store.
    #[derive(Debug, Clone, Copy)]
    struct SliceOp {
        store: bool,
        tile: usize,
        vertical: bool,
        esize: usize,
        v: usize,
        off: u64,
        pred: usize,
        base: usize,
        index: Option<(usize, u32)>,
    }

    /// The statements of a registry template with the declared constants
    /// substituted and the escapes undone, in order.
    pub(crate) fn statements(template: &str, consts: &[(&str, usize)]) -> Vec<String> {
        let mut text = template.replace("{{", "\u{1}").replace("}}", "\u{2}");
        for (name, value) in consts {
            text = text.replace(&format!("{{{name}}}"), &value.to_string());
        }
        let text = text.replace('\u{1}', "{").replace('\u{2}', "}");
        text.lines()
            .map(|line| line.trim().to_owned())
            .filter(|line| !line.is_empty())
            .collect()
    }

    /// The constants of the transposer templates.
    pub(crate) const CONSTS: [(&str, usize); 7] = [
        ("off_src", OFF_SRC),
        ("off_ld", OFF_LD),
        ("off_dst", OFF_DST),
        ("off_depth", OFF_DEPTH),
        ("off_cols", OFF_COLS),
        ("off_width", OFF_WIDTH),
        ("off_tile_step", OFF_TILE_STEP),
    ];

    /// Splits operands at the commas outside brackets and braces.
    pub(crate) fn operands(text: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let (mut depth, mut current) = (0_i32, String::new());
        for ch in text.chars() {
            match ch {
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                _ => {}
            }
            if ch == ',' && depth == 0 {
                parts.push(current.trim().to_owned());
                current.clear();
            } else {
                current.push(ch);
            }
        }
        if !current.trim().is_empty() {
            parts.push(current.trim().to_owned());
        }
        parts
    }

    /// `x<n>` / `w<n>`: the register number and whether it is the `w` form.
    fn gpr(text: &str) -> Option<(usize, bool)> {
        let w = text.starts_with('w');
        let n: usize = text.strip_prefix(['x', 'w'])?.parse().ok()?;
        (n <= 30).then_some((n, w))
    }

    /// `#<decimal>`.
    fn imm(text: &str) -> Option<u64> {
        text.strip_prefix('#')?.parse().ok()
    }

    /// `p<n>` with an optional `.<T>` or `/z` / `/m` suffix: the number and the
    /// element size of the suffix (0 without one).
    fn preg(text: &str) -> Option<(usize, usize)> {
        let body = text.strip_prefix('p')?;
        let (num, rest) = body.split_at(
            body.find(|c: char| !c.is_ascii_digit())
                .unwrap_or(body.len()),
        );
        let n: usize = num.parse().ok()?;
        let esize = match rest {
            "" | "/z" | "/m" => 0,
            ".b" => 1,
            ".h" => 2,
            ".s" => 4,
            ".d" => 8,
            _ => return None,
        };
        (n <= 15).then_some((n, esize))
    }

    /// An element-size suffix letter.
    fn esize_of(letter: &str) -> Option<usize> {
        match letter {
            "b" => Some(1),
            "h" => Some(2),
            "s" => Some(4),
            "d" => Some(8),
            _ => None,
        }
    }

    /// A condition name.
    fn cond(text: &str) -> Option<Cond> {
        match text {
            "lo" | "cc" => Some(Cond::Lo),
            "hs" | "cs" => Some(Cond::Hs),
            "ls" => Some(Cond::Ls),
            "hi" => Some(Cond::Hi),
            "eq" => Some(Cond::Eq),
            "ne" => Some(Cond::Ne),
            _ => None,
        }
    }

    /// The second source operand(s) of an ALU statement.
    fn src(parts: &[String]) -> Option<Src> {
        match parts {
            [one] => imm(one)
                .map(Src::Imm)
                .or_else(|| gpr(one).map(|(n, _)| Src::Reg(n, 0))),
            [reg, shift] => {
                let (n, _) = gpr(reg)?;
                let s = imm(shift.strip_prefix("lsl ")?)?;
                Some(Src::Reg(n, u32::try_from(s).ok()?))
            }
            _ => None,
        }
    }

    /// `[x<n>]` or `[x<n>, x<m>, lsl #<s>]`.
    fn address(text: &str) -> Option<(usize, Option<(usize, u32)>)> {
        let inner = text.strip_prefix('[')?.strip_suffix(']')?;
        let parts = operands(inner);
        let (base, _) = gpr(parts.first()?)?;
        match parts.as_slice() {
            [_] => Some((base, None)),
            [_, index, shift] => {
                let (m, _) = gpr(index)?;
                let s = imm(shift.strip_prefix("lsl ")?)?;
                Some((base, Some((m, u32::try_from(s).ok()?))))
            }
            _ => None,
        }
    }

    /// `{za<t><h|v>.<T>[w<v>, <off>]}`.
    fn slice(text: &str) -> Option<(usize, bool, usize, usize, u64)> {
        let inner = text.strip_prefix("{za")?.strip_suffix("]}")?;
        let (head, select) = inner.split_once('[')?;
        let (tile_dir, letter) = head.split_once('.')?;
        let vertical = tile_dir.ends_with('v');
        let tile: usize = tile_dir.strip_suffix(['h', 'v'])?.parse().ok()?;
        let (v, off) = select.split_once(", ")?;
        let (v, w) = gpr(v)?;
        Some((
            tile,
            vertical,
            esize_of(letter)?,
            if w { v } else { 99 },
            off.parse().ok()?,
        ))
    }

    /// The FPSR statements of the `asm.rs` frame around every template.
    pub(crate) const FRAME: [&str; 5] = [
        "mrs x17, fpsr",
        "msr fpsr, xzr",
        "mrs x16, fpsr",
        "orr x17, x17, x16",
        "msr fpsr, x17",
    ];

    /// Decodes one statement; `labels` resolves branch targets.
    fn decode(text: &str, at: usize, labels: &[(u32, usize)]) -> Result<Op, String> {
        let (mnemonic, rest) = text.split_once(' ').unwrap_or((text, ""));
        let parts = operands(rest);
        let bad = || format!("cannot interpret {text:?}");
        let target = |label: &str| -> Result<usize, String> {
            let (num, dir) = label.split_at(label.len().saturating_sub(1));
            let num: u32 = num.parse().map_err(|_| bad())?;
            let found = match dir {
                "b" => labels.iter().rev().find(|(n, pos)| *n == num && *pos <= at),
                "f" => labels.iter().find(|(n, pos)| *n == num && *pos > at),
                _ => None,
            };
            found.map(|(_, pos)| *pos).ok_or_else(bad)
        };
        let reg = |i: usize| parts.get(i).and_then(|p| gpr(p)).ok_or_else(bad);
        let op = match mnemonic {
            // The frame of `asm.rs`: the FPSR save and restore statement for
            // statement (any other line that names FPSR or `x17` is not the
            // frame and must be interpreted), and the two mode switches (the
            // only statements without operands that start with `sm`).
            _ if FRAME.contains(&text) => Op::Nop,
            m if m.starts_with("sm") && parts.is_empty() => Op::Nop,
            "ldr" => {
                let (d, _) = reg(0)?;
                let addr = parts.get(1..).map(|p| p.join(", ")).ok_or_else(bad)?;
                let inner = addr.strip_prefix("[x0, ").and_then(|a| a.strip_suffix(']'));
                Op::Ldr {
                    d,
                    off: inner.and_then(imm).ok_or_else(bad)?,
                }
            }
            "mov" => {
                let (d, w) = reg(0)?;
                Op::Mov {
                    d,
                    src: src(&parts[1..]).ok_or_else(bad)?,
                    w,
                }
            }
            "add" | "sub" | "and" => {
                let (d, w) = reg(0)?;
                let (n, _) = reg(1)?;
                let op = match mnemonic {
                    "add" => AluOp::Add,
                    "sub" => AluOp::Sub,
                    _ => AluOp::And,
                };
                Op::Alu {
                    op,
                    d,
                    n,
                    src: src(parts.get(2..).ok_or_else(bad)?).ok_or_else(bad)?,
                    w,
                }
            }
            "lsl" => Op::Lsl {
                d: reg(0)?.0,
                n: reg(1)?.0,
                s: parts
                    .get(2)
                    .and_then(|p| imm(p))
                    .and_then(|s| u32::try_from(s).ok())
                    .ok_or_else(bad)?,
            },
            "neg" => Op::Neg {
                d: reg(0)?.0,
                n: reg(1)?.0,
            },
            "cmp" => {
                let (n, w) = reg(0)?;
                Op::Cmp {
                    n,
                    src: src(&parts[1..]).ok_or_else(bad)?,
                    w,
                }
            }
            "csel" => Op::Csel {
                d: reg(0)?.0,
                n: reg(1)?.0,
                m: reg(2)?.0,
                cond: parts.get(3).and_then(|c| cond(c)).ok_or_else(bad)?,
            },
            "b" => Op::Branch {
                cond: None,
                target: target(parts.first().ok_or_else(bad)?)?,
            },
            m if m.starts_with("b.") => Op::Branch {
                cond: Some(cond(&m[2..]).ok_or_else(bad)?),
                target: target(parts.first().ok_or_else(bad)?)?,
            },
            "ptrue" => {
                let (d, esize) = parts.first().and_then(|p| preg(p)).ok_or_else(bad)?;
                ensure!(esize > 0, "{}", bad());
                Op::Ptrue { d, esize }
            }
            "whilelt" => {
                let (d, esize) = parts.first().and_then(|p| preg(p)).ok_or_else(bad)?;
                ensure!(esize > 0, "{}", bad());
                Op::Whilelt {
                    d,
                    n: reg(1)?.0,
                    m: reg(2)?.0,
                    esize,
                }
            }
            "psel" => {
                let (d, _) = parts.first().and_then(|p| preg(p)).ok_or_else(bad)?;
                let (n, _) = parts.get(1).and_then(|p| preg(p)).ok_or_else(bad)?;
                let indexed = parts.get(2).ok_or_else(bad)?;
                let (pm, select) = indexed.split_once('[').ok_or_else(bad)?;
                let (m, esize) = preg(pm).ok_or_else(bad)?;
                let (v, off) = select
                    .strip_suffix(']')
                    .and_then(|s| s.split_once(", "))
                    .ok_or_else(bad)?;
                let (v, w) = gpr(v).ok_or_else(bad)?;
                ensure!(w && esize > 0, "{}", bad());
                Op::Psel {
                    d,
                    n,
                    m,
                    esize,
                    v,
                    imm: off.parse().map_err(|_| bad())?,
                }
            }
            "zero" if rest == "{za}" => Op::ZeroZa,
            "ld1w" | "ld1d" | "st1w" | "st1d" => {
                let store = mnemonic.starts_with("st");
                let (tile, vertical, esize, v, off) =
                    parts.first().and_then(|p| slice(p)).ok_or_else(bad)?;
                let pred_text = parts.get(1).ok_or_else(bad)?;
                ensure!(store != pred_text.ends_with("/z"), "{}", bad());
                let (pred, _) = preg(pred_text).ok_or_else(bad)?;
                let (base, index) = parts.get(2).and_then(|p| address(p)).ok_or_else(bad)?;
                let want = if mnemonic.ends_with('w') { 4 } else { 8 };
                ensure!(esize == want && v < 32, "{}", bad());
                Op::Slice(SliceOp {
                    store,
                    tile,
                    vertical,
                    esize,
                    v,
                    off,
                    pred,
                    base,
                    index,
                })
            }
            _ => return Err(bad()),
        };
        Ok(op)
    }

    /// A decoded template: the statements and their decoded forms.
    pub(crate) struct Program {
        ops: Vec<Op>,
    }

    impl Program {
        /// Decodes a registry template (labels, directives and the frame
        /// included).
        pub(crate) fn new(template: &str) -> Result<Self, String> {
            let mut labels = Vec::new();
            let mut code = Vec::new();
            for line in statements(template, &CONSTS) {
                if line.starts_with('.') {
                    continue;
                }
                if let Some(num) = line.strip_suffix(':') {
                    labels.push((
                        num.parse().map_err(|_| format!("label {line:?}"))?,
                        code.len(),
                    ));
                    continue;
                }
                code.push(line);
            }
            let ops = code
                .iter()
                .enumerate()
                .map(|(at, line)| decode(line, at, &labels))
                .collect::<Result<_, _>>()?;
            Ok(Self { ops })
        }
    }

    /// The machine state the transposers use (SVL 512).
    struct Machine {
        x: [u64; 32],
        /// Operands of the last `cmp`.
        flags: (u64, u64),
        /// One bit per byte lane.
        p: [u64; 16],
        /// ZA as tiles of the element size in use: `[tile][row][col]`.
        za: Vec<u64>,
        /// The element size of the tiles in use (0 before the first access).
        za_esize: usize,
    }

    /// Streaming vector length in bytes.
    pub(crate) const SVL: usize = 64;

    impl Machine {
        fn new() -> Self {
            Self {
                x: [0; 32],
                flags: (0, 0),
                p: [0; 16],
                za: vec![0; SVL * SVL / 4],
                za_esize: 0,
            }
        }

        fn value(&self, src: Src) -> u64 {
            match src {
                Src::Imm(value) => value,
                Src::Reg(n, s) => self.x[n] << s,
            }
        }

        fn holds(&self, cond: Cond) -> bool {
            let (a, b) = self.flags;
            match cond {
                Cond::Lo => a < b,
                Cond::Hs => a >= b,
                Cond::Ls => a <= b,
                Cond::Hi => a > b,
                Cond::Eq => a == b,
                Cond::Ne => a != b,
            }
        }

        fn write(&mut self, d: usize, value: u64, w: bool) {
            self.x[d] = if w { value & 0xffff_ffff } else { value };
        }

        /// Predicate bits of the lanes `0..SVL / esize` where `active(lane)`.
        fn lanes(esize: usize, active: impl Fn(usize) -> bool) -> u64 {
            (0..SVL / esize)
                .filter(|&lane| active(lane))
                .fold(0, |bits, lane| bits | (1 << (lane * esize)))
        }

        /// One tile-slice load or store, every lane checked against the window.
        fn slice(&mut self, op: &SliceOp, window: &Window) -> Result<(), String> {
            if self.za_esize == 0 {
                self.za_esize = op.esize;
            }
            ensure!(self.za_esize == op.esize, "ZA used with two element sizes");
            let dim = SVL / op.esize;
            ensure!(
                op.tile < op.esize,
                "tile za{} does not exist for this size",
                op.tile
            );
            let sel = usize::try_from((self.x[op.v] & 0xffff_ffff) + op.off)
                .map_err(|e| e.to_string())?
                % dim;
            let base = self.x[op.base].wrapping_add(op.index.map_or(0, |(m, s)| self.x[m] << s));
            for lane in 0..dim {
                let (row, col) = if op.vertical {
                    (lane, sel)
                } else {
                    (sel, lane)
                };
                let cell = (op.tile * dim + row) * dim + col;
                let active = self.p[op.pred] >> (lane * op.esize) & 1 == 1;
                let addr = usize::try_from(base.wrapping_add((lane * op.esize) as u64))
                    .map_err(|e| e.to_string())?;
                if !active {
                    if !op.store {
                        self.za[cell] = 0;
                    }
                    continue;
                }
                if op.store {
                    ensure!(
                        window.writable(addr),
                        "write outside the packed buffer at {addr:#x}"
                    );
                    // SAFETY: `writable` proved that `addr` is an aligned element
                    // of the case's packed prefix, which the test owns.
                    unsafe {
                        if op.esize == 4 {
                            (addr as *mut u32).write(self.za[cell] as u32);
                        } else {
                            (addr as *mut u64).write(self.za[cell]);
                        }
                    }
                } else {
                    ensure!(window.readable(addr), "read outside the block at {addr:#x}");
                    // SAFETY: `readable` proved that `addr` is an aligned element
                    // of the case's stored block, which the test owns.
                    self.za[cell] = unsafe {
                        if op.esize == 4 {
                            u64::from((addr as *const u32).read())
                        } else {
                            (addr as *const u64).read()
                        }
                    };
                }
            }
            Ok(())
        }

        /// Runs a program on an argument block (fields by offset).
        fn run(
            &mut self,
            program: &Program,
            fields: &[(usize, u64)],
            window: &Window,
        ) -> Result<(), String> {
            let mut pc = 0;
            let mut steps = 0_u64;
            while let Some(op) = program.ops.get(pc) {
                steps += 1;
                ensure!(steps < 50_000_000, "no exit after {steps} steps");
                pc += 1;
                match *op {
                    Op::Nop => {}
                    Op::Ldr { d, off } => {
                        let field = fields.iter().find(|(o, _)| *o as u64 == off);
                        self.x[d] = field.map(|(_, v)| *v).ok_or("load of an unknown field")?;
                    }
                    Op::Mov { d, src, w } => self.write(d, self.value(src), w),
                    Op::Alu { op, d, n, src, w } => {
                        let (a, b) = (self.x[n], self.value(src));
                        let value = match op {
                            AluOp::Add => a.wrapping_add(b),
                            AluOp::Sub => a.wrapping_sub(b),
                            AluOp::And => a & b,
                        };
                        self.write(d, value, w);
                    }
                    Op::Lsl { d, n, s } => self.x[d] = self.x[n] << s,
                    Op::Neg { d, n } => self.x[d] = self.x[n].wrapping_neg(),
                    Op::Cmp { n, src, w } => {
                        let mask = if w { 0xffff_ffff } else { u64::MAX };
                        self.flags = (self.x[n] & mask, self.value(src) & mask);
                    }
                    Op::Csel { d, n, m, cond } => {
                        self.x[d] = if self.holds(cond) {
                            self.x[n]
                        } else {
                            self.x[m]
                        };
                    }
                    Op::Branch { cond, target } => {
                        if cond.is_none_or(|c| self.holds(c)) {
                            pc = target;
                        }
                    }
                    Op::Ptrue { d, esize } => self.p[d] = Self::lanes(esize, |_| true),
                    Op::Whilelt { d, n, m, esize } => {
                        let (start, end) = (self.x[n] as i64, self.x[m] as i64);
                        self.p[d] =
                            Self::lanes(esize, |lane| start.saturating_add(lane as i64) < end);
                    }
                    Op::Psel {
                        d,
                        n,
                        m,
                        esize,
                        v,
                        imm,
                    } => {
                        let count = (SVL / esize) as u64;
                        let element = ((self.x[v] & 0xffff_ffff) + imm) % count;
                        let on = self.p[m] >> (element as usize * esize) & 1 == 1;
                        self.p[d] = if on { self.p[n] } else { 0 };
                    }
                    Op::ZeroZa => self.za.iter_mut().for_each(|cell| *cell = 0),
                    Op::Slice(ref slice) => self.slice(slice, window)?,
                }
            }
            Ok(())
        }
    }

    /// Runs a template (registered or mutated) on `args` against `window`.
    pub(crate) fn run_template<T>(
        template: &str,
        args: &ZaPackArgs<T>,
        window: &Window,
    ) -> Result<(), String> {
        let program = Program::new(template)?;
        let fields = [
            (OFF_SRC, args.src as u64),
            (OFF_LD, args.ld as u64),
            (OFF_DST, args.dst as u64),
            (OFF_DEPTH, args.depth as u64),
            (OFF_COLS, args.cols as u64),
            (OFF_WIDTH, args.width as u64),
            (OFF_TILE_STEP, args.tile_step as u64),
        ];
        Machine::new().run(&program, &fields, window)
    }
}
