//! An interpreter of the registered macro-kernel templates, for the
//! structural tests of the kernels (test builds only; no tests of its own:
//! they live with the kernels' tests, `tests_kernel_f32_online.rs`).
//!
//! Most of what a kernel must not do leaves no trace in a result: a load of
//! an A row `≥ m` (from the source or from the online-packing buffer) feeds
//! only C rows that are never stored; column panels that read the source
//! again instead of the buffer compute the same C; a second packing pass
//! rewrites the same values; a prefetch hint changes nothing, wherever it
//! points and whether or not `PREFETCH_C` is set. So the tests run the
//! REGISTERED TEMPLATE (`SME_KERNELS`, the exact text that is assembled) in
//! this interpreter of the statements the kernel uses ([`Program`],
//! [`interpret`]): general-purpose arithmetic, flags and branches exactly,
//! predicates lane by lane, and symbolic tokens instead of numbers in the Z
//! registers, the ZA tiles and memory. Each operand (A, B, C, the buffer) is
//! a tagged region with [`GUARD`] guard elements on both sides, and every
//! access is checked against the contract of `args.rs` when it happens and
//! at the end of the call:
//!
//! - an active lane of a load reads only an element the contract names: A
//!   rows `< m` (from the source, or from the buffer after it was written),
//!   B inside its panels (padding columns included), C only in M1 and M3
//!   and only inside the block; a store goes only to the block of C or to
//!   the first `ceil(m / 64) · 64 · kc` buffer elements, buffer stores as
//!   whole vectors;
//! - each tile (row panel × column panel) zeroes, accumulates and reads
//!   exactly its live ZA tiles (`kc` outer products each, every live slice
//!   once), stores exactly its live part of C, and the accumulators its
//!   epilogue reads are `A(i, p) · B(p, j)` summed over `p = 0..kc` in order;
//! - with online packing every A element is read once from the source and
//!   `ceil(n / 16) - 1` times from the buffer, and every buffer element is
//!   written exactly once, with `A(i, p)` or with the zero padding, where
//!   `pack::pack_a` puts it; without it every A element is read once per
//!   column panel and no buffer exists;
//! - with `PREFETCH_C`, between the previous tile's epilogue and the zeroing
//!   of a tile, exactly one hint per 64 bytes of each live column's live
//!   rows of that tile (`C(i0 + 16t, j)`), and no hint anywhere else; with
//!   the bit clear no hint at all.
//!
//! The interpreter fails closed: a statement, operand form or address it
//! does not know is an error. It knows the `.s` forms of the `f32` kernel
//! (16 lanes, four 16 × 16 tiles); a kernel with other forms (the `f64`
//! kernel's `.d` forms, eight 8 × 8 tiles) needs them added here first.
//! The frame lines of `asm.rs` are found by position, never named.

use super::args::{
    MR, OFF_A, OFF_A_DEPTH, OFF_A_PACK, OFF_A_PANEL, OFF_ALPHA, OFF_B, OFF_B_PANEL, OFF_BETA,
    OFF_C, OFF_KC, OFF_LDC, OFF_M, OFF_MODE, OFF_N, OFF_PREFETCH, PREFETCH_C,
};

/// Bytes of one streaming vector (a 512-bit streaming vector length).
const VL: u64 = 64;

/// Lanes of one `.s` vector; rows and columns of one `.s` ZA tile.
pub(super) const LANES: usize = 16;

/// Bytes of one `f32`.
const ELEM: u64 = 4;

/// The `.s` ZA tiles.
const TILES: usize = 4;

/// Columns of one packed B panel.
pub(super) const NR: usize = 16;

/// Guard elements on each side of every operand (interpreter and result
/// tests): wider than one vector, so that a store one vector out of bounds
/// lands in a guard and not in the heap.
pub(super) const GUARD: usize = 64;

/// Interpreted statements after which a run counts as one that never ends.
const STEP_LIMIT: usize = 20_000_000;

/// The token of a zero element: a lane a `/z` load leaves zero, the zero
/// vector, the zero padding of B and of the buffer.
const ZERO: u64 = 0;

/// Token kind of `A(i, p)`.
const KIND_A: u64 = 1;
/// Token kind of `B(p, j)`.
const KIND_B: u64 = 2;
/// Token kind of `C(i, j)`.
const KIND_C: u64 = 3;
/// Token kind of a stale buffer element.
const KIND_STALE: u64 = 4;
/// Token kind of a guard or gap element.
const KIND_GAP: u64 = 5;
/// Token kind of a broadcast register or immediate.
const KIND_DUP: u64 = 6;
/// Token kind of an `fadd` result.
const KIND_FADD: u64 = 7;
/// Token kind of an `fmul` result.
const KIND_FMUL: u64 = 8;
/// Token kind of an `fmla` result.
const KIND_FMLA: u64 = 9;

/// Address of the argument block.
const BASE_ARGS: u64 = 0x10_0000;
/// Address of the first element of A (after its front guard).
const BASE_A: u64 = (1 << 36) + GUARD as u64 * ELEM;
/// Address of the first element of B.
const BASE_B: u64 = (2 << 36) + GUARD as u64 * ELEM;
/// Address of `C(0, 0)`.
const BASE_C: u64 = (3 << 36) + GUARD as u64 * ELEM;
/// Address of the first element of the buffer.
const BASE_BUFFER: u64 = (4 << 36) + GUARD as u64 * ELEM;

/// The splitmix64 finalizer: a bijective mix of 64 bits.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The token of an element of kind `kind` at `(x, y)` (never [`ZERO`]).
fn token(kind: u64, x: usize, y: usize) -> u64 {
    mix(mix(kind.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ x as u64)
        ^ (y as u64).wrapping_mul(0xd1b5_4a32_d192_ed03))
        | 1
}

/// The accumulator after one more outer-product step, `acc + b · a`: a
/// function of the accumulator and both factors that depends on the order.
fn product(acc: u64, b: u64, a: u64) -> u64 {
    mix(acc ^ mix(b.wrapping_add(0x632b_e59b_d9b4_e019) ^ mix(a ^ 0x8cb9_2ba7_2f3d_8dd7)))
}

/// A token made by an epilogue operation from three others.
fn derived(kind: u64, x: u64, y: u64, z: u64) -> u64 {
    mix(mix(kind ^ x) ^ mix(y ^ mix(z)))
}

/// A general-purpose register operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Gp {
    /// `0`–`30`, or `31` for the zero register.
    n: usize,
    /// The 32-bit name `wN`: reads the low half, writes zero-extend.
    w: bool,
}

/// `x0`, the argument block.
const X0: Gp = Gp { n: 0, w: false };

/// The last operand of an arithmetic statement.
#[derive(Debug, Clone, Copy)]
pub(super) enum Src {
    /// `#imm`.
    Imm(u64),
    /// A register shifted left (`xN` or `xN, lsl #s`).
    Reg(Gp, u32),
}

/// A memory operand.
#[derive(Debug, Clone, Copy)]
pub(super) enum Address {
    /// `[xB]` or `[xB, #imm]` (a byte offset).
    Plain(Gp, i64),
    /// `[xB, #k, mul vl]`.
    Vl(Gp, i64),
    /// `[xB, xI, lsl #s]`.
    Index(Gp, Gp, u32),
}

/// A branch condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Cond {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Unsigned higher or same.
    Hs,
    /// Unsigned lower.
    Lo,
    /// Unsigned higher.
    Hi,
    /// Unsigned lower or same.
    Ls,
    /// Signed greater or equal.
    Ge,
    /// Signed less.
    Lt,
    /// Signed greater.
    Gt,
    /// Signed less or equal.
    Le,
    /// Negative.
    Mi,
    /// Positive or zero.
    Pl,
}

/// One decoded statement of a template.
#[derive(Debug, Clone, Copy)]
pub(super) enum Ins {
    /// A label: nothing to do.
    Label,
    /// The frame's exit label: the call ends.
    Exit,
    /// `ldr xD|wD, [x0, #offset]`: a field of the argument block.
    LoadArg {
        /// Destination.
        rd: Gp,
        /// Byte offset in the block.
        offset: u64,
    },
    /// `add` / `sub`.
    Arith {
        /// Destination.
        rd: Gp,
        /// First operand.
        rn: Gp,
        /// Second operand.
        src: Src,
        /// `sub`.
        subtract: bool,
    },
    /// `lsl` / `lsr` by an immediate.
    Shift {
        /// Destination.
        rd: Gp,
        /// Source.
        rn: Gp,
        /// `lsl`.
        left: bool,
        /// The shift.
        amount: u32,
    },
    /// `mov`.
    Mov {
        /// Destination.
        rd: Gp,
        /// Source.
        src: Src,
    },
    /// `cmp`.
    Cmp {
        /// First operand.
        rn: Gp,
        /// Second operand.
        src: Src,
    },
    /// `csel`.
    Csel {
        /// Destination.
        rd: Gp,
        /// Taken when the condition holds.
        rn: Gp,
        /// Taken otherwise.
        rm: Gp,
        /// The condition.
        cond: Cond,
    },
    /// `b` and `b.cond`.
    Branch {
        /// The condition, none for `b`.
        cond: Option<Cond>,
        /// Statement index of the target label.
        target: usize,
    },
    /// `cbz` / `cbnz`.
    Cbz {
        /// The tested register.
        rn: Gp,
        /// `cbz`.
        zero: bool,
        /// Statement index of the target label.
        target: usize,
    },
    /// `tbz` / `tbnz`.
    Tbz {
        /// The tested register.
        rn: Gp,
        /// The tested bit.
        bit: u32,
        /// `tbz`.
        zero: bool,
        /// Statement index of the target label.
        target: usize,
    },
    /// `ptrue pD.s`.
    Ptrue {
        /// Destination.
        pd: usize,
    },
    /// `whilelt pD.s, xN, xM`.
    Whilelt {
        /// Destination.
        pd: usize,
        /// First index.
        rn: Gp,
        /// Limit.
        rm: Gp,
    },
    /// `dup zD.s, wN`.
    DupGp {
        /// Destination.
        zd: usize,
        /// Broadcast register.
        rn: Gp,
    },
    /// `dup zD.s, #imm`.
    DupImm {
        /// Destination.
        zd: usize,
        /// Broadcast immediate.
        imm: i64,
    },
    /// `zero {…}`: the ZA tiles of the mask.
    Zero {
        /// Bit `t`: tile `t`.
        mask: u8,
    },
    /// `ld1w {zD.s}, pG/z, [address]`.
    Load {
        /// Destination.
        zd: usize,
        /// Governing predicate.
        pg: usize,
        /// Address of lane 0.
        at: Address,
    },
    /// `st1w {zS.s}, pG, [address]`.
    Store {
        /// Stored vector.
        zs: usize,
        /// Governing predicate.
        pg: usize,
        /// Address of lane 0.
        at: Address,
    },
    /// `st1w {zaTh.s[wS, off]}, pG, [address]`.
    StoreSlice {
        /// ZA tile.
        tile: usize,
        /// Slice select register.
        ws: Gp,
        /// Slice offset.
        off: u64,
        /// Governing predicate.
        pg: usize,
        /// Address of lane 0.
        at: Address,
    },
    /// `fmopa zaT.s, pN/m, pM/m, zN.s, zM.s`.
    Fmopa {
        /// ZA tile.
        tile: usize,
        /// Predicate of the tile rows (`zN`).
        pn: usize,
        /// Predicate of the tile columns (`zM`).
        pm: usize,
        /// Rows: the B row.
        zn: usize,
        /// Columns: the A vector.
        zm: usize,
    },
    /// `mova zD.s, pG/m, zaTh.s[wS, off]`.
    Mova {
        /// Destination.
        zd: usize,
        /// Governing predicate.
        pg: usize,
        /// ZA tile.
        tile: usize,
        /// Slice select register.
        ws: Gp,
        /// Slice offset.
        off: u64,
    },
    /// `fadd` / `fmul` (`zD = zD op zM`) and `fmla` (`zD += zN · zM`),
    /// merging.
    Float {
        /// [`KIND_FADD`], [`KIND_FMUL`] or [`KIND_FMLA`].
        kind: u64,
        /// Destination (and first source).
        zd: usize,
        /// Governing predicate.
        pg: usize,
        /// Second source (`zD` itself for `fadd` / `fmul`).
        zn: usize,
        /// Third source.
        zm: usize,
    },
    /// `prfm pldl2keep, [address]`.
    Prefetch {
        /// The hinted address.
        at: Address,
    },
}

/// The operands of a statement, split at the commas outside brackets and
/// braces.
fn split_operands(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0_i32, 0_usize);
    for (at, ch) in text.char_indices() {
        match ch {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        parts.push(last);
    }
    parts
}

/// The number written as decimal digits only.
fn digits(text: &str) -> Option<usize> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// `xN`, `wN`, `xzr` or `wzr`.
fn gp(text: &str) -> Result<Gp, String> {
    let bad = || format!("not a general-purpose register: {text:?}");
    let (w, rest) = if let Some(rest) = text.strip_prefix('x') {
        (false, rest)
    } else if let Some(rest) = text.strip_prefix('w') {
        (true, rest)
    } else {
        return Err(bad());
    };
    if rest == "zr" {
        return Ok(Gp { n: 31, w });
    }
    match digits(rest) {
        Some(n) if n <= 30 => Ok(Gp { n, w }),
        _ => Err(bad()),
    }
}

/// A 64-bit register other than the zero register (a base or an index).
fn x_register(text: &str) -> Result<Gp, String> {
    let register = gp(text)?;
    if register.w || register.n == 31 {
        return Err(format!("not a 64-bit address register: {text:?}"));
    }
    Ok(register)
}

/// The value of a declared constant of the kernel: the `args.rs` offsets.
fn constant(name: &str) -> Result<i64, String> {
    let offset = match name {
        "off_a" => OFF_A,
        "off_a_depth" => OFF_A_DEPTH,
        "off_a_panel" => OFF_A_PANEL,
        "off_b" => OFF_B,
        "off_b_panel" => OFF_B_PANEL,
        "off_c" => OFF_C,
        "off_ldc" => OFF_LDC,
        "off_m" => OFF_M,
        "off_n" => OFF_N,
        "off_kc" => OFF_KC,
        "off_alpha" => OFF_ALPHA,
        "off_beta" => OFF_BETA,
        "off_mode" => OFF_MODE,
        "off_a_pack" => OFF_A_PACK,
        "off_prefetch" => OFF_PREFETCH,
        _ => return Err(format!("unknown constant {name:?}")),
    };
    i64::try_from(offset).map_err(|_| format!("constant {name:?} out of range"))
}

/// `#imm`, `#-imm` or a placeholder `#{name}`.
fn immediate(text: &str) -> Result<i64, String> {
    let bad = || format!("not an immediate: {text:?}");
    let body = text.strip_prefix('#').ok_or_else(bad)?;
    let (negative, body) = match body.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, body),
    };
    let value = if let Some(name) = body.strip_prefix('{').and_then(|b| b.strip_suffix('}')) {
        constant(name)?
    } else {
        digits(body)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(bad)?
    };
    Ok(if negative { -value } else { value })
}

/// A shift amount or bit number `#s` below 64.
fn small(text: &str) -> Result<u32, String> {
    u32::try_from(immediate(text)?)
        .ok()
        .filter(|value| *value < 64)
        .ok_or_else(|| format!("not a shift or bit number: {text:?}"))
}

/// The last operands of an arithmetic statement.
fn source(parts: &[&str]) -> Result<Src, String> {
    match parts {
        [value] if value.starts_with('#') => Ok(Src::Imm(immediate(value)? as u64)),
        [register] => Ok(Src::Reg(gp(register)?, 0)),
        [register, shift] => {
            let amount = shift
                .strip_prefix("lsl ")
                .ok_or_else(|| format!("unknown shift {shift:?}"))?;
            Ok(Src::Reg(gp(register)?, small(amount)?))
        }
        _ => Err(format!("unknown operands {parts:?}")),
    }
}

/// A condition code.
fn cond(text: &str) -> Result<Cond, String> {
    Ok(match text {
        "eq" => Cond::Eq,
        "ne" => Cond::Ne,
        "hs" | "cs" => Cond::Hs,
        "lo" | "cc" => Cond::Lo,
        "hi" => Cond::Hi,
        "ls" => Cond::Ls,
        "ge" => Cond::Ge,
        "lt" => Cond::Lt,
        "gt" => Cond::Gt,
        "le" => Cond::Le,
        "mi" => Cond::Mi,
        "pl" => Cond::Pl,
        _ => return Err(format!("unknown condition {text:?}")),
    })
}

/// Whether `cond` holds for the flags `[N, Z, C, V]`.
pub(super) fn holds(cond: Cond, [n, z, c, v]: [bool; 4]) -> bool {
    match cond {
        Cond::Eq => z,
        Cond::Ne => !z,
        Cond::Hs => c,
        Cond::Lo => !c,
        Cond::Hi => c && !z,
        Cond::Ls => !c || z,
        Cond::Ge => n == v,
        Cond::Lt => n != v,
        Cond::Gt => !z && n == v,
        Cond::Le => z || n != v,
        Cond::Mi => n,
        Cond::Pl => !n,
    }
}

/// `a - b` in 64 or 32 bits and its flags `[N, Z, C, V]`.
pub(super) fn subtract(a: u64, b: u64, w: bool) -> [bool; 4] {
    let (bits, mask) = if w { (32, 0xffff_ffff) } else { (64, u64::MAX) };
    let (a, b) = (a & mask, b & mask);
    let result = a.wrapping_sub(b) & mask;
    let sign = |value: u64| (value >> (bits - 1)) & 1 == 1;
    [
        sign(result),
        result == 0,
        a >= b,
        sign(a) != sign(b) && sign(result) != sign(a),
    ]
}

/// `pK` followed by `suffix` (`"/z"`, `"/m"`, `".s"` or nothing).
fn predicate(text: &str, suffix: &str) -> Result<usize, String> {
    text.strip_suffix(suffix)
        .and_then(|name| name.strip_prefix('p'))
        .and_then(digits)
        .filter(|n| *n < 16)
        .ok_or_else(|| format!("not a predicate `p<n>{suffix}`: {text:?}"))
}

/// `zN.s`.
fn zreg(text: &str) -> Result<usize, String> {
    text.strip_suffix(".s")
        .and_then(|name| name.strip_prefix('z'))
        .and_then(digits)
        .filter(|n| *n < 32)
        .ok_or_else(|| format!("not a vector `z<n>.s`: {text:?}"))
}

/// `text` without one pair of enclosing braces.
fn unbraced(text: &str) -> Option<&str> {
    text.strip_prefix('{')
        .and_then(|inner| inner.strip_suffix('}'))
}

/// `{zN.s}`: a list of exactly one vector.
fn zlist(text: &str) -> Result<usize, String> {
    let inner = unbraced(text).ok_or_else(|| format!("not a vector list: {text:?}"))?;
    zreg(inner.trim())
}

/// `zaT.s`.
fn tile(text: &str) -> Result<usize, String> {
    text.strip_suffix(".s")
        .and_then(|name| name.strip_prefix("za"))
        .and_then(digits)
        .filter(|t| *t < TILES)
        .ok_or_else(|| format!("not a `.s` ZA tile: {text:?}"))
}

/// `zaTh.s[wS, off]`, braced or not: a horizontal slice.
fn slice(text: &str) -> Result<(usize, Gp, u64), String> {
    let text = unbraced(text).unwrap_or(text);
    let bad = || format!("not a horizontal `.s` ZA slice: {text:?}");
    let (name, index) = text.split_once('[').ok_or_else(bad)?;
    let za = name
        .strip_prefix("za")
        .and_then(|rest| rest.strip_suffix("h.s"))
        .and_then(digits)
        .filter(|t| *t < TILES)
        .ok_or_else(bad)?;
    let (select, offset) = index
        .strip_suffix(']')
        .and_then(|inside| inside.split_once(','))
        .ok_or_else(bad)?;
    let select = gp(select.trim())?;
    if !select.w || !(12..=15).contains(&select.n) {
        return Err(bad());
    }
    let offset = digits(offset.trim()).filter(|o| *o < 4).ok_or_else(bad)?;
    Ok((za, select, offset as u64))
}

/// `{za}` or `{za0.s, za1.s, …}`.
fn zero_mask(text: &str) -> Result<u8, String> {
    let inner = unbraced(text).ok_or_else(|| format!("not a tile list: {text:?}"))?;
    if inner.trim() == "za" {
        return Ok((1 << TILES) - 1);
    }
    let mut mask = 0_u8;
    for name in inner.split(',') {
        mask |= 1 << tile(name.trim())?;
    }
    Ok(mask)
}

/// A memory operand.
fn address(text: &str) -> Result<Address, String> {
    let bad = || format!("unknown memory operand {text:?}");
    let inner = text
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(bad)?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    match parts.as_slice() {
        [base] => Ok(Address::Plain(x_register(base)?, 0)),
        [base, offset] if offset.starts_with('#') => {
            Ok(Address::Plain(x_register(base)?, immediate(offset)?))
        }
        [base, k, "mul vl"] => Ok(Address::Vl(x_register(base)?, immediate(k)?)),
        [base, index, shift] => {
            let amount = shift.strip_prefix("lsl ").ok_or_else(bad)?;
            Ok(Address::Index(
                x_register(base)?,
                x_register(index)?,
                small(amount)?,
            ))
        }
        _ => Err(bad()),
    }
}

/// The memory operand of `ld1w` / `st1w`: `[xB]`, `[xB, #k, mul vl]` with
/// `-8 ≤ k ≤ 7`, or `[xB, xI, lsl #2]`.
fn vector_address(text: &str) -> Result<Address, String> {
    let at = address(text)?;
    match at {
        Address::Plain(_, 0) | Address::Index(_, _, 2) => Ok(at),
        Address::Vl(_, k) if (-8..=7).contains(&k) => Ok(at),
        _ => Err(format!("not a contiguous `.s` address: {text:?}")),
    }
}

/// The number of a label statement `N:`.
fn label(line: &str) -> Option<usize> {
    line.strip_suffix(':').and_then(digits)
}

/// Decodes one statement; `resolve` turns a branch target (`Nf` / `Nb`)
/// into a statement index.
pub(super) fn decode(
    line: &str,
    resolve: impl Fn(&str) -> Result<usize, String>,
) -> Result<Ins, String> {
    if let Some(number) = label(line) {
        return Ok(if number == 90 { Ins::Exit } else { Ins::Label });
    }
    let (mnemonic, rest) = line.split_once(' ').unwrap_or((line, ""));
    let ops = split_operands(rest);
    let unknown = || "the interpreter does not know this statement".to_string();
    if let Some(code) = mnemonic.strip_prefix("b.") {
        let [target] = ops.as_slice() else {
            return Err(unknown());
        };
        return Ok(Ins::Branch {
            cond: Some(cond(code)?),
            target: resolve(target)?,
        });
    }
    Ok(match (mnemonic, ops.as_slice()) {
        ("ldr", [rd, at]) => match address(at)? {
            Address::Plain(base, offset) if base == X0 && offset >= 0 => Ins::LoadArg {
                rd: gp(rd)?,
                offset: offset as u64,
            },
            _ => return Err("a general-purpose load from outside the argument block".into()),
        },
        ("add" | "sub", [rd, rn, rest @ ..]) => Ins::Arith {
            rd: gp(rd)?,
            rn: gp(rn)?,
            src: source(rest)?,
            subtract: mnemonic == "sub",
        },
        ("lsl" | "lsr", [rd, rn, amount]) => Ins::Shift {
            rd: gp(rd)?,
            rn: gp(rn)?,
            left: mnemonic == "lsl",
            amount: small(amount)?,
        },
        ("mov", [rd, src]) => Ins::Mov {
            rd: gp(rd)?,
            src: source(&[src])?,
        },
        ("cmp", [rn, rest @ ..]) => Ins::Cmp {
            rn: gp(rn)?,
            src: source(rest)?,
        },
        ("csel", [rd, rn, rm, code]) => Ins::Csel {
            rd: gp(rd)?,
            rn: gp(rn)?,
            rm: gp(rm)?,
            cond: cond(code)?,
        },
        ("b", [target]) => Ins::Branch {
            cond: None,
            target: resolve(target)?,
        },
        ("cbz" | "cbnz", [rn, target]) => Ins::Cbz {
            rn: gp(rn)?,
            zero: mnemonic == "cbz",
            target: resolve(target)?,
        },
        ("tbz" | "tbnz", [rn, bit, target]) => Ins::Tbz {
            rn: gp(rn)?,
            bit: small(bit)?,
            zero: mnemonic == "tbz",
            target: resolve(target)?,
        },
        ("ptrue", [pd]) => Ins::Ptrue {
            pd: predicate(pd, ".s")?,
        },
        ("whilelt", [pd, rn, rm]) => Ins::Whilelt {
            pd: predicate(pd, ".s")?,
            rn: x_register(rn)?,
            rm: x_register(rm)?,
        },
        ("dup", [zd, value]) if value.starts_with('#') => Ins::DupImm {
            zd: zreg(zd)?,
            imm: immediate(value)?,
        },
        ("dup", [zd, rn]) => Ins::DupGp {
            zd: zreg(zd)?,
            rn: gp(rn)?,
        },
        ("zero", [list]) => Ins::Zero {
            mask: zero_mask(list)?,
        },
        ("ld1w", [list, pg, at]) => Ins::Load {
            zd: zlist(list)?,
            pg: predicate(pg, "/z")?,
            at: vector_address(at)?,
        },
        ("st1w", [list, pg, at]) if list.contains("za") => {
            let (tile, ws, off) = slice(list)?;
            Ins::StoreSlice {
                tile,
                ws,
                off,
                pg: predicate(pg, "")?,
                at: vector_address(at)?,
            }
        }
        ("st1w", [list, pg, at]) => Ins::Store {
            zs: zlist(list)?,
            pg: predicate(pg, "")?,
            at: vector_address(at)?,
        },
        ("fmopa", [za, pn, pm, zn, zm]) => Ins::Fmopa {
            tile: tile(za)?,
            pn: predicate(pn, "/m")?,
            pm: predicate(pm, "/m")?,
            zn: zreg(zn)?,
            zm: zreg(zm)?,
        },
        ("mova", [zd, pg, za]) => {
            let (tile, ws, off) = slice(za)?;
            Ins::Mova {
                zd: zreg(zd)?,
                pg: predicate(pg, "/m")?,
                tile,
                ws,
                off,
            }
        }
        ("fadd" | "fmul", [zd, pg, zdn, zm]) if zdn == zd => Ins::Float {
            kind: if mnemonic == "fadd" {
                KIND_FADD
            } else {
                KIND_FMUL
            },
            zd: zreg(zd)?,
            pg: predicate(pg, "/m")?,
            zn: zreg(zdn)?,
            zm: zreg(zm)?,
        },
        ("fmla", [zd, pg, zn, zm]) => Ins::Float {
            kind: KIND_FMLA,
            zd: zreg(zd)?,
            pg: predicate(pg, "/m")?,
            zn: zreg(zn)?,
            zm: zreg(zm)?,
        },
        ("prfm", ["pldl2keep", at]) => match address(at)? {
            at @ (Address::Plain(..) | Address::Index(..)) => Ins::Prefetch { at },
            Address::Vl(..) => return Err(unknown()),
        },
        _ => return Err(unknown()),
    })
}

/// A decoded template: the prologue, then the body up to the exit label.
pub(super) struct Program {
    /// The statements.
    ins: Vec<Ins>,
    /// Their text, for messages.
    text: Vec<String>,
}

impl Program {
    /// Decodes a registered template (or a mutated copy of one). The frame
    /// lines of `asm.rs` are found by position: the prologue runs from the
    /// line after the flag save (`mrs x17, fpsr`) to the line before the
    /// mode switch, which is the line before the flag clear (`msr fpsr,
    /// xzr`); the body from the line after the flag clear to the exit
    /// label `90:`.
    pub(super) fn load(template: &str) -> Result<Self, String> {
        let text = template.replace("{{", "{").replace("}}", "}");
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let find = |wanted: &str| {
            lines
                .iter()
                .position(|line| *line == wanted)
                .ok_or_else(|| format!("no `{wanted}` in the template"))
        };
        let (save, clear, exit) = (find("mrs x17, fpsr")?, find("msr fpsr, xzr")?, find("90:")?);
        if save + 1 >= clear || clear >= exit {
            return Err("the frame lines are out of order".into());
        }
        let statements: Vec<&str> = lines[save + 1..clear - 1]
            .iter()
            .chain(&lines[clear + 1..=exit])
            .copied()
            .collect();
        let labels: Vec<(usize, usize)> = statements
            .iter()
            .enumerate()
            .filter_map(|(at, line)| label(line).map(|number| (at, number)))
            .collect();
        let resolve = |from: usize, target: &str| -> Result<usize, String> {
            let bad = || format!("unresolved branch target {target:?}");
            let (number, forward) = if let Some(number) = target.strip_suffix('f') {
                (number, true)
            } else if let Some(number) = target.strip_suffix('b') {
                (number, false)
            } else {
                return Err(bad());
            };
            let number = digits(number).ok_or_else(bad)?;
            let hit = if forward {
                labels
                    .iter()
                    .find(|(at, label)| *label == number && *at > from)
            } else {
                labels
                    .iter()
                    .rev()
                    .find(|(at, label)| *label == number && *at < from)
            };
            hit.map(|(at, _)| *at).ok_or_else(bad)
        };
        let mut ins = Vec::with_capacity(statements.len());
        for (at, line) in statements.iter().enumerate() {
            let decoded =
                decode(line, |target| resolve(at, target)).map_err(|e| format!("{e}: `{line}`"))?;
            ins.push(decoded);
        }
        Ok(Self {
            ins,
            text: statements.iter().map(|line| (*line).to_string()).collect(),
        })
    }
}

/// The register state of the interpreter.
struct Machine {
    /// `x0`–`x30` (index 31 is never used: the zero register reads 0).
    x: [u64; 32],
    /// `[N, Z, C, V]`.
    flags: [bool; 4],
    /// `p0`–`p15`, lane by lane.
    p: [[bool; LANES]; 16],
    /// `z0`–`z31`, as tokens.
    z: [[u64; LANES]; 32],
    /// The `.s` ZA tiles, `[tile][row][column]`, as tokens.
    za: [[[u64; LANES]; LANES]; TILES],
}

impl Machine {
    /// The state on entry: `x0` holds the argument block; every other
    /// general-purpose register holds a value that addresses no operand;
    /// the vector, predicate and ZA state is zero, as streaming mode starts.
    fn new() -> Self {
        let mut x = [0_u64; 32];
        for (n, register) in x.iter_mut().enumerate() {
            *register = 0xdead_0000_0000_0000 | n as u64;
        }
        x[0] = BASE_ARGS;
        Self {
            x,
            flags: [false; 4],
            p: [[false; LANES]; 16],
            z: [[ZERO; LANES]; 32],
            za: [[[ZERO; LANES]; LANES]; TILES],
        }
    }

    /// The value of a register operand.
    fn get(&self, register: Gp) -> u64 {
        match (register.n, register.w) {
            (31, _) => 0,
            (n, true) => self.x[n] & 0xffff_ffff,
            (n, false) => self.x[n],
        }
    }

    /// Writes a register operand (`wN` zero-extends; the zero register
    /// discards).
    fn set(&mut self, register: Gp, value: u64) {
        if register.n != 31 {
            self.x[register.n] = if register.w {
                value & 0xffff_ffff
            } else {
                value
            };
        }
    }

    /// The value of the last operand of an arithmetic statement.
    fn operand(&self, src: Src) -> u64 {
        match src {
            Src::Imm(value) => value,
            Src::Reg(register, shift) => self.get(register) << shift,
        }
    }

    /// The byte address of a memory operand.
    fn address(&self, at: Address) -> u64 {
        match at {
            Address::Plain(base, offset) => self.get(base).wrapping_add(offset as u64),
            Address::Vl(base, k) => self.get(base).wrapping_add((k as u64).wrapping_mul(VL)),
            Address::Index(base, index, shift) => {
                self.get(base).wrapping_add(self.get(index) << shift)
            }
        }
    }
}

/// The geometry of one block: sizes and the layouts of A, B and C.
#[derive(Debug, Clone, Copy)]
pub(super) struct Shape {
    /// Rows.
    pub(super) m: usize,
    /// Columns.
    pub(super) n: usize,
    /// Depth.
    pub(super) kc: usize,
    /// `None`: A in the plain packed layout; `Some(lda)`: column-major A
    /// read in place.
    pub(super) lda: Option<usize>,
    /// Extra elements between B panels (`b_panel = kc · 16 + b_gap`).
    pub(super) b_gap: usize,
    /// `ldc - m`.
    pub(super) ldc_pad: usize,
}

impl Shape {
    /// Packed A, plain B, `ldc = m`.
    pub(super) fn plain(m: usize, n: usize, kc: usize) -> Self {
        Self {
            m,
            n,
            kc,
            lda: None,
            b_gap: 0,
            ldc_pad: 0,
        }
    }

    /// `(a_depth, a_panel)`.
    pub(super) fn a_strides(&self) -> (usize, usize) {
        match self.lda {
            None => (MR, MR * self.kc),
            Some(lda) => (lda, MR),
        }
    }

    /// Where `A(i, p)` is, in elements from the first.
    pub(super) fn a_index(&self, i: usize, p: usize) -> usize {
        let (a_depth, a_panel) = self.a_strides();
        (i / MR) * a_panel + p * a_depth + i % MR
    }

    /// Elements of A up to the last one the kernel may read.
    pub(super) fn a_len(&self) -> usize {
        self.a_index(self.m - 1, self.kc - 1) + 1
    }

    /// `b_panel`.
    pub(super) fn b_panel(&self) -> usize {
        self.kc * NR + self.b_gap
    }

    /// Elements of B up to the end of the last panel.
    pub(super) fn b_len(&self) -> usize {
        (self.n.div_ceil(NR) - 1) * self.b_panel() + self.kc * NR
    }

    /// `ldc`.
    pub(super) fn ldc(&self) -> usize {
        self.m + self.ldc_pad
    }

    /// Elements of the packed block: `ceil(m / 64) · 64 · kc`.
    pub(super) fn pack_len(&self) -> usize {
        self.m.div_ceil(MR) * MR * self.kc
    }

    /// The `(i, p)` of buffer element `e` in the plain packed layout.
    pub(super) fn packed_at(&self, e: usize) -> (usize, usize) {
        let panel = MR * self.kc;
        ((e / panel) * MR + e % MR, (e % panel) / MR)
    }
}

/// `lda = m + pad`, but never 64, the plain `a_depth` (which would hide a
/// kernel that ignores the stride it is given).
pub(super) fn in_place_lda(m: usize, pad: usize) -> usize {
    let lda = m + pad;
    if lda == MR { lda + 1 } else { lda }
}

/// One interpreted call.
#[derive(Debug, Clone, Copy)]
pub(super) struct Case {
    /// The block.
    pub(super) shape: Shape,
    /// The epilogue mode (0 M0, 1 M1, 2 M2, 3 M3).
    pub(super) mode: u64,
    /// Online packing on.
    pub(super) pack: bool,
    /// The prefetch switches.
    pub(super) prefetch: u64,
}

/// Which operand a region of the interpreter's memory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The A source.
    A,
    /// Packed B.
    B,
    /// C.
    C,
    /// The online-packing buffer.
    Buffer,
}

/// One operand in the interpreter's memory: [`GUARD`] elements, the
/// operand's elements, [`GUARD`] elements.
struct Region {
    /// What it holds.
    kind: Kind,
    /// Address of the operand's first element.
    base: u64,
    /// Per element: the coordinates the contract gives it, `None` for a
    /// guard or a gap (A rows `≥ m` in the packed panel or between the
    /// columns of in-place A, gaps between B panels, rows `m..ldc` of C).
    cells: Vec<Option<(usize, usize)>>,
    /// The token each element holds.
    tokens: Vec<u64>,
    /// Loads of each element.
    reads: Vec<u32>,
    /// Stores to each element.
    writes: Vec<u32>,
}

impl Region {
    /// `len` elements at `base`, all gaps until [`Region::set`] names them.
    fn new(kind: Kind, base: u64, len: usize) -> Self {
        let size = GUARD + len + GUARD;
        Self {
            kind,
            base,
            cells: vec![None; size],
            tokens: (0..size)
                .map(|cell| token(KIND_GAP, cell, kind as usize))
                .collect(),
            reads: vec![0; size],
            writes: vec![0; size],
        }
    }

    /// Element `e` is `at` and holds `value`.
    fn set(&mut self, e: usize, at: (usize, usize), value: u64) {
        self.cells[GUARD + e] = Some(at);
        self.tokens[GUARD + e] = value;
    }

    /// The cell (guards included) at byte address `addr`, if it is one of
    /// this region's elements.
    fn cell(&self, addr: u64) -> Option<usize> {
        let start = self.base.checked_sub(GUARD as u64 * ELEM)?;
        let offset = addr.checked_sub(start)?;
        let cell = usize::try_from(offset / ELEM).ok()?;
        (offset % ELEM == 0 && cell < self.cells.len()).then_some(cell)
    }
}

/// What one tile (one row panel × one column panel) did, from its zeroing
/// to the next tile's.
#[derive(Debug, Default)]
struct Tile {
    /// The hints issued before its zeroing (after the previous tile's
    /// epilogue began): `(i, j)` of C.
    prefetches: Vec<(usize, usize)>,
    /// How often each ZA tile was zeroed.
    zeroed: [u32; TILES],
    /// Outer products into each ZA tile.
    products: [usize; TILES],
    /// The ZA slices the epilogue read: tile, slice, contents.
    reads: Vec<(usize, usize, [u64; LANES])>,
    /// The C elements it stored.
    stores: Vec<(usize, usize)>,
    /// The epilogue has begun (a ZA slice was read).
    epilogue: bool,
}

/// What a call did, for the tests' checks that the rules were exercised.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Report {
    /// Tiles computed.
    pub(super) tiles: usize,
    /// Prefetch hints.
    pub(super) hints: usize,
    /// Vector stores into the buffer.
    pub(super) buffer_stores: usize,
}

/// One interpreted call: the machine, the operands and the open tile.
struct Run<'p> {
    /// The template.
    program: &'p Program,
    /// The call.
    case: Case,
    /// Registers.
    machine: Machine,
    /// A, B, C and (with online packing) the buffer.
    regions: Vec<Region>,
    /// The argument block, one 8-byte field per entry.
    fields: [u64; 15],
    /// The tile whose depth loop or epilogue runs.
    tile: Option<Tile>,
    /// Prefetch hints not yet given to a tile.
    pending: Vec<(usize, usize)>,
    /// `A(i, p) · B(p, j)` summed over `p` in order, as tokens, at
    /// `i + j · m`.
    expected: Vec<u64>,
    /// What the call did.
    report: Report,
}

impl<'p> Run<'p> {
    /// The operands of `case` in memory and the argument block that names
    /// them.
    fn new(program: &'p Program, case: Case) -> Self {
        let s = case.shape;
        let (m, n, kc) = (s.m, s.n, s.kc);
        let mut a = Region::new(Kind::A, BASE_A, s.a_len());
        for i in 0..m {
            for p in 0..kc {
                a.set(s.a_index(i, p), (i, p), token(KIND_A, i, p));
            }
        }
        let mut b = Region::new(Kind::B, BASE_B, s.b_len());
        for panel in 0..n.div_ceil(NR) {
            for p in 0..kc {
                for col in 0..NR {
                    let j = panel * NR + col;
                    let value = if j < n { token(KIND_B, p, j) } else { ZERO };
                    b.set(panel * s.b_panel() + p * NR + col, (p, j), value);
                }
            }
        }
        let ldc = s.ldc();
        let mut c = Region::new(Kind::C, BASE_C, ldc * n);
        for j in 0..n {
            for i in 0..m {
                c.set(i + j * ldc, (i, j), token(KIND_C, i, j));
            }
        }
        let mut regions = vec![a, b, c];
        if case.pack {
            let len = s.pack_len();
            let mut buffer = Region::new(Kind::Buffer, BASE_BUFFER, len);
            for e in 0..len {
                buffer.set(e, s.packed_at(e), token(KIND_STALE, e, 0));
            }
            regions.push(buffer);
        }
        let (alpha, beta): (f32, f32) = match case.mode {
            0 => (1.0, 0.0),
            1 => (1.0, 1.0),
            2 => (2.0, 0.0),
            _ => (2.0, 0.5),
        };
        let (a_depth, a_panel) = s.a_strides();
        let mut fields = [0_u64; 15];
        for (offset, value) in [
            (OFF_A, BASE_A),
            (OFF_A_DEPTH, a_depth as u64),
            (OFF_A_PANEL, a_panel as u64),
            (OFF_B, BASE_B),
            (OFF_B_PANEL, s.b_panel() as u64),
            (OFF_C, BASE_C),
            (OFF_LDC, ldc as u64),
            (OFF_M, m as u64),
            (OFF_N, n as u64),
            (OFF_KC, kc as u64),
            (OFF_ALPHA, u64::from(alpha.to_bits())),
            (OFF_BETA, u64::from(beta.to_bits())),
            (OFF_MODE, case.mode),
            (OFF_A_PACK, if case.pack { BASE_BUFFER } else { 0 }),
            (OFF_PREFETCH, case.prefetch),
        ] {
            fields[offset / 8] = value;
        }
        let expected = (0..m * n)
            .map(|index| {
                let (i, j) = (index % m, index / m);
                (0..kc).fold(ZERO, |acc, p| {
                    product(acc, token(KIND_B, p, j), token(KIND_A, i, p))
                })
            })
            .collect();
        Self {
            program,
            case,
            machine: Machine::new(),
            regions,
            fields,
            tile: None,
            pending: Vec::new(),
            expected,
            report: Report::default(),
        }
    }

    /// The region and cell at `addr`.
    fn locate(&self, addr: u64, what: &str) -> Result<(usize, usize), String> {
        self.regions
            .iter()
            .enumerate()
            .find_map(|(at, region)| region.cell(addr).map(|cell| (at, cell)))
            .ok_or_else(|| format!("{what} at {addr:#x}: outside every operand and its guards"))
    }

    /// `Ok` inside the depth loop of a tile (after its zeroing, before its
    /// epilogue).
    fn in_depth_loop(&self, what: &str) -> Result<(), String> {
        match &self.tile {
            Some(tile) if !tile.epilogue => Ok(()),
            _ => Err(format!("{what} outside the depth loop of a tile")),
        }
    }

    /// `Ok` inside the epilogue of a tile.
    fn in_epilogue(&self, what: &str) -> Result<(), String> {
        match &self.tile {
            Some(tile) if tile.epilogue => Ok(()),
            _ => Err(format!("{what} outside the epilogue of a tile")),
        }
    }

    /// One lane of a vector load.
    fn read(&mut self, addr: u64) -> Result<u64, String> {
        let (r, cell) = self.locate(addr, "a load")?;
        let kind = self.regions[r].kind;
        let Some((x, y)) = self.regions[r].cells[cell] else {
            return Err(format!(
                "a load of a guard or gap element of {kind:?} at {addr:#x} (a row ≥ m, a \
                 column gap, or past the operand)"
            ));
        };
        match kind {
            Kind::A | Kind::B => self.in_depth_loop("a load of A or B")?,
            Kind::C => {
                if !matches!(self.case.mode, 1 | 3) {
                    return Err(format!(
                        "a load of C({x}, {y}) in a mode that does not read C"
                    ));
                }
                self.in_epilogue("a load of C")?;
            }
            Kind::Buffer => {
                self.in_depth_loop("a load from the buffer")?;
                if x >= self.case.shape.m {
                    return Err(format!("a load of buffer row {x} ≥ m (depth {y})"));
                }
                if self.regions[r].writes[cell] == 0 {
                    return Err(format!(
                        "a load of buffer element A({x}, {y}) before it was written"
                    ));
                }
            }
        }
        let region = &mut self.regions[r];
        region.reads[cell] += 1;
        Ok(region.tokens[cell])
    }

    /// A vector store of `value` under `pg` at `at`.
    fn store(&mut self, value: [u64; LANES], pg: usize, at: Address) -> Result<(), String> {
        let base = self.machine.address(at);
        let active = self.machine.p[pg];
        let mut buffer_lanes = 0;
        for (lane, on) in active.iter().enumerate() {
            if !on {
                continue;
            }
            let addr = base.wrapping_add(lane as u64 * ELEM);
            let (r, cell) = self.locate(addr, "a store")?;
            let kind = self.regions[r].kind;
            let Some(at) = self.regions[r].cells[cell] else {
                return Err(format!(
                    "a store into a guard or gap element of {kind:?} at {addr:#x}"
                ));
            };
            match kind {
                Kind::C => {
                    self.in_epilogue("a store of C")?;
                    if let Some(tile) = self.tile.as_mut() {
                        tile.stores.push(at);
                    }
                }
                Kind::Buffer => {
                    self.in_depth_loop("a store into the buffer")?;
                    buffer_lanes += 1;
                }
                Kind::A | Kind::B => return Err(format!("a store into {kind:?}")),
            }
            let region = &mut self.regions[r];
            region.writes[cell] += 1;
            region.tokens[cell] = value[lane];
        }
        if buffer_lanes > 0 {
            if buffer_lanes != LANES {
                return Err(format!(
                    "a buffer store of {buffer_lanes} lanes, not a whole vector"
                ));
            }
            self.report.buffer_stores += 1;
        }
        Ok(())
    }

    /// Reads horizontal slice `ws + off` of ZA tile `tile` for the epilogue.
    fn read_slice(&mut self, tile: usize, ws: Gp, off: u64) -> Result<[u64; LANES], String> {
        let index = (self.machine.get(ws).wrapping_add(off) % LANES as u64) as usize;
        let value = self.machine.za[tile][index];
        let Some(record) = self.tile.as_mut() else {
            return Err("a ZA slice read outside a tile".into());
        };
        record.epilogue = true;
        record.reads.push((tile, index, value));
        Ok(value)
    }

    /// A prefetch hint.
    fn prefetch(&mut self, at: Address) -> Result<(), String> {
        let addr = self.machine.address(at);
        let (r, cell) = self.locate(addr, "a prefetch hint")?;
        let region = &self.regions[r];
        let (Kind::C, Some(at)) = (region.kind, region.cells[cell]) else {
            return Err(format!(
                "a prefetch hint outside the block of C ({:?} at {addr:#x})",
                region.kind
            ));
        };
        if self.tile.as_ref().is_some_and(|tile| !tile.epilogue) {
            return Err("a prefetch hint inside the depth loop of a tile".into());
        }
        self.pending.push(at);
        self.report.hints += 1;
        Ok(())
    }

    /// Executes statement `pc`; the next statement, or `None` at the exit.
    fn step(&mut self, pc: usize) -> Result<Option<usize>, String> {
        let flags = self.machine.flags;
        let ins = self.program.ins[pc];
        match ins {
            Ins::Label => {}
            Ins::Exit => return Ok(None),
            Ins::LoadArg { rd, offset } => {
                if self.machine.get(X0) != BASE_ARGS {
                    return Err("x0 no longer holds the argument block".into());
                }
                let field = usize::try_from(offset / 8)
                    .ok()
                    .filter(|field| offset % 8 == 0 && *field < self.fields.len())
                    .ok_or_else(|| format!("argument block offset {offset}"))?;
                self.machine.set(rd, self.fields[field]);
            }
            Ins::Arith {
                rd,
                rn,
                src,
                subtract,
            } => {
                let (a, b) = (self.machine.get(rn), self.machine.operand(src));
                let value = if subtract {
                    a.wrapping_sub(b)
                } else {
                    a.wrapping_add(b)
                };
                self.machine.set(rd, value);
            }
            Ins::Shift {
                rd,
                rn,
                left,
                amount,
            } => {
                let value = self.machine.get(rn);
                self.machine.set(
                    rd,
                    if left {
                        value << amount
                    } else {
                        value >> amount
                    },
                );
            }
            Ins::Mov { rd, src } => {
                let value = self.machine.operand(src);
                self.machine.set(rd, value);
            }
            Ins::Cmp { rn, src } => {
                self.machine.flags =
                    subtract(self.machine.get(rn), self.machine.operand(src), rn.w);
            }
            Ins::Csel { rd, rn, rm, cond } => {
                let value = if holds(cond, flags) {
                    self.machine.get(rn)
                } else {
                    self.machine.get(rm)
                };
                self.machine.set(rd, value);
            }
            Ins::Branch { cond, target } => {
                if cond.is_none_or(|cond| holds(cond, flags)) {
                    return Ok(Some(target));
                }
            }
            Ins::Cbz { rn, zero, target } => {
                if (self.machine.get(rn) == 0) == zero {
                    return Ok(Some(target));
                }
            }
            Ins::Tbz {
                rn,
                bit,
                zero,
                target,
            } => {
                if ((self.machine.get(rn) >> bit) & 1 == 0) == zero {
                    return Ok(Some(target));
                }
            }
            Ins::Ptrue { pd } => self.machine.p[pd] = [true; LANES],
            Ins::Whilelt { pd, rn, rm } => {
                let (start, limit) = (
                    i128::from(self.machine.get(rn) as i64),
                    i128::from(self.machine.get(rm) as i64),
                );
                for (lane, on) in self.machine.p[pd].iter_mut().enumerate() {
                    *on = start + (lane as i128) < limit;
                }
            }
            Ins::DupGp { zd, rn } => {
                self.machine.z[zd] = [token(KIND_DUP, self.machine.get(rn) as usize, 1); LANES];
            }
            Ins::DupImm { zd, imm } => {
                let value = if imm == 0 {
                    ZERO
                } else {
                    token(KIND_DUP, imm as usize, 2)
                };
                self.machine.z[zd] = [value; LANES];
            }
            Ins::Zero { mask } => {
                if self.tile.as_ref().is_none_or(|tile| tile.epilogue) {
                    if let Some(done) = self.tile.take() {
                        self.close(&done)?;
                    }
                    self.tile = Some(Tile {
                        prefetches: core::mem::take(&mut self.pending),
                        ..Tile::default()
                    });
                }
                for t in 0..TILES {
                    if (mask >> t) & 1 == 1 {
                        self.machine.za[t] = [[ZERO; LANES]; LANES];
                        if let Some(record) = self.tile.as_mut() {
                            record.zeroed[t] += 1;
                        }
                    }
                }
            }
            Ins::Load { zd, pg, at } => {
                let base = self.machine.address(at);
                let active = self.machine.p[pg];
                let mut value = [ZERO; LANES];
                for (lane, on) in active.iter().enumerate() {
                    if *on {
                        value[lane] = self.read(base.wrapping_add(lane as u64 * ELEM))?;
                    }
                }
                self.machine.z[zd] = value;
            }
            Ins::Store { zs, pg, at } => self.store(self.machine.z[zs], pg, at)?,
            Ins::StoreSlice {
                tile,
                ws,
                off,
                pg,
                at,
            } => {
                let value = self.read_slice(tile, ws, off)?;
                self.store(value, pg, at)?;
            }
            Ins::Fmopa {
                tile,
                pn,
                pm,
                zn,
                zm,
            } => {
                self.in_depth_loop("an outer product")?;
                if let Some(record) = self.tile.as_mut() {
                    record.products[tile] += 1;
                }
                let (rows, cols) = (self.machine.p[pn], self.machine.p[pm]);
                let (b, a) = (self.machine.z[zn], self.machine.z[zm]);
                for row in 0..LANES {
                    for col in 0..LANES {
                        if rows[row] && cols[col] {
                            let acc = &mut self.machine.za[tile][row][col];
                            *acc = product(*acc, b[row], a[col]);
                        }
                    }
                }
            }
            Ins::Mova {
                zd,
                pg,
                tile,
                ws,
                off,
            } => {
                let value = self.read_slice(tile, ws, off)?;
                for lane in 0..LANES {
                    if self.machine.p[pg][lane] {
                        self.machine.z[zd][lane] = value[lane];
                    }
                }
            }
            Ins::Float {
                kind,
                zd,
                pg,
                zn,
                zm,
            } => {
                let (d, x, y) = (self.machine.z[zd], self.machine.z[zn], self.machine.z[zm]);
                for lane in 0..LANES {
                    if self.machine.p[pg][lane] {
                        self.machine.z[zd][lane] = derived(kind, d[lane], x[lane], y[lane]);
                    }
                }
            }
            Ins::Prefetch { at } => self.prefetch(at)?,
        }
        Ok(Some(pc + 1))
    }

    /// Checks a finished tile against the contract.
    fn close(&mut self, tile: &Tile) -> Result<(), String> {
        let Shape { m, n, kc, .. } = self.case.shape;
        let Some(&(first_i, first_j)) = tile.stores.first() else {
            return Err("a tile that stores no element of C".into());
        };
        let (i0, j0) = (first_i / MR * MR, first_j / NR * NR);
        let (rows, ncols) = ((m - i0).min(MR), (n - j0).min(NR));
        let live = rows.div_ceil(LANES);
        let name = format!("tile C[{i0}.., {j0}..] ({live} live ZA tiles, {ncols} columns)");
        let mut stores = tile.stores.clone();
        stores.sort_unstable();
        let want: Vec<(usize, usize)> = (i0..i0 + rows)
            .flat_map(|i| (j0..j0 + ncols).map(move |j| (i, j)))
            .collect();
        if stores != want {
            return Err(format!(
                "{name}: stores {} elements of C, not exactly its {} live ones",
                stores.len(),
                want.len()
            ));
        }
        for t in 0..TILES {
            let (want_zeroed, want_products) = if t < live { (1, kc) } else { (0, 0) };
            if tile.zeroed[t] != want_zeroed {
                return Err(format!(
                    "{name}: ZA tile {t} zeroed {} times, want {want_zeroed}",
                    tile.zeroed[t]
                ));
            }
            if tile.products[t] != want_products {
                return Err(format!(
                    "{name}: ZA tile {t} gets {} outer products, want {want_products}",
                    tile.products[t]
                ));
            }
        }
        let mut reads: Vec<(usize, usize)> = tile.reads.iter().map(|(t, s, _)| (*t, *s)).collect();
        reads.sort_unstable();
        let want_reads: Vec<(usize, usize)> = (0..live)
            .flat_map(|t| (0..ncols).map(move |s| (t, s)))
            .collect();
        if reads != want_reads {
            return Err(format!(
                "{name}: reads the ZA slices {reads:?}, want slices 0..{ncols} of tiles \
                 0..{live} once each"
            ));
        }
        for (t, s, value) in &tile.reads {
            for (lane, got) in value.iter().enumerate() {
                let (i, j) = (i0 + t * LANES + lane, j0 + s);
                if i < m && *got != self.expected[i + j * m] {
                    return Err(format!(
                        "{name}: the accumulator of C({i}, {j}) is not A({i}, p) · B(p, {j}) \
                         summed over p = 0..{kc} in order"
                    ));
                }
            }
        }
        let mut hints = tile.prefetches.clone();
        hints.sort_unstable();
        let mut want_hints: Vec<(usize, usize)> = if self.case.prefetch & PREFETCH_C != 0 {
            (0..live)
                .flat_map(|t| (j0..j0 + ncols).map(move |j| (i0 + t * LANES, j)))
                .collect()
        } else {
            Vec::new()
        };
        want_hints.sort_unstable();
        if hints != want_hints {
            return Err(format!(
                "{name}: prefetch hints {hints:?} before its depth loop, want {want_hints:?} \
                 (prefetch = {})",
                self.case.prefetch
            ));
        }
        self.report.tiles += 1;
        Ok(())
    }

    /// Checks the whole call once the exit label is reached.
    fn finish(&mut self) -> Result<Report, String> {
        let Some(done) = self.tile.take() else {
            return Err("the call computed no tile".into());
        };
        self.close(&done)?;
        if !self.pending.is_empty() {
            return Err(format!(
                "prefetch hints after the last tile: {:?}",
                self.pending
            ));
        }
        let Shape { m, n, .. } = self.case.shape;
        let (panels_m, panels_n) = (m.div_ceil(MR), n.div_ceil(NR));
        let reads_c = u32::from(matches!(self.case.mode, 1 | 3));
        for region in &self.regions {
            for (cell, at) in region.cells.iter().enumerate() {
                let Some((x, y)) = *at else {
                    continue;
                };
                let (reads, writes) = (region.reads[cell] as usize, region.writes[cell]);
                match region.kind {
                    Kind::A => {
                        let want = if self.case.pack { 1 } else { panels_n };
                        if reads != want {
                            return Err(format!(
                                "A({x}, {y}) read {reads} times from the source, want {want} \
                                 (online packing {})",
                                self.case.pack
                            ));
                        }
                    }
                    Kind::B => {
                        if reads != panels_m {
                            return Err(format!("B({x}, {y}) read {reads} times, want {panels_m}"));
                        }
                    }
                    Kind::C => {
                        if writes != 1 || region.reads[cell] != reads_c {
                            return Err(format!(
                                "C({x}, {y}) stored {writes} times and read {reads} times, want \
                                 once and {reads_c}"
                            ));
                        }
                    }
                    Kind::Buffer => {
                        if writes != 1 {
                            return Err(format!(
                                "the buffer element of A({x}, {y}) written {writes} times, want \
                                 once"
                            ));
                        }
                        let want = if x < m { token(KIND_A, x, y) } else { ZERO };
                        if region.tokens[cell] != want {
                            return Err(format!(
                                "the buffer element of A({x}, {y}) holds something else than \
                                 pack::pack_a's (A({x}, {y}) for a row < m, zero padding else)"
                            ));
                        }
                        let want_reads = if x < m { panels_n - 1 } else { 0 };
                        if reads != want_reads {
                            return Err(format!(
                                "the buffer element of A({x}, {y}) read {reads} times, want \
                                 {want_reads}"
                            ));
                        }
                    }
                }
            }
        }
        Ok(self.report)
    }
}

/// Interprets `program` for `case`: `Ok` with what the call did when every
/// access and the final state keep the contract, the first violation else.
pub(super) fn interpret(program: &Program, case: Case) -> Result<Report, String> {
    let mut run = Run::new(program, case);
    let mut pc = 0;
    for _ in 0..STEP_LIMIT {
        match run.step(pc) {
            Ok(Some(next)) => pc = next,
            Ok(None) => return run.finish(),
            Err(error) => return Err(format!("{error} (at `{}`)", program.text[pc])),
        }
    }
    Err(format!(
        "the call did not reach its exit within {STEP_LIMIT} statements"
    ))
}
