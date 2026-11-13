//! Tests of the `f32` macro-kernel `sme2_gemm_f32` (`kernel_f32.rs`).
//!
//! Every test that executes the kernel starts with `hardware_or_skip`: on a
//! host without SME2 (the M3) it prints a skip line, under QEMU with `-r` it
//! must run. The expected value of every C element is `args::reference`, the
//! executable definition of the contract, compared with `args::same_result`
//! (bitwise, except NaN payloads).
//!
//! Each case builds its operands in owned buffers that are exactly as large
//! as the contract lets the kernel read: A up to its last element for
//! `i < m`, `p < kc` (plain packed or read in place with `a_depth = lda`,
//! `a_panel = 64`), B up to the end of the last panel (optionally with a gap
//! between panels, filled with NaN), and C inside a buffer with `ldc ≥ m`
//! and guard columns before and after the block. Everything outside
//! `C[0..m, 0..n]` holds a signalling-NaN sentinel that must stay bitwise
//! untouched: a store to the border changes it, and so does a border
//! element read and written back through the epilogue's FADD or FMLA,
//! because that arithmetic returns the NaN quieted (other bits). A border
//! element that is only read changes nothing and is not seen here (see the
//! structural test below). In M0 and M2 the block itself holds a NaN that
//! must not leak.
//!
//! - registry and template lint (native);
//! - the row predicates of every A, C load and C store in the registered
//!   template (native, structural; see below);
//! - the harness's layouts against each other (native);
//! - shape sweeps: m around 16/32/48/64/128, n around 16/32/48, kc from 1 to
//!   65, the ten alpha/beta pairs of `tests_contract.rs` (all four modes; M3
//!   with betas that are not powers of two, so that its single rounding
//!   `fma(C, beta, t)` is told apart from a product rounded before the sum),
//!   for both A layouts, and every (m, n) in 1..=20 × 1..=20 with kc ∈ {1, 5};
//! - n around 64 (four column panels, a partial fifth) on a reduced grid,
//!   every pair, both A layouts;
//! - non-plain strides (large `lda`, gaps between B panels, large `ldc`);
//! - special values in A, B, C, alpha and beta;
//! - the FPSR frame: the flags each epilogue mode can raise, exactly;
//! - thread state: ready before, SVCR and TPIDR2_EL0 zero after, the entry
//!   counter, and repeated calls interleaved with scalar and NEON `f64` work.
//!
//! A read of A rows `≥ m` cannot be seen in the results (such a lane only
//! feeds C rows `≥ m`, which are never stored), and neither can a read of C
//! outside the block that is not written back, so
//! `test_sme2_kernel_f32_template_row_predicates` checks the assembled text
//! structurally: every A load of tile `t` is governed by the tile's row
//! predicate `p<t+1>/z`, the predicate whose C stores the border tests
//! observe; `p1`–`p4` are written only by their four `whilelt` lines (a
//! braced destination list or a `pn` alias counts as a write), and no
//! predicate appears anywhere else outside a governing position. Mutated
//! templates that break these rules are rejected by the same checker
//! (`test_sme2_kernel_f32_template_checker_rejects_mutants`).

use std::hint::black_box;

use super::args::{Epilogue, MR, MacroArgs, reference, same_result};
use super::detect;
use super::kernel_f32::{SME_KERNELS, sme2_gemm_f32};
use super::lint::{lint_kernel, lint_streaming_template, report};
use super::test_support::{
    FPSR_DZC, FPSR_IDC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_QC, FPSR_UFC, Need, fpsr_read,
    fpsr_write, hardware_or_skip, kernel_entries, sme_state,
};

/// Columns of one packed B panel.
const NR: usize = 16;

/// Guard columns of sentinels before and after the C block.
const GUARD: usize = 2;

/// The alpha/beta pairs of `tests_contract.rs`: M0, M0 (beta = -0), M1, M2,
/// M2, M3, M3 (alpha = 1), M3 (beta = 1), M3, M3 (alpha = 1). The last two
/// have a beta that is not a power of two: `C · beta` is inexact, so a kernel
/// that rounds it before adding `t` (FMUL then FADD) differs from the single
/// rounding `fma(C, beta, t)` of rule R3.
const PAIRS: [(f32, f32); 10] = [
    (1.0, 0.0),
    (1.0, -0.0),
    (1.0, 1.0),
    (-1.5, 0.0),
    (0.3, 0.0),
    (-1.5, 0.25),
    (1.0, -0.5),
    (2.0, 1.0),
    (0.3, 0.7),
    (1.0, 0.7),
];

/// Column counts of the sweeps: around one, two and three panels.
const SWEEP_N: [usize; 7] = [1, 15, 16, 17, 31, 33, 47];

/// Depths of the sweeps: odd and even, below and above 64.
const SWEEP_KC: [usize; 6] = [1, 2, 3, 7, 64, 65];

/// The NaN outside the block (and in the A and B gaps): a signalling NaN,
/// which no arithmetic returns. A store to the border shows, and so does a
/// border element read and written back through FADD (`acc + C`) or FMLA
/// (`t + C · beta`): the arithmetic returns it quieted (`0x7fc0_dead`), other
/// bits. A quiet-NaN sentinel would come back unchanged from both. A border
/// element that is only read, never written, leaves no trace in any result.
fn sentinel() -> f32 {
    f32::from_bits(0x7f80_dead)
}

/// The NaN inside the block when the mode does not read C (M0, M2): every
/// element must be overwritten with a finite value. Signalling as well, for
/// uniformity; a quiet NaN would be caught the same way, since any leak of it
/// into the block is a NaN where a number is expected.
fn unread() -> f32 {
    f32::from_bits(0x7f80_0bad)
}

/// How A is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// The plain packing: `a_depth = 64`, `a_panel = 64 * kc`.
    Packed,
    /// Column-major A read in place: `a_depth = lda > m`, `a_panel = 64`.
    InPlace,
}

/// Which operand a value is generated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operand {
    /// `A(i, p)`.
    A,
    /// `B(p, j)`.
    B,
    /// `C(i, j)` (only when the mode reads C).
    C,
}

/// The geometry of one kernel call.
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// Rows.
    m: usize,
    /// Columns.
    n: usize,
    /// Depth.
    kc: usize,
    /// A layout.
    layout: Layout,
    /// `ldc - m`.
    ldc_pad: usize,
    /// Extra elements between B panels (`b_panel = kc * 16 + b_gap`).
    b_gap: usize,
    /// For [`Layout::InPlace`]: `lda - m` (at least 1; adjusted so that
    /// `lda` is never 64, the plain `a_depth`).
    lda_pad: usize,
}

impl Shape {
    /// The plain packed layouts with `ldc = m`.
    fn packed(m: usize, n: usize, kc: usize) -> Self {
        Self {
            m,
            n,
            kc,
            layout: Layout::Packed,
            ldc_pad: 0,
            b_gap: 0,
            lda_pad: 0,
        }
    }

    /// `(a_depth, a_panel)` of the layout.
    fn a_strides(&self) -> (usize, usize) {
        match self.layout {
            Layout::Packed => (MR, MR * self.kc),
            Layout::InPlace => {
                let lda = self.m + self.lda_pad.max(1);
                // `lda = 64` would equal the plain `a_depth` and hide a
                // kernel that ignores the given stride.
                (if lda == MR { lda + 1 } else { lda }, MR)
            }
        }
    }

    /// `b_panel`.
    fn b_panel(&self) -> usize {
        self.kc * NR + self.b_gap
    }

    /// `ldc`.
    fn ldc(&self) -> usize {
        self.m + self.ldc_pad
    }
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
    fn normal(&mut self) -> f32 {
        let bits = self.bits();
        let unit = (bits >> 40) as f32 / (1_u64 << 24) as f32;
        let scale = [1.0, 1.0e-3, 37.0, 0.0][(bits & 3) as usize];
        (unit * 2.0 - 1.0) * scale
    }

    /// One of `values`.
    fn pick(&mut self, values: &[f32]) -> f32 {
        values[(self.bits() >> 33) as usize % values.len()]
    }
}

/// The operands of one kernel call in owned buffers.
struct Block {
    /// Geometry.
    shape: Shape,
    /// alpha.
    alpha: f32,
    /// beta.
    beta: f32,
    /// A, exactly up to the last element the kernel may read.
    a: Vec<f32>,
    /// Packed B, exactly up to the end of the last panel.
    b: Vec<f32>,
    /// The initial C buffer: `GUARD` columns, the block, `GUARD` columns.
    c: Vec<f32>,
}

impl Block {
    /// Builds the operands; `value` gives `A(i, p)`, `B(p, j)` and `C(i, j)`.
    fn new(
        shape: Shape,
        alpha: f32,
        beta: f32,
        value: &mut dyn FnMut(Operand, usize, usize) -> f32,
    ) -> Self {
        let Shape { m, n, kc, .. } = shape;
        assert!(m >= 1 && n >= 1 && kc >= 1, "{shape:?}");
        let (a_depth, a_panel) = shape.a_strides();
        let a_len = ((m - 1) / MR) * a_panel + (kc - 1) * a_depth + (m - 1) % MR + 1;
        let mut a = vec![sentinel(); a_len];
        for i in 0..m {
            for p in 0..kc {
                a[(i / MR) * a_panel + p * a_depth + i % MR] = value(Operand::A, i, p);
            }
        }
        let panels = n.div_ceil(NR);
        let b_panel = shape.b_panel();
        let mut b = vec![sentinel(); (panels - 1) * b_panel + kc * NR];
        for panel in 0..panels {
            for p in 0..kc {
                for col in 0..NR {
                    let j = panel * NR + col;
                    // Packed B is zero-padded to whole panels (contract).
                    b[panel * b_panel + p * NR + col] =
                        if j < n { value(Operand::B, p, j) } else { 0.0 };
                }
            }
        }
        let ldc = shape.ldc();
        let reads_c = Epilogue::select(alpha, beta).reads_c();
        let mut c = vec![sentinel(); ldc * (n + 2 * GUARD)];
        for j in 0..n {
            for i in 0..m {
                c[(GUARD + j) * ldc + i] = if reads_c {
                    value(Operand::C, i, j)
                } else {
                    unread()
                };
            }
        }
        Self {
            shape,
            alpha,
            beta,
            a,
            b,
            c,
        }
    }

    /// Normal random values for every operand.
    fn random(shape: Shape, alpha: f32, beta: f32, seed: u64) -> Self {
        let mut generator = Gen::new(seed);
        Self::new(shape, alpha, beta, &mut |_, _, _| generator.normal())
    }

    /// The argument block for this block's operands with C in `c` (a copy
    /// of [`Block::c`]).
    fn args(&self, c: &mut [f32]) -> MacroArgs<f32> {
        let s = self.shape;
        let ldc = s.ldc();
        let packed = MacroArgs::packed(
            self.a.as_ptr(),
            self.b.as_ptr(),
            c[GUARD * ldc..].as_mut_ptr(),
            ldc,
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
            ..packed
        }
    }

    /// The expected C buffer: `reference` on a copy of the initial C.
    fn expected(&self) -> Vec<f32> {
        let mut want = self.c.clone();
        let args = self.args(&mut want);
        // SAFETY: `new` sized A, B and C so that every address the layout
        // names for `i < m`, `p < kc`, `j < ceil(n / 16) * 16` lies inside
        // `self.a`, `self.b` and `want`.
        unsafe { reference(&args) };
        want
    }

    /// Runs the kernel on a copy of the initial C, checking the thread
    /// state around the call.
    #[track_caller]
    fn run(&self) -> Vec<f32> {
        let mut got = self.c.clone();
        let args = self.args(&mut got);
        call(&args);
        got
    }

    /// Checks a C buffer against the expected one: `same_result` inside
    /// the block, the untouched sentinel everywhere else.
    #[track_caller]
    fn check(&self, got: &[f32], want: &[f32], label: &str) {
        let ldc = self.shape.ldc();
        assert_eq!(got.len(), want.len());
        for (index, (got, want)) in got.iter().zip(want).enumerate() {
            let (i, column) = (index % ldc, index / ldc);
            let j = column as isize - GUARD as isize;
            let inside = i < self.shape.m && j >= 0 && (j as usize) < self.shape.n;
            let ok = if inside {
                same_result(*got, *want)
            } else {
                got.to_bits() == sentinel().to_bits()
            };
            assert!(
                ok,
                "{label}: {:?} alpha {} beta {}: C({i}, {j}) = {got} ({:#010x}), want {want} \
                 ({:#010x}){}",
                self.shape,
                self.alpha,
                self.beta,
                got.to_bits(),
                want.to_bits(),
                if inside { "" } else { " outside the block" }
            );
        }
    }

    /// Runs the kernel and checks it against `reference`.
    #[track_caller]
    fn verify(&self, label: &str) {
        let got = self.run();
        self.check(&got, &self.expected(), label);
    }
}

/// One kernel call with the state checks of the contract around it.
#[track_caller]
fn call(args: &MacroArgs<f32>) {
    assert!(detect::thread_ready(), "thread not ready before the kernel");
    let entries = kernel_entries();
    // SAFETY: the thread is ready (checked above); every caller builds the
    // block with `Block::args`, whose buffers cover every address the
    // layout names, with `m, n, kc ≥ 1`, `ldc ≥ m` and the mode chosen by
    // `Epilogue::select`.
    unsafe { sme2_gemm_f32(args) };
    // `kernel_entries()` is process-wide: kernel calls of other test threads
    // also raise it, so this comparison can be satisfied by them and does
    // not prove that this call was counted (the contract sanctions `>`).
    assert!(
        kernel_entries() > entries,
        "the kernel entry was not counted"
    );
    assert_eq!(
        sme_state(),
        Some((0, 0)),
        "SVCR / TPIDR2_EL0 after the kernel"
    );
}

/// Sweeps `ms` × [`SWEEP_N`] × [`SWEEP_KC`] × [`PAIRS`] for one layout.
fn sweep(test: &str, ms: &[usize], layout: Layout) {
    sweep_shapes(test, ms, &SWEEP_N, &SWEEP_KC, layout);
}

/// Sweeps `ms` × `ns` × `kcs` × [`PAIRS`] for one layout, rotating
/// `ldc - m` (and, in place, the B panel gap and `lda - m`).
fn sweep_shapes(test: &str, ms: &[usize], ns: &[usize], kcs: &[usize], layout: Layout) {
    let mut index = 0_usize;
    for &m in ms {
        for &n in ns {
            for &kc in kcs {
                for (alpha, beta) in PAIRS {
                    let mut shape = Shape {
                        layout,
                        ldc_pad: [0, 3, 1][index % 3],
                        ..Shape::packed(m, n, kc)
                    };
                    if layout == Layout::InPlace {
                        shape.b_gap = [0, 5, 16][index % 3];
                        shape.lda_pad = [1, 7, 70][index % 3];
                    }
                    let seed = (m * 1_000_003 + n * 1009 + kc * 17 + index) as u64;
                    Block::random(shape, alpha, beta, seed).verify(test);
                    index += 1;
                }
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f32_registered_and_linted() {
    let names: Vec<&str> = SME_KERNELS.iter().map(|entry| entry.name).collect();
    assert_eq!(names, ["sme2_gemm_f32"]);
    let entry = &SME_KERNELS[0];
    assert!(!entry.f64f64);
    assert!(entry.file.ends_with("kernel_f32.rs"), "{}", entry.file);
    if let Err(errors) = lint_kernel(entry) {
        panic!("sme2_gemm_f32:\n{}", report(&errors));
    }
    // The Wave-4 fields are never read.
    assert!(
        entry
            .consts
            .iter()
            .all(|name| !matches!(*name, "off_a_pack" | "off_prefetch")),
        "{:?}",
        entry.consts
    );
    // Strides come from the block, never from the plain packing.
    for field in ["off_a_depth", "off_a_panel", "off_b_panel", "off_ldc"] {
        assert!(entry.consts.contains(&field), "{field} is not read");
    }
}

/// Returns `Err(message)` from a template check when `condition` is false.
macro_rules! ensure {
    ($condition:expr, $($message:tt)+) => {
        if !$condition {
            return Err(format!($($message)+));
        }
    };
}

/// One predicate register named by a template line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PredicateUse {
    /// The physical register: `pN` and its predicate-as-counter name `pnN`
    /// are the same register `N`.
    pub(super) register: usize,
    /// The line writes it: the register is named in the destination (the
    /// first operand, or any register of a braced first operand such as
    /// `{p4.s, p5.s}` or `{p4.s-p5.s}`), and the line is not a store.
    pub(super) written: bool,
    /// It governs the line: `pK/z`, `pK/m`, or the bare `pK` of a store
    /// (`st1w {…}, pK, [address]`).
    pub(super) governing: bool,
}

/// The predicate registers a template line names, in order. The registry's
/// `{{` / `}}` escapes read as braces; only the operands are scanned, never
/// the mnemonic (`ptrue`, `pext`, `prfm` start with `p`).
pub(super) fn predicate_uses(line: &str) -> Vec<PredicateUse> {
    let line = line.trim().replace("{{", "{").replace("}}", "}");
    let Some((mnemonic, operands)) = line.split_once(' ') else {
        return Vec::new();
    };
    let operands = operands.trim();
    let store = mnemonic.starts_with("st");
    let destination_end = if operands.starts_with('{') {
        operands.find('}').map_or(operands.len(), |close| close + 1)
    } else {
        operands.find(',').unwrap_or(operands.len())
    };
    let identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let bytes = operands.as_bytes();
    let mut uses = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] != b'p' || (at > 0 && identifier(bytes[at - 1])) {
            at += 1;
            continue;
        }
        let digits = if bytes.get(at + 1) == Some(&b'n') {
            at + 2
        } else {
            at + 1
        };
        let mut end = digits;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        let whole = bytes.get(end).is_none_or(|byte| !identifier(*byte));
        match operands[digits..end].parse::<usize>() {
            Ok(register) if whole => {
                let rest = &operands[end..];
                uses.push(PredicateUse {
                    register,
                    written: !store && at < destination_end,
                    governing: rest.starts_with("/z")
                        || rest.starts_with("/m")
                        || (store && rest.starts_with(", [")),
                });
                at = end;
            }
            _ => at += 1,
        }
    }
    // A register range `{pA.s-pB.s}` writes every register from A to B.
    if !store && operands[..destination_end].contains('-') {
        let named: Vec<usize> = uses
            .iter()
            .filter(|named| named.written)
            .map(|named| named.register)
            .collect();
        if let (Some(&first), Some(&last)) = (named.first(), named.last()) {
            for register in first + 1..last {
                uses.push(PredicateUse {
                    register,
                    written: true,
                    governing: false,
                });
            }
        }
    }
    uses
}

/// `template` with `lines` inserted after the one line whose trimmed text is
/// `anchor`, for each `(anchor, lines)` in order (test templates are built
/// from the registered text, never written out).
pub(super) fn template_with(template: &str, edits: &[(&str, &[&str])]) -> String {
    let mut lines: Vec<String> = template.lines().map(str::to_string).collect();
    for (anchor, inserted) in edits {
        let hits: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim() == *anchor)
            .map(|(at, _)| at)
            .collect();
        let [hit] = hits[..] else {
            panic!("anchor {anchor:?} occurs {} times", hits.len());
        };
        for (offset, line) in inserted.iter().enumerate() {
            lines.insert(hit + 1 + offset, (*line).to_string());
        }
    }
    lines.join("\n")
}

/// One memory access of the kernel body, parsed from the registered
/// template: `ld1w {zN.s}, pK/z, [base]`, `[base, #t, mul vl]` or
/// `[base, index, lsl #2]` (and the same shapes for `st1w`).
#[derive(Debug)]
struct Access<'t> {
    /// The whole statement.
    line: &'t str,
    /// The register list without braces (`z4.s`, `za1h.s[w12, 0]`).
    list: &'t str,
    /// The governing predicate as written (`p2/z`, `p3`).
    pred: &'t str,
    /// The base register.
    base: &'t str,
    /// `#t, mul vl`: `Some(t)`; no offset: `Some(0)`; an index register:
    /// `None`.
    vl_offset: Option<usize>,
    /// The index register of `[base, index, lsl #2]`.
    index: Option<&'t str>,
}

impl<'t> Access<'t> {
    /// Parses `mnemonic {list}, pred, [address]`.
    fn parse(line: &'t str) -> Result<Self, String> {
        let unexpected = || format!("unexpected memory access {line:?}");
        let open = line.find('{').ok_or_else(unexpected)?;
        let close = line.find('}').ok_or_else(unexpected)?;
        let list = line[open + 1..close].trim_matches(|c| c == '{' || c == '}');
        let rest = line[close..].trim_start_matches('}');
        let rest = rest.strip_prefix(", ").ok_or_else(unexpected)?;
        let (pred, address) = rest.split_once(", ").ok_or_else(unexpected)?;
        let address = address
            .strip_prefix('[')
            .and_then(|a| a.strip_suffix(']'))
            .ok_or_else(unexpected)?;
        let parts: Vec<&str> = address.split(", ").collect();
        let (vl_offset, index) = match parts.as_slice() {
            [_] => (Some(0), None),
            [_, offset, "mul vl"] => {
                let t = offset
                    .strip_prefix('#')
                    .and_then(|t| t.parse().ok())
                    .ok_or_else(unexpected)?;
                (Some(t), None)
            }
            [_, index, "lsl #2"] => (None, Some(*index)),
            _ => return Err(unexpected()),
        };
        Ok(Self {
            line,
            list,
            pred,
            base: parts[0],
            vl_offset,
            index,
        })
    }

    /// The C tile row offset of `[x13, index, lsl #2]`: tile 0 has no
    /// index, tiles 1, 2, 3 use x14 = 16, x15 = 32, x16 = 48.
    fn c_tile(&self) -> Result<usize, String> {
        match (self.vl_offset, self.index) {
            (Some(0), None) => Ok(0),
            (None, Some("x14")) => Ok(1),
            (None, Some("x15")) => Ok(2),
            (None, Some("x16")) => Ok(3),
            _ => Err(format!(
                "C access with an unexpected address: {:?}",
                self.line
            )),
        }
    }
}

/// The line that defines the row predicate of tile `t`:
/// `p<t+1> = whilelt(i0 + 16t, m)` with `x9 = i0`, `x20 = m` (and, for
/// `t ≥ 1`, `x16 = i0 + 16t` set by the line before it).
fn row_predicate_line(t: usize) -> String {
    if t == 0 {
        "whilelt p1.s, x9, x20".to_string()
    } else {
        format!("whilelt p{}.s, x16, x20", t + 1)
    }
}

/// The structural rules of the `f32` template that no result test can see
/// (see `test_sme2_kernel_f32_template_row_predicates`), checked on a
/// template text: the registered one, or a mutated copy of it.
fn check_row_predicates(template: &str) -> Result<(), String> {
    let template = template.replace("{{", "{").replace("}}", "}");
    let lines: Vec<&str> = template.lines().map(str::trim).collect();
    let start = lines
        .iter()
        .position(|line| *line == "msr fpsr, xzr")
        .ok_or("no body start in the template")?;
    let stop = lines
        .iter()
        .position(|line| *line == "90:")
        .ok_or("no exit label in the template")?;
    let prologue = &lines[..start];
    let body = &lines[start + 1..stop];

    // `x20` is `m` for the whole body.
    ensure!(
        prologue.contains(&"ldr x20, [x0, #{off_m}]"),
        "the prologue does not load m into x20"
    );
    ensure!(
        body.first() == Some(&"ptrue p0.s"),
        "the body does not start with `ptrue p0.s`"
    );
    for line in body {
        let operands = line.split_once(' ').map_or("", |(_, operands)| operands);
        let first = operands.split(", ").next().unwrap_or("");
        ensure!(
            !matches!(first, "x20" | "w20"),
            "the body overwrites m: {line:?}"
        );
    }

    // The predicates, on every line of the template: `p0` is all-true and
    // `p1`–`p4` are `whilelt(i0 + 16t, m)`. Each is written by exactly its
    // own line, whatever the form of a write (a braced destination list,
    // a register range or a `pn` alias name the same registers), and no
    // predicate appears outside a governing position (`pK/z`, `pK/m`, a
    // store's `pK`) anywhere else: a copy, a `pext` source or a `sel`
    // operand is rejected as well.
    let mut writes = [0_usize; 5];
    for (at, line) in lines.iter().enumerate() {
        for PredicateUse {
            register,
            written,
            governing,
        } in predicate_uses(line)
        {
            ensure!(
                register < writes.len(),
                "p{register} is not a predicate of this kernel: {line:?}"
            );
            if !written {
                ensure!(
                    governing,
                    "p{register} outside a governing position (pK/z, pK/m, a store's pK) \
                     in a line that does not define it: {line:?}"
                );
                continue;
            }
            let wanted = if register == 0 {
                "ptrue p0.s".to_string()
            } else {
                row_predicate_line(register - 1)
            };
            ensure!(
                at > start && at < stop && *line == wanted,
                "p{register} written by {line:?}; only `{wanted}` in the body may write it"
            );
            if register >= 2 {
                let add = format!("add x16, x9, #{}", 16 * (register - 1));
                ensure!(
                    lines[at - 1] == add,
                    "`{wanted}` is not preceded by `{add}`"
                );
            }
            writes[register] += 1;
        }
    }
    ensure!(
        writes == [1; 5],
        "p0..p4 must each be written once, by their own line: written {writes:?} times"
    );

    // Every load and store of the body, classified by its base register.
    let (mut a_loads, mut b_loads, mut c_loads, mut c_stores) = (0, 0, 0, 0);
    for line in body {
        let mnemonic = line.split(' ').next().unwrap_or("");
        if mnemonic.starts_with("ld") {
            let access = Access::parse(line)?;
            match access.base {
                // A at depth p (x14) and p + 1 (x16).
                "x14" | "x16" => {
                    let t = access
                        .vl_offset
                        .ok_or_else(|| format!("indexed A load {line:?}"))?;
                    ensure!(t < 4, "A load beyond tile 3: {line:?}");
                    ensure!(access.pred != "p0/z", "all-true A load {line:?}");
                    ensure!(
                        access.pred == format!("p{}/z", t + 1),
                        "A load of tile {t} not governed by its row predicate: {line:?}"
                    );
                    let z: usize = access
                        .list
                        .strip_prefix('z')
                        .and_then(|z| z.strip_suffix(".s"))
                        .and_then(|z| z.parse().ok())
                        .ok_or_else(|| format!("A load into {:?}", access.list))?;
                    ensure!(z < 8 && z % 4 == t, "A vector z{z} for tile {t}: {line:?}");
                    a_loads += 1;
                }
                // Packed B is zero-padded to whole panels: all-true.
                "x15" => {
                    ensure!(access.pred == "p0/z", "B load not all-true: {line:?}");
                    ensure!(
                        matches!(access.list, "z8.s" | "z9.s"),
                        "B load into {line:?}"
                    );
                    ensure!(
                        matches!(access.vl_offset, Some(0 | 1)),
                        "B load offset {line:?}"
                    );
                    b_loads += 1;
                }
                // C (M1 and M3).
                "x13" => {
                    ensure!(access.list == "z17.s", "C load into {line:?}");
                    let t = access.c_tile()?;
                    ensure!(
                        access.pred == format!("p{}/z", t + 1),
                        "C load of tile {t} not governed by its row predicate: {line:?}"
                    );
                    c_loads += 1;
                }
                _ => return Err(format!("load from an unexpected base: {line:?}")),
            }
        } else if mnemonic.starts_with("st") {
            let access = Access::parse(line)?;
            ensure!(access.base == "x13", "store outside C: {line:?}");
            let t = access.c_tile()?;
            ensure!(
                access.pred == format!("p{}", t + 1),
                "C store of tile {t} not governed by its row predicate: {line:?}"
            );
            let za = format!("za{t}h.s[w12, 0]");
            ensure!(
                access.list == "z16.s" || access.list == za,
                "tile {t} stores {:?}: {line:?}",
                access.list
            );
            c_stores += 1;
        }
    }
    // Variants of 1, 2, 3 and 4 tiles: (2 + 1) loads per tile; M1 and M3
    // load C of 4 tiles; each of the 4 modes stores 4 tiles.
    let counts = (a_loads, b_loads, c_loads, c_stores);
    ensure!(
        counts == (3 * (1 + 2 + 3 + 4), 3 * 4, 2 * 4, 4 * 4),
        "(A loads, B loads, C loads, C stores) = {counts:?}"
    );
    Ok(())
}

#[test]
fn test_sme2_kernel_f32_template_row_predicates() {
    // A STRUCTURAL check of the registered template, on purpose: a load of
    // an A row `i ≥ m` cannot be observed in any result. Such a lane only
    // feeds ZA rows that belong to C rows `≥ m`, which are never stored,
    // and the test buffers end where the contract says, but a heap read
    // past them does not fault. Only the predicated `/z` loads keep the
    // kernel off those rows (Wave 4 reads A in place, where they may not
    // exist). So this test asserts the property on the text that is
    // assembled: every A load of tile t is governed by `p<t+1>/z`, the
    // predicate `whilelt(i0 + 16t, m)`, never by the all-true `p0`, and no
    // other instruction writes (or even names, outside a governing
    // position) a predicate. The same `p<t+1>` governs every C load and
    // store of tile t, and those the sentinel-border tests do observe, so
    // a wrong predicate cannot hide behind a right-looking name.
    if let Err(error) = check_row_predicates(SME_KERNELS[0].template) {
        panic!("sme2_gemm_f32: {error}");
    }
}

/// A mutated template the structural checker must reject.
struct Mutant {
    /// What the mutation does.
    label: &'static str,
    /// Lines inserted after an anchor line of the registered template.
    edits: &'static [(&'static str, &'static [&'static str])],
    /// The statement the checker must name (registry escapes as written).
    culprit: &'static str,
    /// Whether the template lint accepts the mutant (the lint is not the
    /// guard here; this records that the hole is real).
    lint_accepts: bool,
}

#[test]
fn test_sme2_kernel_f32_template_checker_rejects_mutants() {
    // Wrong templates that every result test and the lint accept: each
    // overwrites a row predicate (so tile-t A loads may read rows ≥ m) in a
    // form that a first-operand-only parse does not see.
    const MUTANTS: [Mutant; 6] = [
        Mutant {
            label: "review m12: p4 all-true through the pair form, restored for the stores",
            edits: &[
                ("zero {{za}}", &["whilelt {{p4.s, p5.s}}, x9, x20"]),
                (
                    "50:",
                    &["add x16, x9, #48", "whilelt {{p4.s, p5.s}}, x16, x20"],
                ),
            ],
            culprit: "whilelt {p4.s, p5.s}, x9, x20",
            lint_accepts: true,
        },
        Mutant {
            label: "the pair form as a register range",
            edits: &[("zero {{za}}", &["whilelt {{p4.s-p5.s}}, x9, x20"])],
            culprit: "whilelt {p4.s-p5.s}, x9, x20",
            lint_accepts: true,
        },
        Mutant {
            label: "pext into a pair of row predicates",
            edits: &[("zero {{za0.s, za1.s}}", &["pext {{p2.s, p3.s}}, pn8[0]"])],
            culprit: "pext {p2.s, p3.s}, pn8[0]",
            lint_accepts: true,
        },
        Mutant {
            label: "a row predicate copied from the all-true p0 (sel)",
            edits: &[("zero {{za}}", &["sel p4.b, p0, p0.b, p4.b"])],
            culprit: "sel p4.b, p0, p0.b, p4.b",
            lint_accepts: true,
        },
        Mutant {
            label: "a row predicate written through its pn alias",
            edits: &[("zero {{za0.s, za1.s, za2.s}}", &["mov pn3.b, pn8.b"])],
            culprit: "mov pn3.b, pn8.b",
            lint_accepts: true,
        },
        Mutant {
            label: "a row predicate read outside a governing position",
            edits: &[("zero {{za}}", &["cntp x16, p0, p4.s"])],
            culprit: "cntp x16, p0, p4.s",
            lint_accepts: true,
        },
    ];
    let registered = SME_KERNELS[0].template;
    for mutant in &MUTANTS {
        let template = template_with(registered, mutant.edits);
        assert_ne!(template, registered, "{}", mutant.label);
        assert_eq!(
            lint_streaming_template(&template).is_ok(),
            mutant.lint_accepts,
            "{}: the lint's verdict changed",
            mutant.label
        );
        match check_row_predicates(&template) {
            Ok(()) => panic!("{}: the checker accepts the mutant", mutant.label),
            Err(error) => assert!(
                error.contains(&format!("{:?}", mutant.culprit)),
                "{}: rejected for another reason: {error}",
                mutant.label
            ),
        }
    }
}

#[test]
fn test_sme2_kernel_f32_predicate_uses() {
    // The scanner the checkers of both kernels share.
    let uses = |line| {
        predicate_uses(line)
            .into_iter()
            .map(|named| (named.register, named.written, named.governing))
            .collect::<Vec<_>>()
    };
    assert_eq!(uses("whilelt p1.s, x9, x20"), [(1, true, false)]);
    assert_eq!(
        uses("whilelt {{p4.s, p5.s}}, x9, x20"),
        [(4, true, false), (5, true, false)]
    );
    assert_eq!(
        uses("pext {{p2.d-p5.d}}, pn10[0]"),
        [
            (2, true, false),
            (5, true, false),
            (10, false, false),
            (3, true, false),
            (4, true, false)
        ]
    );
    assert_eq!(uses("whilelt pn9.d, x10, x20, vlx4"), [(9, true, false)]);
    assert_eq!(uses("ld1d {{z0.d-z3.d}}, pn9/z, [x14]"), [(9, false, true)]);
    assert_eq!(
        uses("fmopa za0.s, p0/m, p0/m, z8.s, z0.s"),
        [(0, false, true), (0, false, true)]
    );
    assert_eq!(
        uses("st1w {{za1h.s[w12, 0]}}, p2, [x13, x14, lsl #2]"),
        [(2, false, true)]
    );
    assert_eq!(uses("sel p4.b, p0, p0.b, p4.b").len(), 4);
    assert!(uses("prfm pldl1keep, [x14]").is_empty());
    assert!(uses("mrs x17, fpsr").is_empty());
    assert!(uses("ptrue p0.s") == [(0, true, false)]);
    assert!(uses("42:").is_empty());
}

#[test]
fn test_sme2_kernel_f32_harness_layouts_agree() {
    // The harness itself (runs on any host): the same values in the plain
    // packing and in place, with gaps between B panels and a padded C, give
    // the same reference result, and C outside the block keeps the sentinel.
    for (m, n, kc) in [(1, 1, 1), (17, 33, 3), (64, 16, 2), (129, 47, 5)] {
        for (alpha, beta) in PAIRS {
            let value = |operand: Operand, x: usize, y: usize| -> f32 {
                let base = match operand {
                    Operand::A => 1.0,
                    Operand::B => 2.0,
                    Operand::C => 3.0,
                };
                base + ((x * 7 + y * 3) % 11) as f32 * 0.25
            };
            let plain = Block::new(Shape::packed(m, n, kc), alpha, beta, &mut { value });
            let shape = Shape {
                layout: Layout::InPlace,
                ldc_pad: 3,
                b_gap: 5,
                lda_pad: 2,
                ..Shape::packed(m, n, kc)
            };
            let inplace = Block::new(shape, alpha, beta, &mut { value });
            let (want_plain, want_inplace) = (plain.expected(), inplace.expected());
            let (ldc_plain, ldc_inplace) = (m, m + 3);
            for j in 0..n {
                for i in 0..m {
                    let x = want_plain[(GUARD + j) * ldc_plain + i];
                    let y = want_inplace[(GUARD + j) * ldc_inplace + i];
                    assert_eq!(x.to_bits(), y.to_bits(), "{m}x{n}x{kc} C({i}, {j})");
                    assert!(x.is_finite());
                }
            }
            inplace.check(&want_inplace, &want_inplace, "harness");
            assert_eq!(
                inplace.a.len(),
                (m - 1) + (kc - 1) * shape.a_strides().0 + 1,
                "A is exactly as long as the kernel may read"
            );
        }
    }
}

#[test]
fn test_sme2_kernel_f32_sweep_packed_small_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_packed_small_m", Need::F32) {
        return;
    }
    sweep(
        "packed small m",
        &[1, 15, 16, 17, 31, 32, 33],
        Layout::Packed,
    );
}

#[test]
fn test_sme2_kernel_f32_sweep_packed_mid_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_packed_mid_m", Need::F32) {
        return;
    }
    sweep(
        "packed mid m",
        &[47, 48, 49, 63, 64, 65, 70],
        Layout::Packed,
    );
}

#[test]
fn test_sme2_kernel_f32_sweep_packed_large_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_packed_large_m", Need::F32) {
        return;
    }
    sweep("packed large m", &[127, 128, 129], Layout::Packed);
}

#[test]
fn test_sme2_kernel_f32_sweep_inplace_small_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_inplace_small_m", Need::F32) {
        return;
    }
    sweep(
        "in place small m",
        &[1, 15, 16, 17, 31, 32, 33],
        Layout::InPlace,
    );
}

#[test]
fn test_sme2_kernel_f32_sweep_inplace_mid_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_inplace_mid_m", Need::F32) {
        return;
    }
    sweep(
        "in place mid m",
        &[47, 48, 49, 63, 64, 65, 70],
        Layout::InPlace,
    );
}

#[test]
fn test_sme2_kernel_f32_sweep_inplace_large_m() {
    if !hardware_or_skip("test_sme2_kernel_f32_sweep_inplace_large_m", Need::F32) {
        return;
    }
    sweep("in place large m", &[127, 128, 129], Layout::InPlace);
}

#[test]
fn test_sme2_kernel_f32_small_grid() {
    if !hardware_or_skip("test_sme2_kernel_f32_small_grid", Need::F32) {
        return;
    }
    let mut index = 0_usize;
    for m in 1..=20 {
        for n in 1..=20 {
            for kc in [1, 5] {
                let (alpha, beta) = PAIRS[index % PAIRS.len()];
                let layout = if index.is_multiple_of(2) {
                    Layout::Packed
                } else {
                    Layout::InPlace
                };
                let shape = Shape {
                    layout,
                    ldc_pad: [0, 2, 1][index % 3],
                    b_gap: [0, 3][index % 2],
                    lda_pad: 1 + index % 4,
                    ..Shape::packed(m, n, kc)
                };
                Block::random(shape, alpha, beta, index as u64).verify("small grid");
                index += 1;
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f32_columns_around_64_every_mode() {
    if !hardware_or_skip(
        "test_sme2_kernel_f32_columns_around_64_every_mode",
        Need::F32,
    ) {
        return;
    }
    // n = 63, 64, 65: three full column panels and a partial fourth (15
    // columns), four full panels, four full panels and a one-column fifth,
    // after three or four advances of the B panel and C column pointers;
    // m with one tile, with three and four live tiles (49, 64) and a second
    // row panel (65); kc with and without the remainder step; every pair.
    for layout in [Layout::Packed, Layout::InPlace] {
        sweep_shapes(
            "n around 64",
            &[1, 17, 49, 64, 65],
            &[63, 64, 65],
            &[1, 2, 65],
            layout,
        );
    }
}

#[test]
fn test_sme2_kernel_f32_non_plain_strides() {
    if !hardware_or_skip("test_sme2_kernel_f32_non_plain_strides", Need::F32) {
        return;
    }
    // A far apart in depth, wide gaps between B panels, a tall C: every
    // stride differs from the plain packing.
    for (m, n, kc) in [(1, 17, 2), (40, 33, 9), (64, 16, 4), (129, 47, 3)] {
        for (alpha, beta) in PAIRS {
            let shape = Shape {
                layout: Layout::InPlace,
                ldc_pad: 37,
                b_gap: 131,
                lda_pad: 997,
                ..Shape::packed(m, n, kc)
            };
            Block::random(shape, alpha, beta, (m + n + kc) as u64).verify("strides");
            // The plain packing of A with the same B and C strides.
            let shape = Shape {
                layout: Layout::Packed,
                ..shape
            };
            Block::random(shape, alpha, beta, (m * n * kc) as u64).verify("strides");
        }
    }
}

#[test]
fn test_sme2_kernel_f32_nan_c_is_not_read_without_beta() {
    if !hardware_or_skip(
        "test_sme2_kernel_f32_nan_c_is_not_read_without_beta",
        Need::F32,
    ) {
        return;
    }
    // M0 and M2: the block holds NaN (and the border a NaN sentinel); with
    // finite A and B the result is finite everywhere.
    for (m, n, kc) in [(1, 1, 1), (16, 16, 2), (33, 17, 3), (70, 47, 5)] {
        for (alpha, beta) in [(1.0, 0.0), (1.0, -0.0), (-1.5, 0.0), (0.3, -0.0)] {
            for layout in [Layout::Packed, Layout::InPlace] {
                let shape = Shape {
                    layout,
                    ldc_pad: 2,
                    lda_pad: 3,
                    ..Shape::packed(m, n, kc)
                };
                let block = Block::random(shape, alpha, beta, (m + 31 * n) as u64);
                let got = block.run();
                block.check(&got, &block.expected(), "NaN C");
                for j in 0..n {
                    for i in 0..m {
                        let value = got[(GUARD + j) * shape.ldc() + i];
                        assert!(value.is_finite(), "{m}x{n}x{kc} C({i}, {j}) = {value}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_kernel_f32_special_values() {
    if !hardware_or_skip("test_sme2_kernel_f32_special_values", Need::F32) {
        return;
    }
    // NaN, infinities, signed zeros, subnormals (1e-40, and 1e-20 products
    // that round into the subnormal range), values whose products overflow.
    const SPECIALS: [f32; 13] = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0e-40,
        -1.0e-40,
        1.0e-20,
        f32::MIN_POSITIVE,
        1.0e30,
        -3.0,
        1.5,
        7.0,
    ];
    let pairs: [(f32, f32); 14] = [
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, -0.0),
        (-0.0, 0.0),
        (f32::NAN, 0.0),
        (1.0, f32::NAN),
        (f32::NAN, f32::NAN),
        (f32::INFINITY, 0.5),
        (0.5, f32::NEG_INFINITY),
        (1.0e-40, 1.0),
        (1.0e30, 0.0),
        (3.0, 1.0e30),
        (-1.0, 1.0e-40),
        (0.0, 2.0),
    ];
    for (m, n, kc) in [
        (3, 2, 2),
        (17, 17, 3),
        (49, 33, 5),
        (70, 20, 7),
        (129, 47, 4),
    ] {
        for (index, (alpha, beta)) in pairs.into_iter().enumerate() {
            for layout in [Layout::Packed, Layout::InPlace] {
                let shape = Shape {
                    layout,
                    ldc_pad: 1,
                    b_gap: 2,
                    lda_pad: 5,
                    ..Shape::packed(m, n, kc)
                };
                let mut generator = Gen::new((m * 100 + n * 10 + kc + index) as u64);
                let block = Block::new(shape, alpha, beta, &mut |_, _, _| {
                    // Mostly specials, some ordinary values so that sums of
                    // finite products occur as well.
                    if generator.bits().is_multiple_of(4) {
                        generator.normal()
                    } else {
                        generator.pick(&SPECIALS)
                    }
                });
                block.verify("special values");
            }
        }
    }
    // Overflowing and underflowing depth sums without NaN or infinity in the
    // inputs: 1e20 · 1e20 overflows, 1e-20 · 1e-20 is subnormal, 1e-30 ·
    // 1e-30 underflows to zero.
    for (a, b) in [(1.0e20, 1.0e20), (1.0e-20, 1.0e-20), (1.0e-30, -1.0e-30)] {
        for (alpha, beta) in PAIRS {
            let shape = Shape::packed(65, 17, 3);
            let block = Block::new(shape, alpha, beta, &mut |operand, x, y| match operand {
                Operand::A => a * (1.0 + (x % 3) as f32),
                Operand::B => b * (1.0 + (y % 5) as f32),
                Operand::C => (x + y) as f32 * 1.0e-38,
            });
            block.verify("overflow / underflow");
        }
    }
}

/// FPSR after one kernel call on `block` that starts from `before`; the
/// result is checked against `reference`.
fn fpsr_across(block: &Block, before: u64, label: &str) -> u64 {
    let mut got = block.c.clone();
    let args = black_box(block.args(&mut got));
    assert!(detect::thread_ready());
    let saved = fpsr_read();
    fpsr_write(before);
    // SAFETY: as in `call`: the thread is ready and the block covers every
    // address the layout names.
    unsafe { sme2_gemm_f32(&args) };
    let after = fpsr_read();
    fpsr_write(saved);
    block.check(&got, &block.expected(), label);
    after
}

/// The cumulative flags a caller may have set before the call.
const EARLIER_FLAGS: [u64; 6] = [
    0,
    FPSR_DZC,
    FPSR_DZC | FPSR_OFC,
    FPSR_IOC | FPSR_UFC | FPSR_IDC,
    FPSR_IXC,
    FPSR_QC,
];

/// Small integers: every product, sum and multiple by 0.5 or 2 is exact.
fn small_int(operand: Operand, x: usize, y: usize) -> f32 {
    let base = match operand {
        Operand::A => 1 + (x * 3 + y) % 5,
        Operand::B => 1 + (x + 2 * y) % 4,
        Operand::C => 1 + (x + y) % 7,
    };
    base as f32
}

/// One FPSR expectation: the block's values and the flags the kernel adds.
struct FlagCase {
    /// What the case shows.
    label: &'static str,
    /// alpha.
    alpha: f32,
    /// beta.
    beta: f32,
    /// The operand values.
    value: fn(Operand, usize, usize) -> f32,
    /// The flags the call must add (exactly).
    added: u64,
}

#[test]
fn test_sme2_kernel_f32_fpsr_flags_per_mode() {
    if !hardware_or_skip("test_sme2_kernel_f32_fpsr_flags_per_mode", Need::F32) {
        return;
    }
    let third = black_box(1.0_f32) / black_box(3.0_f32);
    assert_eq!(third.to_bits(), (1.0_f32 / 3.0).to_bits());
    // The depth loop (FMOPA) and the stores raise nothing; the epilogue's
    // SVE arithmetic raises what it raises on live lanes only (it is
    // predicated by the row predicate).
    let cases = [
        FlagCase {
            label: "M0 exact",
            alpha: 1.0,
            beta: 0.0,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M0, inexact depth sums: FMOPA raises nothing",
            alpha: 1.0,
            beta: 0.0,
            value: |_, x, y| (1.0 + (x + y) as f32) / 3.0,
            added: 0,
        },
        FlagCase {
            label: "M0, overflowing depth sums: FMOPA raises nothing",
            alpha: 1.0,
            beta: 0.0,
            value: |_, _, _| 1.0e30,
            added: 0,
        },
        FlagCase {
            label: "M1 exact",
            alpha: 1.0,
            beta: 1.0,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M1, inexact acc + C",
            alpha: 1.0,
            beta: 1.0,
            value: |operand, x, y| match operand {
                Operand::C => 1.0 / 1_073_741_824.0,
                _ => small_int(operand, x, y),
            },
            added: FPSR_IXC,
        },
        FlagCase {
            label: "M1, inf + -inf",
            alpha: 1.0,
            beta: 1.0,
            value: |operand, _, _| match operand {
                Operand::A => f32::INFINITY,
                Operand::B => 2.0,
                Operand::C => f32::NEG_INFINITY,
            },
            added: FPSR_IOC,
        },
        FlagCase {
            label: "M2 exact",
            alpha: 0.5,
            beta: 0.0,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M2, inexact alpha",
            alpha: 1.0 / 3.0,
            beta: 0.0,
            value: small_int,
            added: FPSR_IXC,
        },
        FlagCase {
            label: "M2, overflow",
            alpha: 1.0e38,
            beta: 0.0,
            value: |operand, x, y| small_int(operand, x, y) * 4.0,
            added: FPSR_OFC | FPSR_IXC,
        },
        FlagCase {
            label: "M2, inexact underflow",
            alpha: 1.0e-39,
            beta: 0.0,
            // acc = 1/3 (rounded): acc · alpha is subnormal and inexact.
            value: |operand, _, p| match operand {
                Operand::A if p == 0 => 1.0 / 3.0,
                Operand::A => 0.0,
                _ => 1.0,
            },
            added: FPSR_UFC | FPSR_IXC,
        },
        FlagCase {
            label: "M2, alpha = inf: no 0 · inf on padded rows",
            alpha: f32::INFINITY,
            beta: 0.0,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M3 exact",
            alpha: 0.5,
            beta: 2.0,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M3 with alpha = 1: the FMUL by one is exact",
            alpha: 1.0,
            beta: 0.25,
            value: small_int,
            added: 0,
        },
        FlagCase {
            label: "M3, inexact",
            alpha: 1.0 / 3.0,
            beta: 0.5,
            value: small_int,
            added: FPSR_IXC,
        },
    ];
    // m = 5 and 37 leave dead rows in the last tile; n = 3 and 19 leave
    // padded columns in the last panel.
    for (m, n, kc) in [(5, 3, 2), (37, 19, 3), (64, 16, 1)] {
        for case in &cases {
            let shape = Shape::packed(m, n, kc);
            let block = Block::new(shape, case.alpha, case.beta, &mut |o, x, y| {
                (case.value)(o, x, y)
            });
            for before in EARLIER_FLAGS {
                let after = fpsr_across(&block, before, case.label);
                assert_eq!(
                    after,
                    before | case.added,
                    "{}: {m}x{n}x{kc}: FPSR {before:#x} became {after:#x}, want {:#x}",
                    case.label,
                    before | case.added
                );
            }
        }
    }
}

/// Scalar `f64` work whose result depends on every step.
#[inline(never)]
fn scalar_work(seed: f64) -> f64 {
    let mut acc = black_box(seed);
    for step in 0..64 {
        acc = acc * 1.000_000_7 + f64::from(step) * 0.25;
    }
    acc
}

#[test]
fn test_sme2_kernel_f32_repeated_calls_keep_thread_state() {
    if !hardware_or_skip(
        "test_sme2_kernel_f32_repeated_calls_keep_thread_state",
        Need::F32,
    ) {
        return;
    }
    let shape = Shape {
        layout: Layout::InPlace,
        ldc_pad: 3,
        b_gap: 4,
        lda_pad: 2,
        ..Shape::packed(70, 33, 7)
    };
    let blocks = [
        Block::random(shape, -1.5, 0.25, 1),
        Block::random(Shape::packed(17, 47, 5), 1.0, 0.0, 2),
    ];
    let first: Vec<Vec<f32>> = blocks.iter().map(Block::run).collect();
    for (block, got) in blocks.iter().zip(&first) {
        block.check(got, &block.expected(), "first call");
    }
    // The same work without kernel calls, for comparison.
    let x: Vec<f64> = (0..37).map(|v| f64::from(v) * 0.5 - 3.0).collect();
    let y: Vec<f64> = (0..37).map(|v| 1.0 / (f64::from(v) + 1.5)).collect();
    let rounds: u32 = 128;
    let mut quiet = 1.0_f64;
    for round in 0..rounds {
        quiet = scalar_work(quiet) + crate::level1::dot_f64(&x, &y) * f64::from(round);
    }
    let entries = kernel_entries();
    let mut busy = 1.0_f64;
    for round in 0..rounds {
        // An `f64` value live across the call.
        let live = black_box(f64::from(round) * 1.25 + 0.5);
        let dot = crate::level1::dot_f64(&x, &y);
        let block = &blocks[round as usize % blocks.len()];
        assert!(detect::thread_ready(), "round {round}: thread not ready");
        let got = block.run();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            first[round as usize % blocks.len()]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "round {round}: a repeated call gave other bits"
        );
        assert_eq!(live, f64::from(round) * 1.25 + 0.5, "round {round}");
        busy = scalar_work(busy) + dot * f64::from(round);
    }
    assert_eq!(busy.to_bits(), quiet.to_bits(), "f64 work around the calls");
    // Process-wide counter: other test threads' kernel calls count as well,
    // so this lower bound can be met without these rounds being counted.
    assert!(kernel_entries() >= entries + u64::from(rounds));
    assert_eq!(sme_state(), Some((0, 0)));
    assert!(detect::thread_ready());
}
