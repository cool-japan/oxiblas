//! The only way to write a streaming region: [`sme_kernels!`].
//!
//! Stable Rust has no SME intrinsics, so every streaming region is one
//! `asm!` block that enters streaming mode with `smstart`, works on the ZA
//! array and the Z/P registers, and leaves with `smstop`. Such a block is easy
//! to get subtly wrong (an instruction that is illegal in streaming mode, a
//! clobber that is missing, a `;` that Mach-O reads as a comment, a directive
//! that does not carry over, a kernel that is never assembled because nothing
//! calls it yet). The macro removes the choices that lead there:
//!
//! - **One invocation per module, any number of kernels.** Each kernel is an
//!   `unsafe fn(args: &Args)` whose only input is the address of its
//!   `#[repr(C)]` argument block, passed in `x0`. The grammar admits nothing
//!   else: no `reg`, `vreg`, `sym`, `in`, `out` or `inout` operand can be
//!   written, so no placeholder can ever print a vector register or a
//!   register the backend reserves. Named `const` operands (field offsets from
//!   `core::mem::offset_of!`) are the only placeholders.
//! - **A fixed frame.** The macro wraps the kernel's prologue and body in the
//!   frame below and adds the full clobber list (`x0` as a clobbered input,
//!   `x1`–`x17`, `x20`–`x28`, `v0`–`v31`, `p0`–`p15`, `ffr`) and
//!   `options(nostack)`. Nothing else is accepted: no `pure`, `nomem`,
//!   `readonly` or `preserves_flags`.
//! - **Registration and keep-alive.** Each kernel gets a `#[used]` static that
//!   holds its address, so it is code-generated (and therefore assembled) in
//!   every build of the module even before anything calls it: `cargo check`
//!   and clippy never run the assembler, `cargo build` does. In test builds
//!   the invocation also emits `SME_KERNELS`, the registry entries of the
//!   module: name, source file, the exact template string that `asm!`
//!   receives (both are expanded from the same macro rule), and the declared
//!   `const` operand names. The static template lint walks every entry, and a
//!   source scan proves that every kernel in the sources has an entry.
//! - **Docs only.** A kernel takes `///` doc comments (at least one line),
//!   and no other attribute: a `#[cfg]` could hide a kernel from the lint.
//! - **Literal text.** Every prologue and body entry is one string literal
//!   (no `concat!`, no helper macros), and kernel names are written out (the
//!   source scan rejects `streaming fn $name` outside this file), so what the
//!   scan finds in the sources is exactly what the registry holds.
//!
//! # The frame
//!
//! For a kernel with prologue lines `P…` and body lines `B…`, `asm!` receives
//! exactly this template (one statement per line, never `;`):
//!
//! ```text
//! .arch_extension sme2
//! .arch_extension sme-f64f64   (only with `extensions: [f64f64]`)
//! mrs x17, fpsr                the caller's cumulative FP flags
//! P…                           straight-line base A64: loads from the block
//! smstart                      PSTATE.SM = PSTATE.ZA = 1, ZA = 0, FPSR = 0x0800009f
//! msr fpsr, xzr                flags raised inside the region start from zero
//! B…                           the kernel
//! 90:                          the exit label (branch to it with `b 90f`)
//! mrs x16, fpsr                the flags the kernel raised
//! smstop                       PSTATE.SM = PSTATE.ZA = 0, FPSR = 0x0800009f
//! orr x17, x17, x16
//! msr fpsr, x17                caller's flags | kernel's flags
//! ```
//!
//! Every change of PSTATE.SM sets FPSR to `0x0800009f` (`ResetSVEState` in the
//! Arm ARM; measured under QEMU, see the hand-over notes). Without the FPSR
//! lines every call would leave IOC, DZC, OFC, UFC, IXC, IDC and QC set for a
//! caller that inspects the flags. With them a kernel call leaves the
//! caller's flags plus the flags raised by the kernel's SVE vector arithmetic
//! (for example the `fmul` by alpha of the epilogue). The ZA-targeting
//! instructions (FMOPA, and FADD / FMLA on the ZA array) raise no flags at
//! all: the Arm ARM evaluates them with `FPMulAdd_ZA` / `FPAdd_ZA`, which
//! generate no floating-point exceptions and force the default NaN. So a
//! kernel call can leave fewer flags than the NEON kernel would (no IXC from
//! the depth loop), never more than its SVE arithmetic raises.
//! `x17` is reserved for the whole template; `x16` is free inside the body
//! (its last value is overwritten by the frame).
//!
//! # Register conventions
//!
//! | Registers | Use |
//! |---|---|
//! | `x0` | address of the argument block (clobbered input; keep it as the base) |
//! | `x1`–`x15`, `x20`–`x28` | pointers, strides, counters (each kernel documents its map) |
//! | `x16` | free inside the body only; the frame overwrites it before `smstop` |
//! | `x17` | reserved by the FPSR frame: never written or read by a kernel |
//! | `x18`, `x19`, `x29`, `x30`, `sp` | never used (platform, base, frame, link, stack) |
//! | `w8`–`w11` | ZA vector-select for `za.s[wv, off, vgx4]`; offsets 0–7 only |
//! | `w12`–`w15` | tile-slice select for `zaNh.s[ws, off]` and `mova` |
//! | `z0`–`z7` | A vectors (two sets for k-unrolling) |
//! | `z8`–`z15` | B rows; also any `Zm` of an indexed or multi-and-single form |
//! | `z16`–`z27` | C, epilogue and transposer temporaries |
//! | `z28`, `z29` | alpha, beta (`dup z28.s, w20`); NOT usable as an indexed `Zm` |
//! | `p0` | all-true |
//! | `p1`–`p7` | `whilelt` edge predicates (single-slice ZA ld/st need `p0`–`p7`) |
//! | `pn8` | all-true predicate-as-counter |
//! | `pn9`–`pn11` | `whilelt …, vlx4` counters (multi-vector `PNg` must be `pn8`–`pn15`) |
//!
//! Immediate ranges and register ranges the assembler enforces (verified with
//! rustc 1.89 and 1.95 on Mach-O and ELF):
//!
//! - `za.s[w8, off, vgx4]` / `za.d[…]`: `off` 0–7, select register `w8`–`w11`.
//!   With SVL 512, group `v` covers ZA array vectors `v, v+16, v+32, v+48`
//!   (`.s`), i.e. tile `v % 4`, slices `v / 4`, `+4`, `+8`, `+12`.
//! - 4-slice `mova {z0.s-z3.s}, zaNh.s[w12, 0:3]` (and the reverse): offset
//!   `0:3` only; step through slices with `w12`–`w15`.
//! - Single-slice `ld1w` / `st1w` of `zaNh.s[w12, off]`: `off` 0–3;
//!   `ld1d` / `st1d` of `zaNh.d[w12, off]`: `off` 0–1.
//! - Indexed SVE `fmul`: `Zm` in `z0`–`z7` (`.s`) or `z0`–`z15` (`.d`). SME2
//!   indexed and multi-and-single `fmla`: `Zm` in `z0`–`z15`. Lane indices 0–3
//!   (`.s`) and 0–1 (`.d`) select an element of EACH 128-bit segment.
//! - Multi-vector groups start at a multiple of 4; strided groups start at
//!   `z0`–`z3` or `z16`–`z19`. `.s` tiles are `za0`–`za3`, `.d` tiles `za0`–`za7`.
//!
//! Labels: numeric only, never made only of the digits 0 and 1 (`10`, `11`,
//! `100` read as binary literals); `90` belongs to the frame. Escape braces:
//! `zero {{za}}`, `ld1w {{z0.s-z3.s}}, …`.
//!
//! # Example
//!
//! ```text
//! sme_kernels! {
//!     /// Stores one 16 × 16 outer product (see the module docs).
//!     ///
//!     /// # Safety
//!     ///
//!     /// SME2 must be available (`detect::thread_ready()` on this thread) and
//!     /// the pointers in `args` valid as the contract describes.
//!     pub(super) streaming fn outer_f32(args: &MacroArgs<f32>) {
//!         extensions: [],
//!         consts: [
//!             off_a = core::mem::offset_of!(MacroArgs<f32>, a),
//!             off_b = core::mem::offset_of!(MacroArgs<f32>, b),
//!         ],
//!         prologue: [
//!             "ldr x1, [x0, #{off_a}]",
//!             "ldr x2, [x0, #{off_b}]",
//!         ],
//!         body: [
//!             "ptrue p0.s",
//!             "ld1w {{z0.s}}, p0/z, [x1]",
//!             "ld1w {{z1.s}}, p0/z, [x2]",
//!             "fmopa za0.s, p0/m, p0/m, z1.s, z0.s",
//!         ],
//!     }
//! }
//! ```
//!
//! A complete, executed example is the smoke kernel in `tests_smoke.rs`.

/// One registered streaming region: what the template lint checks and what
/// the source scan matches against the kernels it finds in the sources.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelEntry {
    /// The kernel function's name.
    pub(crate) name: &'static str,
    /// `file!()` of the `sme_kernels!` invocation.
    pub(crate) file: &'static str,
    /// The exact template string that `asm!` assembles.
    pub(crate) template: &'static str,
    /// The names of the declared `const` operands, the only placeholders a
    /// template may use.
    pub(crate) consts: &'static [&'static str],
    /// Whether the template enables `sme-f64f64` (f64 outer products).
    pub(crate) f64f64: bool,
}

/// The `.arch_extension` line of an optional extension. Only `f64f64` exists;
/// any other name is a compile error.
// Library builds have no kernel before Wave 2 (the smoke kernels are
// test-only), so the frame rules are unused there until then.
#[cfg_attr(
    not(test),
    allow(
        unused_macros,
        reason = "expanded by every library kernel (Wave 2 onwards)"
    )
)]
macro_rules! sme_extension {
    (f64f64) => {
        ".arch_extension sme-f64f64\n"
    };
}
#[cfg_attr(
    not(test),
    allow(
        unused_imports,
        reason = "expanded by every library kernel (Wave 2 onwards)"
    )
)]
pub(crate) use sme_extension;

/// Whether an extension list enables `sme-f64f64`.
#[cfg(test)]
macro_rules! sme_has_f64f64 {
    () => {
        false
    };
    (f64f64) => {
        true
    };
}
#[cfg(test)]
pub(crate) use sme_has_f64f64;

/// The full template of one kernel: directives, the FPSR frame, the prologue
/// and the body. Used for both `asm!` and the registry, so the lint checks
/// exactly the text that is assembled.
// Library builds have no kernel before Wave 2 (the smoke kernels are
// test-only), so the frame rules are unused there until then.
#[cfg_attr(
    not(test),
    allow(
        unused_macros,
        reason = "expanded by every library kernel (Wave 2 onwards)"
    )
)]
macro_rules! sme_template {
    ([$($ext:ident)?], [$($pro:literal),*], [$($body:literal),*]) => {
        concat!(
            ".arch_extension sme2\n",
            $($crate::level3::sme2::asm::sme_extension!($ext),)?
            "mrs x17, fpsr\n",
            $($pro, "\n",)*
            "smstart\n",
            "msr fpsr, xzr\n",
            $($body, "\n",)*
            "90:\n",
            "mrs x16, fpsr\n",
            "smstop\n",
            "orr x17, x17, x16\n",
            "msr fpsr, x17",
        )
    };
}
#[cfg_attr(
    not(test),
    allow(
        unused_imports,
        reason = "expanded by every library kernel (Wave 2 onwards)"
    )
)]
pub(crate) use sme_template;

/// Defines the streaming kernels of one module; see the module docs.
///
/// An empty invocation (`sme_kernels! {}`) declares a module that has no
/// kernels yet: it registers an empty list, so the module is already part of
/// the registry.
macro_rules! sme_kernels {
    () => {
        /// The streaming kernels of this module (none).
        #[cfg(test)]
        pub(crate) const SME_KERNELS: &[$crate::level3::sme2::asm::KernelEntry] = &[];
    };
    ($(
        $(#[doc = $doc:literal])+
        $vis:vis streaming fn $name:ident(args: &$args:ty) {
            extensions: [$($ext:ident)?],
            consts: [$($cname:ident = $cval:expr),* $(,)?],
            prologue: [$($pro:literal),* $(,)?],
            body: [$($body:literal),* $(,)?] $(,)?
        }
    )+) => {
        $(
            $(#[doc = $doc])+
            #[inline(never)]
            $vis unsafe fn $name(args: &$args) {
                #[cfg(test)]
                $crate::level3::sme2::test_support::note_kernel_entry();
                // SAFETY: the caller upholds the kernel's documented contract
                // (SME2 available and the thread ready, argument block valid).
                // The template is the fixed frame of `asm.rs`: it reads the
                // block through `x0`, saves and restores the caller's FPSR
                // flags, enters and leaves streaming mode itself, and every
                // register it may touch is declared clobbered below.
                unsafe {
                    core::arch::asm!(
                        $crate::level3::sme2::asm::sme_template!(
                            [$($ext)?], [$($pro),*], [$($body),*]
                        ),
                        $($cname = const $cval,)*
                        inout("x0") core::ptr::from_ref(args) => _,
                        out("x1") _, out("x2") _, out("x3") _, out("x4") _,
                        out("x5") _, out("x6") _, out("x7") _, out("x8") _,
                        out("x9") _, out("x10") _, out("x11") _, out("x12") _,
                        out("x13") _, out("x14") _, out("x15") _, out("x16") _,
                        out("x17") _, out("x20") _, out("x21") _, out("x22") _,
                        out("x23") _, out("x24") _, out("x25") _, out("x26") _,
                        out("x27") _, out("x28") _,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                        out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        out("p0") _, out("p1") _, out("p2") _, out("p3") _,
                        out("p4") _, out("p5") _, out("p6") _, out("p7") _,
                        out("p8") _, out("p9") _, out("p10") _, out("p11") _,
                        out("p12") _, out("p13") _, out("p14") _, out("p15") _,
                        out("ffr") _,
                        options(nostack),
                    );
                }
            }

            // Keeps the kernel code-generated, and therefore assembled, in
            // every build, also before anything calls it.
            const _: () = {
                #[used]
                static KEEP_ALIVE: unsafe fn(&$args) = $name;
            };
        )+

        /// The streaming kernels of this module, for the template lint and
        /// the source scan.
        #[cfg(test)]
        pub(crate) const SME_KERNELS: &[$crate::level3::sme2::asm::KernelEntry] = &[$(
            $crate::level3::sme2::asm::KernelEntry {
                name: stringify!($name),
                file: file!(),
                template: $crate::level3::sme2::asm::sme_template!(
                    [$($ext)?], [$($pro),*], [$($body),*]
                ),
                consts: &[$(stringify!($cname)),*],
                f64f64: $crate::level3::sme2::asm::sme_has_f64f64!($($ext)?),
            },
        )+];
    };
}
pub(crate) use sme_kernels;
