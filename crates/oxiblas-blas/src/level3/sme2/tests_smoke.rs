//! End-to-end smoke test of the streaming-region chain: `sme_kernels!` →
//! assembler (Mach-O on the Mac, ELF in the QEMU container) → registry →
//! template lint → `detect::thread_ready()` → the FPSR frame → SVCR and
//! TPIDR2_EL0 back at zero.
//!
//! The two kernels below are real contract kernels (`args.rs`) for the
//! smallest case: one 16 × 16 (`f32`) or 8 × 8 (`f64`) tile, `kc = 1`, modes
//! M0 (direct ZA-slice stores) and M2 (MOVA, FMUL by alpha, store). Their
//! results are compared with `args::reference`, so the argument block, the
//! `const` offsets and the operand orientation are exercised for real. On a
//! host without SME2 the hardware tests print a skip line.
//!
//! FMOPA raises no FP flags, on hardware as under QEMU (the Arm ARM's
//! ZA-targeting `FPMulAdd_ZA` runs with exceptions off), so the FPSR tests
//! raise their flags with the SVE `fmul` of mode M2.

use super::args::{
    Epilogue, MacroArgs, OFF_A, OFF_ALPHA, OFF_B, OFF_C, OFF_LDC, OFF_M, OFF_MODE, OFF_N,
    SmeElement, reference, same_result,
};
use super::asm::sme_kernels;
use super::detect;
use super::test_support::{
    FPSR_AFTER_SM_CHANGE, FPSR_DZC, FPSR_IDC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_QC, FPSR_UFC,
    Need, fpsr_read, fpsr_write, hardware_or_skip, kernel_entries, sme_state,
};

sme_kernels! {
    /// One 16 × 16 `f32` tile with `kc = 1`: `C(i, j) = A(i, 0) * B(0, j)`,
    /// stored directly from ZA (M0) or scaled by alpha (any other mode).
    ///
    /// Register map: `x1` A, `x2` B, `x3` C column, `x4` `ldc` in bytes, `x5`
    /// `m`, `x6` `n`, `x7` mode, `x9` zero, `x10` column index, `w12` slice,
    /// `w20` alpha bits; `z0` A, `z8` B row, `z16` scaled column, `z28` alpha.
    ///
    /// # Safety
    ///
    /// `detect::thread_ready()` must hold on this thread, `1 ≤ m, n ≤ 16`,
    /// `kc = 1`, and the block must be valid as `args.rs` describes.
    pub(super) streaming fn smoke_tile_f32(args: &MacroArgs<f32>) {
        extensions: [],
        consts: [
            off_a = OFF_A,
            off_b = OFF_B,
            off_c = OFF_C,
            off_ldc = OFF_LDC,
            off_m = OFF_M,
            off_n = OFF_N,
            off_mode = OFF_MODE,
            off_alpha = OFF_ALPHA,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_b}]",
            "ldr x3, [x0, #{off_c}]",
            "ldr x4, [x0, #{off_ldc}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "ldr x7, [x0, #{off_mode}]",
            "ldr w20, [x0, #{off_alpha}]",
            "lsl x4, x4, #2",
        ],
        body: [
            "ptrue p0.s",
            "mov x9, #0",
            "whilelt p1.s, x9, x5",
            "zero {{za}}",
            "ld1w {{z0.s}}, p1/z, [x1]",
            "ld1w {{z8.s}}, p0/z, [x2]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "dup z28.s, w20",
            "mov w12, #0",
            "mov x10, #0",
            "20:",
            "cmp x7, #0",
            "b.ne 30f",
            "st1w {{za0h.s[w12, 0]}}, p1, [x3]",
            "b 40f",
            "30:",
            "mova z16.s, p0/m, za0h.s[w12, 0]",
            "fmul z16.s, z16.s, z28.s",
            "st1w {{z16.s}}, p1, [x3]",
            "40:",
            "add x3, x3, x4",
            "add w12, w12, #1",
            "add x10, x10, #1",
            "cmp x10, x6",
            "b.lo 20b",
        ],
    }

    /// One 8 × 8 `f64` tile with `kc = 1`; the `f64` twin of
    /// [`smoke_tile_f32`] (same register map, `x20` alpha bits).
    ///
    /// # Safety
    ///
    /// As for [`smoke_tile_f32`] with `1 ≤ m, n ≤ 8`; FEAT_SME_F64F64 must be
    /// present.
    pub(super) streaming fn smoke_tile_f64(args: &MacroArgs<f64>) {
        extensions: [f64f64],
        consts: [
            off_a = OFF_A,
            off_b = OFF_B,
            off_c = OFF_C,
            off_ldc = OFF_LDC,
            off_m = OFF_M,
            off_n = OFF_N,
            off_mode = OFF_MODE,
            off_alpha = OFF_ALPHA,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_b}]",
            "ldr x3, [x0, #{off_c}]",
            "ldr x4, [x0, #{off_ldc}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "ldr x7, [x0, #{off_mode}]",
            "ldr x20, [x0, #{off_alpha}]",
            "lsl x4, x4, #3",
        ],
        body: [
            "ptrue p0.d",
            "mov x9, #0",
            "whilelt p1.d, x9, x5",
            "zero {{za}}",
            "ld1d {{z0.d}}, p1/z, [x1]",
            "ld1d {{z8.d}}, p0/z, [x2]",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
            "dup z28.d, x20",
            "mov w12, #0",
            "mov x10, #0",
            "20:",
            "cmp x7, #0",
            "b.ne 30f",
            "st1d {{za0h.d[w12, 0]}}, p1, [x3]",
            "b 40f",
            "30:",
            "mova z16.d, p0/m, za0h.d[w12, 0]",
            "fmul z16.d, z16.d, z28.d",
            "st1d {{z16.d}}, p1, [x3]",
            "40:",
            "add x3, x3, x4",
            "add w12, w12, #1",
            "add x10, x10, #1",
            "cmp x10, x6",
            "b.lo 20b",
        ],
    }
}

/// A smoke-kernel call on owned buffers.
struct Tile<T> {
    /// One packed A depth step: 64 rows (rows `≥ m` are never read).
    a: Vec<T>,
    /// One packed B row of `NR` columns (zero past `n`).
    b: Vec<T>,
    /// C with `ldc` rows per column and two spare columns.
    c: Vec<T>,
    /// Leading dimension of C.
    ldc: usize,
    /// Rows.
    m: usize,
    /// Columns.
    n: usize,
}

/// Fills C with a NaN that no kernel produces, to spot stray reads and writes.
fn sentinel<T: SmeElement>() -> T {
    T::from_bits64(if T::NR == 16 {
        0x7fc0_dead
    } else {
        0x7ff8_0000_dead_beef
    })
}

impl<T: SmeElement> Tile<T> {
    /// A tile whose products are exact small integers times `scale`.
    fn new(m: usize, n: usize, scale: T) -> Self {
        let ldc = m + 5;
        let int = |v: usize| -> T { (0..v).fold(T::ZERO, |acc, _| acc + T::ONE) };
        let a = (0..64)
            .map(|i| {
                if i < m {
                    int(i % 7 + 1) * scale
                } else {
                    sentinel()
                }
            })
            .collect();
        let b = (0..T::NR)
            .map(|j| if j < n { int(j % 5 + 2) } else { T::ZERO })
            .collect();
        Self {
            a,
            b,
            c: vec![sentinel(); ldc * (n + 2)],
            ldc,
            m,
            n,
        }
    }

    /// The argument block for alpha (beta = 0: mode M0 or M2).
    fn args(&mut self, alpha: T) -> MacroArgs<T> {
        MacroArgs::packed(
            self.a.as_ptr(),
            self.b.as_ptr(),
            self.c.as_mut_ptr(),
            self.ldc,
            self.m,
            self.n,
            1,
            alpha,
            T::ZERO,
        )
    }

    /// Checks C against `reference` inside the block and against the
    /// sentinel outside it.
    #[track_caller]
    fn check(&self, alpha: T, label: &str) {
        let mut want = vec![sentinel::<T>(); self.c.len()];
        let args = MacroArgs::packed(
            self.a.as_ptr(),
            self.b.as_ptr(),
            want.as_mut_ptr(),
            self.ldc,
            self.m,
            self.n,
            1,
            alpha,
            T::ZERO,
        );
        // SAFETY: the block lies inside `self.a`, `self.b` and `want`, which
        // has the shape of `self.c`.
        unsafe { reference(&args) };
        for (index, (got, want)) in self.c.iter().zip(&want).enumerate() {
            let (i, j) = (index % self.ldc, index / self.ldc);
            let inside = i < self.m && j < self.n;
            let ok = if inside {
                same_result(*got, *want)
            } else {
                got.bits() == sentinel::<T>().bits()
            };
            assert!(
                ok,
                "{label}: C({i}, {j}) = {:#x}, want {:#x}",
                got.bits(),
                want.bits()
            );
        }
    }
}

/// Runs a smoke kernel on the tile and checks the thread state around it.
#[track_caller]
fn run<T: SmeElement>(kernel: unsafe fn(&MacroArgs<T>), tile: &mut Tile<T>, alpha: T) {
    let args = tile.args(alpha);
    assert!(detect::thread_ready(), "thread not ready before the kernel");
    let entries = kernel_entries();
    // SAFETY: the thread is ready (checked above), 1 ≤ m, n ≤ NR, kc = 1,
    // and the block lies inside the tile's buffers.
    unsafe { kernel(&args) };
    assert!(
        kernel_entries() > entries,
        "the kernel entry was not counted"
    );
    assert_eq!(
        sme_state(),
        Some((0, 0)),
        "SVCR / TPIDR2_EL0 after the kernel"
    );
    assert!(detect::thread_ready(), "thread not ready after the kernel");
}

#[test]
fn test_sme2_smoke_kernels_are_registered() {
    let names: Vec<&str> = SME_KERNELS.iter().map(|entry| entry.name).collect();
    assert_eq!(names, ["smoke_tile_f32", "smoke_tile_f64"]);
    assert!(!SME_KERNELS[0].f64f64 && SME_KERNELS[1].f64f64);
    assert!(
        SME_KERNELS
            .iter()
            .all(|entry| entry.file.ends_with("tests_smoke.rs"))
    );
    // Mode selection used by the tests below.
    assert_eq!(Epilogue::select(1.0_f32, 0.0), Epilogue::Store);
    assert_eq!(Epilogue::select(0.5_f64, -0.0), Epilogue::Scale);
}

#[test]
fn test_sme2_smoke_f32_tile() {
    if !hardware_or_skip("test_sme2_smoke_f32_tile", Need::F32) {
        return;
    }
    let third = std::hint::black_box(1.0_f32) / std::hint::black_box(3.0_f32);
    for (m, n) in [(16, 16), (13, 11), (1, 1), (16, 3), (2, 16)] {
        for alpha in [1.0, 0.75, third, -2.5] {
            let mut tile = Tile::new(m, n, 0.5_f32);
            run(smoke_tile_f32, &mut tile, alpha);
            tile.check(alpha, &format!("f32 {m}x{n} alpha {alpha}"));
        }
    }
}

#[test]
fn test_sme2_smoke_f64_tile() {
    if !hardware_or_skip("test_sme2_smoke_f64_tile", Need::F64) {
        return;
    }
    let third = std::hint::black_box(1.0_f64) / std::hint::black_box(3.0_f64);
    for (m, n) in [(8, 8), (5, 7), (1, 1), (8, 2), (3, 8)] {
        for alpha in [1.0, 0.75, third, -2.5] {
            let mut tile = Tile::new(m, n, 0.5_f64);
            run(smoke_tile_f64, &mut tile, alpha);
            tile.check(alpha, &format!("f64 {m}x{n} alpha {alpha}"));
        }
    }
}

/// FPSR after one kernel call that starts from `before`.
fn fpsr_across<T: SmeElement>(
    kernel: unsafe fn(&MacroArgs<T>),
    tile: &mut Tile<T>,
    alpha: T,
    before: u64,
) -> u64 {
    let args = std::hint::black_box(tile.args(alpha));
    assert!(detect::thread_ready());
    let saved = fpsr_read();
    fpsr_write(before);
    // SAFETY: as in `run`.
    unsafe { kernel(&args) };
    let after = fpsr_read();
    fpsr_write(saved);
    after
}

/// The FPSR protocol of `asm.rs` for one kernel.
fn check_fpsr_protocol<T: SmeElement>(kernel: unsafe fn(&MacroArgs<T>), third: T, half: T) {
    let earlier = [
        0,
        FPSR_DZC,
        FPSR_DZC | FPSR_OFC,
        FPSR_IOC | FPSR_UFC | FPSR_IDC,
        FPSR_IXC,
        FPSR_QC,
    ];
    for before in earlier {
        // M0 on exact products and M2 with an exact alpha raise nothing: the
        // caller's flags survive and nothing is added.
        for alpha in [T::ONE, half] {
            let mut tile = Tile::new(T::NR, T::NR, T::ONE);
            let after = fpsr_across(kernel, &mut tile, alpha, before);
            assert_eq!(
                after, before,
                "exact kernel moved FPSR from {before:#x} to {after:#x}"
            );
            tile.check(alpha, "fpsr exact");
        }
        // M2 with alpha = 1/3 on integer products is inexact: IXC only.
        let mut tile = Tile::new(T::NR, T::NR, T::ONE);
        let after = fpsr_across(kernel, &mut tile, third, before);
        assert_eq!(
            after,
            before | FPSR_IXC,
            "inexact kernel moved FPSR from {before:#x} to {after:#x}"
        );
        assert_ne!(after & FPSR_AFTER_SM_CHANGE, FPSR_AFTER_SM_CHANGE);
        tile.check(third, "fpsr inexact");
    }
}

#[test]
fn test_sme2_smoke_fpsr_protocol_f32() {
    if !hardware_or_skip("test_sme2_smoke_fpsr_protocol_f32", Need::F32) {
        return;
    }
    let third = std::hint::black_box(1.0_f32) / std::hint::black_box(3.0_f32);
    check_fpsr_protocol(smoke_tile_f32, third, 0.5);
}

#[test]
fn test_sme2_smoke_fpsr_protocol_f64() {
    if !hardware_or_skip("test_sme2_smoke_fpsr_protocol_f64", Need::F64) {
        return;
    }
    let third = std::hint::black_box(1.0_f64) / std::hint::black_box(3.0_f64);
    check_fpsr_protocol(smoke_tile_f64, third, 0.5);
}

#[test]
fn test_sme2_smoke_many_calls_keep_thread_state() {
    if !hardware_or_skip("test_sme2_smoke_many_calls_keep_thread_state", Need::F32) {
        return;
    }
    // Alternate SME2 calls with NEON-heavy Rust work on the same thread; the
    // thread must stay ready and the f64 values live across calls intact.
    let mut tile = Tile::new(16, 16, 1.0_f32);
    let mut acc = std::hint::black_box(1.0_f64);
    for round in 0..200 {
        run(smoke_tile_f32, &mut tile, 1.0);
        acc = std::hint::black_box(acc * 1.000_001 + f64::from(round));
    }
    tile.check(1.0, "after 200 calls");
    assert!(acc.is_finite() && acc > 1.0);
}
