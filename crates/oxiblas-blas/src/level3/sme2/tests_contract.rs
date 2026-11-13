//! The kernel contract of `args.rs`: the argument block's layout, the mode
//! selection, and the proof that `args::reference` (the executable definition
//! of every C element) equals the NEON blocked path bit for bit.
//!
//! The NEON side is the real one: `macro_panel_multiply_with` with the
//! dispatched NEON micro-kernel on panels packed for its shape, edge tiles
//! included. These tests run on any aarch64 host (no SME needed); under
//! `force-scalar` the dispatched kernel is the portable one, whose epilogue
//! differs, so the comparison is skipped there.

use oxiblas_matrix::MatMut;

use super::args::{
    Epilogue, MR, MacroArgs, OFF_A, OFF_A_DEPTH, OFF_A_PACK, OFF_A_PANEL, OFF_ALPHA, OFF_B,
    OFF_B_PANEL, OFF_BETA, OFF_C, OFF_KC, OFF_LDC, OFF_M, OFF_MODE, OFF_N, OFF_PREFETCH,
    PREFETCH_OFF, SmeElement, reference, same_result,
};
use crate::level3::gemm::macro_panel_multiply_with;
use crate::level3::gemm_kernel::GemmKernel;

#[test]
fn test_sme2_contract_argument_block_layout() {
    // Every field is 8 bytes, in declaration order, for both types.
    let offsets = [
        OFF_A,
        OFF_A_DEPTH,
        OFF_A_PANEL,
        OFF_B,
        OFF_B_PANEL,
        OFF_C,
        OFF_LDC,
        OFF_M,
        OFF_N,
        OFF_KC,
        OFF_ALPHA,
        OFF_BETA,
        OFF_MODE,
        OFF_A_PACK,
        OFF_PREFETCH,
    ];
    for (index, offset) in offsets.iter().enumerate() {
        assert_eq!(*offset, 8 * index, "field {index}");
    }
    assert_eq!(size_of::<MacroArgs<f32>>(), 120);
    assert_eq!(size_of::<MacroArgs<f64>>(), 120);
    assert_eq!(core::mem::offset_of!(MacroArgs<f64>, alpha_bits), OFF_ALPHA);
    assert_eq!(
        core::mem::offset_of!(MacroArgs<f64>, prefetch),
        OFF_PREFETCH
    );
    // `ldr x…, [x0, #off]` takes offsets up to 32760 in steps of 8.
    assert!(offsets.iter().all(|off| off % 8 == 0 && *off <= 32760));
}

#[test]
fn test_sme2_contract_packed_block_and_bits() {
    let (a, b, mut c) = ([0.0_f32; 64], [0.0_f32; 16], [0.0_f32; 4]);
    let args = MacroArgs::packed(
        a.as_ptr(),
        b.as_ptr(),
        c.as_mut_ptr(),
        2,
        2,
        2,
        7,
        1.5,
        -0.0,
    );
    assert_eq!(
        (args.a_depth, args.a_panel, args.b_panel),
        (MR, MR * 7, 7 * 16)
    );
    assert_eq!((args.ldc, args.m, args.n, args.kc), (2, 2, 2, 7));
    assert_eq!(args.alpha_bits, u64::from(1.5_f32.to_bits()));
    assert_eq!(args.beta_bits, u64::from((-0.0_f32).to_bits()));
    assert_eq!(args.mode, Epilogue::Scale as u64);
    assert!(args.a_pack.is_null());
    assert_eq!(args.prefetch, PREFETCH_OFF);
    let args = MacroArgs::<f64>::packed(
        core::ptr::null(),
        core::ptr::null(),
        core::ptr::null_mut(),
        1,
        1,
        1,
        3,
        1.0,
        1.0,
    );
    assert_eq!(args.b_panel, 3 * 8);
    assert_eq!(args.mode, Epilogue::Add as u64);
    // Bit patterns round-trip, f32 zero-extended.
    for value in [
        0.0_f32,
        -0.0,
        1.0,
        -2.5,
        f32::MIN_POSITIVE / 4.0,
        f32::INFINITY,
    ] {
        assert_eq!(value.bits() >> 32, 0);
        assert_eq!(
            <f32 as SmeElement>::from_bits64(value.bits()).to_bits(),
            value.to_bits()
        );
    }
    for value in [
        0.0_f64,
        -0.0,
        1.0,
        -2.5,
        f64::MIN_POSITIVE / 4.0,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(
            <f64 as SmeElement>::from_bits64(value.bits()).to_bits(),
            value.to_bits()
        );
    }
    assert_eq!(
        (f32::NR, f32::TILE_ROWS, f64::NR, f64::TILE_ROWS),
        (16, 16, 8, 8)
    );
    assert!(same_result(f32::NAN, -f32::NAN) && !same_result(0.0_f32, -0.0));
}

#[test]
fn test_sme2_contract_mode_selection_mirrors_neon() {
    use Epilogue::{Add, Scale, ScaleCombine, Store};
    let cases: [(f64, f64, Epilogue); 9] = [
        (1.0, 0.0, Store),
        (1.0, -0.0, Store),
        (1.0, 1.0, Add),
        (2.0, 0.0, Scale),
        (-1.0, -0.0, Scale),
        (2.0, 1.0, ScaleCombine),
        (1.0, 0.5, ScaleCombine),
        (f64::NAN, 0.0, Scale),
        (1.0, f64::NAN, ScaleCombine),
    ];
    for (alpha, beta, mode) in cases {
        assert_eq!(
            Epilogue::select(alpha, beta),
            mode,
            "alpha {alpha}, beta {beta}"
        );
        assert_eq!(Epilogue::select(alpha as f32, beta as f32), mode);
        assert_eq!(mode.reads_c(), matches!(mode, Add | ScaleCombine));
    }
    assert_eq!(
        [Store as u64, Add as u64, Scale as u64, ScaleCombine as u64],
        [0, 1, 2, 3]
    );
}

/// Deterministic test values: mixed signs and magnitudes, a few exact zeros.
fn values<T: SmeElement>(len: usize, seed: u64, from_f64: fn(f64) -> T) -> Vec<T> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            let scale = [1.0, 1.0e-3, 37.0, 0.0][(state & 3) as usize];
            from_f64((unit * 2.0 - 1.0) * scale)
        })
        .collect()
}

/// Packs column-major `a` (`m × kc`, leading dimension `m`) into panels of
/// `mr` rows: `p * mr + i % mr` inside panel `i / mr`, zero-padded.
fn pack_rows<T: SmeElement>(a: &[T], m: usize, kc: usize, mr: usize) -> Vec<T> {
    let panels = m.div_ceil(mr);
    let mut packed = vec![T::ZERO; panels * mr * kc];
    for i in 0..m {
        for p in 0..kc {
            packed[(i / mr) * mr * kc + p * mr + i % mr] = a[i + p * m];
        }
    }
    packed
}

/// Packs row-major `b` (`kc × n`, leading dimension `n`) into panels of `nr`
/// columns: `p * nr + j % nr` inside panel `j / nr`, zero-padded.
fn pack_cols<T: SmeElement>(b: &[T], kc: usize, n: usize, nr: usize) -> Vec<T> {
    let panels = n.div_ceil(nr);
    let mut packed = vec![T::ZERO; panels * nr * kc];
    for p in 0..kc {
        for j in 0..n {
            packed[(j / nr) * nr * kc + p * nr + j % nr] = b[p * n + j];
        }
    }
    packed
}

/// Compares `reference` with the NEON blocked path for one shape and one
/// alpha/beta pair, and the in-place A strides with the packed ones.
fn compare<T>(m: usize, n: usize, kc: usize, alpha: T, beta: T, from_f64: fn(f64) -> T)
where
    T: SmeElement + GemmKernel + oxiblas_core::scalar::Field,
{
    let seed = (m * 1_000_003 + n * 1009 + kc) as u64;
    let a = values(m * kc, seed, from_f64);
    let b = values(kc * n, seed + 1, from_f64);
    let ldc = m + 3;
    let mode = Epilogue::select(alpha, beta);
    // C outside the block holds a sentinel; inside, NaN when C is not read.
    let sentinel = T::from_bits64(if T::NR == 16 {
        0x7fc0_beef
    } else {
        0x7ff8_0000_0000_beef
    });
    let start = values(ldc * (n + 1), seed + 2, from_f64);
    let mut c0 = vec![sentinel; ldc * (n + 1)];
    for j in 0..n {
        for i in 0..m {
            c0[i + j * ldc] = if mode.reads_c() {
                start[i + j * ldc]
            } else {
                T::from_bits64(u64::MAX >> 1)
            };
        }
    }

    // NEON: the dispatched micro-kernel on panels of its own shape.
    let kernel = T::micro_kernel_dispatch();
    let (mr, nr) = (kernel.shape.mr, kernel.shape.nr);
    let neon_a = pack_rows(&a, m, kc, mr);
    let neon_b = pack_cols(&b, kc, n, nr);
    let mut neon_c = c0.clone();
    let view = MatMut::from_strided(&mut neon_c, m, n, ldc);
    let Some(view) = view else {
        panic!("C view {m}x{n} ldc {ldc}");
    };
    macro_panel_multiply_with(alpha, &neon_a, m, kc, &neon_b, n, beta, view, &kernel);

    // The contract, on the SME2 layouts.
    let sme_a = pack_rows(&a, m, kc, MR);
    let sme_b = pack_cols(&b, kc, n, T::NR);
    let mut sme_c = c0.clone();
    let args = MacroArgs::packed(
        sme_a.as_ptr(),
        sme_b.as_ptr(),
        sme_c.as_mut_ptr(),
        ldc,
        m,
        n,
        kc,
        alpha,
        beta,
    );
    // SAFETY: the packed panels cover `ceil(m / 64)` A panels and
    // `ceil(n / NR)` B panels of depth `kc`, and C holds `ldc * (n + 1)`
    // elements with `ldc ≥ m`.
    unsafe { reference(&args) };

    // A read in place (column-major, `lda = m`): same values, other strides.
    let mut inplace_c = c0.clone();
    let inplace = MacroArgs {
        a: a.as_ptr(),
        a_depth: m,
        a_panel: MR,
        c: inplace_c.as_mut_ptr(),
        ..args
    };
    // SAFETY: `A(i, p) = a[i + p * m]` for `i < m`, `p < kc`, inside `a`;
    // B and C as above.
    unsafe { reference(&inplace) };

    for (index, ((neon, sme), inplace)) in neon_c.iter().zip(&sme_c).zip(&inplace_c).enumerate() {
        let (i, j) = (index % ldc, index / ldc);
        let label = format!("{m}x{n}x{kc} {mode:?} C({i}, {j})");
        assert!(
            same_result(*sme, *neon),
            "{label}: contract {:#x}, NEON {:#x}",
            sme.bits(),
            neon.bits()
        );
        assert_eq!(inplace.bits(), sme.bits(), "{label}: in-place A");
        if i >= m || j >= n {
            assert_eq!(sme.bits(), sentinel.bits(), "{label}: outside the block");
        }
    }
}

/// Runs [`compare`] over shapes and every mode for one type.
fn compare_all<T>(from_f64: fn(f64) -> T)
where
    T: SmeElement + GemmKernel + oxiblas_core::scalar::Field,
{
    // M0, M0 (beta = -0), M1, M2, M2, M3, M3 (alpha = 1), M3 (beta = 1),
    // and two M3 pairs whose beta is not a power of two, one with
    // alpha != 1 (the NEON epilogue multiplies by alpha, then FMLA
    // `acc + C * beta`) and one with alpha == 1 (no multiply, then FMLA):
    // `C * beta` is inexact there, so only these tell the single rounding
    // `fma(C, beta, t)` of the contract from `C * beta` rounded before the
    // sum. With a power-of-two beta both give the same bits.
    let pairs = [
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
    for m in [1, 7, 9, 64, 65, 130] {
        for n in [1, 6, 8, 9, 17, 33] {
            for kc in [1, 7, 65] {
                for (alpha, beta) in pairs {
                    compare(m, n, kc, from_f64(alpha), from_f64(beta), from_f64);
                }
            }
        }
    }
}

#[test]
fn test_sme2_contract_reference_equals_neon_f32() {
    if cfg!(feature = "force-scalar") {
        println!("skip: force-scalar dispatches the portable micro-kernel");
        return;
    }
    compare_all::<f32>(|v| v as f32);
}

#[test]
fn test_sme2_contract_reference_equals_neon_f64() {
    if cfg!(feature = "force-scalar") {
        println!("skip: force-scalar dispatches the portable micro-kernel");
        return;
    }
    compare_all::<f64>(|v| v);
}

#[test]
fn test_sme2_contract_reference_special_values() {
    // -0 products, infinities and NaN inputs: positions of NaN match NEON.
    if cfg!(feature = "force-scalar") {
        println!("skip: force-scalar dispatches the portable micro-kernel");
        return;
    }
    const SPECIALS: [f64; 7] = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0e-310,
        -3.0,
    ];
    /// Maps a generated value onto one of the specials.
    fn special(v: f64) -> f64 {
        SPECIALS[(v.abs() * 1.0e6) as usize % SPECIALS.len()]
    }
    fn special_f32(v: f64) -> f32 {
        special(v) as f32
    }
    for (m, n, kc) in [(3, 2, 2), (9, 9, 3), (65, 17, 2)] {
        for (alpha, beta) in [(1.0, 0.0), (1.0, 1.0), (-2.0, 0.0), (0.5, -3.0)] {
            compare(m, n, kc, alpha, beta, special);
            compare(m, n, kc, alpha as f32, beta as f32, special_f32);
        }
    }
}
