//! Tests of the SME2 kernels and packers.

use super::*;
use crate::level3::gemm::{gemm, gemm_transposed, gemm_with_par};
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::Trans;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

/// Deterministic values in [-1, 1).
fn val(i: usize, seed: usize) -> f64 {
    let x = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 1999;
    x as f64 / 999.5 - 1.0
}

/// Checks one micro-kernel call against a scalar reference.
fn check_kernel_f64(k: usize, alpha: f64, beta: f64) {
    let nr = NR_F64;
    let ldc = MR + 3;
    let a: Vec<f64> = (0..MR * k).map(|i| val(i, 1)).collect();
    let b: Vec<f64> = (0..nr * k).map(|i| val(i, 2)).collect();
    let c0: Vec<f64> = (0..ldc * nr)
        .map(|i| if beta == 0.0 { f64::NAN } else { val(i, 3) })
        .collect();
    let mut c = c0.clone();
    // SAFETY: SME2 is checked by the caller; the buffers have the sizes
    // the kernel reads and writes (64 x k, 8 x k, 8 columns of ldc).
    unsafe { micro_kernel_f64_sme2(k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), ldc) };
    for j in 0..nr {
        for i in 0..ldc {
            let got = c[i + j * ldc];
            if i >= MR {
                assert!(
                    got.to_bits() == c0[i + j * ldc].to_bits(),
                    "k={k}: row {i} below the tile written"
                );
                continue;
            }
            let dot: f64 = (0..k).map(|p| a[p * MR + i] * b[p * nr + j]).sum();
            let exp = if beta == 0.0 {
                alpha * dot
            } else {
                alpha * dot + beta * c0[i + j * ldc]
            };
            assert!(
                (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                "f64 k={k} ({i},{j}): {got} vs {exp}"
            );
        }
    }
}

fn check_kernel_f32(k: usize, alpha: f32, beta: f32) {
    let nr = NR_F32;
    let ldc = MR + 5;
    let a: Vec<f32> = (0..MR * k).map(|i| val(i, 4) as f32).collect();
    let b: Vec<f32> = (0..nr * k).map(|i| val(i, 5) as f32).collect();
    let c0: Vec<f32> = (0..ldc * nr)
        .map(|i| {
            if beta == 0.0 {
                f32::NAN
            } else {
                val(i, 6) as f32
            }
        })
        .collect();
    let mut c = c0.clone();
    // SAFETY: as in `check_kernel_f64` (16 columns).
    unsafe { micro_kernel_f32_sme2(k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), ldc) };
    for j in 0..nr {
        for i in 0..ldc {
            let got = c[i + j * ldc];
            if i >= MR {
                assert!(
                    got.to_bits() == c0[i + j * ldc].to_bits(),
                    "k={k}: row {i} below the tile written"
                );
                continue;
            }
            let dot: f64 = (0..k)
                .map(|p| f64::from(a[p * MR + i]) * f64::from(b[p * nr + j]))
                .sum();
            let c_old = f64::from(c0[i + j * ldc]);
            let exp = if beta == 0.0 {
                f64::from(alpha) * dot
            } else {
                f64::from(alpha) * dot + f64::from(beta) * c_old
            };
            let tol = 1e-5 * (1.0 + k as f64);
            assert!(
                (f64::from(got) - exp).abs() <= tol,
                "f32 k={k} ({i},{j}): {got} vs {exp}"
            );
        }
    }
}

/// Checks one macro-kernel call on an `m x n` block; padding of the last
/// panels holds NaN, which must not reach C, and C outside the block must
/// stay unchanged.
fn check_macro_f64(m: usize, n: usize, k: usize, alpha: f64, beta: f64, mode: u8) {
    let nr = NR_F64;
    let (mp, np) = (m.div_ceil(MR), n.div_ceil(nr));
    let ldc = m + 3;
    let mut a = vec![f64::NAN; mp * MR * k];
    for ib in 0..mp {
        for p in 0..k {
            for i in 0..MR.min(m - ib * MR) {
                a[ib * MR * k + p * MR + i] = val(ib * MR + i + p * m, 21);
            }
        }
    }
    // In place: column-major A with column stride lda > m, NaN past row m.
    let lda = m + 2;
    let mut a_cm = vec![f64::NAN; lda * k];
    for p in 0..k {
        for i in 0..m {
            a_cm[i + p * lda] = val(i + p * m, 21);
        }
    }
    for ib in 0..mp {
        for p in 0..k {
            for i in 0..MR.min(m - ib * MR) {
                a[ib * MR * k + p * MR + i] = a_cm[ib * MR + i + p * lda];
            }
        }
    }
    // 0: packed A; 1: column-major A in place; 2: in place with online packing.
    let mut a_pack = vec![f64::NAN; mp * MR * k];
    let (a_ptr, a_panel, a_step, a_pack_ptr) = match mode {
        0 => (a.as_ptr(), MR * k, MR, core::ptr::null_mut()),
        1 => (a_cm.as_ptr(), MR, lda, core::ptr::null_mut()),
        _ => (a_cm.as_ptr(), MR, lda, a_pack.as_mut_ptr()),
    };
    let mut b = vec![f64::NAN; np * nr * k];
    for jb in 0..np {
        for p in 0..k {
            for j in 0..nr.min(n - jb * nr) {
                b[jb * nr * k + p * nr + j] = val(jb * nr + j + p * n, 22);
            }
        }
    }
    let c0: Vec<f64> = (0..ldc * (n + 1)).map(|i| val(i, 23)).collect();
    let mut c = c0.clone();
    // SAFETY: SME2 is checked by the caller; the panels and C have the
    // sizes the macro-kernel reads and writes.
    unsafe {
        macro_kernel_f64_sme2(
            m,
            n,
            k,
            alpha,
            a_ptr,
            a_panel,
            a_step,
            a_pack_ptr,
            b.as_ptr(),
            beta,
            c.as_mut_ptr(),
            ldc,
        )
    };
    for j in 0..=n {
        for i in 0..ldc {
            let got = c[i + j * ldc];
            if i >= m || j >= n {
                assert!(
                    got.to_bits() == c0[i + j * ldc].to_bits(),
                    "{m}x{n}: C({i},{j}) outside the block changed"
                );
                continue;
            }
            let (ib, ii, jb, jj) = (i / MR, i % MR, j / nr, j % nr);
            let dot: f64 = (0..k)
                .map(|p| a[ib * MR * k + p * MR + ii] * b[jb * nr * k + p * nr + jj])
                .sum();
            let exp = alpha * dot + beta * c0[i + j * ldc];
            assert!(
                (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                "{m}x{n}x{k} ({i},{j}): {got} vs {exp}"
            );
        }
    }
}

fn check_macro_f32(m: usize, n: usize, k: usize, alpha: f32, beta: f32, mode: u8) {
    let nr = NR_F32;
    let (mp, np) = (m.div_ceil(MR), n.div_ceil(nr));
    let ldc = m + 5;
    let mut a = vec![f32::NAN; mp * MR * k];
    for ib in 0..mp {
        for p in 0..k {
            for i in 0..MR.min(m - ib * MR) {
                a[ib * MR * k + p * MR + i] = val(ib * MR + i + p * m, 24) as f32;
            }
        }
    }
    // In place: column-major A with column stride lda > m, NaN past row m.
    let lda = m + 2;
    let mut a_cm = vec![f32::NAN; lda * k];
    for p in 0..k {
        for i in 0..m {
            a_cm[i + p * lda] = val(i + p * m, 24) as f32;
        }
    }
    for ib in 0..mp {
        for p in 0..k {
            for i in 0..MR.min(m - ib * MR) {
                a[ib * MR * k + p * MR + i] = a_cm[ib * MR + i + p * lda];
            }
        }
    }
    // 0: packed A; 1: column-major A in place; 2: in place with online packing.
    let mut a_pack = vec![f32::NAN; mp * MR * k];
    let (a_ptr, a_panel, a_step, a_pack_ptr) = match mode {
        0 => (a.as_ptr(), MR * k, MR, core::ptr::null_mut()),
        1 => (a_cm.as_ptr(), MR, lda, core::ptr::null_mut()),
        _ => (a_cm.as_ptr(), MR, lda, a_pack.as_mut_ptr()),
    };
    let mut b = vec![f32::NAN; np * nr * k];
    for jb in 0..np {
        for p in 0..k {
            for j in 0..nr.min(n - jb * nr) {
                b[jb * nr * k + p * nr + j] = val(jb * nr + j + p * n, 25) as f32;
            }
        }
    }
    let c0: Vec<f32> = (0..ldc * (n + 1)).map(|i| val(i, 26) as f32).collect();
    let mut c = c0.clone();
    // SAFETY: as in `check_macro_f64`.
    unsafe {
        macro_kernel_f32_sme2(
            m,
            n,
            k,
            alpha,
            a_ptr,
            a_panel,
            a_step,
            a_pack_ptr,
            b.as_ptr(),
            beta,
            c.as_mut_ptr(),
            ldc,
        )
    };
    for j in 0..=n {
        for i in 0..ldc {
            let got = c[i + j * ldc];
            if i >= m || j >= n {
                assert!(
                    got.to_bits() == c0[i + j * ldc].to_bits(),
                    "{m}x{n}: C({i},{j}) outside the block changed"
                );
                continue;
            }
            let (ib, ii, jb, jj) = (i / MR, i % MR, j / nr, j % nr);
            let dot: f64 = (0..k)
                .map(|p| {
                    f64::from(a[ib * MR * k + p * MR + ii])
                        * f64::from(b[jb * nr * k + p * nr + jj])
                })
                .sum();
            let exp = f64::from(alpha) * dot + f64::from(beta) * f64::from(c0[i + j * ldc]);
            let tol = 1e-5 * (1.0 + k as f64);
            assert!(
                (f64::from(got) - exp).abs() <= tol,
                "{m}x{n}x{k} ({i},{j}): {got} vs {exp}"
            );
        }
    }
}

const BLOCKS: [(usize, usize, usize); 8] = [
    (1, 1, 1),
    (63, 7, 5),
    (64, 8, 4),
    (64, 16, 9),
    (65, 17, 3),
    (130, 9, 33),
    (200, 47, 70),
    (37, 100, 0),
];

#[test]
fn test_sme2_macro_kernel_f64() {
    if !usable_f64() {
        return;
    }
    for (m, n, k) in BLOCKS {
        for mode in 0..3 {
            check_macro_f64(m, n, k, 1.25, 0.0, mode);
            check_macro_f64(m, n, k, -0.5, 0.75, mode);
            check_macro_f64(m, n, k, 1.0, 0.0, mode);
            check_macro_f64(m, n, k, 1.0, 1.0, mode);
        }
    }
}

#[test]
fn test_sme2_macro_kernel_f32() {
    if !usable_f32() {
        return;
    }
    for (m, n, k) in BLOCKS {
        for mode in 0..3 {
            check_macro_f32(m, n, k, 1.25, 0.0, mode);
            check_macro_f32(m, n, k, -0.5, 0.75, mode);
            check_macro_f32(m, n, k, 1.0, 0.0, mode);
            check_macro_f32(m, n, k, 1.0, 1.0, mode);
        }
    }
}

#[test]
fn test_sme2_pack_b() {
    for &(k, n) in &[
        (1, 1),
        (63, 15),
        (64, 16),
        (65, 17),
        (130, 40),
        (200, 8),
        (256, 33),
    ] {
        let ldb = k + 3;
        let src: Vec<f64> = (0..ldb * n).map(|i| val(i, 31)).collect();
        if usable_f64() {
            let len = n.div_ceil(NR_F64) * NR_F64 * k;
            let mut dst = vec![f64::NAN; len];
            // SAFETY: SME2 checked; src holds n columns of ldb >= k, dst holds len.
            unsafe { pack_b_f64_sme2(src.as_ptr(), ldb, k, n, dst.as_mut_ptr()) };
            for (idx, &got) in dst.iter().enumerate() {
                let (panel, rest) = (idx / (NR_F64 * k), idx % (NR_F64 * k));
                let (p, jj) = (rest / NR_F64, rest % NR_F64);
                let j = panel * NR_F64 + jj;
                let exp = if j < n { src[p + j * ldb] } else { 0.0 };
                assert!(got.to_bits() == exp.to_bits(), "f64 k={k} n={n}: ({p},{j})");
            }
        }
        if usable_f32() {
            let src32: Vec<f32> = src.iter().map(|&x| x as f32).collect();
            let len = n.div_ceil(NR_F32) * NR_F32 * k;
            let mut dst = vec![f32::NAN; len];
            // SAFETY: as above.
            unsafe { pack_b_f32_sme2(src32.as_ptr(), ldb, k, n, dst.as_mut_ptr()) };
            for (idx, &got) in dst.iter().enumerate() {
                let (panel, rest) = (idx / (NR_F32 * k), idx % (NR_F32 * k));
                let (p, jj) = (rest / NR_F32, rest % NR_F32);
                let j = panel * NR_F32 + jj;
                let exp = if j < n { src32[p + j * ldb] } else { 0.0 };
                assert!(got.to_bits() == exp.to_bits(), "f32 k={k} n={n}: ({p},{j})");
            }
        }
    }
}

const DEPTHS: [usize; 10] = [0, 1, 2, 3, 4, 5, 7, 8, 33, 257];

#[test]
fn test_sme2_kernel_f64() {
    if !usable_f64() {
        return;
    }
    for k in DEPTHS {
        for (alpha, beta) in [(1.0, 0.0), (2.5, 0.0), (1.0, 1.0), (-0.75, -0.5)] {
            check_kernel_f64(k, alpha, beta);
        }
    }
}

#[test]
fn test_sme2_kernel_f32() {
    if !usable_f32() {
        return;
    }
    for k in DEPTHS {
        for (alpha, beta) in [(1.0, 0.0), (2.5, 0.0), (1.0, 1.0), (-0.75, -0.5)] {
            check_kernel_f32(k, alpha, beta);
        }
    }
}

#[test]
fn test_sme2_shape_is_selected() {
    if usable_f32() {
        let s = f32::micro_kernel_shape();
        assert_eq!((s.mr, s.nr), (MR, NR_F32));
    }
    if usable_f64() {
        let s = f64::micro_kernel_shape();
        assert_eq!((s.mr, s.nr), (MR, NR_F64));
    }
}

fn mat(rows: usize, cols: usize, seed: usize) -> Mat<f64> {
    let mut m = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            m[(i, j)] = val(i + j * rows, seed);
        }
    }
    m
}

/// op(A) op(B) through the public GEMM, against a naive product.
fn check_gemm_f64(m: usize, n: usize, k: usize, ta: Trans, tb: Trans, par: Par) {
    let a = if ta == Trans::NoTrans {
        mat(m, k, 7)
    } else {
        mat(k, m, 7)
    };
    let b = if tb == Trans::NoTrans {
        mat(k, n, 8)
    } else {
        mat(n, k, 8)
    };
    let c0 = mat(m, n, 9);
    let mut c = c0.clone();
    let (alpha, beta) = (1.5, -0.25);
    if ta == Trans::NoTrans && tb == Trans::NoTrans {
        gemm_with_par(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), par);
    } else {
        gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
    }
    let at = |i: usize, p: usize| {
        if ta == Trans::NoTrans {
            a[(i, p)]
        } else {
            a[(p, i)]
        }
    };
    let bt = |p: usize, j: usize| {
        if tb == Trans::NoTrans {
            b[(p, j)]
        } else {
            b[(j, p)]
        }
    };
    for j in 0..n {
        for i in 0..m {
            let dot: f64 = (0..k).map(|p| at(i, p) * bt(p, j)).sum();
            let exp = alpha * dot + beta * c0[(i, j)];
            let got = c[(i, j)];
            assert!(
                (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                "{m}x{n}x{k} {ta:?} {tb:?} ({i},{j}): {got} vs {exp}"
            );
        }
    }
}

#[test]
fn test_sme2_gemm_edges() {
    // Rows and columns around the 64 x 8 tile, depths around the K unroll.
    for &(m, n, k) in &[
        (63, 7, 5),
        (64, 8, 64),
        (65, 9, 67),
        (130, 17, 33),
        (200, 41, 300),
        (1, 40, 50),
        (70, 1, 50),
        // Below SMALL_THRESHOLD, routed to SME2 by `sme2_small`.
        (32, 8, 8),
        (33, 9, 11),
        (16, 16, 64),
        (31, 31, 31),
    ] {
        check_gemm_f64(m, n, k, Trans::NoTrans, Trans::NoTrans, Par::Seq);
    }
    check_gemm_f64(97, 23, 71, Trans::Trans, Trans::NoTrans, Par::Seq);
    check_gemm_f64(97, 23, 71, Trans::NoTrans, Trans::Trans, Par::Seq);
    check_gemm_f64(97, 23, 71, Trans::Trans, Trans::Trans, Par::Seq);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_parallel_concurrent_callers() {
    // Several threads at once: one holds the workers, the others run in place.
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| check_gemm_f64(260, 210, 90, Trans::NoTrans, Trans::NoTrans, Par::Rayon));
        }
    });
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_parallel() {
    // Above the SME2 parallel minimum (2^22 multiply-adds).
    check_gemm_f64(300, 230, 70, Trans::NoTrans, Trans::NoTrans, Par::Rayon);
}

#[test]
fn test_sme2_gemm_f32_matches_f64() {
    for (m, n, k) in [(131, 45, 97), (1, 300, 77), (300, 1, 77), (1, 1, 5)] {
        gemm_f32_matches_f64(m, n, k);
    }
}

fn gemm_f32_matches_f64(m: usize, n: usize, k: usize) {
    let a64 = mat(m, k, 11);
    let b64 = mat(k, n, 12);
    let mut a: Mat<f32> = Mat::zeros(m, k);
    let mut b: Mat<f32> = Mat::zeros(k, n);
    for j in 0..k {
        for i in 0..m {
            a[(i, j)] = a64[(i, j)] as f32;
        }
    }
    for j in 0..n {
        for i in 0..k {
            b[(i, j)] = b64[(i, j)] as f32;
        }
    }
    let mut c: Mat<f32> = Mat::zeros(m, n);
    gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
    for j in 0..n {
        for i in 0..m {
            let exp: f64 = (0..k)
                .map(|p| f64::from(a[(i, p)]) * f64::from(b[(p, j)]))
                .sum();
            assert!((f64::from(c[(i, j)]) - exp).abs() <= 1e-4, "({i},{j})");
        }
    }
}
