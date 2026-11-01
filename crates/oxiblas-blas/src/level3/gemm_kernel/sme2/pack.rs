// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SME2 packers: B and transposed A panels through ZA, B^T rows by copy.

use super::*;

/// Packs `k` rows of `n` columns of a column-major B (column stride `ldb`)
/// into B panels of `NR` columns, `NR` elements per row (the layout the
/// kernels read), zero-padding the last panel to `NR` columns.
///
/// Full 64-row chunks of full panels are transposed through the ZA array:
/// four columns are loaded with four-vector loads and moved into four ZA
/// tiles with one four-slice `MOVA` per tile, and the vertical slices (the
/// packed rows) come out four at a time. This is the four-row ZA transposition
/// of MTGEMM-A's packing of A, applied to B, which is the transposed operand in
/// column-major order. The rest is copied element by element.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `n` columns
///   `ldb` apart, and `dst` for writes of `ceil(n / 16) * 16 * k` elements.
pub(crate) unsafe fn pack_b_f32_sme2(
    src: *const f32,
    ldb: usize,
    k: usize,
    n: usize,
    dst: *mut f32,
) {
    let nr = NR_F32;
    let chunks = k / 64;
    for j0 in (0..n).step_by(nr) {
        let jb = nr.min(n - j0);
        // SAFETY: panel j0 / nr starts inside `dst`; columns j0.. inside `src`.
        let (panel, col) = unsafe { (dst.add(j0 * k), src.add(j0 * ldb)) };
        let mut p0 = 0;
        if jb == nr && chunks > 0 {
            // SAFETY: the caller guarantees SME2; the block reads 64 * chunks
            // rows of 16 columns and writes the same number of packed rows.
            unsafe {
                // Prefetch the next panel only while one follows (zero: none).
                let next = if j0 + 2 * nr <= n { nr * ldb * 4 } else { 0 };
                transpose_chunks_f32(col, ldb * 4, panel, chunks, next)
            };
            p0 = chunks * 64;
        }
        if jb == nr && p0 < k {
            // SAFETY: rows p0..k (fewer than 64) of the 16 columns, and their
            // packed rows inside panel j0 / nr.
            unsafe { transpose_tail_f32(col.add(p0), ldb * 4, panel.add(p0 * nr), k - p0) };
            p0 = k;
        }
        for p in p0..k {
            for jj in 0..nr {
                // SAFETY: p < k and j0 + jj < n for the reads; the write is
                // inside panel j0 / nr.
                unsafe {
                    *panel.add(p * nr + jj) = if jj < jb { *col.add(p + jj * ldb) } else { 0.0 };
                }
            }
        }
    }
}

/// Transposes `chunks` chunks of 64 rows x 16 columns through ZA.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for 64 * `chunks` reads in each of
/// 16 columns `ldb_bytes` apart, and `panel` for 1024 * `chunks` writes.
unsafe fn transpose_chunks_f32(
    col: *const f32,
    ldb_bytes: usize,
    panel: *mut f32,
    chunks: usize,
    prefetch_offset: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue pn8.s",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            // Four columns per step into slices w12..w12+3 of the four tiles.
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            // Core-side L2 prefetch of the same rows of the next panel's columns.
            "cbz x13, 41f",
            "add x9, x4, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x5, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x6, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x7, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "41:",
            "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn8/z, [x4]",
            "ld1w {{z17.s, z21.s, z25.s, z29.s}}, pn8/z, [x5]",
            "ld1w {{z18.s, z22.s, z26.s, z30.s}}, pn8/z, [x6]",
            "ld1w {{z19.s, z23.s, z27.s, z31.s}}, pn8/z, [x7]",
            "mova za0h.s[w12, 0:3], {{z16.s-z19.s}}",
            "mova za1h.s[w12, 0:3], {{z20.s-z23.s}}",
            "mova za2h.s[w12, 0:3], {{z24.s-z27.s}}",
            "mova za3h.s[w12, 0:3], {{z28.s-z31.s}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            // Vertical slice p of tile t is packed row 16t + p.
            "mov w12, #0",
            "mov x8, x2",
            "add x9, x2, #1024",
            "add x10, x2, #2048",
            "add x11, x2, #3072",
            "4:",
            "mova {{z0.s-z3.s}}, za0v.s[w12, 0:3]",
            "mova {{z4.s-z7.s}}, za1v.s[w12, 0:3]",
            "mova {{z8.s-z11.s}}, za2v.s[w12, 0:3]",
            "mova {{z12.s-z15.s}}, za3v.s[w12, 0:3]",
            "st1w {{z0.s-z3.s}}, pn8, [x8]",
            "st1w {{z4.s-z7.s}}, pn8, [x9]",
            "st1w {{z8.s-z11.s}}, pn8, [x10]",
            "st1w {{z12.s-z15.s}}, pn8, [x11]",
            "addvl x8, x8, #4",
            "addvl x9, x9, #4",
            "addvl x10, x10, #4",
            "addvl x11, x11, #4",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 4b",
            "add x0, x0, #256",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x13") prefetch_offset,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x11") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p8") _,
            options(nostack)
        );
    }
}

/// Packs B for the f64 kernels; as [`pack_b_f32_sme2`] with `NR` = 8.
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `n` columns
///   `ldb` apart, and `dst` for writes of `ceil(n / 8) * 8 * k` elements.
pub(crate) unsafe fn pack_b_f64_sme2(
    src: *const f64,
    ldb: usize,
    k: usize,
    n: usize,
    dst: *mut f64,
) {
    let nr = NR_F64;
    let chunks = k / 64;
    for j0 in (0..n).step_by(nr) {
        let jb = nr.min(n - j0);
        // SAFETY: panel j0 / nr starts inside `dst`; columns j0.. inside `src`.
        let (panel, col) = unsafe { (dst.add(j0 * k), src.add(j0 * ldb)) };
        let mut p0 = 0;
        if jb == nr && chunks > 0 {
            // SAFETY: the caller guarantees SME2; the block reads 64 * chunks
            // rows of 8 columns and writes the same number of packed rows.
            unsafe {
                // Prefetch the next panel only while one follows (zero: none).
                let next = if j0 + 2 * nr <= n { nr * ldb * 8 } else { 0 };
                transpose_chunks_f64(col, ldb * 8, panel, chunks, next)
            };
            p0 = chunks * 64;
        }
        if jb == nr && p0 < k {
            // SAFETY: rows p0..k (fewer than 64) of the 8 columns, and their
            // packed rows inside panel j0 / nr.
            unsafe { transpose_tail_f64(col.add(p0), ldb * 8, panel.add(p0 * nr), k - p0) };
            p0 = k;
        }
        for p in p0..k {
            for jj in 0..nr {
                // SAFETY: as in `pack_b_f32_sme2`.
                unsafe {
                    *panel.add(p * nr + jj) = if jj < jb { *col.add(p + jj * ldb) } else { 0.0 };
                }
            }
        }
    }
}

/// Transposes `chunks` chunks of 64 rows x 8 columns through ZA (8 tiles of 8 x 8).
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for 64 * `chunks` reads in each of
/// 8 columns `ldb_bytes` apart, and `panel` for 512 * `chunks` writes.
unsafe fn transpose_chunks_f64(
    col: *const f64,
    ldb_bytes: usize,
    panel: *mut f64,
    chunks: usize,
    prefetch_offset: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue pn8.d",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            // Four columns per step: rows 0-31 into tiles 0-3, rows 32-63 into tiles 4-7.
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            // Core-side L2 prefetch of the same rows of the next panel's columns.
            "cbz x13, 41f",
            "add x9, x4, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x5, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x6, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x7, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "41:",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn8/z, [x4]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn8/z, [x5]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn8/z, [x6]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn8/z, [x7]",
            "mova za0h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za1h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za2h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za3h.d[w12, 0:3], {{z28.d-z31.d}}",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn8/z, [x4, #4, mul vl]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn8/z, [x5, #4, mul vl]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn8/z, [x6, #4, mul vl]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn8/z, [x7, #4, mul vl]",
            "mova za4h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za5h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za6h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za7h.d[w12, 0:3], {{z28.d-z31.d}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            // Vertical slice p of tile t is packed row 8t + p.
            "mov w12, #0",
            "mov x8, x2",
            "add x9, x2, #2048",
            "4:",
            "mova {{z0.d-z3.d}}, za0v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za1v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za2v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za3v.d[w12, 0:3]",
            "st1d {{z0.d-z3.d}}, pn8, [x8]",
            "st1d {{z4.d-z7.d}}, pn8, [x8, #8, mul vl]",
            "st1d {{z8.d-z11.d}}, pn8, [x8, #16, mul vl]",
            "st1d {{z12.d-z15.d}}, pn8, [x8, #24, mul vl]",
            "mova {{z0.d-z3.d}}, za4v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za5v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za6v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za7v.d[w12, 0:3]",
            "st1d {{z0.d-z3.d}}, pn8, [x9]",
            "st1d {{z4.d-z7.d}}, pn8, [x9, #8, mul vl]",
            "st1d {{z8.d-z11.d}}, pn8, [x9, #16, mul vl]",
            "st1d {{z12.d-z15.d}}, pn8, [x9, #24, mul vl]",
            "addvl x8, x8, #4",
            "addvl x9, x9, #4",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 4b",
            "add x0, x0, #512",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x13") prefetch_offset,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p8") _,
            options(nostack)
        );
    }
}

/// Transposes the last `rows` (< 64) rows of one full B panel through ZA, as
/// in `transpose_chunks_f32`, with the loads predicated to the rows.
///
/// # Safety
///
/// SME2 must be present, `rows` in 1..64; `col` must be valid for `rows`
/// reads in each of 16 columns `ldb_bytes` apart, and `panel` for `rows * 16`
/// writes.
unsafe fn transpose_tail_f32(col: *const f32, ldb_bytes: usize, panel: *mut f32, rows: usize) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "zero {{za}}",
            "ptrue p0.s",
            "mov x9, #0",
            "whilelt pn9.s, x9, x3, vlx4",
            "mov x4, x0",
            "mov w12, #0",
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn9/z, [x4]",
            "ld1w {{z17.s, z21.s, z25.s, z29.s}}, pn9/z, [x5]",
            "ld1w {{z18.s, z22.s, z26.s, z30.s}}, pn9/z, [x6]",
            "ld1w {{z19.s, z23.s, z27.s, z31.s}}, pn9/z, [x7]",
            "mova za0h.s[w12, 0:3], {{z16.s-z19.s}}",
            "mova za1h.s[w12, 0:3], {{z20.s-z23.s}}",
            "mova za2h.s[w12, 0:3], {{z24.s-z27.s}}",
            "mova za3h.s[w12, 0:3], {{z28.s-z31.s}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            "mov x8, x2",
            "mov x10, x3",
            "cbz x10, 9f",
            "mov w12, #0",
            "20:",
            "st1w {{za0v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 20b",
            "cbz x10, 9f",
            "mov w12, #0",
            "21:",
            "st1w {{za1v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 21b",
            "cbz x10, 9f",
            "mov w12, #0",
            "22:",
            "st1w {{za2v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 22b",
            "cbz x10, 9f",
            "mov w12, #0",
            "23:",
            "st1w {{za3v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 23b",
            "9:",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            in("x3") rows,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p9") _, out("p10") _,
            options(nostack)
        );
    }
}

/// Transposes the last `rows` (< 64) rows of one full B panel through ZA, as
/// in `transpose_chunks_f64`, with the loads predicated to the rows.
///
/// # Safety
///
/// SME2 must be present, `rows` in 1..64; `col` must be valid for `rows`
/// reads in each of 8 columns `ldb_bytes` apart, and `panel` for `rows * 8`
/// writes.
unsafe fn transpose_tail_f64(col: *const f64, ldb_bytes: usize, panel: *mut f64, rows: usize) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "zero {{za}}",
            "ptrue p0.d",
            "mov x9, #0",
            "whilelt pn9.d, x9, x3, vlx4",
            "mov x9, #32",
            "whilelt pn10.d, x9, x3, vlx4",
            "mov x4, x0",
            "mov w12, #0",
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn9/z, [x4]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn9/z, [x5]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn9/z, [x6]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn9/z, [x7]",
            "mova za0h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za1h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za2h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za3h.d[w12, 0:3], {{z28.d-z31.d}}",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn10/z, [x4, #4, mul vl]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn10/z, [x5, #4, mul vl]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn10/z, [x6, #4, mul vl]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn10/z, [x7, #4, mul vl]",
            "mova za4h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za5h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za6h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za7h.d[w12, 0:3], {{z28.d-z31.d}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            "mov x8, x2",
            "mov x10, x3",
            "cbz x10, 9f",
            "mov w12, #0",
            "20:",
            "st1d {{za0v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 20b",
            "cbz x10, 9f",
            "mov w12, #0",
            "21:",
            "st1d {{za1v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 21b",
            "cbz x10, 9f",
            "mov w12, #0",
            "22:",
            "st1d {{za2v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 22b",
            "cbz x10, 9f",
            "mov w12, #0",
            "23:",
            "st1d {{za3v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 23b",
            "cbz x10, 9f",
            "mov w12, #0",
            "24:",
            "st1d {{za4v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 24b",
            "cbz x10, 9f",
            "mov w12, #0",
            "25:",
            "st1d {{za5v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 25b",
            "cbz x10, 9f",
            "mov w12, #0",
            "26:",
            "st1d {{za6v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 26b",
            "cbz x10, 9f",
            "mov w12, #0",
            "27:",
            "st1d {{za7v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 27b",
            "9:",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            in("x3") rows,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p9") _, out("p10") _,
            options(nostack)
        );
    }
}

/// Copies packed rows of 64 bytes (16 f32 or 8 f64, one B panel row) from a
/// row-ordered source: for each of `panels` panels and each row `p` of the
/// first `rows` rows (a multiple of 16), the 64 bytes at
/// `src + p * ld_bytes + panel * 64` go to `dst + panel * dst_panel_bytes + p * 64`.
/// Sixteen rows of all panels at a time, so the reads stay within 16 source rows.
///
/// # Safety
///
/// SME2 must be present; `src` and `dst` must be valid for those reads and writes.
pub(crate) unsafe fn copy_rows_64b(
    src: *const u8,
    ld_bytes: usize,
    rows: usize,
    panels: usize,
    dst: *mut u8,
    dst_panel_bytes: usize,
) {
    if rows < 16 || panels == 0 {
        return;
    }
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "mov x9, #0",
            "2:",
            "mul x11, x9, x1",
            "add x11, x0, x11",
            "add x12, x2, x9, lsl #6",
            "mov x10, #0",
            "3:",
            "mov x14, x11",
            "mov x15, x12",
            "mov x16, #4",
            "4:",
            "ld1w {{z0.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z1.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z2.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z3.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "st1w {{z0.s-z3.s}}, pn8, [x15]",
            "addvl x15, x15, #4",
            "subs x16, x16, #1",
            "b.ne 4b",
            "add x11, x11, #64",
            "add x12, x12, x5",
            "add x10, x10, #1",
            "cmp x10, x4",
            "b.ne 3b",
            "add x9, x9, #16",
            "cmp x9, x3",
            "b.lo 2b",
            "smstop",
            in("x0") src,
            in("x1") ld_bytes,
            in("x2") dst,
            in("x3") rows,
            in("x4") panels,
            in("x5") dst_panel_bytes,
            out("x9") _, out("x10") _, out("x11") _, out("x12") _,
            out("x14") _, out("x15") _, out("x16") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _,
            options(nostack)
        );
    }
}

/// Packs `chunks * 16` depth steps of one 64-row A panel from a column-major
/// `A^T` (`op(A) = A^T`): `col` points at `A^T[p0, i0]`, so row `i0 + r` of
/// op(A) is column `i0 + r` of `A^T`, `ld_bytes` apart. Each chunk loads 16
/// columns per ZA tile into its horizontal slices and stores the vertical
/// slices (depth steps) as packed panel rows of 64 elements.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for `chunks * 16` reads in each
/// of 64 columns `ld_bytes` apart, and `panel` for `chunks * 16 * 64` writes.
pub(crate) unsafe fn pack_a_t_chunks_f32(
    col: *const f32,
    ld_bytes: usize,
    panel: *mut f32,
    chunks: usize,
    cols: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "2:",
            "zero {{za}}",
            "mov x4, x0",
            "mov x6, x5",
            "mov w12, #0",
            "cbz x6, 39f",
            "20:",
            "ld1w {{za0h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #16",
            "b.ne 20b",
            "mov w12, #0",
            "cbz x6, 39f",
            "21:",
            "ld1w {{za1h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #16",
            "b.ne 21b",
            "mov w12, #0",
            "cbz x6, 39f",
            "12:",
            "ld1w {{za2h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #16",
            "b.ne 12b",
            "mov w12, #0",
            "cbz x6, 39f",
            "13:",
            "ld1w {{za3h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #16",
            "b.ne 13b",
            "39:",
            "mov w12, #0",
            "mov x8, x2",
            "3:",
            "mova {{z0.s-z3.s}}, za0v.s[w12, 0:3]",
            "mova {{z4.s-z7.s}}, za1v.s[w12, 0:3]",
            "mova {{z8.s-z11.s}}, za2v.s[w12, 0:3]",
            "mova {{z12.s-z15.s}}, za3v.s[w12, 0:3]",
            "st1w {{z0.s, z4.s, z8.s, z12.s}}, pn8, [x8, #0, mul vl]",
            "st1w {{z1.s, z5.s, z9.s, z13.s}}, pn8, [x8, #4, mul vl]",
            "st1w {{z2.s, z6.s, z10.s, z14.s}}, pn8, [x8, #8, mul vl]",
            "st1w {{z3.s, z7.s, z11.s, z15.s}}, pn8, [x8, #12, mul vl]",
            "addvl x8, x8, #16",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            "add x0, x0, #64",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ld_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x5") cols,
            out("x4") _, out("x6") _, out("x8") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _,
            options(nostack)
        );
    }
}

/// Packs `chunks * 8` depth steps of one 64-row A panel from a column-major
/// `A^T` (`op(A) = A^T`): `col` points at `A^T[p0, i0]`, so row `i0 + r` of
/// op(A) is column `i0 + r` of `A^T`, `ld_bytes` apart. Each chunk loads 8
/// columns per ZA tile into its horizontal slices and stores the vertical
/// slices (depth steps) as packed panel rows of 64 elements.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for `chunks * 8` reads in each
/// of 64 columns `ld_bytes` apart, and `panel` for `chunks * 8 * 64` writes.
pub(crate) unsafe fn pack_a_t_chunks_f64(
    col: *const f64,
    ld_bytes: usize,
    panel: *mut f64,
    chunks: usize,
    cols: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.d",
            "ptrue pn8.d",
            "2:",
            "zero {{za}}",
            "mov x4, x0",
            "mov x6, x5",
            "mov w12, #0",
            "cbz x6, 39f",
            "20:",
            "ld1d {{za0h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 20b",
            "mov w12, #0",
            "cbz x6, 39f",
            "21:",
            "ld1d {{za1h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 21b",
            "mov w12, #0",
            "cbz x6, 39f",
            "12:",
            "ld1d {{za2h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 12b",
            "mov w12, #0",
            "cbz x6, 39f",
            "13:",
            "ld1d {{za3h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 13b",
            "mov w12, #0",
            "cbz x6, 39f",
            "14:",
            "ld1d {{za4h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 14b",
            "mov w12, #0",
            "cbz x6, 39f",
            "15:",
            "ld1d {{za5h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 15b",
            "mov w12, #0",
            "cbz x6, 39f",
            "16:",
            "ld1d {{za6h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 16b",
            "mov w12, #0",
            "cbz x6, 39f",
            "17:",
            "ld1d {{za7h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "sub x6, x6, #1",
            "cbz x6, 39f",
            "cmp w12, #8",
            "b.ne 17b",
            "39:",
            "mov w12, #0",
            "mov x8, x2",
            "3:",
            "mova {{z0.d-z3.d}}, za0v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za1v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za2v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za3v.d[w12, 0:3]",
            "mova {{z16.d-z19.d}}, za4v.d[w12, 0:3]",
            "mova {{z20.d-z23.d}}, za5v.d[w12, 0:3]",
            "mova {{z24.d-z27.d}}, za6v.d[w12, 0:3]",
            "mova {{z28.d-z31.d}}, za7v.d[w12, 0:3]",
            "st1d {{z0.d, z4.d, z8.d, z12.d}}, pn8, [x8, #0, mul vl]",
            "st1d {{z16.d, z20.d, z24.d, z28.d}}, pn8, [x8, #4, mul vl]",
            "st1d {{z1.d, z5.d, z9.d, z13.d}}, pn8, [x8, #8, mul vl]",
            "st1d {{z17.d, z21.d, z25.d, z29.d}}, pn8, [x8, #12, mul vl]",
            "st1d {{z2.d, z6.d, z10.d, z14.d}}, pn8, [x8, #16, mul vl]",
            "st1d {{z18.d, z22.d, z26.d, z30.d}}, pn8, [x8, #20, mul vl]",
            "st1d {{z3.d, z7.d, z11.d, z15.d}}, pn8, [x8, #24, mul vl]",
            "st1d {{z19.d, z23.d, z27.d, z31.d}}, pn8, [x8, #28, mul vl]",
            "addvl x8, x8, #16",
            "addvl x8, x8, #16",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            "add x0, x0, #64",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ld_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x5") cols,
            out("x4") _, out("x6") _, out("x8") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _,
            options(nostack)
        );
    }
}

/// Packs rows `0..m` of `op(A) = A^T`, depth `k`, from a column-major `A^T`
/// (`src` = `A^T[0, 0]`, column stride `ld`) into 64-row panels of `64 * k`
/// elements (the layout of the packed A panels), zero-padding past `m`.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `m` columns `ld`
///   apart, and `dst` for writes of `ceil(m / 64) * 64 * k` elements.
pub(crate) unsafe fn pack_a_t_f32_sme2(
    src: *const f32,
    ld: usize,
    k: usize,
    m: usize,
    dst: *mut f32,
) {
    let chunks = k / 16;
    for i0 in (0..m).step_by(MR) {
        let ib = MR.min(m - i0);
        // SAFETY: panel i0 / 64 starts inside `dst`, column i0 inside `src`.
        let (panel, col) = unsafe { (dst.add(i0 * k), src.add(i0 * ld)) };
        let mut p0 = 0;
        if chunks > 0 {
            // SAFETY: the caller guarantees SME2; `ib` columns of `chunks * 16`
            // rows (rows of the panel past `ib` are written as zeros).
            unsafe {
                pack_a_t_chunks_f32(col, ld * core::mem::size_of::<f32>(), panel, chunks, ib)
            };
            p0 = chunks * 16;
        }
        for p in p0..k {
            for r in 0..MR {
                // SAFETY: p < k and r < ib for the reads; the write is inside
                // panel i0 / 64.
                unsafe { *panel.add(p * MR + r) = if r < ib { *col.add(p + r * ld) } else { 0.0 } };
            }
        }
    }
}

/// Packs rows `0..m` of `op(A) = A^T`, depth `k`, from a column-major `A^T`
/// (`src` = `A^T[0, 0]`, column stride `ld`) into 64-row panels of `64 * k`
/// elements (the layout of the packed A panels), zero-padding past `m`.
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `m` columns `ld`
///   apart, and `dst` for writes of `ceil(m / 64) * 64 * k` elements.
pub(crate) unsafe fn pack_a_t_f64_sme2(
    src: *const f64,
    ld: usize,
    k: usize,
    m: usize,
    dst: *mut f64,
) {
    let chunks = k / 8;
    for i0 in (0..m).step_by(MR) {
        let ib = MR.min(m - i0);
        // SAFETY: panel i0 / 64 starts inside `dst`, column i0 inside `src`.
        let (panel, col) = unsafe { (dst.add(i0 * k), src.add(i0 * ld)) };
        let mut p0 = 0;
        if chunks > 0 {
            // SAFETY: the caller guarantees SME2; `ib` columns of `chunks * 8`
            // rows (rows of the panel past `ib` are written as zeros).
            unsafe {
                pack_a_t_chunks_f64(col, ld * core::mem::size_of::<f64>(), panel, chunks, ib)
            };
            p0 = chunks * 8;
        }
        for p in p0..k {
            for r in 0..MR {
                // SAFETY: p < k and r < ib for the reads; the write is inside
                // panel i0 / 64.
                unsafe { *panel.add(p * MR + r) = if r < ib { *col.add(p + r * ld) } else { 0.0 } };
            }
        }
    }
}
