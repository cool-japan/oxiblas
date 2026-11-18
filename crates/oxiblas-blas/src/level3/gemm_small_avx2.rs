//! AVX2 + FMA small-GEMM kernels (x86-64): `C += alpha * A * B` on
//! column-major operands, without packing.
//!
//! [`small_f64_avx2`] and [`small_f32_avx2`] are the small kernels of the
//! [`SimdTier::Avx2Fma`](crate::level3::gemm_plan::SimdTier::Avx2Fma) tier.
//! `gemm_small_with` has already applied beta to C (C is zero-filled, not read,
//! when `beta == 0`) and handles `alpha == 0`; these kernels only accumulate.
//!
//! # Register tiles
//!
//! | Type | Rows per tile | Columns per tile | Accumulators |
//! |------|---------------|------------------|--------------|
//! | f64  | 8 (2 × `__m256d`) | `NC` in 1..=4 | `2·NC` |
//! | f32  | 16 (2 × `__m256`) | `NC` in 1..=4 | `2·NC` |
//!
//! C is walked column panel by column panel (`NC = min(4, n - jb)`), and each
//! panel row tile by row tile. A tile's rows are covered by a low and a high
//! vector, each in one of three states: full (plain `loadu` / `storeu`),
//! partly masked (`maskload` / `maskstore` with a lane mask built by comparing
//! the number of live rows against the lane indices), or absent (a high half
//! with no live row is skipped: no load, no store, no pointer). A low half is
//! never absent.
//!
//! # Per-element operation order (position independence)
//!
//! Every element `C[i, j]` is computed by exactly the same sequence of
//! operations, wherever it sits (full tile, row tail, column tail, lane):
//!
//! ```text
//! acc = +0
//! for p in 0..k (ascending): acc = fma(A[i, p], B[p, j], acc)   // one rounding each
//! C[i, j] = fma(acc, alpha, C[i, j])                           // one rounding
//! ```
//!
//! With `k == 0` nothing is added and C is left untouched (as in reference
//! BLAS). Because the sequence depends neither on the tile an element falls
//! in nor on the leading dimensions, the result of an element depends only
//! on the values of row `i` of A, column `j` of B, `C[i, j]` and alpha. So
//! a product computed on column-major copies of transposed operands
//! (`gemm_small_trans_copy`) is bit for bit the `NoTrans`/`NoTrans` product
//! on the same values, and a product computed as a sub-block of a larger one
//! gives the same bits as the stand-alone product. The sequence differs from
//! the portable small loops (which round the product and the sum
//! separately), so it changes the last bits relative to them; the routing
//! policy (`small_simd_x86`) decides whether a call gets here.
//!
//! # Memory accesses
//!
//! Only these elements are read: `A[i, p]` and `C[i, j]` for `i < m`, and
//! `B[p, j]` for `p < k`, `j < n`; only `C[i, j]` (`i < m`, `j < n`) is
//! written. Each lane of a partly masked vector whose row is `>= m` is masked
//! off, and a masked-off lane is never accessed (the masked instructions do
//! not touch memory for masked-off lanes, and do not fault on it). Pointers
//! to partly masked vectors, whose extent runs past the last row, are formed
//! with `wrapping_add`. B and C pointers are formed only for columns
//! `j < n`. So no access leaves the `m × k` area of A, the `k × n` area of B
//! or the `m × n` area of C, also in the last column and when a leading
//! dimension equals the row count.

use core::arch::x86_64::{
    __m256, __m256d, __m256i, _mm256_broadcast_sd, _mm256_broadcast_ss, _mm256_cmpgt_epi32,
    _mm256_cmpgt_epi64, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps,
    _mm256_maskload_pd, _mm256_maskload_ps, _mm256_maskstore_pd, _mm256_maskstore_ps,
    _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_set1_ps, _mm256_setr_epi32,
    _mm256_setr_epi64x, _mm256_setzero_pd, _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps,
};

/// Row coverage of one register tile (a const parameter of the tile code).
///
/// Both halves full.
const ROWS_FULL: u8 = 0;
/// Low half full, high half partly masked.
const ROWS_LO_FULL_HI_MASKED: u8 = 1;
/// Low half partly (or fully) masked, high half absent.
const ROWS_LO_MASKED: u8 = 2;

/// Lanes per `__m256d`.
const F64_LANES: usize = 4;
/// Rows of an `f64` register tile (two `__m256d`).
const F64_TILE_ROWS: usize = 2 * F64_LANES;
/// Lanes per `__m256`.
const F32_LANES: usize = 8;
/// Rows of an `f32` register tile (two `__m256`).
const F32_TILE_ROWS: usize = 2 * F32_LANES;
/// Columns of a full register tile.
const TILE_COLS: usize = 4;

/// Mask with lanes `0..live` set (`live` in `0..=4`), for `__m256d`.
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn mask_f64(live: usize) -> __m256i {
    // Lane l is live iff live > l (64-bit signed compare; live <= 4).
    _mm256_cmpgt_epi64(
        _mm256_set1_epi64x(live as i64),
        _mm256_setr_epi64x(0, 1, 2, 3),
    )
}

/// Mask with lanes `0..live` set (`live` in `0..=8`), for `__m256`.
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn mask_f32(live: usize) -> __m256i {
    // Lane l is live iff live > l (32-bit signed compare; live <= 8).
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(live as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

/// `C += alpha * A * B` for `f64`, column-major (AVX2 + FMA).
///
/// `A` is `m × k` with element `(i, p)` at `a + i + p·lda`, `B` is `k × n`
/// with `(p, j)` at `b + p + j·ldb`, and `C` is `m × n` with `(i, j)` at
/// `c + i + j·ldc`. Each element follows the sequence in the
/// [module documentation](self); `k == 0` leaves C untouched.
///
/// # Safety
///
/// - The CPU supports AVX2 and FMA.
/// - Every element `A[i, p]` (`i < m`, `p < k`) is readable, every `B[p, j]`
///   (`p < k`, `j < n`) is readable, and every `C[i, j]` (`i < m`, `j < n`)
///   is readable and writable, with no other reference to C alive; C does
///   not overlap A or B.
/// - Nothing outside those areas needs to be valid: no other element is
///   accessed (leading dimensions may equal the row counts, and the areas may
///   end exactly at the end of their allocations).
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn small_f64_avx2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut jb = 0;
    while jb < n {
        let nc = (n - jb).min(TILE_COLS);
        // Column jb < n exists, so these pointers address B[0, jb] and C[0, jb].
        let b_panel = b.wrapping_add(jb * ldb);
        let c_panel = c.wrapping_add(jb * ldc);
        match nc {
            4 => panel_f64::<4>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            3 => panel_f64::<3>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            2 => panel_f64::<2>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            _ => panel_f64::<1>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
        }
        jb += nc;
    }
}

/// One column panel of `NC` columns of [`small_f64_avx2`]: full row tiles,
/// then at most one row-tail tile.
///
/// # Safety
///
/// As for [`small_f64_avx2`], with `b` / `c` pointing at the panel's first
/// column, `NC` columns that exist, `m >= 1`, `k >= 1`.
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn panel_f64<const NC: usize>(
    m: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    let full_rows = m - m % F64_TILE_ROWS;
    let unused = mask_f64(0);
    let mut ib = 0;
    while ib < full_rows {
        tile_f64::<NC, ROWS_FULL>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            unused,
            unused,
        );
        ib += F64_TILE_ROWS;
    }
    let tail = m - full_rows;
    if tail > F64_LANES {
        tile_f64::<NC, ROWS_LO_FULL_HI_MASKED>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            unused,
            mask_f64(tail - F64_LANES),
        );
    } else if tail > 0 {
        tile_f64::<NC, ROWS_LO_MASKED>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            mask_f64(tail),
            unused,
        );
    }
}

/// One `f64` register tile: up to 8 rows (by `ROWS`) × `NC` columns.
///
/// `lo_mask` is used only for [`ROWS_LO_MASKED`], `hi_mask` only for
/// [`ROWS_LO_FULL_HI_MASKED`]; masked-off lanes are rows `>= m`.
///
/// # Safety
///
/// `a`, `c` point at the tile's first row (in column 0 of A, column 0 of the
/// panel of C); every live row (by `ROWS` and the masks) of A's first `k`
/// columns and of C's `NC` columns is valid; B's rows `0..k` of `NC` columns
/// are valid; the CPU supports AVX2 and FMA.
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tile_f64<const NC: usize, const ROWS: u8>(
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
    lo_mask: __m256i,
    hi_mask: __m256i,
) {
    let has_hi = ROWS != ROWS_LO_MASKED;
    let mut lo: [__m256d; NC] = [_mm256_setzero_pd(); NC];
    let mut hi: [__m256d; NC] = [_mm256_setzero_pd(); NC];
    for p in 0..k {
        let col = a.wrapping_add(p * lda);
        let a_lo = if ROWS == ROWS_LO_MASKED {
            _mm256_maskload_pd(col, lo_mask)
        } else {
            _mm256_loadu_pd(col)
        };
        let a_hi = if ROWS == ROWS_FULL {
            _mm256_loadu_pd(col.wrapping_add(F64_LANES))
        } else if ROWS == ROWS_LO_FULL_HI_MASKED {
            _mm256_maskload_pd(col.wrapping_add(F64_LANES), hi_mask)
        } else {
            _mm256_setzero_pd()
        };
        for j in 0..NC {
            let bj = _mm256_broadcast_sd(&*b.wrapping_add(p + j * ldb));
            lo[j] = _mm256_fmadd_pd(a_lo, bj, lo[j]);
            if has_hi {
                hi[j] = _mm256_fmadd_pd(a_hi, bj, hi[j]);
            }
        }
    }
    let alpha_v = _mm256_set1_pd(alpha);
    for j in 0..NC {
        let col = c.wrapping_add(j * ldc);
        if ROWS == ROWS_LO_MASKED {
            let c_lo = _mm256_maskload_pd(col, lo_mask);
            _mm256_maskstore_pd(col, lo_mask, _mm256_fmadd_pd(lo[j], alpha_v, c_lo));
        } else {
            let c_lo = _mm256_loadu_pd(col);
            _mm256_storeu_pd(col, _mm256_fmadd_pd(lo[j], alpha_v, c_lo));
        }
        // The high half's pointer is formed only where the high half has
        // live rows (an absent high half gets no pointer at all).
        if ROWS == ROWS_FULL {
            let col_hi = col.wrapping_add(F64_LANES);
            let c_hi = _mm256_loadu_pd(col_hi);
            _mm256_storeu_pd(col_hi, _mm256_fmadd_pd(hi[j], alpha_v, c_hi));
        } else if ROWS == ROWS_LO_FULL_HI_MASKED {
            let col_hi = col.wrapping_add(F64_LANES);
            let c_hi = _mm256_maskload_pd(col_hi, hi_mask);
            _mm256_maskstore_pd(col_hi, hi_mask, _mm256_fmadd_pd(hi[j], alpha_v, c_hi));
        }
    }
}

/// `C += alpha * A * B` for `f32`, column-major (AVX2 + FMA).
///
/// Same layout and contract as [`small_f64_avx2`], with 16-row tiles.
///
/// # Safety
///
/// As for [`small_f64_avx2`].
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn small_f32_avx2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut jb = 0;
    while jb < n {
        let nc = (n - jb).min(TILE_COLS);
        // Column jb < n exists, so these pointers address B[0, jb] and C[0, jb].
        let b_panel = b.wrapping_add(jb * ldb);
        let c_panel = c.wrapping_add(jb * ldc);
        match nc {
            4 => panel_f32::<4>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            3 => panel_f32::<3>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            2 => panel_f32::<2>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
            _ => panel_f32::<1>(m, k, alpha, a, lda, b_panel, ldb, c_panel, ldc),
        }
        jb += nc;
    }
}

/// One column panel of `NC` columns of [`small_f32_avx2`].
///
/// # Safety
///
/// As for [`panel_f64`].
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn panel_f32<const NC: usize>(
    m: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let full_rows = m - m % F32_TILE_ROWS;
    let unused = mask_f32(0);
    let mut ib = 0;
    while ib < full_rows {
        tile_f32::<NC, ROWS_FULL>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            unused,
            unused,
        );
        ib += F32_TILE_ROWS;
    }
    let tail = m - full_rows;
    if tail > F32_LANES {
        tile_f32::<NC, ROWS_LO_FULL_HI_MASKED>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            unused,
            mask_f32(tail - F32_LANES),
        );
    } else if tail > 0 {
        tile_f32::<NC, ROWS_LO_MASKED>(
            k,
            alpha,
            a.wrapping_add(ib),
            lda,
            b,
            ldb,
            c.wrapping_add(ib),
            ldc,
            mask_f32(tail),
            unused,
        );
    }
}

/// One `f32` register tile: up to 16 rows (by `ROWS`) × `NC` columns.
///
/// # Safety
///
/// As for [`tile_f64`].
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tile_f32<const NC: usize, const ROWS: u8>(
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    lo_mask: __m256i,
    hi_mask: __m256i,
) {
    let has_hi = ROWS != ROWS_LO_MASKED;
    let mut lo: [__m256; NC] = [_mm256_setzero_ps(); NC];
    let mut hi: [__m256; NC] = [_mm256_setzero_ps(); NC];
    for p in 0..k {
        let col = a.wrapping_add(p * lda);
        let a_lo = if ROWS == ROWS_LO_MASKED {
            _mm256_maskload_ps(col, lo_mask)
        } else {
            _mm256_loadu_ps(col)
        };
        let a_hi = if ROWS == ROWS_FULL {
            _mm256_loadu_ps(col.wrapping_add(F32_LANES))
        } else if ROWS == ROWS_LO_FULL_HI_MASKED {
            _mm256_maskload_ps(col.wrapping_add(F32_LANES), hi_mask)
        } else {
            _mm256_setzero_ps()
        };
        for j in 0..NC {
            let bj = _mm256_broadcast_ss(&*b.wrapping_add(p + j * ldb));
            lo[j] = _mm256_fmadd_ps(a_lo, bj, lo[j]);
            if has_hi {
                hi[j] = _mm256_fmadd_ps(a_hi, bj, hi[j]);
            }
        }
    }
    let alpha_v = _mm256_set1_ps(alpha);
    for j in 0..NC {
        let col = c.wrapping_add(j * ldc);
        if ROWS == ROWS_LO_MASKED {
            let c_lo = _mm256_maskload_ps(col, lo_mask);
            _mm256_maskstore_ps(col, lo_mask, _mm256_fmadd_ps(lo[j], alpha_v, c_lo));
        } else {
            let c_lo = _mm256_loadu_ps(col);
            _mm256_storeu_ps(col, _mm256_fmadd_ps(lo[j], alpha_v, c_lo));
        }
        // The high half's pointer is formed only where the high half has
        // live rows (an absent high half gets no pointer at all).
        if ROWS == ROWS_FULL {
            let col_hi = col.wrapping_add(F32_LANES);
            let c_hi = _mm256_loadu_ps(col_hi);
            _mm256_storeu_ps(col_hi, _mm256_fmadd_ps(hi[j], alpha_v, c_hi));
        } else if ROWS == ROWS_LO_FULL_HI_MASKED {
            let col_hi = col.wrapping_add(F32_LANES);
            let c_hi = _mm256_maskload_ps(col_hi, hi_mask);
            _mm256_maskstore_ps(col_hi, hi_mask, _mm256_fmadd_ps(hi[j], alpha_v, c_hi));
        }
    }
}
