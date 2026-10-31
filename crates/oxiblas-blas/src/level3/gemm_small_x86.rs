//! AVX2 + FMA kernels for small GEMMs on x86_64.
//!
//! The counterpart of the NEON kernels in `gemm_small`: `C += alpha * A * B`
//! straight from column-major operands, without packing. C is walked in
//! register tiles of two vectors of rows (8 for f64, 16 for f32) by 4
//! columns; for every `p` the tile loads one column sliver of A and
//! broadcasts four elements of B, so 8 FMA chains are in flight. Rows past
//! the end of a column use masked loads and stores, which never touch memory
//! for masked-off lanes.
//!
//! Each element of C is accumulated over `p = 0..k` in order and then added
//! to C as `alpha * acc + c` (one FMA), the same per-element rounding as the
//! portable `gemm_small_acc` loops up to FMA contraction.

use core::arch::x86_64::*;

/// Returns true when the AVX2 small kernels can run on this CPU.
#[inline]
pub(super) fn available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Generates the driver and the register-tile kernel for one element type.
macro_rules! small_gemm_avx2 {
    (
        $driver:ident, $tile:ident, $t:ty, $v:ty, $lanes:expr,
        zero = $zero:ident, set1 = $set1:ident, loadu = $loadu:ident,
        storeu = $storeu:ident, maskload = $maskload:ident,
        maskstore = $maskstore:ident, fmadd = $fmadd:ident,
        bcast = $bcast:ident, row_mask = $row_mask:ident
    ) => {
        /// `C += alpha * A * B` for `m × k` A, `k × n` B and `m × n` C, all
        /// column-major with leading dimensions `lda`, `ldb`, `ldc`.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2 and FMA ([`available`]). `a` must be
        /// valid for reads of the `m × k` column-major matrix with leading
        /// dimension `lda >= m`, `b` likewise for `k × n` with `ldb >= k`, and
        /// `c` valid for reads and writes of `m × n` with `ldc >= m`, not
        /// overlapping A or B.
        #[target_feature(enable = "avx2", enable = "fma")]
        #[allow(clippy::too_many_arguments)]
        pub(super) unsafe fn $driver(
            m: usize,
            n: usize,
            k: usize,
            alpha: $t,
            a: *const $t,
            lda: usize,
            b: *const $t,
            ldb: usize,
            c: *mut $t,
            ldc: usize,
        ) {
            const MR: usize = 2 * $lanes;
            const NR: usize = 4;
            let mut i = 0;
            while i < m {
                let rows = MR.min(m - i);
                let (mask_lo, mask_hi) = $row_mask(rows);
                let a_i = a.wrapping_add(i);
                let c_i = c.wrapping_add(i);
                let mut j = 0;
                while j < n {
                    let b_j = b.wrapping_add(j * ldb);
                    let c_ij = c_i.wrapping_add(j * ldc);
                    // SAFETY: rows `i..i + rows` and columns `j..j + cols`
                    // lie inside A, B and C (caller contract); the masks
                    // disable every lane at or past `rows`.
                    unsafe {
                        match NR.min(n - j) {
                            4 => $tile::<4>(
                                rows, mask_lo, mask_hi, k, alpha, a_i, lda, b_j, ldb, c_ij, ldc,
                            ),
                            3 => $tile::<3>(
                                rows, mask_lo, mask_hi, k, alpha, a_i, lda, b_j, ldb, c_ij, ldc,
                            ),
                            2 => $tile::<2>(
                                rows, mask_lo, mask_hi, k, alpha, a_i, lda, b_j, ldb, c_ij, ldc,
                            ),
                            _ => $tile::<1>(
                                rows, mask_lo, mask_hi, k, alpha, a_i, lda, b_j, ldb, c_ij, ldc,
                            ),
                        }
                    }
                    j += NR;
                }
                i += MR;
            }
        }

        /// One register tile: `rows <= 2 * LANES` rows by `JB` columns.
        ///
        /// # Safety
        ///
        /// As for the driver, restricted to the tile; `mask_lo` / `mask_hi`
        /// must come from the row-mask helper for `rows`.
        #[target_feature(enable = "avx2", enable = "fma")]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $tile<const JB: usize>(
            rows: usize,
            mask_lo: __m256i,
            mask_hi: __m256i,
            k: usize,
            alpha: $t,
            a: *const $t,
            lda: usize,
            b: *const $t,
            ldb: usize,
            c: *mut $t,
            ldc: usize,
        ) {
            let full = rows == 2 * $lanes;
            let mut lo: [$v; JB] = [$zero(); JB];
            let mut hi: [$v; JB] = [$zero(); JB];
            // SAFETY (whole body): every unmasked access is inside the tile
            // (caller contract); masked-off lanes are never accessed, and
            // their addresses are formed with `wrapping_add`.
            unsafe {
                if full {
                    for p in 0..k {
                        let ap = a.add(p * lda);
                        let a_lo = $loadu(ap);
                        let a_hi = $loadu(ap.add($lanes));
                        for jj in 0..JB {
                            let bv = $bcast(&*b.add(p + jj * ldb));
                            lo[jj] = $fmadd(a_lo, bv, lo[jj]);
                            hi[jj] = $fmadd(a_hi, bv, hi[jj]);
                        }
                    }
                } else {
                    for p in 0..k {
                        let ap = a.wrapping_add(p * lda);
                        let a_lo = $maskload(ap, mask_lo);
                        let a_hi = $maskload(ap.wrapping_add($lanes), mask_hi);
                        for jj in 0..JB {
                            let bv = $bcast(&*b.add(p + jj * ldb));
                            lo[jj] = $fmadd(a_lo, bv, lo[jj]);
                            hi[jj] = $fmadd(a_hi, bv, hi[jj]);
                        }
                    }
                }
                let av = $set1(alpha);
                for jj in 0..JB {
                    let cp = c.wrapping_add(jj * ldc);
                    let cp_hi = cp.wrapping_add($lanes);
                    if full {
                        $storeu(cp, $fmadd(lo[jj], av, $loadu(cp)));
                        $storeu(cp_hi, $fmadd(hi[jj], av, $loadu(cp_hi)));
                    } else {
                        $maskstore(cp, mask_lo, $fmadd(lo[jj], av, $maskload(cp, mask_lo)));
                        $maskstore(
                            cp_hi,
                            mask_hi,
                            $fmadd(hi[jj], av, $maskload(cp_hi, mask_hi)),
                        );
                    }
                }
            }
        }
    };
}

/// Lane masks (all ones for lanes `< rows`) of the low and high f64 vectors
/// of an 8-row tile.
#[target_feature(enable = "avx2")]
#[inline]
fn row_mask_f64(rows: usize) -> (__m256i, __m256i) {
    let idx = _mm256_setr_epi64x(0, 1, 2, 3);
    let r = rows as i64;
    (
        _mm256_cmpgt_epi64(_mm256_set1_epi64x(r), idx),
        _mm256_cmpgt_epi64(_mm256_set1_epi64x(r - 4), idx),
    )
}

/// Lane masks of the low and high f32 vectors of a 16-row tile.
#[target_feature(enable = "avx2")]
#[inline]
fn row_mask_f32(rows: usize) -> (__m256i, __m256i) {
    let idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let r = rows as i32;
    (
        _mm256_cmpgt_epi32(_mm256_set1_epi32(r), idx),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(r - 8), idx),
    )
}

small_gemm_avx2!(
    gemm_small_f64_avx2,
    tile_f64,
    f64,
    __m256d,
    4,
    zero = _mm256_setzero_pd,
    set1 = _mm256_set1_pd,
    loadu = _mm256_loadu_pd,
    storeu = _mm256_storeu_pd,
    maskload = _mm256_maskload_pd,
    maskstore = _mm256_maskstore_pd,
    fmadd = _mm256_fmadd_pd,
    bcast = _mm256_broadcast_sd,
    row_mask = row_mask_f64
);

small_gemm_avx2!(
    gemm_small_f32_avx2,
    tile_f32,
    f32,
    __m256,
    8,
    zero = _mm256_setzero_ps,
    set1 = _mm256_set1_ps,
    loadu = _mm256_loadu_ps,
    storeu = _mm256_storeu_ps,
    maskload = _mm256_maskload_ps,
    maskstore = _mm256_maskstore_ps,
    fmadd = _mm256_fmadd_ps,
    bcast = _mm256_broadcast_ss,
    row_mask = row_mask_f32
);

#[cfg(test)]
mod tests {
    use super::*;

    fn val(i: usize, s: usize) -> f64 {
        ((i.wrapping_mul(2_654_435_761) ^ s.wrapping_mul(40_503)) % 1009) as f64 / 504.5 - 1.0
    }

    /// Runs the kernel for `T` on padded column-major operands and compares
    /// with a scalar reference, including that no element outside C (the
    /// padding rows) is written.
    macro_rules! check_type {
        ($t:ty, $kernel:ident, $tol:expr) => {{
            for m in [5usize, 7, 8, 9, 15, 16, 17, 23, 33, 40] {
                for n in [4usize, 5, 6, 7, 8, 13] {
                    for k in [1usize, 2, 7, 32, 65] {
                        let (lda, ldb, ldc) = (m + 3, k + 2, m + 5);
                        let a: Vec<$t> = (0..lda * k).map(|i| val(i, 1) as $t).collect();
                        let b: Vec<$t> = (0..ldb * n).map(|i| val(i, 2) as $t).collect();
                        let c0: Vec<$t> = (0..ldc * n).map(|i| val(i, 3) as $t).collect();
                        let alpha: $t = 0.75;
                        let mut c = c0.clone();
                        // SAFETY: the buffers hold the padded matrices.
                        unsafe {
                            $kernel(
                                m,
                                n,
                                k,
                                alpha,
                                a.as_ptr(),
                                lda,
                                b.as_ptr(),
                                ldb,
                                c.as_mut_ptr(),
                                ldc,
                            )
                        };
                        for j in 0..n {
                            for i in 0..ldc {
                                let got = c[i + j * ldc];
                                if i >= m {
                                    assert_eq!(got, c0[i + j * ldc], "padding written");
                                    continue;
                                }
                                let mut acc = 0.0f64;
                                for p in 0..k {
                                    acc += a[i + p * lda] as f64 * b[p + j * ldb] as f64;
                                }
                                let exp = alpha as f64 * acc + c0[i + j * ldc] as f64;
                                let err = (got as f64 - exp).abs();
                                assert!(
                                    err <= $tol * (1.0 + exp.abs()) * k as f64,
                                    "m={m} n={n} k={k} ({i},{j}): {got} vs {exp}"
                                );
                            }
                        }
                    }
                }
            }
        }};
    }

    #[test]
    fn avx2_small_kernels_match_reference() {
        if !available() {
            return;
        }
        check_type!(f64, gemm_small_f64_avx2, 1e-15);
        check_type!(f32, gemm_small_f32_avx2, 1e-6);
    }
}
