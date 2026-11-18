//! Tests for the GEMM packing routines.

use super::*;
use num_complex::{Complex32, Complex64};
use oxiblas_matrix::Mat;

#[test]
fn test_pack_a_optimized() {
    let a: Mat<f64> = Mat::from_rows(&[
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0],
    ]);

    let mr = 2;
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(16);

    pack_a_optimized(&a.as_ref(), 0, 0, 4, 4, &mut pack, mr);

    // Check first block (rows 0-1, all columns)
    // Should be: [1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
    assert!((pack[0] - 1.0).abs() < 1e-10);
    assert!((pack[1] - 5.0).abs() < 1e-10);
    assert!((pack[2] - 2.0).abs() < 1e-10);
    assert!((pack[3] - 6.0).abs() < 1e-10);
}

#[test]
fn test_pack_b_optimized() {
    let b: Mat<f64> = Mat::from_rows(&[
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0],
    ]);

    let nr = 2;
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(16);

    pack_b_optimized(&b.as_ref(), 0, 0, 4, 4, &mut pack, nr);

    // Check that packing produces correct layout
    // First NR columns, then next NR columns
    assert!((pack[0] - 1.0).abs() < 1e-10);
    assert!((pack[1] - 2.0).abs() < 1e-10);
    assert!((pack[2] - 5.0).abs() < 1e-10);
    assert!((pack[3] - 6.0).abs() < 1e-10);
}

#[test]
fn test_pack_a_partial_block() {
    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

    let mr = 4; // Larger than nrows, will need padding
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(12);

    pack_a_optimized(&a.as_ref(), 0, 0, 3, 3, &mut pack, mr);

    // First column: [1, 4, 7, 0] (padded)
    assert!((pack[0] - 1.0).abs() < 1e-10);
    assert!((pack[1] - 4.0).abs() < 1e-10);
    assert!((pack[2] - 7.0).abs() < 1e-10);
    assert!((pack[3] - 0.0).abs() < 1e-10); // padding
}

#[test]
fn test_packing_config() {
    let config = PackingConfig::default();
    assert!(!config.use_streaming);
    assert!(config.use_unrolling);

    let large_config = PackingConfig::for_large_matrix();
    assert!(large_config.use_streaming);
    assert_eq!(large_config.prefetch_distance, 8);

    let small_config = PackingConfig::for_small_matrix();
    assert!(!small_config.use_streaming);
    assert!(!small_config.use_unrolling);
}

#[test]
fn test_pack_b_simd_f64() {
    let b: Mat<f64> = Mat::from_rows(&[
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        &[33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
        &[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
        &[49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        &[57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
    ]);

    let nr = 4;
    let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(64);
    let mut pack_simd: AlignedVec<f64> = AlignedVec::zeros(64);

    pack_b_optimized(&b.as_ref(), 0, 0, 8, 8, &mut pack_opt, nr);
    pack_b_simd_f64(&b.as_ref(), 0, 0, 8, 8, &mut pack_simd, nr);

    // SIMD version should produce identical results
    for i in 0..64 {
        assert!(
            (pack_opt[i] - pack_simd[i]).abs() < 1e-10,
            "Mismatch at index {}: {} vs {}",
            i,
            pack_opt[i],
            pack_simd[i]
        );
    }
}

#[test]
fn test_pack_b_simd_f32() {
    let b: Mat<f32> = Mat::from_rows(&[
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        &[33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
        &[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
        &[49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        &[57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
    ]);

    let nr = 4;
    let mut pack_opt: AlignedVec<f32> = AlignedVec::zeros(64);
    let mut pack_simd: AlignedVec<f32> = AlignedVec::zeros(64);

    pack_b_optimized(&b.as_ref(), 0, 0, 8, 8, &mut pack_opt, nr);
    pack_b_simd_f32(&b.as_ref(), 0, 0, 8, 8, &mut pack_simd, nr);

    for i in 0..64 {
        assert!(
            (pack_opt[i] - pack_simd[i]).abs() < 1e-5,
            "Mismatch at index {}: {} vs {}",
            i,
            pack_opt[i],
            pack_simd[i]
        );
    }
}

#[test]
fn test_pack_b_simd_large() {
    // Test with a larger matrix to exercise the 8-way unrolling
    let size = 32;
    let data: Vec<Vec<f64>> = (0..size)
        .map(|i| (0..size).map(|j| (i * size + j) as f64).collect())
        .collect();
    let b: Mat<f64> = Mat::from_rows(&data.iter().map(|r| r.as_slice()).collect::<Vec<_>>());

    let nr = 8;
    let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(size * size);
    let mut pack_simd: AlignedVec<f64> = AlignedVec::zeros(size * size);

    pack_b_optimized(&b.as_ref(), 0, 0, size, size, &mut pack_opt, nr);
    pack_b_simd_f64(&b.as_ref(), 0, 0, size, size, &mut pack_simd, nr);

    for i in 0..(size * size) {
        assert!(
            (pack_opt[i] - pack_simd[i]).abs() < 1e-10,
            "Large matrix mismatch at {}: {} vs {}",
            i,
            pack_opt[i],
            pack_simd[i]
        );
    }
}

// ------------------------------------------------------------------
// Regression tests for the packing correctness fixes.
// ------------------------------------------------------------------

/// Naive column-major reference for `pack_a`: re-tiles the panel into
/// `mr`-row blocks with zero padding for the trailing partial block, exactly
/// matching the layout contract of [`pack_a_optimized`].
fn pack_a_reference<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    mr: usize,
) -> Vec<T> {
    let mut out = Vec::new();
    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);
        for p in 0..ncols {
            for ii in 0..mr {
                if ii < ib {
                    out.push(a[(row_start + i + ii, col_start + p)]);
                } else {
                    out.push(T::zero());
                }
            }
        }
    }
    out
}

/// Naive reference for `pack_b`: `nr`-column blocks, row-major within a
/// block, zero padding for the trailing partial block.
fn pack_b_reference<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    nr: usize,
) -> Vec<T> {
    let mut out = Vec::new();
    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        for p in 0..nrows {
            for jj in 0..nr {
                if jj < jb {
                    out.push(b[(row_start + p, col_start + j + jj)]);
                } else {
                    out.push(T::zero());
                }
            }
        }
    }
    out
}

/// Row-major sequential test matrix `[[1,2,..],[..]]` of the given shape.
fn seq_mat_f64(nrows: usize, ncols: usize) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = (0..nrows)
        .map(|i| (0..ncols).map(|j| (i * ncols + j + 1) as f64).collect())
        .collect();
    Mat::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>())
}

fn seq_mat_f32(nrows: usize, ncols: usize) -> Mat<f32> {
    let rows: Vec<Vec<f32>> = (0..nrows)
        .map(|i| (0..ncols).map(|j| (i * ncols + j + 1) as f32).collect())
        .collect();
    Mat::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>())
}

#[test]
fn test_pack_a_contiguous_matches_reference() {
    // For every shape/`mr` the fast path and the fallback must both agree
    // with the naive reference AND with the general `pack_a_optimized` path.
    //
    // The (8x8, mr=4) case is the direct regression for the original bug:
    // the old flat-copy fast path fired for the whole contiguous matrix
    // regardless of `mr`, emitting plain column-major order instead of the
    // 4-row-tiled layout `pack_a_optimized` produces. The (8x4, mr=8) case
    // covers the wrong `row_stride == ncols` contiguity test on a non-square
    // matrix.
    let cases = [
        (8usize, 8usize, 8usize), // fast-path eligible on x86_64 (rs==nrows==mr)
        (8, 8, 4),                // contiguous but multi-block -> must delegate
        (8, 4, 8),                // non-square: old `== ncols` check was wrong
        (5, 7, 3),                // padded (non-contiguous) -> fallback
        (3, 3, 4),                // partial block with zero padding
    ];

    for (nrows, ncols, mr) in cases {
        let a = seq_mat_f64(nrows, ncols);
        let a_ref = a.as_ref();
        let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
        let total = reference.len();

        let mut pack_contig: AlignedVec<f64> = AlignedVec::zeros(total);
        let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(total);
        pack_a_contiguous(&a_ref, 0, 0, nrows, ncols, &mut pack_contig, mr);
        pack_a_optimized(&a_ref, 0, 0, nrows, ncols, &mut pack_opt, mr);

        for k in 0..total {
            assert!(
                (pack_contig[k] - reference[k]).abs() < 1e-12,
                "pack_a_contiguous vs reference mismatch at {k} for {nrows}x{ncols} mr={mr}: {} vs {}",
                pack_contig[k],
                reference[k]
            );
            assert!(
                (pack_contig[k] - pack_opt[k]).abs() < 1e-12,
                "pack_a_contiguous vs pack_a_optimized mismatch at {k} for {nrows}x{ncols} mr={mr}",
            );
        }
    }
}

#[test]
fn test_pack_a_contiguous_fast_path_layout() {
    // A genuinely contiguous, single-micro-panel matrix (row_stride == nrows,
    // nrows == mr) that takes the flat-copy fast path. 16 rows is a multiple
    // of the f64 cache-line element count under both 64-byte (8) and 128-byte
    // (16) padding, so the matrix is contiguous on every supported target. We
    // assert the precondition so the fast path is actually exercised, then
    // check it reproduces the reference layout.
    let nrows = 16;
    let ncols = 6;
    let a = seq_mat_f64(nrows, ncols);
    let a_ref = a.as_ref();
    assert_eq!(
        a_ref.row_stride(),
        a_ref.nrows(),
        "expected a contiguous {nrows}-row matrix on this target"
    );
    let mr = a_ref.nrows();
    let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(reference.len());
    pack_a_contiguous(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
    for k in 0..reference.len() {
        assert!(
            (pack[k] - reference[k]).abs() < 1e-12,
            "fast-path mismatch at {k}"
        );
    }
}

#[test]
fn test_pack_b_streaming_matches_optimized_f64() {
    // Shapes cover full blocks, zero-padded partial blocks, and total element
    // counts that are and are not multiples of the 4-element (32-byte) AVX
    // lane, exercising both the streaming lanes and the ordinary tail on AVX
    // hardware. Elsewhere this validates the identical fallback.
    let cases = [
        (8usize, 8usize, 4usize),
        (3, 4, 4), // total 12 -> three full lanes, no tail
        (3, 6, 4), // partial second block (jb=2, zero-padded), total 24
        (5, 5, 6), // nr > ncols -> single partial block, total 30 (tail 2)
        (16, 12, 8),
    ];
    for (nrows, ncols, nr) in cases {
        let b = seq_mat_f64(nrows, ncols);
        let b_ref = b.as_ref();
        let reference = pack_b_reference(&b_ref, 0, 0, nrows, ncols, nr);
        let total = reference.len();

        let mut pack_stream: AlignedVec<f64> = AlignedVec::zeros(total);
        let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(total);
        pack_b_streaming(&b_ref, 0, 0, nrows, ncols, &mut pack_stream, nr);
        pack_b_optimized(&b_ref, 0, 0, nrows, ncols, &mut pack_opt, nr);

        for k in 0..total {
            assert!(
                (pack_stream[k] - reference[k]).abs() < 1e-12,
                "streaming vs reference mismatch at {k} for {nrows}x{ncols} nr={nr}: {} vs {}",
                pack_stream[k],
                reference[k]
            );
            assert!(
                (pack_stream[k] - pack_opt[k]).abs() < 1e-12,
                "streaming vs optimized mismatch at {k} for {nrows}x{ncols} nr={nr}",
            );
        }
    }
}

#[test]
fn test_pack_b_streaming_matches_optimized_f32() {
    // f32 uses an 8-element (32-byte) lane; (5,3,2) yields total 20 = two
    // full lanes plus a 4-element tail.
    let cases = [(4usize, 8usize, 4usize), (7, 5, 4), (5, 3, 2), (20, 16, 8)];
    for (nrows, ncols, nr) in cases {
        let b = seq_mat_f32(nrows, ncols);
        let b_ref = b.as_ref();
        let reference = pack_b_reference(&b_ref, 0, 0, nrows, ncols, nr);
        let total = reference.len();

        let mut pack_stream: AlignedVec<f32> = AlignedVec::zeros(total);
        pack_b_streaming(&b_ref, 0, 0, nrows, ncols, &mut pack_stream, nr);

        for k in 0..total {
            assert!(
                (pack_stream[k] - reference[k]).abs() < 1e-5,
                "f32 streaming mismatch at {k} for {nrows}x{ncols} nr={nr}",
            );
        }
    }
}

#[test]
fn test_pack_b_streaming_subregion() {
    // Pack an interior region (row_start/col_start != 0) of a larger matrix,
    // exercising the streaming gather with a non-zero origin and a padded
    // leading dimension.
    let b = seq_mat_f64(10, 10);
    let b_ref = b.as_ref();
    let (row_start, col_start, nrows, ncols, nr) = (2usize, 3usize, 6usize, 5usize, 4usize);
    let reference = pack_b_reference(&b_ref, row_start, col_start, nrows, ncols, nr);
    let total = reference.len();

    let mut pack_stream: AlignedVec<f64> = AlignedVec::zeros(total);
    pack_b_streaming(
        &b_ref,
        row_start,
        col_start,
        nrows,
        ncols,
        &mut pack_stream,
        nr,
    );

    for k in 0..total {
        assert!(
            (pack_stream[k] - reference[k]).abs() < 1e-12,
            "subregion streaming mismatch at {k}"
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_pack_a_simd_f64_matches_reference() {
    // With the corrected guard the AVX2 path fires for real column-major
    // storage (any row_stride, including the padded strides these matrices
    // have), so on AVX2 hardware this exercises `pack_a_simd_contiguous_f64`
    // and checks it against the scalar reference.
    let cases = [(8usize, 8usize, 4usize), (16, 5, 8), (10, 7, 4)];
    for (nrows, ncols, mr) in cases {
        let a = seq_mat_f64(nrows, ncols);
        let a_ref = a.as_ref();
        let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
        let mut pack: AlignedVec<f64> = AlignedVec::zeros(reference.len());
        pack_a_simd_f64(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
        for k in 0..reference.len() {
            assert!(
                (pack[k] - reference[k]).abs() < 1e-12,
                "pack_a_simd_f64 mismatch at {k} for {nrows}x{ncols} mr={mr}"
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_pack_a_simd_f32_matches_reference() {
    let cases = [(16usize, 8usize, 8usize), (20, 5, 8)];
    for (nrows, ncols, mr) in cases {
        let a = seq_mat_f32(nrows, ncols);
        let a_ref = a.as_ref();
        let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
        let mut pack: AlignedVec<f32> = AlignedVec::zeros(reference.len());
        pack_a_simd_f32(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
        for k in 0..reference.len() {
            assert!(
                (pack[k] - reference[k]).abs() < 1e-5,
                "pack_a_simd_f32 mismatch at {k} for {nrows}x{ncols} mr={mr}"
            );
        }
    }
}

// ------------------------------------------------------------------
// Compile-time MR/NR packers: byte-identical layout for every shape.
// ------------------------------------------------------------------

/// Distinct, non-symmetric sample values for each element type.
trait Sample: Field + bytemuck::Zeroable {
    fn sample(seed: usize) -> Self;
    fn poison() -> Self;
    fn is_poison(&self) -> bool;
}

impl Sample for f64 {
    fn sample(seed: usize) -> Self {
        seed as f64 * 0.5 + 1.0
    }
    fn poison() -> Self {
        f64::NAN
    }
    fn is_poison(&self) -> bool {
        self.is_nan()
    }
}

impl Sample for f32 {
    fn sample(seed: usize) -> Self {
        (seed % 4096) as f32 * 0.5 + 1.0
    }
    fn poison() -> Self {
        f32::NAN
    }
    fn is_poison(&self) -> bool {
        self.is_nan()
    }
}

impl Sample for Complex64 {
    fn sample(seed: usize) -> Self {
        Complex64::new(seed as f64 + 1.0, -(seed as f64) * 0.25)
    }
    fn poison() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn is_poison(&self) -> bool {
        self.re.is_nan() && self.im.is_nan()
    }
}

impl Sample for Complex32 {
    fn sample(seed: usize) -> Self {
        Complex32::new((seed % 4096) as f32 + 1.0, -((seed % 512) as f32))
    }
    fn poison() -> Self {
        Complex32::new(f32::NAN, f32::NAN)
    }
    fn is_poison(&self) -> bool {
        self.re.is_nan() && self.im.is_nan()
    }
}

fn sample_mat<T: Sample>(rows: usize, cols: usize) -> Mat<T> {
    let mut m: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            m[(i, j)] = T::sample(i * 1009 + j * 7 + 3);
        }
    }
    m
}

/// Regions `(row_start, col_start, nrows, ncols)` of a 37 x 41 matrix: full
/// panels, partial panels, a nonzero origin, and empty regions.
const REGIONS: [(usize, usize, usize, usize); 8] = [
    (0, 0, 37, 41),
    (0, 0, 32, 16),
    (3, 5, 29, 23),
    (1, 2, 7, 9),
    (5, 0, 16, 1),
    (0, 7, 1, 30),
    (10, 10, 0, 5),
    (10, 10, 5, 0),
];

/// Micro-panel sizes: every MR / NR of the crate's kernels plus run-time ones.
const PANEL_SIZES: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16];

/// Packs every region at every panel size with both packers into a buffer
/// pre-filled with NaN, and checks the packed prefix against the naive
/// references (exact equality) and that nothing past it was written.
fn check_packers_match_reference<T: Sample>(src: MatRef<'_, T>) {
    for &(r0, c0, rows, cols) in &REGIONS {
        if r0 + rows > src.nrows() || c0 + cols > src.ncols() {
            continue;
        }
        for &panel in &PANEL_SIZES {
            let expected = pack_a_reference(&src, r0, c0, rows, cols, panel);
            let mut pack: AlignedVec<T> = AlignedVec::filled(expected.len() + 3, T::poison());
            pack_a_optimized(&src, r0, c0, rows, cols, &mut pack, panel);
            let ctx = format!("pack_a {r0},{c0} {rows}x{cols} mr={panel}");
            assert!(
                pack.as_slice()[..expected.len()] == expected[..],
                "{ctx}: layout"
            );
            assert!(
                pack.as_slice()[expected.len()..]
                    .iter()
                    .all(Sample::is_poison),
                "{ctx}: wrote past the packed prefix"
            );

            let expected = pack_b_reference(&src, r0, c0, rows, cols, panel);
            let mut pack: AlignedVec<T> = AlignedVec::filled(expected.len() + 3, T::poison());
            pack_b_optimized(&src, r0, c0, rows, cols, &mut pack, panel);
            let ctx = format!("pack_b {r0},{c0} {rows}x{cols} nr={panel}");
            assert!(
                pack.as_slice()[..expected.len()] == expected[..],
                "{ctx}: layout"
            );
            assert!(
                pack.as_slice()[expected.len()..]
                    .iter()
                    .all(Sample::is_poison),
                "{ctx}: wrote past the packed prefix"
            );
        }
    }
}

fn check_packers_all_views<T: Sample>() {
    // Plain storage (37 rows: the leading dimension is padded).
    let plain = sample_mat::<T>(37, 41);
    check_packers_match_reference(plain.as_ref());
    // A view into a larger matrix: leading dimension well above the row count.
    let big = sample_mat::<T>(53, 47);
    check_packers_match_reference(big.as_ref().submatrix(9, 4, 37, 41));
    // A transposed view (rows not adjacent in memory): element-wise path.
    let stored = sample_mat::<T>(41, 37);
    let transposed = stored.as_ref().transpose();
    assert_ne!(transposed.col_stride(), 1);
    check_packers_match_reference(transposed);
}

#[test]
fn test_packers_match_reference_f64() {
    check_packers_all_views::<f64>();
}

#[test]
fn test_packers_match_reference_f32() {
    check_packers_all_views::<f32>();
}

#[test]
fn test_packers_match_reference_c64() {
    check_packers_all_views::<Complex64>();
}

#[test]
fn test_packers_match_reference_c32() {
    check_packers_all_views::<Complex32>();
}

#[test]
fn test_packers_tall_panels_hit_neon_tails() {
    // Row counts around the 2-row (f64 x 6) and 4-row (f32 x 8) register
    // transposes, and many full panels, so every tail runs.
    for rows in [1usize, 2, 3, 4, 5, 6, 7, 9, 64, 131] {
        let b64 = sample_mat::<f64>(rows, 6 * 5 + 4);
        let expected = pack_b_reference(&b64.as_ref(), 0, 0, rows, 34, 6);
        let mut pack: AlignedVec<f64> = AlignedVec::filled(expected.len(), f64::NAN);
        pack_b_optimized(&b64.as_ref(), 0, 0, rows, 34, &mut pack, 6);
        assert!(pack.as_slice() == &expected[..], "f64 x 6, {rows} rows");

        let b32 = sample_mat::<f32>(rows, 8 * 3 + 5);
        let expected = pack_b_reference(&b32.as_ref(), 0, 0, rows, 29, 8);
        let mut pack: AlignedVec<f32> = AlignedVec::filled(expected.len(), f32::NAN);
        pack_b_optimized(&b32.as_ref(), 0, 0, rows, 29, &mut pack, 8);
        assert!(pack.as_slice() == &expected[..], "f32 x 8, {rows} rows");
    }
}

#[test]
#[should_panic(expected = "buffer too small")]
fn test_pack_a_optimized_rejects_short_buffer() {
    let a = sample_mat::<f64>(9, 4);
    // ceil(9 / 8) * 8 * 4 = 64 elements are needed.
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(63);
    pack_a_optimized(&a.as_ref(), 0, 0, 9, 4, &mut pack, 8);
}

#[test]
#[should_panic(expected = "buffer too small")]
fn test_pack_b_optimized_rejects_short_buffer() {
    let b = sample_mat::<f32>(4, 9);
    let mut pack: AlignedVec<f32> = AlignedVec::zeros(4 * 16 - 1);
    pack_b_optimized(&b.as_ref(), 0, 0, 4, 9, &mut pack, 8);
}

#[test]
#[should_panic(expected = "region outside the matrix")]
fn test_pack_b_optimized_rejects_region_outside() {
    let b = sample_mat::<f64>(4, 9);
    let mut pack: AlignedVec<f64> = AlignedVec::zeros(256);
    pack_b_optimized(&b.as_ref(), 1, 0, 4, 9, &mut pack, 6);
}

// ------------------------------------------------------------------
// Per-thread packing-buffer cache.
// ------------------------------------------------------------------

/// Everything parked in the slots of the four element types on this thread.
fn parked_bytes() -> usize {
    let mut total = 0;
    for slot in [PackSlot::A, PackSlot::B] {
        total += pack_cache_capacity::<f64>(slot) * 8;
        total += pack_cache_capacity::<f32>(slot) * 4;
        total += pack_cache_capacity::<Complex64>(slot) * 16;
        total += pack_cache_capacity::<Complex32>(slot) * 8;
    }
    total
}

#[test]
fn test_pack_buffer_reuses_and_grows() {
    let first_ptr = {
        let buf = PackBuffer::<f64>::take(PackSlot::A, 1000);
        assert!(buf.len() >= 1000);
        buf.as_ptr()
    };
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 1000);
    // A smaller request reuses the same allocation (no re-zeroing, no free).
    {
        let buf = PackBuffer::<f64>::take(PackSlot::A, 10);
        assert_eq!(buf.as_ptr(), first_ptr);
        assert_eq!(buf.len(), 1000);
        // The slot is empty while the buffer is on loan.
        assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 0);
    }
    // A larger request grows it.
    {
        let buf = PackBuffer::<f64>::take(PackSlot::A, 5000);
        assert!(buf.len() >= 5000);
    }
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 5000);
    assert_eq!(pack_cache_bytes(), parked_bytes());
}

#[test]
fn test_pack_buffer_slots_and_types_are_separate() {
    drop(PackBuffer::<f64>::take(PackSlot::A, 64));
    drop(PackBuffer::<f64>::take(PackSlot::B, 128));
    drop(PackBuffer::<f32>::take(PackSlot::A, 256));
    drop(PackBuffer::<Complex64>::take(PackSlot::B, 32));
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 64);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), 128);
    assert_eq!(pack_cache_capacity::<f32>(PackSlot::A), 256);
    assert_eq!(pack_cache_capacity::<f32>(PackSlot::B), 0);
    assert_eq!(pack_cache_capacity::<Complex64>(PackSlot::B), 32);
    assert_eq!(pack_cache_bytes(), parked_bytes());
    assert_eq!(pack_cache_bytes(), 64 * 8 + 128 * 8 + 256 * 4 + 32 * 16);
}

#[test]
fn test_pack_buffer_nested_loan_allocates_its_own() {
    let outer = PackBuffer::<f32>::take(PackSlot::B, 300);
    {
        // The slot is empty while `outer` is on loan, so a nested GEMM on this
        // thread gets a separate buffer instead of aliasing (or panicking).
        let inner = PackBuffer::<f32>::take(PackSlot::B, 200);
        assert_ne!(inner.as_ptr(), outer.as_ptr());
    }
    assert_eq!(pack_cache_capacity::<f32>(PackSlot::B), 200);
    drop(outer);
    // The outer loan replaces what the nested one parked.
    assert_eq!(pack_cache_capacity::<f32>(PackSlot::B), 300);
    assert_eq!(pack_cache_bytes(), parked_bytes());
}

#[test]
fn test_pack_buffer_policy_frees_oversized_buffers() {
    let too_big = PACK_CACHE_MAX_BYTES / 8 + 1;
    {
        let buf = PackBuffer::<f64>::take(PackSlot::B, too_big);
        assert!(buf.len() >= too_big);
    }
    assert_eq!(
        pack_cache_capacity::<f64>(PackSlot::B),
        0,
        "oversized buffer kept"
    );
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);

    // Two buffers that together exceed the limit: the one that does not fit
    // is freed, and the total stays bounded.
    let half = PACK_CACHE_MAX_BYTES / 8 / 2 + 16;
    drop(PackBuffer::<f64>::take(PackSlot::A, half));
    drop(PackBuffer::<f64>::take(PackSlot::B, half));
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
    assert_eq!(pack_cache_bytes(), parked_bytes());
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), half);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), 0);
}

#[test]
fn test_pack_cache_limit_is_per_thread_and_restored() {
    assert_eq!(pack_cache_limit(), PACK_CACHE_MAX_BYTES);
    // Start from empty f64 slots so only this test's buffers are parked.
    drop(PackBuffer::<f64>::take(
        PackSlot::A,
        PACK_CACHE_MAX_BYTES / 8 + 1,
    ));
    drop(PackBuffer::<f64>::take(
        PackSlot::B,
        PACK_CACHE_MAX_BYTES / 8 + 1,
    ));
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 0);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), 0);
    {
        // A lower limit: 4 KiB.
        let _guard = set_pack_cache_limit(4096);
        assert_eq!(pack_cache_limit(), 4096);
        drop(PackBuffer::<f64>::take(PackSlot::A, 513));
        assert_eq!(
            pack_cache_capacity::<f64>(PackSlot::A),
            0,
            "4104 bytes kept"
        );
        drop(PackBuffer::<f64>::take(PackSlot::A, 512));
        assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), 512);
        // The limit is per thread: another thread keeps the default.
        let other = std::thread::spawn(pack_cache_limit).join();
        assert_eq!(other.ok(), Some(PACK_CACHE_MAX_BYTES));
    }
    assert_eq!(pack_cache_limit(), PACK_CACHE_MAX_BYTES, "guard restores");
    {
        // A higher limit keeps a buffer the default frees: 12 MiB of f64.
        let big = (12 << 20) / 8;
        let _guard = set_pack_cache_limit(16 << 20);
        drop(PackBuffer::<f64>::take(PackSlot::B, big));
        assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), big);
        assert!(pack_cache_bytes() <= 16 << 20);
        // Back under the default: the parked buffer is freed when it is
        // next put back.
        drop(_guard);
        drop(PackBuffer::<f64>::take(PackSlot::B, 1));
        assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), 0);
    }
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
    assert_eq!(pack_cache_bytes(), parked_bytes());
}

#[test]
fn test_pack_cache_limit_nested_guards_restore_the_previous_limit() {
    // Each guard restores the limit that was in force when it was created,
    // not the default: dropping the inner guard brings back the outer
    // guard's limit, dropping the outer one the default.
    assert_eq!(pack_cache_limit(), PACK_CACHE_MAX_BYTES);
    let outer = set_pack_cache_limit(4 << 10);
    assert_eq!(pack_cache_limit(), 4 << 10);
    let inner = set_pack_cache_limit(16 << 10);
    assert_eq!(pack_cache_limit(), 16 << 10);
    drop(inner);
    assert_eq!(pack_cache_limit(), 4 << 10, "inner guard restores 4 KiB");
    drop(outer);
    assert_eq!(
        pack_cache_limit(),
        PACK_CACHE_MAX_BYTES,
        "outer guard restores the default"
    );
}

#[test]
fn test_pack_buffer_allocations_count_growth_only() {
    let before = pack_cache_allocations();
    drop(PackBuffer::<f32>::take(PackSlot::A, 4096));
    let grown = pack_cache_allocations();
    assert!(grown <= before + 1);
    // Reusing the parked buffer (same or smaller length) allocates nothing.
    for len in [4096usize, 100, 1] {
        drop(PackBuffer::<f32>::take(PackSlot::A, len));
        assert_eq!(pack_cache_allocations(), grown, "len {len}");
    }
    // Growing allocates exactly once.
    drop(PackBuffer::<f32>::take(PackSlot::A, 8193));
    assert_eq!(pack_cache_allocations(), grown + 1);
}

#[test]
fn test_pack_buffer_fresh_elements_are_zero() {
    let buf = PackBuffer::<Complex32>::take(PackSlot::A, 77);
    assert!(buf.iter().all(|v| v.re == 0.0 && v.im == 0.0));
}
