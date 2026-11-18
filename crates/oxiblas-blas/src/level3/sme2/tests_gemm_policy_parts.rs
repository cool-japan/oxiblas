//! Tests of the driver policy across parts: the thresholds that
//! `parallel.rs` applies per part, with several parts on both sides of
//! them, and the online-packing and few-rows paths that
//! `tests_gemm_policy.rs` reaches with one part only.
//!
//! - The reuse limit is applied to the rows of a part
//!   (`round_up(part rows, 64) · kc · elem <= 4 MiB`), and the B cap to the
//!   columns of a part (`kc ×` the part's columns): a call whose whole A copy
//!   or whole packed B is above the limit but whose parts are below it runs
//!   each part with the per-part choice. The pure functions pin the rule at
//!   a KC that does not divide 4 MiB (the rounding of the rows to whole
//!   panels matters there), and the layout pins it for one and two parts.
//! - Through `try_gemm` (SME2 code: `backend_or_skip`): the same calls with
//!   one, two and three parts, bit for bit the NEON blocked route, the
//!   layout against the harness's spec and every part's policy record
//!   against the spec's walk (`assert_matches_neon`), plus the branch each
//!   test is about counted in every part.
//!
//! Every counter is read after the gate call of the test, on the test's
//! own thread.

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::Par;

#[cfg(feature = "parallel")]
use super::detect::override_units;
use super::driver::{A_REUSE_BYTES, AMode, Route, Shape, Tuning, a_reuse_fits};
#[cfg(feature = "parallel")]
use super::driver::{PolicyCount, TuningOverride, override_tuning};
use super::parallel::{Layout, Split};
#[cfg(feature = "parallel")]
use super::test_support::backend_or_skip;
#[cfg(feature = "parallel")]
use super::tests_gemm_harness::{Case, Outcome, Store, TestElem, assert_matches_neon, run_case};
use crate::level3::trsm::Trans;

/// A blocked shape whose A is `op(A) = A`, column-major with leading
/// dimension `lda`.
fn no_trans(m: usize, n: usize, kc: usize, lda: usize) -> Shape {
    Shape {
        m,
        n,
        kc,
        a_trans: Trans::NoTrans,
        lda,
        route: Route::Blocked,
    }
}

/// The policy records of every part of `out`.
#[cfg(feature = "parallel")]
fn part_policies<T>(out: &Outcome<T>) -> Vec<PolicyCount> {
    out.report
        .as_ref()
        .expect("run report")
        .blocks
        .iter()
        .map(|blocks| blocks.policy)
        .collect()
}

// ---------------------------------------------------------------------------
// Pure functions (run everywhere)
// ---------------------------------------------------------------------------

/// The reuse limit rounds the rows up to whole 64-row panels: at
/// `kc = 2000` (which does not divide 4 MiB) 257 rows are 4 112 000 bytes
/// of `f64` as rows but 5 120 000 as panels, so they are not reused; 256
/// rows are 4 096 000 bytes and are. `f32` likewise at 512 / 513 rows.
#[test]
fn test_sme2_gemm_policy_parts_reuse_rounds_rows_to_panels() {
    for (rows, fits) in [(192, true), (256, true), (257, false), (262, false)] {
        assert_eq!(
            a_reuse_fits(rows, 2000, 8, A_REUSE_BYTES),
            fits,
            "f64 rows {rows} at kc 2000"
        );
    }
    for (rows, fits) in [(448, true), (512, true), (513, false), (524, false)] {
        assert_eq!(
            a_reuse_fits(rows, 2000, 4, A_REUSE_BYTES),
            fits,
            "f32 rows {rows} at kc 2000"
        );
    }
    // Through the layout: 257 rows of f64 at kc 2000 (lda 260: online) are
    // rebuilt, 256 reused.
    let tuning = Tuning::of::<f64>();
    for (m, mode) in [(256, AMode::OnlineReuse), (257, AMode::OnlineRebuild)] {
        let layout = Layout::new::<f64>(&no_trans(m, 17, 2000, m + 3), 1, &tuning).expect("layout");
        assert_eq!(layout.parts[0].a_mode, mode, "f64 m {m} at kc 2000");
    }
}

/// The reuse limit is the part's: a row split of 384 rows of `f64` at
/// `kc = 2048` (6 MiB as one copy, rebuilt) into two parts of 192 rows
/// (3 MiB each, reused) and three of 128 (2 MiB each, reused).
#[test]
fn test_sme2_gemm_policy_parts_reuse_limit_per_part() {
    let tuning = Tuning::of::<f64>();
    let shape = no_trans(384, 17, 2048, 387);
    let one = Layout::new::<f64>(&shape, 1, &tuning).expect("layout");
    assert_eq!(one.parts.len(), 1);
    assert_eq!(one.parts[0].a_mode, AMode::OnlineRebuild, "one part");
    assert_eq!(one.parts[0].a_len, 384 * 2048, "one part: an MC-row buffer");
    for (parts, rows) in [(2, 192), (3, 128)] {
        let layout = Layout::new::<f64>(&shape, parts, &tuning).expect("layout");
        assert_eq!(layout.split, Split::Rows, "{parts} parts");
        assert_eq!(layout.parts.len(), parts, "{parts} parts");
        for (index, part) in layout.parts.iter().enumerate() {
            assert_eq!(part.rows.len, rows, "{parts} parts: part {index} rows");
            assert_eq!(
                (part.a_mode, part.a_len),
                (AMode::OnlineReuse, rows * 2048),
                "{parts} parts: part {index}"
            );
        }
    }
}

/// The B cap is the part's: 1500 columns of `f64` at `kc = 512` (6 MiB of
/// packed B, one MC block of 33 rows) take the 2 MiB cap as one part, but
/// each of two parts of 752 / 748 columns (3 MiB) keeps the 4 MiB cap and
/// one column block.
#[test]
fn test_sme2_gemm_policy_parts_b_cap_per_part() {
    let tuning = Tuning::of::<f64>();
    let shape = Shape::blocked(33, 1500, 512);
    let one = Layout::new::<f64>(&shape, 1, &tuning).expect("layout");
    assert_eq!(one.parts[0].b_cap, 2 << 20, "one part");
    assert_eq!(
        1500_usize.div_ceil(one.parts[0].nc),
        3,
        "one part: column blocks"
    );
    let two = Layout::new::<f64>(&shape, 2, &tuning).expect("layout");
    assert_eq!(two.split, Split::Cols);
    let widths: Vec<usize> = two.parts.iter().map(|part| part.cols.len).collect();
    assert_eq!(widths, [752, 748]);
    for (index, part) in two.parts.iter().enumerate() {
        assert_eq!(part.b_cap, 4 << 20, "two parts: part {index}");
        assert_eq!(
            part.nc, part.cols.len,
            "two parts: part {index}: one column block"
        );
    }
}

// ---------------------------------------------------------------------------
// Through `try_gemm` (SME2 code)
// ---------------------------------------------------------------------------

/// The reuse limit per part as real calls: one part above the limit
/// (rebuilt), two (and for `f64` three) parts below it (each reused, the
/// reused copy read by the second column block), all bit for bit NEON.
/// `f64`: 384 rows at kc 2048; `f32`: 768 rows at kc 2048 (6 MiB as one
/// copy, 3 MiB per part of two).
#[cfg(feature = "parallel")]
fn reuse_limit_per_part<T: TestElem>(test: &str, m: usize, pars: &[(Par, usize)]) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(3);
    let elem = core::mem::size_of::<T>();
    let kc = 2048;
    // NC = NR: two column blocks (the reused copy is read by the second).
    let _t = override_tuning(TuningOverride {
        par_min_macs: Some(0),
        b_bytes: Some(kc * elem * T::NR),
        b_one_block_bytes: Some(kc * elem * T::NR),
        ..TuningOverride::default()
    });
    for &(par, parts) in pars {
        let case = Case::<T>::new(m, T::NR + 1, kc)
            .explicit_kc(kc)
            .scalars(1.0, -0.5)
            .par(par);
        let out = run_case(&case);
        let label = format!("{test} {par:?}");
        assert_eq!(out.kc, kc, "{label}");
        assert_matches_neon(&out, &label);
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts.len(), parts, "{label}: parts");
        let policies = part_policies(&out);
        for (index, (part, p)) in report.layout.parts.iter().zip(&policies).enumerate() {
            let label = format!("{label}: part {index} ({} rows) {p:?}", part.rows.len);
            if parts == 1 {
                assert_eq!(part.a_mode, AMode::OnlineRebuild, "{label}");
                assert!(p.a_online_rebuilt > 0 && p.a_reused == 0, "{label}");
            } else {
                assert_eq!(part.a_mode, AMode::OnlineReuse, "{label}");
                assert!(p.a_online_fill > 0 && p.a_reused > 0, "{label}");
                assert_eq!(p.a_online_rebuilt, 0, "{label}");
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f64_reuse_limit_per_part() {
    reuse_limit_per_part::<f64>(
        "test_sme2_gemm_policy_parts_f64_reuse_limit_per_part",
        384,
        &[
            (Par::Seq, 1),
            (Par::RayonWith(2), 2),
            (Par::RayonWith(3), 3),
        ],
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f32_reuse_limit_per_part() {
    reuse_limit_per_part::<f32>(
        "test_sme2_gemm_policy_parts_f32_reuse_limit_per_part",
        768,
        &[(Par::Seq, 1), (Par::RayonWith(2), 2)],
    );
}

/// The B cap per part as real calls: a part of one MC block whose packed B
/// of a depth block is above 4 MiB takes the 2 MiB cap (three column
/// blocks); the same call in two column parts keeps 4 MiB in each (one
/// column block each). `f64`: 33 × 1500 at kc 512; `f32`: 49 × 1500 at
/// kc 1024.
#[cfg(feature = "parallel")]
fn b_cap_per_part<T: TestElem>(test: &str, m: usize, kc: usize) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(2);
    let _t = override_tuning(TuningOverride {
        par_min_macs: Some(0),
        ..TuningOverride::default()
    });
    let n = 1500;
    for (par, parts, cap, blocks) in [
        (Par::Seq, 1, 2 << 20, 3),
        (Par::RayonWith(2), 2, 4 << 20, 1),
    ] {
        let case = Case::<T>::new(m, n, kc)
            .explicit_kc(kc)
            .trans(Trans::NoTrans, Trans::Trans)
            .stores(Store::ColMajor { pad: 2 }, Store::RowMajor { pad: 1 })
            .scalars(0.5, 0.0)
            .nan_c()
            .par(par);
        let out = run_case(&case);
        let label = format!("{test} {par:?}");
        assert_eq!(out.kc, kc, "{label}");
        assert_matches_neon(&out, &label);
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts.len(), parts, "{label}: parts");
        for (index, part) in report.layout.parts.iter().enumerate() {
            assert_eq!(part.b_cap, cap, "{label}: part {index} cap");
            assert_eq!(
                part.cols.len.div_ceil(part.nc),
                blocks,
                "{label}: part {index} column blocks"
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f64_b_cap_per_part() {
    b_cap_per_part::<f64>("test_sme2_gemm_policy_parts_f64_b_cap_per_part", 33, 512);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f32_b_cap_per_part() {
    b_cap_per_part::<f32>("test_sme2_gemm_policy_parts_f32_b_cap_per_part", 49, 1024);
}

/// Online packing reused across jc in every part of a column split (each
/// part holds all rows: three row blocks of MC 64, NC = NR, three depth
/// blocks with a short last one), and rebuilt likewise; bit for bit NEON
/// for one to three parts, the reused copy read in every part.
#[cfg(feature = "parallel")]
fn online_column_parts<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(3);
    let elem = core::mem::size_of::<T>();
    // 163 columns: 21 panels of f64, 11 of f32; more columns than rows.
    let (m, n) = (130, 163);
    for reuse in [true, false] {
        let _t = override_tuning(TuningOverride {
            par_min_macs: Some(0),
            in_place_bytes: Some(0),
            a_reuse_bytes: if reuse { None } else { Some(0) },
            mc_target: Some(64),
            b_bytes: Some(40 * elem * T::NR),
            b_one_block_bytes: Some(40 * elem * T::NR),
            ..TuningOverride::default()
        });
        for (par, parts) in [
            (Par::Seq, 1),
            (Par::RayonWith(2), 2),
            (Par::RayonWith(3), 3),
        ] {
            let case = Case::<T>::new(m, n, 100)
                .explicit_kc(40)
                .stores(Store::ColMajor { pad: 7 }, Store::ColMajor { pad: 0 })
                .scalars(-1.0, 1.0)
                .c_guard(64)
                .par(par);
            let out = run_case(&case);
            let label = format!("{test} reuse {reuse} {par:?}");
            assert_eq!(out.kc, 40, "{label}");
            assert_matches_neon(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert_eq!(report.layout.parts.len(), parts, "{label}: parts");
            if parts > 1 {
                assert_eq!(report.layout.split, Split::Cols, "{label}");
            }
            for (index, (part, p)) in report
                .layout
                .parts
                .iter()
                .zip(part_policies(&out))
                .enumerate()
            {
                let label = format!("{label}: part {index} {p:?}");
                assert_eq!(part.rows.len, m, "{label}");
                if reuse {
                    assert_eq!(part.a_mode, AMode::OnlineReuse, "{label}");
                    assert!(p.a_online_fill > 0 && p.a_reused > 0, "{label}");
                } else {
                    assert_eq!(part.a_mode, AMode::OnlineRebuild, "{label}");
                    assert!(p.a_online_rebuilt > 0 && p.a_reused == 0, "{label}");
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f32_online_column_parts() {
    online_column_parts::<f32>("test_sme2_gemm_policy_parts_f32_online_column_parts");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f64_online_column_parts() {
    online_column_parts::<f64>("test_sme2_gemm_policy_parts_f64_online_column_parts");
}

/// Few rows with a square `op(A)` (`m == k`): the stored `k × m` matrix of
/// `op(A) = Aᵀ` has the shape of the `m × k` view the kernel takes, so a
/// route that handed it over instead of the copy would be accepted and give
/// `A · op(B)` in place of `Aᵀ · op(B)`. Both orientations of A, one to
/// three parts. (One depth block: a KC of at least 32 that splits a depth
/// of at most 48 is fitted below 32, which declines.)
#[cfg(feature = "parallel")]
fn few_rows_square<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(3);
    let _t = override_tuning(TuningOverride {
        par_min_macs: Some(0),
        ..TuningOverride::default()
    });
    let m = T::FEW_ROWS - 1;
    for trans_a in [Trans::Trans, Trans::NoTrans] {
        for (par, parts) in [
            (Par::Seq, 1),
            (Par::RayonWith(2), 2),
            (Par::RayonWith(3), 3),
        ] {
            let case = Case::<T>::new(m, 5 * T::NR + 1, m)
                .trans(trans_a, Trans::NoTrans)
                .stores(Store::ColMajor { pad: 0 }, Store::ColMajor { pad: 2 })
                .par(par);
            let out = run_case(&case);
            let label = format!("{test} {trans_a:?} {par:?}");
            assert_eq!(out.kc, m, "{label}");
            assert_matches_neon(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert_eq!(
                report.route,
                Route::FewRows {
                    copied: trans_a != Trans::NoTrans
                },
                "{label}"
            );
            assert_eq!(report.layout.parts.len(), parts, "{label}: parts");
            for (index, p) in part_policies(&out).iter().enumerate() {
                assert_eq!(p.few_rows, 1, "{label}: part {index}");
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f32_few_rows_square() {
    few_rows_square::<f32>("test_sme2_gemm_policy_parts_f32_few_rows_square");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_parts_f64_few_rows_square() {
    few_rows_square::<f64>("test_sme2_gemm_policy_parts_f64_few_rows_square");
}
