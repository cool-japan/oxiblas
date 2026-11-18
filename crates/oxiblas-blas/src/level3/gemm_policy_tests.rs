//! Tests of the policy constants and of the default policy per build.

use super::*;

#[test]
fn test_gemm_plan_policy_field_table() {
    // The values, field by field (numbers written out, not taken from the
    // constants under test).
    let gate = (512 * 512, 128 * 128);
    let all = [
        LEGACY_POLICY,
        ADOPTED_AARCH64,
        CANDIDATE_X86,
        CANDIDATE_AARCH64,
    ];
    for policy in all {
        assert_eq!(policy.thin_par_gate, gate);
        assert_eq!(policy.b_caps, ISSUE_23_B_CAPS);
        let threshold = policy.thin_par_threshold();
        assert_eq!(
            (threshold.min_elements, threshold.min_work_per_thread),
            gate
        );
    }
    assert_eq!(
        all.map(|p| p.thin),
        [
            ThinPolicy::Off,
            ThinPolicy::Scoped,
            ThinPolicy::Scoped,
            ThinPolicy::Scoped
        ]
    );
    assert_eq!(all.map(|p| p.small_simd_x86), [false, false, true, false]);
    assert_eq!(
        all.map(|p| p.small_simd_max_work),
        [32_768, 32_768, 512_000, 32_768]
    );
    assert_eq!(
        all.map(|p| p.small_trans_copy),
        [
            None,
            Some((1728, 32_768)),
            Some((512, 32_768)),
            Some((1728, 32_768))
        ]
    );
    assert_eq!(
        all.map(|p| p.small_trans_copy_a_only),
        [false, true, false, true]
    );
    assert_eq!(
        all.map(|p| p.f32_avx2),
        [
            F32Avx2Policy::Legacy8x8,
            F32Avx2Policy::Legacy8x8,
            F32Avx2Policy::Adaptive,
            F32Avx2Policy::Legacy8x8
        ]
    );
    assert_eq!(
        all.map(|p| p.per_tile_min_threads),
        [usize::MAX, usize::MAX, 5, usize::MAX]
    );
    let expected_cap = if cfg!(oxiblas_gemm_bcap_all) {
        BCapScope::All
    } else {
        BCapScope::Sequential
    };
    assert_eq!(CANDIDATE_B_CAP, expected_cap);
    assert_eq!(
        all.map(|p| p.b_cap),
        [BCapScope::Off, BCapScope::Off, expected_cap, expected_cap]
    );
    // The adopted policy is the legacy one with exactly three fields
    // changed, and the aarch64 candidate the adopted one with the B cap.
    let reverted = GemmPolicy {
        thin: ThinPolicy::Off,
        small_trans_copy: None,
        small_trans_copy_a_only: false,
        ..ADOPTED_AARCH64
    };
    assert_eq!(reverted, LEGACY_POLICY);
    let uncapped = GemmPolicy {
        b_cap: BCapScope::Off,
        ..CANDIDATE_AARCH64
    };
    assert_eq!(uncapped, ADOPTED_AARCH64);
    // The scope's constants.
    assert_eq!(THIN_MIN_WORK, 1024);
    let limits = |l: ThinLimits| (l.min_vec, l.min_depth);
    assert_eq!(
        [THIN_COL_AXPY, THIN_COL_DOT, THIN_ROW_DOT, THIN_ROW_AXPY].map(limits),
        [(2, 2), (2, 16), (4, 2), (4, 4)]
    );
    assert_eq!(SMALL_TRANS_COPY_FULL_RANGE, (512, 32_768));
    assert_eq!(SMALL_TRANS_COPY_A_ONLY_RANGE, (1728, 32_768));
}

/// The policy a build uses for its public entry points, by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Named {
    Legacy,
    AdoptedAarch64,
    CandidateX86,
    CandidateAarch64,
}

/// A target architecture of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    Aarch64,
    X86_64,
    /// Every other target (no candidate, nothing measured).
    Other,
}

/// `DEFAULT_POLICY` for every combination of (target, `force-scalar`,
/// `--cfg oxiblas_gemm_legacy`, `--cfg oxiblas_gemm_candidate`) that can be
/// built, stated row by row. `--cfg oxiblas_gemm_bcap_all` only changes the
/// candidates' B cap (`BCapScope::All` instead of `Sequential`), and
/// `max-simd-128` / `max-simd-256` change nothing here (NEON is 128 bits
/// wide; on x86-64 the default is the legacy policy with or without them).
const DEFAULT_TABLE: [(Arch, bool, bool, bool, Named); 24] = {
    use Arch::{Aarch64, Other, X86_64};
    use Named::{AdoptedAarch64, CandidateAarch64, CandidateX86, Legacy};
    [
        // (arch, force-scalar, legacy cfg, candidate cfg) -> default
        (Aarch64, false, false, false, AdoptedAarch64),
        (Aarch64, false, false, true, CandidateAarch64),
        (Aarch64, false, true, false, Legacy),
        (Aarch64, false, true, true, Legacy),
        (Aarch64, true, false, false, Legacy),
        (Aarch64, true, false, true, CandidateAarch64),
        (Aarch64, true, true, false, Legacy),
        (Aarch64, true, true, true, Legacy),
        (X86_64, false, false, false, Legacy),
        (X86_64, false, false, true, CandidateX86),
        (X86_64, false, true, false, Legacy),
        (X86_64, false, true, true, Legacy),
        (X86_64, true, false, false, Legacy),
        (X86_64, true, false, true, CandidateX86),
        (X86_64, true, true, false, Legacy),
        (X86_64, true, true, true, Legacy),
        (Other, false, false, false, Legacy),
        (Other, false, false, true, Legacy),
        (Other, false, true, false, Legacy),
        (Other, false, true, true, Legacy),
        (Other, true, false, false, Legacy),
        (Other, true, false, true, Legacy),
        (Other, true, true, false, Legacy),
        (Other, true, true, true, Legacy),
    ]
};

#[test]
fn test_gemm_plan_default_policy_selection() {
    let arch = if cfg!(target_arch = "aarch64") {
        Arch::Aarch64
    } else if cfg!(target_arch = "x86_64") {
        Arch::X86_64
    } else {
        Arch::Other
    };
    let build = (
        arch,
        cfg!(feature = "force-scalar"),
        cfg!(oxiblas_gemm_legacy),
        cfg!(oxiblas_gemm_candidate),
    );
    let rows: Vec<Named> = DEFAULT_TABLE
        .iter()
        .filter(|r| (r.0, r.1, r.2, r.3) == build)
        .map(|r| r.4)
        .collect();
    assert_eq!(rows.len(), 1, "{build:?}");
    let named = rows[0];
    let expected = match named {
        Named::Legacy => LEGACY_POLICY,
        Named::AdoptedAarch64 => ADOPTED_AARCH64,
        Named::CandidateX86 => CANDIDATE_X86,
        Named::CandidateAarch64 => CANDIDATE_AARCH64,
    };
    assert_eq!(DEFAULT_POLICY, expected, "{build:?} {named:?}");
    // The B cap of the default: only a candidate has one, and it covers
    // both drivers exactly with `--cfg oxiblas_gemm_bcap_all`.
    let cap = match named {
        Named::Legacy | Named::AdoptedAarch64 => BCapScope::Off,
        Named::CandidateX86 | Named::CandidateAarch64 if cfg!(oxiblas_gemm_bcap_all) => {
            BCapScope::All
        }
        Named::CandidateX86 | Named::CandidateAarch64 => BCapScope::Sequential,
    };
    assert_eq!(DEFAULT_POLICY.b_cap, cap, "{build:?}");
    // The candidate of the target, whatever the default is.
    let candidate = match arch {
        Arch::Aarch64 => CANDIDATE_AARCH64,
        Arch::X86_64 => CANDIDATE_X86,
        Arch::Other => LEGACY_POLICY,
    };
    assert_eq!(CANDIDATE_POLICY, candidate);
    // The table itself: every combination exactly once, the legacy cfg wins
    // over the candidate cfg, and only aarch64 without `force-scalar` and
    // without either cfg has the adopted policy.
    for arch in [Arch::Aarch64, Arch::X86_64, Arch::Other] {
        for scalar in [false, true] {
            for legacy in [false, true] {
                for cand in [false, true] {
                    let key = (arch, scalar, legacy, cand);
                    let found: Vec<Named> = DEFAULT_TABLE
                        .iter()
                        .filter(|r| (r.0, r.1, r.2, r.3) == key)
                        .map(|r| r.4)
                        .collect();
                    assert_eq!(found.len(), 1, "{key:?}");
                    if legacy {
                        assert_eq!(found[0], Named::Legacy, "{key:?}");
                    }
                    let adopted = arch == Arch::Aarch64 && !scalar && !legacy && !cand;
                    assert_eq!(found[0] == Named::AdoptedAarch64, adopted, "{key:?}");
                }
            }
        }
    }
    // The portable tier is the only one a context can be lowered to.
    let ctx = super::super::gemm_plan::GemmContext::detect(LEGACY_POLICY).with_portable_tier();
    assert_eq!(ctx.tier(), super::super::gemm_plan::SimdTier::Portable);
    assert_eq!(ctx.policy, LEGACY_POLICY);
}

/// The adopted policy needs the NEON tier: with `force-scalar` (the
/// portable tier, where nothing was measured) and off aarch64 the adopted
/// policy of the build is the legacy one. The policy constant follows this
/// crate's features; a build where only oxiblas-core has `force-scalar`
/// keeps the adopted policy on the portable tier, where its thin route is
/// out of scope (`RouteInputs::thin_in_scope`) and its copy has no SIMD
/// small kernels. Compiles in every feature set.
#[test]
fn test_gemm_plan_adopted_policy_needs_the_neon_tier() {
    use super::super::gemm_plan::{SimdTier, core_forces_scalar};
    let neon_build = cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")));
    if neon_build {
        assert_eq!(ADOPTED_POLICY, ADOPTED_AARCH64);
        let expected = if core_forces_scalar() {
            SimdTier::Portable
        } else {
            SimdTier::Neon
        };
        assert_eq!(SimdTier::current(), expected);
    } else {
        assert_eq!(ADOPTED_POLICY, LEGACY_POLICY);
    }
    if cfg!(feature = "force-scalar") {
        assert_eq!(SimdTier::current(), SimdTier::Portable);
        assert_eq!(ADOPTED_POLICY, LEGACY_POLICY);
        if !cfg!(oxiblas_gemm_candidate) {
            assert_eq!(DEFAULT_POLICY, LEGACY_POLICY);
        }
    }
    // Without a cfg the default is the adopted policy of the build.
    if !cfg!(oxiblas_gemm_legacy) && !cfg!(oxiblas_gemm_candidate) {
        assert_eq!(DEFAULT_POLICY, ADOPTED_POLICY);
    }
    // The legacy cfg restores 0.2.3 on every target, also over the
    // candidate cfg.
    if cfg!(oxiblas_gemm_legacy) {
        assert_eq!(DEFAULT_POLICY, LEGACY_POLICY);
    }
}
