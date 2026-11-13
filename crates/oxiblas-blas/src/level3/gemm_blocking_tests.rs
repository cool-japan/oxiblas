//! Tests for the block sizes of one GEMM call: the default blocking keeps the
//! canonical KC whatever micro-kernel a plan carries, and the B-block cap is
//! decided once, from the blocking source and the policy.

use super::*;
use crate::level3::autotune::{CacheInfo, ISSUE_23_B_CAPS};
use crate::level3::gemm_plan::{GemmPolicy, LEGACY_POLICY};

const fn shape(mr: usize, nr: usize) -> MicroKernelShape {
    MicroKernelShape { mr, nr }
}

/// Synthetic cache descriptions. The first one (32 KiB L1d with a 4 MiB L2)
/// makes the L1 bound of the cache model bind differently for an 8×8 and a
/// 16×6 f32 kernel; the last one (8 KiB L1d with a 4 MiB L2) does the same
/// for the f64 kernels, whose NR ranges from 2 to 6.
fn caches() -> [CacheInfo; 6] {
    [
        CacheInfo {
            l1d: 32 << 10,
            l2: 4 << 20,
            l3: 16 << 20,
            line_size: 64,
        },
        CacheInfo {
            l1d: 48 << 10,
            l2: (5 << 20) / 4,
            l3: 30 << 20,
            line_size: 64,
        },
        CacheInfo {
            l1d: 32 << 10,
            l2: 256 << 10,
            l3: 8 << 20,
            line_size: 64,
        },
        CacheInfo {
            l1d: 128 << 10,
            l2: 16 << 20,
            l3: 8 << 20,
            line_size: 128,
        },
        CacheInfo {
            l1d: 64 << 10,
            l2: 512 << 10,
            l3: 32 << 20,
            line_size: 64,
        },
        CacheInfo {
            l1d: 8 << 10,
            l2: 4 << 20,
            l3: 16 << 20,
            line_size: 64,
        },
    ]
}

/// `m × k × n` on both sides of the auto-tune threshold, covering every
/// branch of the adaptive rules (tiny, balanced, tall-thin, short-wide,
/// inner-product, panel-panel) and the `for_kernel` path.
const DIMS: [(usize, usize, usize); 14] = [
    (64, 64, 64),
    (300, 200, 400),
    (511, 511, 511),
    (1, 511, 1),
    (512, 512, 512),
    (1000, 850, 1000),
    (1000, 1000, 1000),
    (4096, 64, 64),
    (64, 64, 4096),
    (64, 4096, 64),
    (2048, 100, 2048),
    (512, 4, 4),
    (3000, 2000, 700),
    (520, 9000, 37),
];

/// The micro-kernel shapes of the f32 kernels on every tier (AVX-512,
/// AVX2 / NEON, SSE4.2 / scalar) plus the AVX2 16×6 candidate.
const F32_SHAPES: [MicroKernelShape; 4] = [shape(16, 16), shape(8, 8), shape(4, 4), shape(16, 6)];

/// The micro-kernel shapes of the f64 kernels on every tier (AVX-512,
/// AVX2 / NEON, SSE4.2, scalar).
const F64_SHAPES: [MicroKernelShape; 4] = [shape(16, 6), shape(8, 6), shape(4, 2), shape(4, 4)];

/// The fitted block of `dim` must be either the whole dimension or a
/// multiple of `unit`.
fn assert_fitted(block: usize, dim: usize, unit: usize, ctx: &str) {
    assert!(block >= 1, "{ctx}: empty block");
    assert!(
        block == dim || block.is_multiple_of(unit),
        "{ctx}: block {block} of {dim} is neither the dimension nor a multiple of {unit}"
    );
}

fn check_canonical_kc<T: Field>(shapes: &[MicroKernelShape]) {
    let name = core::any::type_name::<T>();
    for (ci, cache) in caches().iter().enumerate() {
        for canonical in shapes {
            for plan_shape in shapes {
                for &(m, k, n) in &DIMS {
                    let ctx = format!(
                        "{name} cache#{ci} canonical {canonical:?} plan {plan_shape:?} {m}x{k}x{n}"
                    );
                    let reference =
                        default_blocking_with::<T>(|| *cache, m, k, n, canonical, canonical);
                    let planned =
                        default_blocking_with::<T>(|| *cache, m, k, n, plan_shape, canonical);
                    assert_eq!(planned.kc, reference.kc, "{ctx}: KC moved");
                    // MC / NC follow the plan's kernel.
                    let fitted = planned.fit_to_dims(m, k, n, plan_shape);
                    assert_fitted(fitted.mc, m, plan_shape.mr, &format!("{ctx} MC"));
                    assert_fitted(fitted.nc, n, plan_shape.nr, &format!("{ctx} NC"));
                    assert_eq!(fitted.kc, reference.fit_to_dims(m, k, n, canonical).kc);
                    // With the canonical kernel the default blocking is the
                    // 0.2.3 one: `auto_tuned` from the threshold on (here on
                    // the synthetic cache), `for_kernel` below.
                    if plan_shape == canonical {
                        let expected = if m.max(k).max(n) >= AUTO_TUNE_THRESHOLD {
                            let t = crate::level3::autotune::compute_blocking_adaptive_with(
                                || *cache,
                                m,
                                k,
                                n,
                                core::mem::size_of::<T>(),
                                canonical,
                                canonical,
                            );
                            (t.mc, t.kc, t.nc)
                        } else {
                            let t = GemmBlocking::for_kernel::<T>(canonical);
                            (t.mc, t.kc, t.nc)
                        };
                        assert_eq!((planned.mc, planned.kc, planned.nc), expected, "{ctx}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_blocking_default_kc_is_canonical_for_every_plan_kernel() {
    check_canonical_kc::<f32>(&F32_SHAPES);
    check_canonical_kc::<f64>(&F64_SHAPES);
}

#[test]
fn test_gemm_blocking_canonical_kc_rule_is_not_vacuous() {
    // On a 32 KiB L1d / 4 MiB L2 cache the KC of the cache model differs
    // between the f32 8×8 and 16×6 kernels (the L1 bound binds for one of
    // them), so taking KC from the plan's kernel would move the KC partition
    // of a 1000 × 850 × 1000 product: this is what the canonical-KC rule
    // prevents.
    let cache = caches()[0];
    let (m, k, n) = (1000, 850, 1000);
    let (s8x8, s16x6) = (shape(8, 8), shape(16, 6));
    let own_8x8 = default_blocking_with::<f32>(|| cache, m, k, n, &s8x8, &s8x8);
    let own_16x6 = default_blocking_with::<f32>(|| cache, m, k, n, &s16x6, &s16x6);
    assert_ne!(
        own_8x8.kc, own_16x6.kc,
        "the synthetic cache must make the per-shape KC differ"
    );
    let planned = default_blocking_with::<f32>(|| cache, m, k, n, &s16x6, &s8x8);
    assert_eq!(planned.kc, own_8x8.kc);
    assert_ne!(planned.kc, own_16x6.kc);
    // MC is re-derived for the plan's kernel from that KC.
    assert!(planned.mc.is_multiple_of(s16x6.mr));
}

#[test]
fn test_gemm_blocking_host_default_is_0_2_3_for_the_canonical_kernel() {
    fn check<T: Field + GemmKernel>() {
        let canonical = T::micro_kernel_shape();
        for &(m, k, n) in &DIMS {
            let got = default_blocking::<T>(m, k, n, &canonical);
            let expected = if m.max(k).max(n) >= AUTO_TUNE_THRESHOLD {
                GemmBlocking::auto_tuned::<T>(m, k, n, &canonical)
            } else {
                GemmBlocking::for_kernel::<T>(&canonical)
            };
            assert_eq!(
                (got.mc, got.kc, got.nc),
                (expected.mc, expected.kc, expected.nc),
                "{} {m}x{k}x{n}",
                core::any::type_name::<T>()
            );
            // Any other kernel shape keeps the canonical KC on the host too.
            for other in [shape(16, 6), shape(4, 4), shape(8, 8)] {
                assert_eq!(default_blocking::<T>(m, k, n, &other).kc, got.kc);
            }
        }
    }
    check::<f64>();
    check::<f32>();
}

/// The KC that the default blocking of `T` uses for a plan whose kernel has
/// `shape`, on `cache`: the KC of the canonical kernel of `T`, never the one
/// the cache model would give `shape` itself. Returns how many of the checked
/// cases would have moved KC had the plan's own shape been used.
fn check_canonical_kc_wiring<T: Field + GemmKernel>(shapes: &[MicroKernelShape]) -> usize {
    let canonical = T::micro_kernel_shape();
    let mut discriminating = 0;
    for (ci, cache) in caches().iter().enumerate() {
        for plan_shape in shapes {
            for &(m, k, n) in &DIMS {
                let ctx = format!(
                    "{} cache#{ci} plan {plan_shape:?} {m}x{k}x{n}",
                    core::any::type_name::<T>()
                );
                let reference = default_blocking_in::<T>(|| *cache, m, k, n, &canonical);
                let planned = default_blocking_in::<T>(|| *cache, m, k, n, plan_shape);
                assert_eq!(planned.kc, reference.kc, "{ctx}: KC moved");
                let own = default_blocking_with::<T>(|| *cache, m, k, n, plan_shape, plan_shape);
                if own.kc != reference.kc {
                    discriminating += 1;
                }
            }
        }
    }
    discriminating
}

#[test]
fn test_gemm_blocking_default_takes_kc_from_the_canonical_kernel_of_the_type() {
    // `default_blocking` only adds the host's caches to `default_blocking_in`,
    // which is where the canonical shape of the type is chosen. Checked with
    // the synthetic caches, so the check bites on every host and in every
    // feature set: at least one case per type must have a plan shape whose
    // own KC differs from the canonical one (otherwise undoing the rule would
    // go unnoticed).
    let f32_cases = check_canonical_kc_wiring::<f32>(&F32_SHAPES);
    let f64_cases = check_canonical_kc_wiring::<f64>(&F64_SHAPES);
    assert!(f32_cases > 0, "no f32 case discriminates the canonical KC");
    assert!(f64_cases > 0, "no f64 case discriminates the canonical KC");
}

// ------------------------------------------------------------------
// The blocking source decides whether the B-block cap may apply.
// ------------------------------------------------------------------

/// A product whose packed B block is far above every cap: one MC block, two
/// KC blocks and more than 4 MiB of packed B.
const CAP_DIMS: (usize, usize, usize) = (16, 512, (4 << 20) / (256 * 8) + 452);

const CAP_TARGETS: GemmBlocking = GemmBlocking {
    mc: 64,
    kc: 256,
    nc: 1 << 20,
};

fn plan_with(b_cap: BCapScope) -> GemmPlan<f64> {
    GemmPlan::<f64>::new(GemmPolicy {
        b_cap,
        b_caps: ISSUE_23_B_CAPS,
        ..LEGACY_POLICY
    })
}

#[test]
fn test_gemm_blocking_source_explicit_is_never_capped() {
    let (m, k, n) = CAP_DIMS;
    let shape = f64::micro_kernel_shape();
    let fitted = CAP_TARGETS.fit_to_dims(m, k, n, &shape);
    let capped_nc = capped_nc(
        m,
        k,
        n,
        fitted.mc,
        fitted.kc,
        fitted.nc,
        shape.nr,
        8,
        &ISSUE_23_B_CAPS,
    );
    assert!(capped_nc < fitted.nc, "the cap must engage on these blocks");
    let same = |b: &GemmBlocking, nc: usize| (b.mc, b.kc, b.nc) == (fitted.mc, fitted.kc, nc);
    for scope in [BCapScope::Off, BCapScope::Sequential, BCapScope::All] {
        let plan = plan_with(scope);
        let explicit = call_blocking(BlockingSource::Explicit(CAP_TARGETS), &plan, m, k, n);
        assert!(
            same(explicit.sequential(), fitted.nc),
            "{scope:?} explicit seq"
        );
        let heuristic = call_blocking(BlockingSource::Heuristic(CAP_TARGETS), &plan, m, k, n);
        let expected_seq = match scope {
            BCapScope::Off => fitted.nc,
            BCapScope::Sequential | BCapScope::All => capped_nc,
        };
        assert!(
            same(heuristic.sequential(), expected_seq),
            "{scope:?} heuristic seq"
        );
        #[cfg(feature = "parallel")]
        {
            assert!(
                same(explicit.parallel(), fitted.nc),
                "{scope:?} explicit par"
            );
            let expected_par = match scope {
                BCapScope::Off | BCapScope::Sequential => fitted.nc,
                BCapScope::All => capped_nc,
            };
            assert!(
                same(heuristic.parallel(), expected_par),
                "{scope:?} heuristic par"
            );
        }
    }
}

#[test]
fn test_gemm_blocking_source_default_is_cap_eligible() {
    // Default blocking: the default targets of the plan's kernel, fitted,
    // then capped by the rule exactly as heuristic targets are.
    let shape = f64::micro_kernel_shape();
    for (m, k, n) in [
        CAP_DIMS,
        (16, 4096, 65_536),
        (300, 512, 40_000),
        (64, 64, 64),
    ] {
        let fitted = default_blocking::<f64>(m, k, n, &shape).fit_to_dims(m, k, n, &shape);
        let rule = capped_nc(
            m,
            k,
            n,
            fitted.mc,
            fitted.kc,
            fitted.nc,
            shape.nr,
            8,
            &ISSUE_23_B_CAPS,
        );
        for scope in [BCapScope::Off, BCapScope::Sequential, BCapScope::All] {
            let blocks = call_blocking(BlockingSource::Default, &plan_with(scope), m, k, n);
            let seq = blocks.sequential();
            assert_eq!((seq.mc, seq.kc), (fitted.mc, fitted.kc));
            let expected = if scope == BCapScope::Off {
                fitted.nc
            } else {
                rule
            };
            assert_eq!(seq.nc, expected, "{m}x{k}x{n} {scope:?}");
        }
    }
    // Under the legacy policy nothing is capped, whatever the source.
    let legacy = GemmPlan::<f64>::new(LEGACY_POLICY);
    let (m, k, n) = CAP_DIMS;
    for source in [
        BlockingSource::Default,
        BlockingSource::Heuristic(CAP_TARGETS),
        BlockingSource::Explicit(CAP_TARGETS),
    ] {
        let blocks = call_blocking(source, &legacy, m, k, n);
        let targets = match source {
            BlockingSource::Default => default_blocking::<f64>(m, k, n, &shape),
            BlockingSource::Heuristic(t) | BlockingSource::Explicit(t) => t,
        };
        let fitted = targets.fit_to_dims(m, k, n, &shape);
        let seq = blocks.sequential();
        assert_eq!((seq.mc, seq.kc, seq.nc), (fitted.mc, fitted.kc, fitted.nc));
    }
}

#[test]
fn test_gemm_blocking_source_never_moves_result_bits() {
    // The source only decides NC (through the cap): explicit and heuristic
    // targets under every capping scope give the bits of the uncapped legacy
    // run, sequentially and (with `parallel`) through the 2D driver, which
    // gets the capped NC under `All`.
    let (m, k, n) = CAP_DIMS;
    let targets = CAP_TARGETS;
    let shape = f64::micro_kernel_shape();
    let fitted = targets.fit_to_dims(m, k, n, &shape);
    for scope in [BCapScope::Sequential, BCapScope::All] {
        let blocks = call_blocking(
            BlockingSource::Heuristic(targets),
            &plan_with(scope),
            m,
            k,
            n,
        );
        assert!(
            blocks.sequential().nc < fitted.nc,
            "{scope:?}: the cap must engage on these blocks"
        );
        #[cfg(feature = "parallel")]
        if scope == BCapScope::All {
            assert!(blocks.parallel().nc < fitted.nc, "All: parallel NC capped");
        }
    }
    let fill = |rows: usize, cols: usize, seed: usize| {
        let mut out: Mat<f64> = Mat::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                out[(i, j)] = ((i * 31 + j * 17 + seed) % 23) as f64 / 7.0 - 1.5;
            }
        }
        out
    };
    let a = fill(m, k, 1);
    let b = fill(k, n, 5);
    let run = |source: BlockingSource, plan: &GemmPlan<f64>, par: Par| {
        let mut c: Mat<f64> = Mat::filled(m, n, f64::NAN);
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            0.5,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
            par,
            source,
            plan,
        );
        c
    };
    let reference = run(
        BlockingSource::Explicit(targets),
        &GemmPlan::<f64>::new(LEGACY_POLICY),
        Par::Seq,
    );
    #[cfg(feature = "parallel")]
    let pars = [Par::Seq, Par::RayonWith(4)];
    #[cfg(not(feature = "parallel"))]
    let pars = [Par::Seq];
    for scope in [BCapScope::Sequential, BCapScope::All] {
        let plan = plan_with(scope);
        for source in [
            BlockingSource::Explicit(targets),
            BlockingSource::Heuristic(targets),
        ] {
            for par in pars {
                let c = run(source, &plan, par);
                for j in 0..n {
                    for i in 0..m {
                        assert_eq!(
                            c[(i, j)].to_bits(),
                            reference[(i, j)].to_bits(),
                            "{scope:?} {source:?} {par:?} ({i},{j})"
                        );
                    }
                }
            }
        }
    }
}
