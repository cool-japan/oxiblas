//! The router under the adopted aarch64 policy against a model written from
//! the measured scope alone, and the narrowing against the router without
//! the storage and tier conditions.
//!
//! - [`measured_route`]: [`ADOPTED_AARCH64`] restated from what the deciding
//!   measurement timed (an Apple M3, the NEON tier, operands stored
//!   column-major): the thin route for thin products with both operands
//!   stored column-major on a tier with SIMD thin kernels, the small
//!   transposed copy for `TN` and `TT` products of column-major operands on
//!   the NEON small kernels, and the route of 0.2.3 ([`legacy_route`]) for
//!   every other product.
//! - [`wide_route`]: the router without the two conditions (its thin scope
//!   takes every thin product above 32³ whatever its storage and tier; its
//!   a-only copy takes every product whose `op(A)` is not read in place).
//!   The router must give, for every input and policy, either that route
//!   or the route of [`LEGACY_POLICY`], and exactly that route for
//!   column-major operands on a tier with SIMD thin kernels.
//!
//! The grid: `m, n, k` from [`SIZES`], the nine transpose pairs, the four
//! storages of each operand (column-major only, row-major only, both unit
//! strides, neither), the four tiers, and `f64`, `f32` and a complex type.

use super::super::gemm_policy::{
    ADOPTED_AARCH64, CANDIDATE_AARCH64, CANDIDATE_X86, GemmPolicy, LEGACY_POLICY, ThinPolicy,
};
use super::call_probe::{clear_route, last_route};
use super::*;
use crate::level3::gemm::{BlockingSource, gemm_planned, gemm_transposed};
use core::any::TypeId;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::MatMut;

const SIZES: [usize; 16] = [
    1, 2, 3, 4, 5, 8, 12, 16, 31, 32, 33, 64, 128, 1024, 4096, 65536,
];
const TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
const TIERS: [SimdTier; 4] = [
    SimdTier::Portable,
    SimdTier::Sse42,
    SimdTier::Neon,
    SimdTier::Avx2Fma,
];
/// `(unit column stride, unit row stride)` of one stored operand.
const STORES: [(bool, bool); 4] = [(true, false), (false, true), (true, true), (false, false)];

/// Whether `T` is a real type with SIMD kernels, as the router sees it.
fn real_of<T: 'static>() -> bool {
    let id = TypeId::of::<T>();
    id == TypeId::of::<f64>() || id == TypeId::of::<f32>()
}

/// Calls `f` once for every input of the grid of element type `T`.
fn for_grid<T: 'static>(mut f: impl FnMut(&RouteInputs)) {
    let simd_real = real_of::<T>();
    for &m in &SIZES {
        for &n in &SIZES {
            for &k in &SIZES {
                if m.checked_mul(n).and_then(|mn| mn.checked_mul(k)).is_none() {
                    continue;
                }
                for &trans_a in &TRANS {
                    for &trans_b in &TRANS {
                        for &(a_col, a_row) in &STORES {
                            for &(b_col, b_row) in &STORES {
                                for &tier in &TIERS {
                                    f(&RouteInputs {
                                        m,
                                        n,
                                        k,
                                        trans_a,
                                        trans_b,
                                        a_unit_col_stride: a_col,
                                        a_unit_row_stride: a_row,
                                        b_unit_col_stride: b_col,
                                        b_unit_row_stride: b_row,
                                        simd_real,
                                        tier,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The route of 0.2.3: the small loops up to 32³ multiply-adds (the
/// `NoTrans`/`NoTrans` ones, which include the SIMD small kernels, or the
/// transposed ones), the blocked driver above.
fn legacy_route(i: &RouteInputs) -> GemmRoute {
    let work = i.m * i.n * i.k;
    let nn = i.trans_a == Trans::NoTrans && i.trans_b == Trans::NoTrans;
    if work > 32 * 32 * 32 {
        GemmRoute::Blocked
    } else if nn {
        GemmRoute::Small
    } else {
        GemmRoute::SmallTrans
    }
}

/// [`ADOPTED_AARCH64`] restated from the measured scope.
fn measured_route(i: &RouteInputs) -> GemmRoute {
    let work = i.m * i.n * i.k;
    let both_col_major = i.a_unit_col_stride && i.b_unit_col_stride;
    let a_notrans = i.trans_a == Trans::NoTrans;
    let b_notrans = i.trans_b == Trans::NoTrans;
    // The thin route: an f64 / f32 product with one row or one column of
    // C whose large operand the GEMV can read, both operands stored
    // column-major, on NEON or AVX2 + FMA.
    let large_readable = if i.n == 1 {
        i.a_unit_col_stride || i.a_unit_row_stride
    } else {
        i.b_unit_col_stride || i.b_unit_row_stride
    };
    let thin_capable = i.simd_real && (i.m == 1 || i.n == 1) && large_readable;
    let thin_tier = matches!(i.tier, SimdTier::Neon | SimdTier::Avx2Fma);
    if thin_capable && both_col_major && thin_tier && work >= 1024 {
        let vec_len = i.m.max(i.n);
        // (least vector length, least depth) of the four small layouts,
        // each with the vector operand not transposed.
        let small_layout = if i.n == 1 {
            match (b_notrans, a_notrans) {
                (true, true) => Some((2, 2)),
                (true, false) => Some((2, 16)),
                (false, _) => None,
            }
        } else {
            match (a_notrans, b_notrans) {
                (true, true) => Some((4, 2)),
                (true, false) => Some((4, 4)),
                (false, _) => None,
            }
        };
        let in_small_scope =
            small_layout.is_some_and(|(min_vec, min_depth)| vec_len >= min_vec && i.k >= min_depth);
        if work > 32 * 32 * 32 || in_small_scope {
            return GemmRoute::Thin;
        }
    }
    // The small transposed copy: TN and TT (a conjugate transpose of a real
    // matrix is a transpose) of column-major operands, 12³ to 32³, on the
    // NEON small kernels (`m, n >= 4`).
    let neon_small = i.simd_real && i.tier == SimdTier::Neon && i.m >= 4 && i.n >= 4;
    if neon_small && !a_notrans && both_col_major && (12 * 12 * 12..=32 * 32 * 32).contains(&work) {
        return GemmRoute::SmallTransCopy;
    }
    legacy_route(i)
}

/// The scope of the thin route without the storage and tier conditions.
fn wide_thin_in_scope(i: &RouteInputs, work: usize) -> bool {
    if work > SMALL_THRESHOLD {
        return true;
    }
    if !(i.a_unit_col_stride && i.b_unit_col_stride) {
        return false;
    }
    let a_notrans = i.trans_a == Trans::NoTrans;
    let b_notrans = i.trans_b == Trans::NoTrans;
    let limits = if i.n == 1 {
        if !b_notrans {
            return false;
        }
        if a_notrans {
            THIN_COL_AXPY
        } else {
            THIN_COL_DOT
        }
    } else {
        if !a_notrans {
            return false;
        }
        if b_notrans {
            THIN_ROW_DOT
        } else {
            THIN_ROW_AXPY
        }
    };
    i.m.max(i.n) >= limits.min_vec && i.k >= limits.min_depth
}

/// The router without the storage and tier conditions.
fn wide_route(i: &RouteInputs, policy: &GemmPolicy) -> GemmRoute {
    let work = i.work();
    let nn = i.both_notrans();
    let tiny_routes_only = policy.thin != ThinPolicy::All
        && policy
            .small_trans_copy
            .is_none_or(|(lo, _)| lo >= ROUTE_TINY_WORK);
    if work < ROUTE_TINY_WORK && tiny_routes_only {
        return if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        };
    }
    let col_major = nn && i.a_unit_col_stride && i.b_unit_col_stride;
    let simd_small = i.simd_small(policy);
    if simd_small && col_major && work <= policy.small_simd_max_work {
        return GemmRoute::Small;
    }
    if (work >= THIN_MIN_WORK || policy.thin == ThinPolicy::All) && i.thin_ok() {
        if work >= THIN_MIN_WORK && wide_thin_in_scope(i, work) {
            if policy.thin != ThinPolicy::Off {
                return GemmRoute::Thin;
            }
        } else if policy.thin == ThinPolicy::All {
            return GemmRoute::Thin;
        }
    }
    let a_in_place = i.trans_a == Trans::NoTrans && i.a_unit_col_stride;
    let copy = matches!(policy.small_trans_copy, Some((lo, hi)) if lo <= work && work <= hi)
        && !(policy.small_trans_copy_a_only && a_in_place);
    if simd_small && !col_major && copy {
        return GemmRoute::SmallTransCopy;
    }
    if work <= SMALL_THRESHOLD {
        return if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        };
    }
    GemmRoute::Blocked
}

fn model_matches<T: 'static>() -> [usize; 5] {
    // Thin, SmallTransCopy, Small, SmallTrans, Blocked.
    let mut counts = [0usize; 5];
    for_grid::<T>(|i| {
        let got = choose_route(i, &ADOPTED_AARCH64);
        let want = measured_route(i);
        assert_eq!(got, want, "adopted {i:?}");
        assert_eq!(choose_route(i, &CANDIDATE_AARCH64), want, "candidate {i:?}");
        let slot = match got {
            GemmRoute::Thin => 0,
            GemmRoute::SmallTransCopy => 1,
            GemmRoute::Small => 2,
            GemmRoute::SmallTrans => 3,
            GemmRoute::Blocked => 4,
        };
        counts[slot] += 1;
    });
    counts
}

#[test]
fn test_gemm_route_storage_model_matches_router() {
    for counts in [model_matches::<f64>(), model_matches::<f32>()] {
        // Both adopted routes occur, and so does every legacy route.
        assert!(counts.iter().all(|&c| c > 0), "{counts:?}");
    }
    let complex = model_matches::<num_complex::Complex64>();
    assert_eq!(complex[..2], [0, 0], "{complex:?}");
}

#[test]
fn test_gemm_route_storage_narrowing_against_wider_rule() {
    let policies: [(&str, GemmPolicy); 5] = [
        ("legacy", LEGACY_POLICY),
        ("adopted", ADOPTED_AARCH64),
        ("candidate aarch64", CANDIDATE_AARCH64),
        ("candidate x86", CANDIDATE_X86),
        (
            "thin all",
            GemmPolicy {
                thin: ThinPolicy::All,
                ..ADOPTED_AARCH64
            },
        ),
    ];
    // Moves back to the legacy route: thin by storage, thin by tier, copy.
    let mut moved = [0usize; 3];
    let mut check = |i: &RouteInputs| {
        let legacy = choose_route(i, &LEGACY_POLICY);
        assert_eq!(legacy, legacy_route(i), "legacy {i:?}");
        let both_col_major = i.a_unit_col_stride && i.b_unit_col_stride;
        let thin_tier = matches!(i.tier, SimdTier::Neon | SimdTier::Avx2Fma);
        for (name, policy) in &policies {
            let now = choose_route(i, policy);
            let wide = wide_route(i, policy);
            assert!(
                now == wide || now == legacy,
                "{name} {i:?}: {now:?}, wider {wide:?}, legacy {legacy:?}"
            );
            if both_col_major && thin_tier {
                assert_eq!(now, wide, "{name} column-major on a thin tier {i:?}");
            }
            if policy.thin == ThinPolicy::Scoped && i.thin_ok() && !thin_tier {
                assert_eq!(now, legacy, "{name} thin off a thin tier {i:?}");
            }
            if *name == "adopted" && !both_col_major {
                assert_eq!(now, legacy, "{name} not column-major {i:?}");
            }
            if *name == "legacy" {
                assert_eq!(now, wide, "{name} unchanged {i:?}");
            }
            if policy.thin == ThinPolicy::All {
                // Every tier and storage: the tier and storage conditions
                // are the scope's, not `ThinPolicy::All`'s.
                assert_eq!(
                    now == GemmRoute::Thin,
                    wide == GemmRoute::Thin,
                    "{name} {i:?}"
                );
            }
            if *name == "candidate x86" {
                // No a-only copy: only the thin scope can narrow.
                assert!(now == wide || wide == GemmRoute::Thin, "{name} {i:?}");
            }
            if *name == "adopted" && now != wide {
                match wide {
                    GemmRoute::Thin if !both_col_major => moved[0] += 1,
                    GemmRoute::Thin => moved[1] += 1,
                    GemmRoute::SmallTransCopy => moved[2] += 1,
                    other => panic!("adopted {i:?}: moved from {other:?}"),
                }
            }
        }
    };
    for_grid::<f64>(&mut check);
    for_grid::<f32>(&mut check);
    for_grid::<num_complex::Complex64>(&mut check);
    assert!(moved.iter().all(|&c| c > 0), "{moved:?}");
}

/// An element type of the public calls.
trait Real: GemmKernel + bytemuck::Zeroable + core::fmt::Debug {
    const EPS: f64;
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
}

impl Real for f64 {
    const EPS: f64 = f64::EPSILON;
    fn of(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl Real for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn of(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// How the operands of a call are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    /// Both stored column-major (sub-matrices of larger column-major
    /// matrices: leading dimension `rows + 3`).
    ColMajor,
    /// A is the transposed view of its column-major storage, used with the
    /// other transpose flag (the same `op(A)`).
    AView,
    /// B likewise.
    BView,
}

/// The other transpose flag: `op` of the transposed view under it is `op`
/// of the stored matrix under `t` (for a real matrix `ConjTrans` is
/// `Trans`).
fn flip(t: Trans) -> Trans {
    if t == Trans::NoTrans {
        Trans::Trans
    } else {
        Trans::NoTrans
    }
}

/// A `rows × cols` matrix stored column-major with leading dimension
/// `rows + 3`; element `(i, j)` at `i + j·(rows + 3)`.
fn col_major_data(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    let ld = rows + 3;
    let mut data = vec![7.0e5; ld * cols];
    for j in 0..cols {
        for i in 0..rows {
            data[i + j * ld] = ((i * 5 + j * 11 + seed * 3) % 19) as f64 / 8.0 - 1.05;
        }
    }
    data
}

/// The operand as passed: the column-major matrix with flag `t`, or its
/// transposed view with the other flag.
fn operand<T: Real>(
    data: &[T],
    rows: usize,
    cols: usize,
    t: Trans,
    as_view: bool,
) -> (MatRef<'_, T>, Trans) {
    let ld = rows + 3;
    let view = if as_view {
        MatRef::from_strided(data, cols, rows, 1, ld)
    } else {
        MatRef::from_strided(data, rows, cols, ld, 1)
    };
    match view {
        Some(v) if as_view => (v, flip(t)),
        Some(v) => (v, t),
        None => panic!("operand out of bounds"),
    }
}

/// `op(X)(i, j)` of a column-major `rows × cols` matrix with leading
/// dimension `rows + 3` under flag `t`.
fn op_at(data: &[f64], rows: usize, t: Trans, i: usize, j: usize) -> f64 {
    let ld = rows + 3;
    if t == Trans::NoTrans {
        data[i + j * ld]
    } else {
        data[j + i * ld]
    }
}

fn c_view<T: Real>(c: &mut [T], m: usize, n: usize) -> MatMut<'_, T> {
    match MatMut::from_strided(c, m, n, m) {
        Some(v) => v,
        None => panic!("C out of bounds"),
    }
}

type Shape = ((usize, usize, usize), (Trans, Trans));

/// The shapes: thin above 32³ (every transpose pair), thin from 1024 to
/// 32³ (the four layouts with the vector operand not transposed), the copy
/// class (`TN` and `TT`, a conjugate transpose counting as one), and `NT`
/// products in the copy's range, which are not copied.
fn shapes() -> Vec<Shape> {
    use Trans::{ConjTrans as C, NoTrans as N, Trans as T};
    let mut out = Vec::new();
    let big = [
        (1, 40, 900),
        (300, 1, 130),
        (1, 1, 40000),
        (64, 1, 600),
        (1, 64, 600),
        (1, 2, 20000),
        (3000, 1, 12),
    ];
    for &s in &big {
        for t in [(N, N), (T, N), (N, T), (T, C)] {
            out.push((s, t));
        }
    }
    let small: [Shape; 16] = [
        ((64, 1, 32), (N, N)),
        ((2, 1, 600), (N, N)),
        ((100, 1, 20), (N, N)),
        ((4, 1, 300), (N, N)),
        ((64, 1, 16), (T, N)),
        ((40, 1, 100), (C, N)),
        ((2, 1, 1000), (T, N)),
        ((128, 1, 16), (T, N)),
        ((1, 64, 32), (N, N)),
        ((1, 4, 300), (N, N)),
        ((1, 200, 10), (N, N)),
        ((1, 4, 256), (N, N)),
        ((1, 64, 16), (N, T)),
        ((1, 8, 200), (N, C)),
        ((1, 4, 512), (N, T)),
        ((1, 128, 8), (N, T)),
    ];
    out.extend(small);
    let copy = [
        (12, 12, 12),
        (16, 16, 16),
        (20, 24, 30),
        (32, 32, 32),
        (4, 40, 40),
        (40, 4, 30),
    ];
    for &s in &copy {
        for t in [(T, N), (T, T), (C, N), (T, C)] {
            out.push((s, t));
        }
    }
    // In the copy's range with A not transposed: never copied (measured:
    // the copy loses there), the route of 0.2.3.
    out.extend([
        ((16, 16, 16), (N, T)),
        ((20, 24, 30), (N, C)),
        ((12, 12, 12), (N, T)),
    ]);
    out
}

/// One call of `shape` in `storage` with `alpha`, `beta` through the public
/// entry point and through a `LEGACY_POLICY` plan; checks the value of the
/// public call against a triple loop. Returns the public call's route, the
/// legacy plan's route, whether the bits are equal, and the route inputs.
fn public_call<T: Real>(
    shape: Shape,
    storage: Storage,
    (alpha, beta): (f64, f64),
) -> (Option<GemmRoute>, Option<GemmRoute>, bool, RouteInputs) {
    let ((m, n, k), (ta, tb)) = shape;
    let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
    let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
    let a64 = col_major_data(ar, ac, m + 2 * k);
    let b64 = col_major_data(br, bc, 3 * n + k);
    let a: Vec<T> = a64.iter().map(|&v| T::of(v)).collect();
    let b: Vec<T> = b64.iter().map(|&v| T::of(v)).collect();
    let (av, fa) = operand(&a, ar, ac, ta, storage == Storage::AView);
    let (bv, fb) = operand(&b, br, bc, tb, storage == Storage::BView);
    // beta == 0: C holds NaN, which must not be read.
    let c0: Vec<f64> = (0..m * n)
        .map(|i| {
            if beta == 0.0 {
                f64::NAN
            } else {
                (i % 13) as f64 / 5.0 - 1.2
            }
        })
        .collect();
    let mut c: Vec<T> = c0.iter().map(|&v| T::of(v)).collect();
    let mut c_legacy = c.clone();
    clear_route();
    gemm_transposed(
        fa,
        fb,
        T::of(alpha),
        av,
        bv,
        T::of(beta),
        c_view(&mut c, m, n),
    );
    let route = last_route();
    clear_route();
    let legacy_plan = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
    gemm_planned(
        fa,
        fb,
        T::of(alpha),
        av,
        bv,
        T::of(beta),
        c_view(&mut c_legacy, m, n),
        Par::Seq,
        BlockingSource::Default,
        &legacy_plan,
    );
    let legacy_route = last_route();
    let what = format!("{shape:?} {storage:?} alpha {alpha} beta {beta}");
    for j in 0..n {
        for i in 0..m {
            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
            for p in 0..k {
                let prod = op_at(&a64, ar, ta, i, p) * op_at(&b64, br, tb, p, j);
                exact += prod;
                magnitude += prod.abs();
            }
            let old = if beta == 0.0 {
                0.0
            } else {
                beta * c0[i + j * m]
            };
            let reference = alpha * exact + old;
            let bound = alpha.abs() * magnitude + old.abs();
            let tol = 2.0 * (k as f64 + 2.0) * T::EPS * bound + f64::MIN_POSITIVE;
            let got = c[i + j * m].to_f64();
            assert!(
                (got - reference).abs() <= tol,
                "{what}: ({i},{j}) {got} vs {reference}"
            );
        }
    }
    let same_bits = c.iter().zip(&c_legacy).all(|(x, y)| x.bits() == y.bits());
    let tier = SimdTier::current();
    let inputs = RouteInputs::new(fa, fb, &av, &bv, m, n, k, tier);
    (route, legacy_route, same_bits, inputs)
}

/// Whether `p` and `q` have the same route fields (every field that
/// `choose_route` reads).
fn same_routes(p: &GemmPolicy, q: &GemmPolicy) -> bool {
    p.thin == q.thin
        && p.small_simd_x86 == q.small_simd_x86
        && p.small_simd_max_work == q.small_simd_max_work
        && p.small_trans_copy == q.small_trans_copy
        && p.small_trans_copy_a_only == q.small_trans_copy_a_only
}

fn public_calls<T: Real>() -> [usize; 3] {
    let scalars = [0.0, 1.0, -1.0, 0.5];
    let tier = SimdTier::current();
    // The build's default policy has both adopted routes (the default and
    // the candidate policy of aarch64), or it gives the routes of 0.2.3
    // here: the legacy policy, and the x86-64 candidate off the AVX2 + FMA
    // tier (the caller skips that candidate on its tier).
    let adopted_policy = same_routes(&DEFAULT_POLICY, &ADOPTED_AARCH64);
    assert!(
        adopted_policy || same_routes(&DEFAULT_POLICY, &LEGACY_POLICY) || tier != SimdTier::Avx2Fma,
        "a default policy this test has no model of"
    );
    if cfg!(all(
        target_arch = "aarch64",
        not(feature = "force-scalar"),
        not(oxiblas_gemm_legacy)
    )) {
        assert!(adopted_policy, "the default policy of aarch64");
    }
    // Column-major calls on the NEON tier under the adopted policy: thin,
    // copied, and the route of 0.2.3.
    let mut adopted = [0usize; 3];
    for shape in shapes() {
        for storage in [Storage::ColMajor, Storage::AView, Storage::BView] {
            for &alpha in &scalars {
                for &beta in &scalars {
                    let (route, legacy, same_bits, inputs) =
                        public_call::<T>(shape, storage, (alpha, beta));
                    let what = format!("{shape:?} {storage:?} alpha {alpha} beta {beta}");
                    let col_major = inputs.a_unit_col_stride && inputs.b_unit_col_stride;
                    assert_eq!(col_major, storage == Storage::ColMajor, "{what}");
                    if alpha == 0.0 {
                        // A quick return before any route decision.
                        assert_eq!((route, legacy), (None, None), "{what}");
                        assert!(same_bits, "{what}");
                        continue;
                    }
                    assert_eq!(legacy, Some(legacy_route(&inputs)), "{what}");
                    if storage == Storage::ColMajor {
                        let want = if adopted_policy {
                            measured_route(&inputs)
                        } else {
                            legacy_route(&inputs)
                        };
                        assert_eq!(route, Some(want), "{what} tier {tier:?}");
                        if want == legacy_route(&inputs) {
                            // Outside the adopted routes (another tier, or
                            // a build without them): the bits of 0.2.3.
                            assert!(same_bits, "{what}: bits differ from LEGACY_POLICY");
                        }
                        if adopted_policy && tier == SimdTier::Neon {
                            match want {
                                GemmRoute::Thin => adopted[0] += 1,
                                GemmRoute::SmallTransCopy => adopted[1] += 1,
                                _ => adopted[2] += 1,
                            }
                        }
                    } else {
                        // A view operand: the route and the bits of 0.2.3.
                        assert_eq!(route, legacy, "{what} tier {tier:?}");
                        assert!(same_bits, "{what}: bits differ from LEGACY_POLICY");
                    }
                }
            }
        }
    }
    adopted
}

/// Public calls of [`shapes`] in the three storages. The models of this
/// file describe two default policies: the adopted aarch64 policy (the
/// default and the candidate of aarch64) and the routes of 0.2.3 (every
/// other default build and the legacy cfg). The x86-64 candidate
/// (`--cfg oxiblas_gemm_candidate` on x86-64) has routes of its own on the
/// AVX2 + FMA tier, among them the copy of products with a view as an
/// operand: its public calls are checked against the stated rule of that
/// build in `route_public_check_tests`, and this test skips there. Off
/// that tier the candidate gives the routes of 0.2.3 and the test runs.
#[test]
fn test_gemm_route_storage_public_views_keep_legacy_bits() {
    let shapes = shapes();
    assert!(shapes.len() >= 60, "{}", shapes.len());
    let tier = SimdTier::current();
    println!(
        "tier {tier:?}, default policy thin {:?}",
        DEFAULT_POLICY.thin
    );
    if !(same_routes(&DEFAULT_POLICY, &ADOPTED_AARCH64)
        || same_routes(&DEFAULT_POLICY, &LEGACY_POLICY))
    {
        // Only the x86-64 candidate is such a policy.
        if !cfg!(all(oxiblas_gemm_candidate, target_arch = "x86_64")) {
            panic!("a default policy that is neither the adopted nor the legacy one");
        }
        if tier == SimdTier::Avx2Fma {
            println!(
                "test_gemm_route_storage_public_views_keep_legacy_bits: SKIPPED (the x86-64 candidate policy on the AVX2 + FMA tier)"
            );
            return;
        }
    }
    for adopted in [public_calls::<f64>(), public_calls::<f32>()] {
        if DEFAULT_POLICY.thin == ThinPolicy::Scoped && tier == SimdTier::Neon {
            // 12 alpha/beta pairs with alpha != 0 per shape.
            assert_eq!(adopted, [44 * 12, 24 * 12, 3 * 12], "{adopted:?}");
        } else {
            assert_eq!(adopted, [0, 0, 0], "{adopted:?}");
        }
    }
}
