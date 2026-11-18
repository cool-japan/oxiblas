//! Public GEMM calls over the scope table, the copy rule and their limits:
//! route, value, parallel bits.
//!
//! For each sample (a shape, a transpose pair, the storage of A and B) and
//! each `alpha, beta` in `{0, 1, -1, 0.5}`, `f64` and `f32`:
//!
//! - the route the public call took (`call_probe::last_route`) is the one
//!   the stated rule of this build gives (`route_model_tests::stated_route`;
//!   the rule of the build is chosen here from the cfg table, not read from
//!   `DEFAULT_POLICY`; the routes follow the tier of the call), and a call
//!   with `alpha == 0` takes no route;
//! - under the adopted rule a call with an operand not stored column-major
//!   takes the route of 0.2.3, and the thin route is taken only with both
//!   operands stored column-major on a tier with SIMD thin kernels;
//! - the result is within `γ(k + 2)·(|alpha|·Σ|a·b| + |beta·c|)` of a
//!   double-double reference (`γ(n) = n·u / (1 - n·u)`, `u` half the
//!   machine epsilon of the element type), plus the reference's own
//!   rounding to `f64`: every route computes each element as a sum of `k`
//!   products in some order (fused or not), scaled by `alpha` and added to
//!   `beta·c`, so each product passes through at most `k + 2` roundings;
//! - with `beta == 0` C is filled with NaN first and must not be read;
//! - C has a padded leading dimension whose padding must stay untouched;
//! - a call with `Par::Rayon` gives the bits of the sequential call;
//! - a call that the rule sends to a route of 0.2.3 gives the bits of a
//!   plan of `LEGACY_POLICY` (but for the x86-64 candidate's small route on
//!   the AVX2 + FMA tier, which runs that tier's small kernels).
//!
//! Storage: column-major with a padded leading dimension (`Col`), a
//! transposed view of padded column-major storage (`Row`), and a view with
//! no unit stride (`Neither`).

use super::call_probe::{clear_route, last_route};
use super::route_model_tests::{
    RULES_ADOPTED, RULES_CANDIDATE_X86, RULES_LEGACY, Rules, stated_route,
};
use super::route_tests::simd_thin_tier;
use super::*;
use crate::level3::gemm::{BlockingSource, gemm, gemm_planned, gemm_transposed_with_par, read_op};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

/// `f64` / `f32`.
trait Elem: GemmKernel + bytemuck::Zeroable + core::fmt::Debug {
    /// Machine epsilon.
    const EPS: f64;
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
}

impl Elem for f64 {
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

impl Elem for f32 {
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

/// The stated rule of this build, from the cfg table: the legacy cfg wins,
/// then the candidate cfg (the aarch64 candidate routes as the adopted
/// policy), then aarch64 without `force-scalar` has the adopted rule.
fn build_rules() -> Rules {
    if cfg!(oxiblas_gemm_legacy) {
        RULES_LEGACY
    } else if cfg!(oxiblas_gemm_candidate) {
        if cfg!(target_arch = "x86_64") {
            RULES_CANDIDATE_X86
        } else if cfg!(target_arch = "aarch64") {
            RULES_ADOPTED
        } else {
            RULES_LEGACY
        }
    } else if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))) {
        RULES_ADOPTED
    } else {
        RULES_LEGACY
    }
}

/// How a stored operand is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Store {
    /// Column-major, leading dimension `rows + 3`.
    Col,
    /// A `transpose()`-like view: adjacent columns, row distance `cols + 2`.
    Row,
    /// No unit stride: rows 2 apart, columns `2·rows + 1` apart.
    Neither,
}

/// A value with rounding in its sums: mantissas of about 10 bits over five
/// binades, both signs.
fn value(i: usize, j: usize, seed: usize) -> f64 {
    let h = (i.wrapping_mul(2_654_435_761) ^ j.wrapping_mul(40_503) ^ seed.wrapping_mul(97_531))
        .wrapping_mul(2_246_822_519)
        >> 7;
    let mant = (h % 2001) as f64 - 1000.0;
    let scale = [0.25, 0.5, 1.0, 2.0, 4.0][(h / 2001) % 5];
    mant / 997.0 * scale
}

/// A stored `rows × cols` matrix with the values of `seed`.
struct Stored<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    store: Store,
}

impl<T: Elem> Stored<T> {
    fn new(rows: usize, cols: usize, store: Store, seed: usize) -> Self {
        let (rs, cs) = Self::strides(rows, cols, store);
        let len = (rows.max(1) - 1) * cs + (cols.max(1) - 1) * rs + 1;
        // Fill the gaps with a large value: a route that read one would be
        // far outside the error bound.
        let mut data = vec![T::of(1.0e6); len];
        for j in 0..cols {
            for i in 0..rows {
                data[i * cs + j * rs] = T::of(value(i, j, seed));
            }
        }
        Self {
            data,
            rows,
            cols,
            store,
        }
    }

    /// `(row_stride, col_stride)`: the distance of two columns, of two rows.
    fn strides(rows: usize, cols: usize, store: Store) -> (usize, usize) {
        match store {
            Store::Col => (rows + 3, 1),
            Store::Row => (1, cols + 2),
            Store::Neither => (2 * rows + 1, 2),
        }
    }

    fn view(&self) -> MatRef<'_, T> {
        let (rs, cs) = Self::strides(self.rows, self.cols, self.store);
        match MatRef::from_strided(&self.data, self.rows, self.cols, rs, cs) {
            Some(v) => v,
            None => panic!("view out of bounds"),
        }
    }
}

/// `op(X)[i, p]` of a stored operand.
fn op_at<T: Elem>(x: &MatRef<'_, T>, trans: Trans, i: usize, p: usize) -> f64 {
    read_op(x, trans, i, p).to_f64()
}

/// A double-double sum.
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    fn add(self, x: f64) -> Self {
        let s = self.hi + x;
        let bb = s - self.hi;
        let e = (self.hi - (s - bb)) + (x - bb) + self.lo;
        let hi = s + e;
        Self {
            hi,
            lo: e - (hi - s),
        }
    }

    /// Adds `a·b` exactly (the product's error from a fused multiply-add).
    fn add_prod(self, a: f64, b: f64) -> Self {
        let p = a * b;
        self.add(p).add(a.mul_add(b, -p))
    }
}

/// One element's exact sum `Σ_p op(A)[i,p]·op(B)[p,j]` (double-double) and
/// `Σ_p |op(A)[i,p]·op(B)[p,j]|`.
fn sums<T: Elem>(
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    (ta, tb): (Trans, Trans),
    k: usize,
    (i, j): (usize, usize),
) -> (Dd, f64) {
    let mut s = Dd { hi: 0.0, lo: 0.0 };
    let mut mag = 0.0f64;
    for p in 0..k {
        let (x, y) = (op_at(a, ta, i, p), op_at(b, tb, p, j));
        s = s.add_prod(x, y);
        mag += (x * y).abs();
    }
    (s, mag)
}

/// `γ(n)` of the element type.
fn gamma<T: Elem>(n: usize) -> f64 {
    let nu = n as f64 * T::EPS / 2.0;
    nu / (1.0 - nu)
}

/// One sample.
type Sample = ((usize, usize, usize), (Trans, Trans), (Store, Store));

/// The C of a call: `m × n` with leading dimension `m + 2`, padding 7.0.
fn c_storage<T: Elem>(m: usize, n: usize, beta_zero: bool) -> Vec<T> {
    let ld = m + 2;
    let mut data = vec![T::of(7.0); ld * n];
    for j in 0..n {
        for i in 0..m {
            data[i + j * ld] = if beta_zero {
                T::of(f64::NAN)
            } else {
                T::of(value(i, j, 99))
            };
        }
    }
    data
}

fn c_view<T: Elem>(data: &mut [T], m: usize, n: usize) -> MatMut<'_, T> {
    match MatMut::from_strided(data, m, n, m + 2) {
        Some(v) => v,
        None => panic!("C out of bounds"),
    }
}

/// The routes taken by the checked calls (alpha != 0), by kind.
type RouteCounts = [usize; 5];

/// Checks one sample for `T` under every `alpha, beta`.
fn check_sample<T: Elem>(((m, k, n), (ta, tb), (sa, sb)): Sample, counts: &mut RouteCounts) {
    let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
    let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
    let a = Stored::<T>::new(ar, ac, sa, 3 + m + 7 * k);
    let b = Stored::<T>::new(br, bc, sb, 5 + n + 11 * k);
    let (av, bv) = (a.view(), b.view());
    let ctx = format!(
        "{} {:?} {ta:?}/{tb:?} {sa:?}/{sb:?}",
        core::any::type_name::<T>(),
        (m, k, n)
    );
    let inp = RouteInputs::new(ta, tb, &av, &bv, m, n, k, SimdTier::current());
    // The storage is what the sample means.
    let units = |s: Store| match s {
        Store::Col => (true, false),
        Store::Row => (false, true),
        Store::Neither => (false, false),
    };
    assert_eq!(
        (inp.a_unit_col_stride, inp.a_unit_row_stride),
        units(sa),
        "{ctx}: A"
    );
    assert_eq!(
        (inp.b_unit_col_stride, inp.b_unit_row_stride),
        units(sb),
        "{ctx}: B"
    );
    let expected = stated_route(&inp, build_rules());
    let legacy_route = stated_route(&inp, RULES_LEGACY);
    // The measured scope of the adopted rule, stated without the model: an
    // operand not stored column-major keeps the route of 0.2.3 (and with it
    // its bits, checked below); the thin route needs a tier with SIMD thin
    // kernels.
    let both_col = sa == Store::Col && sb == Store::Col;
    if build_rules() == RULES_ADOPTED && !both_col {
        assert_eq!(
            expected, legacy_route,
            "{ctx}: outside the measured storage"
        );
    }
    if expected == GemmRoute::Thin {
        assert!(both_col && simd_thin_tier(inp.tier), "{ctx}: thin route");
    }

    let refs: Vec<(Dd, f64)> = (0..n)
        .flat_map(|j| (0..m).map(move |i| (i, j)))
        .map(|ij| sums(&av, &bv, (ta, tb), k, ij))
        .collect();
    let scalars = [0.0, 1.0, -1.0, 0.5];
    for alpha in scalars {
        for beta in scalars {
            let run = |par: Par, entry_gemm: bool| {
                let mut data = c_storage::<T>(m, n, beta == 0.0);
                clear_route();
                if entry_gemm {
                    gemm(T::of(alpha), av, bv, T::of(beta), c_view(&mut data, m, n));
                } else {
                    gemm_transposed_with_par(
                        ta,
                        tb,
                        T::of(alpha),
                        av,
                        bv,
                        T::of(beta),
                        c_view(&mut data, m, n),
                        par,
                    );
                }
                (data, last_route())
            };
            let what = format!("{ctx} alpha {alpha} beta {beta}");
            let nn = (ta, tb) == (Trans::NoTrans, Trans::NoTrans);
            let (seq, route) = run(Par::Seq, nn);
            if alpha == 0.0 {
                assert_eq!(route, None, "{what}: alpha == 0 takes no route");
            } else {
                assert_eq!(route, Some(expected), "{what}");
                counts[expected as usize] += 1;
            }
            let ld = m + 2;
            for j in 0..n {
                for i in 0..m {
                    let got = seq[i + j * ld].to_f64();
                    let (s, mag) = refs[i + j * m];
                    let c0 = if beta == 0.0 { 0.0 } else { value(i, j, 99) };
                    // C0 as stored in T.
                    let c0 = T::of(c0).to_f64();
                    // alpha and beta are 0, ±1 or 1/2: the scaling is exact.
                    let exact = Dd {
                        hi: alpha * s.hi,
                        lo: alpha * s.lo,
                    }
                    .add(beta * c0);
                    let reference = exact.hi + exact.lo;
                    let bound = alpha.abs() * mag + (beta * c0).abs();
                    let tol = gamma::<T>(k + 2) * bound * (1.0 + 1e-12)
                        + f64::EPSILON / 2.0 * reference.abs()
                        + f64::MIN_POSITIVE;
                    assert!(
                        (got - reference).abs() <= tol,
                        "{what}: ({i},{j}) {got} vs {reference} (tol {tol})"
                    );
                }
                for pad in m..ld {
                    assert_eq!(seq[pad + j * ld].to_f64(), 7.0, "{what}: padding");
                }
            }
            #[cfg(feature = "parallel")]
            {
                let (par, par_route) = run(Par::Rayon, false);
                assert_eq!(par_route, route, "{what}: Rayon route");
                let same = seq.iter().zip(&par).all(|(x, y)| x.bits() == y.bits());
                assert!(same, "{what}: Rayon bits differ from sequential");
            }
            // A route of 0.2.3 gives the bits of the legacy policy, except the
            // small route of the x86-64 candidate where it runs the AVX2 + FMA
            // small kernels (`small_simd_x86`, from 5 × 4) instead of 0.2.3's
            // loops.
            let x86_small_kernels = build_rules() == RULES_CANDIDATE_X86
                && expected == GemmRoute::Small
                && inp.tier == SimdTier::Avx2Fma
                && m >= 5
                && n >= 4;
            if alpha != 0.0 && expected == legacy_route && !x86_small_kernels {
                let mut data = c_storage::<T>(m, n, beta == 0.0);
                let plan = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
                gemm_planned(
                    ta,
                    tb,
                    T::of(alpha),
                    av,
                    bv,
                    T::of(beta),
                    c_view(&mut data, m, n),
                    Par::Seq,
                    BlockingSource::Default,
                    &plan,
                );
                let same = seq.iter().zip(&data).all(|(x, y)| x.bits() == y.bits());
                assert!(same, "{what}: bits differ from the legacy plan");
            }
        }
    }
}

/// `(m, k, n)` and the transposes.
type ShapeTrans = ((usize, usize, usize), (Trans, Trans));

const N: Trans = Trans::NoTrans;
const T: Trans = Trans::Trans;
const C: Trans = Trans::ConjTrans;

/// The thin shapes: every row of the scope table on both sides of each
/// limit, with `(m, k, n)` and the transposes.
const THIN: [ShapeTrans; 64] = [
    // n == 1, NN (least m 2, least k 2, floor 1024).
    ((2, 512, 1), (N, N)),
    ((2, 511, 1), (N, N)),
    ((512, 2, 1), (N, N)),
    ((511, 2, 1), (N, N)),
    ((1024, 1, 1), (N, N)),
    ((342, 3, 1), (N, N)),
    ((341, 3, 1), (N, N)),
    ((2, 16_384, 1), (N, N)),
    ((2, 16_385, 1), (N, N)),
    ((1024, 32, 1), (N, N)),
    ((1025, 32, 1), (N, N)),
    // n == 1, TN (least m 2, least k 16).
    ((64, 16, 1), (T, N)),
    ((64, 15, 1), (T, N)),
    ((63, 16, 1), (T, N)),
    ((2, 512, 1), (T, N)),
    ((2, 511, 1), (C, N)),
    ((2184, 15, 1), (T, N)),
    ((2185, 15, 1), (T, N)),
    ((128, 16, 1), (C, N)),
    // m == 1, NN (least n 4, least k 2).
    ((1, 256, 4), (N, N)),
    ((1, 255, 4), (N, N)),
    ((1, 342, 3), (N, N)),
    ((1, 2, 512), (N, N)),
    ((1, 2, 511), (N, N)),
    ((1, 1, 1024), (N, N)),
    ((1, 10_923, 3), (N, N)),
    // m == 1, NT (least n 4, least k 4).
    ((1, 4, 256), (N, T)),
    ((1, 4, 255), (N, T)),
    ((1, 3, 512), (N, T)),
    ((1, 256, 4), (N, C)),
    ((1, 256, 3), (N, T)),
    ((1, 3, 10_922), (N, T)),
    ((1, 3, 10_923), (N, T)),
    // The vector operand transposed and both transposed: no thin route
    // up to 32³, the thin route above.
    ((64, 16, 1), (N, T)),
    ((64, 16, 1), (T, T)),
    ((64, 513, 1), (N, T)),
    ((64, 513, 1), (C, C)),
    ((1, 16, 64), (T, N)),
    ((1, 16, 64), (T, T)),
    ((1, 513, 64), (T, N)),
    ((1, 513, 64), (T, C)),
    ((1, 512, 64), (T, T)),
    // Dot products: no thin route up to 32³.
    ((1, 1024, 1), (N, N)),
    ((1, 1024, 1), (T, T)),
    ((1, 32_768, 1), (N, N)),
    ((1, 32_769, 1), (N, N)),
    ((1, 32_769, 1), (T, N)),
    ((1, 65_536, 1), (C, C)),
    // Below the floor.
    ((32, 31, 1), (N, N)),
    ((1, 31, 32), (N, N)),
    ((4, 255, 1), (N, N)),
    ((1, 255, 4), (N, T)),
    // Above 32³ in every layout.
    ((4096, 9, 1), (N, N)),
    ((4096, 9, 1), (T, N)),
    ((1, 9, 4096), (N, N)),
    ((1, 9, 4096), (N, T)),
    ((32_769, 1, 1), (N, N)),
    ((1, 1, 32_769), (N, N)),
    // The parallel thin path (at least 512² and 128² per thread).
    ((8192, 64, 1), (N, N)),
    ((8192, 64, 1), (T, N)),
    ((1, 64, 8192), (N, N)),
    ((1, 64, 8192), (N, T)),
    ((1, 4096, 128), (N, N)),
    ((4096, 128, 1), (T, T)),
];

/// The copy shapes and their neighbours (m, n of the NEON kernels' least
/// size 4, the range's ends 12³ and 32³).
const COPY: [ShapeTrans; 22] = [
    ((12, 12, 12), (T, N)),
    ((12, 12, 12), (N, T)),
    ((12, 12, 12), (T, T)),
    ((12, 12, 12), (N, N)),
    ((11, 157, 1), (T, N)),
    ((11, 1, 157), (T, N)),
    ((157, 1, 11), (C, N)),
    ((4, 1, 432), (T, N)),
    ((4, 108, 4), (T, C)),
    ((3, 32, 32), (T, N)),
    ((32, 32, 3), (T, T)),
    ((32, 32, 32), (T, N)),
    ((32, 32, 32), (C, C)),
    ((32, 32, 32), (N, T)),
    ((9, 11, 331), (T, N)),
    ((33, 32, 32), (T, N)),
    ((7, 13, 19), (T, N)),
    ((8, 8, 8), (T, N)),
    ((13, 11, 12), (T, T)),
    ((16, 16, 16), (N, T)),
    ((16, 16, 16), (N, N)),
    ((64, 64, 64), (T, N)),
];

/// The storages of a sample: both column-major, then each operand as a
/// transposed view and without a unit stride.
const STORES: [(Store, Store); 5] = [
    (Store::Col, Store::Col),
    (Store::Row, Store::Col),
    (Store::Col, Store::Row),
    (Store::Neither, Store::Col),
    (Store::Col, Store::Neither),
];

/// The generated shapes: for each of the four small layouts of the scope
/// table (and the `ConjTrans` twins of the two with a transposed operand)
/// the nine points `v·k = 1024` with the two vectors and dots next to them,
/// every factorisation of `v·k = 1023` and `1025` (the floor), the least
/// vector length and depth with their neighbours (from the floor on), and a
/// point above 32³; the other four layouts (and two `ConjTrans` forms) at
/// two small points and two above 32³; the dot products next to 1024 and
/// 32³ in four transpose pairs; and the cubes 11³ to 13³ and 31³ to 33³ in
/// all nine transpose pairs (the copy's two ends).
fn generated() -> Vec<ShapeTrans> {
    let mut out = Vec::new();
    let shape = |column: bool, v: usize, k: usize| if column { (v, k, 1) } else { (1, k, v) };
    let divisors = |w: usize| (1..=w).filter(move |d| w.is_multiple_of(*d));
    // (a column of C, the transposes, least max(m, n), least k)
    let layouts = [
        (true, (N, N), 2, 2),
        (true, (T, N), 2, 16),
        (true, (C, N), 2, 16),
        (false, (N, N), 4, 2),
        (false, (N, T), 4, 4),
        (false, (N, C), 4, 4),
    ];
    for (column, trans, min_vec, min_depth) in layouts {
        let mut points: Vec<(usize, usize)> = vec![
            (1, 1024),
            (2, 512),
            (4, 256),
            (8, 128),
            (16, 64),
            (32, 32),
            (64, 16),
            (128, 8),
            (256, 4),
            (512, 2),
            (1024, 1),
            (33, 1000),
        ];
        for w in [1023usize, 1025] {
            points.extend(divisors(w).map(|v| (v, w / v)));
        }
        for v in [min_vec - 1, min_vec, min_vec + 1] {
            points.push((v, 1024usize.div_ceil(v)));
        }
        for k in [min_depth - 1, min_depth, min_depth + 1] {
            points.push((1024usize.div_ceil(k), k));
        }
        out.extend(
            points
                .into_iter()
                .map(|(v, k)| (shape(column, v, k), trans)),
        );
    }
    for (column, trans) in [
        (true, (N, T)),
        (true, (T, T)),
        (true, (C, C)),
        (false, (T, N)),
        (false, (T, T)),
        (false, (C, T)),
    ] {
        for (v, k) in [(64, 16), (2, 512), (1024, 64), (33, 1000)] {
            out.push((shape(column, v, k), trans));
        }
    }
    for k in [1023usize, 1024, 32_768, 32_769] {
        for trans in [(N, N), (T, N), (N, T), (T, T)] {
            out.push(((1, k, 1), trans));
        }
    }
    for s in [11usize, 12, 13, 31, 32, 33] {
        for ta in [N, T, C] {
            for tb in [N, T, C] {
                out.push(((s, s, s), (ta, tb)));
            }
        }
    }
    out
}

/// The distinct `(shape, transposes)` of the sample.
fn shapes() -> Vec<ShapeTrans> {
    let mut all: Vec<ShapeTrans> = Vec::new();
    for item in THIN.iter().chain(COPY.iter()).copied().chain(generated()) {
        if !all.contains(&item) {
            all.push(item);
        }
    }
    all
}

/// Every shape of the sample in every storage pair.
fn samples() -> Vec<Sample> {
    let mut all = Vec::new();
    for (shape, trans) in shapes() {
        for stores in STORES {
            all.push((shape, trans, stores));
        }
    }
    all
}

fn check_all<T: Elem>(part: usize, parts: usize) -> RouteCounts {
    let mut counts = [0usize; 5];
    for (index, sample) in samples().into_iter().enumerate() {
        if index % parts == part {
            check_sample::<T>(sample, &mut counts);
        }
    }
    counts
}

/// The route counts of every sample under this build's rule (alpha != 0:
/// 12 calls per sample), for the expectations below.
fn expected_counts() -> RouteCounts {
    let mut counts = [0usize; 5];
    for ((m, k, n), (ta, tb), (sa, sb)) in samples() {
        let units = |s: Store| (s == Store::Col, s == Store::Row);
        let ((ac, ar), (bc, br)) = (units(sa), units(sb));
        let inp = RouteInputs {
            m,
            n,
            k,
            trans_a: ta,
            trans_b: tb,
            a_unit_col_stride: ac,
            a_unit_row_stride: ar,
            b_unit_col_stride: bc,
            b_unit_row_stride: br,
            simd_real: true,
            tier: SimdTier::current(),
        };
        counts[stated_route(&inp, build_rules()) as usize] += 12;
    }
    counts
}

#[test]
fn test_gemm_plan_public_route_check_f64_part0() {
    check_all::<f64>(0, 3);
}

#[test]
fn test_gemm_plan_public_route_check_f64_part1() {
    check_all::<f64>(1, 3);
}

#[test]
fn test_gemm_plan_public_route_check_f64_part2() {
    check_all::<f64>(2, 3);
}

#[test]
fn test_gemm_plan_public_route_check_f32_part0() {
    check_all::<f32>(0, 3);
}

#[test]
fn test_gemm_plan_public_route_check_f32_part1() {
    check_all::<f32>(1, 3);
}

#[test]
fn test_gemm_plan_public_route_check_f32_part2() {
    check_all::<f32>(2, 3);
}

#[test]
fn test_gemm_plan_public_route_check_sample_routes() {
    // The sample has at least 300 distinct shapes with their transposes
    // (each in five storage pairs), and under the adopted rule on the NEON
    // tier it reaches the thin route, the copy and the legacy routes; on a
    // tier without SIMD thin kernels (also a build where only oxiblas-core
    // has `force-scalar`) neither the thin route nor the copy; under 0.2.3's
    // rule never the thin route or the copy.
    let distinct = shapes().len();
    assert!(distinct >= 300, "{distinct}");
    assert_eq!(samples().len(), 5 * distinct);
    let counts = expected_counts();
    let thin = counts[GemmRoute::Thin as usize];
    let copy = counts[GemmRoute::SmallTransCopy as usize];
    let tier = SimdTier::current();
    if build_rules() == RULES_LEGACY {
        assert_eq!((thin, copy), (0, 0));
    } else if simd_thin_tier(tier) {
        assert!(thin > 0, "{counts:?}");
        if tier == SimdTier::Neon {
            assert!(copy > 0, "{counts:?}");
        }
    } else {
        assert_eq!(thin, 0, "{tier:?} {counts:?}");
        if build_rules() == RULES_ADOPTED {
            assert_eq!(copy, 0, "{tier:?} {counts:?}");
        }
    }
    for route in [GemmRoute::Small, GemmRoute::SmallTrans, GemmRoute::Blocked] {
        assert!(counts[route as usize] > 0, "{route:?} {counts:?}");
    }
}
