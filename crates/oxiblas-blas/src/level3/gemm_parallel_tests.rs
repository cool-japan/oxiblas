//! Tests for the two-dimensional parallel GEMM driver: bitwise equality with
//! the sequential driver for every thread count and both pack scopes, the
//! tile grids, the per-call micro-kernel the driver must run, the scope the
//! plan selects, and the per-tile workers' buffer reuse.

use super::*;
use crate::level3::autotune::KC_BLOCK_UNIT;
use crate::level3::gemm::{
    BlockingSource, PARALLEL_GATE, gemm_planned, gemm_transposed_with_blocking,
    gemm_transposed_with_par,
};
use crate::level3::gemm_kernel::f64_kernel_for;
use crate::level3::gemm_plan::{
    CANDIDATE_AARCH64, CANDIDATE_X86, DEFAULT_POLICY, GemmPlan, GemmPolicy, GemmRoute,
    LEGACY_POLICY,
};
use num_complex::{Complex32, Complex64};
use oxiblas_core::simd::dispatch::GemmKernelKind;
use oxiblas_matrix::Mat;

/// Both pack scopes of the 2D driver.
const SCOPES: [PackScope; 2] = [PackScope::Shared, PackScope::PerTile];

/// Element helpers for the bitwise comparisons.
trait Probe: Field + GemmKernel + bytemuck::Zeroable {
    /// Exact bit patterns (real, imaginary).
    fn bits(self) -> (u64, u64);
    /// True when every component is finite.
    fn finite(self) -> bool;
    /// Deterministic, non-symmetric sample value.
    fn sample(seed: usize) -> Self;
    /// A NaN (NaN in every component).
    fn nan() -> Self;
}

fn unit_sample(seed: usize) -> f64 {
    ((seed.wrapping_mul(2_654_435_761) >> 7) % 1009) as f64 / 504.5 - 1.0
}

impl Probe for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn sample(seed: usize) -> Self {
        unit_sample(seed)
    }
    fn nan() -> Self {
        f64::NAN
    }
}

impl Probe for f32 {
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn sample(seed: usize) -> Self {
        unit_sample(seed) as f32
    }
    fn nan() -> Self {
        f32::NAN
    }
}

impl Probe for Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn sample(seed: usize) -> Self {
        Complex64::new(unit_sample(seed), unit_sample(seed.wrapping_add(7_919)))
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
}

impl Probe for Complex32 {
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn sample(seed: usize) -> Self {
        Complex32::new(
            unit_sample(seed) as f32,
            unit_sample(seed.wrapping_add(7_919)) as f32,
        )
    }
    fn nan() -> Self {
        Complex32::new(f32::NAN, f32::NAN)
    }
}

fn fill<T: Probe>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            out[(i, j)] = T::sample(i * 7_919 + j * 104_729 + seed * 1_299_709);
        }
    }
    out
}

/// The stored operand of `op(X)` for an `rows × cols` logical operand.
fn operand<T: Probe>(trans: Trans, rows: usize, cols: usize, seed: usize) -> Mat<T> {
    if trans == Trans::NoTrans {
        fill::<T>(rows, cols, seed)
    } else {
        fill::<T>(cols, rows, seed)
    }
}

fn assert_bits_eq<T: Probe>(got: &Mat<T>, expected: &Mat<T>, ctx: &str) {
    for j in 0..got.ncols() {
        for i in 0..got.nrows() {
            assert_eq!(
                got[(i, j)].bits(),
                expected[(i, j)].bits(),
                "{ctx} at ({i},{j}): {} vs {}",
                got[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

/// The blocking `gemm_transposed_with_par` selects, fitted to the problem.
fn fitted_blocking<T: Probe>(m: usize, k: usize, n: usize) -> GemmBlocking {
    let shape = T::micro_kernel_shape();
    let blk = if m >= 512 || k >= 512 || n >= 512 {
        GemmBlocking::auto_tuned::<T>(m, k, n, &shape)
    } else {
        GemmBlocking::for_kernel::<T>(&shape)
    };
    fit(&blk, m, k, n, &shape)
}

fn fit(blk: &GemmBlocking, m: usize, k: usize, n: usize, shape: &MicroKernelShape) -> GemmBlocking {
    GemmBlocking {
        mc: balanced_block(m, blk.mc, shape.mr),
        kc: balanced_block(k, blk.kc, KC_BLOCK_UNIT),
        nc: balanced_block(n, blk.nc, shape.nr),
    }
}

/// A LEGACY plan forced onto the blocked route with `kernel` and `scope`.
fn blocked_plan<T: Probe>(kernel: MicroKernel<T>, scope: PackScope) -> GemmPlan<T> {
    GemmPlan {
        kernel,
        ..GemmPlan::<T>::new(LEGACY_POLICY)
    }
    .with_forced_route(GemmRoute::Blocked)
    .with_forced_scope(scope)
}

/// Runs the 2D driver directly with `scope` and asserts that it engaged with
/// that scope (and not the other one).
#[allow(clippy::too_many_arguments)]
fn run_direct<T: Probe>(
    (ta, tb): (Trans, Trans),
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    kernel: &MicroKernel<T>,
    scope: PackScope,
    par: Par,
    ctx: &str,
) {
    let before = scope_runs();
    let engaged =
        try_gemm_blocked_parallel(ta, tb, alpha, a, b, beta, c, blocking, kernel, scope, par);
    assert!(engaged, "{ctx}: 2D driver declined");
    let after = scope_runs();
    let expected = match scope {
        PackScope::Shared => (before.0 + 1, before.1),
        PackScope::PerTile => (before.0, before.1 + 1),
    };
    assert_eq!(after, expected, "{ctx}: scope counter");
}

/// Runs `C = alpha * op(A) * op(B) + beta * C` sequentially, then on the 2D
/// driver with every scope of `scopes` (directly, and through the planned
/// funnel with that scope forced) and through the public entry point, for
/// every thread count, asserting bitwise-identical results. With `poison_c`,
/// C starts as NaN and `beta == 0`, so C must not be read.
fn check<T: Probe>(
    dims: (usize, usize, usize),
    trans: (Trans, Trans),
    blocking: Option<GemmBlocking>,
    threads: &[usize],
    poison_c: bool,
    scopes: &[PackScope],
) {
    let legs: Vec<(usize, &[PackScope])> = threads.iter().map(|&t| (t, scopes)).collect();
    check_legs::<T>(dims, trans, blocking, &legs, poison_c);
}

/// [`check`] with its own scopes per thread count: each leg `(threads,
/// scopes)` runs the 2D driver on `threads` threads with every scope of
/// `scopes` (directly and planned), and the public entry point once.
fn check_legs<T: Probe>(
    (m, k, n): (usize, usize, usize),
    (ta, tb): (Trans, Trans),
    blocking: Option<GemmBlocking>,
    legs: &[(usize, &[PackScope])],
    poison_c: bool,
) {
    let kernel = T::micro_kernel_dispatch();
    let shape = kernel.shape;
    let a = operand::<T>(ta, m, k, 1);
    let b = operand::<T>(tb, k, n, 2);
    let alpha = T::sample(5);
    let (beta, c_init) = if poison_c {
        (T::zero(), Mat::filled(m, n, T::nan()))
    } else {
        (T::sample(9), fill::<T>(m, n, 3))
    };
    let fitted = match blocking {
        Some(blk) => fit(&blk, m, k, n, &shape),
        None => fitted_blocking::<T>(m, k, n),
    };
    let ctx = format!("{m}x{k}x{n} {ta:?}/{tb:?} {fitted:?}");

    let mut c_seq = c_init.clone();
    gemm_transposed_with_blocking(
        ta,
        tb,
        alpha,
        a.as_ref(),
        b.as_ref(),
        beta,
        c_seq.as_mut(),
        Par::Seq,
        &fitted,
    );
    if poison_c {
        for j in 0..n {
            for i in 0..m {
                assert!(c_seq[(i, j)].finite(), "{ctx}: beta = 0 read C");
            }
        }
    }

    for &(t, scopes) in legs {
        let par = Par::RayonWith(t);
        assert!(
            PARALLEL_GATE.should_parallelize(m * n * k, par),
            "{ctx}: gate, t={t}"
        );

        for &scope in scopes {
            let mut c_par = c_init.clone();
            run_direct(
                (ta, tb),
                alpha,
                &a.as_ref(),
                &b.as_ref(),
                beta,
                &mut c_par.as_mut(),
                &fitted,
                &kernel,
                scope,
                par,
                &format!("{ctx} {scope:?} t={t}"),
            );
            assert_bits_eq(&c_par, &c_seq, &format!("{ctx} {scope:?} direct t={t}"));

            let mut c_plan = c_init.clone();
            gemm_planned(
                ta,
                tb,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c_plan.as_mut(),
                par,
                BlockingSource::Explicit(fitted),
                &blocked_plan(kernel, scope),
            );
            assert_bits_eq(&c_plan, &c_seq, &format!("{ctx} {scope:?} planned t={t}"));
        }

        let mut c_api = c_init.clone();
        gemm_transposed_with_blocking(
            ta,
            tb,
            alpha,
            a.as_ref(),
            b.as_ref(),
            beta,
            c_api.as_mut(),
            par,
            &fitted,
        );
        assert_bits_eq(&c_api, &c_seq, &format!("{ctx} api t={t}"));
    }
}

/// Shapes (m, k, n): square-ish with two KC blocks (f64; a dimension of 512
/// or more selects the auto-tuned blocking), tall, wide, and a deep k with
/// five KC blocks. Each is above the 8-thread parallel gate (80³ per
/// thread) and splits into several tiles.
const SHAPES: [(usize, usize, usize); 4] = [
    (257, 517, 150),
    (2048, 64, 64),
    (64, 64, 2048),
    (37, 4099, 33),
];

/// Thread counts of the per-tile legs of the issue #16 tests, one per
/// transpose combination in turn: the shared scope runs on every thread
/// count of a test, the per-tile scope (covered on every thread count by the
/// `per_tile` tests) on one of these per combination.
const PER_TILE_LEG_THREADS: [usize; 4] = [2, 3, 5, 8];

/// The legs of an issue #16 check: the shared scope on each of `threads`,
/// and the per-tile scope on `per_tile` threads (with the shared scope too
/// when `per_tile` is one of `threads`).
fn issue_16_legs(threads: &[usize], per_tile: usize) -> Vec<(usize, &'static [PackScope])> {
    let mut legs: Vec<(usize, &'static [PackScope])> = threads
        .iter()
        .map(|&t| {
            let scopes: &'static [PackScope] = if t == per_tile {
                &SCOPES
            } else {
                &[PackScope::Shared]
            };
            (t, scopes)
        })
        .collect();
    if !threads.contains(&per_tile) {
        legs.push((per_tile, &[PackScope::PerTile]));
    }
    legs
}

#[test]
fn test_issue_16_parallel_matches_sequential_bitwise_f64() {
    let combos = [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::ConjTrans),
        (Trans::Trans, Trans::Trans),
    ];
    for &shape in &SHAPES {
        for (idx, &combo) in combos.iter().enumerate() {
            let legs = issue_16_legs(&[2, 3, 8], PER_TILE_LEG_THREADS[idx]);
            check_legs::<f64>(shape, combo, None, &legs, false);
        }
        check::<f64>(
            shape,
            (Trans::NoTrans, Trans::NoTrans),
            None,
            &[5],
            true,
            &SCOPES,
        );
    }
}

#[test]
fn test_issue_16_parallel_matches_sequential_bitwise_f32() {
    for &shape in &SHAPES {
        check::<f32>(
            shape,
            (Trans::NoTrans, Trans::NoTrans),
            None,
            &[2, 8],
            false,
            &SCOPES,
        );
        check::<f32>(
            shape,
            (Trans::Trans, Trans::ConjTrans),
            None,
            &[3],
            false,
            &SCOPES,
        );
        check::<f32>(
            shape,
            (Trans::NoTrans, Trans::Trans),
            None,
            &[4],
            true,
            &SCOPES,
        );
    }
}

#[test]
fn test_issue_16_parallel_matches_sequential_bitwise_c64() {
    // 213 * 157 * 137 = 4.58M multiply-adds clears the 8-thread gate
    // (PARALLEL_GATE: 80^3 per thread, i.e. 160^3 on 8 threads).
    // Every combination runs the shared scope on 2, 3 and 8 threads and the
    // per-tile scope on one of 2, 3, 5 and 8.
    let all = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    for (ia, &ta) in all.iter().enumerate() {
        for (ib, &tb) in all.iter().enumerate() {
            let combo = 3 * ia + ib;
            let per_tile = PER_TILE_LEG_THREADS[combo % PER_TILE_LEG_THREADS.len()];
            let legs = issue_16_legs(&[2, 3, 8], per_tile);
            check_legs::<Complex64>((213, 157, 137), (ta, tb), None, &legs, false);
        }
    }
    for &shape in &[(2048usize, 64usize, 64usize), (64, 64, 4096)] {
        check::<Complex64>(
            shape,
            (Trans::ConjTrans, Trans::ConjTrans),
            None,
            &[3],
            false,
            &SCOPES,
        );
        check::<Complex64>(
            shape,
            (Trans::NoTrans, Trans::NoTrans),
            None,
            &[5],
            true,
            &SCOPES,
        );
    }
}

#[test]
fn test_issue_16_parallel_bitwise_with_many_blocks() {
    // Small explicit targets: four NC blocks, four KC blocks and several
    // MC sub-blocks per tile exercise every loop of the 2D driver.
    let blocking = GemmBlocking {
        mc: 64,
        kc: 48,
        nc: 96,
    };
    let combos = [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
        (Trans::ConjTrans, Trans::ConjTrans),
    ];
    for &combo in &combos {
        check::<f64>(
            (200, 150, 300),
            combo,
            Some(blocking),
            &[2, 3, 5, 8],
            false,
            &SCOPES,
        );
    }
    check::<f64>(
        (200, 150, 300),
        combos[0],
        Some(blocking),
        &[4],
        true,
        &SCOPES,
    );
    check::<f32>(
        (200, 150, 300),
        combos[1],
        Some(blocking),
        &[3],
        false,
        &SCOPES,
    );
    check::<Complex64>(
        (200, 150, 300),
        combos[3],
        Some(blocking),
        &[3],
        false,
        &SCOPES,
    );
}

/// Transposed views (col_stride != 1) for A and B, and a C view with a
/// leading dimension larger than its row count, on the 2D driver with
/// `scope` and `blocking` (`None`: the default fitted blocking).
fn check_strided_views(scope: PackScope, blocking: Option<GemmBlocking>, threads: usize) {
    let (m, k, n) = (160usize, 150usize, 120usize);
    let a_store = fill::<f64>(k, m, 1);
    let b_store = fill::<f64>(n, k, 2);
    let a_view = a_store.as_ref().transpose();
    let b_view = b_store.as_ref().transpose();
    assert_ne!(a_view.col_stride(), 1);
    let kernel = f64::micro_kernel_dispatch();
    let fitted = match blocking {
        Some(blk) => fit(&blk, m, k, n, &kernel.shape),
        None => fitted_blocking::<f64>(m, k, n),
    };
    let ctx = format!("strided views {scope:?} {fitted:?} t={threads}");

    let mut c_seq_big = fill::<f64>(m + 5, n + 3, 3);
    let mut c_par_big = c_seq_big.clone();
    gemm_transposed_with_blocking(
        Trans::NoTrans,
        Trans::NoTrans,
        0.75,
        a_view,
        b_view,
        -1.25,
        c_seq_big.as_mut().submatrix(2, 1, m, n),
        Par::Seq,
        &fitted,
    );
    run_direct(
        (Trans::NoTrans, Trans::NoTrans),
        0.75,
        &a_view,
        &b_view,
        -1.25,
        &mut c_par_big.as_mut().submatrix(2, 1, m, n),
        &fitted,
        &kernel,
        scope,
        Par::RayonWith(threads),
        &ctx,
    );
    // The whole backing store must agree, including the untouched border.
    assert_bits_eq(&c_par_big, &c_seq_big, &ctx);
}

#[test]
fn test_issue_16_parallel_bitwise_strided_views() {
    for scope in SCOPES {
        check_strided_views(scope, None, 4);
    }
}

#[test]
fn test_issue_16_tile_grid_valid_and_balanced() {
    for &(mr, nr) in &[(8usize, 6usize), (8, 8), (16, 6), (4, 4), (2, 2)] {
        let shape = MicroKernelShape { mr, nr };
        for &m in &[1usize, 7, 64, 100, 513, 2048, 4096] {
            for &w in &[1usize, 6, 64, 517, 2048, 5712] {
                for &threads in &[2usize, 3, 4, 8, 12, 64] {
                    let grid = TileGrid::choose(m, w, &shape, threads);
                    let (mu, nu) = (m.div_ceil(mr), w.div_ceil(nr));
                    let ctx = format!("{mr}x{nr} m={m} w={w} t={threads}: {grid:?}");
                    assert!((1..=mu).contains(&grid.rows), "{ctx}");
                    assert!((1..=nu).contains(&grid.cols), "{ctx}");
                    assert!(grid.count() < 2 * TILES_PER_THREAD * threads, "{ctx}");
                    if mu * nu >= 8 {
                        assert!(grid.count() >= 2, "{ctx}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_16_split_tiles_cover_c_exactly_once() {
    let shape = MicroKernelShape { mr: 8, nr: 6 };
    for &(m, w, threads) in &[
        (37usize, 50usize, 3usize),
        (513, 300, 8),
        (8, 6, 2),
        (64, 4096, 12),
    ] {
        for grid in [
            TileGrid::choose(m, w, &shape, threads),
            TileGrid::choose_per_tile(m, w, &shape, threads),
        ] {
            let mut c: Mat<f64> = Mat::zeros(m, w);
            let tiles = split_tiles(c.as_mut(), grid, &shape);
            assert_eq!(tiles.len(), grid.count());
            let mut covered_rows = 0;
            for mut tile in tiles {
                let (rows, cols) = (tile.view.nrows(), tile.view.ncols());
                assert!(rows > 0 && cols > 0, "empty tile");
                assert_eq!(tile.row0 % shape.mr, 0, "row start off the micro grid");
                assert_eq!(tile.col0 % shape.nr, 0, "column start off the micro grid");
                assert!(tile.row0 + rows <= m && tile.col0 + cols <= w);
                assert!(rows <= grid.tallest_rows(m, shape.mr), "{grid:?}");
                assert!(cols <= grid.widest_cols(w, shape.nr), "{grid:?}");
                // Tag each element with its global row so a misplaced view is
                // caught, not just a double-counted one.
                for j in 0..cols {
                    for i in 0..rows {
                        tile.view[(i, j)] += 1.0 + (tile.row0 + i) as f64 * 1e3;
                    }
                }
                covered_rows += rows * cols;
            }
            assert_eq!(covered_rows, m * w);
            for j in 0..w {
                for i in 0..m {
                    let expected = 1.0 + i as f64 * 1e3;
                    assert_eq!(c[(i, j)], expected, "({i},{j}) misplaced or double-covered");
                }
            }
        }
    }
}

#[test]
fn test_issue_16_square_problems_split_across_threads() {
    // With the NC-only split, a square problem with n <= NC was a single
    // work item. Every NC block must now offer at least one tile per
    // thread, in both scopes.
    let threads = 12;
    for &n in &[512usize, 1024, 2048, 4096] {
        for (label, shape, blk) in [
            (
                "f64",
                f64::micro_kernel_shape(),
                GemmBlocking::auto_tuned::<f64>(n, n, n, &f64::micro_kernel_shape()),
            ),
            (
                "f32",
                f32::micro_kernel_shape(),
                GemmBlocking::auto_tuned::<f32>(n, n, n, &f32::micro_kernel_shape()),
            ),
        ] {
            let nc = balanced_block(n, blk.nc, shape.nr);
            let last_w = n - (n - 1) / nc * nc;
            for w in [nc, last_w] {
                for grid in [
                    TileGrid::choose(n, w, &shape, threads),
                    TileGrid::choose_per_tile(n, w, &shape, threads),
                ] {
                    assert!(
                        grid.count() >= threads,
                        "{label} n={n} w={w}: only {} tiles ({grid:?})",
                        grid.count()
                    );
                }
            }
        }
    }
}

#[test]
fn test_issue_16_parallel_alpha_zero_and_api_paths() {
    // The funnel handles alpha == 0 before the 2D driver: NaN operands
    // must not leak, whatever the thread count.
    let n = 200;
    let a: Mat<f64> = Mat::filled(n, n, f64::NAN);
    let b: Mat<f64> = Mat::filled(n, n, f64::INFINITY);
    for &t in &[2usize, 7] {
        let mut c: Mat<f64> = Mat::filled(n, n, 2.0);
        gemm_transposed_with_par(
            Trans::Trans,
            Trans::NoTrans,
            0.0,
            a.as_ref(),
            b.as_ref(),
            0.5,
            c.as_mut(),
            Par::RayonWith(t),
        );
        for j in 0..n {
            for i in 0..n {
                assert_eq!(c[(i, j)].to_bits(), 1.0f64.to_bits(), "t={t} ({i},{j})");
            }
        }
    }
}

// ------------------------------------------------------------------
// The per-call micro-kernel.
// ------------------------------------------------------------------

/// Whether the f64 micro-kernel of `kind` may run in this build on this CPU.
///
/// A kernel is left out wherever the build caps SIMD below its width:
/// `force-scalar` runs no SIMD kernel at all, `max-simd-128` no 256- or
/// 512-bit kernel, `max-simd-256` no 512-bit kernel. So a capped run never
/// executes a kernel the build excludes; the 128-bit kernels run where the
/// CPU has them (NEON on every aarch64 CPU, SSE4.2 when the CPU reports it).
fn kind_runnable(kind: GemmKernelKind) -> bool {
    match kind {
        GemmKernelKind::Scalar => true,
        GemmKernelKind::Neon => cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))),
        GemmKernelKind::Sse42 => {
            #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
            {
                std::arch::is_x86_feature_detected!("sse4.2")
            }
            #[cfg(not(all(target_arch = "x86_64", not(feature = "force-scalar"))))]
            {
                false
            }
        }
        GemmKernelKind::Avx2 => {
            #[cfg(all(
                target_arch = "x86_64",
                not(any(feature = "force-scalar", feature = "max-simd-128"))
            ))]
            {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                not(any(feature = "force-scalar", feature = "max-simd-128"))
            )))]
            {
                false
            }
        }
        GemmKernelKind::Avx512 => {
            #[cfg(all(
                target_arch = "x86_64",
                not(any(
                    feature = "force-scalar",
                    feature = "max-simd-128",
                    feature = "max-simd-256"
                ))
            ))]
            {
                std::arch::is_x86_feature_detected!("avx512f")
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                not(any(
                    feature = "force-scalar",
                    feature = "max-simd-128",
                    feature = "max-simd-256"
                ))
            )))]
            {
                false
            }
        }
    }
}

/// Every f64 micro-kernel of this crate that may run here ([`kind_runnable`]).
fn runnable_f64_kernels() -> Vec<(GemmKernelKind, MicroKernel<f64>)> {
    [
        GemmKernelKind::Scalar,
        GemmKernelKind::Neon,
        GemmKernelKind::Sse42,
        GemmKernelKind::Avx2,
        GemmKernelKind::Avx512,
    ]
    .into_iter()
    .filter(|&kind| kind_runnable(kind))
    .map(|kind| (kind, f64_kernel_for(kind)))
    .collect()
}

#[test]
fn test_gemm_parallel_honours_per_call_kernel() {
    // Every tile must run the plan's micro-kernel, not a re-dispatched one,
    // in both scopes. The plan carries the portable 6×4 test kernel, whose
    // shape differs from every f64 kernel of this crate (scalar 4×4, NEON and
    // AVX2 8×6, SSE4.2 4×2, AVX-512 16×6), so from the dispatch kernel in
    // every feature set and on every CPU, and which runs no SIMD code (also
    // under `force-scalar`): the removed `macro_panel_multiply` re-dispatched
    // and asserted equal shapes, so such a plan panicked in the 2D driver,
    // and a driver that packs with one shape and runs another kernel
    // produces garbage.
    let other = shaped_f64::<6, 4>();
    for kind in [
        GemmKernelKind::Scalar,
        GemmKernelKind::Neon,
        GemmKernelKind::Sse42,
        GemmKernelKind::Avx2,
        GemmKernelKind::Avx512,
    ] {
        assert_ne!(other.shape, f64_kernel_for(kind).shape, "{kind:?}");
    }
    assert_ne!(other.shape, f64::micro_kernel_dispatch().shape);
    let (m, k, n) = (203usize, 157usize, 189usize);
    // Several MC, KC and NC blocks, fitted to the kernel's shape.
    let targets = GemmBlocking {
        mc: 64,
        kc: 48,
        nc: 96,
    };
    let fitted = fit(&targets, m, k, n, &other.shape);
    assert!(PARALLEL_GATE.should_parallelize(m * n * k, Par::RayonWith(4)));
    for (ta, tb) in [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
    ] {
        let a = operand::<f64>(ta, m, k, 11);
        let b = operand::<f64>(tb, k, n, 12);
        for beta in [0.0f64, -0.5] {
            let c0 = if beta == 0.0 {
                Mat::filled(m, n, f64::NAN)
            } else {
                fill::<f64>(m, n, 13)
            };
            let ctx = format!("{:?} {ta:?}/{tb:?} beta={beta}", other.shape);
            let mut c_seq = c0.clone();
            gemm_planned(
                ta,
                tb,
                0.75,
                a.as_ref(),
                b.as_ref(),
                beta,
                c_seq.as_mut(),
                Par::Seq,
                BlockingSource::Explicit(targets),
                &blocked_plan(other, PackScope::Shared),
            );
            for scope in SCOPES {
                let plan = blocked_plan(other, scope);
                let mut c_par = c0.clone();
                gemm_planned(
                    ta,
                    tb,
                    0.75,
                    a.as_ref(),
                    b.as_ref(),
                    beta,
                    c_par.as_mut(),
                    Par::RayonWith(4),
                    BlockingSource::Explicit(targets),
                    &plan,
                );
                assert_bits_eq(&c_par, &c_seq, &format!("{ctx}: planned {scope:?}"));

                let mut c_direct = c0.clone();
                run_direct(
                    (ta, tb),
                    0.75,
                    &a.as_ref(),
                    &b.as_ref(),
                    beta,
                    &mut c_direct.as_mut(),
                    &fitted,
                    &other,
                    scope,
                    Par::RayonWith(4),
                    &format!("{ctx} {scope:?}"),
                );
                assert_bits_eq(&c_direct, &c_seq, &format!("{ctx}: direct {scope:?}"));
            }

            // And it is the product (a packing/kernel shape mismatch would
            // stride past the panels and produce garbage).
            for j in (0..n).step_by(17) {
                for i in (0..m).step_by(13) {
                    let mut acc = 0.0f64;
                    for p in 0..k {
                        acc += read_op(&a.as_ref(), ta, i, p) * read_op(&b.as_ref(), tb, p, j);
                    }
                    let expected = if beta == 0.0 {
                        0.75 * acc
                    } else {
                        0.75 * acc + beta * c0[(i, j)]
                    };
                    let got = c_seq[(i, j)];
                    assert!(
                        (got - expected).abs() <= 1e-12 * (k as f64),
                        "{ctx} ({i},{j}): {got} vs {expected}"
                    );
                }
            }
        }
    }
}

/// Defines `$kernel::<MR, NR>`, a portable `MR × NR` micro-kernel of any
/// shape for the real type `$t`, and `$make::<MR, NR>()`, its
/// [`MicroKernel`], so the drivers can be checked with every shape a plan
/// may carry (8×8, 8×6, 16×6, 4×4, …) on every target. Per element:
/// `acc = fma(a, b, acc)` over `p` in order, then `alpha · acc` when
/// `beta == 0` (C not read) or `fma(alpha, acc, beta · c)`.
macro_rules! shaped_kernel {
    ($t:ty, $kernel:ident, $make:ident) => {
        /// A portable micro-kernel of any `MR × NR` shape (see
        /// `shaped_kernel!`).
        ///
        /// # Safety
        ///
        /// `a` holds `MR · k` and `b` holds `k · NR` readable elements; the
        /// `MR × NR` tile at `c` (column `j` at `c + j · c_stride`) is
        /// writable and, when `beta != 0`, readable.
        unsafe fn $kernel<const MR: usize, const NR: usize>(
            k: usize,
            alpha: $t,
            a: *const $t,
            b: *const $t,
            beta: $t,
            c: *mut $t,
            c_stride: usize,
        ) {
            let mut acc = [[0.0 as $t; NR]; MR];
            for p in 0..k {
                for (i, row) in acc.iter_mut().enumerate() {
                    // SAFETY: `p * MR + i < k * MR`, inside `a` (contract
                    // above).
                    let av = unsafe { *a.add(p * MR + i) };
                    for (j, slot) in row.iter_mut().enumerate() {
                        // SAFETY: `p * NR + j < k * NR`, inside `b`.
                        let bv = unsafe { *b.add(p * NR + j) };
                        *slot = av.mul_add(bv, *slot);
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    // SAFETY: `i < MR` and `j < NR`, inside the C tile
                    // (contract).
                    let cij = unsafe { c.add(i + j * c_stride) };
                    let out = if beta == 0.0 {
                        alpha * value
                    } else {
                        // SAFETY: as above; C is readable when `beta != 0`.
                        alpha.mul_add(value, beta * unsafe { *cij })
                    };
                    // SAFETY: as above, the tile is writable.
                    unsafe { *cij = out };
                }
            }
        }

        fn $make<const MR: usize, const NR: usize>() -> MicroKernel<$t> {
            MicroKernel {
                shape: MicroKernelShape { mr: MR, nr: NR },
                kernel: $kernel::<MR, NR>,
            }
        }
    };
}

shaped_kernel!(f64, shaped_kernel_f64, shaped_f64);
shaped_kernel!(f32, shaped_kernel_f32, shaped_f32);

/// Both scopes give the sequential driver's bits with each of `kernels`
/// (the blocking is fitted to its shape), and each scope runs its own runner.
fn check_kernel_shapes<T: Probe>(kernels: &[MicroKernel<T>]) {
    let (m, k, n) = (171usize, 133usize, 157usize);
    let targets = GemmBlocking {
        mc: 48,
        kc: 40,
        nc: 64,
    };
    let (ta, tb) = (Trans::Trans, Trans::NoTrans);
    let a = operand::<T>(ta, m, k, 21);
    let b = operand::<T>(tb, k, n, 22);
    let c0 = fill::<T>(m, n, 23);
    let (alpha, beta) = (T::sample(24), T::sample(25));
    for &kernel in kernels {
        let ctx = format!("{} {:?}", core::any::type_name::<T>(), kernel.shape);
        let run = |scope: PackScope, par: Par| {
            let mut c = c0.clone();
            gemm_planned(
                ta,
                tb,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c.as_mut(),
                par,
                BlockingSource::Explicit(targets),
                &blocked_plan(kernel, scope),
            );
            c
        };
        let seq = run(PackScope::Shared, Par::Seq);
        for scope in SCOPES {
            for threads in [2usize, 5] {
                let before = scope_runs();
                let got = run(scope, Par::RayonWith(threads));
                let after = scope_runs();
                let ran = match scope {
                    PackScope::Shared => (after.0 - before.0, after.1 - before.1),
                    PackScope::PerTile => (after.1 - before.1, after.0 - before.0),
                };
                assert_eq!(ran, (1, 0), "{ctx} {scope:?}: runner (this, other)");
                assert_bits_eq(&got, &seq, &format!("{ctx} {scope:?} t={threads}"));
            }
        }
    }
}

#[test]
fn test_gemm_parallel_per_tile_any_kernel_shape_bitwise() {
    // Whatever shape the plan's kernel has, both scopes give the sequential
    // driver's bits: kernels of the shapes this crate's plans carry (among
    // them f32 16×6, the x86 candidate's per-call kernel, which reaches the
    // per-tile scope from five threads), odd ones, and every crate f64
    // kernel that may run here.
    let mut f64_kernels = vec![
        shaped_f64::<8, 8>(),
        shaped_f64::<8, 6>(),
        shaped_f64::<16, 6>(),
        shaped_f64::<4, 4>(),
        shaped_f64::<16, 16>(),
        shaped_f64::<3, 5>(),
        shaped_f64::<1, 1>(),
    ];
    f64_kernels.extend(runnable_f64_kernels().into_iter().map(|(_, kernel)| kernel));
    check_kernel_shapes::<f64>(&f64_kernels);
    check_kernel_shapes::<f32>(&[
        shaped_f32::<16, 6>(),
        shaped_f32::<8, 8>(),
        shaped_f32::<16, 16>(),
        shaped_f32::<3, 5>(),
        f32::micro_kernel_dispatch(),
    ]);
}

// ------------------------------------------------------------------
// The per-tile scope.
// ------------------------------------------------------------------

/// Thread-count pairs of the per-tile checks, one per transpose combination
/// in turn: every combination runs on two thread counts, and every
/// consecutive pair of combinations covers 2, 3, 5 and 8.
const PER_TILE_THREAD_PAIRS: [[usize; 2]; 2] = [[2, 5], [3, 8]];

/// [`check`] with the per-tile scope only, over every transpose combination
/// of the type (each on two of the thread counts 2, 3, 5 and 8, all four
/// over every two combinations), and a poisoned C.
fn check_per_tile<T: Probe>(shapes: &[(usize, usize, usize)], transes: &[Trans]) {
    for &shape in shapes {
        let mut combo = 0usize;
        for &ta in transes {
            for &tb in transes {
                check::<T>(
                    shape,
                    (ta, tb),
                    None,
                    &PER_TILE_THREAD_PAIRS[combo % PER_TILE_THREAD_PAIRS.len()],
                    false,
                    &[PackScope::PerTile],
                );
                combo += 1;
            }
        }
        check::<T>(
            shape,
            (Trans::NoTrans, Trans::NoTrans),
            None,
            &[3, 8],
            true,
            &[PackScope::PerTile],
        );
        // The default blocking may hold all of k in one KC block (the cache
        // model's KC can exceed these depths); small explicit targets make
        // several KC, NC and MC blocks for every type.
        let small = GemmBlocking {
            mc: 64,
            kc: 72,
            nc: 120,
        };
        for combo in [
            (Trans::NoTrans, Trans::NoTrans),
            (Trans::Trans, Trans::ConjTrans),
        ] {
            check::<T>(
                shape,
                combo,
                Some(small),
                &[3, 5],
                false,
                &[PackScope::PerTile],
            );
        }
        check::<T>(
            shape,
            (Trans::ConjTrans, Trans::NoTrans),
            Some(small),
            &[8],
            true,
            &[PackScope::PerTile],
        );
    }
}

#[test]
fn test_gemm_parallel_per_tile_matches_sequential_bitwise_f64() {
    // A dimension of 512 or more (the auto-tuned blocking) with two default
    // KC blocks in both shapes; the small explicit blocking of
    // `check_per_tile` adds several KC, NC and MC blocks.
    check_per_tile::<f64>(
        &[(203, 517, 157), (37, 1100, 129)],
        &[Trans::NoTrans, Trans::Trans, Trans::ConjTrans],
    );
}

#[test]
fn test_gemm_parallel_per_tile_matches_sequential_bitwise_f32() {
    check_per_tile::<f32>(
        &[(150, 1000, 140), (64, 300, 1500)],
        &[Trans::NoTrans, Trans::Trans, Trans::ConjTrans],
    );
}

#[test]
fn test_gemm_parallel_per_tile_matches_sequential_bitwise_c64() {
    check_per_tile::<Complex64>(
        &[(213, 157, 137)],
        &[Trans::NoTrans, Trans::Trans, Trans::ConjTrans],
    );
}

#[test]
fn test_gemm_parallel_per_tile_matches_sequential_bitwise_c32() {
    check_per_tile::<Complex32>(
        &[(213, 157, 137)],
        &[Trans::NoTrans, Trans::Trans, Trans::ConjTrans],
    );
    // And the shared scope with several KC, NC and MC blocks.
    check::<Complex32>(
        (203, 150, 301),
        (Trans::ConjTrans, Trans::Trans),
        Some(GemmBlocking {
            mc: 64,
            kc: 48,
            nc: 96,
        }),
        &[3, 5],
        false,
        &SCOPES,
    );
}

#[test]
fn test_gemm_parallel_per_tile_many_blocks_and_strided_views() {
    // Several NC blocks (the last narrower), KC blocks and MC sub-blocks per
    // tile, with strided (transposed-view) operands and a C view.
    let blocking = GemmBlocking {
        mc: 24,
        kc: 32,
        nc: 54,
    };
    for threads in [2usize, 3, 5, 8] {
        check_strided_views(PackScope::PerTile, Some(blocking), threads);
        check_strided_views(PackScope::PerTile, None, threads);
    }
    for (ta, tb) in [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::ConjTrans),
        (Trans::ConjTrans, Trans::Trans),
    ] {
        check::<f64>(
            (203, 150, 301),
            (ta, tb),
            Some(blocking),
            &[2, 3, 5, 8],
            false,
            &[PackScope::PerTile],
        );
        check::<Complex64>(
            (203, 150, 301),
            (ta, tb),
            Some(blocking),
            &[3],
            false,
            &[PackScope::PerTile],
        );
    }
}

#[test]
fn test_gemm_parallel_per_tile_declines_a_single_tile() {
    // A C of one micro-tile (m = MR, n = NR) cannot be split into two tiles:
    // the per-tile runner (and the shared one) must return false without
    // touching C or counting a run, whatever the thread count, so the caller
    // runs the sequential driver. k and alpha are nonzero, so only the grid
    // can make the runner decline.
    let kernel = f64::micro_kernel_dispatch();
    let (m, k, n) = (kernel.shape.mr, 37usize, kernel.shape.nr);
    let a = fill::<f64>(m, k, 41);
    let b = fill::<f64>(k, n, 42);
    let c0 = fill::<f64>(m, n, 43);
    let blocking = fitted_blocking::<f64>(m, k, n);
    for threads in [2usize, 4, 8] {
        let par = Par::RayonWith(threads);
        assert_eq!(
            TileGrid::choose_per_tile(m, blocking.nc.min(n), &kernel.shape, threads).count(),
            1
        );
        let mut c = c0.clone();
        let before = scope_runs();
        let engaged = run_per_tile(
            Trans::NoTrans,
            Trans::NoTrans,
            0.75,
            &a.as_ref(),
            &b.as_ref(),
            -0.5,
            &mut c.as_mut(),
            &blocking,
            &kernel,
            par,
        );
        assert!(!engaged, "t={threads}: a single tile must be declined");
        assert_eq!(scope_runs(), before, "t={threads}: a declined call counted");
        assert_bits_eq(&c, &c0, &format!("per-tile runner t={threads}: C touched"));
        // The same through the scope dispatch, for both scopes.
        for scope in SCOPES {
            let engaged = try_gemm_blocked_parallel(
                Trans::NoTrans,
                Trans::NoTrans,
                0.75,
                &a.as_ref(),
                &b.as_ref(),
                -0.5,
                &mut c.as_mut(),
                &blocking,
                &kernel,
                scope,
                par,
            );
            assert!(
                !engaged,
                "{scope:?} t={threads}: a single tile must be declined"
            );
            assert_eq!(scope_runs(), before, "{scope:?} t={threads}: counted");
            assert_bits_eq(&c, &c0, &format!("{scope:?} t={threads}: C touched"));
        }
    }
}

/// The candidate grids of [`TileGrid::choose_per_tile`], recomputed: `rows`
/// up to the target tile count, `cols = min(⌈target / rows⌉, nu)`.
fn per_tile_candidates(
    m: usize,
    w: usize,
    shape: &MicroKernelShape,
    threads: usize,
) -> impl Iterator<Item = TileGrid> {
    let (mu, nu) = (m.div_ceil(shape.mr), w.div_ceil(shape.nr));
    let target = (PER_TILE_TILES_PER_THREAD * threads)
        .min(mu * nu / 4)
        .max(1);
    (1..=target.min(mu)).map(move |rows| TileGrid {
        rows,
        cols: target.div_ceil(rows).min(nu),
    })
}

/// Rounds of tiles per thread times the largest tile, in micro-tiles.
fn per_tile_span(
    grid: TileGrid,
    m: usize,
    w: usize,
    shape: &MicroKernelShape,
    threads: usize,
) -> usize {
    let (mu, nu) = (m.div_ceil(shape.mr), w.div_ceil(shape.nr));
    grid.count().div_ceil(threads) * mu.div_ceil(grid.rows) * nu.div_ceil(grid.cols)
}

#[test]
fn test_gemm_parallel_per_tile_grid_valid_and_near_square() {
    for &(mr, nr) in &[(8usize, 6usize), (8, 8), (16, 6), (4, 4), (2, 2)] {
        let shape = MicroKernelShape { mr, nr };
        for &m in &[1usize, 7, 64, 100, 513, 2048, 4096] {
            for &w in &[1usize, 6, 64, 517, 2048, 5712] {
                for &threads in &[1usize, 2, 3, 4, 5, 8, 12, 64] {
                    let grid = TileGrid::choose_per_tile(m, w, &shape, threads);
                    let (mu, nu) = (m.div_ceil(mr), w.div_ceil(nr));
                    let ctx = format!("{mr}x{nr} m={m} w={w} t={threads}: {grid:?}");
                    assert!((1..=mu).contains(&grid.rows), "{ctx}");
                    assert!((1..=nu).contains(&grid.cols), "{ctx}");
                    // About four tiles per thread, never twice that.
                    assert!(
                        grid.count() < 2 * PER_TILE_TILES_PER_THREAD * threads,
                        "{ctx}"
                    );
                    if mu * nu >= 8 {
                        assert!(grid.count() >= 2, "{ctx}");
                    }
                    // Enough micro-tiles: at least one tile per thread (the
                    // tie-break on packed elements may give up a tile when
                    // that does not enlarge the largest one).
                    if mu * nu >= 4 * PER_TILE_TILES_PER_THREAD * threads {
                        assert!(grid.count() >= threads, "{ctx}");
                    }
                    // The estimated span (rounds of the largest tile) is
                    // within 1/8 of the best candidate's.
                    let shortest = per_tile_candidates(m, w, &shape, threads)
                        .map(|g| per_tile_span(g, m, w, &shape, threads))
                        .min()
                        .unwrap_or(0);
                    let span = per_tile_span(grid, m, w, &shape, threads);
                    assert!(span * 8 <= shortest * 9, "{ctx}: span {span} vs {shortest}");
                }
            }
        }
    }
    // A squarer grid that costs a whole extra round loses: 512² on 11
    // threads takes 4 × 11 tiles (4 rounds of 128 × 48) rather than 7 × 7
    // (5 rounds of 80 × 78).
    assert_eq!(
        TileGrid::choose_per_tile(512, 512, &MicroKernelShape { mr: 8, nr: 6 }, 11),
        TileGrid { rows: 4, cols: 11 }
    );
    // Square blocks get near-square tiles (f64 2048² on 8 threads: 6 × 6
    // tiles of about 344 × 342, where ranking by the largest tile alone
    // would give 29 × 2 tiles of 72 × 1026): within a factor of two of each
    // other in elements from 2048² on, and of 3.5 below, where a squarer
    // grid would cost a whole extra round of tiles on some thread counts.
    let f64_shape = MicroKernelShape { mr: 8, nr: 6 };
    assert_eq!(
        TileGrid::choose_per_tile(2048, 2048, &f64_shape, 8),
        TileGrid { rows: 6, cols: 6 }
    );
    for &(mr, nr) in &[(8usize, 6usize), (8, 8), (16, 6), (4, 4)] {
        let shape = MicroKernelShape { mr, nr };
        for &n in &[512usize, 1000, 2048, 4096] {
            // Bound on the tile aspect ratio, in tenths.
            let bound = if n >= 2048 { 20 } else { 35 };
            for threads in 2..=16usize {
                let grid = TileGrid::choose_per_tile(n, n, &shape, threads);
                let (h, w) = (grid.tallest_rows(n, mr), grid.widest_cols(n, nr));
                assert!(
                    10 * h <= bound * w && 10 * w <= bound * h,
                    "{mr}x{nr} n={n} t={threads}: {grid:?} tiles {h} x {w}"
                );
            }
        }
    }
}

#[test]
fn test_gemm_parallel_scope_selection() {
    // The pure rule: the x86 candidate takes the per-tile scope from five
    // threads on, LEGACY and the aarch64 candidate never do, and a forced
    // scope wins.
    let at = |policy: GemmPolicy, threads: usize| GemmPlan::<f64>::new(policy).scope(threads);
    for threads in 1..=16usize {
        let x86 = if threads >= 5 {
            PackScope::PerTile
        } else {
            PackScope::Shared
        };
        assert_eq!(at(CANDIDATE_X86, threads), x86, "x86 candidate t={threads}");
        assert_eq!(at(LEGACY_POLICY, threads), PackScope::Shared);
        assert_eq!(at(CANDIDATE_AARCH64, threads), PackScope::Shared);
        for scope in SCOPES {
            let forced = GemmPlan::<f64>::new(LEGACY_POLICY).with_forced_scope(scope);
            assert_eq!(forced.scope(threads), scope);
        }
    }

    // And the scope the funnel actually takes, counted in the driver: the
    // x86 candidate's plan (blocked route forced, so the call reaches the 2D
    // driver on every target) runs Shared on 4 threads and PerTile on 5 and
    // 8, with the bits of the sequential driver either way.
    let (m, k, n) = (197usize, 211usize, 183usize);
    let a = fill::<f64>(m, k, 31);
    let b = fill::<f64>(k, n, 32);
    let c0 = fill::<f64>(m, n, 33);
    let run = |policy: GemmPolicy, par: Par| {
        let plan = GemmPlan::<f64>::new(policy).with_forced_route(GemmRoute::Blocked);
        let mut c = c0.clone();
        let before = scope_runs();
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            0.5,
            a.as_ref(),
            b.as_ref(),
            -2.0,
            c.as_mut(),
            par,
            BlockingSource::Default,
            &plan,
        );
        let after = scope_runs();
        (c, (after.0 - before.0, after.1 - before.1))
    };
    let (seq, none) = run(CANDIDATE_X86, Par::Seq);
    assert_eq!(none, (0, 0), "Par::Seq must not reach the 2D driver");
    for (policy, label) in [
        (CANDIDATE_X86, "x86 candidate"),
        (LEGACY_POLICY, "LEGACY"),
        (CANDIDATE_AARCH64, "aarch64 candidate"),
    ] {
        for threads in [2usize, 4, 5, 8] {
            let par = Par::RayonWith(threads);
            assert!(PARALLEL_GATE.should_parallelize(m * n * k, par));
            let (c, taken) = run(policy, par);
            let expected = if policy.per_tile_min_threads <= threads {
                (0, 1)
            } else {
                (1, 0)
            };
            assert_eq!(taken, expected, "{label} t={threads}: (Shared, PerTile)");
            assert_bits_eq(&c, &seq, &format!("{label} t={threads}"));
        }
    }

    // The public entry point follows the build's policy (`DEFAULT_POLICY`:
    // LEGACY, or the candidate of the target under
    // `--cfg oxiblas_gemm_candidate`).
    for threads in [4usize, 5] {
        let mut c = c0.clone();
        let before = scope_runs();
        gemm_transposed_with_par(
            Trans::NoTrans,
            Trans::NoTrans,
            0.5,
            a.as_ref(),
            b.as_ref(),
            -2.0,
            c.as_mut(),
            Par::RayonWith(threads),
        );
        let after = scope_runs();
        // The rule checked above, for the build's policy.
        let expected = match GemmPlan::<f64>::new(DEFAULT_POLICY).scope(threads) {
            PackScope::Shared => (1, 0),
            PackScope::PerTile => (0, 1),
        };
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            expected,
            "public API t={threads}"
        );
        assert_bits_eq(&c, &seq, &format!("public API t={threads}"));
    }
}

#[test]
fn test_gemm_parallel_per_tile_workers_reuse_buffers() {
    use crate::level3::gemm_packing::{
        PACK_CACHE_MAX_BYTES, PackSlot, pack_cache_allocations, pack_cache_bytes,
        pack_cache_capacity,
    };
    // Every per-tile task borrows its A and B buffers from the cache of the
    // worker that runs it; a worker allocates them once (all tasks of these
    // calls need the same lengths) and reuses them in every later task and
    // call. Without the cache every task of every call would allocate both.
    let workers = 4usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("thread pool");
    let (m, k, n) = (300usize, 260usize, 280usize);
    let a = fill::<f64>(m, k, 61);
    let b = fill::<f64>(k, n, 62);
    let plan = GemmPlan::<f64>::new(LEGACY_POLICY)
        .with_forced_route(GemmRoute::Blocked)
        .with_forced_scope(PackScope::PerTile);
    // Two KC blocks: the B buffer of a tile is refilled inside the task.
    let targets = GemmBlocking {
        mc: 96,
        kc: 136,
        nc: 1 << 20,
    };
    let mut c_seq: Mat<f64> = Mat::zeros(m, n);
    gemm_planned(
        Trans::NoTrans,
        Trans::NoTrans,
        1.0,
        a.as_ref(),
        b.as_ref(),
        0.0,
        c_seq.as_mut(),
        Par::Seq,
        BlockingSource::Explicit(targets),
        &plan,
    );
    let calls = 6usize;
    let per_tile_runs = pool.install(|| {
        let before = scope_runs().1;
        for call in 0..calls {
            let mut c = Mat::filled(m, n, f64::NAN);
            gemm_planned(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                a.as_ref(),
                b.as_ref(),
                0.0,
                c.as_mut(),
                Par::Rayon,
                BlockingSource::Explicit(targets),
                &plan,
            );
            assert_bits_eq(&c, &c_seq, &format!("call {call}"));
        }
        scope_runs().1 - before
    });
    assert_eq!(per_tile_runs, calls, "every call must run per tile");
    // (allocations, cached A elements, cached B elements, cached bytes) per
    // worker; the pool's threads are new, so their counters start at zero.
    let stats = pool.broadcast(|_| {
        (
            pack_cache_allocations(),
            pack_cache_capacity::<f64>(PackSlot::A),
            pack_cache_capacity::<f64>(PackSlot::B),
            pack_cache_bytes(),
        )
    });
    let tiles = TileGrid::choose_per_tile(m, n, &f64::micro_kernel_shape(), workers).count();
    assert!(tiles >= 2 * workers, "{tiles} tiles");
    let mut busy = 0;
    for (worker, &(allocations, cap_a, cap_b, bytes)) in stats.iter().enumerate() {
        let ctx = format!("worker {worker}: {stats:?}");
        assert!(allocations <= 2, "{ctx}: allocated in more than one task");
        assert!(bytes <= PACK_CACHE_MAX_BYTES, "{ctx}");
        if allocations > 0 {
            busy += 1;
            assert!(cap_a > 0 && cap_b > 0, "{ctx}: buffers not kept");
        }
    }
    assert!(busy > 0, "no worker ran a tile: {stats:?}");
    let total: usize = stats.iter().map(|s| s.0).sum();
    assert!(
        total <= 2 * workers && total < 2 * tiles * calls,
        "{total} allocations for {calls} calls of {tiles} tiles"
    );
}
