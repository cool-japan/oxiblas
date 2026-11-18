//! The case tables of the groups (const arrays of [`Shape`]).

use super::super::trsm::Trans;

/// Element type of a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dtype {
    F32,
    F64,
}

impl Dtype {
    /// The name in labels and the `dtype` column.
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

/// Both element types, f64 first.
pub(super) const BOTH: [Dtype; 2] = [Dtype::F64, Dtype::F32];
/// Only f32.
#[cfg(target_arch = "x86_64")]
pub(super) const F32_ONLY: [Dtype; 1] = [Dtype::F32];

/// The transposes of A and B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TransPair {
    pub(super) a: Trans,
    pub(super) b: Trans,
}

impl TransPair {
    pub(super) const NN: Self = Self {
        a: Trans::NoTrans,
        b: Trans::NoTrans,
    };
    pub(super) const TN: Self = Self {
        a: Trans::Trans,
        b: Trans::NoTrans,
    };
    pub(super) const NT: Self = Self {
        a: Trans::NoTrans,
        b: Trans::Trans,
    };
    pub(super) const TT: Self = Self {
        a: Trans::Trans,
        b: Trans::Trans,
    };

    /// `NN`, `TN`, `NT` or `TT` (`C` for a conjugate transpose).
    pub(super) fn name(self) -> String {
        let letter = |t: Trans| match t {
            Trans::NoTrans => 'N',
            Trans::Trans => 'T',
            Trans::ConjTrans => 'C',
        };
        [letter(self.a), letter(self.b)].iter().collect()
    }
}

/// `beta` of a case (alpha is always 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Beta {
    /// `C = op(A)·op(B)`.
    Zero,
    /// `C += op(A)·op(B)` (the LAPACK update shape).
    One,
}

/// For which element types a shape is in the quick subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Quick {
    No,
    Both,
    F64,
    F32,
}

impl Quick {
    /// Whether a case of `dtype` is in the quick subset.
    pub(super) fn includes(self, dtype: Dtype) -> bool {
        match self {
            Self::No => false,
            Self::Both => true,
            Self::F64 => dtype == Dtype::F64,
            Self::F32 => dtype == Dtype::F32,
        }
    }
}

/// One row of a case table: `op(A)` is `m × k`, `op(B)` is `k × n`.
/// `target`: the group names it an intended target (B-cap and cache-limit
/// groups; the other groups derive their intended targets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) trans: TransPair,
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) quick: Quick,
    pub(super) target: bool,
    pub(super) beta: Beta,
}

/// The shape `trans` `m × k` by `k × n`: not quick, not a named target,
/// `beta = 0`.
const fn shape(trans: TransPair, m: usize, k: usize, n: usize) -> Shape {
    Shape {
        trans,
        m,
        k,
        n,
        quick: Quick::No,
        target: false,
        beta: Beta::Zero,
    }
}

const fn nn(m: usize, k: usize, n: usize) -> Shape {
    shape(TransPair::NN, m, k, n)
}

const fn tn(m: usize, k: usize, n: usize) -> Shape {
    shape(TransPair::TN, m, k, n)
}

const fn nt(m: usize, k: usize, n: usize) -> Shape {
    shape(TransPair::NT, m, k, n)
}

const fn tt(m: usize, k: usize, n: usize) -> Shape {
    shape(TransPair::TT, m, k, n)
}

impl Shape {
    /// In the quick subset for `quick`.
    const fn quick(mut self, quick: Quick) -> Self {
        self.quick = quick;
        self
    }

    /// A named (intended) target.
    const fn target(mut self) -> Self {
        self.target = true;
        self
    }

    /// With `beta = 1`.
    const fn beta_one(mut self) -> Self {
        self.beta = Beta::One;
        self
    }
}

use Quick::{Both as QB, F32 as QS, F64 as QD};

#[cfg(target_arch = "x86_64")]
/// `ab_seq_f32_kernel` (f32 only): squares, m-narrow and n-narrow shapes,
/// TN/NT cubes.
pub(super) const F32_KERNEL: [Shape; 25] = [
    nn(64, 64, 64).quick(QB),
    nn(128, 128, 128),
    nn(256, 256, 256).quick(QB),
    nn(512, 512, 512),
    nn(1024, 1024, 1024),
    nn(2048, 2048, 2048),
    nn(4096, 4096, 4096),
    nn(8, 4096, 4096).quick(QB),
    nn(12, 4096, 4096),
    nn(16, 4096, 4096).quick(QB),
    nn(24, 4096, 4096),
    nn(4096, 4096, 8),
    nn(4096, 4096, 12),
    nn(4096, 4096, 16),
    nn(4096, 4096, 24),
    tn(33, 33, 33).quick(QB),
    tn(48, 48, 48),
    tn(64, 64, 64),
    tn(96, 96, 96),
    tn(128, 128, 128).quick(QB),
    nt(33, 33, 33),
    nt(48, 48, 48),
    nt(64, 64, 64),
    nt(96, 96, 96),
    nt(128, 128, 128),
];

#[cfg(target_arch = "x86_64")]
/// `ab_seq_small_simd`: NN cubes and rectangles, and a `beta = 1` subset.
pub(super) const SMALL_SIMD: [Shape; 21] = [
    nn(5, 5, 5),
    nn(8, 8, 8),
    nn(12, 12, 12).quick(QD),
    nn(16, 16, 16),
    nn(24, 24, 24),
    nn(32, 32, 32),
    nn(40, 40, 40),
    nn(48, 48, 48).quick(QS),
    nn(64, 64, 64),
    nn(80, 80, 80),
    nn(96, 96, 96).quick(QD),
    nn(4, 64, 4).quick(QD),
    nn(5, 256, 4),
    nn(8, 512, 8),
    nn(16, 128, 64),
    nn(64, 8, 64),
    nn(40, 32, 80),
    nn(8, 8, 8).beta_one(),
    nn(32, 32, 32).beta_one().quick(QB),
    nn(64, 64, 64).beta_one(),
    nn(16, 128, 64).beta_one(),
];

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
/// `ab_seq_small_trans_copy` on every architecture: TN, NT and TT cubes up
/// to 32 and rectangles, and a `beta = 1` subset.
pub(super) const SMALL_TRANS_COPY: [Shape; 31] = [
    tn(4, 4, 4),
    tn(6, 6, 6).quick(QD),
    tn(8, 8, 8),
    tn(12, 12, 12),
    tn(16, 16, 16),
    tn(24, 24, 24),
    tn(32, 32, 32),
    tn(16, 64, 8).quick(QD),
    tn(32, 8, 48),
    tn(8, 128, 16),
    nt(4, 4, 4),
    nt(6, 6, 6),
    nt(8, 8, 8),
    nt(12, 12, 12),
    nt(16, 16, 16).quick(QS),
    nt(24, 24, 24),
    nt(32, 32, 32),
    nt(16, 64, 8),
    nt(32, 8, 48),
    nt(8, 128, 16),
    tt(4, 4, 4),
    tt(6, 6, 6),
    tt(8, 8, 8),
    tt(12, 12, 12),
    tt(16, 16, 16),
    tt(24, 24, 24),
    tt(32, 32, 32),
    tn(16, 16, 16).beta_one().quick(QB),
    nt(16, 16, 16).beta_one(),
    tt(16, 16, 16).beta_one(),
    tn(16, 64, 8).beta_one(),
];

#[cfg(target_arch = "x86_64")]
/// `ab_seq_small_trans_copy` on x86-64 only: the cubes 40 to 80 (the wider
/// copy range of the AVX2 small kernels).
pub(super) const SMALL_TRANS_COPY_X86: [Shape; 12] = [
    tn(40, 40, 40),
    tn(48, 48, 48),
    tn(64, 64, 64),
    tn(80, 80, 80),
    nt(40, 40, 40),
    nt(48, 48, 48),
    nt(64, 64, 64),
    nt(80, 80, 80),
    tt(40, 40, 40),
    tt(48, 48, 48).quick(QS),
    tt(64, 64, 64),
    tt(80, 80, 80),
];

/// The sizes of a thin group: rows `(1, s, s)` and columns `(s, s, 1)` with
/// both transposes of the large operand, dots `(1, k, 1)` (NN and TT), and
/// the non-thin neighbours `(2, s, s)`, `(s, s, 2)`. The first size of each
/// list is in the quick subset.
///
/// With `boundary`, the group adds the boundary set of the measured scope
/// ([`THIN_BOUNDARY_DIAG`] and the lists after it): the shapes on which the
/// scope was decided with the keep rule.
///
/// The grid adds, for every vector length `v` of `grid_v` and depth `k` of
/// `grid_k` with `v·k <= THIN_GRID_MAX_WORK`, the four layouts of that GEMV:
/// columns `(v, k, 1)` as NN (the axpy form: `k` columns of `v` rows) and TN
/// (the dot form: `v` dots of length `k`), rows `(1, k, v)` as NN (the dot
/// form) and NT (the axpy form). It is where the thin route has to prove
/// itself against the small route (few rows, short dots, tiny products),
/// and the measurement a scope is derived from.
#[derive(Debug, Clone, Copy)]
pub(super) struct ThinSizes {
    pub(super) sizes: &'static [usize],
    pub(super) dots: &'static [usize],
    pub(super) near: &'static [usize],
    /// The `beta = 1` subset (its first shape is quick).
    pub(super) beta_one: &'static [Shape],
    /// The boundary set of the measured scope.
    pub(super) boundary: bool,
    /// Vector lengths of the grid (empty: no grid).
    pub(super) grid_v: &'static [usize],
    /// Depths of the grid.
    pub(super) grid_k: &'static [usize],
    /// Vector lengths and depths of the coarse grid of the other four
    /// layouts: columns `(v, k, 1)` as NT and TT, rows `(1, k, v)` as TN and
    /// TT (the same two GEMV forms; the vector of the other operand is read
    /// from a row or a column).
    pub(super) other_v: &'static [usize],
    pub(super) other_k: &'static [usize],
}

/// Largest `v·k` of a grid shape (larger ones are covered by the squares).
pub(super) const THIN_GRID_MAX_WORK: usize = 1 << 22;

/// The boundary set, part 1: the points `(v, k)` with `v·k` = 1024 or 2048
/// (from the floor of the scope on) and with `v·k` = 512 (below it), each as
/// the four small layouts: columns `(v, k, 1)` NN and TN, rows `(1, k, v)`
/// NN and NT.
pub(super) const THIN_BOUNDARY_DIAG: [(usize, usize); 17] = [
    (2, 1024),
    (4, 256),
    (8, 128),
    (16, 64),
    (32, 32),
    (64, 16),
    (128, 8),
    (256, 4),
    (1024, 2),
    (2, 256),
    (4, 128),
    (8, 64),
    (16, 32),
    (32, 16),
    (64, 8),
    (128, 4),
    (256, 2),
];

/// The boundary set, part 2: points just outside a layout's limit, above
/// the floor (TN columns below 16 deep or of 2 dots; NN and NT rows of 2
/// columns; NT rows 2 deep).
pub(super) const THIN_BOUNDARY_OUTSIDE: [Shape; 9] = [
    tn(128, 8, 1),
    tn(256, 8, 1),
    tn(2, 4096, 1),
    nn(1, 1024, 2),
    nt(1, 1024, 2),
    nn(1, 4096, 2),
    nt(1, 4096, 2),
    nt(1, 2, 1024),
    nt(1, 2, 4096),
];

/// The boundary set, part 3: the four layouts outside the scope (NT and TT
/// columns, TN and TT rows), each at 1024 multiply-adds and above the small
/// threshold.
pub(super) const THIN_BOUNDARY_OTHER: [Shape; 8] = [
    nt(64, 16, 1),
    tt(64, 16, 1),
    tn(1, 16, 64),
    tt(1, 16, 64),
    nt(1024, 64, 1),
    tt(1024, 64, 1),
    tn(1, 64, 1024),
    tt(1, 64, 1024),
];

/// `ab_seq_thin`: the decision group of the thin route (the original shapes,
/// two more dots and the boundary set).
pub(super) const SEQ_THIN: ThinSizes = ThinSizes {
    sizes: &[256, 1024, 4096, 16_384],
    dots: &[4096, 65_536, 1 << 20, 1024, 32_768],
    near: &[256, 1024, 4096],
    beta_one: &[
        nn(1, 256, 256),
        nn(1, 1024, 1024),
        nn(1024, 1024, 1),
        tn(1024, 1024, 1),
        nn(1, 65_536, 1),
    ],
    boundary: true,
    grid_v: &[],
    grid_k: &[],
    other_v: &[],
    other_k: &[],
};

/// `ab_seq_thin_grid`: the information group (the original shapes, more
/// dots, the dense grid of the four small layouts and the coarse grid of
/// the other four).
pub(super) const SEQ_THIN_GRID: ThinSizes = ThinSizes {
    sizes: &[256, 1024, 4096, 16_384],
    dots: &[4096, 65_536, 1 << 20, 8, 64, 512, 32_768],
    near: &[256, 1024, 4096],
    beta_one: &[
        nn(1, 256, 256),
        nn(1, 1024, 1024),
        nn(1024, 1024, 1),
        tn(1024, 1024, 1),
        nn(1, 65_536, 1),
    ],
    boundary: false,
    grid_v: &[
        2, 4, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 4096, 16_384, 32_768,
    ],
    grid_k: &[2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 32_768],
    other_v: &[2, 4, 8, 16, 64, 1024, 32_768],
    other_k: &[2, 4, 16, 64, 256, 4096],
};

/// `ab_par_thin`.
#[cfg(feature = "parallel")]
pub(super) const PAR_THIN: ThinSizes = ThinSizes {
    sizes: &[1024, 4096, 16_384],
    dots: &[65_536, 1 << 20],
    near: &[1024, 4096],
    beta_one: &[nn(1, 4096, 4096), nn(4096, 4096, 1)],
    boundary: false,
    grid_v: &[],
    grid_k: &[],
    other_v: &[],
    other_k: &[],
};

/// The shapes of a thin group (see [`ThinSizes`]).
pub(super) fn thin_shapes(sizes: &ThinSizes) -> Vec<Shape> {
    let quick = |i: usize| if i == 0 { QB } else { Quick::No };
    let mut out = Vec::new();
    for (i, &s) in sizes.sizes.iter().enumerate() {
        let rows = [nn(1, s, s), nn(s, s, 1), nt(1, s, s), tn(s, s, 1)];
        out.extend(rows.map(|shape| shape.quick(quick(i))));
    }
    for (i, &k) in sizes.dots.iter().enumerate() {
        out.extend([nn(1, k, 1), tt(1, k, 1)].map(|shape| shape.quick(quick(i))));
    }
    for (i, &s) in sizes.near.iter().enumerate() {
        out.extend([nn(2, s, s), nn(s, s, 2)].map(|shape| shape.quick(quick(i))));
    }
    for (i, shape) in sizes.beta_one.iter().enumerate() {
        out.push(shape.beta_one().quick(quick(i)));
    }
    // The squares above are grid points too: no shape twice.
    let mut more = Vec::new();
    if sizes.boundary {
        for &(v, k) in &THIN_BOUNDARY_DIAG {
            more.extend([nn(v, k, 1), tn(v, k, 1), nn(1, k, v), nt(1, k, v)]);
        }
        more.extend(THIN_BOUNDARY_OUTSIDE);
        more.extend(THIN_BOUNDARY_OTHER);
    }
    for &v in sizes.grid_v {
        for &k in sizes.grid_k {
            if v.saturating_mul(k) <= THIN_GRID_MAX_WORK {
                more.extend([nn(v, k, 1), tn(v, k, 1), nn(1, k, v), nt(1, k, v)]);
            }
        }
    }
    for &v in sizes.other_v {
        for &k in sizes.other_k {
            if v.saturating_mul(k) <= THIN_GRID_MAX_WORK {
                more.extend([nt(v, k, 1), tt(v, k, 1), tn(1, k, v), tt(1, k, v)]);
            }
        }
    }
    let key = |s: &Shape| (s.trans, s.m, s.k, s.n, s.beta);
    for shape in more {
        if !out.iter().any(|o| key(o) == key(&shape)) {
            out.push(shape);
        }
    }
    out
}

/// The B-cap groups: the issue #23 thin-M shapes (one MC block) and a
/// shallow one (several MC blocks, one KC block) as named targets; small B
/// blocks, deep `k` and squares elsewhere.
pub(super) const B_CAP: [Shape; 8] = [
    nn(64, 512, 32_768).target().quick(QS),
    nn(128, 256, 16_384).target().quick(QD),
    nn(32, 1024, 32_768).target(),
    nn(1024, 256, 16_384).target(),
    nn(64, 256, 1024).quick(QD),
    nn(4096, 2048, 256),
    nn(1024, 1024, 1024).quick(QS),
    nn(2048, 2048, 2048),
];

/// `ab_seq_pack_cache`, both runs: thin-M shapes whose packed blocks can
/// outgrow 8 MiB, and the shallow B-cap shape (several MC blocks, one KC
/// block: with the cap on, its B block is capped at the room the limit
/// leaves beside A, so it is the case where the limit matters with the cap)
/// as named targets; small squares elsewhere.
pub(super) const PACK_CACHE: [Shape; 6] = [
    nn(64, 512, 32_768).target(),
    nn(128, 256, 16_384).target().quick(QD),
    nn(1024, 256, 16_384).target().quick(QS),
    nn(256, 256, 256).quick(QD),
    nn(512, 512, 512),
    nn(1024, 1024, 1024),
];

/// `ab_seq_pack_cache`, the cap-off run only: a large square (with the cap
/// on it repeats the cap-off measurement). 4096³ is trimmed to keep one pass
/// within its time budget: it was the heaviest case of all (about 18 of 70
/// minutes on the M3, 10 of 82 on the x86 desktop) and shows the effect of
/// 2048³ at a tenth of its weight in the ratio.
pub(super) const PACK_CACHE_CAP_OFF: [Shape; 1] = [nn(2048, 2048, 2048).target()];

#[cfg(feature = "parallel")]
/// `ab_par_pack_scope`: squares, a tall and a wide shape, TN and NT. 4096³ is
/// trimmed to keep one pass within its time budget (it took about 33 of the
/// 42 minutes of this group over the x86 thread counts 2 to 16).
pub(super) const PACK_SCOPE: [Shape; 9] = [
    nn(96, 96, 96).quick(QD),
    nn(256, 256, 256).quick(QD),
    nn(512, 512, 512).quick(QS),
    nn(1024, 1024, 1024),
    nn(2048, 2048, 2048),
    nn(8192, 512, 256),
    nn(256, 512, 8192),
    tn(1024, 1024, 1024),
    nt(2048, 2048, 2048),
];
