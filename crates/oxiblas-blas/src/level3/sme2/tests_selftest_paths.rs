//! The path table of the first-use self-test (`selftest.rs`): every path
//! class of both macro-kernels is taken by the self-test's own block list.
//!
//! From the block list ([`self_test_blocks`]) it computes, for the tile
//! height and panel width of each element type, every path a block takes
//! through that type's macro-kernel, and requires the set of path classes
//! that the kernels' structure defines (the [`KernelPaths`] descriptions).
//! The descriptions are tied to the kernels by `tests_selftest_text.rs`,
//! which derives the structure from the registered template text and
//! requires it to equal them ([`model_matches_text`]); the complete tables
//! below run that comparison too.
//!
//! Everything here is plain Rust and runs on every host.

use core::cell::RefCell;
use std::collections::BTreeSet;

use super::args::{Epilogue, MR, MacroArgs, PREFETCH_C, SmeElement};
use super::kernel_f32::SME_KERNELS as F32_KERNELS;
use super::kernel_f64::SME_KERNELS as F64_KERNELS;
use super::oracle::oracle_block;
use super::selftest::{
    ALayout, CASES, Case, KernelFn, SCALARS, SWEEP_BLOCKS_PER_VARIANT, SWEEP_MIDDLE_BLOCKS,
    online_cases, online_sweep_cases, self_test_blocks, self_test_calls, self_test_with,
    sweep_cases,
};
use super::tests_oracle::TestFloat;
use super::tests_selftest_text::model_matches_text;

/// The structure of one macro-kernel as data: what decides the path a block
/// takes through it.
///
/// Both kernels (`kernel_f32.rs`, `kernel_f64.rs`) walk column panels of
/// `nr` columns and, inside each, row panels of 64 rows. For one (row panel,
/// column panel) pair the path is:
///
/// - the row variant: the number of live tiles `L = ceil(rows / tile_rows)`,
///   `1..=variants`, where `rows = min(64, m - i0)`; each variant has its own
///   zeroing, depth loop and remainder, and the epilogue's tile chain stops
///   after tile `L`;
/// - the depth: the loop unrolled by `unroll` makes `kc / unroll` passes
///   (none, or two and more, which takes its back branch and pointer steps)
///   and then a remainder of `kc % unroll` steps;
/// - the epilogue mode, M0 to M3, and for M3 the branch of the contract
///   (`alpha == 1`, `beta == 1`, neither) that the scalars select;
/// - the column panel: full (`nr` columns) or the partial last one;
/// - the position of the row panel: the only one, the first of several
///   (always full), a full one between two others, or the last one after a
///   full one. The second row panel of a block gets its A and C pointers
///   from the first application of the row-panel advance, the third from the
///   second: a middle panel implies a third panel behind it with the same
///   depth, A layout, epilogue and column panel, so requiring the middle
///   panel's classes requires the second application in each of them;
/// - the A layout: packed, or in place;
/// - the last live tile of the row panel: full, or holding a single row (the
///   lower edge of the variant, where the row predicates and the tile chain
///   cut);
/// - the pass over A ([`Pass`]): the plain variants on the call's A, the
///   packing twins in column panel 0 of a call with online packing, or the
///   plain variants on the buffer in its later column panels (whose A layout
///   is then the packed one);
/// - the prefetch block before the depth loop: run (`PREFETCH_C`), or
///   skipped.
///
/// A change of a kernel's structure (another row variant, another unrolling
/// of the depth loop, another epilogue mode or branch, another edge that the
/// code treats on its own) must be reflected here and in [`paths_of_call`];
/// the table then fails until the sweep of `selftest.rs` takes the new paths.
/// What the kernel text shows of it (see `derive_structure` in
/// `tests_selftest_text.rs`) is compared with the registered templates by
/// [`model_matches_text`]: the panel width (`nr`, from the column-panel step
/// and the clamp of the live columns), the row variants and the packing
/// twins and which row counts take them, the depth steps per pass and the
/// remainder, the four epilogue modes and their tile chains, the switch to
/// the buffer after column panel 0, the prefetch block hint for hint, and
/// that the text has no other branch. So a change of the text there that is
/// not made here fails as well. The tile height is the elements of one
/// 64-byte vector (a constant: SVL 512), checked against the tiles the text
/// uses and the dispatch thresholds.
#[derive(Clone, Copy, Debug)]
pub(super) struct KernelPaths {
    /// The element type.
    pub(super) name: &'static str,
    /// Rows of one ZA tile (`T::TILE_ROWS`).
    pub(super) tile_rows: usize,
    /// Columns of one B panel (`T::NR`).
    pub(super) nr: usize,
    /// Row variants: `1..=variants` live tiles.
    pub(super) variants: usize,
    /// Depth steps per pass of the unrolled depth loop.
    pub(super) unroll: usize,
    /// Online packing of A: packing twins of the variants in column panel 0
    /// and the plain variants on the buffer after it.
    pub(super) packing: bool,
    /// The prefetch block of C before each tile's depth loop.
    pub(super) prefetch: bool,
}

/// `kernel_f32.rs`: 16-row tiles, 16-column panels, variants of 1 to 4
/// tiles (labels 41 to 48 and 61 to 64) and their packing twins (65 to 78),
/// a depth loop unrolled by two, the prefetch block (32, 33) (checked
/// against the text by `test_sme2_selftest_paths_model_matches_text`).
pub(super) const F32_PATHS: KernelPaths = KernelPaths {
    name: "f32",
    tile_rows: 16,
    nr: 16,
    variants: 4,
    unroll: 2,
    packing: true,
    prefetch: true,
};

/// `kernel_f64.rs`: 8-row tiles, 8-column panels, variants of 1 to 8 tiles
/// (labels 41 to 48, 61 to 68 and 71 to 78) and their packing twins (141 to
/// 178), a depth loop unrolled by two, the prefetch block (32, 33) (checked
/// against the text as for [`F32_PATHS`]).
pub(super) const F64_PATHS: KernelPaths = KernelPaths {
    name: "f64",
    tile_rows: 8,
    nr: 8,
    variants: 8,
    unroll: 2,
    packing: true,
    prefetch: true,
};

/// The epilogue of a call: the mode and, for M3, the branch of the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EpilogueClass {
    /// `alpha == 1`, `beta == 0`.
    M0,
    /// `alpha == 1`, `beta == 1`.
    M1,
    /// `alpha != 1`, `beta == 0`.
    M2,
    /// M3 with `alpha == 1`.
    M3AlphaOne,
    /// M3 with `beta == 1`.
    M3BetaOne,
    /// M3 with neither.
    M3General,
}

/// Every [`EpilogueClass`].
const EPILOGUES: [EpilogueClass; 6] = [
    EpilogueClass::M0,
    EpilogueClass::M1,
    EpilogueClass::M2,
    EpilogueClass::M3AlphaOne,
    EpilogueClass::M3BetaOne,
    EpilogueClass::M3General,
];

/// The column panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Columns {
    /// `nr` columns.
    Full,
    /// The partial last panel.
    Partial,
}

/// The position of the row panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Position {
    /// The first and last row panel (`m ≤ 64`).
    Only,
    /// The first of several row panels (always 64 full rows).
    Leading,
    /// A row panel that is neither the first nor the last (always 64 full
    /// rows; only in blocks of three or more row panels). It gets its
    /// pointers from the first application of the row-panel advance; the
    /// panel after it, from the second (see [`KernelPaths`]).
    Middle,
    /// The last row panel after a full one.
    Last,
}

/// The A layout of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layout {
    /// `a_depth = 64`, `a_panel = 64 · kc`.
    Packed,
    /// Any other strides (in place: `a_depth = lda`, `a_panel = 64`).
    InPlace,
}

/// The pass over A of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Pass {
    /// The plain variants on the call's A (no online packing).
    Plain,
    /// The packing twins: column panel 0 of a call with online packing.
    Packing,
    /// The plain variants on the buffer: the later column panels of a call
    /// with online packing.
    Buffer,
}

/// The last live tile of the row panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LastTile {
    /// All `tile_rows` rows live.
    Full,
    /// One live row.
    OneRow,
    /// Between the two (never required).
    Other,
}

/// One path through a kernel: what one (row panel, column panel) pair of one
/// call takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Path {
    /// Live tiles (the row variant).
    tiles: usize,
    /// Passes of the unrolled depth loop: 0, 1, or 2 for two and more.
    passes: usize,
    /// Depth steps after the loop (`kc % unroll`).
    remainder: usize,
    /// Mode and M3 branch.
    epilogue: EpilogueClass,
    /// Full or partial column panel.
    columns: Columns,
    /// Position of the row panel.
    position: Position,
    /// A layout.
    layout: Layout,
    /// The last live tile.
    last_tile: LastTile,
    /// The pass over A.
    pass: Pass,
    /// Whether the prefetch block runs.
    prefetch: bool,
}

/// A class of paths: a [`Path`] reduced to some of its dimensions (the
/// others are `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Pattern {
    /// See [`Path`].
    tiles: Option<usize>,
    /// See [`Path`].
    passes: Option<usize>,
    /// See [`Path`].
    remainder: Option<usize>,
    /// See [`Path`].
    epilogue: Option<EpilogueClass>,
    /// See [`Path`].
    columns: Option<Columns>,
    /// See [`Path`].
    position: Option<Position>,
    /// See [`Path`].
    layout: Option<Layout>,
    /// See [`Path`].
    last_tile: Option<LastTile>,
    /// See [`Path`].
    pass: Option<Pass>,
    /// See [`Path`].
    prefetch: Option<bool>,
}

/// A dimension of [`Path`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dim {
    /// [`Path::tiles`].
    Tiles,
    /// [`Path::passes`].
    Passes,
    /// [`Path::remainder`].
    Remainder,
    /// [`Path::epilogue`].
    Epilogue,
    /// [`Path::columns`].
    Columns,
    /// [`Path::position`].
    Position,
    /// [`Path::layout`].
    Layout,
    /// [`Path::last_tile`].
    LastTile,
    /// [`Path::pass`].
    Pass,
    /// [`Path::prefetch`].
    Prefetch,
}

/// One requirement: every combination of the values of `dims` that a
/// possible path of one of the passes `passes` has must be taken by some
/// call (other dimensions free; every requirement keeps [`Dim::Pass`] or
/// names all passes, so a path of one pass never stands in for another).
#[derive(Clone, Copy, Debug)]
struct Requirement {
    /// The dimensions combined.
    dims: &'static [Dim],
    /// The passes whose paths it requires.
    passes: &'static [Pass],
}

/// The requirements.
///
/// 1. Every plain row variant, with every depth (`kc = 1`, even, odd), in
///    every epilogue mode and M3 branch, with a full and a partial column
///    panel, as the only row panel and as the last one after a full one,
///    packed and in place; and the full variant also as the first of
///    several row panels and as a row panel between two others.
/// 2. The same with the last live tile full and holding one row, per
///    position of the row panel ...
/// 3. ... and per A layout.
/// 4. to 6. The same for the packing twins (column panel 0, so its column
///    panel is not combined), and 5. also for the plain variants on the
///    buffer (whose layout is the packed one).
/// 7. The prefetch block run and skipped before each pass, in a full and a
///    partial column panel.
/// 8. The prefetch block run and skipped before each pass, at every exit of
///    its tile chain (the live tiles) and in every position of the row
///    panel.
const REQUIREMENTS: [Requirement; 8] = [
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Columns,
            Dim::Position,
            Dim::Layout,
        ],
        passes: &[Pass::Plain],
    },
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Columns,
            Dim::Position,
            Dim::LastTile,
        ],
        passes: &[Pass::Plain],
    },
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Columns,
            Dim::Layout,
            Dim::LastTile,
        ],
        passes: &[Pass::Plain],
    },
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Position,
            Dim::Layout,
        ],
        passes: &[Pass::Packing],
    },
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Position,
            Dim::LastTile,
        ],
        passes: &[Pass::Packing, Pass::Buffer],
    },
    Requirement {
        dims: &[
            Dim::Pass,
            Dim::Tiles,
            Dim::Passes,
            Dim::Remainder,
            Dim::Epilogue,
            Dim::Layout,
            Dim::LastTile,
        ],
        passes: &[Pass::Packing],
    },
    Requirement {
        dims: &[Dim::Prefetch, Dim::Pass, Dim::Columns],
        passes: &[Pass::Plain, Pass::Packing, Pass::Buffer],
    },
    Requirement {
        dims: &[Dim::Prefetch, Dim::Pass, Dim::Tiles, Dim::Position],
        passes: &[Pass::Plain, Pass::Packing, Pass::Buffer],
    },
];

/// The passes over A of a kernel with structure `desc`.
fn passes_of(desc: &KernelPaths) -> Vec<Pass> {
    if desc.packing {
        vec![Pass::Plain, Pass::Packing, Pass::Buffer]
    } else {
        vec![Pass::Plain]
    }
}

/// Every path that the structure `desc` makes possible and that the table
/// requires: the passes are none or two and more (one pass never takes the
/// loop's back branch), the last tile full or holding one row, the buffer
/// read in the packed layout.
fn possible_paths(desc: &KernelPaths) -> Vec<Path> {
    let prefetches: &[bool] = if desc.prefetch {
        &[false, true]
    } else {
        &[false]
    };
    let mut paths = Vec::new();
    for (pass, &prefetch) in passes_of(desc)
        .into_iter()
        .flat_map(|pass| prefetches.iter().map(move |prefetch| (pass, prefetch)))
    {
        for path in plain_paths(desc) {
            if pass != Pass::Buffer || path.layout == Layout::Packed {
                paths.push(Path {
                    pass,
                    prefetch,
                    ..path
                });
            }
        }
    }
    paths
}

/// The paths of [`possible_paths`] for the plain pass without prefetch.
fn plain_paths(desc: &KernelPaths) -> Vec<Path> {
    let mut paths = Vec::new();
    for tiles in 1..=desc.variants {
        for passes in [0, 2] {
            for remainder in 0..desc.unroll {
                // kc ≥ 1: without a pass there is a remainder.
                if passes == 0 && remainder == 0 {
                    continue;
                }
                for epilogue in EPILOGUES {
                    for columns in [Columns::Full, Columns::Partial] {
                        for position in [
                            Position::Only,
                            Position::Leading,
                            Position::Middle,
                            Position::Last,
                        ] {
                            for layout in [Layout::Packed, Layout::InPlace] {
                                for last_tile in [LastTile::Full, LastTile::OneRow] {
                                    // A leading or middle row panel holds 64 rows.
                                    let leading_ok =
                                        !matches!(position, Position::Leading | Position::Middle)
                                            || (tiles == desc.variants
                                                && last_tile == LastTile::Full);
                                    // One row is a full tile when tiles have one row.
                                    let edge_ok =
                                        last_tile != LastTile::OneRow || desc.tile_rows > 1;
                                    if leading_ok && edge_ok {
                                        paths.push(Path {
                                            tiles,
                                            passes,
                                            remainder,
                                            epilogue,
                                            columns,
                                            position,
                                            layout,
                                            last_tile,
                                            pass: Pass::Plain,
                                            prefetch: false,
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
    paths
}

/// `path` reduced to the dimensions `dims`.
fn project(path: &Path, dims: &[Dim]) -> Pattern {
    let keep = |dim: Dim| dims.contains(&dim);
    Pattern {
        tiles: keep(Dim::Tiles).then_some(path.tiles),
        passes: keep(Dim::Passes).then_some(path.passes),
        remainder: keep(Dim::Remainder).then_some(path.remainder),
        epilogue: keep(Dim::Epilogue).then_some(path.epilogue),
        columns: keep(Dim::Columns).then_some(path.columns),
        position: keep(Dim::Position).then_some(path.position),
        layout: keep(Dim::Layout).then_some(path.layout),
        last_tile: keep(Dim::LastTile).then_some(path.last_tile),
        pass: keep(Dim::Pass).then_some(path.pass),
        prefetch: keep(Dim::Prefetch).then_some(path.prefetch),
    }
}

/// Every class of paths the table requires for `desc`.
fn required(desc: &KernelPaths) -> BTreeSet<Pattern> {
    let paths = possible_paths(desc);
    REQUIREMENTS
        .iter()
        .flat_map(|requirement| {
            paths
                .iter()
                .filter(|path| requirement.passes.contains(&path.pass))
                .map(|path| project(path, requirement.dims))
        })
        .collect()
}

/// What decides the path of a kernel call: the fields of its argument block
/// that the kernel branches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Call {
    /// Rows.
    m: usize,
    /// Columns.
    n: usize,
    /// Depth.
    kc: usize,
    /// A step between depth steps.
    a_depth: usize,
    /// A step between row panels.
    a_panel: usize,
    /// Bits of alpha.
    alpha_bits: u64,
    /// Bits of beta.
    beta_bits: u64,
    /// Online packing (a non-null `a_pack`).
    a_pack: bool,
    /// The `prefetch` field.
    prefetch: u64,
}

impl Call {
    /// The call of a recorded argument block.
    fn of_args<T: SmeElement>(args: &MacroArgs<T>) -> Self {
        Self {
            m: args.m,
            n: args.n,
            kc: args.kc,
            a_depth: args.a_depth,
            a_panel: args.a_panel,
            alpha_bits: args.alpha_bits,
            beta_bits: args.beta_bits,
            a_pack: !args.a_pack.is_null(),
            prefetch: args.prefetch,
        }
    }

    /// The call that the self-test makes for `case` and the pair `scalars`
    /// (as `Operands::new` lays A out).
    fn of_case<T: TestFloat>(case: &Case, (alpha, beta): (f64, f64)) -> Self {
        let (a_depth, a_panel) = match case.layout {
            ALayout::Packed => (MR, MR * case.kc),
            ALayout::InPlace => (case.m + case.lda_pad, MR),
        };
        Self {
            m: case.m,
            n: case.n,
            kc: case.kc,
            a_depth,
            a_panel,
            alpha_bits: T::of(alpha).bits(),
            beta_bits: T::of(beta).bits(),
            a_pack: case.online.packs(),
            prefetch: case.online.prefetch(),
        }
    }
}

/// The epilogue class of a call of a `T` kernel.
fn epilogue_of<T: SmeElement>(call: &Call) -> EpilogueClass {
    let alpha = T::from_bits64(call.alpha_bits);
    let beta = T::from_bits64(call.beta_bits);
    match Epilogue::select(alpha, beta) {
        Epilogue::Store => EpilogueClass::M0,
        Epilogue::Add => EpilogueClass::M1,
        Epilogue::Scale => EpilogueClass::M2,
        Epilogue::ScaleCombine if alpha == T::ONE => EpilogueClass::M3AlphaOne,
        Epilogue::ScaleCombine if beta == T::ONE => EpilogueClass::M3BetaOne,
        Epilogue::ScaleCombine => EpilogueClass::M3General,
    }
}

/// Every path one call of a `T` kernel with structure `desc` takes: one per
/// (row panel, column panel) pair, in the kernel's order. With online
/// packing column panel 0 runs the packing twins on the call's A and the
/// later column panels the plain variants on the buffer (packed layout).
fn paths_of_call<T: SmeElement>(desc: &KernelPaths, call: &Call) -> Vec<Path> {
    let epilogue = epilogue_of::<T>(call);
    let given = if call.a_depth == MR && call.a_panel == MR * call.kc {
        Layout::Packed
    } else {
        Layout::InPlace
    };
    let passes = (call.kc / desc.unroll).min(2);
    let remainder = call.kc % desc.unroll;
    let prefetch = desc.prefetch && call.prefetch & PREFETCH_C != 0;
    let mut paths = Vec::new();
    for j0 in (0..call.n).step_by(desc.nr) {
        let columns = if call.n - j0 >= desc.nr {
            Columns::Full
        } else {
            Columns::Partial
        };
        let (pass, layout) = match (desc.packing && call.a_pack, j0 == 0) {
            (false, _) => (Pass::Plain, given),
            (true, true) => (Pass::Packing, given),
            (true, false) => (Pass::Buffer, Layout::Packed),
        };
        for i0 in (0..call.m).step_by(MR) {
            let rows = (call.m - i0).min(MR);
            let position = match (i0 == 0, i0 + MR < call.m) {
                (true, false) => Position::Only,
                (true, true) => Position::Leading,
                (false, true) => Position::Middle,
                (false, false) => Position::Last,
            };
            let last_tile = match rows % desc.tile_rows {
                0 => LastTile::Full,
                1 => LastTile::OneRow,
                _ => LastTile::Other,
            };
            paths.push(Path {
                tiles: rows.div_ceil(desc.tile_rows),
                passes,
                remainder,
                epilogue,
                columns,
                position,
                layout,
                last_tile,
                pass,
                prefetch,
            });
        }
    }
    paths
}

/// The calls the self-test makes for the blocks `cases`, in order.
fn listed_calls<T: TestFloat>(cases: impl Iterator<Item = Case>) -> Vec<Call> {
    cases
        .flat_map(|case| {
            SCALARS
                .iter()
                .map(move |&pair| Call::of_case::<T>(&case, pair))
        })
        .collect()
}

/// The classes of `desc` that no call of `calls` takes.
fn missing<T: SmeElement>(desc: &KernelPaths, calls: &[Call]) -> Vec<Pattern> {
    let taken: BTreeSet<Path> = calls
        .iter()
        .flat_map(|call| paths_of_call::<T>(desc, call))
        .collect();
    // Every class a taken path belongs to (a class is a projection).
    let classes: BTreeSet<Pattern> = REQUIREMENTS
        .iter()
        .flat_map(|requirement| {
            taken
                .iter()
                .map(move |path| project(path, requirement.dims))
        })
        .collect();
    required(desc)
        .into_iter()
        .filter(|pattern| !classes.contains(pattern))
        .collect()
}

/// The registered template of the macro-kernel of `T` (the exact text that
/// `asm!` assembles).
pub(super) fn kernel_template<T: SmeElement>() -> &'static str {
    let kernels = if T::NR == 16 {
        F32_KERNELS
    } else {
        F64_KERNELS
    };
    let Some(entry) = kernels.first() else {
        panic!("no registered kernel for NR {}", T::NR);
    };
    entry.template
}

/// The table for one type: the description matches the type and the
/// kernel's registered text, and every required class is taken by the
/// self-test's own block list.
fn table_is_complete<T: TestFloat>(desc: &KernelPaths) {
    assert_eq!(desc.tile_rows, T::TILE_ROWS, "{}: tile rows", desc.name);
    assert_eq!(desc.nr, T::NR, "{}: panel width", desc.name);
    assert_eq!(desc.variants, MR / T::TILE_ROWS, "{}: variants", desc.name);
    if let Err(error) = model_matches_text(desc, kernel_template::<T>()) {
        panic!(
            "{}: the model differs from the kernel text: {error}",
            desc.name
        );
    }
    let calls = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    let classes = required(desc).len();
    let gaps = missing::<T>(desc, &calls);
    println!(
        "{}: {} calls, {} required path classes, {} not taken",
        desc.name,
        calls.len(),
        classes,
        gaps.len()
    );
    assert!(
        gaps.is_empty(),
        "{}: {} of {classes} path classes not taken by the self-test, first: {:?}",
        desc.name,
        gaps.len(),
        gaps.first()
    );
}

#[test]
fn test_sme2_selftest_paths_all_taken_f32() {
    table_is_complete::<f32>(&F32_PATHS);
}

#[test]
fn test_sme2_selftest_paths_all_taken_f64() {
    table_is_complete::<f64>(&F64_PATHS);
}

/// The table on `desc` changed by `change` (a class the kernels do not
/// have) reports gaps.
fn new_class_is_a_gap<T: TestFloat>(desc: &KernelPaths) {
    let calls = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    assert!(missing::<T>(desc, &calls).is_empty(), "{}", desc.name);
    let more_variants = KernelPaths {
        variants: desc.variants + 1,
        ..*desc
    };
    assert!(
        !missing::<T>(&more_variants, &calls).is_empty(),
        "{}: another row variant",
        desc.name
    );
    let unrolled_by_4 = KernelPaths { unroll: 4, ..*desc };
    assert!(
        !missing::<T>(&unrolled_by_4, &calls).is_empty(),
        "{}: a depth loop unrolled by 4",
        desc.name
    );
    // No calls at all: every class is missing.
    assert_eq!(missing::<T>(desc, &[]).len(), required(desc).len());
}

#[test]
fn test_sme2_selftest_paths_new_class_is_a_gap() {
    new_class_is_a_gap::<f32>(&F32_PATHS);
    new_class_is_a_gap::<f64>(&F64_PATHS);
}

/// The sweeps and the online cases (without [`CASES`]) take every required
/// class, and without any one of their blocks there is a gap: every block
/// takes a class that no other block of them takes. (A class may also be
/// taken by a block of `CASES`, which tests other claims and may change.)
fn every_sweep_block_counts<T: TestFloat>(desc: &KernelPaths) {
    let sweep: Vec<Case> = sweep_cases(T::TILE_ROWS, T::NR).collect();
    let online: Vec<Case> = online_sweep_cases(T::TILE_ROWS, T::NR).collect();
    let per_sweep = desc.variants * SWEEP_BLOCKS_PER_VARIANT + SWEEP_MIDDLE_BLOCKS;
    assert_eq!((sweep.len(), online.len()), (per_sweep, per_sweep));
    let blocks: Vec<Case> = online_cases(T::NR)
        .into_iter()
        .chain(sweep)
        .chain(online)
        .collect();
    let alone = listed_calls::<T>(blocks.iter().copied());
    assert!(
        missing::<T>(desc, &alone).is_empty(),
        "{}: the sweeps and the online cases alone",
        desc.name
    );
    for skip in 0..blocks.len() {
        let without = blocks
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != skip)
            .map(|(_, &case)| case);
        let calls = listed_calls::<T>(without);
        assert!(
            !missing::<T>(desc, &calls).is_empty(),
            "{}: block {skip} ({:?}) is not needed",
            desc.name,
            blocks.get(skip)
        );
    }
}

#[test]
fn test_sme2_selftest_paths_every_sweep_block_counts() {
    every_sweep_block_counts::<f32>(&F32_PATHS);
    every_sweep_block_counts::<f64>(&F64_PATHS);
}

/// Without the blocks of three row panels of one sweep (`online`: the online
/// sweep's, else the plain one's) the whole list has gaps, all of them
/// classes of a middle row panel of that sweep's passes; and every depth
/// class and A layout of such a panel is among them (but packed `kc = 1`
/// of the plain pass, which the `130 × 33 × 1` block of [`CASES`] also
/// takes; the online case `130 × (2 · NR + 1) × 3` makes one pass, which no
/// class requires).
fn middle_blocks_are_needed<T: TestFloat>(desc: &KernelPaths, online: bool) {
    let per_variant = desc.variants * SWEEP_BLOCKS_PER_VARIANT;
    let sweep: Vec<Case> = sweep_cases(T::TILE_ROWS, T::NR).collect();
    let online_sweep: Vec<Case> = online_sweep_cases(T::TILE_ROWS, T::NR).collect();
    let (dropped, kept) = if online {
        (&online_sweep, &sweep)
    } else {
        (&sweep, &online_sweep)
    };
    let (front, middle) = dropped.split_at(per_variant);
    assert_eq!(middle.len(), SWEEP_MIDDLE_BLOCKS, "{}", desc.name);
    assert!(
        middle
            .iter()
            .all(|case| case.m > 2 * MR && case.m <= 3 * MR),
        "{}: {middle:?}",
        desc.name
    );
    let calls = listed_calls::<T>(
        CASES
            .into_iter()
            .chain(online_cases(T::NR))
            .chain(front.iter().copied())
            .chain(kept.iter().copied()),
    );
    let gaps = missing::<T>(desc, &calls);
    assert!(
        !gaps.is_empty(),
        "{}: the middle blocks are not needed",
        desc.name
    );
    let passes: &[Pass] = if online {
        &[Pass::Packing, Pass::Buffer]
    } else {
        &[Pass::Plain]
    };
    assert!(
        gaps.iter().all(|gap| gap.position == Some(Position::Middle)
            && gap.pass.is_some_and(|pass| passes.contains(&pass))),
        "{}: {gaps:?}",
        desc.name
    );
    // Each depth class and each layout of a middle panel is among the gaps
    // (the buffer's layout is the packed one, so the twins' classes count).
    let first = passes.first().copied();
    for layout in [Layout::Packed, Layout::InPlace] {
        for (passes, remainder) in [(0, 1), (2, 0), (2, 1)] {
            let found = gaps.iter().any(|gap| {
                gap.pass == first
                    && gap.layout == Some(layout)
                    && gap.passes == Some(passes)
                    && gap.remainder == Some(remainder)
            });
            let in_cases = !online && layout == Layout::Packed && passes == 0;
            assert_eq!(
                found, !in_cases,
                "{}: middle panel {first:?} {layout:?} passes {passes} remainder {remainder}",
                desc.name
            );
        }
    }
}

#[test]
fn test_sme2_selftest_paths_middle_blocks_are_needed() {
    middle_blocks_are_needed::<f32>(&F32_PATHS, false);
    middle_blocks_are_needed::<f64>(&F64_PATHS, false);
    middle_blocks_are_needed::<f32>(&F32_PATHS, true);
    middle_blocks_are_needed::<f64>(&F64_PATHS, true);
}

/// The table requires the online paths of `desc` (a kernel that packs and
/// prefetches), and only the blocks that use them take them:
///
/// - the required classes, per pass and per prefetch value (pinned: a
///   change of the requirements must change these numbers deliberately);
/// - without the blocks with online packing (the online cases and the
///   online sweep) every class of the packing twins and of the buffer of
///   every live-tile count and depth class is missing, and nothing else;
/// - with every block's prefetch switched off, every prefetch class of every
///   pass is missing, and nothing else;
/// - a model without online packing (or without prefetch) requires none of
///   them, and the whole list then takes all it requires.
fn online_classes_are_required<T: TestFloat>(desc: &KernelPaths, pinned: [usize; 5]) {
    let classes = required(desc);
    let count = |pass: Pass| {
        classes
            .iter()
            .filter(|class| class.pass == Some(pass) && class.prefetch.is_none())
            .count()
    };
    let prefetch_classes = classes
        .iter()
        .filter(|class| class.prefetch.is_some())
        .count();
    let got = [
        classes.len(),
        count(Pass::Plain),
        count(Pass::Packing),
        count(Pass::Buffer),
        prefetch_classes,
    ];
    println!(
        "{}: required classes (all, plain, packing, buffer, prefetch) {got:?}",
        desc.name
    );
    assert_eq!(got, pinned, "{}", desc.name);

    let blocks: Vec<Case> = self_test_blocks(T::TILE_ROWS, T::NR).collect();
    let calls = listed_calls::<T>(blocks.iter().copied());
    assert!(missing::<T>(desc, &calls).is_empty(), "{}", desc.name);
    // Without online packing.
    let unpacked = listed_calls::<T>(blocks.iter().copied().filter(|case| !case.online.packs()));
    let gaps = missing::<T>(desc, &unpacked);
    assert!(
        gaps.iter()
            .all(|gap| matches!(gap.pass, Some(Pass::Packing | Pass::Buffer))),
        "{}: {gaps:?}",
        desc.name
    );
    for pass in [Pass::Packing, Pass::Buffer] {
        for tiles in 1..=desc.variants {
            for (passes, remainder) in [(0, 1), (2, 0), (2, 1)] {
                let found = gaps.iter().any(|gap| {
                    gap.pass == Some(pass)
                        && gap.tiles == Some(tiles)
                        && gap.passes == Some(passes)
                        && gap.remainder == Some(remainder)
                });
                assert!(
                    found,
                    "{}: {pass:?} {tiles} tiles, passes {passes} remainder {remainder}",
                    desc.name
                );
            }
        }
    }
    // Without prefetch.
    let quiet: Vec<Call> = calls
        .iter()
        .map(|call| Call {
            prefetch: 0,
            ..*call
        })
        .collect();
    let gaps = missing::<T>(desc, &quiet);
    assert!(
        !gaps.is_empty() && gaps.iter().all(|gap| gap.prefetch == Some(true)),
        "{}: {gaps:?}",
        desc.name
    );
    for pass in [Pass::Plain, Pass::Packing, Pass::Buffer] {
        assert!(
            gaps.iter().any(|gap| gap.pass == Some(pass)),
            "{}: prefetch before {pass:?}",
            desc.name
        );
    }
    // Models without the online paths require none of them.
    let without = KernelPaths {
        packing: false,
        prefetch: false,
        ..*desc
    };
    let plain = required(&without);
    assert!(
        plain
            .iter()
            .all(|class| class.pass == Some(Pass::Plain) && class.prefetch != Some(true)),
        "{}",
        desc.name
    );
    assert!(missing::<T>(&without, &calls).is_empty(), "{}", desc.name);
}

#[test]
fn test_sme2_selftest_paths_online_classes_are_required() {
    online_classes_are_required::<f32>(&F32_PATHS, [3_312, 1_944, 972, 324, 72]);
    online_classes_are_required::<f64>(&F64_PATHS, [6_240, 3_672, 1_836, 612, 120]);
}

thread_local! {
    /// Calls seen by [`recording_oracle`] on this thread.
    static RECORDED: RefCell<Vec<Call>> = const { RefCell::new(Vec::new()) };
}

/// Stand-in: the oracle, recording each call.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn recording_oracle<T: SmeElement>(args: &MacroArgs<T>) {
    RECORDED.with(|recorded| recorded.borrow_mut().push(Call::of_args(args)));
    // SAFETY: forwarded from this function's contract.
    unsafe { oracle_block(args) };
}

/// `self_test_with` makes exactly the calls of the block list the table is
/// computed from, in order.
fn runs_its_list<T: TestFloat>() {
    RECORDED.with(|recorded| recorded.borrow_mut().clear());
    let kernel: KernelFn<T> = recording_oracle::<T>;
    // SAFETY: the stand-in is plain Rust.
    assert!(unsafe { self_test_with(kernel) });
    let recorded = RECORDED.with(|recorded| recorded.take());
    let listed = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    assert_eq!(recorded.len(), self_test_calls::<T>());
    assert!(
        recorded == listed,
        "{}: the run differs from the list",
        T::NR
    );
}

#[test]
fn test_sme2_selftest_paths_run_equals_list() {
    runs_its_list::<f32>();
    runs_its_list::<f64>();
}
