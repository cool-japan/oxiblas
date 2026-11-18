//! GEMM routing policies: which optional paths a GEMM call may take, and the
//! policy the public entry points use ([`DEFAULT_POLICY`]).
//!
//! A [`GemmPolicy`] holds one switch or threshold per optional path; the
//! router (`gemm_plan::choose_route`) and the blocked drivers read it. The
//! policies of this crate:
//!
//! - [`LEGACY_POLICY`] reproduces 0.2.3 (its promise about result bits and
//!   how it is checked are stated there).
//! - [`ADOPTED_AARCH64`] is [`LEGACY_POLICY`] with the fields that passed the
//!   project's A/B keep rule on aarch64 (measured on an Apple M3, NEON tier,
//!   operands stored column-major): the thin route within the measured
//!   scope ([`ThinPolicy::Scoped`]) and the small transposed copy of a
//!   transposed A ([`SMALL_TRANS_COPY_A_ONLY_RANGE`]), both only where both
//!   operands are stored column-major, the thin route only on a tier with
//!   SIMD thin kernels. [`ADOPTED_POLICY`] is it on aarch64 builds without
//!   `force-scalar`, [`LEGACY_POLICY`] everywhere else.
//! - [`CANDIDATE_AARCH64`] and [`CANDIDATE_X86`] hold the paths that are
//!   still candidates; [`CANDIDATE_POLICY`] is the one of the target
//!   architecture. Candidates for a later measurement, not in any policy:
//!   the two adopted routes for operands that are not stored column-major
//!   (the A/B harness has no case with such storage yet).
//!
//! [`DEFAULT_POLICY`], per build:
//!
//! | build | `DEFAULT_POLICY` |
//! |---|---|
//! | any target, `--cfg oxiblas_gemm_legacy` (wins over the candidate cfg) | [`LEGACY_POLICY`] |
//! | any target, `--cfg oxiblas_gemm_candidate` | [`CANDIDATE_POLICY`] of the target |
//! | aarch64 without `force-scalar` | [`ADOPTED_AARCH64`] (its thin route needs the NEON tier at run time) |
//! | aarch64 with `force-scalar`; x86-64; every other target | [`LEGACY_POLICY`] |
//!
//! `max-simd-128` and `max-simd-256` do not change the tier on aarch64 (NEON
//! is 128 bits wide), so the adopted policy applies with them. The policy
//! follows this crate's `force-scalar`, the tier of a call oxiblas-core's
//! (which Cargo can turn on through another crate of the build without
//! this crate's): such a build has [`ADOPTED_AARCH64`] on the portable tier,
//! where the thin route is out of its scope and the copy has no SIMD small
//! kernels, so every call takes the route of 0.2.3.
//!
//! # Result bits under the default policy
//!
//! - Under [`LEGACY_POLICY`] every GEMM call keeps the bits of 0.2.3, with
//!   the NaN rule stated there.
//! - Under the default policy of an aarch64 build without `force-scalar`,
//!   on the NEON tier, two kinds of real `f64` / `f32` products compute in
//!   another order than 0.2.3, so their results agree with 0.2.3 to
//!   rounding and not bit for bit; both kinds have BOTH operands stored
//!   column-major (`col_stride() == 1`: a `Mat` or a sub-matrix of one):
//!   - the thin products the thin route takes (`m == 1` or `n == 1`): every
//!     such product of more than 32³ multiply-adds (`m·n·k`, all four
//!     transpose combinations), and, from 1024 multiply-adds up to 32³, the
//!     four layouts with the vector operand not transposed, from the least
//!     sizes of [`THIN_COL_AXPY`], [`THIN_COL_DOT`], [`THIN_ROW_DOT`] and
//!     [`THIN_ROW_AXPY`] (`RouteInputs::thin_in_scope` states the rule); a
//!     conjugate transpose of a real matrix counts as a transpose. Their
//!     order is the thin route's contract (the module docs of `gemm_thin`:
//!     the NEON GEMV kernels; with the `sme2` feature the SME2 GEMV may
//!     compute such a call, with the same bits apart from NaN signs and
//!     payloads);
//!   - the small transposed copy: products of 12³ to 32³ multiply-adds with
//!     `m >= 4` and `n >= 4` and A transposed (`TN` and `TT`). Their result
//!     is bit for bit that of the `NoTrans`/`NoTrans` NEON small kernels on
//!     column-major operands holding the same values (the contract of
//!     `gemm_small::gemm_small_trans_copy`).
//!
//!   A real product with a transposed or strided view as an operand (an
//!   operand whose `col_stride()` is not 1) keeps the bits of 0.2.3, as
//!   does every product on another tier. The rule is about each real
//!   `f64` / `f32` product that the library computes through its GEMM
//!   core, whoever issues it, and the storage that counts is that of that
//!   product's operands: a direct call, the complex entry points that
//!   compute through real products on column-major scratch of their own
//!   (`gemm_transposed_c64` / `gemm_transposed_c32`, the 3M scheme of
//!   `gemm3m_c64` / `gemm3m_c32`, also when the caller passes views), and
//!   every routine that computes through GEMM. Their results can differ
//!   from 0.2.3 in the last bits where such an inner product takes one of
//!   the two routes. Every other call keeps the bits of 0.2.3, with the NaN
//!   rule of [`LEGACY_POLICY`]. Parallel results stay bit for bit equal to
//!   the sequential ones on every route.
//! - `--cfg oxiblas_gemm_legacy` restores the routes of 0.2.3, and with them
//!   its bits, on every target.
//! - With `force-scalar` on aarch64, on x86-64 and on every other target the
//!   default is [`LEGACY_POLICY`]: nothing changed there. A build where only
//!   oxiblas-core has `force-scalar` keeps the routes of 0.2.3 too (the
//!   portable tier, above).

use super::autotune::{BBlockCaps, ISSUE_23_B_CAPS};
use super::gemm_small::SMALL_THRESHOLD;
use oxiblas_core::parallel::ParThreshold;

/// Where the issue #23 B-block cap ([`BBlockCaps`]) applies.
///
/// Calls with explicit (caller-chosen) blocking are never capped, whatever
/// the policy says; see `gemm::BlockingSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BCapScope {
    /// Nowhere (0.2.3).
    Off,
    /// In the sequential blocked driver; the 2D parallel driver keeps the
    /// fitted NC.
    Sequential,
    /// In both blocked drivers: the cap is applied before the driver is
    /// chosen, so the 2D parallel driver gets the capped NC too. The
    /// candidates use it only when built with `--cfg oxiblas_gemm_bcap_all`
    /// (A/B runs; see [`CANDIDATE_B_CAP`]).
    All,
}

/// Which f32 micro-kernel the blocked route of a plan resolves on the
/// AVX2 + FMA tier.
///
/// The canonical f32 kernel of that tier (`f32::micro_kernel_dispatch()`)
/// is the 8×8 kernel under every policy; the policy only chooses the kernel
/// of one call's plan (`gemm_kernel::kernel_for_problem`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F32Avx2Policy {
    /// Always the canonical 8×8 kernel (0.2.3).
    Legacy8x8,
    /// The kernel shape chosen per problem (16×6 where it pays off).
    Adaptive,
}

/// How the 2D parallel driver keeps its packed panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackScope {
    /// One packed B block per KC step, shared by every tile (0.2.3).
    Shared,
    /// Every tile packs its own part of B (and A) on the worker that runs it.
    PerTile,
}

/// Which thin products (`m == 1` or `n == 1`, `f64`/`f32`, large operand with
/// a unit stride: `RouteInputs::thin_ok`) run as one GEMV (`gemm_thin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThinPolicy {
    /// None (0.2.3).
    Off,
    /// The thin products of the measured scope
    /// (`RouteInputs::thin_in_scope`, from [`THIN_MIN_WORK`] on): both
    /// operands stored column-major, on a tier with SIMD thin kernels (NEON,
    /// AVX2 + FMA); every other thin product keeps the route of 0.2.3.
    Scoped,
    /// Every thin product, on every tier and with any storage the thin
    /// route can read (tests and one A/B variant).
    All,
}

/// Least `m·n·k` of a thin product that [`ThinPolicy::Scoped`] sends to the
/// thin route.
///
/// The router compares the size with it before it looks at the thin policy
/// at all, so below it a call takes exactly the same decisions under
/// [`ThinPolicy::Off`] and [`ThinPolicy::Scoped`]: the smallest products,
/// thin or not, cannot get slower by the scope. From it on, a product that
/// the scope sends back to the small loops has paid for the scope's tests
/// (about 1.5 ns on an Apple M3), which is around 1 % of the fastest such
/// call (the small loops need at least 0.12 ns per multiply-add).
pub(crate) const THIN_MIN_WORK: usize = 1024;

/// The scope's limits for one layout of a small thin product (both
/// operands stored column-major, on a tier with SIMD thin kernels): the
/// least length `max(m, n)` of the vector of C and the least depth `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ThinLimits {
    /// Least `max(m, n)`.
    pub(crate) min_vec: usize,
    /// Least `k`.
    pub(crate) min_depth: usize,
}

/// `n == 1`, `op(A) = A`, `op(B) = B`: the axpy form over A's columns.
pub(crate) const THIN_COL_AXPY: ThinLimits = ThinLimits {
    min_vec: 2,
    min_depth: 2,
};
/// `n == 1`, `op(A) = Aᵀ`, `op(B) = B`: one dot product per column of the
/// stored A.
pub(crate) const THIN_COL_DOT: ThinLimits = ThinLimits {
    min_vec: 2,
    min_depth: 16,
};
/// `m == 1` (`n > 1`), `op(A) = A`, `op(B) = B`: one dot product per column
/// of B.
pub(crate) const THIN_ROW_DOT: ThinLimits = ThinLimits {
    min_vec: 4,
    min_depth: 2,
};
/// `m == 1` (`n > 1`), `op(A) = A`, `op(B) = Bᵀ`: the axpy form over the
/// columns of the stored B.
pub(crate) const THIN_ROW_AXPY: ThinLimits = ThinLimits {
    min_vec: 4,
    min_depth: 4,
};

/// Which optional GEMM paths a call may take.
///
/// Each field is one switch or threshold; a policy never changes what is
/// computed, only the route (and with it the rounding of the last bits) and
/// the block sizes. The policies below list the fields in the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmPolicy {
    /// Which thin products (`m == 1` or `n == 1`, `f64`/`f32`, large operand
    /// with a unit stride) run as one GEMV (`gemm_thin`).
    pub(crate) thin: ThinPolicy,
    /// The work gate of the parallel thin path, as the two numbers of a
    /// [`ParThreshold`]: the least total work (`m·n·k`) and the least work
    /// per thread (see [`GemmPolicy::thin_par_threshold`]).
    pub(crate) thin_par_gate: (usize, usize),
    /// The AVX2 + FMA tier has SIMD small kernels (`m >= 5`, `n >= 4`).
    pub(crate) small_simd_x86: bool,
    /// Largest `m·n·k` that `NoTrans`/`NoTrans` problems with column-major
    /// operands send to the tier's SIMD small kernels. Problems up to
    /// [`SMALL_THRESHOLD`] take the small path in any case.
    pub(crate) small_simd_max_work: usize,
    /// `[min, max]` of `m·n·k` for which transposed or strided problems of a
    /// tier with SIMD small kernels are copied to column-major scratch and run
    /// on those kernels (`gemm_small_trans_copy`); `None` never copies.
    pub(crate) small_trans_copy: Option<(usize, usize)>,
    /// The copy is taken only for the measured class: A transposed and both
    /// operands stored column-major (`RouteInputs::a_transposed_col_major`;
    /// `TN` and `TT`). `NoTrans` A of column-major operands keeps the small
    /// transposed loops, which already run along A's columns, and a product
    /// with an operand not stored column-major keeps the route of 0.2.3.
    pub(crate) small_trans_copy_a_only: bool,
    /// The f32 micro-kernel of a plan on the AVX2 + FMA tier.
    pub(crate) f32_avx2: F32Avx2Policy,
    /// Thread count from which the 2D parallel driver uses
    /// [`PackScope::PerTile`]; `usize::MAX` never does.
    pub(crate) per_tile_min_threads: usize,
    /// Where the B-block cap applies.
    pub(crate) b_cap: BCapScope,
    /// The B-block cap rule.
    pub(crate) b_caps: BBlockCaps,
}

impl GemmPolicy {
    /// The work gate of the parallel thin path ([`GemmPolicy::thin_par_gate`]).
    pub(crate) const fn thin_par_threshold(&self) -> ParThreshold {
        ParThreshold::new(self.thin_par_gate.0, self.thin_par_gate.1)
    }
}

/// Work gate of the parallel thin path in every policy: at least 512²
/// multiply-adds in total and 128² per thread.
pub(crate) const THIN_PAR_GATE: (usize, usize) = (512 * 512, 128 * 128);

/// The policy that reproduces 0.2.3: no thin route, the small path up to
/// [`SMALL_THRESHOLD`] (with the NEON small kernels on aarch64) and no copy,
/// the canonical f32 kernel, the shared pack scope, everything else blocked
/// with the fitted NC.
///
/// What "reproduces 0.2.3" promises for every GEMM call under this policy:
/// every result that is not a NaN has the bits of 0.2.3, and every NaN
/// result sits where 0.2.3 has one. A NaN computed from one NaN operand
/// carries that NaN, as in 0.2.3; but where an operation meets two NaNs (an
/// add or a multiply whose operands are both NaNs), which of the two the
/// result carries (its sign and payload) may differ from 0.2.3. Rust and
/// LLVM leave that choice unspecified: it is the operand order the compiler
/// gives the machine instruction, not the order in the source, and 0.2.3
/// itself makes it differently in different build profiles (of 2160 calls
/// with distinct NaN payloads on aarch64, 27 `NoTrans`/`NoTrans` calls
/// differ between its O3 + thin-LTO and its O2 release builds, and 550 in a
/// debug build). In this version the small transposed loops for
/// `op(A) = A` with adjacent rows ([`small_trans_axpy`], from 8 rows, and
/// [`small_trans_dot_columns`], 3 to 7 rows) keep a different one of the
/// two NaNs than a release build of 0.2.3: the register blocks keep the
/// accumulator's NaN of an add where 0.2.3 keeps the product's, and the
/// dot columns keep `op(B)`'s NaN of a multiply where 0.2.3 keeps A's.
///
/// How the promise is checked: the legacy-bits check (the public GEMM API of
/// this tree against a 0.2.3 build, on aarch64 with NEON, where this tree
/// must be built with `--cfg oxiblas_gemm_legacy` for its public API to use
/// this policy, and with `force-scalar`) compares results bit for bit, but
/// its inputs carry no
/// NaN other than the canonical quiet NaN (`f64::NAN`, `f32::NAN`), so
/// every NaN result has the same bits and it compares NaN positions only.
/// The bit-for-bit tests of the small transposed loops (`gemm_small_trans`
/// against a copy of 0.2.3's loops, with distinct NaN payloads) compare the
/// payloads too, except in the one family of inputs where two NaNs meet,
/// which they compare by NaN positions.
///
/// It is the default policy of every build but aarch64 without
/// `force-scalar`, and of every build with `--cfg oxiblas_gemm_legacy`
/// ([`DEFAULT_POLICY`]): on aarch64 a comparison of the public GEMM API
/// with 0.2.3 bit for bit needs a build with that cfg.
///
/// [`small_trans_axpy`]: super::gemm_small::small_trans_axpy
/// [`small_trans_dot_columns`]: super::gemm_small::small_trans_dot_columns
pub(crate) const LEGACY_POLICY: GemmPolicy = GemmPolicy {
    thin: ThinPolicy::Off,
    thin_par_gate: THIN_PAR_GATE,
    small_simd_x86: false,
    small_simd_max_work: SMALL_THRESHOLD,
    small_trans_copy: None,
    small_trans_copy_a_only: false,
    f32_avx2: F32Avx2Policy::Legacy8x8,
    per_tile_min_threads: usize::MAX,
    b_cap: BCapScope::Off,
    b_caps: ISSUE_23_B_CAPS,
};

/// `m·n·k` range of the small transposed copy of every transposed or
/// strided product (the copy as first designed, a candidate on x86-64):
/// from 8³ up to the small threshold.
pub(crate) const SMALL_TRANS_COPY_FULL_RANGE: (usize, usize) = (8 * 8 * 8, SMALL_THRESHOLD);

/// `m·n·k` range of the small transposed copy where only products with A
/// transposed and both operands stored column-major are copied
/// ([`GemmPolicy::small_trans_copy_a_only`]): from 12³ up to the small
/// threshold; adopted on aarch64.
///
/// Measured on an Apple M3 (NEON, A/B group `ab_seq_small_trans_copy`, five
/// passes, operands stored column-major): 1.6 to 4.6 times faster than the
/// generic transposed loop of 0.2.3 from 12³ (geometric mean of the 28
/// targets, `TN` and `TT`, 2.72 to 2.74, least target 1.63). At 8³ the copy
/// of a product with only A transposed was 1.01 to 1.07 times faster over
/// three runs, too close to the rule's limit to rely on. Where `op(A)` is
/// read in place (`NoTrans` A of column-major operands), 0.2.3's
/// specialised loops along A's columns are as fast or faster for products
/// with a long `k`. Products with an operand stored otherwise were not
/// measured and are not copied.
pub(crate) const SMALL_TRANS_COPY_A_ONLY_RANGE: (usize, usize) = (12 * 12 * 12, SMALL_THRESHOLD);

/// The policy adopted on aarch64 (NEON tier): [`LEGACY_POLICY`] with the
/// thin route within the measured scope ([`ThinPolicy::Scoped`]) and the
/// small transposed copy of the products with A transposed
/// ([`SMALL_TRANS_COPY_A_ONLY_RANGE`], `small_trans_copy_a_only`), both only
/// for operands stored column-major, the thin route only on a tier with
/// SIMD thin kernels: what was measured. Every other product takes the
/// route of [`LEGACY_POLICY`].
///
/// Both fields passed the project's keep rule (medians of 22 interleaved
/// rounds, an A/A' control within 1.5 %, geometric mean of the target cases
/// at least 1.05, every target case at least 1.00, every other case at least
/// 0.97), measured on an Apple M3 with the A/B groups `ab_seq_thin`,
/// `ab_par_thin` and `ab_seq_small_trans_copy`, every case with operands
/// stored column-major:
///
/// - the scoped thin route: 840 targets of the dense thin grid at least 1.24
///   times faster (geometric mean 5.4); on the decision set of `ab_seq_thin`
///   120 targets with geometric mean 4.8 and least ratio 1.245 (`NN`
///   2×1024×1 f64; 1.01 in one of eight passes, in which both routes ran
///   slower for that case); parallel (4 threads) geometric mean 5.8, least
///   2.79. The whole thin route ([`ThinPolicy::All`]) failed the rule: small
///   thin products lose where 0.2.3 has a tight loop (the scope's limits,
///   `RouteInputs::thin_in_scope`);
/// - the copy: see [`SMALL_TRANS_COPY_A_ONLY_RANGE`].
///
/// What the two fields do to result bits: the module docs.
pub(crate) const ADOPTED_AARCH64: GemmPolicy = GemmPolicy {
    thin: ThinPolicy::Scoped,
    small_trans_copy: Some(SMALL_TRANS_COPY_A_ONLY_RANGE),
    small_trans_copy_a_only: true,
    ..LEGACY_POLICY
};

/// The adopted policy of this build: [`ADOPTED_AARCH64`] on aarch64 without
/// `force-scalar`, [`LEGACY_POLICY`] everywhere else.
///
/// The measurement was made on the NEON tier. With `force-scalar` the tier
/// is `Portable`: there the thin route would run the portable GEMV kernels
/// and the copy has no SIMD small kernels to run on, and nothing was
/// measured, so such a build keeps the legacy default. On every other
/// target nothing was measured either. A build where only oxiblas-core has
/// `force-scalar` gets this policy's aarch64 value on the portable tier;
/// there the thin route is out of the scope (`RouteInputs::thin_in_scope`
/// needs a tier with SIMD thin kernels) and the copy has no SIMD small
/// kernels, so its calls take the routes of 0.2.3.
pub(crate) const ADOPTED_POLICY: GemmPolicy =
    if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))) {
        ADOPTED_AARCH64
    } else {
        LEGACY_POLICY
    };

/// The B-block cap scope of both candidates: [`BCapScope::Sequential`], or
/// [`BCapScope::All`] when built with `--cfg oxiblas_gemm_bcap_all` (so the
/// `All` variant can be measured on its own, from the same source).
pub(crate) const CANDIDATE_B_CAP: BCapScope = if cfg!(oxiblas_gemm_bcap_all) {
    BCapScope::All
} else {
    BCapScope::Sequential
};

/// Candidate policy for x86-64 (nothing measured there yet): the AVX2 + FMA
/// small kernels up to 80³, the adaptive f32 kernel, per-tile packing from 5
/// threads, the B-block cap, the thin route within the scope measured on
/// aarch64 (the best available prior; on x86-64 that is the AVX2 + FMA
/// tier, so an SSE4.2 host keeps the routes of 0.2.3 for thin products) and
/// the small transposed copy of every transposed or strided product
/// ([`SMALL_TRANS_COPY_FULL_RANGE`]).
pub(crate) const CANDIDATE_X86: GemmPolicy = GemmPolicy {
    thin: ThinPolicy::Scoped,
    thin_par_gate: THIN_PAR_GATE,
    small_simd_x86: true,
    small_simd_max_work: 80 * 80 * 80,
    small_trans_copy: Some(SMALL_TRANS_COPY_FULL_RANGE),
    small_trans_copy_a_only: false,
    f32_avx2: F32Avx2Policy::Adaptive,
    per_tile_min_threads: 5,
    b_cap: CANDIDATE_B_CAP,
    b_caps: ISSUE_23_B_CAPS,
};

/// Candidate policy for aarch64: [`ADOPTED_AARCH64`] plus the B-block cap
/// ([`CANDIDATE_B_CAP`]), which did not pass the keep rule there (geometric
/// mean 1.018 in the sequential driver, below 1.05).
pub(crate) const CANDIDATE_AARCH64: GemmPolicy = GemmPolicy {
    b_cap: CANDIDATE_B_CAP,
    b_caps: ISSUE_23_B_CAPS,
    ..ADOPTED_AARCH64
};

/// The candidate policy of the target architecture. Targets without a
/// candidate keep [`LEGACY_POLICY`]: nothing has been measured there.
pub(crate) const CANDIDATE_POLICY: GemmPolicy = if cfg!(target_arch = "x86_64") {
    CANDIDATE_X86
} else if cfg!(target_arch = "aarch64") {
    CANDIDATE_AARCH64
} else {
    LEGACY_POLICY
};

/// The policy of the public GEMM entry points: [`LEGACY_POLICY`] when built
/// with `--cfg oxiblas_gemm_legacy` (which wins over the candidate cfg),
/// [`CANDIDATE_POLICY`] when built with `--cfg oxiblas_gemm_candidate`, and
/// [`ADOPTED_POLICY`] otherwise (the table and the rule about result bits:
/// the module docs).
pub(crate) const DEFAULT_POLICY: GemmPolicy = if cfg!(oxiblas_gemm_legacy) {
    LEGACY_POLICY
} else if cfg!(oxiblas_gemm_candidate) {
    CANDIDATE_POLICY
} else {
    ADOPTED_POLICY
};

#[cfg(test)]
#[path = "gemm_policy_tests.rs"]
mod tests;
