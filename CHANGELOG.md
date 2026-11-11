# Changelog

All notable changes to OxiBLAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-11-12

Fixes and additions for GitHub issues #4 and #6 through #16 (#16 only in part). The QZ
decomposition is rewritten as a LAPACK `DGGES` equivalent, Level 1/2/3 and CBLAS
behaviour is aligned with reference BLAS 3.12 (increments, `alpha == 0` and
`beta == 0`, `crotg`), 78 CBLAS entry points and several Level-1 natives are added,
and GEMM gets balanced blocking, a 2D parallel driver and aarch64 small-matrix
kernels.

A second round of work, listed after the issue entries in each section below (in new
subsections under Fixed and Changed), adds performance work across Level 1-3 and the
LAPACK drivers: aarch64 NEON kernels for DOT, NRM2 and GEMV, lower per-call GEMM
overhead, triangle-only SYRK/SYR2K/GEMMT and TRMM, a blocked TRSM, in-place LU (with
recursive panels) and Cholesky, and `SymmetricEvdDc`, `SvdDc` and `HermitianEvdDc`
rebuilt on the LAPACK `DSYEVD`, `dgesdd`/`dbdsdc` and `ZHEEVD` schemes, with
`SymmetricEvd` sharing the blocked reduction. It fixes many wrong results in the
sparse solvers, preconditioners and factorizations and in the LAPACK SVD and
eigensolvers, makes singularity, pivot and breakdown tests scale-invariant, closes
soundness holes (out-of-range pointer arithmetic and overlapping views in
`oxiblas-matrix`, unchecked sparse-format invariants, the CBLAS Level-1 slices,
`tensor::contract_2d`, the GEMM packers), documents every `unsafe` block in
`oxiblas-blas`, `oxiblas-matrix`, `oxiblas-lapack`, `oxiblas-sparse` and
`oxiblas-ndarray`, and removes the remaining `unwrap()`/`expect()` calls from library
code.

No public item is removed or renamed and no signature changes; the new items are
listed under Added, and behaviour changes under Fixed and Changed ("Behaviour changes
in the second round" collects them). The minimum supported Rust version rises from
1.85 to 1.89. The published performance figures were wrong and are replaced by new
measurements; see Fixed. Figures quoted in the second-round entries are same-session
before/after measurements by the change's author on one thread of an Apple M3 unless
stated otherwise; the README tables are measured separately.

### Added

#### LAPACK

- `Qz::compute_sorted` (`DGGES` with `SORT = 'S'`) and `Qz::reorder` (`DTGSEN` with
  `IJOB = 0`) move the selected generalized eigenvalues to the leading positions of
  the generalized Schur form. `select` sees the unscaled eigenvalues, and a complex
  pair moves as a whole when either member is selected. `QzOrdering` (fields `sdim`
  and `selection_changed`, i.e. `DGGES` `SDIM` and `INFO = N+2`) and
  `QzReorderError { Qz(QzError), SwapFailed { position } }` are re-exported from
  `oxiblas_lapack::evd`; both are `#[non_exhaustive]`, so read the fields of
  `QzOrdering` and do not construct it. `SwapFailed` is `DTGSEN` `INFO = 1`
  (`DGGES` `INFO = N+3`): `compute_sorted` then discards the factors and `reorder`
  leaves the decomposition unchanged. Condition estimates (`IJOB != 0`) are not
  implemented. If the internally scaled `A` and `B` still differ in norm by more than
  about `1/eps`, every swap that moves a 2x2 block is rejected; pre-scale `A` or `B`
  (see the `compute_sorted` docs). (#6)
- Declared differences of `Qz::compute_sorted` and `Qz::reorder` from `DGGES` with
  `SORT = 'S'` (#6):
  - 2x2 blocks whose `T` diagonal comes out negative after a swap are sign-normalized
    by negating the corresponding rows of `S` and `T` and columns of `Q`, an
    extension of `DTGSEN`, which leaves them negative. `S`, `T` and `Q` can therefore
    differ from Netlib by these sign flips; `Z`, `Q*S*Z^T`, `Q*T*Z^T` and the
    eigenvalues are unaffected.
  - `reorder` scales `S` and `T` into `[SMLNUM, BIGNUM]` around `DTGSEN` and unscales
    with the `DGGES` guards, so they may change bitwise even when no swap is needed.
  - After a rejected swap Netlib returns the partial factors, and its final check can
    turn `INFO = N+3` into `N+2`; here `compute_sorted` discards the factors and
    `reorder` restores `self`.
  - The `DTGEX2` acceptance norms use the classical `DLASSQ` recurrence, which moves
    the thresholds at the ulp level.
  - We expect reference LAPACK to reject the same unbalanced pencils (see above); this
    was derived by analysis and not checked against a LAPACK binary.
- `SymmetricEvd::compute_with_par`, `SymmetricEvdDc::compute_with_par` and
  `SvdDc::compute_with_par` (feature `parallel`) run the GEMMs of the reduction, of
  the divide-and-conquer merges (`SymmetricEvdDc`, `SvdDc`) and of the
  back-transformation with the given `Par`. Every other stage stays on the calling
  thread, and the result is bitwise identical to `compute` for every `Par`.
  `compute` runs on the calling thread, as in 0.2.2. `ParallelSvdDc::compute` is now
  `SvdDc::compute_with_par(a, Par::Rayon)` (see Fixed).

#### BLAS

- Level-1 natives in `oxiblas_blas::level1`, written from the Netlib 3.12 sources:
  `cabs1` (previously crate-private), `scabs1`, `dcabs1`; the complex 2-norm `cnrm2`,
  `scnrm2`, `dznrm2` and `cnrm2_strided`; `scal_real`, `csscal`, `zdscal` (complex
  vector times a real scalar, componentwise, immediate return for `alpha == 1`, no
  zero-fill for `alpha == 0`); `rotm_strided` and `zdrot_strided`. The strided forms
  take positive, negative and zero increments. (#15)
- `trsm_with_par` and `trsm_in_place_with_par` in `oxiblas_blas::level3` (always
  available; `Par::Rayon` needs the `parallel` feature) take an explicit `Par` and
  solve independent panels of right-hand sides concurrently: 128 columns
  (`Side::Left`) or rows (`Side::Right`) per panel, 256 when there are at least 1024
  right-hand sides. The panel width depends only on the shape, so every `Par` gives
  bitwise the result of `trsm`/`trsm_in_place`, which run sequentially (see Changed).
- GEMM kernel-layer items used by the reworked drivers: the provided trait method
  `GemmKernel::micro_kernel_dispatch` with `gemm_kernel::{MicroKernel, MicroKernelFn}`
  (the micro-kernel's shape and function, resolved once per call),
  `gemm_packing::PACK_CACHE_MAX_BYTES` (8 MiB, see Changed), and the doc-hidden
  `GemmKernel::with_pack_cache` and `gemm_packing::PackSlot`. `MicroKernelShape` now
  derives `PartialEq` and `Eq`. Both new trait methods have default bodies, so
  existing `GemmKernel` implementations still compile.

#### CBLAS

- 78 reference CBLAS entry points (30 Level 1/3, 48 Level 2). Matrix routines work
  in both layouts, a row-major call being mapped onto the column-major kernels as
  netlib `cblas_*.c` does (flip `uplo`/`side`, swap `m`/`n`, toggle `Trans`,
  conjugate where the reference does). The new Level-1 entry points with vector
  arguments accept every increment (zero reuses element 0, a negative one starts at
  `(1-n)*inc`), the new Level-2 ones reject `inc == 0` as a silent no-op, and quick
  returns follow the reference. Finite results agree with a reference build to
  rounding, not bit for bit. The crate now exports 152 `cblas_*` symbols (74 before):
  every routine declared in netlib `cblas.h` 3.12.1 except the `cblas_xerbla` error
  handler (3.12.0 lacks `?gemmtr`). Parameter counts and kinds match the header; the
  deviations are listed below. (#14)
  - Level 1 (16): `cblas_{s,d}rotg`, `cblas_{s,d}rotmg`, `cblas_{s,d}rot`,
    `cblas_{s,d}rotm`, `cblas_{c,z}rotg`, `cblas_csrot`, `cblas_zdrot`,
    `cblas_sdsdot`, `cblas_dsdot`, `cblas_scabs1`, `cblas_dcabs1`.
  - Level 2 (48): `cblas_{s,d,c,z}gbmv`, `cblas_{s,d}sbmv`, `cblas_{c,z}hbmv`,
    `cblas_{s,d,c,z}tbmv`, `cblas_{s,d,c,z}tbsv`, `cblas_{s,d}spmv`,
    `cblas_{s,d}spr`, `cblas_{s,d}spr2`, `cblas_{c,z}hpmv`, `cblas_{c,z}hpr`,
    `cblas_{c,z}hpr2`, `cblas_{s,d,c,z}tpmv`, `cblas_{s,d,c,z}tpsv`,
    `cblas_{c,z}geru`, `cblas_{c,z}gerc`, `cblas_{c,z}her`, `cblas_{c,z}her2`,
    `cblas_{c,z}trmv`, `cblas_{c,z}trsv`.
  - Level 3 (14): `cblas_{s,d,c,z}gemmtr`, `cblas_{c,z}symm`, `cblas_{c,z}syrk`,
    `cblas_{c,z}syr2k`, `cblas_{c,z}trmm`, `cblas_{c,z}trsm`.
- Declared deviation: complex `cblas_{c,z}syrk` and `cblas_{c,z}syr2k` with
  `ConjTrans` in column-major layout. Reference CBLAS forwards `'C'` to Fortran
  `C/ZSYRK` and `C/ZSYR2K`, which reject it (`INFO = 2`), and `cblas_xerbla` then
  terminates the process; here it is an invalid argument, a silent no-op that leaves
  `C` untouched. In row-major layout `ConjTrans` maps to `'N'` exactly like `Trans`,
  as in reference CBLAS 3.12.0 and 3.12.1. (#14)
- Declared deviation: `cblas_?gemmtr` follows 3.12.1, the first release that ships
  it. It adds the `?gemm`-style quick return `(alpha == 0 || k == 0) && beta == 1`
  (`C` bitwise untouched), which ZGEMMTR 3.12.1 lacks: it returns early only for
  `n == 0`, while the reference `?syrk`/`?syr2k` do have this quick return. Together
  with the `k == 0 -> C := beta*C` shortcut of `?gemmtr`/`?syrk`/`?syr2k`, it
  differs from the reference only in signed zeros, NaN or ±Inf `alpha` (with
  `k == 0`) and, for complex `?gemmtr` with `beta == 1`, Inf components of `C`, which
  the reference's `(1,0)*C` turns into NaN. (#14)
- Other declared deviations of the CBLAS entry points (#14):
  - There is no `cblas_xerbla`: invalid arguments and null pointers make the call a
    silent no-op, and a null `A`/`B` is a no-op even when `alpha == 0` means it would
    never be read, so `beta` is not applied in that case.
  - The new Level-2 routines `?tbmv`, `?tbsv`, `?tpmv`, `?tpsv`, `?trmv` and `?trsv`,
    and the existing real `?trmv`/`?trsv`, do not reproduce the reference's
    per-column skip of a zero `x(j)` in their `NoTrans` forms. An Inf/NaN in `A`
    facing a zero `x(j)` can therefore reach the result; finite results agree to
    rounding. The rank updates do skip, like the reference.
  - A `NonUnit` triangular solve (`?tbsv`, `?tpsv`, `?trsv`) stops at the first
    exactly-zero diagonal entry it reaches and writes back the partially solved `x`.
    Reference BLAS never stops. In the `NoTrans` forms it skips the division and the
    column update while the current `x(j)` is exactly zero, which gives a finite
    solution of a consistent singular system. Otherwise it divides by zero and
    propagates Inf/NaN; the `Trans`/`ConjTrans` forms always divide.
  - `rotmg` reproduces the 3.12.0/3.12.1 rescaling loops, in which a second rescale
    pass resets `h12` and `h21` (netlib master fixed this).
  - In row-major layout the conjugation paths can produce `+0.0` where reference
    CBLAS produces `-0.0` in an imaginary part, or the reverse; the values compare
    equal.
  - `cblas_i{s,d,c,z}amax` return a 32-bit `int`, but `cblas.h` declares the return
    type `CBLAS_INDEX` (`size_t`). These four entry points predate 0.2.3.
  - The `layout`, `trans`, `uplo`, `side` and `diag` parameters of every Level-2 and
    Level-3 entry point, new and old, are `#[repr(C)]` Rust enums carrying the
    `cblas.h` values (101/102, 111-113, 121/122, 131/132, 141/142). A C caller that
    passes any other value causes undefined behaviour, where reference CBLAS reports
    it through `cblas_xerbla`. This predates 0.2.3 and also affects the existing
    entry points.
  - There is no `_64` (ILP64) API. The symbols are `#[no_mangle]` functions in the
    `oxiblas-blas` rlib; this workspace builds no C-linkable library (`cdylib` or
    `staticlib`).

#### Matrix

- `HermitianMat::element(row, col)` returns the logical element by value in either
  triangle; `get` keeps returning the raw stored slot.

#### Sparse

- `pcg` is re-exported from `oxiblas_sparse::linalg` and `linalg::prelude` (and so
  from `oxiblas_sparse::prelude`); before, it was only in `linalg::iterative`.
  `oxiblas::prelude` exports it next to `gmres`, `cg` and `bicgstab`.

#### Examples and tooling

- Example `evd_svd_bench`
  (`cargo run --release -p oxiblas --example evd_svd_bench -- [--reps N] [SIZE ...]`)
  times `SymmetricEvd`, `SymmetricEvdDc`, `Svd` and `SvdDc` on fixed-seed random
  matrices (default sizes 200, 500, 1000 and 2000) and prints residual and
  orthogonality errors next to the median times.
- `crates/oxiblas-blas/tools/gen_tri_kernel_neon.py` regenerates
  `level3/trsm/tri_kernel_neon.rs`, the NEON tile kernels of TRSM and TRMM, which now
  carries an `@generated` header with the regeneration command. The script ships in
  the `oxiblas-blas` package; it is not used by the build.

### Fixed

- GEMM blocking no longer splits a dimension that fits in one block. Each
  cache-derived block was clamped to its dimension and then rounded down to a
  multiple of MR/NR, so f64 on the 8x6 kernel gave blocks `[252, 4]` for `n = 256`
  and `[4092, 4]` for `n = 4096`, and rescaling already clamped values gave
  `[168, 88]` for `4096x512x256`; every extra NC block re-packs all of `A`.
  `compute_blocking`, `compute_blocking_adaptive` and `GemmBlocking::asymmetric` now
  scale unclamped targets, and the GEMM core fits every blocking it receives
  (`for_kernel` and `custom` included) with `balanced_block(dim, target, unit)`: a
  dimension that fits is one block of exactly that size, a larger one is split into
  the fewest near-equal blocks (multiples of MR for MC, NR for NC and 8 for KC).
  Auto-tuned MC/KC/NC for f64 on the 8x6 kernel, with the cache sizes detected on an
  Apple M3: (1024,1024,256) 832/441/252 -> 512/344/256; (4096,512,256)
  1248/294/168 -> 1024/256/256; (64,64,4096) 40/64/4096 -> 64/64/4096; (2048,64,64)
  1248/64/36 -> 1024/64/64. Sequential throughput of two rectangular shapes on an
  Apple M3 is lower than before (64x64x4096 by about 10%, 37x4099x129 by 5-12%);
  the drop is reproducible and not explained by the blocking change. (#4)
- `Qz::compute` returned factors with `B != Q T Z^T`, a `T` that was not upper
  triangular, wrong eigenvalues, `beta < 0`, and `NotConverged` for larger pencils
  (for example `n = 12`) and for badly scaled ones; infinite eigenvalues were never
  deflated. See Changed for the rewrite. (#6)
- Row-major `cblas_{s,d}trsm` and `cblas_{s,d}trmm` were a silent no-op when
  `m != n`: the order of `A` was taken from the flipped side together with the
  unswapped `m`/`n`, the internal dimension check failed, and the wrapper dropped the
  `Err` (`trsm` also left `B` scaled by `alpha`). The order of `A` now comes from the
  caller's side. The other row-major Level-3 wrappers (`symm`, `syrk`, `syr2k`,
  `hemm`, `herk`, `her2k`) were checked against a dense reference for `m != n` and
  `n != k` and needed no change. (#7)
- `trsm_c32` and `trsm_c32_in_place` divided by the unconjugated diagonal for
  `Trans::ConjTrans` with `Diag::NonUnit`, at four sites in
  `level3/trsm/complex32.rs` (the blocked path included); `trsm_c64` and the generic
  `trsm` already conjugated it. (#8)
- `alpha == 0` no longer reads `A`/`B` and `beta == 0` no longer reads `C` (reference
  BLAS), so NaN/Inf in an ignored operand cannot reach the result in `gemm` and every
  path behind it (small, blocked and parallel; `gemm_transposed*`,
  `gemm_with_blocking`, `gemm_auto*`, `gemm_small`), `gemm3m_c64`/`gemm3m_c32` (and
  with them `gemm_transposed_c64/c32` and HEMM via GEMM), `gemm_strassen*`,
  `gemm_winograd*`, `gemm_cache_oblivious*`, the batched family (`gemm_batched*`,
  `gemm_strided_batched*`, `gemm_batched_c64/c32`, which also skips the transposed
  copies for `alpha == 0`) and `cblas_{s,d,c,z}gemm`. `scal_par` with `alpha == 0`
  multiplies like `scal` instead of zero-filling. (#9)
- `cblas_{s,d,c,z}gemv` follow the reference order for `alpha == 0`: quick return for
  `alpha == 0 && beta == 1`, scale (or zero-fill without reading `y` for
  `beta == 0`) only when `beta != 1`, then return before touching `A` and `x`. The
  complex wrappers no longer rescale `y` for `beta == 1` (`Inf+0i` became
  `Inf+NaN*i`). (#9)
- The same class in Level 3, audited against the reference: native `symm` (which
  also read `C` for `beta == 0`) and `hemm` had no `alpha == 0` exit;
  `syrk`/`syr2k`/`herk`/`her2k` sent `alpha == 0` into the GEMM path that expands `A`
  first (`syr2k` via GEMM also read `C` for `beta == 0`); the `cblas_{c,z}herk` and
  `cblas_{c,z}her2k` kernels read `A`/`B` for `alpha == 0` and `C` for `beta == 0`;
  and `cblas_{s,d}trsm` pre-scaled `B` by `alpha` and then solved with `A`. `trmm`
  (native and CBLAS) was already correct. `hemm_c64`/`hemm_c32` leave `C` unread for
  `beta == 0` on the 3M path without zero-filling it first, so a `-0.0` product
  stays `-0.0` as in `ZHEMM`. (#9)
- CBLAS Level-1 `cblas_?dot`, `?dotu_sub`, `?dotc_sub`, `?axpy`, `?copy` and `?swap`
  rejected `inc == 0` (the Level-2 rule); they now run the reference loop, which
  reuses element 0. The CBLAS Level-2 wrappers keep rejecting `inc == 0`, as the
  reference does. `?axpy` returns before touching `y` for `alpha == 0` on every
  stride (before, only on unit-stride real input). (#10)
- `nrm2_strided`, `cblas_{s,d}nrm2` and `cblas_{sc,dz}nrm2` returned 0 for
  `incx <= 0` (the pre-3.10 rule). They follow Netlib 3.10+: a negative increment
  walks the vector backwards and `incx == 0` gives `sqrt(n) * |x[0]|`. The complex
  norms go through the new `cnrm2_strided`, so `cblas_scnrm2` and `cblas_dznrm2` of
  `[Inf+Inf*i]` return `Inf` instead of NaN. (#10)
- `cblas_{c,z}scal` and `cblas_{cs,zd}scal` scaled the vector back to front for
  `incx < 0` and turned `inf+0i` into `inf+NaN*i` for `alpha == 1`. They now
  quick-return for `n <= 0`, `incx <= 0` or `alpha == 1`, as the reference `IF` does.
  (#10)
- `cblas_{s,d}gemv` now reject `incx == 0 || incy == 0` (a no-op that leaves `y`
  untouched, as `cblas_{c,z}gemv` already did); they used to alias every element onto
  element 0. (#10)
- `StridedSlice::from_raw_parts` and `StridedSliceMut::from_raw_parts` return an
  empty view, without touching the pointer, when the extent overflows `usize` (they
  panicked) or when its elements span more than `isize::MAX` bytes (they built a
  slice that `core::slice::from_raw_parts` forbids: undefined behaviour, an abort in
  debug builds). Both need contract-violating input, since no buffer is that large.
  Through the `extern "C"` wrappers that build strided views (`cblas_scnrm2`,
  `cblas_dznrm2` and the new rot, rotm and dot entry points), where a panic aborts
  the process, the first case needs a 32-bit `usize`, while the second is reachable
  on 64-bit targets too, e.g. with `n` and `|inc|` near `2^31`. (#10)
- `iamax_strided` and `iamin_strided` ranked complex elements by modulus for
  non-unit strides while the contiguous kernels use `|Re| + |Im|`. Both now rank by
  `cabs1` on every stride, with the same tie and NaN rules. (#11)
- `crotg` overflowed or underflowed (NaN, 0 or Inf) once an input exceeded about
  `sqrt(MAX)` (1e19 in f32, 1e154 in f64) or was tiny, because it divided complex by
  complex. It is now a port of LAPACK 3.12 `zrotg.f90`/`crotg.f90` (real-scalar
  divisions only, `safmax = 2/safmin`); the signature and `ComplexRotgResult` are
  unchanged. It follows the 3.10+ conventions: `b == 0` gives `c = 1, s = 0, r = a`
  (also for `a == b == 0`, which gave `c = 0, s = 1`), and `a == 0` gives `c = 0`,
  `r = |b|`, `s = conj(b)/|b|` (before: `r = b`, `s = 1`). (#12)
- `gemv`, `gemv_with_par` and `gemv_strided` applied `y := beta * y` (zero-filled `y`
  for `beta == 0`) when `A` was empty; like reference `?GEMV` they now return
  immediately and leave `y` untouched. `gemv_add_inplace` (so `gemv_add` and
  `gemv_sum2`) no longer poisons `y` with `alpha * 0` for an infinite or NaN `alpha`
  and an empty `op(A)`. (#13)
- `her` and `her2` kept the stored imaginary part of the diagonal. They now write
  `real(A(j,j)) + real(update)` and skip the column update when `x(j) == 0` (`her2`:
  `x(j) == 0 && y(j) == 0`), applying only `A(j,j) = real(A(j,j))`, like
  `?HER`/`?HER2`; `alpha == 0` and `n == 0` still leave `A` untouched. (#13)
- Native `level2::{ger, gerc, syr, spr, syr2, spr2, hpr, hpr2}` skip the column
  update of an exactly-zero scaling element, as reference BLAS does: `y(j)` for
  `ger`/`gerc`, `x(j)` for `syr`/`spr`/`hpr`, and `x(j)` and `y(j)` together for
  `syr2`/`spr2`/`hpr2` (`-0.0` counts as zero). An Inf/NaN in the other operand no
  longer reaches that column as `Inf * 0 = NaN`. `hpr`/`hpr2` still drop the diagonal
  imaginary part of a skipped column. `ger`/`gerc` now run column by column with the
  reference association `A(i,j) + x(i)*(alpha*y(j))` (`gerc`: `alpha*conj(y(j))`),
  and `syr2`/`spr2` evaluate `A(i,j) + x(i)*(alpha*y(j)) + y(i)*(alpha*x(j))` left to
  right, so finite results can differ from 0.2.2 in the last bit and intermediate
  over/underflow can differ. This also applies to `cblas_?ger`, `?syr`, `?syr2` and
  the new `?geru`, `?gerc`, `?spr`, `?spr2`, `?hpr`, `?hpr2`; in `RowMajor`,
  GER/GERU/GERC skip the row of a zero `x(i)`, as reference CBLAS does. (#14, #13)
- `level2::hpr` and `hpr2` mirror `her` and `her2` operation for operation, including
  the diagonals `real(A) + real(temp*x(j))` and
  `real(A) + real(temp1*x(j) + temp2*y(j))`. Packed and dense storage now give
  bitwise identical results for NaN-free input (with NaN input the NaN sign/payload
  may differ): `cblas_?hpr`/`?hpr2` match `cblas_?her`/`?her2`, and
  `cblas_?spr`/`?spr2` match `cblas_?syr`/`?syr2`, in both layouts. Diagonals of
  finite `hpr`/`hpr2` input can differ from 0.2.2 in the last bit. `her2`/`hpr2` now
  form the second temp as `conj(alpha*x(j))`, as ZHER2/ZHPR2 do, instead of
  `conj(alpha)*conj(x(j))`. The two temps differ only in the sign of an exact-zero
  (or NaN) imaginary part, so this changes only signed zeros and NaN signs, and the
  signed zeros now match the reference. (#13, #14)
- Native `rotg` and `rotmg` deviated from the reference in finite arithmetic. `rotg`
  is now a transcription of `drotg.f90`/`srotg.f90`: it scaled by `|a| + |b|` and
  overflowed to NaN for huge finite inputs, returned `-0.0` for `s`/`z` (`b == 0`,
  `a < 0`) and for `c` (`a == 0`, `b < 0`) where the reference gives `+0.0`, and
  differed in the last bit in ordinary f32 cases. `rotmg` compared against exactly
  `2^24` and `2^-24`; it now uses the reference `GAMSQ`/`RGAMSQ` literals
  (`16777216`/`5.9604645e-8` in f64, `1.67772e7`/`5.96046e-8` in f32) and still
  rescales by exactly `2^24`, so inputs near `2^24` or `2^-24` rescale like the
  reference. `cblas_?rotg` and `cblas_?rotmg` inherit both changes. Found while
  adding the CBLAS entry points. (#14)
- The new `cblas_{s,d}rot`, `cblas_{s,d}rotm`, `cblas_csrot`, `cblas_zdrot`,
  `cblas_sdsdot` and `cblas_dsdot` return without touching the operands (`alpha` for
  `sdsdot`, `0` for `dsdot`), instead of aborting the C caller with a length-mismatch
  panic or building an oversized slice, when called with an `(n, inc)` that no buffer
  can hold: an extent `(n-1)*|inc|+1` that overflows `usize` (32-bit targets such as
  wasm32) or spans more than `isize::MAX` bytes (64-bit targets too, e.g. with `n`
  and `|inc|` near `2^31`). The unit-increment path of `sdsdot`/`dsdot` applies the
  same byte bound to `n`, which only a 32-bit target can exceed. Contract-violating
  input only. (#14)
- Contiguous `rotm` handled only the flags `-2`, `-1`, `0` and `1` and silently
  ignored any other value; it now classifies the flag like `drotm.f` (`flag + 2 == 0`
  is the identity, `flag < 0` the full `H`, `flag == 0` the unit diagonal, anything
  else, NaN included, the unit off-diagonal). Contiguous `zdrot` multiplied by
  `from_real(c)`, which gave NaN cross terms for infinite input; it now scales the
  real and imaginary parts separately, as `zdrot.f` does. (#15)
- Blocked GEMM (`m*n*k > 32^3`) read the wrong elements when an operand was a view
  with `col_stride != 1`, such as `MatRef::transpose()`: the NoTrans fast packers
  `pack_a_optimized` and `pack_b_optimized` assume unit column stride. They are now
  used only for NoTrans operands with `col_stride == 1`, and every other operand is
  read through `read_op`, which honours both strides. No call site in this workspace
  was affected (LAPACK Cholesky transposes into an owned matrix); external callers
  that passed such views to `gemm`, `gemm_with_par` or `gemm_with_blocking` got
  wrong results. (No issue number; found while working on #4 and #16.)
- `gemm_small_f32_simd` and `gemm_small_f64_simd` (aarch64) zero-filled `C` on their
  tiny-problem path instead of accumulating into it. They now accumulate, fall back
  to the portable loops for strided or tiny operands, and assert consistent shapes
  (they panic on a mismatch, because the NEON tiles address the operands through raw
  pointers). (#16)
- The `compare-openblas` benchmark never ran OpenBLAS. `oxiblas-blas` exports
  `#[no_mangle]` `cblas_*` symbols, so the old bench's `cblas_*` calls, direct or
  through `ndarray`, bound to OxiBLAS at static link, and the linker then dropped
  `libopenblas`. Both sides of every comparison were OxiBLAS. `benches/comparison.rs`
  now loads OpenBLAS at run time with `dlopen` and refuses to run if a symbol resolves
  into its own image. It pins OpenBLAS to one thread for the sequential groups (OpenMP
  builds also need `omp_set_num_threads`) and checks that by CPU time. New groups
  `gemm_parallel_comparison` and `sgemm_parallel_comparison` compare `Par::Rayon` with
  multi-threaded OpenBLAS at the same thread count, which `RAYON_NUM_THREADS` sets for
  both. The `ndarray-linalg`, `blas-src` and `cblas-sys` dependencies are removed from
  the workspace. (No issue number.)
- Published performance figures were wrong in two ways, and every README is corrected:
  - GFLOPS were computed as 16 × Gelem/s, but real GEMM does 2 flops per element, so
    they were 8 times too high. The old "427 Gf/s" on an Apple M3 was 53 GFLOPS.
  - Every "OpenBLAS" column and every "% of OpenBLAS" ratio came from the benchmark
    above and compared OxiBLAS with itself. None of them holds. Measured against real
    OpenBLAS 0.3.30, GEMM on one thread was close, and DOT, NRM2, GEMV, TRSM and SYRK
    were slower before the second-round kernel work below. The README has the current
    numbers.
  - The one-thread TRSM figures at n = 256 and 512 were not one-thread figures: the
    benchmark crate enables `oxiblas-blas/parallel`, with which `trsm_in_place` ran its
    GEMM updates on the Rayon pool from order 256 (see Changed). The published 5.67 ms
    at n = 512 took 6.55-6.93 ms on one thread. The 32.2 GFLOPS of TRSM at n = 1024 in
    the table against naive loops did not reproduce (25.4-25.6 in later runs of the
    same code); the TRMM figure next to it did.

  Older entries in this file keep the figures as published at the time.
- The `syrk_syr2k_bench` and `trmm_trsm_bench` examples counted twice the flops of
  SYRK, SYR2K and TRMM, so their GFLOPS were 2 times too high.

#### BLAS

- `gemv` and `gemv_with_par` with f64 `NoTrans`, `m*n > 4096` and a view whose
  `col_stride() != 1` (for example `MatRef::transpose()`) read the wrong elements on
  aarch64 (NEON) and on x86_64 with AVX2. Such views now take the stride-aware
  portable code.
- The SIMD `nrm2` paths (NEON and AVX2, f64 and f32) returned Inf or NaN when the
  largest `|x_i|` was below `2^-1024` (f64) or `2^-128` (f32). They now return the
  norm.
- `pack_a_optimized`, `pack_b_optimized`, `pack_a_contiguous`, `pack_b_streaming` and
  the SIMD `pack_a` wrappers wrote past an undersized buffer or read outside the
  matrix. They now panic on an undersized buffer, and views with a non-unit
  `col_stride` are packed correctly. The transpose-aware packing inside GEMM checks
  the buffer length in release builds too.
- `get_cache_info` on Apple Silicon read the efficiency-core `hw.l1dcachesize` and
  `hw.l2cachesize`. It now reads the performance-core `hw.perflevel0.*` keys, falling
  back key by key, takes L2 as the per-core share (`l2cachesize / cpusperl2`, 4 MiB
  on an M3) and treats empty or failing keys as missing. The auto-tuned GEMM blocking
  on an M3 is unchanged (tested); chips with a different per-core L2 share (an M1 has
  3 MiB) can get different blocking.
- `trsm_in_place`, `trsm_c32_in_place` and `trsm_c64_in_place` report `Singular`
  before touching `B`, which is left unchanged (it was partly solved). `trsm`,
  `trsm_c32` and `trsm_c64` with `alpha == 1` copy `B` instead of multiplying it by
  `1 + 0i`, which turned infinite entries into NaN. `trmm` no longer spreads Inf or
  NaN in `B` through the structural zeros of the triangle (reference semantics).
- Complex TRSM (`trsm_c32`, `trsm_c64`, and `trsm`/`trsm_in_place` on complex types)
  gave 0, NaN or inaccurate solutions for a diagonal entry of modulus above about
  `1.3e154` (`1.8e19` for `Complex32`) or whose squared modulus underflows. Such
  entries are now detected and divided with the robust Baudin-Smith algorithm of
  LAPACK `?LADIV`; real TRSM is unchanged.
- Complex `syrk`/`syr2k` with `beta == 1` no longer turn an infinite entry of `C`
  into NaN, and `gemmt` with `alpha == 0` and `beta == 1` leaves `C` untouched.
- `her_new`, `her2_new`, `syr_new` and `syr2_new` computed direct products. They now
  run the in-place reference kernels on a zero matrix and mirror the result, so a
  column with a zero `x(j)` (`her2`/`syr2`: zero `x(j)` and `y(j)`) is skipped as in
  `her`/`syr`, a `-0` product is stored as `+0`, and finite results can differ from
  0.2.2 in the last bit.
- `tensor::contract_2d` read past `a` or `b` (undefined behaviour) on its GEMM path
  (a dimension of 32 or more) when a slice was shorter than its shape, and panicked
  on the small path. It now returns `TensorError::DimensionMismatch`, also when the
  shape product overflows.
- The `oxiblas-blas` unit tests build on 32-bit targets (wasm32, i686) again:
  Strassen's next-power-of-two helper shifted a `usize` by 32, which
  `deny(arithmetic_overflow)` rejects.

#### CBLAS

- The unit-stride paths of the CBLAS Level-1 wrappers that predate 0.2.3
  (`cblas_?dot`, `?nrm2`, `?asum`, `?scal`, `?axpy`, `?copy`, `?swap`, `i?amax` and
  the complex `?copy`/`?swap`) check that `n` elements fit in `isize::MAX` bytes
  before building a slice; on a 32-bit target a larger `n` built a slice that
  `slice::from_raw_parts` forbids (undefined behaviour). Contract-violating input
  only; the strided (`inc != 1`) paths still lack this guard (see TODO.md).

#### LAPACK

- The LU and Cholesky families compared every pivot with the absolute `eps·n`, so
  the outcome depended on the scale of the input: they rejected well-conditioned
  matrices of small magnitude and accepted numerically singular ones with entries
  larger than 1. 0.2.2 listed this as fixed, but the absolute test was still in
  place. Each pivot is now compared with a reference from its own
  column of the input, computed once from the finite entries and shared by every
  code path (unblocked, blocked, recursive, parallel):
  - `Lu`, `LuRook`, `LuFullPiv` and `BandLu`: `|u_jj| <= eps·n·s_j` is singular, with
    `s_j` the largest entry of the original column `j` (rook and full pivoting follow
    their column interchanges; `BandLu` uses the stored band).
  - `Cholesky` (every variant), `HermitianCholesky`, `BandCholesky`,
    `PackedCholesky`: the radicand `a_jj − Σ|l_jk|²` must exceed `eps·n·a_jj`.
  - `Ldlt`, `PackedLdlt`: `|d_j| <= eps·n·s_j` over the whole symmetric column
    `j`; `Ldlt::inertia` counts `|d_j| <= eps·s_j` as zero.
  - Unchanged: `LuFullPiv::rank` keeps its global `n·eps·‖A‖` threshold, and an
    explicit tolerance (`Ldlt::compute_with_tol`) is absolute.

  The outcome is now invariant under power-of-two scaling. Newly accepted: matrices
  scaled by `1e-30`, `MIN_POSITIVE·I`, `diag(1, 1e-16)`. Still accepted:
  `diag(1e20, 1)`, `diag(1e6, 1e-10)`, stiffness matrices with a `1e20` penalty
  term. Newly rejected: cancellation residues in columns after the first, for example
  `[[1,2,3],[4,5,6],[5,7,9]]` gives `Singular { index: 2 }` (0.2.2: `Ok`, determinant
  about `-3e-15`). A NaN pivot or radicand propagates into the factors instead of
  being reported (LAPACK `dpotrf` reports `INFO = j`). `utils::det`, `solve` and the
  other solvers built on these factorizations inherit the rule.
- `SvdDc` computed the bidiagonal SVD by Cuppen's method on `BᵀB` and formed `U` as
  `B·V/σ`. That was not backward stable once `σ_min/σ_max` fell below about `1e-4`
  (a graded n = 60 matrix: residual `3e-8`, one singular value 19% too small; f32
  with condition `1e3`: residual `3e-4`), and `‖UᵀU − I‖` was `3e-7` to `3e-6` at
  n = 200 to 2000. It now follows LAPACK `dgesdd`: divide and conquer on the
  bidiagonal itself, ported from `dbdsdc` (`dlasd0`-`dlasd5`, `dlaed6`, and
  `dlasdq`/`dbdsqr` for blocks of at most 25 rows). Residual and loss of
  orthogonality are `O(n·eps)`, every singular value is within `O(n·eps·‖A‖)`, and
  the bidiagonal SVD agrees with reference LAPACK 3.12 `dbdsdc`/`dqds` to `1.5e-15`
  relative on a test bidiagonal. `ComplexSvdDc`, `TruncatedSvd` and `RandomizedSvd`
  share the bidiagonal solver. Deviations from LAPACK (power-of-two input scaling, no
  `|d_i| < eps` to `±eps` replacement, the avoided `dlasd4` largest-root `TAU = RHO`
  quirk, a bisection fallback) are documented in `svd::bidiag_dc`; below
  `eps·‖B‖` only the absolute bound holds.
- `SvdDc` and `ComplexSvdDc` depended on the magnitude of the input: with more than
  25 columns, entries below about `1e-7` or above `1e154` gave `Ok` with singular
  values up to 19% off; with 25 or fewer, entries above `1e80` or below `1e-160` gave
  `NotConverged`. The input is now scaled by a power of two as in `dgesdd` (and the
  bidiagonal as in `dbdsdc`), so the results are exactly scale invariant for
  power-of-two scalings. `SvdDc` and `ComplexSvdDc` also returned `NotConverged`
  for every matrix of at most 25 columns whose bidiagonal has a zero diagonal entry
  (for example `[[0,1],[0,1]]`); the bidiagonal QR is now a `dbdsqr` port with
  exact-zero chasing. Rank-deficient input no longer produces NaN from subnormal
  reduction noise, completing `U` no longer takes `O(n⁴)` time, and singular values
  are sorted with a NaN-safe total order.
- `ComplexSvdDc` returned all-zero singular values for small matrices (entries near
  `1e-150`) and overflowed above about `1e154`.
- `ParallelSvdDc` returned `Ok` with wrong factors for most inputs larger than 3x3
  (random 200x200: residual 1.4, `‖VᵀV − I‖ = 1`). `ParallelSvdDc::compute` now
  runs `SvdDc` with its GEMMs on the Rayon pool and returns exactly the factors of
  `SvdDc::compute`; the parallelism is GEMM-only.
- `Svd::compute` (one-sided Jacobi) completed the basis of `U` for rank-deficient or
  tall input in `O(m⁴)` time, restarting its candidate scan for every column (300x31:
  1.93 s, 1000x20: 282 s in release builds; now 0.0037 s and 0.077 s). The completed
  columns are reorthogonalized (DGKS) and are orthogonal to about `3e-14` instead of
  `1e-11`; singular values, `V` and the columns of `U` for nonzero singular values
  are bit-identical.
- `SymmetricEvd` and `SymmetricEvdDc` overflowed for entries near `1e±300`; the input
  is now scaled into the safe range as in `DSYEVD`. Non-finite input fails at once
  with `NotConverged`. `SymmetricEvdDc` splits the tridiagonal at negligible
  off-diagonal entries (a diagonal input returns its exact diagonal), and its leaf
  blocks use the `dsteqr` deflation test.
- The secular-equation solver of `SymmetricEvdDc` (also used by `HermitianEvdDc`)
  could accept an unconverged root. It now starts from a `DLAED4`-style first
  iterate, uses cancellation-free quadratic roots, a two-pole model for the last root
  and the `DLAED4` stopping bound including `|tau|·f'`, and returns
  `SecularEquationFailed` instead of accepting an unconverged iterate. Roots next to
  poles with tiny weights are accurate.
- `HermitianEvdDc` returned non-finite or wrong eigenvalues for n > 100: its private
  secular solver took an inverted Newton step, used absolute tolerances and never
  checked convergence. It is now LAPACK `ZHEEVD` (see Changed); NaN or infinite input
  returns `NotConverged`.
- `compute_with_block_size(a, 0)` and `compute_blocked_par_with_block_size(a, 0)` of
  `Lu` and `Cholesky`, `Qr::compute_blocked(a, 0)`,
  `Hessenberg::compute_blocked_with_block_size(a, 0)` and
  `BidiagFactors::compute_blocked_with_block_size(a, 0)` looped forever for any
  non-empty matrix; a block size of 0 is now treated as 1.
- `GeneralEvd::compute` and `compute_full` paired eigenvectors with the wrong
  eigenvalues (`verify()` gave about 2.24 for `[[2, 1], [1, 3]]`, and `diag(2, 3)`
  paired λ = 3 with `e1`). Several defects compounded: `Schur` returned the 2x2
  eigenvalues as `[λmax, λmin]` whatever the order on the diagonal of `T`, left 2x2
  blocks with real eigenvalues unsplit, and its eigenvectors treated every
  eigenvalue as a 1x1 block. The real Schur form is now computed as LAPACK
  `DLAHQR`/`DLANV2` do (Francis double shift, Ahues–Kressner deflation, exceptional
  shifts, every 2x2 block standardized: real eigenvalues split, complex blocks with
  equal diagonal entries), eigenvalues are listed in the order of `T`'s diagonal,
  and eigenvectors come from `DTREVC`-style back-substitution (perturbed
  near-singular pivots, overflow-safe rescaling) transformed back with `Q`. Every
  eigenpair now satisfies `‖A·v − λ·v‖ = O(n·eps·‖A‖)`, complex pairs and defective
  matrices included, and `compute_full` takes its left eigenvectors from the same
  Schur form, paired by index (`uᴴ·A = λ·uᴴ`).
- `Schur::compute` returned `NotConverged` for many ordinary matrices (about half of
  random matrices with n >= 5, companion matrices of `x^n − 1`): a deflated 2x2 block
  with real eigenvalues stalled the double-shift iteration.
- `trevc_right` back-substituted a complex pair with the conjugate eigenvalue and
  treated 2x2 blocks above a column as 1x1 (NaN for a block with real eigenvalues);
  `trsna_s`/`trsna_sep` now detect blocks exactly as `Schur` does.
- `ComplexGeneralEvd` returned wrong eigenvalues for n >= 3 (its private QR step was
  not a similarity transform, off by up to a third of `‖A‖`) and left eigenvectors
  that were not paired with their eigenvalues. It now uses `ComplexSchur` and
  `ZTREVC`-style back-substitution. `ComplexSchur` converges at any scale,
  `ComplexHessenberg` is a similarity for small-norm inputs, and
  `ComplexSchur::residual` is relative at every scale. `Schur`, `ComplexSchur`,
  `GeneralEvd` and `ComplexGeneralEvd` scale by an exact power of two internally, so
  inputs around `1e±150` work.

#### Sparse

- The iterative solvers of `linalg::iterative` tested breakdown against absolute
  thresholds and could report convergence from a drifted recursive residual. Breakdown
  tests are now relative (scale-invariant), and a breakdown is decided by the true
  residual `‖b − A·x‖` (a converged result if it meets the tolerance, `Breakdown`
  otherwise). `gmres`, `pgmres`, `fgmres`, `block_gmres`, `idrs`, `pidrs`, `qmr` and
  `pminres` also confirm normal convergence with the true residual (and report it as
  `residual_norm`). Norms are overflow-safe, and systems far from unit scale (largest
  magnitude of `A` or `b` outside `2^±63` for f64, `2^±7` for f32) are solved on a
  copy rescaled by powers of four and mapped back exactly. `(2^±520·A, 2^±520·b)`
  converges with the same iteration counts and solutions as `(A, b)`, `b` near `1e160`
  no longer converges falsely, and well-scaled results are bit for bit unchanged.
- The GMRES family restarted after every step: its lucky-breakdown test read a
  sub-diagonal entry that had already been zeroed, so `gmres(.., restart = 30, ..)`
  behaved like GMRES(1). On `laplacian_2d(100, 100)` with `tol = 1e-8`, GMRES(30) now
  needs 1,398 iterations instead of 37,571. Convergence is confirmed with the true
  residual (one extra product with `A`, not counted in `iterations`), and
  `residual_norm` is that true residual.
- Solver-specific fixes: `block_cg` and `block_pcg` reported a breakdown as
  convergence; `block_gmres` assembled its block Hessenberg matrix wrongly; `minres`
  and `pminres` could report false convergence, and `pminres` used the wrong
  coefficient (`vᵀ·M·A·v`); `ptfqmr` mixed preconditioned and unpreconditioned
  quantities (it is now right-preconditioned, with two preconditioner applications
  per iteration); `idrs`/`pidrs` did not clear `f` between cycles; `bicgstab` and
  `tfqmr` tested the previous `rho` for breakdown.
- `pcg` and `block_pcg` check `rᵀz` and `pᵀAp`: when either is negligible relative
  to the norms of its factors, is NaN, or changes sign relative to the first
  iteration (indefinite `A` or preconditioner), the true residual decides between a
  converged result and `Breakdown`, instead of a division that returned Inf or NaN.
  Negative-definite `A`/`M` pairs converge, as their iterates are those of the
  corresponding positive definite problem. IC(0)-preconditioned
  `pcg` with `tol = 1e-10` on `laplacian_2d(10, 10)` failed at iteration 13 under the
  old absolute `|pᵀAp| <= eps` test and now converges.
- Preconditioner setup is scale-invariant: building on `s·A` succeeds exactly when
  building on `A` does and gives exactly the correspondingly scaled preconditioner
  (tested with `2^±64` on 17 variants). The "zero diagonal" errors of `Jacobi`,
  `GaussSeidel`, `SOR`, `SSOR`, `Polynomial`, the `AMG`/`SAMG` diagonal inverse and
  the Schwarz local Jacobi fire only for exact zeros (`Jacobi::new` rejected
  `|d| < 1e-14`). The pivots of `BlockJacobi` (Gauss-Jordan), the Schwarz `ExactLU`
  and `ILU0` local solvers and `SPAI` use the dense LU rule; the negligible-pivot test
  of `AINV`, the `SPAI` Tikhonov term and the `AMG` interpolation denominator are
  relative.
- `IC0` dropped the coupling terms of the elimination: `l_ij` now subtracts
  `Σ_{k<j} l_ik·l_jk` over the sparsity pattern, so `L·Lᵀ` matches `A` on its pattern
  (for a 9-point stencil it missed by up to about 0.25). Factors of tridiagonal and
  5-point matrices are unchanged. Unsorted CSR rows are handled.
- `ICT` read row `j` of `L` instead of column `j`, so it never updated the
  off-diagonal entries and never created fill-in. It now eliminates correctly with
  fill-in; a missing diagonal entry is `ZeroDiagonal`; the drop rule is
  `|l_ij|·l_jj <= tau·‖row_i(A)‖₂` (Saad's ILUT rule applied to `U = D·Lᵀ`; the old
  rule could not be scale-invariant) with an overflow-safe row norm, and `p` caps the
  off-diagonal entries per row.
- The Chebyshev `Polynomial` preconditioner grew its recurrence by `2·(d/c)` instead
  of `2·(c/d)`, so it was not a Chebyshev polynomial and could be indefinite. It now
  follows Saad, Algorithm 12.1; degrees 0 and 1 are unchanged, a zero-width interval
  no longer gives NaN, and invalid eigenvalue bounds return
  `PreconditionerError::InvalidMatrix`.
- Sparse factorizations: a pivot or radicand is negligible when it is at or below
  `eps·m·s`, with `s` the reference from the original column (LU) or diagonal entry
  (LLᵀ) and `m` the number of terms that formed it, instead of the order `n` (an
  identity of order `1e5` with one block `[[1, 1], [1, 1 + 1e-12]]` no longer fails).
  - `SparseCholesky`: a radicand at or below the threshold (or NaN) is
    `NotPositiveDefinite`, `ZeroDiagonal` means a diagonal entry that is not stored,
    only exact zeros are dropped, and `L(s·A) = √s·L(A)`. `IC0` and `ICT` use the
    same rule.
  - `SparseLU` compared pivots with the absolute `eps` and dropped small entries; it
    now keeps every nonzero. Its row swaps did not swap the multipliers already
    computed, so any swap after the first step gave `P·A != L·U` and wrong solutions
    (the shift-invert and generalized sparse eigensolvers solve with it).
  - `ILU0` tests every pivot, the last row included, and reports a missing diagonal
    entry; `ILU0::apply` divided by the multipliers twice and now applies
    `(L·U)⁻¹`; unsorted rows are eliminated in column order.
  - `ILUT` and `ILUTP` drop `|l_ij·u_jj| <= tau·‖row_i(A)‖₂` (overflow-safe norm),
    test pivots relatively, and track fill-in exactly (small fill-in was lost, leaked
    or stored twice). `ILUTP` with `pivot_tol = 0` never pivots and follows SPARSKIT
    `permtol` otherwise, and it renumbers `U` after a column swap (complete `ILUTP`
    did not reproduce `A·Q`: error 1.39, now `1.5e-16`).
  - `SAMG` returns a `MatrixConstruction` error instead of falling back to a silent
    zero operator or panicking.
- Sparse formats:
  - `CsrMatrix::new` and `CscMatrix::new` reject a first pointer other than 0 (with
    `nrows == 0`, `row_ptrs == [nnz]` smuggled in `nnz` stray entries), report an
    overflowing `nrows + 1` as an error, and `CscMatrix::new` rejects duplicate rows
    in a column (`DuplicateEntry`), as `CsrMatrix::new` did for columns.
  - `CscMatrix::transpose` returned its input unchanged; it now returns the
    transpose.
  - `spadd` and `rank1_update` accept unsorted rows.
  - `to_csr` of ELL, SELL, HYB and BSR, and `BscMatrix::to_csc`, sum a position that
    is stored twice instead of emitting a duplicate.
  - `HybMatrix::new` and `SellMatrix::new` reject out-of-range indices, and
    `HybMatrix::from_ell` ignores padding.
  - `csr_to_dia` lost the entries of wide matrices; offsets are normalized.
  - The internal `rows_to_csr` helpers of the sparse factorizations were unsound for
    fewer than `nrows` rows; they are now correct by construction.

#### Core

- `SimdCapabilities::cache_line_bytes` (and the `multiver` mirror) reported a fixed 64.
  They now report the compile-time `CACHE_LINE_SIZE`: 128 on aarch64 (chosen for
  Apple cores; many other aarch64 cores have 64-byte lines) and 64 elsewhere.
- `Real::signum` of `f16` and `QuadFloat` follows `f32::signum`: `±0` gives `±1`, NaN
  gives NaN, and `QuadFloat` maps `-inf` to `-1` (it gave `+1`).
- With `sve-nightly`, `oxiblas-core` now declares
  `feature(stdarch_aarch64_sve)` under
  `cfg(all(feature = "sve-nightly", target_arch = "aarch64", target_feature = "sve"))`,
  which the SVE kernels need. Compile-checked with a nightly toolchain (2026-09-06)
  and `-C target-feature=+sve`; no SVE hardware was used.
- `QuadFloat` division, `recip` and negative `powi` were only double precision
  (`1/3` was `fl(1/3) + 0` and `3/3 != 1`), because twofloat 0.8.4 forms the
  residual without FMA. They now use an FMA-based double-double division accurate to
  about `0.5·2^-106`. Special values follow IEEE 754: `x/±0 = ±inf`,
  `x/±inf = ±0`, `±inf/x = ±inf`, with signed zeros (NaN before). `tan`, `atan`,
  `atan2`, `log`, `log2`, `log10`, `%` and `hypot` use the new division,
  `atan(±inf) = ±π/2`, and `powi(x, i32::MIN)` no longer panics in debug builds.

#### Matrix

- `MatRef::ptr_at`, `MatMut::ptr_at` and `MatMut::ptr_at_mut` formed out-of-range
  pointers with in-bounds pointer arithmetic, which is undefined behaviour reachable
  from safe code. They now use wrapping arithmetic: any indices are safe to pass, and
  only the pointer of an in-range element may be dereferenced.
- `MatMut::split_cols`, `split_rows` and `col_as_slice(_mut)` no longer form offsets
  past the allocation (padded submatrices, tight `from_strided` views, empty views);
  `MatMut::submatrix` checks its bounds without overflow and accepts empty
  submatrices at the right and bottom edges; `split_cols`/`split_rows` panic instead
  of returning overlapping mutable views of a `MatMut::new` view whose `row_stride` is
  smaller than its row count; `MatRef::diagonal` no longer overflows for huge strides.
  Checked with Miri (15 soundness tests, 22 reproducer runs).
- `HermitianMat::set(row, col, z)` on the non-stored triangle stored `z` without
  conjugating it, so the matrix read back `conj(z)` at `(row, col)`. It now gives
  `A[row, col] = z` and `A[col, row] = conj(z)`; a write to the diagonal (also
  `set_diagonal`) keeps only the real part.

#### Examples and tooling

- `lu_bench` printed `nb = 64` whatever block size it used; it now prints the real one
  (n < 64: n; below 256: 32; below 1024: 64; else 128), and notes that its matrices
  never interchange rows. `cholesky_bench` prints the block size from
  `optimal_block_size_cholesky`.
- `sparse_matrices` passed `gmres` its `max_iter` and `restart` swapped.
- Example text that no longer matched the code: `trmm_trsm_bench` (blocked TRMM/TRSM,
  no dense expansion), `syrk_syr2k_bench` (triangle-only kernel),
  `extended_precision` (the naive/pairwise/Kahan dot contrast, `dsdot`),
  `tensor_operations` (22 einsum patterns, not 24), `symm_hemm_bench` and
  `complex_gemm_bench` (nominal `8n³` flops against about `6n³` for 3M; NEON on
  aarch64 only) and `sparse_matrices` (no ILUT, no `parallel` feature needed, no
  preconditioned solve).

### Changed

- The minimum supported Rust version (`rust-version` in `Cargo.toml`, and `msrv` in
  `clippy.toml`) rises from 1.85 to 1.89, which the code already needed: the x86_64
  kernels use AVX-512 intrinsics and target features stabilized in Rust 1.89,
  `memory::numa` uses let chains (Rust 1.88), and `nalgebra` 0.35 (the `nalgebra`
  feature, part of `full`) requires 1.89. The clippy suggestions that the higher MSRV
  enables (`is_multiple_of`, let chains) are applied. Of these, only
  `memory::is_aligned(ptr, 0)` behaves differently: it returns `false` for a non-null
  pointer instead of panicking on a division by zero.
- `Qz::compute` is rewritten as a Pure Rust equivalent of LAPACK 3.12 `DGGES` with
  `JOBVSL = JOBVSR = 'V'` and `SORT = 'N'`, in the new module `evd/qz/` (replacing
  `evd/qz.rs`): norm scaling, `DGGBAL`/`DGGBAK` with `'P'`, QR of `B`, `DGGHRD`,
  `DHGEQZ` with single and Francis double shifts and relative deflation tests,
  zero-chasing deflation of infinite eigenvalues (exact `beta = 0`), `DLAG2`/`DLASV2`
  standardization of 2x2 blocks, and the `DGGES` unscaling guards. Eigenvalues are
  returned as `DHGEQZ` computes them: `beta >= 0` (enforced by negating column `j` of
  `S`, `T` and `Z`, so `n = 1` with a negative `B` gives `Z = -1`), and a complex
  pair stores the member with `alpha_imag > 0` first (the two `beta`s differ, so only
  the ratios `alpha/beta` are conjugate). `Qz`, `QzError` (the same four variants)
  and `GeneralizedEigenvalue` keep their public API. The factors `S`, `T`, `Q` and `Z`
  differ from 0.2.2. The tests check the LAPACK `ddrges` ratios (threshold 30) and
  the exact Schur structure; results are not claimed to match Netlib bit for bit.
  (#6)
- `GeneralizedEigenvalue::is_finite` is now `beta != 0` and `is_real` is now
  `alpha_imag == 0`; they compared `|beta| > eps` and `|alpha_imag| <= eps` against
  the machine epsilon, so the answer depended on the scale of the pencil. `value` and
  `value_complex` follow. A singular pencil (`alpha == beta == 0`) is reported as not
  finite. Non-finite input to `Qz::compute` returns
  `QzError::NotConverged { iterations: 0 }`. The exact `beta = 0` holds for
  `Qz::compute`. After `compute_sorted` or `reorder` the members of a complex pair
  come from `DLAG2` on the final 2x2 block (shared `beta`, exactly negated
  `alpha_imag`, unless a `DGGES` unscale guard rescaled one), and an infinite
  eigenvalue that took part in a swap has a rounding-level `beta`, so `is_finite`
  and `num_infinite` can count it as finite. (#6)
- GEMM block sizes are targets fitted to each call: `gemm_with_blocking` and
  `gemm_transposed_with_blocking` refit any `GemmBlocking` they receive
  (`custom`, `asymmetric`, `for_kernel`, `auto_tuned`) with `balanced_block`. The
  `mc`/`nc` from `compute_blocking` may therefore be the full dimension and need not
  be multiples of MR/NR, and `custom(.., kc = 1, ..)` runs with `kc = 8` when
  `k > 8` (the struct still reports `kc = 1`). KC is rebalanced, which changes the
  floating-point summation order of GEMM wherever it is used, LAPACK included:
  results are not bitwise equal to 0.2.2. (#4)
- Native `trsm`, `trsm_c32` and `trsm_c64` with `alpha == 0` validate the shapes and
  return zeros without reading `A` or `B`, as reference `?TRSM` does. A singular `A`
  used to give `Err(Singular)`, and NaN/Inf in either operand leaked into the
  result. (#9)
- `herk` and `her2k` (native and CBLAS) follow the reference quick return: for
  `(alpha == 0 || k == 0) && beta == 1`, `C` is left untouched, a non-real diagonal
  included; whenever `C` is updated its diagonal is still made real. Declared
  deviation: NaN and Inf propagate differently from Netlib `ZHERK`/`ZHER2K` as
  gfortran compiles them with default floating-point flags, which turn a real scalar
  times a complex value into a complex multiply by `(r, 0)`, so `2*(Inf, 1)` is
  `(Inf, NaN)`. (#9)
  - For `beta` other than 0 and 1, the native routines scale off-diagonal elements
    by a complex multiply by `beta + 0i`, as the reference does, but multiply
    diagonal elements the same way before keeping the real part, where the reference
    scales the real part only (`BETA*DBLE(C(J,J))`): an infinite or NaN imaginary
    part on the diagonal gives NaN instead of `beta*Re(C(j,j))`. The CBLAS kernels
    scale the diagonal as the reference does and the off-diagonal elements
    componentwise (`2*(Inf, 1)` gives `(Inf, 2)`).
  - For `beta == 1` the reference leaves `C` unscaled for `NoTrans` but still
    multiplies its off-diagonal elements by `(1, 0)` for `ConjTrans`. The CBLAS
    kernels, and the native routines when `n < 32` or `k < 8`, leave `C` unscaled
    in both cases; otherwise the native routines (GEMM path) multiply in both, the
    diagonal included.
  - NaN and Inf in `A` or `B`: the reference `NoTrans` update skips column `L` when
    `A(J,L) == 0` (`ZHER2K`: when `A(J,L)` and `B(J,L)` are both zero), and no path
    here has that skip. The CBLAS `NoTrans` kernels (pre-existing), like the native
    GEMM path, form dot products and apply `alpha` to the sums, where the reference
    multiplies `alpha` into each `A(J,L)` / `B(J,L)` first as a complex multiply
    (`ZHERK`: `TEMP = ALPHA*DCONJG(A(J,L))`). TODO.md lists the remaining
    differences in where `alpha` is applied; alignment is a 0.2.4 candidate.
- Performance: `Par::Rayon` GEMM uses a new 2D (M x N) driver
  (`level3/gemm_parallel.rs`, feature `parallel`). The old driver split `C` over NC
  column blocks only; NC is sized for the last-level cache, so every problem with
  `n <= NC`, square ones included, was a single work item. Each NC block of `C` is
  now cut into a grid of tiles aligned to the micro-tile grid (about three per
  thread), the `B` panel of each KC step is packed once, in parallel, into a shared
  buffer, and the tiles run as one rayon task each on disjoint `MatMut` views.
  Results are bitwise identical to `Par::Seq` for any thread count (tested on
  aarch64 for f32, f64 and c64, every transpose combination, multi-block and strided
  operands). `Par::RayonWith(n)` still does not install a pool of `n` threads: `n`
  feeds the gate and sizes the tile grid, and the tasks run on the ambient rayon
  pool (see the documentation note below). (#16)
- Performance: the parallel GEMM gate moved from 64^3 total / 32^2 per thread to
  128^3 total / 80^3 per thread (`m*n*k`); below it `Par::Rayon` runs the sequential
  path. With integer division the gate opens at 128^3 on up to 4 threads, at 137^3
  on 5 threads and at 160^3 on 8 threads. The constants were tuned on one Apple M3
  (4 P + 4 E cores) with 4 and 8 threads; other thread counts are extrapolated.
  (#16)
- Performance: on aarch64, `gemm_small` (problems up to `m*n*k <= 32^3`) sends f32
  and f64 operands with `m, n >= 4` and column-major `A` and `B` to the NEON 4x4
  kernels, and uses the portable loops otherwise, including under `force-scalar`.
  Results can differ from 0.2.2 in the last bit because the NEON kernels use FMA.
  Only the parallel driver and this dispatch address #16; the SME2 path (Apple M4)
  is not part of this release. (#16)
- Documentation (no behaviour change):
  - `GeneralizedEvd` reduces to `B⁻¹·A` through an LU of `B` and needs a nonsingular
    `B`; it is not a QZ decomposition. Use `Qz` for a singular `B` or for generalized
    Schur forms. (#6)
  - `Par::RayonWith(n)` uses `n` in the parallel gate and, in GEMV and GEMM, to size
    the work partition (GEMM: the tile grid), but runs on the ambient rayon pool. Use
    `rayon::ThreadPool::install` or `RAYON_NUM_THREADS` to bound the thread count.
    The `oxiblas-core` helpers that dispatch through `run_in_pool` (for example
    `for_each_range`) do install an `n`-thread pool. (#16)
  - Corrected stale docs: the HEMM 3M threshold (64), the `GemmBlocking::for_kernel`
    values, the `gemm_with_par` auto-tuning condition, the `gemm_small` threshold,
    the `tbsv` example, the nrm2 algorithm wording and the CBLAS module docs. (#4, #9,
    #10, #14, #16)
  - Every README, `CONTRIBUTING.md`, `SECURITY.md`, `TODO.md` and the internal CI notes
    are brought in line with the code: API names, feature flags, tables of routines,
    code samples (self-contained and compiled, apart from two Criterion templates),
    sparse convergence figures, the `oxiblas-ffi` interface, and the fact that no CI
    workflow is enabled.

#### Behaviour changes in the second round

Each item is described in detail in the entries that follow or under Fixed.

- TRSM runs sequentially unless asked: `trsm`, `trsm_in_place`, `trsm_c32`,
  `trsm_c64` and their in-place forms stay on the calling thread, like `gemm`, also
  with the `parallel` feature, where 0.2.2 ran their GEMM updates on the Rayon pool
  from order 256. Pass a `Par` to `trsm_with_par`/`trsm_in_place_with_par` instead.
- `SymmetricEvd::compute`, `SymmetricEvdDc::compute` and `SvdDc::compute` stay on the
  calling thread with the `parallel` feature, as in 0.2.2; `compute_with_par` is the
  parallel form. `ParallelSvdDc` delegates to `SvdDc::compute_with_par`.
- Tolerances move in both directions. Dense LU and Cholesky (per-column rule) accept
  small-magnitude and badly scaled matrices they rejected and reject cancellation
  residues they accepted. The sparse factorizations and preconditioners (`eps·m·s`
  thresholds, exact-zero diagonal tests) accept scaled and large short-row matrices
  they rejected and reject numerically singular pivots of large-magnitude matrices
  that an absolute `eps` let through; `SparseLU` no longer drops small entries.
- `IC0` (on stencils with couplings, such as 9-point), `ICT` (now with fill-in),
  `ILU0::apply`, `ILUT`, `ILUTP` and `SparseLU` with row interchanges compute
  different (correct) factors, so preconditioned iteration counts and the solutions
  of the sparse eigensolvers that use `SparseLU` change.
- Iteration counts change for the GMRES family (no longer restarted after every
  step), IDR(s) (new shadow space; `iterations` counts every product with `A`),
  `ptfqmr` (right preconditioning) and the solvers that now confirm convergence
  with the true residual. `minres`/`pminres` return `Err(Breakdown)` where
  they returned `Ok` with an unmet true residual.
- For systems far from unit scale the solvers hand rescaled vectors to preconditioner
  closures, which must therefore be linear.
- `her_new`, `her2_new`, `syr_new` and `syr2_new` skip zero columns and store `+0` for
  `-0` products, as `her`/`syr` do.
- Results can differ from 0.2.2 in the last bit for: blocked, recursive and parallel
  LU and Cholesky; TRSM (reciprocals and blocked FMA order); SYRK and SYR2K with
  `beta != 0` across several k-blocks, or when a shape changes path; the aarch64 DOT,
  NRM2 and GEMV kernels and the unit-stride `cblas_?gemv` that now use them; CBLAS
  GEMM with transposed operands; `HermitianEvdDc` for n <= 100. Bitwise identical:
  the LU, `LuFullPiv`, `LuRook` and `Cholesky::compute` loop interchanges, and every
  `Par` against `Par::Seq` in the reworked drivers.
- Eigenvector signs and bases within repeated eigenspaces of `SymmetricEvd` and
  `SymmetricEvdDc` can differ from 0.2.2. On its divide-and-conquer path (more than 25
  columns) `SvdDc` returns exact-zero singular values as values of order `eps·‖A‖`,
  like `dgesdd`.
- The new DOT, NRM2 and GEMV kernels are aarch64 only. On x86_64 these routines keep
  their kernels and speed; only the AVX2 fixes listed under Fixed (strided views in
  GEMV, NRM2 with a tiny maximum) change their results.
- `force-scalar` on aarch64 now also keeps the generic `dot`, `nrm2` and `cnrm2`,
  `gemv`/`gemv_with_par`, the unit-stride `cblas_?gemv`, the TRSM/TRMM tile kernels
  and the GEMM B-panel packers on portable code. The typed `*_f64`/`*_f32` kernels
  and the x86_64 Level-1 and `gemv` kernels ignore the SIMD feature flags.
- Memory: each thread that ran a GEMM keeps up to 8 MiB of packing buffers between
  calls (Rayon workers included). Peak memory is about `4n²` elements for
  `SymmetricEvdDc`, `2n²` for `SymmetricEvd`, and 10-11 `n²` for `SvdDc` at
  n = 2000 (about 350 MB in f64).
- `CsrMatrix::new` and `CscMatrix::new` reject a first pointer other than 0, and
  `CscMatrix::new` duplicate rows; the `Display` text of `InvalidRowPtrs` and
  `InvalidColPtrs` changed. `CscMatrix::transpose` returns the transpose.
- `MatRef::ptr_at` and `MatMut::ptr_at(_mut)` accept any indices (wrapping
  arithmetic); only an in-range pointer may be dereferenced. `MatMut::split_cols` and
  `split_rows` panic where they returned overlapping mutable views.
- `QuadFloat` division by `±0` or `±inf` and `inf / x` follow IEEE 754 (they gave
  NaN), and `signum(±0)` is `±1`.
- `HermitianEvdDc` ignores the imaginary parts of the diagonal (as `ZHEEVD`) and
  rejects NaN/Inf with `NotConverged`; `SvdDc` and `ComplexSvdDc` reject NaN/Inf with
  `NotConverged` before computing anything; `SymmetricEvd`/`SymmetricEvdDc` fail at
  once with `NotConverged` on non-finite input.
- New errors: `PreconditionerError::InvalidMatrix` from the Chebyshev `Polynomial`
  preconditioner for invalid eigenvalue bounds, `PreconditionerError::MatrixConstruction`
  from `SAMG` instead of a zero operator or a panic, `SecularEquationFailed` from
  `SymmetricEvdDc`/`HermitianEvdDc` instead of an unconverged root, and
  `TensorError::DimensionMismatch` from `contract_2d` for short slices.

#### BLAS

- GEMM per-call overhead: packing buffers are kept per thread and reused (at most
  `PACK_CACHE_MAX_BYTES` = 8 MiB per thread stays cached), edge tiles go through an
  `MR × NR` scratch tile, the micro-kernel is resolved once per call, the packers are
  specialized for the compile-time `MR`/`NR`, and on aarch64 the f64 (`NR = 6`) and
  f32 (`NR = 8`) B panels are packed with NEON register transposes. One thread:
  DGEMM 64: 14.3 → 11.7 µs, 128: 93.9 → 84.2 µs; SGEMM 64: 8.20 → 5.06 µs, 128:
  50.3 → 39.7 µs; 1-12% faster from 256 to 1024; 0.4-10% faster on 4 and 8 threads
  from 512 to 2048.
- Transposed real operands are packed by the fast packers in place (up to 18% faster
  with a transposed `A` at n = 33-128), and small real GEMMs (`m*n*k <= 32³`) with a
  transposed `B`, a column-major `NoTrans` `A` and `m >= 8` use the axpy loop order.
- DOT on aarch64: `dot` for f64/f32 (and with it the unit-stride `cblas_ddot` and
  `cblas_sdot`), the chunks of `dot_par`, and `dot_f64`/`dot_f32` run a NEON kernel
  with 8 FMA accumulators, a pairwise tree and a two-level sum (flushed every 1024
  f64 or 2048 f32 elements). f64: 1K 268 → 93 ns, 1M 277 → 151 µs; f32 1K
  204 → 48 ns.
- NRM2 on aarch64: `nrm2` (f64/f32, from 8 elements), `nrm2_f64`/`nrm2_f32`,
  `cnrm2`, `scnrm2`/`dznrm2`, the chunks of `nrm2_par` and the unit-stride
  `cblas_dnrm2`/`cblas_snrm2` use a one-pass NEON port of the LAPACK 3.12 `dnrm2`
  (Blue's constants, three accumulators, no division per element) instead of two
  passes with scaling. f64: 1K 2.85 µs → 126 ns, 1M 2.92 ms → 117 µs (increasing
  input; 562 → 125 ns and 539 → 114 µs on random data). NaN, Inf, zero and overflow
  behaviour is unchanged, results are within 6 ulps of an exact `dnrm2.f90` port, and
  vectors shorter than 8 keep the `dlassq` path.
- GEMV on aarch64: `gemv` and `gemv_with_par` on column-major f64/f32 views run NEON
  kernels at every size, an axpy form over 8 columns per pass for `NoTrans` (the
  reference `DGEMV` order) and a 4-column, 16-chain dot form for `Trans`/`ConjTrans`.
  f64 `NoTrans` 100²: 1.81 → 0.55 µs, 1000²: 257 → 87 µs, 10000²: 39.2 → 13.7 ms;
  f64 `Trans` 1000²: 395 → 91 µs; f32 `NoTrans` 1000²: 877 → 43 µs. With
  `Par::Rayon`, threads own disjoint row (`NoTrans`) or column (`Trans`) ranges of
  `y`, and the result equals the sequential one bit for bit (f64 `Trans` 5000² on 4
  threads: 19.3 → 2.31 ms). `dot_par` and `nrm2_par` use the new kernels per chunk
  (4 threads, 1M: `dot_par` 96.9 → 58.4 µs, `nrm2_par` 802 → 50 µs).
- TRSM is rewritten for every side, triangle, transpose, diagonal and element type:
  diagonal blocks of order 128 are packed once with the reciprocals of their
  diagonal and solved by register-tiled kernels (NEON for f32/f64 on aarch64), and
  each off-diagonal block is one in-place GEMM on views of `A` and `B`, with no
  temporaries. `trsm_c32`/`trsm_c64` use the same driver with 3M GEMM updates. It
  runs sequentially (see above; 0.2.2 parallelized itself with the `parallel`
  feature). One thread, `trsm_in_place` Left/Lower/NoTrans, f64: 64: 49.1 → 9.74 µs,
  128: 228.8 → 57.2 µs, 256: 1127 → 402 µs, 512: 6.55 → 2.98 ms (OpenBLAS in the same
  run: 19.9 µs, 97.8 µs, 537 µs, 3.30 ms); n = 1024: 25.6 → 49.1 GFLOPS. A
  non-finite reciprocal falls back to exact division. Results can differ in the last
  bit, and for complex input the same entries are non-finite but NaN and Inf can trade
  places.
- TRMM is triangle-only and in place: diagonal blocks are packed (the other triangle,
  and the diagonal for `TrmmDiag::Unit`, are never read) and multiplied by the same
  tile kernels, and off-diagonal blocks are GEMMs on views; the dense expansion of
  the triangle and the full GEMM are gone. n = 1024, one thread: 26.0 → 49.3 GFLOPS.
- SYRK, SYR2K and GEMMT run on a triangle-only blocked kernel (`level3/tri_gemm.rs`)
  that reuses the GEMM packers (transposed operands packed in place), packing buffers
  and micro-kernel, and computes only the micro-tiles that meet the referenced
  triangle; diagonal and edge tiles go through a scratch tile, and the other triangle
  is never read or written. The transposed copies, the `n × n` temporary, the full
  GEMM (twice the flops) and the triangle copy are gone. SYR2K is one pass with
  `k' = 2k` (`alpha·[A B]·[B A]ᵀ`). `beta` is applied once, and `beta == 0` is folded
  into the first k-block, which stores without reading `C`. DSYRK (n = k, Lower,
  NoTrans), one thread: 64: 15.4 → 7.19 µs, 128: 94.6 → 48.4 µs, 256: 676 → 340 µs,
  512: 5.21 → 2.57 ms (OpenBLAS in the same run: 7.13 µs, 48.6 µs, 361 µs, 2.63 ms);
  `syrk_syr2k_bench` at n = 1024: SYRK 26.4 → 53.8, SYR2K 26.3 → 52.8 GFLOPS. f32,
  GEMMT and the complex `Trans` forms are about twice as fast as well.
- The switch between the naive loops and the blocked kernel of SYRK, SYR2K and GEMMT
  (`n >= 32 && k >= 8` before) is re-measured per routine, transpose, type and
  `beta == 0`. Real short-k problems now use the kernel (DSYRK n = 512 with
  k = 1..4: 1.8-9× faster); complex `NoTrans` SYRK and SYR2K always use the loops,
  which beat the scalar 2×2 complex micro-kernel.

#### CBLAS

- `cblas_dgemm` and `cblas_sgemm` with a transposed operand, more than `32³`
  multiply-adds and `n > 1` run on the blocked, transpose-aware GEMM, which reads the
  transposed operand in place; before, only `NoTrans`/`NoTrans` did. 2.4-3.7× faster
  for `NoTrans`/`Trans` and 4.8-39× for a transposed `A` at n = 33-512. The complex
  `cblas_{c,z}gemm` keep their path (routing them was measured slower).
- On aarch64, `cblas_dgemv` and `cblas_sgemv` with `incx == incy == 1` call
  `level2::gemv` after the unchanged argument checks (1000² column-major `NoTrans`:
  658 → 81 µs for f64, 650 → 41 µs for f32). The unit-stride driver is kept out of
  line (`#[inline(never)]`); the strided column-major `Trans` case measures 4-6%
  slower than before the fast path, which was traced to code placement (the loops are
  byte-identical, and 64-byte-aligned builds agree within 0.7%) and accepted.
- `cblas_strsm` and `cblas_dtrsm` skip the scaling pass when `alpha == 1`.

#### LAPACK

- LU with partial pivoting: the elimination loops of `Lu::compute` and of the panels
  run column by column (bit-identical results; the unblocked factorization went from
  0.37-0.63 to 9.1-10.7 GFLOPS). `compute_blocked`, `compute_with_block_size`,
  `compute_recursive` and `compute_blocked_par*` work in place on the factor: panels
  wider than 16 columns are factored recursively (LAPACK `?getrf2`), the block row of
  `U` is one TRSM and the trailing update one GEMM (`alpha = -1`, `beta = 1`) on
  views, and row interchanges are applied column by column (`?laswp`), to the left
  columns once per panel. `compute_recursive` now runs the recursive algorithm its
  documentation describes (it was a right-looking blocked LU with panels of at most
  64). The parallel drivers solve their TRSM panels in parallel and return bitwise the
  factors of `compute_with_block_size` with the same block size. One thread, f64,
  `compute_blocked` on random matrices: 12.5/17.2/23.4/23.0 → 18.8/27.3/31.9/36.2
  GFLOPS at n = 256/512/768/1024 (from a baseline that already had the faster loops
  and TRSM); `compute_recursive` 11.9/16.5/23.7 → 18.8/27.4/36.1 at n = 256/512/1024;
  4 threads, `compute_blocked_par`: 19.9/31.1/44.0 → 36.4/64.4/77.4 at
  n = 512/1024/2048 (f32 at 2048: 71.7 → 140.8). Results can differ from 0.2.2 in the
  last bit; the pivots of the test matrices are unchanged.
- `LuFullPiv` and `LuRook` eliminate column by column (bit-identical; the full-pivot
  search runs down the columns with a row-major tie rule). One thread, n = 256/512/768:
  `LuFullPiv` 0.46/0.35/0.47 → 2.72/2.93/3.07 GFLOPS, `LuRook` 0.59/0.43/0.61 →
  6.92/8.11/9.08.
- Cholesky: `compute_blocked`, `compute_with_block_size`, `compute_recursive` and
  `compute_blocked_par*` work in place: TRSM on views, then GEMM on the off-diagonal
  blocks of the lower triangle and the triangle-only SYRK on diagonal blocks of order
  at most 64, about half the multiply-adds of the old full `L21·L21ᵀ` temporary; the
  copy of `A22` and the full SYRK of the recursive driver are gone. With a parallel
  `Par` the disjoint update blocks run concurrently, and `compute_blocked_par` equals
  `compute_blocked` bitwise. `Cholesky::compute` and the diagonal blocks run column
  by column (bit-identical). One thread, `compute_blocked`: 17.1/21.1/22.6/20.8 →
  26.2/33.7/38.2/38.6 GFLOPS at n = 256/512/768/1024, then 6-11% more from the SYRK
  diagonal blocks; `compute`: 2.2/1.8/2.0/1.6 → 11.6/11.0/11.3/11.4. 4 threads,
  `compute_blocked_par`: 34.0/39.8/52.7 → 54.7/94.9/128.8 at n = 512/1024/2048, then
  3-7% more.
- `SymmetricEvdDc` follows LAPACK `DSYEVD`: a blocked tridiagonal reduction of the
  lower triangle (`DSYTRD`/`DLATRD` panels and a GEMM trailing update; the reflectors
  are kept and `Q` is never formed), `DLAED0`-`DLAED4`-style divide and conquer
  (merges gather the deflated and grouped columns into a workspace allocated once per
  call, and form eigenvectors with two GEMMs over the nonzero blocks of
  `diag(Q1, Q2)`), and a compact-WY back-transformation (`DORMTR`). `SymmetricEvd`
  shares the reduction and back-transformation; its implicit QR records the rotations
  of each sweep and applies them to L1-sized row blocks as a wavefront of up to 32
  sweeps, skipping zero rows, then forms `V = Q·Z` by GEMM. One thread, random f64
  (best round of the A/B on a busy machine): `SymmetricEvdDc` 0.0217/0.730/2.86/29.6 s →
  0.0024/0.0225/0.138/0.977 s and `SymmetricEvd` 0.0109/0.521/1.41/14.1 s →
  0.0035/0.0394/0.271/2.04 s at n = 200/500/1000/2000, so `SymmetricEvdDc` is now the
  faster of the two at every size; the robust secular solver (see Fixed) made it
  another 4-6% faster at n = 500-1000.
- `SvdDc` follows LAPACK `dgesdd`: power-of-two scaling, a blocked bidiagonalization
  (`dgebrd`/`dlabrd` panels with GEMM trailing updates), the bidiagonal divide and
  conquer of `dbdsdc` (see Fixed), and `U`, `V` formed by applying the reflectors as
  compact-WY blocks (`dlarft`/`dlarfb`) instead of from the identity. f32 and f64 run
  on the `oxiblas-blas` GEMM; other real types run the same algorithm with portable
  loops. One thread, random matrices: 0.033/1.19/4.56/49.6 s → 0.0040/0.041/0.27/2.3 s
  at n = 200/500/1000/2000 (separate runs on a busy machine), 3-8% of the time of the
  one-sided Jacobi `Svd`, which 0.2.2's `SvdDc` was slower than at n = 500.
- `HermitianEvdDc` follows LAPACK `ZHEEVD`: reduction to a real symmetric
  tridiagonal with a unitary diagonal phase scaling, the divide and conquer of
  `SymmetricEvdDc` in the real type, and a GEMM back-transformation (complex times
  real as in `ZLACRM`). Inputs outside about `[1e-146, 1e146]` are scaled by a power
  of two. Eigenvalues agree with `HermitianEvd` to within `0.5·n·eps·‖A‖`, and it
  is 2.3× faster than `HermitianEvd` at n = 200 (11.6 against 26.8 ms) and n = 500
  (176 against 400 ms). For n <= 100 results can change in the last bit (stedc
  instead of complex Givens QR). Complex types other than `Complex32`/`Complex64`
  use the same reduction followed by the implicit QR iteration.
- `Lu::solve` no longer makes permutation passes that had no effect (results are
  unchanged).
- Eigenvalues of `Schur`, `GeneralEvd`, `ComplexSchur` and `ComplexGeneralEvd` are
  listed in the order of the Schur factor's diagonal (a real 2x2 matrix no longer
  returns `[λmax, λmin]`), and the real Schur factor is standardized, so `T` and `Q`
  can differ from 0.2.2 (for a complex 2x2 input they are no longer `T = A`,
  `Q = I`). `GeneralEvd` and `ComplexGeneralEvd` eigenvectors have unit norm, and
  the largest component of a complex eigenvector is real and positive, as in
  `DGEEV`/`ZGEEV`; column `k` of `eigenvectors_real()` plus `i·eigenvectors_imag()`
  belongs to `eigenvalues()[k]` (now documented). `compute_full` runs one Schur
  decomposition instead of two, and `ComplexGeneralEvd::eigenvalues_only` now also
  accumulates `Q` (about twice the work of the old values-only loop). A NaN or
  infinite 2x2 input gives NaN eigenvalues.

#### Sparse

- IDR(s) builds its shadow space from a deterministic SplitMix64 sequence, replaces
  the recursive residual by the true one when they drift apart, clamps `s` to
  `[1, n]`, counts every product with `A` in `iterations`, and ends
  `residual_history` with the reported (true) residual.
- `minres` and `pminres` return `Err(Breakdown)` on a Lanczos breakdown whose true
  residual misses the tolerance; they returned `Ok` with `converged` set.

#### Matrix

- The `mmap` module and each of its safe constructors document the conditions they
  cannot check: no other writer and no truncation of the mapped file while a mapping
  is alive, and an element type from the supported list. "Process sharing" is
  narrowed to read-only mappings. Making this explicit in the API is planned for
  0.3.0 (see TODO.md).

#### Code quality

- `#![warn(clippy::undocumented_unsafe_blocks)]` is enabled in `oxiblas-blas`,
  `oxiblas-matrix`, `oxiblas-lapack`, `oxiblas-sparse` and `oxiblas-ndarray`, and
  every `unsafe` block there has a `// SAFETY:` comment. `oxiblas-core` does not
  enable the lint yet (434 blocks would be flagged).
- The remaining 41 `unwrap()`, `expect()` and `unwrap_or_else(|| panic!(..))` calls
  are gone from the library code of `oxiblas-core`, `oxiblas-matrix`,
  `oxiblas-lapack` and `oxiblas-sparse`; most code is restructured so it cannot fail
  (for example `CowMat` copy-on-write through `Arc::make_mut`), and panics that are
  part of an API contract are explicit, documented `panic!`s with unchanged messages.
  `Mat`, `BandedMat` and `SymmetricBandedMat` constructors, the banded setters and
  `Arena::alloc_zeroed`/`alloc_vec` gain `# Panics` sections. Output for valid input
  is unchanged; `GeneralEvd::verify` returns 0 when only one eigenvector part is
  present, a state no constructor produces.
- Parallel Strassen and `herk`/`her2k` contain no `unreachable!()`; results are
  bit-identical.
- `oxiblas-core` failed `clippy -D warnings` when built with `max-simd-128` or
  `max-simd-256` alone: `detect_simd_level` ended in a needless `return`, and the
  complex SIMD tests left an import unused. `detect_simd_level` now applies the cap in
  one expression (same results), and the test checks the capped level and backend.

#### Tests and benchmarks

- `test_complex_simd_path_is_taken` assumed that every x86_64 host has AVX2 and failed
  on SSE-only CPUs, for example under Rosetta 2. It now compares the selected SIMD
  level and complex backend with runtime feature detection, so it checks the scalar
  fallback on SSE-only CPUs and the SIMD kernel on AVX2/FMA CPUs. The workspace tests
  pass under Rosetta 2 (`--target x86_64-apple-darwin`, SSE4.2 only), where the AVX2
  and AVX-512 paths are compiled but not executed.

- The large `SvdDc` accuracy tests use prescribed singular values in debug builds;
  the random-matrix comparisons against the Jacobi `Svd` are release-only (4 tests,
  skipped in debug; run them with `--release` or `--run-ignored all`), which made the
  debug `oxiblas-lapack` suite several times faster. The `lapack_compat` fixtures for
  `SvdDc` with NaN or Inf input assert `Err(NotConverged)`.
- `lapack_svd`: group `svd_standard` is renamed `svd_jacobi` (comparison label
  `jacobi_svd`), the inputs are full-rank fixed-seed random matrices, and the
  divide-and-conquer group adds n = 500. `lapack_evd`: `symmetric_evd_large` uses 10
  samples and adds n = 1000, and the new groups `symmetric_evd_dc_large`
  (300/500/1000), `symmetric_evd_random` and `symmetric_evd_dc_random`
  (100/200/500/1000) are added.

#### Documentation

- Corrected API docs: `MatRef` strides (`row_stride` is the leading dimension only
  when `col_stride == 1`), the `cblas_?dot`/`?nrm2` paths, the `axpy` module, the
  `gemv` `KC_BLOCK` (portable paths only), `bidiag_reduce` "blocked" (rank-1 chunks),
  `utils::det` ("numerically zero"), `CACHE_LINE_SIZE`/`DEFAULT_ALIGN` (every
  aarch64 target), `GemmKernel::micro_kernel`'s `c_stride` (the leading dimension),
  `PackingConfig::prefetch_distance` (unused), the GEMM prefetch notes, the SYR2K
  sweep, the TRMM operation count, the SYMM threshold (the order of `A`), the LU and
  Cholesky module docs (`compute()` is always unblocked), the
  `compute_blocked_par` and `compute_recursive` docs, the element types of
  `Lu::compute_auto`, the `SymmetricEvdDc` errors, the `new_unchecked` contracts of
  the sparse formats, and ambiguous or broken intra-doc links. The docs of private items
  build without warnings as well (`cargo doc --document-private-items`).
- The README performance figures are re-measured on 2026-11-09, after the second
  round, on an Apple M3 against OpenBLAS 0.3.30 (best of 5 runs on one thread, 3 on 4
  threads and 5 on 8). On one thread, f64 GEMM is at 92-99% of OpenBLAS and f32 GEMM
  at 101-120% (before the second round: 73-96% and 74-98%); DOT is at 109-137% (was
  48-59%), NRM2 14-16× faster (was 63-64%), GEMV 97-222% (was 36-70%), TRSM 117-200%
  (was 41-60%) and SYRK 99-108% (was 39-51%). Blocked LU reaches 37.9 GFLOPS and
  blocked Cholesky 40.9 at n = 1024 (was 3.8 and 18.1). The `comparison` bench's
  doc comment now describes the ratio the READMEs use (speed, not time) and the thread
  count of the parallel groups (`RAYON_NUM_THREADS` when set).

## [0.2.2] - 2026-08-06

A production-readiness hardening release. A systematic multi-agent audit swept the
entire workspace (`oxiblas-core`, `oxiblas-matrix`, `oxiblas-blas`, `oxiblas-lapack`,
`oxiblas-ndarray`, `oxiblas-sparse`), followed by adversarial re-verification of every
proposed fix, resulting in a large batch of correctness, honesty, and documentation
fixes — plus a follow-on hardening pass over that audit closure: soundness fixes for
the sibling call sites the original audit's fixes missed, plus workspace hygiene
(lint/dependency/CI policy, doctest and file-size compliance, fuzz coverage).
No public API breaking changes; a handful of previously-`unsafe fn`-eligible
constructors are now correctly marked `unsafe fn` (see Changed).

### Fixed
- Numerous numerical correctness bugs across BLAS, LAPACK, and sparse routines,
  including Hermitian/symmetric diagonal handling, SVD and eigensolver convergence
  and deflation, sparse factorization fill-in, incremental SVD, and IRAM restart
  logic
- Several fabricated or stub code paths replaced with real, verified algorithms
  (e.g. a dead-code MRRR eigensolver path, an untested symmetric divide-and-conquer
  EVD merge, and fake complex-routine aliases)
- Panics on edge-case inputs (empty, singular, or near-singular matrices) converted
  to proper `Result`-based error handling instead of `unwrap()`/`expect()`/`panic!()`
- Assorted silent-fallback and misleading-documentation issues that could mask
  incorrect results
- `PackedRef::new`/`PackedMut::new` (packed.rs) and `BandedRef::new`/`BandedMut::new`
  (banded.rs) were still safe `pub fn` over raw pointers with no validation, even
  though the sibling `MatRef::new`/`MatMut::new` had already been hardened to
  `unsafe fn` — safe code could construct an out-of-bounds view and read/write past
  the backing allocation. All four are now `unsafe fn` with `# Safety` docs.
- `wasm32.rs` SIMD `extract()`/`insert()` still reached `unreachable_unchecked()` on
  an out-of-range lane index from safe caller code in release builds, after the
  identical bug had been fixed for x86_64/aarch64 — ported the
  `lane_index_out_of_range` panic helper to all six wasm32 call sites.
- `MmapMatMut::create`'s size computation (`row_stride * ncols * size_of::<T>()`)
  used unchecked arithmetic, so an oversized `(nrows, ncols)` could wrap to a small
  file while the returned struct still recorded the original huge dimensions,
  letting ordinary safe accessors write past the mapping. Now uses checked
  arithmetic and re-validates the layout against the mapped file size.
- The four CBLAS modules added to close the "CBLAS surface is incomplete" audit
  finding (`hermitian.rs`, `triangular_symmetric.rs`, `complex_level2.rs`,
  `complex_level1.rs`) shipped with zero `lda`/`ldb`/`ldc`/null-pointer validation,
  reproducing the exact bug class `cblas/basic.rs` had already been hardened
  against. Added a shared `cblas/validate.rs` module and wired
  `gemv_params_valid`/`gemm_params_valid`/-family checks plus null checks into
  every `extern "C"` entry point in the four files, no-oping (matching the
  `xerbla`-then-return convention) on invalid input instead of indexing OOB.
- `CscMatrix`'s `Index` impl still panicked via `expect()` on a structurally-zero
  element after the identical `CsrMatrix` impl had been removed for the same
  reason; removed to match.
- `packed_len()` (`n*(n+1)/2`) and the banded `ldab*ncols` allocation-size
  expressions were still plain, wrapping-in-release arithmetic after `mat.rs`'s
  allocations had been routed through a checked `checked_dim_mul` helper; both now
  use checked multiplication and panic on overflow instead of silently
  under-allocating.
- Matrix Market reader (`mtx.rs`) pre-allocated `Vec::with_capacity(header.nnz)`
  directly from the untrusted size line, so a ~30-byte crafted file
  (`3 3 18446744073709551615`) could trigger a capacity-overflow abort before any
  data line was read. `read_header` now rejects `nnz > nrows*ncols` (coordinate
  format, checked with `saturating_mul`) and uses checked multiplication for the
  array-format `nrows*ncols` cell count, so malformed headers return a
  `MtxError` instead of aborting the process.
- 57 of 64 public-API doctests marked `` ```ignore `` (never compiled or verified)
  across `oxiblas-core`, `oxiblas-matrix`, `oxiblas-lapack`, `oxiblas-ndarray`, and
  `oxiblas-sparse` are now real, executed doctests with concrete inputs and
  assertions on the results (the audit that closed this finding had only fixed 7 of
  the 64).
- `oxiblas-core`'s no_std test-code gating was incomplete for the x86_64 SIMD
  modules: `simd/x86_64/functions.rs` (a `thread_local!` override plus 16
  `is_x86_feature_detected!` call sites and a `println!`) and 4 direct-register
  test functions in `simd/complex.rs` compiled only under `std`, but were not
  `#[cfg(feature = "std")]`-gated — invisible to `--no-default-features` checks
  run on an aarch64 host, since that module tree is itself
  `#[cfg(target_arch = "x86_64")]`-gated and was never actually cross-compiled.
  Confirmed and fixed by adding `--target x86_64-apple-darwin` to the no_std
  verification command (see `TODO.md`); see `TODO.md` for the full file list.
- Two pre-existing test/feature-interaction bugs surfaced by the above
  verification pass (not caused by it): `simd::dispatch::tests::test_aarch64_neon_always_present`
  and `simd::multiver::tests::test_aarch64_neon_always_true` both asserted NEON
  is unconditionally present on AArch64 without accounting for `dispatch.rs`'s
  own `limited_to()` masking, which correctly reports `has_neon = false` under
  the `force-scalar` feature (always active under `--all-features`). Fixed by
  checking `SimdCapabilities::simd_ceiling_bytes() >= 16` before asserting,
  mirroring the already-correct sibling test `test_compute_respects_feature_ceiling`.

### Changed
- Dependencies updated to latest compatible versions
- Documentation corrected in numerous places to accurately describe actual behavior
- `PackedRef::new`, `PackedMut::new`, `BandedRef::new`, `BandedMut::new` are now
  `unsafe fn` (see Fixed above) — existing safe call sites inside this workspace
  were updated to `unsafe { ... }` blocks with `// SAFETY:` justifications;
  downstream callers constructing these types directly from raw pointers will need
  to do the same.
- Hoisted the last four per-crate dependency version pins
  (`oxiblas-ndarray`'s `faer`, `oxiblas-benchmarks`'s `ndarray-linalg`/`blas-src`/
  `cblas-sys`) into `[workspace.dependencies]` so every version lives in one place.
- `oxiblas-blas`'s crate-wide `#![allow(clippy::missing_safety_doc)]`,
  `clippy::ptr_as_ptr`, `clippy::transmute_ptr_to_ref`, `clippy::cast_possible_wrap`,
  `clippy::transmute_undefined_repr`, and `clippy::missing_transmute_annotations`
  lint suppressions were removed; the 74 CBLAS entry points that were missing a
  `# Safety` doc section now have one, and 8 NEON transmute call sites in
  `level1/nrm2.rs` now carry explicit `transmute::<From, To>()` turbofish
  annotations instead of relying on suppressed lints.
- Split 4 source files that had grown past the workspace's 2000-line limit into
  cohesive modules with the public API re-exported from the original path:
  `oxiblas-blas/src/level2/gemv.rs`, `oxiblas-blas/src/cblas/basic.rs`,
  `oxiblas-sparse/src/linalg/eigenvalue/tests.rs`, and
  `oxiblas-sparse/src/linalg/eigenvalue/special.rs`.

### Added
- `[package.metadata.docs.rs]` with `all-features = true` on every publishable
  crate so feature-gated APIs are visible on docs.rs
- `rustfmt.toml`, `clippy.toml` (MSRV pinned to the workspace's `rust-version`),
  `deny.toml` (COOLJAPAN dependency-ban list, `cargo deny check bans` clean),
  `SECURITY.md`, and `CONTRIBUTING.md` at the workspace root.
- `fuzz/` (an independent `cargo-fuzz` workspace, excluded from the main
  workspace so `cargo build`/`clippy --workspace` never need a nightly
  toolchain) with two real libFuzzer targets: `mtx_matrix_market` (the Matrix
  Market parser) and `mmap_header` (the `.oxiblas` memory-mapped-matrix header
  parser) — the two untrusted-input parsers in the workspace.

## [0.2.1] - 2026-03-16

### Fixed
- Fixed compilation error in sparse eigenvalue test imports
- Fixed unnecessary type casts in BLAS level 3 autotune module

### Changed
- Updated dependencies to latest versions

## [0.2.0] - 2026-03-06

### Added

#### LAPACK

- Recursive cache-oblivious factorizations: `Cholesky::compute_recursive()`, `Lu::compute_recursive()`, `Qr::compute_recursive()` - divide-and-conquer algorithms with automatic cache adaptation
- Parallel blocked factorizations: `Cholesky::compute_blocked_par()`, `Lu::compute_blocked_par()` (requires "parallel" feature) - multi-threaded Level 3 BLAS updates via rayon
- Complex bidiagonal reduction: `ComplexBidiagFactors` supporting `Complex64` and `Complex32` matrices
- Mixed-precision iterative refinement variants: `mixed_precision_solve` (LU), `mixed_precision_solve_cholesky`, `mixed_precision_solve_symmetric`, and `mixed_precision_solve_qr` - f32 factorization with f64 residual computation
- LAPACK integration test suite: 61 tests covering LU, Cholesky, QR, SVD, EVD, and solve operations in `tests/lapack_compat.rs`

#### BLAS

- Batched BLAS operations: `gemm_batched`, `gemm_strided_batched`, `axpy_batched`, `gemv_batched` with parallel variants (`gemm_batched_par`, etc.)
- Runtime auto-tuning infrastructure: `RuntimeAutoTuner` for dynamic block size selection, `gemm_auto_tuned()` convenience function (requires "runtime-tuning" feature)

#### Sparse

- Multifrontal sparse factorizations: `MultifrontalCholesky` and `MultifrontalLU` with elimination tree construction and supernodal aggregation
- Advanced sparse LU pivoting strategies: threshold pivoting (`SparseLuThreshold`), static pivoting (`SparseLuStaticPivot` - SuperLU-style), and Bunch-Kaufman LDL^T factorization (`SparseLdlt`)
- Standard test matrix generators: `laplacian_2d`, `laplacian_3d`, `tridiagonal`, `diagonal`, `arrow_matrix`, `random_spd`, `poisson_1d`
- Memory usage integration tests: 27 tests verifying sparse operation memory behavior

#### Core

- Runtime SIMD dispatch infrastructure: `SimdCapabilities` (runtime CPU feature detection), `SimdDispatcher`, `KernelSelector`, and `simd_dispatch!` macro for function multi-versioning
- no_std support for `oxiblas-core` and `oxiblas-matrix`: add `default-features = false` to use in embedded or no-std environments (requires `alloc`)
- Feature constants module in the main `oxiblas` crate: `oxiblas::features::{HAS_PARALLEL, HAS_SPARSE, HAS_F16, HAS_F128, HAS_RUNTIME_TUNING, ...}` for compile-time feature introspection

#### ndarray Integration

- Parallel GEMM: `matmul_par` for ndarray `Array2` (requires "parallel" feature)
- Sparse integration functions: `array2_to_csr`, `csr_to_array2`, `spmv_ndarray`, `sparse_solve_ndarray`

#### Performance Regression Framework

- Performance regression framework: `PerfBaseline`, `PerfMeasurement`, `RegressionChecker` types with JSON storage and configurable degradation threshold
- Performance regression CLI binary (`regress`): subcommands `capture`, `check`, `report`, `list` for CI-integrated throughput tracking
  - `regress capture [--output baseline.json]` — run all quick benchmarks (GEMM f64/f32, Cholesky) and save JSON baseline
  - `regress check [--baseline baseline.json] [--threshold 5.0]` — compare current performance vs baseline, PASS/FAIL per operation
  - `regress report [--baseline baseline.json]` — formatted table of baseline measurements
  - `regress list` — print all available benchmark names

#### BLAS

- SSE4.2 intermediate GEMM micro-kernels: `F64x2Sse` (F64×2, 4×4 tiles) and `F32x4Sse` (F32×4, 4×4 tiles) filling the gap between scalar and AVX2 on x86_64 CPUs without AVX2

#### Core

- NUMA-aware memory allocation: `NumaVec<T>`, `MatNuma<T>`, Linux real NUMA topology detection, NUMA-local allocation fallback
- Thread pool customization: `set_global_thread_pool`, `OxiblasThreadConfig`, `with_thread_count` for fine-grained parallel execution control

#### Documentation and Benchmarks

- Algorithm Selection Guide added to README: when to use blocked vs. recursive vs. parallel variants
- Performance comparison tables in README: LAPACK factorization speedups (Cholesky 6-10×, LU 14-23×, QR 3-7×)
- Library comparison table in README: OxiBLAS vs. ndarray-linalg vs. nalgebra vs. faer across key criteria
- Benchmark size variation suites: tiny/non-power-of-2/rectangular/large matrix benchmarks
- Precision benchmarks: f16/f32/f64/f128 throughput comparisons
- Cross-platform performance comparison: tested platforms, SIMD feature detection summary

### Fixed

- Blocked QR factorization (WY representation): corrected T matrix construction (was using T, must use T^T per DLARFT specification) and corrected block reflector application; 3-7× speedup for large matrices now fully realized

### Changed

- Refactored `oxiblas-core/src/scalar.rs` (2,846 lines) into 8 focused module files under `scalar/` directory; all files remain under the 2,000-line policy limit
- Retired `oxiblas-ffi` crate from workspace (Pure Rust ecosystem policy); crate directory preserved as deprecated archive
- Project statistics updated: ~169,900 lines of Rust across 359 files, 2,835 passing tests + 195 doctests

### Removed

- `oxiblas-ffi` removed from workspace members (`Cargo.toml` members list); the crate directory is retained as a deprecated archive but is no longer built or published

### Code Quality

- Zero `unwrap()` calls in all production code across the entire workspace
- All source files are under 2,000 lines (100% compliant with refactoring policy)
- 16 previously oversized files (51,890 lines total) refactored into 113 modules; v0.2.0 adds the `scalar.rs` refactoring for a total of 114 modules

## [0.1.2] - 2025-12-30

### Added

- **Complex Number Support in ndarray Integration**: Added comprehensive complex-specific LAPACK functions to `oxiblas-ndarray`:
  - `svd_complex_ndarray`: SVD decomposition for complex matrices using one-sided Jacobi algorithm
  - `qr_complex_ndarray`: QR decomposition for complex matrices (unitary Q)
  - `cholesky_hermitian_ndarray`: Cholesky decomposition for Hermitian positive definite matrices
  - `eig_hermitian_ndarray`: Eigenvalue decomposition for Hermitian matrices (real eigenvalues, complex eigenvectors)
  - `ComplexSvdResult` and `HermitianEvdResult` types for complex decomposition results

### Changed

- **Relaxed Trait Bounds**: Removed unnecessary `Real` trait bound from `solve` and `solve_multiple` functions in both `oxiblas-lapack` and `oxiblas-ndarray`, enabling proper support for complex number types (`Complex<f32>`, `Complex<f64>`)

### Fixed

- Minor code formatting improvements in symmetric eigenvalue decomposition tests

## [0.1.1] - 2025-12-29

### Fixed

- **Symmetric Eigenvalue Decomposition**: Fixed critical bug in QR algorithm for tridiagonal matrices where off-diagonal elements were incorrectly stored using `hypot(x, z)` (always positive) instead of `c * x - s * z` (preserves sign). This caused eigenvectors to be computed incorrectly for matrices requiring multiple QR iterations, while eigenvalues remained correct. The fix ensures proper accumulation of Givens rotations into the eigenvector matrix.

## [0.1.0] - 2025-12-27

### Initial Release

OxiBLAS 0.1.0 is the first public release of a pure Rust BLAS/LAPACK implementation.

#### Features

**Core Library (`oxiblas-core`)**
- SIMD layer with AVX2/FMA, AVX-512, and ARM NEON support
- Portable scalar fallback for all platforms
- Extended precision support: f16, f32, f64, f128 (quad precision)
- Complex number support: Complex32, Complex64
- Parallel execution with rayon (optional `parallel` feature)

**BLAS Operations (`oxiblas-blas`)**
- **Level 1** (11 operations): dot, axpy, nrm2, scal, iamax, swap, copy, rot, rotg, rotm, rotmg, asum
- **Level 2** (15 operations): gemv, ger, syr, spr, symv, sbmv, tbmv, tbsv, tpmv, tpsv, spmv, her, hpr, hbmv, hpmv
- **Level 3** (11 operations): gemm, syrk, trsm, syr2k, symm, hemm, herk, her2k
- Complete packed and banded matrix support
- Hermitian and symmetric operations for complex/real types

**LAPACK Operations (`oxiblas-lapack`)**
- **Factorizations**: LU (getrf), Cholesky (potrf), QR (geqrf, geqr2)
- **Decompositions**: SVD (gesvd), Eigenvalue (syev, geev), Schur (gees), Hessenberg (gehrd)
- **Linear Solvers**: Triangular (trtrs), General (gesv, getrs), Tridiagonal (gtsv, ptsv), Least squares (gels)
- **Advanced**: Condition number estimation, matrix inversion, balancing

**Sparse Linear Algebra (`oxiblas-sparse`)**
- **9 sparse formats**: CSR, CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL-C-σ
- **Basic operations**: SpMV (generic + SIMD), SpMM, sparse addition, triangular solve
- **Iterative solvers** (10 variants):
  - CG, PCG (preconditioned conjugate gradient)
  - BiCGStab (stabilized bi-conjugate gradient)
  - GMRES, MINRES (minimal residual methods)
  - IDR(s), TFQMR, QMR (quasi-minimal residual variants)
  - Block-CG, Block-GMRES (multiple right-hand sides)
- **Eigenvalue solvers**: Lanczos, Arnoldi, IRAM (Implicitly Restarted Arnoldi)
- **SVD solvers**: Truncated SVD, Randomized SVD
- **Factorizations**: Sparse LU, Sparse QR, Sparse Cholesky
- **Preconditioners**: Jacobi, Gauss-Seidel, SOR, ILU0, ILUT, ILUTP, IC0, ICT, AMG, SPAI, AINV, Schwarz
- **Ordering algorithms**: RCM, AMD, Nested Dissection

**Matrix Utilities (`oxiblas-matrix`)**
- Flexible matrix types with multiple layouts (row-major, column-major)
- View and mutable view support
- Efficient memory management and slicing

**Extended Features**
- **Tensor operations**: Einstein summation (24 patterns), batched operations
- **Advanced summation**: Kahan (compensated), pairwise, superaccurate
- ~~**C FFI**: Drop-in replacement for C BLAS/LAPACK (`oxiblas-ffi`)~~ (RETIRED v0.2.0)
- **ndarray integration**: Seamless interop with rust-ndarray (`oxiblas-ndarray`)

**Benchmarks (`oxiblas-benchmarks`)**
- **12 comprehensive benchmark suites** (13 files, 121 benchmark functions)
- Criterion.rs framework with HTML reports and statistical analysis
- OpenBLAS comparison suite (optional feature)
- Complete coverage: BLAS L1/L2/L3, LAPACK, sparse operations, iterative/eigenvalue/SVD solvers
- Performance: DGEMM achieves 79% of OpenBLAS speed on large matrices

#### Performance

**macOS (Apple M3, ARM64 NEON):**
- **GEMM**: 79% of OpenBLAS performance (1024×1024 matrices)
- **Peak**: 20.9 GFLOPS on large matrices

**Linux x86_64 (Intel Xeon E5-2623 v4, AVX2/FMA):**
- **DGEMM f64**: 80-112% of OpenBLAS performance across all sizes
  - 1024×1024: **102% of OpenBLAS** (213 vs 208 GFLOPS) - **Faster than OpenBLAS!**
  - 256×256: 95% of OpenBLAS (220 vs 232 GFLOPS) - Peak performance
  - 512×512: 80% of OpenBLAS (194 vs 244 GFLOPS) - Optimization target
- **SGEMM f32**: 94-112% of OpenBLAS performance
  - 64×64: **112% of OpenBLAS** (253 vs 225 GFLOPS) - **Faster than OpenBLAS!**
  - 128×128: 100% of OpenBLAS (328 vs 328 GFLOPS) - Identical performance
- **Cache-aware tuning**: 13-20% performance improvement with platform-specific optimization
  - Fine-tuned blocking parameters: KC=192, MC=128 (optimized for 256KB L2 cache)
  - Increased prefetch distance: 12 iterations (Intel Xeon E5-2600 latency)

**General:**
- **GEMM-based operations**: 6-15× speedup over naive implementations
- **Cache-aware**: BLIS-style blocking for optimal memory hierarchy usage
- **Platform detection**: Linux sysfs, macOS sysctl, x86_64 CPUID fallback
- **Parallel**: Efficient multi-threading with rayon
- **SIMD**: Hand-tuned kernels for AVX2/FMA and ARM NEON

#### Documentation

- Comprehensive API documentation with examples
- README with quick start guide
- TODO.md tracking development priorities
- 121 benchmark functions demonstrating all operations

#### Code Quality

- ~156,000 lines of Rust code
- Zero clippy warnings policy enforced
- Modular architecture with 9 specialized crates
- Complete test coverage for all operations
- Safe Rust with minimal unsafe blocks (only in SIMD kernels)

#### Known Limitations

- SVD performance not yet optimized (future work)
- Some LAPACK operations may be slower than commercial libraries
- No GPU acceleration (CPU-only)
- Requires Rust 1.85+ for edition 2024 features

#### Dependencies

**Required**:
- num-complex 0.4
- num-traits 0.2
- bytemuck 1.21

**Optional**:
- rayon 1.10 (parallel feature)
- ndarray 0.17 (ndarray integration)
- nalgebra 0.33 (interop)
- half 2.4 (f16 support)
- twofloat 0.8 (f128 support)

#### Crate Structure

- `oxiblas` - Main convenience crate, re-exports all modules
- `oxiblas-core` - Core types, traits, SIMD layer
- `oxiblas-matrix` - Matrix types and utilities
- `oxiblas-blas` - BLAS Level 1/2/3 implementations
- `oxiblas-lapack` - LAPACK factorizations and solvers
- `oxiblas-sparse` - Sparse matrix formats and algorithms
- ~~`oxiblas-ffi` - C FFI bindings (CBLAS/LAPACKE compatible)~~ (RETIRED v0.2.0)
- `oxiblas-ndarray` - Integration with rust-ndarray
- `oxiblas-benchmarks` - Comprehensive performance benchmarks

---

## Release Checklist (v0.1.0 — historical)

This checklist tracked readiness for the original v0.1.0 release (2025-12-27) and is
complete; it is retained here for historical reference only and requires no further
action. See the dated entries above for what shipped in each subsequent release.

- [x] All tests pass
- [x] Zero clippy warnings
- [x] Documentation complete
- [x] Benchmarks comprehensive
- [x] LICENSE file (Apache-2.0)
- [x] README up to date
- [x] CHANGELOG created
- [x] Version 0.1.0 in all Cargo.toml
- [x] Examples functional
- [x] cargo publish --dry-run succeeds
- [x] Git tags created

---

[Unreleased]: https://github.com/cool-japan/oxiblas/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/cool-japan/oxiblas/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/cool-japan/oxiblas/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cool-japan/oxiblas/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cool-japan/oxiblas/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cool-japan/oxiblas/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiblas/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxiblas/releases/tag/v0.1.0
