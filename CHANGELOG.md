# Changelog

All notable changes to OxiBLAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - Unreleased

Fixes and additions for GitHub issues #4 and #6 through #16 (#16 only in part). The QZ
decomposition is rewritten as a LAPACK `DGGES` equivalent, Level 1/2/3 and CBLAS
behaviour is aligned with reference BLAS 3.12 (increments, `alpha == 0` and
`beta == 0`, `crotg`), 78 CBLAS entry points and several Level-1 natives are added,
and GEMM gets balanced blocking, a 2D parallel driver and aarch64 small-matrix
kernels. No public item is removed or renamed and no signature changes; behaviour
changes are listed under Fixed and Changed.

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

#### BLAS

- Level-1 natives in `oxiblas_blas::level1`, written from the Netlib 3.12 sources:
  `cabs1` (previously crate-private), `scabs1`, `dcabs1`; the complex 2-norm `cnrm2`,
  `scnrm2`, `dznrm2` and `cnrm2_strided`; `scal_real`, `csscal`, `zdscal` (complex
  vector times a real scalar, componentwise, immediate return for `alpha == 1`, no
  zero-fill for `alpha == 0`); `rotm_strided` and `zdrot_strided`. The strided forms
  take positive, negative and zero increments. (#15)
- SME2 GEMM path for f32 and f64 on AArch64 (Apple M4 and later), in stable `asm!` (no
  C, no FFI). It is selected at run time on macOS when `FEAT_SME2` (and
  `FEAT_SME_F64F64` for f64) is present and the streaming vector length is 512 bits;
  otherwise the NEON kernels run, and the SIMD-control features (`force-scalar`,
  `max-simd-128`, `max-simd-256`) turn it off. It covers all four transpose
  combinations of column-major operands and all sizes from 24^3 up (smaller problems
  with fewer than 24 rows keep `gemm_small`), plus direct NEON kernels for `m == 1`
  and `n == 1`. Design: 64 x 16 (f32) and 64 x 8 (f64) ZA tiles; one streaming-mode
  region per packed block; a column-major A read in place, or packed while it is read
  for the first columns; B (or a transposed A) transposed through ZA with a core-side
  L2 prefetch; C moved directly between memory and ZA; per-thread scratch buffers kept
  between calls; and with `Par::Rayon`, C split into one part per SME unit (one per
  performance cluster) run by persistent spinning workers. On an Apple M4 Pro this
  is about 15x (f32) and 7x (f64) the NEON path, and level with or ahead of
  Accelerate on squares and the 24 workloads of arXiv:2512.21473. `GemmKernel` gains
  four defaulted methods for this (`macro_kernel`, `pack_b`, `pack_a_t`,
  `gemm_thin`), which other implementations need not override. `oxiblas-blas` now
  depends on `libc` on macOS. (#16)
- `materialize_op` (internal) copies a column-major source directly for `Trans`, and
  small products with transposed operands copy them to column-major for the
  small-matrix kernels instead of the element-wise loop. (#16)

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

### Changed

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

[Unreleased]: https://github.com/cool-japan/oxiblas/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/cool-japan/oxiblas/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cool-japan/oxiblas/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cool-japan/oxiblas/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cool-japan/oxiblas/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiblas/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxiblas/releases/tag/v0.1.0
