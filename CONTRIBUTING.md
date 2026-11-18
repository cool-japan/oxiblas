# Contributing to OxiBLAS

Thanks for your interest in improving OxiBLAS, a pure-Rust BLAS/LAPACK
implementation for the [SciRS2](https://github.com/cool-japan/scirs)
ecosystem. This document covers the practical rules for sending a change.

## Workspace layout

OxiBLAS is a Cargo workspace (`Cargo.toml` at the repo root) with these
publishable members under `crates/`:

- `oxiblas-core` — scalar types, SIMD abstraction, memory primitives
- `oxiblas-matrix` — dense/packed/banded/memory-mapped matrix storage
- `oxiblas-blas` — BLAS levels 1-3 and the CBLAS C-ABI surface
- `oxiblas-lapack` — LAPACK-equivalent factorizations and solvers
- `oxiblas-sparse` — sparse matrix formats, solvers, and preconditioners
- `oxiblas-ndarray` — optional `ndarray` interop
- `oxiblas` — the facade crate re-exporting the above

`crates/oxiblas-benchmarks` is a `publish = false` development-only crate
and `crates/oxiblas-ffi` is retired (excluded from the workspace, not built
by CI); neither is a target for routine contributions.

## Before you start

For anything beyond a small fix, please open an issue first to discuss the
approach — this project has strict internal policies (see below) that
sometimes constrain the shape of an acceptable change more than usual.

## Code quality requirements

These are enforced in review. The GitHub Actions workflows are disabled (see
`.github/README-internal.md`), so run the checks locally before sending a
change:

- **No warnings.** `cargo clippy --workspace --exclude oxiblas-benchmarks
  --all-targets -- -D warnings` must be clean. `msrv` in `clippy.toml`
  tracks `rust-version` in `Cargo.toml`.
- **No `unwrap()`/`expect()` in production code.** Tests and doctests are
  exempt; production code must return a `Result`/`Option` or use a
  documented, checked invariant instead.
- **Pure Rust by default.** Default features must not pull in C/C++/Fortran
  dependencies. Any FFI must be behind a non-default feature flag. See
  `deny.toml` for the banned-crate list (COOLJAPAN ecosystem replacements
  such as `oxiblas` itself in place of OpenBLAS bindings).
- **Formatting.** `cargo fmt --all -- --check` must pass; see
  `rustfmt.toml`.
- **snake_case / standard Rust naming conventions** throughout.
- **File size.** Keep source files under 2000 lines; split cohesive
  modules out and re-export from the original path when a file grows past
  that.
- **Workspace-level dependency versions.** Add new dependencies to
  `[workspace.dependencies]` in the root `Cargo.toml` and reference them
  from member crates with `dep-name.workspace = true`; do not pin versions
  in individual crate manifests (`keywords`/`categories` may still differ
  per crate).
- **`unsafe` code.** This crate has a large, deliberate `unsafe` surface
  (SIMD kernels, raw-pointer matrix views, memory-mapped I/O, CBLAS ABI
  entry points). Every `unsafe fn` needs a `# Safety` doc section stating
  its preconditions, and every `unsafe { ... }` block needs a `// SAFETY:`
  comment justifying why the preconditions hold at that call site
  (`clippy::undocumented_unsafe_blocks` enforces this in `oxiblas-blas`,
  `oxiblas-matrix`, `oxiblas-lapack`, `oxiblas-sparse` and `oxiblas-ndarray`;
  `oxiblas-core` still has undocumented blocks). Prefer
  a safe API with a validated constructor (see `Mat::from_slice` /
  `checked_dim_mul`) over an unchecked one where the two are not
  meaningfully different in performance.

## Testing

- `cargo build --workspace --exclude oxiblas-benchmarks`
- `cargo nextest run --workspace --exclude oxiblas-benchmarks` (or
  `cargo test` if `nextest` is not installed)
- `cargo test --doc --workspace --exclude oxiblas-benchmarks` for doctests
- New public-facing behavior needs a unit test; bug fixes need a regression
  test that fails without the fix.
- Four SVD tests that compare with the one-sided Jacobi `Svd` are too slow for
  debug builds and are ignored there; run them with `--release`, or with
  `--run-ignored all` under nextest.
- The GEMM A/B harness of `oxiblas-blas` (`level3::gemm_ab_tests`) consists of
  timing tests marked `#[ignore = "timing harness"]`. They decide the GEMM
  defaults and are not part of the normal run; their module docs say how to run
  a group.

`--all-features` builds and passes on stable, but it turns on `force-scalar`,
which replaces the GEMM micro-kernels with the scalar one (and, on aarch64,
the NEON DOT, NRM2, GEMV and TRSM/TRMM kernels with portable code), so it
does not replace the default run above. It also turns on `sme2`: the backend
then compiles, but the SIMD caps keep it off at run time.
To cover other configurations, select features explicitly, for example
`--features parallel`, or `--features runtime-tuning` on `oxiblas-blas`.
`oxiblas-core`'s `sve-nightly` needs a nightly toolchain only together with
`-C target-feature=+sve`; `lib.rs` then declares the `stdarch_aarch64_sve`
feature gate. That configuration is compile-checked, but has not run on SVE
hardware.

`crates/oxiblas-blas/src/level3/trsm/tri_kernel_neon.rs` is generated: edit
`crates/oxiblas-blas/tools/gen_tri_kernel_neon.py` and regenerate it with the
command in the file's header instead of editing it by hand. The same holds for
the marked regions of `crates/oxiblas-blas/src/level3/sme2/kernel_f64.rs` and
`few_rows.rs` (beside it), written by `gen_sme2_kernel_f64.py` and
`gen_sme2_few_rows.py` in the same `tools/` directory. Check them with
`python3 -I crates/oxiblas-blas/tools/gen_sme2_kernel_f64.py --check` and
`python3 -I crates/oxiblas-blas/tools/gen_sme2_few_rows.py --check` (exit 0
when the committed output is current).

The `sme2` feature (SME2 backend, aarch64): on a machine without SME2 nothing
of the backend executes natively, so
`cargo test -p oxiblas-blas --features sme2,parallel --lib sme2` builds and
assembles it while the tests that need the backend skip.
`crates/oxiblas-blas/tools/sme2_qemu.sh` runs them under QEMU user-mode
emulation (Docker, or `--native` on a Linux host with qemu-aarch64 10.1 or
newer, rustup's `aarch64-unknown-linux-gnu` target, rsync and, on x86-64, an
aarch64 cross linker): a positive pass on a CPU with a 512-bit streaming
vector length, where
`OXIBLAS_TEST_REQUIRE_SME2=1` makes a skipped test a failure, and a negative
pass with 256 bits, where `OXIBLAS_TEST_EXPECT_NO_SME2=1` makes a test that
finds SME2 a failure; `OXIBLAS_TEST_SME2_MARKERS=1` makes the tests print the
marker lines that the verdict counts. Each pass then runs the integration tests
`sme2_public_path` and `gemm_rayon_pool` as a release and as a debug build. The
result is one archive, `target/sme2-qemu/sme2-qemu-results.tar.gz`. On SME2
hardware, `--hardware` runs the positive pass natively, without Docker and QEMU;
on macOS a test with `fpsr` in its name can then fail at random with the FPSR
`0x800009f` in its message, a known limit of 0.2.4 (see the CHANGELOG), not a
wrong result. The script needs `sme2_qemu_verdicts.sh` and
`sme2_qemu_selftest.sh` beside it; `--self-test` checks its own logic without
Docker, QEMU or cargo, and `--help` lists the options.

`crates/oxiblas-blas/tools/x86_gemm_runbook.sh` collects, in one run on an
x86-64 Linux machine with AVX2 and FMA, the tests, A/B passes and benchmarks
that decide the x86-64 GEMM defaults, and packs them into
`target/x86-runbook/x86-runbook-results.tar.gz`.
`crates/oxiblas-blas/tools/gemm_tiny_call_check.sh` compares the per-call time
of small GEMM products with a base release (v0.2.3 by default) and writes
`target/gemm_tiny_call/summary.txt`; run it after changing the GEMM entry
points or routes.

The optional `compare-openblas` benchmark links no BLAS crate. It loads an
OpenBLAS shared library when it runs, and is not part of the normal development
loop.

## Commit / PR process

- Keep commits focused; describe the *why*, not just the *what*.
- Update `CHANGELOG.md` for user-visible changes, under the section of the
  next release (the topmost `## [X.Y.Z] - Unreleased` heading).
- Do not bump the crate version yourself; maintainers handle releases.
- Security-relevant issues should go through `SECURITY.md`, not a public
  PR/issue, until a fix is ready.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
