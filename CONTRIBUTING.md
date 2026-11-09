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

`--all-features` builds and passes on stable, but it turns on `force-scalar`,
which replaces the GEMM micro-kernels with the scalar one (and, on aarch64,
the NEON DOT, NRM2, GEMV and TRSM/TRMM kernels with portable code), so it
does not replace the default run above.
To cover other configurations, select features explicitly, for example
`--features parallel`, or `--features runtime-tuning` on `oxiblas-blas`.
`oxiblas-core`'s `sve-nightly` needs a nightly toolchain only together with
`-C target-feature=+sve`; `lib.rs` then declares the `stdarch_aarch64_sve`
feature gate. That configuration is compile-checked, but has not run on SVE
hardware.

`crates/oxiblas-blas/src/level3/trsm/tri_kernel_neon.rs` is generated: edit
`crates/oxiblas-blas/tools/gen_tri_kernel_neon.py` and regenerate it with the
command in the file's header instead of editing it by hand.

The optional `compare-openblas` benchmark links no BLAS crate. It loads an
OpenBLAS shared library when it runs, and is not part of the normal development
loop.

## Commit / PR process

- Keep commits focused; describe the *why*, not just the *what*.
- Update `CHANGELOG.md` for user-visible changes, under the section of the
  next release (currently `## [0.2.3] - Unreleased`).
- Do not bump the crate version yourself; maintainers handle releases.
- Security-relevant issues should go through `SECURITY.md`, not a public
  PR/issue, until a fix is ready.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
