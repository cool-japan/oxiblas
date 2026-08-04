# Security Policy

## Supported Versions

OxiBLAS follows [Semantic Versioning](https://semver.org/). Until a `1.0.0`
release, only the latest `0.2.x` release receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

OxiBLAS is a pure-Rust numerical library with a large `unsafe` surface
(SIMD kernels, packed/banded matrix storage, memory-mapped matrices, and a
C-ABI-compatible CBLAS layer). Memory-safety issues, undefined behavior, and
soundness holes reachable from safe APIs are treated as security bugs, in
addition to conventional vulnerabilities such as denial-of-service via
untrusted input (e.g. Matrix Market file parsing).

**Please do not open a public GitHub issue for a suspected vulnerability.**

Instead, report it privately by emailing **contact@cooljapan.tech** with:

- A description of the issue and its potential impact.
- Steps to reproduce, ideally a minimal example (crate, feature flags,
  target triple, and whether the repro requires `unsafe` on the caller's
  side or is reachable from 100% safe code).
- The affected crate(s) and version(s) (`oxiblas`, `oxiblas-core`,
  `oxiblas-matrix`, `oxiblas-blas`, `oxiblas-lapack`, `oxiblas-sparse`,
  `oxiblas-ndarray`).

You should expect an initial response within 5 business days. We will
coordinate a disclosure timeline with you once the report is triaged; please
allow us to release a fix before any public disclosure.

## Scope

In scope: all crates published from this workspace
(`crates/oxiblas*`, excluding the retired and unpublished
`crates/oxiblas-ffi`, which is excluded from the workspace, carries
`publish = false`, and is not built or tested by CI).

Out of scope: `oxiblas-benchmarks` (an unpublished, `publish = false`
development-only crate) and vulnerabilities that require enabling its
opt-in `compare-openblas` feature, which links a system OpenBLAS.
