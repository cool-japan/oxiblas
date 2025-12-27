# OxiBLAS CI/CD Infrastructure

This directory contains GitHub Actions workflows for automated testing, benchmarking, and releasing.

## Workflows

### CI (`ci.yml`)
Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **test**: Runs test suite on Ubuntu, macOS, and Windows with stable and nightly Rust
  - All features enabled
  - No default features
  - Doc tests
- **clippy**: Linting with `-D warnings` (zero-tolerance policy)
- **fmt**: Code formatting checks
- **docs**: Documentation build with broken link detection
- **coverage**: Code coverage via tarpaulin, uploaded to Codecov
- **security-audit**: Dependency vulnerability scanning via `cargo-audit`

### Benchmarks (`benchmarks.yml`)
Performance tracking and regression detection.

**Triggers:**
- Push to `main`
- Pull requests to `main`
- Weekly schedule (Sunday 00:00 UTC)
- Manual dispatch

**Jobs:**
- **benchmark**: Runs all benchmark suites on Ubuntu and macOS
  - BLAS Level 1/2/3
  - GEMM variants
  - Sparse operations
  - LAPACK operations
  - Results stored as artifacts for 30 days
- **benchmark-compare**: PR comparison against base branch
  - Runs baseline on base branch
  - Compares PR performance
  - Reports significant regressions
- **performance-tracking**: Long-term performance data (main branch only)
  - Stores results for 90 days
  - Tracks GEMM and BLAS3 performance over time
- **benchmark-matrix**: Comprehensive benchmarking (weekly/manual)
  - Full benchmark suite
  - All matrix sizes
  - Results retained for 90 days

### Release (`release.yml`)
Automated release process triggered by version tags (`v*`).

**Jobs:**
- **create-release**: Creates GitHub release with changelog
- **publish-crates**: Sequential publication to crates.io
  - Publishes in dependency order with 30s delays
  - oxiblas-core → oxiblas-matrix → oxiblas-blas → oxiblas-lapack → oxiblas-sparse → oxiblas-ffi → oxiblas-ndarray → oxiblas
- **build-documentation**: Builds and deploys rustdoc to GitHub Pages

## Performance Tracking

### Artifacts Retention
- **Benchmark results**: 30 days (regular runs)
- **Performance data**: 90 days (tracking trends)
- **Comprehensive benchmarks**: 90 days (weekly/manual)

### Benchmark Comparison
Pull requests automatically compare performance against the base branch for critical operations (BLAS Level 3, GEMM). Significant regressions are flagged in the PR.

### Historical Data
Main branch benchmarks are stored with commit SHA for long-term performance analysis. Data includes:
- GEMM performance (f32/f64/c32/c64)
- BLAS Level 3 operations (SYRK, TRSM, etc.)
- Sparse operations

## Usage

### Running Benchmarks Manually
```bash
gh workflow run benchmarks.yml
```

### Triggering a Release
```bash
git tag v0.1.0
git push origin v0.1.0
```

### Viewing Benchmark Results
1. Navigate to Actions tab in GitHub
2. Select "Performance Benchmarks" workflow
3. Click on a run
4. Download artifacts under "Artifacts" section
5. Open `criterion/report/index.html` in browser

## Required Secrets

For full CI/CD functionality, configure these secrets in repository settings:

- `CARGO_REGISTRY_TOKEN`: crates.io API token for publishing
- GitHub token (automatic): Used for creating releases and deploying docs

## Local Testing

Test workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
sudo apt install act  # Ubuntu

# Run CI workflow
act -j test

# Run benchmarks
act -j benchmark
```

## Performance Regression Detection

The benchmark-compare job automatically detects performance regressions in PRs by:
1. Checking out the base branch
2. Running baseline benchmarks
3. Checking out the PR branch
4. Comparing against baseline
5. Reporting significant changes (>5% regression)

## Optimization Guidelines

- **Caching**: All workflows cache cargo registry, index, and build artifacts
- **Matrix strategy**: Tests run in parallel across OS/Rust versions
- **Conditional execution**: Performance tracking only on main branch
- **Artifact cleanup**: Automatic cleanup via retention policies
