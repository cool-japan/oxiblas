//! Benchmarks for sparse eigenvalue, SVD, and advanced factorization operations.
//!
//! Includes:
//! - Eigenvalue solvers: Lanczos (symmetric), Arnoldi (general), IRAM (memory-efficient)
//! - SVD solvers: Truncated SVD, Randomized SVD (fast approximate)
//! - Factorizations: Sparse LU, Sparse QR
//! - Preconditioners: ILU0, ILUT (incomplete LU)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_sparse::linalg::{
    Arnoldi, ILU0, ILUT, IRAM, IRAMConfig, Lanczos, LanczosConfig, RandomizedSparseSvd,
    RandomizedSparseSvdConfig, SparseLU, SparseQR, TruncatedSVD, TruncatedSVDConfig,
    WhichEigenvalues,
};
use oxiblas_sparse::{CooMatrixBuilder, CsrMatrix};
use std::hint::black_box;

/// Create a symmetric sparse matrix for eigenvalue problems.
fn create_symmetric_matrix(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Diagonal
    for i in 0..n {
        builder.add(i, i, 2.0 + (i % 10) as f64 * 0.1);
    }

    // Symmetric off-diagonal entries (tridiagonal pattern)
    for i in 1..n {
        let val = -1.0 / (i as f64 + 1.0);
        builder.add(i, i - 1, val);
        builder.add(i - 1, i, val);
    }

    builder.build().to_csr()
}

/// Create a non-symmetric sparse matrix for general eigenvalue problems.
fn create_nonsymmetric_matrix(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Diagonal
    for i in 0..n {
        builder.add(i, i, 3.0 + (i % 10) as f64 * 0.1);
    }

    // Asymmetric off-diagonal entries
    for i in 1..n {
        builder.add(i, i - 1, -1.0); // sub-diagonal
    }
    for i in 0..n - 1 {
        builder.add(i, i + 1, -0.5); // super-diagonal (different from sub)
    }

    builder.build().to_csr()
}

fn bench_lanczos_extremal(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_lanczos_extremal");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_symmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = LanczosConfig {
            num_eigenvalues: 5,
            max_iterations: 50,
            tolerance: 1e-8,
            which: WhichEigenvalues::LargestMagnitude,
            compute_eigenvectors: false,
            krylov_dimension: 15,
            full_reorthogonalization: true,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let lanczos = Lanczos::new(black_box(config.clone()));
                let _ = lanczos.compute(black_box(&a), black_box(None));
            });
        });
    }

    group.finish();
}

fn bench_lanczos_smallest(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_lanczos_smallest");

    for n in [100, 500].iter() {
        let n = *n;
        let a = create_symmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = LanczosConfig {
            num_eigenvalues: 5,
            max_iterations: 50,
            tolerance: 1e-8,
            which: WhichEigenvalues::SmallestMagnitude,
            compute_eigenvectors: false,
            krylov_dimension: 15,
            full_reorthogonalization: true,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let lanczos = Lanczos::new(black_box(config.clone()));
                let _ = lanczos.compute(black_box(&a), black_box(None));
            });
        });
    }

    group.finish();
}

fn bench_arnoldi(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_arnoldi");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = LanczosConfig {
            num_eigenvalues: 5,
            max_iterations: 50,
            tolerance: 1e-8,
            which: WhichEigenvalues::LargestMagnitude,
            compute_eigenvectors: false,
            krylov_dimension: 15,
            full_reorthogonalization: true,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let arnoldi = Arnoldi::new(black_box(config.clone()));
                let _ = arnoldi.compute(black_box(&a), black_box(None));
            });
        });
    }

    group.finish();
}

fn bench_iram(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_iram");

    for n in [100, 500].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = IRAMConfig {
            num_eigenvalues: 5,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 50,
            tolerance: 1e-8,
            compute_eigenvectors: false, // Faster without vectors
            krylov_dimension: 15,
            symmetric: false,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let iram = IRAM::new(black_box(config.clone()));
                let _ = iram.compute(black_box(&a), black_box(None));
            });
        });
    }

    group.finish();
}

fn bench_truncated_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_truncated_svd");

    for n in [100, 500].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = TruncatedSVDConfig {
            num_singular_values: 5,
            max_iterations: 50,
            tolerance: 1e-8,
            compute_vectors: false, // Faster without computing vectors
            krylov_dimension: 15,
            full_reorthogonalization: true,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let svd = TruncatedSVD::new(black_box(config.clone()));
                let _ = svd.compute(black_box(&a));
            });
        });
    }

    group.finish();
}

fn bench_randomized_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_randomized_svd");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let config = RandomizedSparseSvdConfig {
            num_singular_values: 5,
            oversampling: 5,
            power_iterations: 1,
            seed: None,
            compute_vectors: false, // Faster without computing vectors
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let svd = RandomizedSparseSvd::new(black_box(config.clone()));
                let _ = svd.compute(black_box(&a));
            });
        });
    }

    group.finish();
}

fn bench_sparse_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_lu");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let a_csc = a.to_csc();
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = SparseLU::new(black_box(&a_csc));
            });
        });
    }

    group.finish();
}

fn bench_sparse_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_qr");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let a_csc = a.to_csc();
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = SparseQR::new(black_box(&a_csc));
            });
        });
    }

    group.finish();
}

fn bench_ilu0(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_ilu0");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = ILU0::new(black_box(&a));
            });
        });
    }

    group.finish();
}

fn bench_ilut(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_ilut");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let drop_tol = 1e-3;
        let max_fill = 10;

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = ILUT::new(black_box(&a), black_box(drop_tol), black_box(max_fill));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lanczos_extremal,
    bench_lanczos_smallest,
    bench_arnoldi,
    bench_iram,
    bench_truncated_svd,
    bench_randomized_svd,
    bench_sparse_lu,
    bench_sparse_qr,
    bench_ilu0,
    bench_ilut
);
criterion_main!(benches);
