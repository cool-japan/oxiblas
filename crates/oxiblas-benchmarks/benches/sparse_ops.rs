//! Benchmarks for sparse matrix operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_sparse::linalg::iterative::pcg;
use oxiblas_sparse::linalg::{
    SparseCholesky, bicgstab, block_cg, block_gmres, cg, gmres, idrs, minres, qmr, tfqmr,
};
use oxiblas_sparse::ops::{spadd, spmm_sparse, spmv, spmv_f64_simd, sptrsv_lower, sptrsv_upper};
use oxiblas_sparse::{CooMatrixBuilder, CsrMatrix, DiaMatrix};
use std::hint::black_box;

/// Create a random-ish sparse matrix with given dimensions and density.
fn create_sparse_matrix(n: usize, density: f64) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Add diagonal entries (always)
    for i in 0..n {
        builder.add(i, i, (i + 1) as f64 * 2.0);
    }

    // Add off-diagonal entries based on density
    let nnz_offdiag = ((n * n) as f64 * density) as usize;
    for k in 0..nnz_offdiag {
        let i = (k * 17) % n;
        let j = (k * 31 + 7) % n;
        if i != j {
            let val = ((k % 100) as f64 * 0.01) + 0.1;
            builder.add(i, j, val);
        }
    }

    builder.build().to_csr()
}

/// Create a lower triangular sparse matrix.
fn create_lower_triangular(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Diagonal
    for i in 0..n {
        builder.add(i, i, (i + 1) as f64 + 1.0);
    }

    // Sub-diagonal entries
    for i in 1..n {
        builder.add(i, i - 1, -0.5);
        if i > 1 {
            builder.add(i, i - 2, -0.25);
        }
    }

    builder.build().to_csr()
}

/// Create an upper triangular sparse matrix.
fn create_upper_triangular(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Diagonal
    for i in 0..n {
        builder.add(i, i, (i + 1) as f64 + 1.0);
    }

    // Super-diagonal entries
    for i in 0..n - 1 {
        builder.add(i, i + 1, -0.5);
        if i < n - 2 {
            builder.add(i, i + 2, -0.25);
        }
    }

    builder.build().to_csr()
}

/// Create an SPD sparse matrix (symmetric positive definite).
fn create_spd_matrix(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Strong diagonal
    for i in 0..n {
        builder.add(i, i, (n as f64) + 1.0);
    }

    // Symmetric off-diagonal entries
    for i in 1..n {
        builder.add(i, i - 1, -0.5);
        builder.add(i - 1, i, -0.5);
    }

    builder.build().to_csr()
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmv");

    for (n, density) in [(1000, 0.01), (5000, 0.001), (10000, 0.0005)].iter() {
        let n = *n;
        let density = *density;
        let a = create_sparse_matrix(n, density);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        let label = format!("{}x{}_{}nnz", n, n, nnz);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                spmv(
                    black_box(1.0),
                    black_box(&a),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_spmv_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmv_simd");

    for (n, density) in [(1000, 0.01), (5000, 0.001), (10000, 0.0005)].iter() {
        let n = *n;
        let density = *density;
        let a = create_sparse_matrix(n, density);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        let label = format!("{}x{}_{}nnz", n, n, nnz);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                spmv_f64_simd(
                    black_box(1.0),
                    black_box(&a),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_spmm_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmm");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let density = 0.02;
        let a = create_sparse_matrix(n, density);
        let b = create_sparse_matrix(n, density);
        let nnz_a = a.nnz();
        let nnz_b = b.nnz();

        // Approximate FLOP count for SpMM
        group.throughput(Throughput::Elements((nnz_a * nnz_b / n) as u64));

        let label = format!("{}x{}", n, n);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                let _ = spmm_sparse(black_box(&a), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_spadd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_add");

    for n in [500, 1000, 5000].iter() {
        let n = *n;
        let density = 0.01;
        let a = create_sparse_matrix(n, density);
        let b = create_sparse_matrix(n, density);
        let nnz = a.nnz() + b.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let label = format!("{}x{}", n, n);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                let _ = spadd(black_box(1.0), black_box(&a), black_box(1.0), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_sptrsv_lower(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_trsv_lower");

    for n in [500, 1000, 5000].iter() {
        let n = *n;
        let a = create_lower_triangular(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                sptrsv_lower(black_box(&a), black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_sptrsv_upper(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_trsv_upper");

    for n in [500, 1000, 5000].iter() {
        let n = *n;
        let a = create_upper_triangular(n);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                sptrsv_upper(black_box(&a), black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_csr_to_csc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_csr_to_csc");

    for (n, density) in [(1000, 0.01), (5000, 0.001), (10000, 0.0005)].iter() {
        let n = *n;
        let density = *density;
        let a = create_sparse_matrix(n, density);
        let nnz = a.nnz();

        group.throughput(Throughput::Elements(nnz as u64));

        let label = format!("{}x{}_{}nnz", n, n, nnz);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                let _ = black_box(&a).to_csc();
            });
        });
    }

    group.finish();
}

fn bench_cg_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_cg");

    // Use smaller sizes for iterative solver since they're more expensive
    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_spd_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = cg(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_sparse_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_cholesky");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_spd_matrix(n);
        let a_csc = a.to_csc();
        let nnz = a.nnz();

        // Cholesky is O(n^3) in worst case, but sparse structure helps
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = SparseCholesky::new(black_box(&a_csc));
            });
        });
    }

    group.finish();
}

fn bench_dia_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_dia_spmv");

    for n in [1000, 5000, 10000].iter() {
        let n = *n;

        // Create a tridiagonal matrix
        let sub = vec![-1.0f64; n - 1];
        let main = vec![2.0f64; n];
        let super_diag = vec![-1.0f64; n - 1];
        let dia = DiaMatrix::tridiagonal(sub, main, super_diag).unwrap();

        group.throughput(Throughput::Elements((3 * n) as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                dia.matvec(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

/// Create a non-symmetric but diagonally dominant sparse matrix.
fn create_nonsymmetric_matrix(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    // Strong diagonal for diagonal dominance
    for i in 0..n {
        builder.add(i, i, (n as f64) + 2.0);
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

fn bench_bicgstab_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_bicgstab");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = bicgstab(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_gmres_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_gmres");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        // GMRES with restart = 30
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = gmres(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(30), // restart
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_minres_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_minres");

    // MINRES works on symmetric matrices (SPD or indefinite)
    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_spd_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = minres(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_idrs_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_idrs");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        // IDR(s) with s=4
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = idrs(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(4), // s parameter
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_tfqmr_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_tfqmr");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = tfqmr(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_qmr_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_qmr");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_nonsymmetric_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = qmr(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_pcg_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_pcg");

    for n in [100, 500, 1000].iter() {
        let n = *n;
        let a = create_spd_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let x0: Vec<f64> = vec![0.0; n];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements(nnz as u64));

        // Extract diagonal for Jacobi preconditioner
        let mut diag = vec![1.0; n];
        for (i, diag_val) in diag.iter_mut().enumerate().take(n) {
            for j in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                if a.col_indices()[j] == i {
                    *diag_val = a.values()[j];
                    break;
                }
            }
        }

        // Jacobi preconditioner: M^{-1} r = r / diag
        let precond = move |r: &[f64]| -> Vec<f64> {
            r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect()
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = pcg(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(&precond),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_block_cg_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_block_cg");

    // Block-CG for multiple right-hand sides
    for n in [100, 500].iter() {
        let n = *n;
        let nrhs = 3;
        let a = create_spd_matrix(n);

        // Create multiple right-hand sides
        let b: Vec<Vec<f64>> = (0..nrhs)
            .map(|k| (0..n).map(|i| ((i + k * 10) % 100) as f64 * 0.01).collect())
            .collect();
        let x0: Vec<Vec<f64>> = vec![vec![0.0; n]; nrhs];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements((nnz * nrhs) as u64));

        let label = format!("{}x{}_{}rhs", n, n, nrhs);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                let _ = block_cg(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

fn bench_block_gmres_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_block_gmres");

    // Block-GMRES for multiple right-hand sides
    for n in [100, 500].iter() {
        let n = *n;
        let nrhs = 3;
        let a = create_nonsymmetric_matrix(n);

        // Create multiple right-hand sides
        let b: Vec<Vec<f64>> = (0..nrhs)
            .map(|k| (0..n).map(|i| ((i + k * 10) % 100) as f64 * 0.01).collect())
            .collect();
        let x0: Vec<Vec<f64>> = vec![vec![0.0; n]; nrhs];

        let nnz = a.nnz();
        group.throughput(Throughput::Elements((nnz * nrhs) as u64));

        let label = format!("{}x{}_{}rhs", n, n, nrhs);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |bench, _| {
            bench.iter(|| {
                let _ = block_gmres(
                    black_box(&a),
                    black_box(&b),
                    black_box(&x0),
                    black_box(30), // restart
                    black_box(1e-8),
                    black_box(100),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_spmv,
    bench_spmv_simd,
    bench_spmm_sparse,
    bench_spadd,
    bench_sptrsv_lower,
    bench_sptrsv_upper,
    bench_csr_to_csc,
    bench_cg_solver,
    bench_sparse_cholesky,
    bench_dia_spmv,
    bench_bicgstab_solver,
    bench_gmres_solver,
    bench_minres_solver,
    bench_idrs_solver,
    bench_tfqmr_solver,
    bench_qmr_solver,
    bench_pcg_solver,
    bench_block_cg_solver,
    bench_block_gmres_solver
);
criterion_main!(benches);
