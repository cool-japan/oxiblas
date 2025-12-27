//! Cache-oblivious blocking benchmarks.
//!
//! Benchmarks for cache-oblivious traversal and blocking utilities.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_core::blocking::{
    BlockRange, BlockVisitor, RecursiveTask, cache_oblivious_traverse, gemm_block_sizes,
    morton_decode, morton_index, trsm_block_size,
};
use std::hint::black_box;

/// Matrix sizes to benchmark.
const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];

/// Simple visitor that counts blocks and elements.
struct CountingVisitor {
    block_count: usize,
    element_count: usize,
}

impl BlockVisitor for CountingVisitor {
    type Error = ();

    fn visit_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<(), ()> {
        self.block_count += 1;
        self.element_count += (row_end - row_start) * (col_end - col_start);
        Ok(())
    }
}

/// Visitor that simulates matrix operation.
struct MatrixVisitor<'a> {
    data: &'a mut [f64],
    cols: usize,
}

impl<'a> BlockVisitor for MatrixVisitor<'a> {
    type Error = ();

    fn visit_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<(), ()> {
        for row in row_start..row_end {
            for col in col_start..col_end {
                let idx = row * self.cols + col;
                if idx < self.data.len() {
                    self.data[idx] = black_box(self.data[idx] + 1.0);
                }
            }
        }
        Ok(())
    }
}

/// Benchmark cache-oblivious traversal with varying thresholds.
fn bench_traverse_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("traverse/thresholds");

    let size = 1024usize;
    let thresholds: &[usize] = &[16, 32, 64, 128, 256];

    for &threshold in thresholds {
        group.bench_with_input(
            BenchmarkId::from_parameter(threshold),
            &threshold,
            |b, &threshold| {
                let task = RecursiveTask::from_dims(size, size);

                b.iter(|| {
                    let mut visitor = CountingVisitor {
                        block_count: 0,
                        element_count: 0,
                    };
                    cache_oblivious_traverse(&mut visitor, task, threshold).unwrap();
                    black_box((visitor.block_count, visitor.element_count))
                });
            },
        );
    }
    group.finish();
}

/// Benchmark cache-oblivious traversal with varying matrix sizes.
fn bench_traverse_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("traverse/sizes");

    let threshold = 64usize;

    for &size in SIZES {
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let task = RecursiveTask::from_dims(size, size);

            b.iter(|| {
                let mut visitor = CountingVisitor {
                    block_count: 0,
                    element_count: 0,
                };
                cache_oblivious_traverse(&mut visitor, task, threshold).unwrap();
                black_box((visitor.block_count, visitor.element_count))
            });
        });
    }
    group.finish();
}

/// Benchmark actual matrix traversal with cache-oblivious vs row-major.
fn bench_matrix_access_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_pattern");

    let size = 512usize;
    let threshold = 64usize;

    // Cache-oblivious traversal
    group.throughput(Throughput::Elements((size * size) as u64));
    group.bench_function("cache_oblivious", |b| {
        let mut data = vec![0.0f64; size * size];
        let task = RecursiveTask::from_dims(size, size);

        b.iter(|| {
            let mut visitor = MatrixVisitor {
                data: &mut data,
                cols: size,
            };
            cache_oblivious_traverse(&mut visitor, task, threshold).unwrap();
            black_box(&data);
        });
    });

    // Row-major traversal
    group.bench_function("row_major", |b| {
        let mut data = vec![0.0f64; size * size];

        b.iter(|| {
            for row in 0..size {
                for col in 0..size {
                    let idx = row * size + col;
                    data[idx] = black_box(data[idx] + 1.0);
                }
            }
            black_box(&data);
        });
    });

    group.finish();
}

/// Benchmark Morton index encoding/decoding.
fn bench_morton(c: &mut Criterion) {
    let mut group = c.benchmark_group("morton");

    let count = 10_000u32;

    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for x in 0..100 {
                for y in 0..100 {
                    sum = sum.wrapping_add(morton_index(x, y));
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for z in 0..count as u64 {
                let (x, y) = morton_decode(z);
                sum = sum.wrapping_add(x as u64 + y as u64);
            }
            black_box(sum)
        });
    });

    group.bench_function("roundtrip", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for x in 0..100 {
                for y in 0..100 {
                    let z = morton_index(x, y);
                    let (dx, dy) = morton_decode(z);
                    sum = sum.wrapping_add(dx as u64 + dy as u64);
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Benchmark block size calculations.
fn bench_block_size_calc(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_size");

    group.bench_function("gemm", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for m in (64..=1024).step_by(64) {
                for n in (64..=1024).step_by(64) {
                    for k in (64..=1024).step_by(64) {
                        let (bm, bn, bk) = gemm_block_sizes::<f64>(m, n, k);
                        sum += bm + bn + bk;
                    }
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("trsm", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for n in (64..=1024).step_by(64) {
                for nrhs in (1..=64).step_by(8) {
                    let block = trsm_block_size::<f64>(n, nrhs);
                    sum += block;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Benchmark BlockRange operations.
fn bench_block_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_range");

    group.bench_function("split_recursive", |b| {
        b.iter(|| {
            let mut ranges = vec![BlockRange::new(0, 1024)];
            let mut next = Vec::new();

            // Recursively split until base case
            while !ranges.is_empty() {
                for range in ranges.drain(..) {
                    if range.len() > 32 {
                        let (left, right) = range.split();
                        next.push(left);
                        next.push(right);
                    }
                }
                std::mem::swap(&mut ranges, &mut next);
            }
            black_box(ranges.len())
        });
    });

    group.bench_function("split_at", |b| {
        b.iter(|| {
            let range = BlockRange::new(0, 1024);
            let mut sum = 0usize;
            for split_point in (64..=960).step_by(64) {
                let (left, right) = range.split_at(split_point);
                sum += left.len() + right.len();
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_traverse_thresholds,
    bench_traverse_sizes,
    bench_matrix_access_pattern,
    bench_morton,
    bench_block_size_calc,
    bench_block_range,
);
criterion_main!(benches);
