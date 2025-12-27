//! SIMD performance benchmarks.
//!
//! Benchmarks for various SIMD operations across different register sizes and platforms.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_core::memory::AlignedVec;
use oxiblas_core::simd::{SimdRegister, SimdScalar};
use std::hint::black_box;

/// Sizes to benchmark (in number of elements).
const SIZES: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];

/// Benchmark vector addition using SIMD (f64).
fn bench_f64_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/add");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f64> = AlignedVec::zeros(size);
            let bb: AlignedVec<f64> = AlignedVec::zeros(size);
            let mut result: AlignedVec<f64> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f64 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    let vc = va.add(vb);
                    unsafe { vc.store_unaligned(result.as_mut_ptr().add(offset)) };
                }

                // Handle remainder
                for i in (chunks * lanes)..size {
                    result[i] = unsafe { *a.as_ptr().add(i) };
                }

                black_box(&result);
            });
        });
    }
    group.finish();
}

/// Benchmark vector FMA using SIMD (f64).
fn bench_f64_fma(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/fma");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f64> = AlignedVec::zeros(size);
            let bb: AlignedVec<f64> = AlignedVec::zeros(size);
            let cc: AlignedVec<f64> = AlignedVec::zeros(size);
            let mut result: AlignedVec<f64> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f64 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    let vc = unsafe { Reg::load_unaligned(cc.as_ptr().add(offset)) };
                    let vr = va.mul_add(vb, vc);
                    unsafe { vr.store_unaligned(result.as_mut_ptr().add(offset)) };
                }

                black_box(&result);
            });
        });
    }
    group.finish();
}

/// Benchmark horizontal sum (f64).
fn bench_f64_reduce_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/reduce_sum");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let data: AlignedVec<f64> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f64 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                let mut acc = Reg::zero();
                for i in 0..chunks {
                    let offset = i * lanes;
                    let v = unsafe { Reg::load_unaligned(data.as_ptr().add(offset)) };
                    acc = acc.add(v);
                }

                let result = acc.reduce_sum();
                black_box(result)
            });
        });
    }
    group.finish();
}

/// Benchmark dot product (f64).
fn bench_f64_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/dot");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f64> = AlignedVec::zeros(size);
            let bb: AlignedVec<f64> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f64 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                let mut acc = Reg::zero();
                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    acc = va.mul_add(vb, acc);
                }

                let result = acc.reduce_sum();
                black_box(result)
            });
        });
    }
    group.finish();
}

/// Benchmark memory bandwidth (f64).
fn bench_f64_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/memcpy");

    for &size in SIZES {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f64>()) as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let src: AlignedVec<f64> = AlignedVec::zeros(size);
            let mut dst: AlignedVec<f64> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f64 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let v = unsafe { Reg::load_unaligned(src.as_ptr().add(offset)) };
                    unsafe { v.store_unaligned(dst.as_mut_ptr().add(offset)) };
                }

                black_box(&dst);
            });
        });
    }
    group.finish();
}

/// Benchmark vector addition using SIMD (f32).
fn bench_f32_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32/add");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f32> = AlignedVec::zeros(size);
            let bb: AlignedVec<f32> = AlignedVec::zeros(size);
            let mut result: AlignedVec<f32> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f32 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    let vc = va.add(vb);
                    unsafe { vc.store_unaligned(result.as_mut_ptr().add(offset)) };
                }

                black_box(&result);
            });
        });
    }
    group.finish();
}

/// Benchmark vector FMA using SIMD (f32).
fn bench_f32_fma(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32/fma");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f32> = AlignedVec::zeros(size);
            let bb: AlignedVec<f32> = AlignedVec::zeros(size);
            let cc: AlignedVec<f32> = AlignedVec::zeros(size);
            let mut result: AlignedVec<f32> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f32 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    let vc = unsafe { Reg::load_unaligned(cc.as_ptr().add(offset)) };
                    let vr = va.mul_add(vb, vc);
                    unsafe { vr.store_unaligned(result.as_mut_ptr().add(offset)) };
                }

                black_box(&result);
            });
        });
    }
    group.finish();
}

/// Benchmark horizontal sum (f32).
fn bench_f32_reduce_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32/reduce_sum");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let data: AlignedVec<f32> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f32 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                let mut acc = Reg::zero();
                for i in 0..chunks {
                    let offset = i * lanes;
                    let v = unsafe { Reg::load_unaligned(data.as_ptr().add(offset)) };
                    acc = acc.add(v);
                }

                let result = acc.reduce_sum();
                black_box(result)
            });
        });
    }
    group.finish();
}

/// Benchmark dot product (f32).
fn bench_f32_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32/dot");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a: AlignedVec<f32> = AlignedVec::zeros(size);
            let bb: AlignedVec<f32> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f32 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                let mut acc = Reg::zero();
                for i in 0..chunks {
                    let offset = i * lanes;
                    let va = unsafe { Reg::load_unaligned(a.as_ptr().add(offset)) };
                    let vb = unsafe { Reg::load_unaligned(bb.as_ptr().add(offset)) };
                    acc = va.mul_add(vb, acc);
                }

                let result = acc.reduce_sum();
                black_box(result)
            });
        });
    }
    group.finish();
}

/// Benchmark memory bandwidth (f32).
fn bench_f32_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32/memcpy");

    for &size in SIZES {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let src: AlignedVec<f32> = AlignedVec::zeros(size);
            let mut dst: AlignedVec<f32> = AlignedVec::zeros(size);

            b.iter(|| {
                type Reg = <f32 as SimdScalar>::Simd256;
                let lanes = Reg::LANES;
                let chunks = size / lanes;

                for i in 0..chunks {
                    let offset = i * lanes;
                    let v = unsafe { Reg::load_unaligned(src.as_ptr().add(offset)) };
                    unsafe { v.store_unaligned(dst.as_mut_ptr().add(offset)) };
                }

                black_box(&dst);
            });
        });
    }
    group.finish();
}

/// Compare aligned vs unaligned loads.
fn bench_aligned_vs_unaligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64/alignment");

    let size = 16384usize;
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("aligned", |b| {
        let data: AlignedVec<f64> = AlignedVec::zeros(size);
        let mut result: AlignedVec<f64> = AlignedVec::zeros(size);

        b.iter(|| {
            type Reg = <f64 as SimdScalar>::Simd256;
            let lanes = Reg::LANES;
            let chunks = size / lanes;

            for i in 0..chunks {
                let offset = i * lanes;
                let v = unsafe { Reg::load_aligned(data.as_ptr().add(offset)) };
                unsafe { v.store_aligned(result.as_mut_ptr().add(offset)) };
            }

            black_box(&result);
        });
    });

    group.bench_function("unaligned", |b| {
        let data: AlignedVec<f64> = AlignedVec::zeros(size);
        let mut result: AlignedVec<f64> = AlignedVec::zeros(size);

        b.iter(|| {
            type Reg = <f64 as SimdScalar>::Simd256;
            let lanes = Reg::LANES;
            let chunks = size / lanes;

            for i in 0..chunks {
                let offset = i * lanes;
                let v = unsafe { Reg::load_unaligned(data.as_ptr().add(offset)) };
                unsafe { v.store_unaligned(result.as_mut_ptr().add(offset)) };
            }

            black_box(&result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_f64_add,
    bench_f64_fma,
    bench_f64_reduce_sum,
    bench_f64_dot,
    bench_f64_memcpy,
    bench_f32_add,
    bench_f32_fma,
    bench_f32_reduce_sum,
    bench_f32_dot,
    bench_f32_memcpy,
    bench_aligned_vs_unaligned,
);
criterion_main!(benches);
