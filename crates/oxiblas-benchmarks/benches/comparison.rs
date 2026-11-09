//! Comparison benchmarks: OxiBLAS vs OpenBLAS.
//!
//! This benchmark suite directly compares OxiBLAS performance against OpenBLAS
//! for common BLAS operations. Run with:
//!
//! ```bash
//! cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas
//! ```
//!
//! ## How OpenBLAS is called
//!
//! OpenBLAS is loaded at run time with `dlopen`, and each routine is resolved
//! with `dlsym` on that library handle. Do not link it instead: `oxiblas-blas`
//! exports `#[no_mangle]` `cblas_*` symbols, so any `cblas_*` reference in
//! this binary binds to OxiBLAS at link time and OpenBLAS never runs. That is
//! what happened before 0.2.3, when the `openblas` rows went through
//! `ndarray`'s `blas` feature and `cblas-sys`.
//!
//! The library is `OPENBLAS_LIB` if that is set. Otherwise the default library
//! names are tried, then `brew --prefix openblas`. Startup panics if a routine
//! resolves into this binary, and prints the library path, build
//! configuration and kernel in use. Unix only.
//!
//! Both sides work on the same `Mat` buffers, so layout, alignment and leading
//! dimensions match, and neither side allocates inside the measured loop.
//! The sequential groups run OpenBLAS on one thread, whatever
//! `OPENBLAS_NUM_THREADS` says: startup sets the count through
//! `openblas_set_num_threads` and, in OpenMP builds, `omp_set_num_threads`, and
//! panics if a threaded GEMM still uses more than one core of CPU time. The
//! `*_parallel_comparison` groups give both libraries the same thread count:
//! `RAYON_NUM_THREADS` when it is set, otherwise every available core.
//!
//! ## Benchmarks Included
//!
//! ### Level 1 (Vector Operations)
//! - DDOT (f64 dot product)
//! - DAXPY (f64 vector addition)
//! - DNRM2 (f64 vector norm)
//!
//! ### Level 2 (Matrix-Vector Operations)
//! - DGEMV (f64 matrix-vector multiplication)
//!
//! ### Level 3 (Matrix-Matrix Operations)
//! - DGEMM (f64 matrix multiplication) - Square and rectangular matrices
//! - SGEMM (f32 matrix multiplication)
//! - DGEMM and SGEMM on several threads (see above for the count)
//! - DTRSM (f64 triangular solve with multiple RHS)
//! - DSYRK (f64 symmetric rank-k update)
//!
//! ## Results Interpretation
//!
//! Criterion reports each library and size separately:
//! - Time: lower is better
//! - Throughput: higher is better. The GEMM groups count `n³` elements per
//!   call, so GFLOPS = 2 × Gelem/s.
//!
//! Criterion prints no ratio between the two libraries. The READMEs give
//! OxiBLAS's speed as a percentage of OpenBLAS's, that is OpenBLAS time /
//! OxiBLAS time × 100%, so above 100% means OxiBLAS is faster.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::num::NonZero;
use std::thread;

// OxiBLAS imports
use oxiblas_blas::level1::{axpy, dot, nrm2};
use oxiblas_blas::level2::{GemvTrans, gemv};
use oxiblas_blas::level3::{Diag, Side, Trans, Uplo, gemm, gemm_with_par, syrk, trsm_in_place};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

/// OpenBLAS routines resolved at run time from the loaded library.
mod openblas {
    use oxiblas_matrix::Mat;
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::process::Command;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    const COL_MAJOR: c_int = 102;
    const NO_TRANS: c_int = 111;
    const LOWER: c_int = 122;
    const NON_UNIT: c_int = 131;
    const LEFT: c_int = 141;

    type Dot = unsafe extern "C" fn(c_int, *const f64, c_int, *const f64, c_int) -> f64;
    type Axpy = unsafe extern "C" fn(c_int, f64, *const f64, c_int, *mut f64, c_int);
    type Nrm2 = unsafe extern "C" fn(c_int, *const f64, c_int) -> f64;
    type Gemv = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        f64,
        *const f64,
        c_int,
        *const f64,
        c_int,
        f64,
        *mut f64,
        c_int,
    );
    type Gemm<T> = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        T,
        *const T,
        c_int,
        *const T,
        c_int,
        T,
        *mut T,
        c_int,
    );
    type Trsm = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        f64,
        *const f64,
        c_int,
        *mut f64,
        c_int,
    );
    type Syrk = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        f64,
        *const f64,
        c_int,
        f64,
        *mut f64,
        c_int,
    );
    type SetThreads = unsafe extern "C" fn(c_int);
    type GetString = unsafe extern "C" fn() -> *const c_char;

    /// The routines these benchmarks call.
    pub struct OpenBlas {
        ddot: Dot,
        daxpy: Axpy,
        dnrm2: Nrm2,
        dgemv: Gemv,
        dgemm: Gemm<f64>,
        sgemm: Gemm<f32>,
        dtrsm: Trsm,
        dsyrk: Syrk,
        set_threads: SetThreads,
        // OpenMP builds take their thread count from `omp_get_max_threads()`,
        // so `openblas_set_num_threads` alone does not bound them.
        omp_set_threads: Option<SetThreads>,
    }

    /// Returns the process-wide OpenBLAS, loading it on first use.
    pub fn get() -> &'static OpenBlas {
        static OPENBLAS: OnceLock<OpenBlas> = OnceLock::new();
        OPENBLAS.get_or_init(OpenBlas::load)
    }

    /// Candidate library paths, in the order they are tried.
    fn library_candidates() -> Vec<String> {
        if let Ok(path) = std::env::var("OPENBLAS_LIB") {
            return vec![path];
        }
        let mut candidates: Vec<String> =
            ["libopenblas.dylib", "libopenblas.so.0", "libopenblas.so"]
                .map(String::from)
                .to_vec();
        // Homebrew's formula is keg-only, so the dynamic loader does not search it.
        if let Ok(output) = Command::new("brew").args(["--prefix", "openblas"]).output()
            && output.status.success()
        {
            let prefix = String::from_utf8_lossy(&output.stdout);
            candidates.push(format!("{}/lib/libopenblas.dylib", prefix.trim()));
        }
        candidates
    }

    fn open_library() -> *mut c_void {
        let mut tried = Vec::new();
        for path in library_candidates() {
            let c_path = CString::new(path.as_str()).expect("library path contains a NUL byte");
            // SAFETY: `c_path` is NUL-terminated. Loading runs only the
            // initialisers of OpenBLAS and of the libraries it depends on.
            let handle =
                unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return handle;
            }
            tried.push(path);
        }
        panic!("could not load OpenBLAS (tried {tried:?}); set OPENBLAS_LIB to its path");
    }

    /// Returns the load address and path of the image that contains `addr`.
    fn image_of(addr: *const c_void) -> (*mut c_void, String) {
        // SAFETY: `Dl_info` is plain data, for which all-zero bytes are valid.
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid out-pointer. `addr` is only looked up, not read.
        let found = unsafe { libc::dladdr(addr, &mut info) };
        assert_ne!(found, 0, "dladdr found no image for {addr:?}");
        // SAFETY: on success `dli_fname` is a NUL-terminated path owned by the loader.
        let path = unsafe { CStr::from_ptr(info.dli_fname) };
        (info.dli_fbase, path.to_string_lossy().into_owned())
    }

    /// Resolves `name` from the loaded library and rejects a definition that
    /// lives in this binary.
    ///
    /// # Safety
    /// `F` must be the `extern "C"` function pointer type of `name`.
    unsafe fn resolve<F: Copy>(handle: *mut c_void, name: &str) -> F {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        let symbol = CString::new(name).expect("symbol name contains a NUL byte");
        // SAFETY: `handle` came from a successful `dlopen`, and `symbol` is
        // NUL-terminated.
        let addr = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
        assert!(
            !addr.is_null(),
            "{name} is missing from the OpenBLAS library"
        );
        let (own_image, own_path) = image_of(get as *const () as *const c_void);
        let (image, path) = image_of(addr);
        assert_ne!(
            image, own_image,
            "{name} resolved to {own_path}, not to OpenBLAS ({path})"
        );
        // SAFETY: `addr` is non-null, and the caller guarantees that `F` is the
        // pointer type of `name`.
        unsafe { std::mem::transmute_copy(&addr) }
    }

    /// Resolves an optional symbol from the loaded library or its dependencies.
    ///
    /// # Safety
    /// `F` must be the `extern "C"` function pointer type of `name`.
    unsafe fn resolve_optional<F: Copy>(handle: *mut c_void, name: &str) -> Option<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        let symbol = CString::new(name).expect("symbol name contains a NUL byte");
        // SAFETY: `handle` came from a successful `dlopen`, and `symbol` is
        // NUL-terminated.
        let addr = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
        // SAFETY: `addr` is non-null here, and the caller guarantees that `F` is
        // the pointer type of `name`.
        (!addr.is_null()).then(|| unsafe { std::mem::transmute_copy(&addr) })
    }

    /// Returns the CPU time used so far by every thread of this process.
    fn process_cpu_time() -> Duration {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid out-pointer.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
        assert_eq!(status, 0, "clock_gettime failed");
        Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
    }

    impl OpenBlas {
        fn load() -> Self {
            let handle = open_library();
            // SAFETY: each field's type is the `cblas.h` prototype of its symbol.
            let blas = unsafe {
                Self {
                    ddot: resolve(handle, "cblas_ddot"),
                    daxpy: resolve(handle, "cblas_daxpy"),
                    dnrm2: resolve(handle, "cblas_dnrm2"),
                    dgemv: resolve(handle, "cblas_dgemv"),
                    dgemm: resolve(handle, "cblas_dgemm"),
                    sgemm: resolve(handle, "cblas_sgemm"),
                    dtrsm: resolve(handle, "cblas_dtrsm"),
                    dsyrk: resolve(handle, "cblas_dsyrk"),
                    set_threads: resolve(handle, "openblas_set_num_threads"),
                    omp_set_threads: resolve_optional(handle, "omp_set_num_threads"),
                }
            };
            blas.set_num_threads(1);
            let cores = blas.cores_used_by_gemm();
            assert!(
                cores < 1.5,
                "OpenBLAS used {cores:.1} cores with one thread requested"
            );

            // SAFETY: these are the `openblas_config.h` prototypes.
            let (config, core): (GetString, GetString) = unsafe {
                (
                    resolve(handle, "openblas_get_config"),
                    resolve(handle, "openblas_get_corename"),
                )
            };
            let (_, path) = image_of(blas.dgemm as *const () as *const c_void);
            // SAFETY: both strings are static and NUL-terminated, and the calls
            // take no arguments.
            let (config, core) = unsafe {
                (
                    CStr::from_ptr(config()).to_string_lossy(),
                    CStr::from_ptr(core()).to_string_lossy(),
                )
            };
            eprintln!("OpenBLAS under test: {path}");
            eprintln!("  config: {config}");
            eprintln!("  kernel: {core}, sequential groups measured at {cores:.2} cores");
            blas
        }

        /// CPU seconds per wall-clock second of a GEMM large enough to thread.
        fn cores_used_by_gemm(&self) -> f64 {
            let n = 1024;
            let a = Mat::<f64>::filled(n, n, 1.0);
            let b = Mat::<f64>::filled(n, n, 2.0);
            let mut c = Mat::<f64>::zeros(n, n);
            let cpu_start = process_cpu_time();
            let wall_start = Instant::now();
            self.dgemm(&a, &b, &mut c);
            (process_cpu_time() - cpu_start).as_secs_f64() / wall_start.elapsed().as_secs_f64()
        }

        /// Sets the number of OpenBLAS worker threads.
        pub fn set_num_threads(&self, threads: usize) {
            // SAFETY: both functions take any positive count.
            unsafe {
                (self.set_threads)(threads as c_int);
                if let Some(omp_set_threads) = self.omp_set_threads {
                    omp_set_threads(threads as c_int);
                }
            }
        }

        pub fn ddot(&self, x: &[f64], y: &[f64]) -> f64 {
            assert_eq!(x.len(), y.len());
            // SAFETY: both slices hold `x.len()` elements at unit increment.
            unsafe { (self.ddot)(x.len() as c_int, x.as_ptr(), 1, y.as_ptr(), 1) }
        }

        pub fn daxpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
            assert_eq!(x.len(), y.len());
            // SAFETY: both slices hold `x.len()` elements at unit increment.
            unsafe { (self.daxpy)(x.len() as c_int, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
        }

        pub fn dnrm2(&self, x: &[f64]) -> f64 {
            // SAFETY: `x` holds `x.len()` elements at unit increment.
            unsafe { (self.dnrm2)(x.len() as c_int, x.as_ptr(), 1) }
        }

        /// `y := A * x`.
        pub fn dgemv(&self, a: &Mat<f64>, x: &[f64], y: &mut [f64]) {
            assert_eq!(x.len(), a.ncols());
            assert_eq!(y.len(), a.nrows());
            // SAFETY: `a` is a live column-major matrix with leading dimension
            // `row_stride()`, and the vector lengths were checked above.
            unsafe {
                (self.dgemv)(
                    COL_MAJOR,
                    NO_TRANS,
                    a.nrows() as c_int,
                    a.ncols() as c_int,
                    1.0,
                    a.as_ptr(),
                    a.row_stride() as c_int,
                    x.as_ptr(),
                    1,
                    0.0,
                    y.as_mut_ptr(),
                    1,
                );
            }
        }

        /// `C := A * B`.
        pub fn dgemm(&self, a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>) {
            assert_eq!(a.ncols(), b.nrows());
            assert_eq!((a.nrows(), b.ncols()), (c.nrows(), c.ncols()));
            // SAFETY: all three are live column-major matrices with leading
            // dimension `row_stride()`, and the shapes were checked above.
            unsafe {
                (self.dgemm)(
                    COL_MAJOR,
                    NO_TRANS,
                    NO_TRANS,
                    c.nrows() as c_int,
                    c.ncols() as c_int,
                    a.ncols() as c_int,
                    1.0,
                    a.as_ptr(),
                    a.row_stride() as c_int,
                    b.as_ptr(),
                    b.row_stride() as c_int,
                    0.0,
                    c.as_mut_ptr(),
                    c.row_stride() as c_int,
                );
            }
        }

        /// `C := A * B`.
        pub fn sgemm(&self, a: &Mat<f32>, b: &Mat<f32>, c: &mut Mat<f32>) {
            assert_eq!(a.ncols(), b.nrows());
            assert_eq!((a.nrows(), b.ncols()), (c.nrows(), c.ncols()));
            // SAFETY: all three are live column-major matrices with leading
            // dimension `row_stride()`, and the shapes were checked above.
            unsafe {
                (self.sgemm)(
                    COL_MAJOR,
                    NO_TRANS,
                    NO_TRANS,
                    c.nrows() as c_int,
                    c.ncols() as c_int,
                    a.ncols() as c_int,
                    1.0,
                    a.as_ptr(),
                    a.row_stride() as c_int,
                    b.as_ptr(),
                    b.row_stride() as c_int,
                    0.0,
                    c.as_mut_ptr(),
                    c.row_stride() as c_int,
                );
            }
        }

        /// Solves `A * X = B` for lower triangular, non-unit `A`, overwriting `x`.
        pub fn dtrsm(&self, a: &Mat<f64>, x: &mut Mat<f64>) {
            assert_eq!(a.nrows(), a.ncols());
            assert_eq!(a.nrows(), x.nrows());
            // SAFETY: both are live column-major matrices with leading dimension
            // `row_stride()`, and the shapes were checked above.
            unsafe {
                (self.dtrsm)(
                    COL_MAJOR,
                    LEFT,
                    LOWER,
                    NO_TRANS,
                    NON_UNIT,
                    x.nrows() as c_int,
                    x.ncols() as c_int,
                    1.0,
                    a.as_ptr(),
                    a.row_stride() as c_int,
                    x.as_mut_ptr(),
                    x.row_stride() as c_int,
                );
            }
        }

        /// `C := A * A^T` on the lower triangle.
        pub fn dsyrk(&self, a: &Mat<f64>, c: &mut Mat<f64>) {
            assert_eq!(c.nrows(), c.ncols());
            assert_eq!(a.nrows(), c.nrows());
            // SAFETY: both are live column-major matrices with leading dimension
            // `row_stride()`, and the shapes were checked above.
            unsafe {
                (self.dsyrk)(
                    COL_MAJOR,
                    LOWER,
                    NO_TRANS,
                    c.nrows() as c_int,
                    a.ncols() as c_int,
                    1.0,
                    a.as_ptr(),
                    a.row_stride() as c_int,
                    0.0,
                    c.as_mut_ptr(),
                    c.row_stride() as c_int,
                );
            }
        }
    }
}

/// Benchmark DGEMM: C = alpha * A * B + beta * C
fn bench_gemm_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("gemm_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let a_mat = Mat::from_slice(n, n, &a_data);
        let b_mat = Mat::from_slice(n, n, &b_data);
        let mut c_mat = Mat::zeros(n, n);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.dgemm(black_box(&a_mat), black_box(&b_mat), black_box(&mut c_mat)));
        });
    }

    group.finish();
}

/// Benchmark DGEMM with rectangular matrices
fn bench_gemm_rectangular_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("gemm_rectangular_comparison");

    for &(m, k, n) in [(128, 256, 128), (256, 512, 256), (512, 256, 512)].iter() {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a_data: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.01).collect();
        let a_mat = Mat::from_slice(m, k, &a_data);
        let b_mat = Mat::from_slice(k, n, &b_data);
        let mut c_mat = Mat::zeros(m, n);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", &param), &param, |bench, _| {
            bench.iter(|| blas.dgemm(black_box(&a_mat), black_box(&b_mat), black_box(&mut c_mat)));
        });
    }

    group.finish();
}

/// Benchmark DGEMV: y = alpha * A * x + beta * y
fn bench_gemv_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("gemv_comparison");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let a_mat = Mat::from_slice(n, n, &a_data);
        let mut y = vec![0.0; n];

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemv(
                    black_box(GemvTrans::NoTrans),
                    black_box(1.0),
                    black_box(a_mat.as_ref()),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.dgemv(black_box(&a_mat), black_box(&x), black_box(&mut y)));
        });
    }

    group.finish();
}

/// Benchmark DDOT: dot product of two vectors
fn bench_dot_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("dot_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.02).collect();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| dot(black_box(&x), black_box(&y)));
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.ddot(black_box(&x), black_box(&y)));
        });
    }

    group.finish();
}

/// Benchmark DAXPY: y = alpha * x + y
fn bench_axpy_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("axpy_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let mut y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.02).collect();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| axpy(black_box(2.5), black_box(&x), black_box(&mut y)));
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.daxpy(black_box(2.5), black_box(&x), black_box(&mut y)));
        });
    }

    group.finish();
}

/// Benchmark DNRM2: Euclidean norm of a vector
fn bench_nrm2_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("nrm2_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| nrm2(black_box(&x)));
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.dnrm2(black_box(&x)));
        });
    }

    group.finish();
}

/// Benchmark SGEMM (single precision): C = alpha * A * B + beta * C
fn bench_sgemm_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("sgemm_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let a_mat: Mat<f32> = Mat::from_slice(n, n, &a_data);
        let b_mat: Mat<f32> = Mat::from_slice(n, n, &b_data);
        let mut c_mat: Mat<f32> = Mat::zeros(n, n);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0f32),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0f32),
                    black_box(c_mat.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.sgemm(black_box(&a_mat), black_box(&b_mat), black_box(&mut c_mat)));
        });
    }

    group.finish();
}

/// Thread count of the parallel groups.
///
/// `RAYON_NUM_THREADS` sizes the pool `Par::Rayon` runs on, so OpenBLAS follows
/// it, and both libraries always get the same count.
fn parallel_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|threads| threads.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, NonZero::get))
}

/// Benchmark DGEMM on `parallel_threads()` threads
fn bench_gemm_parallel_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let threads = parallel_threads();
    let mut group = c.benchmark_group("gemm_parallel_comparison");
    group.sample_size(20);

    for size in [512, 1024, 2048].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let a_mat = Mat::from_slice(n, n, &a_data);
        let b_mat = Mat::from_slice(n, n, &b_data);
        let mut c_mat = Mat::zeros(n, n);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm_with_par(
                    black_box(1.0),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                    Par::Rayon,
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            blas.set_num_threads(threads);
            bench.iter(|| blas.dgemm(black_box(&a_mat), black_box(&b_mat), black_box(&mut c_mat)));
            blas.set_num_threads(1);
        });
    }

    group.finish();
}

/// Benchmark SGEMM on `parallel_threads()` threads
fn bench_sgemm_parallel_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let threads = parallel_threads();
    let mut group = c.benchmark_group("sgemm_parallel_comparison");
    group.sample_size(20);

    for size in [512, 1024, 2048].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let a_mat: Mat<f32> = Mat::from_slice(n, n, &a_data);
        let b_mat: Mat<f32> = Mat::from_slice(n, n, &b_data);
        let mut c_mat: Mat<f32> = Mat::zeros(n, n);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm_with_par(
                    black_box(1.0f32),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0f32),
                    black_box(c_mat.as_mut()),
                    Par::Rayon,
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            blas.set_num_threads(threads);
            bench.iter(|| blas.sgemm(black_box(&a_mat), black_box(&b_mat), black_box(&mut c_mat)));
            blas.set_num_threads(1);
        });
    }

    group.finish();
}

/// Benchmark DTRSM: Solve op(A) * X = alpha * B for X
fn bench_trsm_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("trsm_comparison");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create a lower triangular matrix with positive diagonal (for stability)
        let a_data: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx % n;
                let j = idx / n;
                if i >= j {
                    if i == j { (i + 1) as f64 } else { 0.1 }
                } else {
                    0.0
                }
            })
            .collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01 + 1.0).collect();

        let a_mat = Mat::from_slice(n, n, &a_data);
        let b_mat = Mat::from_slice(n, n, &b_data);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter_batched_ref(
                || b_mat.clone(),
                |x| {
                    let _ = trsm_in_place(
                        black_box(Side::Left),
                        black_box(Uplo::Lower),
                        black_box(Trans::NoTrans),
                        black_box(Diag::NonUnit),
                        black_box(a_mat.as_ref()),
                        black_box(x.as_mut()),
                    );
                },
                BatchSize::LargeInput,
            );
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter_batched_ref(
                || b_mat.clone(),
                |x| blas.dtrsm(black_box(&a_mat), black_box(x)),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark DSYRK: C = alpha * A * A^T + beta * C (symmetric rank-k update)
fn bench_syrk_comparison(c: &mut Criterion) {
    let blas = openblas::get();
    let mut group = c.benchmark_group("syrk_comparison");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        let k = *size;
        group.throughput(Throughput::Elements((n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i as f64) * 0.01).collect();
        let a_mat = Mat::from_slice(n, k, &a_data);
        let mut c_mat = Mat::zeros(n, n);

        // beta == 0 overwrites C without reading it, so C needs no reset.

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                let _ = syrk(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a_mat.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| blas.dsyrk(black_box(&a_mat), black_box(&mut c_mat)));
        });
    }

    group.finish();
}

criterion_group!(
    comparison_benches,
    bench_gemm_comparison,
    bench_gemm_rectangular_comparison,
    bench_sgemm_comparison,
    bench_gemm_parallel_comparison,
    bench_sgemm_parallel_comparison,
    bench_trsm_comparison,
    bench_syrk_comparison,
    bench_gemv_comparison,
    bench_dot_comparison,
    bench_axpy_comparison,
    bench_nrm2_comparison,
);

criterion_main!(comparison_benches);
