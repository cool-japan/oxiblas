// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SME2 GEMM micro-kernels for AArch64 (Apple M4 and later).
//!
//! The kernels use the ZA matrix storage of the Scalable Matrix Extension with
//! a 512-bit streaming vector length (SVL). A micro-tile of C is 64 x 16 (f32,
//! four 16 x 16 ZA tiles) or 64 x 8 (f64, eight 8 x 8 ZA tiles). Each K step
//! takes 64 elements of the packed A panel and NR elements of the packed B
//! panel, and adds their outer products with `FMOPA`, with B as the row
//! operand: horizontal slice `j` of ZA tile `t` holds rows `16t..16t + 16`
//! (f32) of column `j` of C, which is contiguous in column-major storage.
//!
//! The SME2 GEMM path (driven by `sme2_sequential` and `sme2_split` in
//! `level3/gemm.rs`) also has, in this module and its submodules:
//!
//! - macro-kernels that run all micro-tiles of a packed block in one
//!   streaming-mode region, read a column-major A in place (optionally writing
//!   a packed copy during the first column block), move C directly between
//!   memory and ZA when `alpha == 1` and `beta` is 0 or 1, prefetch the next C
//!   tile into L2 and skip the `FMOPA`s of ZA tiles past the last row
//!   (`macro_kernel.rs`);
//! - packers that transpose B, or a transposed A, through ZA four slices at a
//!   time, with a core-side L2 prefetch of the next B panel, and copy the rows
//!   of a transposed B (`pack.rs`);
//! - persistent worker threads, one per SME unit beyond the first, that spin
//!   between calls (`Workers`), and a per-thread scratch for packed panels
//!   (`with_scratch`).
//!
//! The design follows MTGEMM-A, an SME GEMM for the Apple M4 Pro
//! (<https://github.com/tesch1/mtgemm-a>).
//!
//! Rust has no SME intrinsics, so each kernel is one `asm!` block that enters
//! streaming mode (`smstart`), computes the whole micro-tile, writes C and
//! leaves streaming mode (`smstop`). Nothing runs between `smstart` and
//! `smstop` except the instructions in the block, so the compiler never sees
//! streaming mode. `smstart` and `smstop` zero the vector registers, so the
//! blocks declare all of `v0`-`v31` (and the predicates) as clobbered; this
//! includes the callee-saved `d8`-`d15`.

use std::sync::OnceLock;

mod macro_kernel;
mod pack;

pub(crate) use macro_kernel::*;
pub(crate) use pack::*;

/// Micro-tile rows (MR) of both kernels.
pub(crate) const MR: usize = 64;
/// Micro-tile columns (NR) of the f32 kernel.
pub(super) const NR_F32: usize = 16;
/// Micro-tile columns (NR) of the f64 kernel.
pub(super) const NR_F64: usize = 8;

/// Streaming vector length, in bytes, that the kernels are written for.
const SVL_BYTES: u64 = 64;

/// What the SME2 kernels can use on this CPU.
#[derive(Debug, Clone, Copy)]
struct Sme2Support {
    f32: bool,
    f64: bool,
}

/// Detects SME2 once per process (and `FEAT_SME_F64F64` for f64).
fn support() -> Sme2Support {
    static SUPPORT: OnceLock<Sme2Support> = OnceLock::new();
    *SUPPORT.get_or_init(detect)
}

/// Returns `true` when the f32 SME2 kernel can run on this CPU.
pub(super) fn usable_f32() -> bool {
    support().f32
}

/// Returns `true` when the f64 SME2 kernel can run on this CPU.
pub(super) fn usable_f64() -> bool {
    support().f64
}

#[cfg(target_os = "macos")]
fn detect() -> Sme2Support {
    let sme2 = sysctl_flag(c"hw.optional.arm.FEAT_SME2");
    // RDSVL is undefined without SME, so it runs only after the SME2 check.
    // SAFETY: SME2 (and hence SME) is implemented when `sme2` is true.
    let svl_ok = sme2 && unsafe { streaming_vector_bytes() } == SVL_BYTES;
    Sme2Support {
        f32: svl_ok,
        f64: svl_ok && sysctl_flag(c"hw.optional.arm.FEAT_SME_F64F64"),
    }
}

/// Untested outside macOS: other systems use the NEON kernels.
#[cfg(not(target_os = "macos"))]
fn detect() -> Sme2Support {
    Sme2Support {
        f32: false,
        f64: false,
    }
}

/// Reads an integer `sysctl` flag; `false` when it is missing or zero.
#[cfg(target_os = "macos")]
fn sysctl_flag(name: &core::ffi::CStr) -> bool {
    let mut value: libc::c_int = 0;
    let mut len: libc::size_t = core::mem::size_of::<libc::c_int>();
    // SAFETY: `name` is NUL-terminated, `value` and `len` describe a valid
    // `c_int` buffer, and no new value is written.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && value != 0
}

/// Smallest number of multiply-adds that runs on more than one SME unit, for
/// f32; f64 (a quarter of the f32 rate) gains from a quarter of that.
#[cfg(feature = "parallel")]
const PAR_MIN_WORK: usize = 3 << 20;

/// Persistent worker threads for the SME units beyond the first (the calling
/// thread uses one unit), as in MTGEMM-A: each worker spins on an atomic for
/// [`SPIN`] after a job before it sleeps, so back-to-back GEMM calls pay no
/// wake-up, and asks for the user-interactive QoS class, so that macOS runs it
/// on a performance core.
#[cfg(feature = "parallel")]
pub(crate) struct Workers {
    slots: Vec<std::sync::Arc<Slot>>,
    call: std::sync::Mutex<()>,
}

/// How long a worker spins for the next job before it sleeps.
#[cfg(feature = "parallel")]
const SPIN: std::time::Duration = std::time::Duration::from_micros(200);

/// One worker's mailbox. `job` is written only by the thread that holds
/// `Workers::call`, before it bumps `posted`, and read by the worker after it
/// sees the new `posted`; the caller waits for `done` before it returns.
#[cfg(feature = "parallel")]
struct Slot {
    job: core::cell::UnsafeCell<Option<*mut (dyn FnMut() + Send + 'static)>>,
    posted: std::sync::atomic::AtomicU64,
    done: std::sync::atomic::AtomicU64,
    panicked: std::sync::atomic::AtomicBool,
    sleeping: std::sync::atomic::AtomicBool,
    lock: std::sync::Mutex<()>,
    wake: std::sync::Condvar,
}

// SAFETY: `job` is only accessed under the posted/done handshake described on
// `Slot`, so the caller and the worker never touch it at the same time.
#[cfg(feature = "parallel")]
unsafe impl Sync for Slot {}
// SAFETY: as above; the raw job pointer is only dereferenced by the worker
// while the caller keeps the job alive.
#[cfg(feature = "parallel")]
unsafe impl Send for Slot {}

#[cfg(feature = "parallel")]
impl Slot {
    fn work(&self) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        set_interactive_qos();
        let mut seen = 0u64;
        loop {
            let t0 = std::time::Instant::now();
            while self.posted.load(Acquire) == seen && t0.elapsed() < SPIN {
                core::hint::spin_loop();
            }
            if self.posted.load(Acquire) == seen {
                let mut guard = self
                    .lock
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                self.sleeping.store(true, Release);
                while self.posted.load(Acquire) == seen {
                    guard = self
                        .wake
                        .wait(guard)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                self.sleeping.store(false, Release);
            }
            seen = self.posted.load(Acquire);
            // SAFETY: the caller wrote `job` before bumping `posted` and keeps
            // the job alive until it sees `done == seen`.
            if let Some(job) = unsafe { *self.job.get() } {
                let ok =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)() }));
                if ok.is_err() {
                    self.panicked.store(true, Release);
                }
            }
            self.done.store(seen, Release);
        }
    }
}

#[cfg(feature = "parallel")]
impl Workers {
    /// Number of worker threads.
    pub(crate) fn count(&self) -> usize {
        self.slots.len()
    }

    /// Runs `jobs[i]` on worker `i` and `main` on the calling thread, and
    /// returns when all have finished; `false` (nothing run) when another call
    /// holds the workers. Panics if a job panicked.
    pub(crate) fn run(
        &self,
        jobs: &mut [&mut (dyn FnMut() + Send)],
        main: &mut dyn FnMut(),
    ) -> bool {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let Ok(_call) = self.call.try_lock() else {
            return false;
        };
        let mut tickets = Vec::with_capacity(jobs.len());
        for (slot, job) in self.slots.iter().zip(jobs.iter_mut()) {
            let ptr: *mut (dyn FnMut() + Send + '_) = &mut **job;
            // SAFETY: only the lifetime is erased; this call waits below until
            // the worker has finished the job.
            let ptr: *mut (dyn FnMut() + Send + 'static) = unsafe { core::mem::transmute(ptr) };
            // SAFETY: the worker reads `job` only after it sees the new
            // `posted`, and it has finished the previous job (`done` was waited for).
            unsafe { *slot.job.get() = Some(ptr) };
            let ticket = {
                let _guard = slot
                    .lock
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let t = slot.posted.load(Acquire) + 1;
                slot.posted.store(t, Release);
                t
            };
            if slot.sleeping.load(Acquire) {
                slot.wake.notify_one();
            }
            tickets.push(ticket);
        }
        main();
        let mut panicked = false;
        for (slot, ticket) in self.slots.iter().zip(tickets) {
            while slot.done.load(Acquire) != ticket {
                core::hint::spin_loop();
            }
            // SAFETY: the worker has finished; clear the pointer to the job.
            unsafe { *slot.job.get() = None };
            panicked |= slot.panicked.swap(false, Acquire);
        }
        assert!(!panicked, "an SME2 worker job panicked");
        true
    }
}

/// Returns the SME2 worker threads (one less than there are SME units), when
/// there are at least two units and the problem has at least [`PAR_MIN_WORK`]
/// multiply-adds (f32; a quarter of that for 8-byte `elem_size`).
#[cfg(feature = "parallel")]
pub(crate) fn workers(work: usize, elem_size: usize) -> Option<&'static Workers> {
    static WORKERS: OnceLock<Option<Workers>> = OnceLock::new();
    let min_work = if elem_size == 8 {
        PAR_MIN_WORK / 4
    } else {
        PAR_MIN_WORK
    };
    if work < min_work || sme_units() < 2 {
        return None;
    }
    WORKERS
        .get_or_init(|| {
            let mut slots = Vec::new();
            for i in 1..sme_units() {
                let slot = std::sync::Arc::new(Slot {
                    job: core::cell::UnsafeCell::new(None),
                    posted: std::sync::atomic::AtomicU64::new(0),
                    done: std::sync::atomic::AtomicU64::new(0),
                    panicked: std::sync::atomic::AtomicBool::new(false),
                    sleeping: std::sync::atomic::AtomicBool::new(false),
                    lock: std::sync::Mutex::new(()),
                    wake: std::sync::Condvar::new(),
                });
                let worker = std::sync::Arc::clone(&slot);
                let spawned = std::thread::Builder::new()
                    .name(format!("oxiblas-sme2-{i}"))
                    .spawn(move || worker.work());
                if spawned.is_err() {
                    break;
                }
                slots.push(slot);
            }
            (!slots.is_empty()).then(|| Workers {
                slots,
                call: std::sync::Mutex::new(()),
            })
        })
        .as_ref()
}

/// Asks for the user-interactive QoS class for the calling thread.
#[cfg(all(feature = "parallel", target_os = "macos"))]
fn set_interactive_qos() {
    // SAFETY: the call only changes the scheduling class of this thread; a
    // failure leaves the thread at its default class, which is still correct.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(all(feature = "parallel", not(target_os = "macos")))]
fn set_interactive_qos() {}

/// Runs `f` on a per-thread scratch buffer of `len` elements of `T`, kept
/// between calls (as MTGEMM-A keeps its packing buffers): no allocation and
/// no zeroing per GEMM call. The contents are whatever an earlier call left;
/// `T` must be `f32` or `f64`, for which every bit pattern is a value.
pub(crate) fn with_scratch<T: Copy, R>(len: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    use oxiblas_core::memory::AlignedVec;
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<AlignedVec<u64>> = RefCell::new(AlignedVec::zeros(0));
    }
    assert!(
        matches!(core::mem::size_of::<T>(), 4 | 8) && core::mem::align_of::<T>() <= 8,
        "SME2 scratch is for f32 and f64"
    );
    let words = (len * core::mem::size_of::<T>()).div_ceil(8);
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < words {
                *buf = AlignedVec::zeros(words);
            }
            // SAFETY: the buffer holds at least `len` elements of `T` (4 or 8
            // bytes, alignment <= 8), and every bit pattern is an f32 or f64.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<T>(), len) };
            f(slice)
        }
        Err(_) => {
            let mut buf: AlignedVec<u64> = AlignedVec::zeros(words);
            // SAFETY: as above, for a fresh buffer.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<T>(), len) };
            f(slice)
        }
    })
}

/// Returns the number of performance-core clusters, i.e. of SME units on Apple
/// silicon (one per P-cluster); at least 1.
#[cfg(all(feature = "parallel", target_os = "macos"))]
pub(super) fn sme_units() -> usize {
    static UNITS: OnceLock<usize> = OnceLock::new();
    *UNITS.get_or_init(|| {
        let cores = sysctl_int(c"hw.perflevel0.physicalcpu");
        let per_l2 = sysctl_int(c"hw.perflevel0.cpusperl2");
        if cores > 0 && per_l2 > 0 {
            (cores / per_l2).max(1)
        } else {
            1
        }
    })
}

/// Returns the number of SME units; 1 outside macOS.
#[cfg(all(feature = "parallel", not(target_os = "macos")))]
pub(super) fn sme_units() -> usize {
    1
}

/// Reads a non-negative integer `sysctl` value; 0 when it is missing.
#[cfg(all(feature = "parallel", target_os = "macos"))]
fn sysctl_int(name: &core::ffi::CStr) -> usize {
    let mut value: libc::c_int = 0;
    let mut len: libc::size_t = core::mem::size_of::<libc::c_int>();
    // SAFETY: as in `sysctl_flag`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        usize::try_from(value).unwrap_or(0)
    } else {
        0
    }
}

/// Returns the streaming vector length in bytes.
///
/// # Safety
///
/// The CPU must implement SME (`RDSVL` is undefined otherwise).
#[cfg(target_os = "macos")]
unsafe fn streaming_vector_bytes() -> u64 {
    let svl: u64;
    // SAFETY: the caller guarantees SME; RDSVL reads a system value only.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme",
            "rdsvl {svl}, #1",
            svl = out(reg) svl,
            options(nomem, nostack, preserves_flags)
        );
    }
    svl
}

/// SME2 micro-kernel for f32: `C = alpha * A * B + beta * C` on a 64 x 16 tile.
///
/// `a` is a packed A panel (`k` steps of 64 elements), `b` a packed B panel
/// (`k` steps of 16 elements), and C is column-major with column stride
/// `c_stride`. With `beta == 0`, C is not read.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`.
/// - `a` must be valid for `64 * k` reads and `b` for `16 * k` reads.
/// - `c` must be valid for reads and writes of 64 elements in each of 16
///   columns `c_stride` apart.
pub(super) unsafe fn micro_kernel_f32_sme2(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    let ldc_bytes = c_stride * core::mem::size_of::<f32>();
    let beta_zero = u64::from(beta == 0.0);
    // SAFETY: the caller guarantees SME2, SVL 512 and the buffer sizes above;
    // the block enters and leaves streaming mode itself.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "zero {{za}}",
            "ptrue p0.s",
            "ptrue pn8.s",
            // K loop, four steps per iteration: one x4 load of B gives 4 steps.
            "lsr x9, x0, #2",
            "cbz x9, 2f",
            "1:",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x2]",
            "ld1w {{z0.s-z3.s}}, pn8/z, [x1]",
            "ld1w {{z16.s-z19.s}}, pn8/z, [x1, #4, mul vl]",
            "ld1w {{z20.s-z23.s}}, pn8/z, [x1, #8, mul vl]",
            "ld1w {{z24.s-z27.s}}, pn8/z, [x1, #12, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z4.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z4.s, z3.s",
            "fmopa za0.s, p0/m, p0/m, z5.s, z16.s",
            "fmopa za1.s, p0/m, p0/m, z5.s, z17.s",
            "fmopa za2.s, p0/m, p0/m, z5.s, z18.s",
            "fmopa za3.s, p0/m, p0/m, z5.s, z19.s",
            "fmopa za0.s, p0/m, p0/m, z6.s, z20.s",
            "fmopa za1.s, p0/m, p0/m, z6.s, z21.s",
            "fmopa za2.s, p0/m, p0/m, z6.s, z22.s",
            "fmopa za3.s, p0/m, p0/m, z6.s, z23.s",
            "fmopa za0.s, p0/m, p0/m, z7.s, z24.s",
            "fmopa za1.s, p0/m, p0/m, z7.s, z25.s",
            "fmopa za2.s, p0/m, p0/m, z7.s, z26.s",
            "fmopa za3.s, p0/m, p0/m, z7.s, z27.s",
            "addvl x1, x1, #16",
            "addvl x2, x2, #4",
            "subs x9, x9, #1",
            "b.ne 1b",
            "2:",
            // Remaining 0-3 K steps.
            "ands x9, x0, #3",
            "b.eq 4f",
            "3:",
            "ld1w {{z4.s}}, p0/z, [x2]",
            "ld1w {{z0.s-z3.s}}, pn8/z, [x1]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z4.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z4.s, z3.s",
            "addvl x1, x1, #4",
            "addvl x2, x2, #1",
            "subs x9, x9, #1",
            "b.ne 3b",
            "4:",
            // C: column j = slice j of the four tiles, 64 contiguous elements.
            "dup z30.s, w4",
            "dup z31.s, w5",
            "mov w12, #0",
            "5:",
            "mova z0.s, p0/m, za0h.s[w12, 0]",
            "mova z1.s, p0/m, za1h.s[w12, 0]",
            "mova z2.s, p0/m, za2h.s[w12, 0]",
            "mova z3.s, p0/m, za3h.s[w12, 0]",
            "fmul z0.s, z0.s, z30.s",
            "fmul z1.s, z1.s, z30.s",
            "fmul z2.s, z2.s, z30.s",
            "fmul z3.s, z3.s, z30.s",
            "cbnz x6, 6f",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x3]",
            "fmla z0.s, p0/m, z4.s, z31.s",
            "fmla z1.s, p0/m, z5.s, z31.s",
            "fmla z2.s, p0/m, z6.s, z31.s",
            "fmla z3.s, p0/m, z7.s, z31.s",
            "6:",
            "st1w {{z0.s-z3.s}}, pn8, [x3]",
            "add x3, x3, x7",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 5b",
            "smstop",
            inout("x0") k => _,
            inout("x1") a => _,
            inout("x2") b => _,
            inout("x3") c => _,
            in("x4") u64::from(alpha.to_bits()),
            in("x5") u64::from(beta.to_bits()),
            in("x6") beta_zero,
            in("x7") ldc_bytes,
            out("x9") _,
            out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _,
            options(nostack)
        );
    }
}

/// SME2 micro-kernel for f64: `C = alpha * A * B + beta * C` on a 64 x 8 tile.
///
/// `a` is a packed A panel (`k` steps of 64 elements), `b` a packed B panel
/// (`k` steps of 8 elements), and C is column-major with column stride
/// `c_stride`. With `beta == 0`, C is not read.
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`.
/// - `a` must be valid for `64 * k` reads and `b` for `8 * k` reads.
/// - `c` must be valid for reads and writes of 64 elements in each of 8
///   columns `c_stride` apart.
pub(super) unsafe fn micro_kernel_f64_sme2(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    let ldc_bytes = c_stride * core::mem::size_of::<f64>();
    let beta_zero = u64::from(beta == 0.0);
    // SAFETY: the caller guarantees SME2, FEAT_SME_F64F64, SVL 512 and the
    // buffer sizes above; the block enters and leaves streaming mode itself.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            ".arch_extension sme-f64f64",
            "smstart",
            "zero {{za}}",
            "ptrue p0.d",
            "ptrue pn8.d",
            // K loop, four steps per iteration: one x4 load of B gives 4 steps.
            "lsr x9, x0, #2",
            "cbz x9, 2f",
            "1:",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x2]",
            "ld1d {{z0.d-z3.d}}, pn8/z, [x1]",
            "ld1d {{z16.d-z19.d}}, pn8/z, [x1, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn8/z, [x1, #8, mul vl]",
            "ld1d {{z24.d-z27.d}}, pn8/z, [x1, #12, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "ld1d {{z0.d-z3.d}}, pn8/z, [x1, #16, mul vl]",
            "ld1d {{z16.d-z19.d}}, pn8/z, [x1, #20, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z5.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z5.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z5.d, z23.d",
            "fmopa za4.d, p0/m, p0/m, z5.d, z24.d",
            "fmopa za5.d, p0/m, p0/m, z5.d, z25.d",
            "fmopa za6.d, p0/m, p0/m, z5.d, z26.d",
            "fmopa za7.d, p0/m, p0/m, z5.d, z27.d",
            "ld1d {{z20.d-z23.d}}, pn8/z, [x1, #24, mul vl]",
            "ld1d {{z24.d-z27.d}}, pn8/z, [x1, #28, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z6.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z6.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z6.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z6.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z6.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z6.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z6.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z6.d, z19.d",
            "fmopa za0.d, p0/m, p0/m, z7.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z7.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z7.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z7.d, z23.d",
            "fmopa za4.d, p0/m, p0/m, z7.d, z24.d",
            "fmopa za5.d, p0/m, p0/m, z7.d, z25.d",
            "fmopa za6.d, p0/m, p0/m, z7.d, z26.d",
            "fmopa za7.d, p0/m, p0/m, z7.d, z27.d",
            "addvl x1, x1, #16",
            "addvl x1, x1, #16",
            "addvl x2, x2, #4",
            "subs x9, x9, #1",
            "b.ne 1b",
            "2:",
            // Remaining 0-3 K steps.
            "ands x9, x0, #3",
            "b.eq 4f",
            "3:",
            "ld1d {{z4.d}}, p0/z, [x2]",
            "ld1d {{z0.d-z3.d}}, pn8/z, [x1]",
            "ld1d {{z16.d-z19.d}}, pn8/z, [x1, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "addvl x1, x1, #8",
            "addvl x2, x2, #1",
            "subs x9, x9, #1",
            "b.ne 3b",
            "4:",
            // C: column j = slice j of the eight tiles, 64 contiguous elements.
            "dup z30.d, x4",
            "dup z31.d, x5",
            "mov w12, #0",
            "5:",
            "mova z0.d, p0/m, za0h.d[w12, 0]",
            "mova z1.d, p0/m, za1h.d[w12, 0]",
            "mova z2.d, p0/m, za2h.d[w12, 0]",
            "mova z3.d, p0/m, za3h.d[w12, 0]",
            "mova z16.d, p0/m, za4h.d[w12, 0]",
            "mova z17.d, p0/m, za5h.d[w12, 0]",
            "mova z18.d, p0/m, za6h.d[w12, 0]",
            "mova z19.d, p0/m, za7h.d[w12, 0]",
            "fmul z0.d, z0.d, z30.d",
            "fmul z1.d, z1.d, z30.d",
            "fmul z2.d, z2.d, z30.d",
            "fmul z3.d, z3.d, z30.d",
            "fmul z16.d, z16.d, z30.d",
            "fmul z17.d, z17.d, z30.d",
            "fmul z18.d, z18.d, z30.d",
            "fmul z19.d, z19.d, z30.d",
            "cbnz x6, 6f",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x3]",
            "ld1d {{z20.d-z23.d}}, pn8/z, [x3, #4, mul vl]",
            "fmla z0.d, p0/m, z4.d, z31.d",
            "fmla z1.d, p0/m, z5.d, z31.d",
            "fmla z2.d, p0/m, z6.d, z31.d",
            "fmla z3.d, p0/m, z7.d, z31.d",
            "fmla z16.d, p0/m, z20.d, z31.d",
            "fmla z17.d, p0/m, z21.d, z31.d",
            "fmla z18.d, p0/m, z22.d, z31.d",
            "fmla z19.d, p0/m, z23.d, z31.d",
            "6:",
            "st1d {{z0.d-z3.d}}, pn8, [x3]",
            "st1d {{z16.d-z19.d}}, pn8, [x3, #4, mul vl]",
            "add x3, x3, x7",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 5b",
            "smstop",
            inout("x0") k => _,
            inout("x1") a => _,
            inout("x2") b => _,
            inout("x3") c => _,
            in("x4") alpha.to_bits(),
            in("x5") beta.to_bits(),
            in("x6") beta_zero,
            in("x7") ldc_bytes,
            out("x9") _,
            out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _,
            options(nostack)
        );
    }
}

#[cfg(test)]
mod tests;
