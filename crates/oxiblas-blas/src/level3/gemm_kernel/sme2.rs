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
//! This is the layout of the packed SME2 kernels in OpenBLAS (pull request
//! OpenMathLib/OpenBLAS#6074, `KERNEL.VORTEXM4`).
//!
//! Rust has no SME intrinsics, so each kernel is one `asm!` block that enters
//! streaming mode (`smstart`), computes the whole micro-tile, writes C and
//! leaves streaming mode (`smstop`). Nothing runs between `smstart` and
//! `smstop` except the instructions in the block, so the compiler never sees
//! streaming mode. `smstart` and `smstop` zero the vector registers, so the
//! blocks declare all of `v0`-`v31` (and the predicates) as clobbered; this
//! includes the callee-saved `d8`-`d15`.

use std::sync::OnceLock;

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

/// SME2 macro-kernel for f32: `C = alpha * A * B + beta * C` on an `m x n`
/// block, in one streaming-mode region.
///
/// A panel `i` (rows `64i..`) starts at `a + i * a_panel_stride`, and its
/// depth steps are `a_step` elements apart (packed panels: `64 * k` and 64;
/// a column-major A read in place: 64 and its column stride). `b` holds
/// `ceil(n / 16)` packed B panels of `16 * k` elements, in the layout of
/// [`micro_kernel_f32_sme2`]. A loads and C stores are predicated to the
/// block, so rows past `m` are not read and columns of the last B panel past
/// `n` may hold any values. C is column-major with column stride `c_stride`.
/// With `beta == 0`, C is not read.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`, and `m`, `n` > 0.
/// - `a` must be valid for reads of rows `0..m` of each depth step of each
///   panel, and `b` for the B panel reads above.
/// - `c` must be valid for reads and writes of `m` elements in each of `n`
///   columns `c_stride` apart.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn macro_kernel_f32_sme2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_panel_stride: usize,
    a_step: usize,
    a_pack: *mut f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    let size = core::mem::size_of::<f32>();
    let ldc_bytes = c_stride * size;
    let a_panel_bytes = a_panel_stride * size;
    let a_step_bytes = a_step * size;
    let b_panel_bytes = NR_F32 * k * size;
    // Bit 0: beta == 0 (C is not read); bit 1: alpha == 1 and beta 0 or 1, so
    // C is loaded into ZA before the depth loop and ZA is stored as it is.
    let direct = alpha == 1.0 && (beta == 0.0 || beta == 1.0);
    let mode = u64::from(beta == 0.0) | (u64::from(direct) << 1);
    // SAFETY: the caller guarantees SME2, SVL 512, m, n > 0 and the buffer
    // sizes above; the block enters and leaves streaming mode itself.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "dup z30.s, w9",
            "dup z31.s, w5",
            "mov x14, #0",
            // Column blocks of 16: x14 = j0, x2 = B panel, x3 = C column j0.
            "2:",
            "mov x15, #0",
            "mov x16, x1",
            "mov x21, x3",
            // Row blocks of 64: x15 = i0, x16 = A panel, x21 = C(i0, j0).
            "3:",
            "zero {{za}}",
            // Core-side L2 prefetch of the next C tile in this column block (rows i0 + 64.., 16 columns).
            "add x28, x15, #64",
            "cmp x28, x8",
            "b.hs 13f",
            "add x28, x21, #256",
            "mov x9, #16",
            "12:",
            "prfm pldl2keep, [x28]",
            "prfm pldl2keep, [x28, #128]",
            "add x28, x28, x7",
            "subs x9, x9, #1",
            "b.ne 12b",
            "13:",
            // Rows of this tile: A loads and C stores are predicated to m - i0.
            "sub x24, x8, x15",
            "mov x25, #0",
            "whilelt pn9.s, x25, x24, vlx4",
            // Direct mode (alpha == 1, beta 0 or 1): C moves between memory and ZA slices.
            "tbz x6, #1, 14f",
            "whilelt p1.s, x25, x24",
            "mov x27, #16",
            "whilelt p2.s, x27, x24",
            "mov x27, #32",
            "whilelt p3.s, x27, x24",
            "mov x27, #48",
            "whilelt p4.s, x27, x24",
            "tbnz x6, #0, 14f",
            "sub x26, x10, x14",
            "mov x27, #16",
            "cmp x26, x27",
            "csel x26, x26, x27, lo",
            "mov w12, #0",
            "mov x27, x21",
            "15:",
            "ld1w {{za0h.s[w12, 0]}}, p1/z, [x27]",
            "add x9, x27, #64",
            "ld1w {{za1h.s[w12, 0]}}, p2/z, [x9]",
            "add x20, x27, #128",
            "ld1w {{za2h.s[w12, 0]}}, p3/z, [x20]",
            "add x28, x27, #192",
            "ld1w {{za3h.s[w12, 0]}}, p4/z, [x28]",
            "add x27, x27, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 15b",
            "14:",
            "mov x22, x16",
            "mov x23, x2",
            "lsr x9, x0, #2",
            // First column block with a pack buffer (x4): read A, keep a packed copy.
            "cbnz x14, 29f",
            "cbz x4, 29f",
            "mul x5, x15, x0",
            "add x5, x4, x5, lsl #2",
            "cbz x9, 31f",
            "30:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "st1w {{z0.s-z3.s}}, pn8, [x5, #0, mul vl]",
            "ld1w {{z16.s-z19.s}}, pn9/z, [x24]",
            "st1w {{z16.s-z19.s}}, pn8, [x5, #4, mul vl]",
            "ld1w {{z20.s-z23.s}}, pn9/z, [x25]",
            "st1w {{z20.s-z23.s}}, pn8, [x5, #8, mul vl]",
            "ld1w {{z24.s-z27.s}}, pn9/z, [x26]",
            "st1w {{z24.s-z27.s}}, pn8, [x5, #12, mul vl]",
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
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "addvl x5, x5, #16",
            "subs x9, x9, #1",
            "b.ne 30b",

            "31:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "32:",
            "ld1w {{z4.s}}, p0/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "st1w {{z0.s-z3.s}}, pn8, [x5, #0, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z4.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z4.s, z3.s",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "addvl x5, x5, #4",
            "subs x9, x9, #1",
            "b.ne 32b",

            "b 7f",
            "29:",
            "cmp x24, #16",
            "b.ls 53f",
            "cmp x24, #32",
            "b.ls 63f",
            "cbz x9, 5f",
            "4:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "ld1w {{z16.s-z19.s}}, pn9/z, [x24]",
            "ld1w {{z20.s-z23.s}}, pn9/z, [x25]",
            "ld1w {{z24.s-z27.s}}, pn9/z, [x26]",
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
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 4b",
            "5:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "6:",
            "ld1w {{z4.s}}, p0/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z4.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z4.s, z3.s",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 6b",
            "b 7f",
            // At most 16 rows: only the first 1 ZA tile(s) carry data.
            "53:",
            "cbz x9, 51f",
            "50:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "ld1w {{z16.s-z19.s}}, pn9/z, [x24]",
            "ld1w {{z20.s-z23.s}}, pn9/z, [x25]",
            "ld1w {{z24.s-z27.s}}, pn9/z, [x26]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za0.s, p0/m, p0/m, z5.s, z16.s",
            "fmopa za0.s, p0/m, p0/m, z6.s, z20.s",
            "fmopa za0.s, p0/m, p0/m, z7.s, z24.s",
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 50b",

            "51:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "52:",
            "ld1w {{z4.s}}, p0/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 52b",

            "b 7f",
            // At most 32 rows: only the first 2 ZA tile(s) carry data.
            "63:",
            "cbz x9, 61f",
            "60:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1w {{z4.s-z7.s}}, pn8/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "ld1w {{z16.s-z19.s}}, pn9/z, [x24]",
            "ld1w {{z20.s-z23.s}}, pn9/z, [x25]",
            "ld1w {{z24.s-z27.s}}, pn9/z, [x26]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "fmopa za0.s, p0/m, p0/m, z5.s, z16.s",
            "fmopa za1.s, p0/m, p0/m, z5.s, z17.s",
            "fmopa za0.s, p0/m, p0/m, z6.s, z20.s",
            "fmopa za1.s, p0/m, p0/m, z6.s, z21.s",
            "fmopa za0.s, p0/m, p0/m, z7.s, z24.s",
            "fmopa za1.s, p0/m, p0/m, z7.s, z25.s",
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 60b",

            "61:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "62:",
            "ld1w {{z4.s}}, p0/z, [x23]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x22]",
            "fmopa za0.s, p0/m, p0/m, z4.s, z0.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z1.s",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 62b",

            "b 7f",
            "7:",
            // Store: rows predicated to m - i0, columns min(16, n - j0).
            "sub x24, x8, x15",
            "mov x25, #0",
            "whilelt pn9.s, x25, x24, vlx4",
            "sub x26, x10, x14",
            "mov x27, #16",
            "cmp x26, x27",
            "csel x26, x26, x27, lo",
            "mov w12, #0",
            "mov x25, x21",
            "tbnz x6, #1, 16f",
            "8:",
            "mova z0.s, p0/m, za0h.s[w12, 0]",
            "mova z1.s, p0/m, za1h.s[w12, 0]",
            "mova z2.s, p0/m, za2h.s[w12, 0]",
            "mova z3.s, p0/m, za3h.s[w12, 0]",
            "fmul z0.s, z0.s, z30.s",
            "fmul z1.s, z1.s, z30.s",
            "fmul z2.s, z2.s, z30.s",
            "fmul z3.s, z3.s, z30.s",
            "tbnz x6, #0, 9f",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x25]",
            "fmla z0.s, p0/m, z4.s, z31.s",
            "fmla z1.s, p0/m, z5.s, z31.s",
            "fmla z2.s, p0/m, z6.s, z31.s",
            "fmla z3.s, p0/m, z7.s, z31.s",
            "9:",
            "st1w {{z0.s-z3.s}}, pn9, [x25]",
            "add x25, x25, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 8b",
            "b 18f",
            "16:",
            "st1w {{za0h.s[w12, 0]}}, p1, [x25]",
            "add x9, x25, #64",
            "st1w {{za1h.s[w12, 0]}}, p2, [x9]",
            "add x20, x25, #128",
            "st1w {{za2h.s[w12, 0]}}, p3, [x20]",
            "add x28, x25, #192",
            "st1w {{za3h.s[w12, 0]}}, p4, [x28]",
            "add x25, x25, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 16b",
            "18:",
            "add x15, x15, #64",
            "add x16, x16, x11",
            "add x21, x21, #256",
            "cmp x15, x8",
            "b.lo 3b",
            // After the first column block, read A from the packed copy.
            "cbnz x14, 33f",
            "cbz x4, 33f",
            "mov x1, x4",
            "lsl x11, x0, #8",
            "mov x17, #256",
            "33:",
            "add x14, x14, #16",
            "add x2, x2, x13",
            "add x3, x3, x7, lsl #4",
            "cmp x14, x10",
            "b.lo 2b",
            "smstop",
            in("x0") k,
            inout("x1") a => _,
            inout("x2") b => _,
            inout("x3") c => _,
            inout("x4") a_pack => _,
            inout("x9") u64::from(alpha.to_bits()) => _,
            inout("x5") u64::from(beta.to_bits()) => _,
            in("x6") mode,
            in("x7") ldc_bytes,
            in("x8") m,
            in("x10") n,
            inout("x11") a_panel_bytes => _,
            in("x13") b_panel_bytes,
            inout("x17") a_step_bytes => _,
            out("x12") _, out("x14") _, out("x15") _, out("x16") _,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _, out("x24") _, out("x25") _,
            out("x26") _, out("x27") _, out("x28") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
            out("p8") _, out("p9") _,
            options(nostack)
        );
    }
}

/// SME2 macro-kernel for f64: `C = alpha * A * B + beta * C` on an `m x n`
/// block, in one streaming-mode region.
///
/// As [`macro_kernel_f32_sme2`], with packed B panels of `8 * k` elements
/// and the layout of [`micro_kernel_f64_sme2`].
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`, and `m`, `n` > 0.
/// - `a` must be valid for reads of rows `0..m` of each depth step of each
///   panel, and `b` for `ceil(n / 8) * 8 * k` reads.
/// - `c` must be valid for reads and writes of `m` elements in each of `n`
///   columns `c_stride` apart.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn macro_kernel_f64_sme2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_panel_stride: usize,
    a_step: usize,
    a_pack: *mut f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    let size = core::mem::size_of::<f64>();
    let ldc_bytes = c_stride * size;
    let a_panel_bytes = a_panel_stride * size;
    let a_step_bytes = a_step * size;
    let b_panel_bytes = NR_F64 * k * size;
    // Bit 0: beta == 0 (C is not read); bit 1: alpha == 1 and beta 0 or 1, so
    // a full tile loads C into ZA before the depth loop and stores ZA as it is.
    let direct = alpha == 1.0 && (beta == 0.0 || beta == 1.0);
    let mode = u64::from(beta == 0.0) | (u64::from(direct) << 1);
    // SAFETY: the caller guarantees SME2, FEAT_SME_F64F64, SVL 512, m, n > 0
    // and the buffer sizes above; the block enters and leaves streaming mode
    // itself.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            ".arch_extension sme-f64f64",
            "smstart",
            "ptrue p0.d",
            "ptrue pn8.d",
            "dup z30.d, x9",
            "dup z31.d, x5",
            "mov x14, #0",
            // Column blocks of 8: x14 = j0, x2 = B panel, x3 = C column j0.
            "2:",
            "mov x15, #0",
            "mov x16, x1",
            "mov x21, x3",
            // Row blocks of 64: x15 = i0, x16 = A panel, x21 = C(i0, j0).
            "3:",
            "zero {{za}}",
            // Core-side L2 prefetch of the next C tile in this column block (rows i0 + 64.., 8 columns).
            "add x28, x15, #64",
            "cmp x28, x8",
            "b.hs 13f",
            "add x28, x21, #512",
            "mov x9, #8",
            "12:",
            "prfm pldl2keep, [x28]",
            "prfm pldl2keep, [x28, #128]",
            "prfm pldl2keep, [x28, #256]",
            "prfm pldl2keep, [x28, #384]",
            "add x28, x28, x7",
            "subs x9, x9, #1",
            "b.ne 12b",
            "13:",
            // Rows of this tile: A loads and C stores are predicated to m - i0.
            "sub x24, x8, x15",
            "mov x25, #0",
            "whilelt pn9.d, x25, x24, vlx4",
            "mov x25, #32",
            "whilelt pn10.d, x25, x24, vlx4",
            // Direct mode (alpha == 1, beta 0 or 1, full 64-row tile): C moves between memory and ZA slices.
            "mov x20, x6",
            "cmp x24, #64",
            "b.hs 19f",
            "and x20, x20, #1",
            "19:",
            "tbz x20, #1, 14f",
            "tbnz x20, #0, 14f",
            "sub x26, x10, x14",
            "mov x27, #8",
            "cmp x26, x27",
            "csel x26, x26, x27, lo",
            "mov w12, #0",
            "mov x27, x21",
            "15:",
            "ld1d {{za0h.d[w12, 0]}}, p0/z, [x27]",
            "add x9, x27, #64",
            "ld1d {{za1h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #128",
            "ld1d {{za2h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #192",
            "ld1d {{za3h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #256",
            "ld1d {{za4h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #320",
            "ld1d {{za5h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #384",
            "ld1d {{za6h.d[w12, 0]}}, p0/z, [x9]",
            "add x9, x27, #448",
            "ld1d {{za7h.d[w12, 0]}}, p0/z, [x9]",
            "add x27, x27, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 15b",
            "14:",
            "mov x22, x16",
            "mov x23, x2",
            "lsr x9, x0, #2",
            // First column block with a pack buffer (x4): read A, keep a packed copy.
            "cbnz x14, 29f",
            "cbz x4, 29f",
            "mul x5, x15, x0",
            "add x5, x4, x5, lsl #3",
            "cbz x9, 31f",
            "30:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "st1d {{z0.d-z3.d}}, pn8, [x5, #0, mul vl]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "st1d {{z16.d-z19.d}}, pn8, [x5, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x24]",
            "st1d {{z20.d-z23.d}}, pn8, [x5, #8, mul vl]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x24, #4, mul vl]",
            "st1d {{z24.d-z27.d}}, pn8, [x5, #12, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x25]",
            "st1d {{z0.d-z3.d}}, pn8, [x5, #16, mul vl]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x25, #4, mul vl]",
            "st1d {{z16.d-z19.d}}, pn8, [x5, #20, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z5.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z5.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z5.d, z23.d",
            "fmopa za4.d, p0/m, p0/m, z5.d, z24.d",
            "fmopa za5.d, p0/m, p0/m, z5.d, z25.d",
            "fmopa za6.d, p0/m, p0/m, z5.d, z26.d",
            "fmopa za7.d, p0/m, p0/m, z5.d, z27.d",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x26]",
            "st1d {{z20.d-z23.d}}, pn8, [x5, #24, mul vl]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x26, #4, mul vl]",
            "st1d {{z24.d-z27.d}}, pn8, [x5, #28, mul vl]",
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
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "addvl x5, x5, #16",
            "addvl x5, x5, #16",
            "subs x9, x9, #1",
            "b.ne 30b",

            "31:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "32:",
            "ld1d {{z4.d}}, p0/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "st1d {{z0.d-z3.d}}, pn8, [x5, #0, mul vl]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "st1d {{z16.d-z19.d}}, pn8, [x5, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "addvl x5, x5, #8",
            "subs x9, x9, #1",
            "b.ne 32b",

            "b 7f",
            "29:",
            "cmp x24, #8",
            "b.ls 53f",
            "cmp x24, #16",
            "b.ls 63f",
            "cmp x24, #32",
            "b.ls 73f",
            "cbz x9, 5f",
            "4:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x24]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x24, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x25]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x25, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z5.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z5.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z5.d, z23.d",
            "fmopa za4.d, p0/m, p0/m, z5.d, z24.d",
            "fmopa za5.d, p0/m, p0/m, z5.d, z25.d",
            "fmopa za6.d, p0/m, p0/m, z5.d, z26.d",
            "fmopa za7.d, p0/m, p0/m, z5.d, z27.d",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x26]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x26, #4, mul vl]",
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
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 4b",
            "5:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "6:",
            "ld1d {{z4.d}}, p0/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "fmopa za4.d, p0/m, p0/m, z4.d, z16.d",
            "fmopa za5.d, p0/m, p0/m, z4.d, z17.d",
            "fmopa za6.d, p0/m, p0/m, z4.d, z18.d",
            "fmopa za7.d, p0/m, p0/m, z4.d, z19.d",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 6b",
            "b 7f",
            // At most 8 rows: only the first 1 ZA tile(s) carry data.
            "53:",
            "cbz x9, 51f",
            "50:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x24]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x24, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x25]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x25, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x26]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x26, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z6.d, z0.d",
            "fmopa za0.d, p0/m, p0/m, z7.d, z20.d",
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 50b",

            "51:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "52:",
            "ld1d {{z4.d}}, p0/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 52b",

            "b 7f",
            // At most 16 rows: only the first 2 ZA tile(s) carry data.
            "63:",
            "cbz x9, 61f",
            "60:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x24]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x24, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x25]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x25, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z5.d, z21.d",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x26]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x26, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z6.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z6.d, z1.d",
            "fmopa za0.d, p0/m, p0/m, z7.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z7.d, z21.d",
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 60b",

            "61:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "62:",
            "ld1d {{z4.d}}, p0/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 62b",

            "b 7f",
            // At most 32 rows: only the first 4 ZA tile(s) carry data.
            "73:",
            "cbz x9, 71f",
            "70:",
            "add x24, x22, x17",
            "add x25, x24, x17",
            "add x26, x25, x17",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x24]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x24, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x25]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x25, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z5.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z5.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z5.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z5.d, z23.d",
            "ld1d {{z20.d-z23.d}}, pn9/z, [x26]",
            "ld1d {{z24.d-z27.d}}, pn10/z, [x26, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z6.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z6.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z6.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z6.d, z3.d",
            "fmopa za0.d, p0/m, p0/m, z7.d, z20.d",
            "fmopa za1.d, p0/m, p0/m, z7.d, z21.d",
            "fmopa za2.d, p0/m, p0/m, z7.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z7.d, z23.d",
            "add x22, x26, x17",
            "addvl x23, x23, #4",
            "subs x9, x9, #1",
            "b.ne 70b",

            "71:",
            "ands x9, x0, #3",
            "b.eq 7f",
            "72:",
            "ld1d {{z4.d}}, p0/z, [x23]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x22]",
            "ld1d {{z16.d-z19.d}}, pn10/z, [x22, #4, mul vl]",
            "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
            "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
            "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
            "add x22, x22, x17",
            "addvl x23, x23, #1",
            "subs x9, x9, #1",
            "b.ne 72b",

            "b 7f",
            "7:",
            // Store: rows predicated to m - i0 (two halves of 32), columns min(8, n - j0).
            "sub x24, x8, x15",
            "mov x25, #0",
            "whilelt pn9.d, x25, x24, vlx4",
            "mov x25, #32",
            "whilelt pn10.d, x25, x24, vlx4",
            "sub x26, x10, x14",
            "mov x27, #8",
            "cmp x26, x27",
            "csel x26, x26, x27, lo",
            "mov w12, #0",
            "mov x25, x21",
            "tbnz x20, #1, 16f",
            "8:",
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
            "tbnz x20, #0, 9f",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x25]",
            "ld1d {{z20.d-z23.d}}, pn10/z, [x25, #4, mul vl]",
            "fmla z0.d, p0/m, z4.d, z31.d",
            "fmla z1.d, p0/m, z5.d, z31.d",
            "fmla z2.d, p0/m, z6.d, z31.d",
            "fmla z3.d, p0/m, z7.d, z31.d",
            "fmla z16.d, p0/m, z20.d, z31.d",
            "fmla z17.d, p0/m, z21.d, z31.d",
            "fmla z18.d, p0/m, z22.d, z31.d",
            "fmla z19.d, p0/m, z23.d, z31.d",
            "9:",
            "st1d {{z0.d-z3.d}}, pn9, [x25]",
            "st1d {{z16.d-z19.d}}, pn10, [x25, #4, mul vl]",
            "add x25, x25, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 8b",
            "b 18f",
            "16:",
            "st1d {{za0h.d[w12, 0]}}, p0, [x25]",
            "add x9, x25, #64",
            "st1d {{za1h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #128",
            "st1d {{za2h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #192",
            "st1d {{za3h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #256",
            "st1d {{za4h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #320",
            "st1d {{za5h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #384",
            "st1d {{za6h.d[w12, 0]}}, p0, [x9]",
            "add x9, x25, #448",
            "st1d {{za7h.d[w12, 0]}}, p0, [x9]",
            "add x25, x25, x7",
            "add w12, w12, #1",
            "cmp w12, w26",
            "b.ne 16b",
            "18:",
            "add x15, x15, #64",
            "add x16, x16, x11",
            "add x21, x21, #512",
            "cmp x15, x8",
            "b.lo 3b",
            // After the first column block, read A from the packed copy.
            "cbnz x14, 33f",
            "cbz x4, 33f",
            "mov x1, x4",
            "lsl x11, x0, #9",
            "mov x17, #512",
            "33:",
            "add x14, x14, #8",
            "add x2, x2, x13",
            "add x3, x3, x7, lsl #3",
            "cmp x14, x10",
            "b.lo 2b",
            "smstop",
            in("x0") k,
            inout("x1") a => _,
            inout("x2") b => _,
            inout("x3") c => _,
            inout("x4") a_pack => _,
            inout("x9") alpha.to_bits() => _,
            inout("x5") beta.to_bits() => _,
            in("x6") mode,
            in("x7") ldc_bytes,
            in("x8") m,
            in("x10") n,
            inout("x11") a_panel_bytes => _,
            in("x13") b_panel_bytes,
            inout("x17") a_step_bytes => _,
            out("x12") _, out("x14") _, out("x15") _, out("x16") _,
            out("x20") _, out("x21") _, out("x22") _, out("x23") _, out("x24") _, out("x25") _,
            out("x26") _, out("x27") _, out("x28") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p8") _, out("p9") _, out("p10") _,
            options(nostack)
        );
    }
}

/// Packs `k` rows of `n` columns of a column-major B (column stride `ldb`)
/// into B panels of `NR` columns, `NR` elements per row (the layout the
/// kernels read), zero-padding the last panel to `NR` columns.
///
/// Full 64-row chunks of full panels are transposed through the ZA array:
/// four columns are loaded with four-vector loads and moved into four ZA
/// tiles with one four-slice `MOVA` per tile, and the vertical slices (the
/// packed rows) come out four at a time. This is the four-row ZA transposition
/// of MTGEMM-A's packing of A, applied to B, which is the transposed operand in
/// column-major order. The rest is copied element by element.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `n` columns
///   `ldb` apart, and `dst` for writes of `ceil(n / 16) * 16 * k` elements.
pub(super) unsafe fn pack_b_f32_sme2(
    src: *const f32,
    ldb: usize,
    k: usize,
    n: usize,
    dst: *mut f32,
) {
    let nr = NR_F32;
    let chunks = k / 64;
    for j0 in (0..n).step_by(nr) {
        let jb = nr.min(n - j0);
        // SAFETY: panel j0 / nr starts inside `dst`; columns j0.. inside `src`.
        let (panel, col) = unsafe { (dst.add(j0 * k), src.add(j0 * ldb)) };
        let mut p0 = 0;
        if jb == nr && chunks > 0 {
            // SAFETY: the caller guarantees SME2; the block reads 64 * chunks
            // rows of 16 columns and writes the same number of packed rows.
            unsafe {
                // Prefetch the next panel only while one follows (zero: none).
                let next = if j0 + 2 * nr <= n { nr * ldb * 4 } else { 0 };
                transpose_chunks_f32(col, ldb * 4, panel, chunks, next)
            };
            p0 = chunks * 64;
        }
        if jb == nr && p0 < k {
            // SAFETY: rows p0..k (fewer than 64) of the 16 columns, and their
            // packed rows inside panel j0 / nr.
            unsafe { transpose_tail_f32(col.add(p0), ldb * 4, panel.add(p0 * nr), k - p0) };
            p0 = k;
        }
        for p in p0..k {
            for jj in 0..nr {
                // SAFETY: p < k and j0 + jj < n for the reads; the write is
                // inside panel j0 / nr.
                unsafe {
                    *panel.add(p * nr + jj) = if jj < jb { *col.add(p + jj * ldb) } else { 0.0 };
                }
            }
        }
    }
}

/// Transposes `chunks` chunks of 64 rows x 16 columns through ZA.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for 64 * `chunks` reads in each of
/// 16 columns `ldb_bytes` apart, and `panel` for 1024 * `chunks` writes.
unsafe fn transpose_chunks_f32(
    col: *const f32,
    ldb_bytes: usize,
    panel: *mut f32,
    chunks: usize,
    prefetch_offset: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue pn8.s",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            // Four columns per step into slices w12..w12+3 of the four tiles.
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            // Core-side L2 prefetch of the same rows of the next panel's columns.
            "cbz x13, 41f",
            "add x9, x4, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x5, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x6, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "add x9, x7, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "41:",
            "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn8/z, [x4]",
            "ld1w {{z17.s, z21.s, z25.s, z29.s}}, pn8/z, [x5]",
            "ld1w {{z18.s, z22.s, z26.s, z30.s}}, pn8/z, [x6]",
            "ld1w {{z19.s, z23.s, z27.s, z31.s}}, pn8/z, [x7]",
            "mova za0h.s[w12, 0:3], {{z16.s-z19.s}}",
            "mova za1h.s[w12, 0:3], {{z20.s-z23.s}}",
            "mova za2h.s[w12, 0:3], {{z24.s-z27.s}}",
            "mova za3h.s[w12, 0:3], {{z28.s-z31.s}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            // Vertical slice p of tile t is packed row 16t + p.
            "mov w12, #0",
            "mov x8, x2",
            "add x9, x2, #1024",
            "add x10, x2, #2048",
            "add x11, x2, #3072",
            "4:",
            "mova {{z0.s-z3.s}}, za0v.s[w12, 0:3]",
            "mova {{z4.s-z7.s}}, za1v.s[w12, 0:3]",
            "mova {{z8.s-z11.s}}, za2v.s[w12, 0:3]",
            "mova {{z12.s-z15.s}}, za3v.s[w12, 0:3]",
            "st1w {{z0.s-z3.s}}, pn8, [x8]",
            "st1w {{z4.s-z7.s}}, pn8, [x9]",
            "st1w {{z8.s-z11.s}}, pn8, [x10]",
            "st1w {{z12.s-z15.s}}, pn8, [x11]",
            "addvl x8, x8, #4",
            "addvl x9, x9, #4",
            "addvl x10, x10, #4",
            "addvl x11, x11, #4",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 4b",
            "add x0, x0, #256",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x13") prefetch_offset,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x11") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p8") _,
            options(nostack)
        );
    }
}

/// Packs B for the f64 kernels; as [`pack_b_f32_sme2`] with `NR` = 8.
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `n` columns
///   `ldb` apart, and `dst` for writes of `ceil(n / 8) * 8 * k` elements.
pub(super) unsafe fn pack_b_f64_sme2(
    src: *const f64,
    ldb: usize,
    k: usize,
    n: usize,
    dst: *mut f64,
) {
    let nr = NR_F64;
    let chunks = k / 64;
    for j0 in (0..n).step_by(nr) {
        let jb = nr.min(n - j0);
        // SAFETY: panel j0 / nr starts inside `dst`; columns j0.. inside `src`.
        let (panel, col) = unsafe { (dst.add(j0 * k), src.add(j0 * ldb)) };
        let mut p0 = 0;
        if jb == nr && chunks > 0 {
            // SAFETY: the caller guarantees SME2; the block reads 64 * chunks
            // rows of 8 columns and writes the same number of packed rows.
            unsafe {
                // Prefetch the next panel only while one follows (zero: none).
                let next = if j0 + 2 * nr <= n { nr * ldb * 8 } else { 0 };
                transpose_chunks_f64(col, ldb * 8, panel, chunks, next)
            };
            p0 = chunks * 64;
        }
        if jb == nr && p0 < k {
            // SAFETY: rows p0..k (fewer than 64) of the 8 columns, and their
            // packed rows inside panel j0 / nr.
            unsafe { transpose_tail_f64(col.add(p0), ldb * 8, panel.add(p0 * nr), k - p0) };
            p0 = k;
        }
        for p in p0..k {
            for jj in 0..nr {
                // SAFETY: as in `pack_b_f32_sme2`.
                unsafe {
                    *panel.add(p * nr + jj) = if jj < jb { *col.add(p + jj * ldb) } else { 0.0 };
                }
            }
        }
    }
}

/// Transposes `chunks` chunks of 64 rows x 8 columns through ZA (8 tiles of 8 x 8).
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for 64 * `chunks` reads in each of
/// 8 columns `ldb_bytes` apart, and `panel` for 512 * `chunks` writes.
unsafe fn transpose_chunks_f64(
    col: *const f64,
    ldb_bytes: usize,
    panel: *mut f64,
    chunks: usize,
    prefetch_offset: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue pn8.d",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            // Four columns per step: rows 0-31 into tiles 0-3, rows 32-63 into tiles 4-7.
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            // Core-side L2 prefetch of the same rows of the next panel's columns.
            "cbz x13, 41f",
            "add x9, x4, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x5, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x6, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "add x9, x7, x13",
            "prfm pldl2keep, [x9]",
            "prfm pldl2keep, [x9, #128]",
            "prfm pldl2keep, [x9, #256]",
            "prfm pldl2keep, [x9, #384]",
            "41:",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn8/z, [x4]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn8/z, [x5]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn8/z, [x6]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn8/z, [x7]",
            "mova za0h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za1h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za2h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za3h.d[w12, 0:3], {{z28.d-z31.d}}",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn8/z, [x4, #4, mul vl]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn8/z, [x5, #4, mul vl]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn8/z, [x6, #4, mul vl]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn8/z, [x7, #4, mul vl]",
            "mova za4h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za5h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za6h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za7h.d[w12, 0:3], {{z28.d-z31.d}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            // Vertical slice p of tile t is packed row 8t + p.
            "mov w12, #0",
            "mov x8, x2",
            "add x9, x2, #2048",
            "4:",
            "mova {{z0.d-z3.d}}, za0v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za1v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za2v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za3v.d[w12, 0:3]",
            "st1d {{z0.d-z3.d}}, pn8, [x8]",
            "st1d {{z4.d-z7.d}}, pn8, [x8, #8, mul vl]",
            "st1d {{z8.d-z11.d}}, pn8, [x8, #16, mul vl]",
            "st1d {{z12.d-z15.d}}, pn8, [x8, #24, mul vl]",
            "mova {{z0.d-z3.d}}, za4v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za5v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za6v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za7v.d[w12, 0:3]",
            "st1d {{z0.d-z3.d}}, pn8, [x9]",
            "st1d {{z4.d-z7.d}}, pn8, [x9, #8, mul vl]",
            "st1d {{z8.d-z11.d}}, pn8, [x9, #16, mul vl]",
            "st1d {{z12.d-z15.d}}, pn8, [x9, #24, mul vl]",
            "addvl x8, x8, #4",
            "addvl x9, x9, #4",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 4b",
            "add x0, x0, #512",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            in("x13") prefetch_offset,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p8") _,
            options(nostack)
        );
    }
}

/// Transposes the last `rows` (< 64) rows of one full B panel through ZA, as
/// in `transpose_chunks_f32`, with the loads predicated to the rows.
///
/// # Safety
///
/// SME2 must be present, `rows` in 1..64; `col` must be valid for `rows`
/// reads in each of 16 columns `ldb_bytes` apart, and `panel` for `rows * 16`
/// writes.
unsafe fn transpose_tail_f32(col: *const f32, ldb_bytes: usize, panel: *mut f32, rows: usize) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "zero {{za}}",
            "ptrue p0.s",
            "mov x9, #0",
            "whilelt pn9.s, x9, x3, vlx4",
            "mov x4, x0",
            "mov w12, #0",
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn9/z, [x4]",
            "ld1w {{z17.s, z21.s, z25.s, z29.s}}, pn9/z, [x5]",
            "ld1w {{z18.s, z22.s, z26.s, z30.s}}, pn9/z, [x6]",
            "ld1w {{z19.s, z23.s, z27.s, z31.s}}, pn9/z, [x7]",
            "mova za0h.s[w12, 0:3], {{z16.s-z19.s}}",
            "mova za1h.s[w12, 0:3], {{z20.s-z23.s}}",
            "mova za2h.s[w12, 0:3], {{z24.s-z27.s}}",
            "mova za3h.s[w12, 0:3], {{z28.s-z31.s}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            "mov x8, x2",
            "mov x10, x3",
            "cbz x10, 9f",
            "mov w12, #0",
            "20:",
            "st1w {{za0v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 20b",
            "cbz x10, 9f",
            "mov w12, #0",
            "21:",
            "st1w {{za1v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 21b",
            "cbz x10, 9f",
            "mov w12, #0",
            "22:",
            "st1w {{za2v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 22b",
            "cbz x10, 9f",
            "mov w12, #0",
            "23:",
            "st1w {{za3v.s[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #16",
            "b.ne 23b",
            "9:",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            in("x3") rows,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p9") _, out("p10") _,
            options(nostack)
        );
    }
}

/// Transposes the last `rows` (< 64) rows of one full B panel through ZA, as
/// in `transpose_chunks_f64`, with the loads predicated to the rows.
///
/// # Safety
///
/// SME2 must be present, `rows` in 1..64; `col` must be valid for `rows`
/// reads in each of 8 columns `ldb_bytes` apart, and `panel` for `rows * 8`
/// writes.
unsafe fn transpose_tail_f64(col: *const f64, ldb_bytes: usize, panel: *mut f64, rows: usize) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "zero {{za}}",
            "ptrue p0.d",
            "mov x9, #0",
            "whilelt pn9.d, x9, x3, vlx4",
            "mov x9, #32",
            "whilelt pn10.d, x9, x3, vlx4",
            "mov x4, x0",
            "mov w12, #0",
            "3:",
            "add x5, x4, x1",
            "add x6, x5, x1",
            "add x7, x6, x1",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn9/z, [x4]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn9/z, [x5]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn9/z, [x6]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn9/z, [x7]",
            "mova za0h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za1h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za2h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za3h.d[w12, 0:3], {{z28.d-z31.d}}",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn10/z, [x4, #4, mul vl]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn10/z, [x5, #4, mul vl]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn10/z, [x6, #4, mul vl]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn10/z, [x7, #4, mul vl]",
            "mova za4h.d[w12, 0:3], {{z16.d-z19.d}}",
            "mova za5h.d[w12, 0:3], {{z20.d-z23.d}}",
            "mova za6h.d[w12, 0:3], {{z24.d-z27.d}}",
            "mova za7h.d[w12, 0:3], {{z28.d-z31.d}}",
            "add x4, x7, x1",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            "mov x8, x2",
            "mov x10, x3",
            "cbz x10, 9f",
            "mov w12, #0",
            "20:",
            "st1d {{za0v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 20b",
            "cbz x10, 9f",
            "mov w12, #0",
            "21:",
            "st1d {{za1v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 21b",
            "cbz x10, 9f",
            "mov w12, #0",
            "22:",
            "st1d {{za2v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 22b",
            "cbz x10, 9f",
            "mov w12, #0",
            "23:",
            "st1d {{za3v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 23b",
            "cbz x10, 9f",
            "mov w12, #0",
            "24:",
            "st1d {{za4v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 24b",
            "cbz x10, 9f",
            "mov w12, #0",
            "25:",
            "st1d {{za5v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 25b",
            "cbz x10, 9f",
            "mov w12, #0",
            "26:",
            "st1d {{za6v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 26b",
            "cbz x10, 9f",
            "mov w12, #0",
            "27:",
            "st1d {{za7v.d[w12, 0]}}, p0, [x8]",
            "add x8, x8, #64",
            "add w12, w12, #1",
            "sub x10, x10, #1",
            "cbz x10, 9f",
            "cmp w12, #8",
            "b.ne 27b",
            "9:",
            "smstop",
            inout("x0") col => _,
            in("x1") ldb_bytes,
            inout("x2") panel => _,
            in("x3") rows,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _, out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p9") _, out("p10") _,
            options(nostack)
        );
    }
}

/// Copies packed rows of 64 bytes (16 f32 or 8 f64, one B panel row) from a
/// row-ordered source: for each of `panels` panels and each row `p` of the
/// first `rows` rows (a multiple of 16), the 64 bytes at
/// `src + p * ld_bytes + panel * 64` go to `dst + panel * dst_panel_bytes + p * 64`.
/// Sixteen rows of all panels at a time, so the reads stay within 16 source rows.
///
/// # Safety
///
/// SME2 must be present; `src` and `dst` must be valid for those reads and writes.
pub(crate) unsafe fn copy_rows_64b(
    src: *const u8,
    ld_bytes: usize,
    rows: usize,
    panels: usize,
    dst: *mut u8,
    dst_panel_bytes: usize,
) {
    if rows < 16 || panels == 0 {
        return;
    }
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "mov x9, #0",
            "2:",
            "mul x11, x9, x1",
            "add x11, x0, x11",
            "add x12, x2, x9, lsl #6",
            "mov x10, #0",
            "3:",
            "mov x14, x11",
            "mov x15, x12",
            "mov x16, #4",
            "4:",
            "ld1w {{z0.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z1.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z2.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "ld1w {{z3.s}}, p0/z, [x14]",
            "add x14, x14, x1",
            "st1w {{z0.s-z3.s}}, pn8, [x15]",
            "addvl x15, x15, #4",
            "subs x16, x16, #1",
            "b.ne 4b",
            "add x11, x11, #64",
            "add x12, x12, x5",
            "add x10, x10, #1",
            "cmp x10, x4",
            "b.ne 3b",
            "add x9, x9, #16",
            "cmp x9, x3",
            "b.lo 2b",
            "smstop",
            in("x0") src,
            in("x1") ld_bytes,
            in("x2") dst,
            in("x3") rows,
            in("x4") panels,
            in("x5") dst_panel_bytes,
            out("x9") _, out("x10") _, out("x11") _, out("x12") _,
            out("x14") _, out("x15") _, out("x16") _,
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

/// Packs `chunks * 16` depth steps of one 64-row A panel from a column-major
/// `A^T` (`op(A) = A^T`): `col` points at `A^T[p0, i0]`, so row `i0 + r` of
/// op(A) is column `i0 + r` of `A^T`, `ld_bytes` apart. Each chunk loads 16
/// columns per ZA tile into its horizontal slices and stores the vertical
/// slices (depth steps) as packed panel rows of 64 elements.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for `chunks * 16` reads in each
/// of 64 columns `ld_bytes` apart, and `panel` for `chunks * 16 * 64` writes.
pub(super) unsafe fn pack_a_t_chunks_f32(
    col: *const f32,
    ld_bytes: usize,
    panel: *mut f32,
    chunks: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            "20:",
            "ld1w {{za0h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 20b",
            "mov w12, #0",
            "21:",
            "ld1w {{za1h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 21b",
            "mov w12, #0",
            "12:",
            "ld1w {{za2h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 12b",
            "mov w12, #0",
            "13:",
            "ld1w {{za3h.s[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 13b",
            "mov w12, #0",
            "mov x8, x2",
            "3:",
            "mova {{z0.s-z3.s}}, za0v.s[w12, 0:3]",
            "mova {{z4.s-z7.s}}, za1v.s[w12, 0:3]",
            "mova {{z8.s-z11.s}}, za2v.s[w12, 0:3]",
            "mova {{z12.s-z15.s}}, za3v.s[w12, 0:3]",
            "st1w {{z0.s, z4.s, z8.s, z12.s}}, pn8, [x8, #0, mul vl]",
            "st1w {{z1.s, z5.s, z9.s, z13.s}}, pn8, [x8, #4, mul vl]",
            "st1w {{z2.s, z6.s, z10.s, z14.s}}, pn8, [x8, #8, mul vl]",
            "st1w {{z3.s, z7.s, z11.s, z15.s}}, pn8, [x8, #12, mul vl]",
            "addvl x8, x8, #16",
            "add w12, w12, #4",
            "cmp w12, #16",
            "b.ne 3b",
            "add x0, x0, #64",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ld_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            out("x4") _, out("x8") _, out("x12") _,
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

/// Packs `chunks * 8` depth steps of one 64-row A panel from a column-major
/// `A^T` (`op(A) = A^T`): `col` points at `A^T[p0, i0]`, so row `i0 + r` of
/// op(A) is column `i0 + r` of `A^T`, `ld_bytes` apart. Each chunk loads 8
/// columns per ZA tile into its horizontal slices and stores the vertical
/// slices (depth steps) as packed panel rows of 64 elements.
///
/// # Safety
///
/// SME2 must be present; `col` must be valid for `chunks * 8` reads in each
/// of 64 columns `ld_bytes` apart, and `panel` for `chunks * 8 * 64` writes.
pub(super) unsafe fn pack_a_t_chunks_f64(
    col: *const f64,
    ld_bytes: usize,
    panel: *mut f64,
    chunks: usize,
) {
    // SAFETY: guaranteed by the caller; streaming mode starts and ends here.
    unsafe {
        core::arch::asm!(
            ".arch_extension sme2",
            "smstart",
            "ptrue p0.d",
            "ptrue pn8.d",
            "2:",
            "mov x4, x0",
            "mov w12, #0",
            "20:",
            "ld1d {{za0h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 20b",
            "mov w12, #0",
            "21:",
            "ld1d {{za1h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 21b",
            "mov w12, #0",
            "12:",
            "ld1d {{za2h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 12b",
            "mov w12, #0",
            "13:",
            "ld1d {{za3h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 13b",
            "mov w12, #0",
            "14:",
            "ld1d {{za4h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 14b",
            "mov w12, #0",
            "15:",
            "ld1d {{za5h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 15b",
            "mov w12, #0",
            "16:",
            "ld1d {{za6h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 16b",
            "mov w12, #0",
            "17:",
            "ld1d {{za7h.d[w12, 0]}}, p0/z, [x4]",
            "add x4, x4, x1",
            "add w12, w12, #1",
            "cmp w12, #8",
            "b.ne 17b",
            "mov w12, #0",
            "mov x8, x2",
            "3:",
            "mova {{z0.d-z3.d}}, za0v.d[w12, 0:3]",
            "mova {{z4.d-z7.d}}, za1v.d[w12, 0:3]",
            "mova {{z8.d-z11.d}}, za2v.d[w12, 0:3]",
            "mova {{z12.d-z15.d}}, za3v.d[w12, 0:3]",
            "mova {{z16.d-z19.d}}, za4v.d[w12, 0:3]",
            "mova {{z20.d-z23.d}}, za5v.d[w12, 0:3]",
            "mova {{z24.d-z27.d}}, za6v.d[w12, 0:3]",
            "mova {{z28.d-z31.d}}, za7v.d[w12, 0:3]",
            "st1d {{z0.d, z4.d, z8.d, z12.d}}, pn8, [x8, #0, mul vl]",
            "st1d {{z16.d, z20.d, z24.d, z28.d}}, pn8, [x8, #4, mul vl]",
            "st1d {{z1.d, z5.d, z9.d, z13.d}}, pn8, [x8, #8, mul vl]",
            "st1d {{z17.d, z21.d, z25.d, z29.d}}, pn8, [x8, #12, mul vl]",
            "st1d {{z2.d, z6.d, z10.d, z14.d}}, pn8, [x8, #16, mul vl]",
            "st1d {{z18.d, z22.d, z26.d, z30.d}}, pn8, [x8, #20, mul vl]",
            "st1d {{z3.d, z7.d, z11.d, z15.d}}, pn8, [x8, #24, mul vl]",
            "st1d {{z19.d, z23.d, z27.d, z31.d}}, pn8, [x8, #28, mul vl]",
            "addvl x8, x8, #16",
            "addvl x8, x8, #16",
            "add w12, w12, #4",
            "cmp w12, #8",
            "b.ne 3b",
            "add x0, x0, #64",
            "add x2, x2, #4096",
            "subs x3, x3, #1",
            "b.ne 2b",
            "smstop",
            inout("x0") col => _,
            in("x1") ld_bytes,
            inout("x2") panel => _,
            inout("x3") chunks => _,
            out("x4") _, out("x8") _, out("x12") _,
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

/// Packs rows `0..m` of `op(A) = A^T`, depth `k`, from a column-major `A^T`
/// (`src` = `A^T[0, 0]`, column stride `ld`) into 64-row panels of `64 * k`
/// elements (the layout of the packed A panels), zero-padding past `m`.
///
/// # Safety
///
/// - [`usable_f32`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `m` columns `ld`
///   apart, and `dst` for writes of `ceil(m / 64) * 64 * k` elements.
pub(super) unsafe fn pack_a_t_f32_sme2(
    src: *const f32,
    ld: usize,
    k: usize,
    m: usize,
    dst: *mut f32,
) {
    let chunks = k / 16;
    for i0 in (0..m).step_by(MR) {
        let ib = MR.min(m - i0);
        // SAFETY: panel i0 / 64 starts inside `dst`, column i0 inside `src`.
        let (panel, col) = unsafe { (dst.add(i0 * k), src.add(i0 * ld)) };
        let mut p0 = 0;
        if ib == MR && chunks > 0 {
            // SAFETY: the caller guarantees SME2; 64 columns of `chunks * 16` rows.
            unsafe { pack_a_t_chunks_f32(col, ld * core::mem::size_of::<f32>(), panel, chunks) };
            p0 = chunks * 16;
        }
        for p in p0..k {
            for r in 0..MR {
                // SAFETY: p < k and r < ib for the reads; the write is inside
                // panel i0 / 64.
                unsafe { *panel.add(p * MR + r) = if r < ib { *col.add(p + r * ld) } else { 0.0 } };
            }
        }
    }
}

/// Packs rows `0..m` of `op(A) = A^T`, depth `k`, from a column-major `A^T`
/// (`src` = `A^T[0, 0]`, column stride `ld`) into 64-row panels of `64 * k`
/// elements (the layout of the packed A panels), zero-padding past `m`.
///
/// # Safety
///
/// - [`usable_f64`] must have returned `true`.
/// - `src` must be valid for reads of `k` elements in each of `m` columns `ld`
///   apart, and `dst` for writes of `ceil(m / 64) * 64 * k` elements.
pub(super) unsafe fn pack_a_t_f64_sme2(
    src: *const f64,
    ld: usize,
    k: usize,
    m: usize,
    dst: *mut f64,
) {
    let chunks = k / 8;
    for i0 in (0..m).step_by(MR) {
        let ib = MR.min(m - i0);
        // SAFETY: panel i0 / 64 starts inside `dst`, column i0 inside `src`.
        let (panel, col) = unsafe { (dst.add(i0 * k), src.add(i0 * ld)) };
        let mut p0 = 0;
        if ib == MR && chunks > 0 {
            // SAFETY: the caller guarantees SME2; 64 columns of `chunks * 8` rows.
            unsafe { pack_a_t_chunks_f64(col, ld * core::mem::size_of::<f64>(), panel, chunks) };
            p0 = chunks * 8;
        }
        for p in p0..k {
            for r in 0..MR {
                // SAFETY: p < k and r < ib for the reads; the write is inside
                // panel i0 / 64.
                unsafe { *panel.add(p * MR + r) = if r < ib { *col.add(p + r * ld) } else { 0.0 } };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level3::gemm::{gemm, gemm_transposed, gemm_with_par};
    use crate::level3::gemm_kernel::GemmKernel;
    use crate::level3::trsm::Trans;
    use oxiblas_core::parallel::Par;
    use oxiblas_matrix::Mat;

    /// Deterministic values in [-1, 1).
    fn val(i: usize, seed: usize) -> f64 {
        let x = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 1999;
        x as f64 / 999.5 - 1.0
    }

    /// Checks one micro-kernel call against a scalar reference.
    fn check_kernel_f64(k: usize, alpha: f64, beta: f64) {
        let nr = NR_F64;
        let ldc = MR + 3;
        let a: Vec<f64> = (0..MR * k).map(|i| val(i, 1)).collect();
        let b: Vec<f64> = (0..nr * k).map(|i| val(i, 2)).collect();
        let c0: Vec<f64> = (0..ldc * nr)
            .map(|i| if beta == 0.0 { f64::NAN } else { val(i, 3) })
            .collect();
        let mut c = c0.clone();
        // SAFETY: SME2 is checked by the caller; the buffers have the sizes
        // the kernel reads and writes (64 x k, 8 x k, 8 columns of ldc).
        unsafe {
            micro_kernel_f64_sme2(k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), ldc)
        };
        for j in 0..nr {
            for i in 0..ldc {
                let got = c[i + j * ldc];
                if i >= MR {
                    assert!(
                        got.to_bits() == c0[i + j * ldc].to_bits(),
                        "k={k}: row {i} below the tile written"
                    );
                    continue;
                }
                let dot: f64 = (0..k).map(|p| a[p * MR + i] * b[p * nr + j]).sum();
                let exp = if beta == 0.0 {
                    alpha * dot
                } else {
                    alpha * dot + beta * c0[i + j * ldc]
                };
                assert!(
                    (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                    "f64 k={k} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }

    fn check_kernel_f32(k: usize, alpha: f32, beta: f32) {
        let nr = NR_F32;
        let ldc = MR + 5;
        let a: Vec<f32> = (0..MR * k).map(|i| val(i, 4) as f32).collect();
        let b: Vec<f32> = (0..nr * k).map(|i| val(i, 5) as f32).collect();
        let c0: Vec<f32> = (0..ldc * nr)
            .map(|i| {
                if beta == 0.0 {
                    f32::NAN
                } else {
                    val(i, 6) as f32
                }
            })
            .collect();
        let mut c = c0.clone();
        // SAFETY: as in `check_kernel_f64` (16 columns).
        unsafe {
            micro_kernel_f32_sme2(k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), ldc)
        };
        for j in 0..nr {
            for i in 0..ldc {
                let got = c[i + j * ldc];
                if i >= MR {
                    assert!(
                        got.to_bits() == c0[i + j * ldc].to_bits(),
                        "k={k}: row {i} below the tile written"
                    );
                    continue;
                }
                let dot: f64 = (0..k)
                    .map(|p| f64::from(a[p * MR + i]) * f64::from(b[p * nr + j]))
                    .sum();
                let c_old = f64::from(c0[i + j * ldc]);
                let exp = if beta == 0.0 {
                    f64::from(alpha) * dot
                } else {
                    f64::from(alpha) * dot + f64::from(beta) * c_old
                };
                let tol = 1e-5 * (1.0 + k as f64);
                assert!(
                    (f64::from(got) - exp).abs() <= tol,
                    "f32 k={k} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }

    /// Checks one macro-kernel call on an `m x n` block; padding of the last
    /// panels holds NaN, which must not reach C, and C outside the block must
    /// stay unchanged.
    fn check_macro_f64(m: usize, n: usize, k: usize, alpha: f64, beta: f64, mode: u8) {
        let nr = NR_F64;
        let (mp, np) = (m.div_ceil(MR), n.div_ceil(nr));
        let ldc = m + 3;
        let mut a = vec![f64::NAN; mp * MR * k];
        for ib in 0..mp {
            for p in 0..k {
                for i in 0..MR.min(m - ib * MR) {
                    a[ib * MR * k + p * MR + i] = val(ib * MR + i + p * m, 21);
                }
            }
        }
        // In place: column-major A with column stride lda > m, NaN past row m.
        let lda = m + 2;
        let mut a_cm = vec![f64::NAN; lda * k];
        for p in 0..k {
            for i in 0..m {
                a_cm[i + p * lda] = val(i + p * m, 21);
            }
        }
        for ib in 0..mp {
            for p in 0..k {
                for i in 0..MR.min(m - ib * MR) {
                    a[ib * MR * k + p * MR + i] = a_cm[ib * MR + i + p * lda];
                }
            }
        }
        // 0: packed A; 1: column-major A in place; 2: in place with online packing.
        let mut a_pack = vec![f64::NAN; mp * MR * k];
        let (a_ptr, a_panel, a_step, a_pack_ptr) = match mode {
            0 => (a.as_ptr(), MR * k, MR, core::ptr::null_mut()),
            1 => (a_cm.as_ptr(), MR, lda, core::ptr::null_mut()),
            _ => (a_cm.as_ptr(), MR, lda, a_pack.as_mut_ptr()),
        };
        let mut b = vec![f64::NAN; np * nr * k];
        for jb in 0..np {
            for p in 0..k {
                for j in 0..nr.min(n - jb * nr) {
                    b[jb * nr * k + p * nr + j] = val(jb * nr + j + p * n, 22);
                }
            }
        }
        let c0: Vec<f64> = (0..ldc * (n + 1)).map(|i| val(i, 23)).collect();
        let mut c = c0.clone();
        // SAFETY: SME2 is checked by the caller; the panels and C have the
        // sizes the macro-kernel reads and writes.
        unsafe {
            macro_kernel_f64_sme2(
                m,
                n,
                k,
                alpha,
                a_ptr,
                a_panel,
                a_step,
                a_pack_ptr,
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                ldc,
            )
        };
        for j in 0..=n {
            for i in 0..ldc {
                let got = c[i + j * ldc];
                if i >= m || j >= n {
                    assert!(
                        got.to_bits() == c0[i + j * ldc].to_bits(),
                        "{m}x{n}: C({i},{j}) outside the block changed"
                    );
                    continue;
                }
                let (ib, ii, jb, jj) = (i / MR, i % MR, j / nr, j % nr);
                let dot: f64 = (0..k)
                    .map(|p| a[ib * MR * k + p * MR + ii] * b[jb * nr * k + p * nr + jj])
                    .sum();
                let exp = alpha * dot + beta * c0[i + j * ldc];
                assert!(
                    (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                    "{m}x{n}x{k} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }

    fn check_macro_f32(m: usize, n: usize, k: usize, alpha: f32, beta: f32, mode: u8) {
        let nr = NR_F32;
        let (mp, np) = (m.div_ceil(MR), n.div_ceil(nr));
        let ldc = m + 5;
        let mut a = vec![f32::NAN; mp * MR * k];
        for ib in 0..mp {
            for p in 0..k {
                for i in 0..MR.min(m - ib * MR) {
                    a[ib * MR * k + p * MR + i] = val(ib * MR + i + p * m, 24) as f32;
                }
            }
        }
        // In place: column-major A with column stride lda > m, NaN past row m.
        let lda = m + 2;
        let mut a_cm = vec![f32::NAN; lda * k];
        for p in 0..k {
            for i in 0..m {
                a_cm[i + p * lda] = val(i + p * m, 24) as f32;
            }
        }
        for ib in 0..mp {
            for p in 0..k {
                for i in 0..MR.min(m - ib * MR) {
                    a[ib * MR * k + p * MR + i] = a_cm[ib * MR + i + p * lda];
                }
            }
        }
        // 0: packed A; 1: column-major A in place; 2: in place with online packing.
        let mut a_pack = vec![f32::NAN; mp * MR * k];
        let (a_ptr, a_panel, a_step, a_pack_ptr) = match mode {
            0 => (a.as_ptr(), MR * k, MR, core::ptr::null_mut()),
            1 => (a_cm.as_ptr(), MR, lda, core::ptr::null_mut()),
            _ => (a_cm.as_ptr(), MR, lda, a_pack.as_mut_ptr()),
        };
        let mut b = vec![f32::NAN; np * nr * k];
        for jb in 0..np {
            for p in 0..k {
                for j in 0..nr.min(n - jb * nr) {
                    b[jb * nr * k + p * nr + j] = val(jb * nr + j + p * n, 25) as f32;
                }
            }
        }
        let c0: Vec<f32> = (0..ldc * (n + 1)).map(|i| val(i, 26) as f32).collect();
        let mut c = c0.clone();
        // SAFETY: as in `check_macro_f64`.
        unsafe {
            macro_kernel_f32_sme2(
                m,
                n,
                k,
                alpha,
                a_ptr,
                a_panel,
                a_step,
                a_pack_ptr,
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                ldc,
            )
        };
        for j in 0..=n {
            for i in 0..ldc {
                let got = c[i + j * ldc];
                if i >= m || j >= n {
                    assert!(
                        got.to_bits() == c0[i + j * ldc].to_bits(),
                        "{m}x{n}: C({i},{j}) outside the block changed"
                    );
                    continue;
                }
                let (ib, ii, jb, jj) = (i / MR, i % MR, j / nr, j % nr);
                let dot: f64 = (0..k)
                    .map(|p| {
                        f64::from(a[ib * MR * k + p * MR + ii])
                            * f64::from(b[jb * nr * k + p * nr + jj])
                    })
                    .sum();
                let exp = f64::from(alpha) * dot + f64::from(beta) * f64::from(c0[i + j * ldc]);
                let tol = 1e-5 * (1.0 + k as f64);
                assert!(
                    (f64::from(got) - exp).abs() <= tol,
                    "{m}x{n}x{k} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }

    const BLOCKS: [(usize, usize, usize); 8] = [
        (1, 1, 1),
        (63, 7, 5),
        (64, 8, 4),
        (64, 16, 9),
        (65, 17, 3),
        (130, 9, 33),
        (200, 47, 70),
        (37, 100, 0),
    ];

    #[test]
    fn test_sme2_macro_kernel_f64() {
        if !usable_f64() {
            return;
        }
        for (m, n, k) in BLOCKS {
            for mode in 0..3 {
                check_macro_f64(m, n, k, 1.25, 0.0, mode);
                check_macro_f64(m, n, k, -0.5, 0.75, mode);
                check_macro_f64(m, n, k, 1.0, 0.0, mode);
                check_macro_f64(m, n, k, 1.0, 1.0, mode);
            }
        }
    }

    #[test]
    fn test_sme2_macro_kernel_f32() {
        if !usable_f32() {
            return;
        }
        for (m, n, k) in BLOCKS {
            for mode in 0..3 {
                check_macro_f32(m, n, k, 1.25, 0.0, mode);
                check_macro_f32(m, n, k, -0.5, 0.75, mode);
                check_macro_f32(m, n, k, 1.0, 0.0, mode);
                check_macro_f32(m, n, k, 1.0, 1.0, mode);
            }
        }
    }

    #[test]
    fn test_sme2_pack_b() {
        for &(k, n) in &[
            (1, 1),
            (63, 15),
            (64, 16),
            (65, 17),
            (130, 40),
            (200, 8),
            (256, 33),
        ] {
            let ldb = k + 3;
            let src: Vec<f64> = (0..ldb * n).map(|i| val(i, 31)).collect();
            if usable_f64() {
                let len = n.div_ceil(NR_F64) * NR_F64 * k;
                let mut dst = vec![f64::NAN; len];
                // SAFETY: SME2 checked; src holds n columns of ldb >= k, dst holds len.
                unsafe { pack_b_f64_sme2(src.as_ptr(), ldb, k, n, dst.as_mut_ptr()) };
                for (idx, &got) in dst.iter().enumerate() {
                    let (panel, rest) = (idx / (NR_F64 * k), idx % (NR_F64 * k));
                    let (p, jj) = (rest / NR_F64, rest % NR_F64);
                    let j = panel * NR_F64 + jj;
                    let exp = if j < n { src[p + j * ldb] } else { 0.0 };
                    assert!(got.to_bits() == exp.to_bits(), "f64 k={k} n={n}: ({p},{j})");
                }
            }
            if usable_f32() {
                let src32: Vec<f32> = src.iter().map(|&x| x as f32).collect();
                let len = n.div_ceil(NR_F32) * NR_F32 * k;
                let mut dst = vec![f32::NAN; len];
                // SAFETY: as above.
                unsafe { pack_b_f32_sme2(src32.as_ptr(), ldb, k, n, dst.as_mut_ptr()) };
                for (idx, &got) in dst.iter().enumerate() {
                    let (panel, rest) = (idx / (NR_F32 * k), idx % (NR_F32 * k));
                    let (p, jj) = (rest / NR_F32, rest % NR_F32);
                    let j = panel * NR_F32 + jj;
                    let exp = if j < n { src32[p + j * ldb] } else { 0.0 };
                    assert!(got.to_bits() == exp.to_bits(), "f32 k={k} n={n}: ({p},{j})");
                }
            }
        }
    }

    const DEPTHS: [usize; 10] = [0, 1, 2, 3, 4, 5, 7, 8, 33, 257];

    #[test]
    fn test_sme2_kernel_f64() {
        if !usable_f64() {
            return;
        }
        for k in DEPTHS {
            for (alpha, beta) in [(1.0, 0.0), (2.5, 0.0), (1.0, 1.0), (-0.75, -0.5)] {
                check_kernel_f64(k, alpha, beta);
            }
        }
    }

    #[test]
    fn test_sme2_kernel_f32() {
        if !usable_f32() {
            return;
        }
        for k in DEPTHS {
            for (alpha, beta) in [(1.0, 0.0), (2.5, 0.0), (1.0, 1.0), (-0.75, -0.5)] {
                check_kernel_f32(k, alpha, beta);
            }
        }
    }

    #[test]
    fn test_sme2_shape_is_selected() {
        if usable_f32() {
            let s = f32::micro_kernel_shape();
            assert_eq!((s.mr, s.nr), (MR, NR_F32));
        }
        if usable_f64() {
            let s = f64::micro_kernel_shape();
            assert_eq!((s.mr, s.nr), (MR, NR_F64));
        }
    }

    fn mat(rows: usize, cols: usize, seed: usize) -> Mat<f64> {
        let mut m = Mat::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                m[(i, j)] = val(i + j * rows, seed);
            }
        }
        m
    }

    /// op(A) op(B) through the public GEMM, against a naive product.
    fn check_gemm_f64(m: usize, n: usize, k: usize, ta: Trans, tb: Trans, par: Par) {
        let a = if ta == Trans::NoTrans {
            mat(m, k, 7)
        } else {
            mat(k, m, 7)
        };
        let b = if tb == Trans::NoTrans {
            mat(k, n, 8)
        } else {
            mat(n, k, 8)
        };
        let c0 = mat(m, n, 9);
        let mut c = c0.clone();
        let (alpha, beta) = (1.5, -0.25);
        if ta == Trans::NoTrans && tb == Trans::NoTrans {
            gemm_with_par(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), par);
        } else {
            gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
        }
        let at = |i: usize, p: usize| {
            if ta == Trans::NoTrans {
                a[(i, p)]
            } else {
                a[(p, i)]
            }
        };
        let bt = |p: usize, j: usize| {
            if tb == Trans::NoTrans {
                b[(p, j)]
            } else {
                b[(j, p)]
            }
        };
        for j in 0..n {
            for i in 0..m {
                let dot: f64 = (0..k).map(|p| at(i, p) * bt(p, j)).sum();
                let exp = alpha * dot + beta * c0[(i, j)];
                let got = c[(i, j)];
                assert!(
                    (got - exp).abs() <= 1e-12 * (1.0 + k as f64),
                    "{m}x{n}x{k} {ta:?} {tb:?} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }

    #[test]
    fn test_sme2_gemm_edges() {
        // Rows and columns around the 64 x 8 tile, depths around the K unroll.
        for &(m, n, k) in &[
            (63, 7, 5),
            (64, 8, 64),
            (65, 9, 67),
            (130, 17, 33),
            (200, 41, 300),
            (1, 40, 50),
            (70, 1, 50),
            // Below SMALL_THRESHOLD, routed to SME2 by `sme2_small`.
            (32, 8, 8),
            (33, 9, 11),
            (16, 16, 64),
            (31, 31, 31),
        ] {
            check_gemm_f64(m, n, k, Trans::NoTrans, Trans::NoTrans, Par::Seq);
        }
        check_gemm_f64(97, 23, 71, Trans::Trans, Trans::NoTrans, Par::Seq);
        check_gemm_f64(97, 23, 71, Trans::NoTrans, Trans::Trans, Par::Seq);
        check_gemm_f64(97, 23, 71, Trans::Trans, Trans::Trans, Par::Seq);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sme2_gemm_parallel_concurrent_callers() {
        // Several threads at once: one holds the workers, the others run in place.
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    check_gemm_f64(260, 210, 90, Trans::NoTrans, Trans::NoTrans, Par::Rayon)
                });
            }
        });
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sme2_gemm_parallel() {
        // Above the SME2 parallel minimum (2^22 multiply-adds).
        check_gemm_f64(300, 230, 70, Trans::NoTrans, Trans::NoTrans, Par::Rayon);
    }

    #[test]
    fn test_sme2_gemm_f32_matches_f64() {
        for (m, n, k) in [(131, 45, 97), (1, 300, 77), (300, 1, 77), (1, 1, 5)] {
            gemm_f32_matches_f64(m, n, k);
        }
    }

    fn gemm_f32_matches_f64(m: usize, n: usize, k: usize) {
        let a64 = mat(m, k, 11);
        let b64 = mat(k, n, 12);
        let mut a: Mat<f32> = Mat::zeros(m, k);
        let mut b: Mat<f32> = Mat::zeros(k, n);
        for j in 0..k {
            for i in 0..m {
                a[(i, j)] = a64[(i, j)] as f32;
            }
        }
        for j in 0..n {
            for i in 0..k {
                b[(i, j)] = b64[(i, j)] as f32;
            }
        }
        let mut c: Mat<f32> = Mat::zeros(m, n);
        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        for j in 0..n {
            for i in 0..m {
                let exp: f64 = (0..k)
                    .map(|p| f64::from(a[(i, p)]) * f64::from(b[(p, j)]))
                    .sum();
                assert!((f64::from(c[(i, j)]) - exp).abs() <= 1e-4, "({i},{j})");
            }
        }
    }
}
