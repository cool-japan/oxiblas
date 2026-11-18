//! The calling thread's floating-point control and status registers (FPCR,
//! FPSR), read and written with plain AArch64 `mrs` / `msr` (no SME
//! instruction; both registers exist on every AArch64 CPU and are
//! accessible at EL0).
//!
//! The detection's first-use check runs inside [`with_default_fp_state`]:
//! FPCR is 0 for the check (round to nearest, no flush-to-zero, default NaN
//! off, no trap enabled), and afterwards FPCR and FPSR hold, bit for bit,
//! what the caller had. So the verdict does not depend on the caller's
//! modes, a trap the caller enabled cannot fire on the check's NaN and
//! overflow cases, and the check's cumulative flags never reach the caller.
//!
//! The four templates are the only assembly here, one register access each.
//! The file is registered in `lint_scan::ASM_SITES` (four sites). The
//! template lint has a kind for the FPSR accessors only, so
//! `tests_asm_sites.rs` pins the exact text of all four, lints the FPSR ones
//! with that kind and the FPCR ones with the same kind after the register
//! name (the one finding left must be the register), and requires every
//! assembly site of this file to take its text from one template macro,
//! each macro at exactly one site.

#[cfg(test)]
#[path = "tests_fpstate.rs"]
mod tests;

/// FPCR read.
macro_rules! fpcr_read_template {
    () => {
        "mrs {fpcr}, fpcr"
    };
}

/// FPCR write.
macro_rules! fpcr_write_template {
    () => {
        "msr fpcr, {fpcr}"
    };
}

/// FPSR read.
macro_rules! fpsr_read_template {
    () => {
        "mrs {fpsr}, fpsr"
    };
}

/// FPSR write.
macro_rules! fpsr_write_template {
    () => {
        "msr fpsr, {fpsr}"
    };
}

/// The two FPCR templates (read, write), for the template tests.
#[cfg(test)]
pub(super) const FPCR_TEMPLATES: [&str; 2] = [fpcr_read_template!(), fpcr_write_template!()];

/// The two FPSR templates (read, write), for the template lint.
#[cfg(test)]
pub(super) const FPSR_TEMPLATES: [&str; 2] = [fpsr_read_template!(), fpsr_write_template!()];

/// The calling thread's FPCR.
pub(crate) fn fpcr_read() -> u64 {
    let fpcr: u64;
    // SAFETY: reading FPCR is always permitted at EL0 on AArch64 and touches
    // no memory, stack or condition flags.
    unsafe {
        core::arch::asm!(
            fpcr_read_template!(),
            fpcr = out(reg) fpcr,
            options(nomem, nostack, preserves_flags),
        );
    }
    fpcr
}

/// Sets the calling thread's FPCR.
///
/// The block is not `nomem`, so the compiler keeps memory accesses on their
/// side of the write.
///
/// # Safety
///
/// Rust code assumes the default floating-point environment (FPCR = 0 on
/// AArch64). The caller writes either 0 or a value the thread held before
/// (restoring it), and must not leave a value on the thread that the code
/// running after it does not expect.
pub(crate) unsafe fn fpcr_write(fpcr: u64) {
    // SAFETY: writing FPCR is permitted at EL0 on AArch64; it touches no
    // memory, stack or condition flags. What the new modes mean for the code
    // that runs next is the caller's contract.
    unsafe {
        core::arch::asm!(
            fpcr_write_template!(),
            fpcr = in(reg) fpcr,
            options(nostack, preserves_flags),
        );
    }
}

/// The calling thread's FPSR.
pub(crate) fn fpsr_read() -> u64 {
    let fpsr: u64;
    // SAFETY: reading FPSR is always permitted at EL0 on AArch64 and touches
    // no memory, stack or condition flags.
    unsafe {
        core::arch::asm!(
            fpsr_read_template!(),
            fpsr = out(reg) fpsr,
            options(nomem, nostack, preserves_flags),
        );
    }
    fpsr
}

/// Sets the calling thread's FPSR.
///
/// The block is not `nomem`, so the compiler keeps memory accesses on their
/// side of the write.
pub(crate) fn fpsr_write(fpsr: u64) {
    // SAFETY: writing FPSR is always permitted at EL0 on AArch64. It changes
    // only the cumulative exception and saturation flags, which Rust code
    // never relies on; it touches no stack or condition flags.
    unsafe {
        core::arch::asm!(
            fpsr_write_template!(),
            fpsr = in(reg) fpsr,
            options(nostack, preserves_flags),
        );
    }
}

/// Restores the saved FPCR and FPSR when dropped, also while a panic
/// unwinds (see [`with_default_fp_state`]).
struct Restore {
    /// The caller's FPCR.
    fpcr: u64,
    /// The caller's FPSR.
    fpsr: u64,
}

impl Drop for Restore {
    fn drop(&mut self) {
        // SAFETY: the value is the one this thread held before
        // `with_default_fp_state` wrote 0.
        unsafe { fpcr_write(self.fpcr) };
        // FPSR last: writing FPCR leaves FPSR alone, and nothing after this
        // line raises a flag.
        fpsr_write(self.fpsr);
    }
}

/// Runs `f` with FPCR = 0 and afterwards gives the calling thread back its
/// FPCR and FPSR, bit for bit (also when `f` unwinds).
///
/// FPCR = 0 is the default environment that Rust code assumes: round to
/// nearest, no flush-to-zero, default NaN off, alternative half-precision
/// and alternate handling off, no trap enabled. The cumulative flags that
/// `f` raises are discarded.
///
/// The compiler assumes the default floating-point environment and may move
/// floating-point arithmetic across the register accesses (they order only
/// memory and other assembly). So this function is never inlined, which
/// keeps the caller's arithmetic out of the region, and `f` should be a call
/// that the compiler cannot see through (the first-use checks run in an
/// `#[inline(never)]` function), which keeps the region's arithmetic inside.
#[inline(never)]
pub(crate) fn with_default_fp_state<R>(f: impl FnOnce() -> R) -> R {
    let restore = Restore {
        fpcr: fpcr_read(),
        fpsr: fpsr_read(),
    };
    // SAFETY: 0 is the default floating-point environment, the one Rust code
    // assumes; `restore` writes the caller's value back when it drops.
    unsafe { fpcr_write(0) };
    let result = f();
    drop(restore);
    result
}
