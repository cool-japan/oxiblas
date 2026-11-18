//! Tests of the FPCR / FPSR accessors and of `with_default_fp_state`, on
//! any AArch64 CPU (FPCR and FPSR are base A64; no SME needed).

use std::hint::black_box;

use super::*;
use crate::level3::sme2::test_support::{FPSR_DZC, FPSR_IOC, FPSR_IXC, FPSR_QC};

/// FPCR.RMode = round towards zero (bits 23:22 = 0b11).
const FPCR_RZ: u64 = 0b11 << 22;
/// FPCR.FZ: flush-to-zero.
const FPCR_FZ: u64 = 1 << 24;
/// FPCR.DN: default NaN.
const FPCR_DN: u64 = 1 << 25;
/// Modes a caller may run with: none of them is the default.
const CALLER_MODES: u64 = FPCR_RZ | FPCR_FZ | FPCR_DN;
/// Flags a caller may hold: none of them is raised by the code below.
const CALLER_FLAGS: u64 = FPSR_DZC | FPSR_QC;

/// Gives the test thread back its FPCR and FPSR when dropped (also when an
/// assertion fails).
struct Keep {
    /// The thread's FPCR.
    fpcr: u64,
    /// The thread's FPSR.
    fpsr: u64,
}

impl Keep {
    /// Saves the calling thread's FPCR and FPSR.
    fn new() -> Self {
        Self {
            fpcr: fpcr_read(),
            fpsr: fpsr_read(),
        }
    }
}

impl Drop for Keep {
    fn drop(&mut self) {
        // SAFETY: the value the test thread held before.
        unsafe { fpcr_write(self.fpcr) };
        fpsr_write(self.fpsr);
    }
}

/// What a stand-in check sees and does inside the default state: FPCR, the
/// bits of `1 / 3` in `f32` (round to nearest gives `0x3eaa_aaab`, round
/// towards zero `0x3eaa_aaaa`), then FPSR after an inexact division and an
/// invalid `0 · ∞`.
fn inside() -> (u64, u32, u64) {
    let fpcr = fpcr_read();
    let third = black_box(1.0_f32) / black_box(3.0_f32);
    let invalid = black_box(0.0_f64) * black_box(f64::INFINITY);
    black_box(invalid);
    (fpcr, third.to_bits(), fpsr_read())
}

#[test]
fn test_sme2_fpstate_fpcr_and_fpsr_round_trip() {
    let keep = Keep::new();
    let mut read = Vec::new();
    for fpcr in [0, FPCR_RZ, FPCR_FZ, FPCR_DN, CALLER_MODES] {
        // SAFETY: a test thread; `keep` restores the original value.
        unsafe { fpcr_write(fpcr) };
        read.push((fpcr, fpcr_read()));
    }
    for fpsr in [0, FPSR_IXC, CALLER_FLAGS] {
        fpsr_write(fpsr);
        read.push((fpsr, fpsr_read()));
    }
    drop(keep);
    for (wrote, got) in read {
        assert_eq!(got, wrote, "wrote {wrote:#x}, read {got:#x}");
    }
}

#[test]
fn test_sme2_fpstate_default_state_inside_and_restored_after() {
    let keep = Keep::new();
    // SAFETY: a test thread; `keep` restores the original value.
    unsafe { fpcr_write(CALLER_MODES) };
    fpsr_write(CALLER_FLAGS);
    // The caller's modes are in force before: 1 / 3 rounds towards zero.
    let (_, third_before, _) = inside();
    fpsr_write(CALLER_FLAGS);
    let (fpcr_in, third_in, fpsr_in) = with_default_fp_state(inside);
    let (fpcr_after, fpsr_after) = (fpcr_read(), fpsr_read());
    drop(keep);
    assert_eq!(third_before, 0x3eaa_aaaa, "the caller's rounding mode");
    assert_eq!(fpcr_in, 0, "FPCR inside the default state");
    assert_eq!(third_in, 0x3eaa_aaab, "round to nearest inside");
    assert_eq!(
        fpsr_in & (FPSR_IXC | FPSR_IOC),
        FPSR_IXC | FPSR_IOC,
        "flags raised inside: {fpsr_in:#x}"
    );
    assert_eq!(fpcr_after, CALLER_MODES, "the caller's FPCR afterwards");
    assert_eq!(fpsr_after, CALLER_FLAGS, "the caller's FPSR afterwards");
}

#[test]
fn test_sme2_fpstate_restored_when_unwinding() {
    let keep = Keep::new();
    // SAFETY: a test thread; `keep` restores the original value.
    unsafe { fpcr_write(CALLER_MODES) };
    fpsr_write(CALLER_FLAGS);
    // `resume_unwind` skips the panic hook, so the log shows no message.
    let unwound = std::panic::catch_unwind(|| {
        with_default_fp_state(|| {
            black_box(inside());
            std::panic::resume_unwind(Box::new("unwinds inside the default state"))
        })
    });
    let (fpcr_after, fpsr_after) = (fpcr_read(), fpsr_read());
    drop(keep);
    assert!(unwound.is_err());
    assert_eq!(fpcr_after, CALLER_MODES, "FPCR after unwinding");
    assert_eq!(fpsr_after, CALLER_FLAGS, "FPSR after unwinding");
}
