//! Test-only switches of the calling thread's SME state, so that tests can
//! make `detect::thread_ready()` read a real "not ready" state through the
//! real register probe.
//!
//! Under QEMU every thread that runs a test is in the state that the SME
//! procedure-call standard calls "off": SVCR = 0 and TPIDR2_EL0 = 0. The
//! guards below put the calling thread into another state between ordinary
//! Rust statements, each through one register write, and restore the
//! previous state when they are dropped, also while a failing assertion
//! unwinds:
//!
//! - [`ZaStorageOn`]: `smstart za` turns ZA storage on outside streaming
//!   mode (PSTATE.ZA = 1, so SVCR = 2: the state of a function that owns
//!   live ZA contents and calls ordinary code); the drop turns it off again
//!   (`smstop za`) unless it was on before.
//! - [`Tpidr2Set`]: `msr tpidr2_el0` stores the address of [`TPIDR2_BLOCK`],
//!   a zeroed TPIDR2 block, so that TPIDR2_EL0 is non-zero (what a pending
//!   lazy save of a caller's ZA contents looks like to the readiness rule);
//!   the drop writes the previous value back. The block is a static and
//!   names no save buffer, so a routine of the standard that looks at it
//!   (an unwinder that turns ZA off, `__arm_tpidr2_save`) finds nothing to
//!   save, even if a guard is leaked.
//!
//! Neither switch changes PSTATE.SM, so no instruction becomes illegal while
//! a guard lives, and the readiness check reads the changed state. Two limits
//! hold for code inside a guard's lifetime: with ZA storage on (and TPIDR2_EL0
//! zero) ZA is active in the procedure-call standard's sense, so nothing may
//! call a routine that uses ZA; and no thread may be created (no rayon pool,
//! no `std::thread::spawn`, no first parallel call that builds rayon's global
//! pool), because a new thread may start with PSTATE.ZA = 1 and stay not
//! ready for the rest of the process. The tests keep both limits.
//!
//! Streaming mode itself (SVCR bit 0) has no switch. Ordinary Rust code must
//! not run in streaming mode: the compiler emits Advanced SIMD and scalar
//! floating-point instructions, which are illegal there without
//! FEAT_SME_FA64 (an Apple M4 has none, and the QEMU runs model none), and
//! the module's rules confine streaming mode to the single region of a
//! `sme_kernels!` template, from its `smstart` to its `smstop`, which may
//! not call anything. That bit is covered by the test of the readiness rule
//! itself (`ThreadState::ready` for every SVCR value).
//!
//! The three templates are the only assembly here: one register write each,
//! after `.arch_extension sme`. The file is registered in
//! `lint_scan::ASM_SITES` (three sites) and in
//! `lint_scan::STREAMING_TEXT_FILES` (the extension line). The template lint
//! has no kind for a state switch, so
//! `test_sme2_state_switch_templates_are_linted` pins the exact text and
//! runs the lint's probe kind on it: every rule a non-streaming template
//! must keep (alphabet, braces and placeholders, separators and comments,
//! directives, registers) has to pass, and the one finding left has to be
//! that the instruction is not one of the probe's reads.
//!
//! Every switch needs SME, so a guard is `None` on a host without SME
//! hardware (where the instructions are undefined).

use std::marker::PhantomData;

use super::detect;

/// SVCR bit 1: PSTATE.ZA (ZA storage on).
const SVCR_ZA: u64 = 1 << 1;

/// ZA storage on, outside streaming mode.
macro_rules! za_on_template {
    () => {
        ".arch_extension sme\nsmstart za"
    };
}

/// ZA storage off.
macro_rules! za_off_template {
    () => {
        ".arch_extension sme\nsmstop za"
    };
}

/// TPIDR2_EL0 written from a general-purpose register.
macro_rules! tpidr2_write_template {
    () => {
        ".arch_extension sme\nmsr tpidr2_el0, {value}"
    };
}

/// The three switch templates, for their test.
pub(super) const STATE_SWITCH_TEMPLATES: [&str; 3] = [
    za_on_template!(),
    za_off_template!(),
    tpidr2_write_template!(),
];

/// A TPIDR2 block of the SME procedure-call standard: the address of a ZA
/// save buffer (8 bytes), the number of saved ZA slices (2 bytes) and six
/// reserved bytes. All zero here: no buffer, nothing to save.
#[repr(C, align(16))]
pub(super) struct Tpidr2Block(pub(super) [u64; 2]);

/// The block whose address [`Tpidr2Set`] stores in TPIDR2_EL0 (see the
/// module docs). A static, so the address stays valid for the whole run.
pub(super) static TPIDR2_BLOCK: Tpidr2Block = Tpidr2Block([0; 2]);

/// ZA storage switched on outside streaming mode (SVCR = 2) while the guard
/// lives; see the module docs.
///
/// The guard is neither `Send` nor `Sync`: the state belongs to the thread
/// that created it, which must also drop it.
#[derive(Debug)]
#[must_use = "ZA storage is on only while the guard lives"]
pub(super) struct ZaStorageOn {
    /// ZA storage was on already (the drop then leaves it on).
    was_on: bool,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

impl ZaStorageOn {
    /// Turns ZA storage on for the calling thread; `None` without SME
    /// hardware ([`detect::candidate`] is `None`), where the instruction
    /// would be undefined. It never starts the first-use check.
    pub(super) fn new() -> Option<Self> {
        detect::candidate()?;
        // SAFETY: the candidate is present only after the operating system
        // reported FEAT_SME for this process and the detection executed this
        // probe.
        let was_on = unsafe { detect::probe_thread_state() }.svcr & SVCR_ZA != 0;
        // SAFETY: SME is implemented and enabled (above). SMSTART ZA sets
        // PSTATE.ZA (and zeroes the ZA storage when it was off); it leaves
        // PSTATE.SM, every general-purpose, vector and predicate register,
        // the condition flags, FPSR, the stack and memory alone, so the
        // ordinary code that runs after it is unaffected.
        unsafe {
            core::arch::asm!(za_on_template!(), options(nomem, nostack, preserves_flags));
        }
        Some(Self {
            was_on,
            _not_send: PhantomData,
        })
    }
}

impl Drop for ZaStorageOn {
    fn drop(&mut self) {
        if self.was_on {
            return;
        }
        // SAFETY: the guard exists only where SME is implemented and enabled
        // (`new`). SMSTOP ZA clears PSTATE.ZA (the ZA contents are dropped,
        // and nothing of this test's lives there); like SMSTART ZA it
        // touches no register, flag, stack or memory that Rust uses, and it
        // is a no-op when ZA storage is already off (an unwinder may have
        // turned it off).
        unsafe {
            core::arch::asm!(za_off_template!(), options(nomem, nostack, preserves_flags));
        }
    }
}

/// TPIDR2_EL0 pointing at [`TPIDR2_BLOCK`] while the guard lives; see the
/// module docs.
///
/// The guard is neither `Send` nor `Sync`: the register belongs to the
/// thread that created it, which must also drop it.
#[derive(Debug)]
#[must_use = "TPIDR2_EL0 is set only while the guard lives"]
pub(super) struct Tpidr2Set {
    /// The value to restore.
    previous: u64,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

impl Tpidr2Set {
    /// The value the guard stores in TPIDR2_EL0: the address of
    /// [`TPIDR2_BLOCK`] (never zero).
    pub(super) fn value() -> u64 {
        core::ptr::addr_of!(TPIDR2_BLOCK).addr() as u64
    }

    /// Stores [`Tpidr2Set::value`] in the calling thread's TPIDR2_EL0;
    /// `None` without SME hardware ([`detect::candidate`] is `None`), where
    /// the register does not exist. It never starts the first-use check.
    pub(super) fn new() -> Option<Self> {
        detect::candidate()?;
        // SAFETY: the candidate is present only after the operating system
        // reported FEAT_SME for this process and the detection executed this
        // probe.
        let previous = unsafe { detect::probe_thread_state() }.tpidr2;
        // SAFETY: SME is implemented and enabled (above), and the value is
        // the address of a static zeroed TPIDR2 block (module docs).
        unsafe { write_tpidr2(Self::value()) };
        Some(Self {
            previous,
            _not_send: PhantomData,
        })
    }
}

impl Drop for Tpidr2Set {
    fn drop(&mut self) {
        // SAFETY: the guard exists only where SME is implemented and enabled
        // (`new`), and `previous` is what the register held before.
        unsafe { write_tpidr2(self.previous) };
    }
}

/// Writes the calling thread's TPIDR2_EL0.
///
/// # Safety
///
/// SME must be implemented and enabled for this process
/// (`detect::candidate()` is `Some`), and `value` must be zero, the value the register held before,
/// or the address of a TPIDR2 block that stays valid while it is stored
/// there (a routine of the procedure-call standard may read the block).
unsafe fn write_tpidr2(value: u64) {
    // SAFETY: the caller guarantees SME. MSR TPIDR2_EL0 writes only that
    // register (no memory, stack or flags); nothing in this crate reads it
    // except the readiness probe, which is what the tests want it to see.
    unsafe {
        core::arch::asm!(
            tpidr2_write_template!(),
            value = in(reg) value,
            options(nomem, nostack, preserves_flags)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::lint::{Rule, lint_probe_template, report};
    use super::super::lint_scan::{lex, macro_calls};
    use super::super::test_support::{Need, hardware_or_skip, kernel_ready, sme_state};
    use super::*;

    #[test]
    fn test_sme2_state_switch_templates_are_linted() {
        let instructions = ["smstart za", "smstop za", "msr tpidr2_el0, {value}"];
        for (template, instruction) in STATE_SWITCH_TEMPLATES.iter().zip(instructions) {
            assert_eq!(*template, format!(".arch_extension sme\n{instruction}"));
            let errors = match lint_probe_template(template) {
                Ok(()) => panic!("the probe kind accepted a state switch: {template:?}"),
                Err(errors) => errors,
            };
            assert_eq!(errors.len(), 1, "{template:?}:\n{}", report(&errors));
            assert_eq!(
                errors[0].rule,
                Rule::ProbeInstruction,
                "{}",
                report(&errors)
            );
            assert_eq!(errors[0].line, 2, "{}", report(&errors));
        }
        // The probe kind is a real check here: a second statement, a
        // separator or an FP/SIMD register in a switch adds a finding.
        for broken in [
            ".arch_extension sme\nsmstart za\nsmstop za",
            ".arch_extension sme\nsmstart za; nop",
            ".arch_extension sme\nmsr tpidr2_el0, {value:d}",
            "smstart za",
        ] {
            let errors = lint_probe_template(broken).err().unwrap_or_default();
            assert!(errors.len() > 1, "{broken:?}: {}", report(&errors));
        }
        assert_eq!(TPIDR2_BLOCK.0, [0, 0], "the TPIDR2 block names a buffer");
        assert_ne!(Tpidr2Set::value(), 0);
    }

    #[test]
    fn test_sme2_state_switch_sites_use_the_templates() {
        // This file is in `STREAMING_TEXT_FILES`, so the source scan accepts
        // any streaming text in it, and the test above lints the templates
        // only through `STATE_SWITCH_TEMPLATES`. Each assembly site must
        // therefore take its text from one of the three template macros, each
        // macro at exactly one site: a template written out at a site (a
        // plain `smstart`, which would enter streaming mode) escapes both.
        let lexed = lex(include_str!("test_sme_state.rs"));
        // The first argument of each invocation (string literals are empty
        // in the lexed code, so a literal template reads as `""`).
        let mut templates: Vec<&str> = macro_calls(&lexed.code, "asm")
            .into_iter()
            .map(|after_bang| {
                let args = lexed.code[after_bang..].trim_start();
                let args = args.strip_prefix('(').unwrap_or("");
                let mut depth = 0_usize;
                let end = args
                    .char_indices()
                    .find(|&(_, ch)| match ch {
                        '(' | '[' | '{' => {
                            depth += 1;
                            false
                        }
                        ')' | ']' | '}' if depth > 0 => {
                            depth -= 1;
                            false
                        }
                        ',' | ')' | ']' | '}' => true,
                        _ => false,
                    })
                    .map_or(args.len(), |(at, _)| at);
                args[..end].trim()
            })
            .collect();
        templates.sort_unstable();
        assert_eq!(
            templates,
            [
                "tpidr2_write_template!()",
                "za_off_template!()",
                "za_on_template!()"
            ],
            "the assembly sites of this file and their templates"
        );
    }

    #[test]
    fn test_sme2_state_switches_without_sme() {
        // Without SME hardware no switch runs (it would be undefined).
        if detect::candidate().is_none() {
            assert!(ZaStorageOn::new().is_none());
            assert!(Tpidr2Set::new().is_none());
        }
    }

    #[test]
    fn test_sme2_state_switches_restore() {
        let test = "test_sme2_state_switches_restore";
        if !hardware_or_skip(test, Need::F32) {
            return;
        }
        assert_eq!(sme_state(), Some((0, 0)), "{test}: state before");
        assert!(kernel_ready(), "{test}: not ready before");
        {
            let _za = ZaStorageOn::new().expect("SME present");
            assert_eq!(sme_state(), Some((SVCR_ZA, 0)), "{test}: ZA on");
            assert!(!detect::thread_ready(), "{test}: ready with ZA on");
            {
                // Nested: the inner guard sees ZA on and leaves it on.
                let _again = ZaStorageOn::new().expect("SME present");
                assert_eq!(sme_state(), Some((SVCR_ZA, 0)));
            }
            assert_eq!(sme_state(), Some((SVCR_ZA, 0)), "{test}: inner drop");
        }
        assert_eq!(sme_state(), Some((0, 0)), "{test}: ZA off again");
        {
            let _tpidr2 = Tpidr2Set::new().expect("SME present");
            assert_eq!(sme_state(), Some((0, Tpidr2Set::value())), "{test}: TPIDR2");
            assert!(!detect::thread_ready(), "{test}: ready with TPIDR2 set");
        }
        assert_eq!(sme_state(), Some((0, 0)), "{test}: TPIDR2 restored");
        assert!(kernel_ready(), "{test}: not ready after");
        // Unwinding drops the guards too (`resume_unwind` skips the panic
        // hook, so the log shows no panic message for it).
        for state in 0..2 {
            let unwound = std::panic::catch_unwind(|| {
                let _za = (state == 0).then(|| ZaStorageOn::new().expect("SME present"));
                let _tpidr2 = (state == 1).then(|| Tpidr2Set::new().expect("SME present"));
                assert_ne!(sme_state(), Some((0, 0)), "{test}: switch {state}");
                std::panic::resume_unwind(Box::new("unwinds with the switch on"));
            });
            assert!(unwound.is_err(), "{test}: switch {state} did not unwind");
            assert_eq!(sme_state(), Some((0, 0)), "{test}: switch {state} restored");
        }
        assert!(kernel_ready(), "{test}: not ready at the end");
    }
}
