//! Run-time detection of SME2 hardware, in Pure Rust (no FFI, no libc).
//!
//! [`hw`] runs the detection once per process. Each step runs only when the
//! previous one succeeded:
//!
//! 1. **Environment.** `OXIBLAS_DISABLE_SME2` set to anything other than an
//!    empty string or `0` turns the backend off ([`env_disables`]).
//! 2. **Operating-system report**, obtained without executing any SME
//!    instruction. On macOS this is one call of `/usr/sbin/sysctl` (an
//!    absolute path, so a different `sysctl` earlier in `PATH` is never
//!    used) for the `hw.optional.arm.*` feature keys and
//!    `hw.optional.arm.sme_max_svl_b`, which must be 64. On Linux and Android
//!    it is the `AT_HWCAP2` word of `/proc/self/auxv`, which qemu-user
//!    emulates as well. Other operating systems give no report, so SME2 stays
//!    off there. FEAT_SME and FEAT_SME2 are required; `f64` additionally
//!    needs FEAT_SME_F64F64.
//! 3. **Register probe.** Only after the operating system reported SME2 does
//!    [`hw_from_report`] execute `RDSVL`, which must return 64 bytes (a
//!    512-bit streaming vector length). On a CPU without SME the instruction
//!    is undefined and raises SIGILL, which is why the probe comes last.
//!
//! [`thread_ready`] repeats the register probe on every call, on the thread
//! that is about to run SME code.

use std::any::TypeId;
use std::ffi::OsStr;
use std::sync::OnceLock;

use super::SME2_ALLOWED;

#[cfg(test)]
#[path = "tests_detect.rs"]
mod tests;

/// Environment variable that turns the SME2 backend off for the whole process.
const DISABLE_ENV: &str = "OXIBLAS_DISABLE_SME2";

/// Streaming vector length, in bytes, that the kernels are written for (512
/// bits, the SVL of the Apple M4).
const REQUIRED_SVL_BYTES: u64 = 64;

/// SME2 hardware available to this process, as found by [`hw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sme2Hw {
    /// FEAT_SME_F64F64 is present, so `f64` products may use SME2 as well.
    pub(crate) f64: bool,
}

/// SME features as reported by the operating system, read without executing
/// any SME instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OsSme {
    /// FEAT_SME.
    sme: bool,
    /// FEAT_SME2.
    sme2: bool,
    /// FEAT_SME_F64F64.
    f64f64: bool,
    /// Maximum streaming vector length in bytes, when the operating system
    /// reports one (macOS `hw.optional.arm.sme_max_svl_b`). Linux reports
    /// none, so there only `RDSVL` tells.
    max_svl_bytes: Option<u64>,
}

impl OsSme {
    /// Whether the report allows the backend: SME and SME2, and a 512-bit
    /// maximum streaming vector length when the operating system reports one.
    fn offers_sme2(&self) -> bool {
        self.sme
            && self.sme2
            && self
                .max_svl_bytes
                .is_none_or(|svl| svl == REQUIRED_SVL_BYTES)
    }
}

/// The calling thread's SME state, as read by [`probe_thread_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ThreadState {
    /// `SVCR`: bit 0 is PSTATE.SM (streaming mode), bit 1 is PSTATE.ZA (ZA
    /// storage on).
    pub(super) svcr: u64,
    /// `TPIDR2_EL0`: non-zero while a lazy save of a caller's ZA contents is
    /// pending (SME procedure-call standard).
    pub(super) tpidr2: u64,
    /// `RDSVL #1`: the streaming vector length in bytes.
    pub(super) svl_bytes: u64,
}

/// The detection result, computed once per process by [`hw`].
static HW: OnceLock<Option<Sme2Hw>> = OnceLock::new();

/// The SME2 hardware of this process, detected once (see the module docs).
///
/// `None` means the SME2 backend must not run: it was disabled through the
/// environment, the operating system does not report SME2, or the streaming
/// vector length is not 512 bits.
pub(crate) fn hw() -> Option<&'static Sme2Hw> {
    HW.get_or_init(detect_hw).as_ref()
}

/// Whether the SME2 backend may run for element type `T` in this process.
///
/// `f32` needs the hardware of [`hw`], and `f64` additionally needs
/// FEAT_SME_F64F64; every other type is served by the other backends. Always
/// `false`, without running the detection, when the build caps SIMD
/// (`SME2_ALLOWED` in the parent module). In test builds it is also `false`
/// on a thread that holds a `test_support::force_off` guard.
pub(crate) fn enabled<T: 'static>() -> bool {
    if !SME2_ALLOWED {
        return false;
    }
    #[cfg(test)]
    if super::test_support::forced_off() {
        return false;
    }
    let Some(hw) = hw() else {
        return false;
    };
    let ty = TypeId::of::<T>();
    ty == TypeId::of::<f32>() || (ty == TypeId::of::<f64>() && hw.f64)
}

/// Whether the calling thread may enter a streaming region now.
///
/// `true` only when [`hw`] found SME2 hardware and the thread is in the state
/// that the SME procedure-call standard calls "off": SVCR is zero (neither
/// streaming mode nor ZA storage is on), TPIDR2_EL0 is zero (no lazy save of
/// a caller's ZA contents is pending), and the streaming vector length is
/// still 512 bits (Linux lets a thread change it). In any other state the
/// backend declines rather than commit or clobber someone else's ZA contents.
///
/// The check reads three registers. It must run on the thread that executes
/// the SME code, before any output is written. Without SME2 hardware it
/// returns `false` without executing any SME instruction.
pub(crate) fn thread_ready() -> bool {
    if hw().is_none() {
        return false;
    }
    // SAFETY: `hw()` is `Some` only after the operating system reported
    // FEAT_SME for this process and the detection already executed this
    // probe, so the SME register reads and RDSVL are implemented and enabled.
    let state = unsafe { probe_thread_state() };
    state.svcr == 0 && state.tpidr2 == 0 && state.svl_bytes == REQUIRED_SVL_BYTES
}

/// Runs the detection steps of the module docs; called once by [`hw`].
fn detect_hw() -> Option<Sme2Hw> {
    if env_disables(std::env::var_os(DISABLE_ENV).as_deref()) {
        return None;
    }
    let os = os_report()?;
    hw_from_report(os, || {
        // SAFETY: `hw_from_report` calls this closure only after the operating
        // system reported FEAT_SME and FEAT_SME2 for this process, so the
        // probe's SME instructions are implemented and enabled.
        unsafe { probe_thread_state() }.svl_bytes
    })
}

/// Whether a value of `OXIBLAS_DISABLE_SME2` turns the backend off.
///
/// Any value other than an empty string or `0` does, including values that
/// are not valid Unicode. Taking the value as an argument keeps this testable
/// without changing the process environment.
fn env_disables(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Combines the operating-system report with the register probe.
///
/// `read_svl_bytes` executes `RDSVL`. It is called only when the report
/// offers SME2, because on a CPU without SME the instruction raises SIGILL.
fn hw_from_report(os: OsSme, read_svl_bytes: impl FnOnce() -> u64) -> Option<Sme2Hw> {
    if !os.offers_sme2() || read_svl_bytes() != REQUIRED_SVL_BYTES {
        return None;
    }
    Some(Sme2Hw { f64: os.f64f64 })
}

/// The register probe: SVCR, TPIDR2_EL0 and `RDSVL #1`.
///
/// All three are legal inside and outside streaming mode, and none touches
/// memory, the stack or the condition flags. The template is also checked by
/// the static template lint (`tests_lint`), which allows exactly these reads.
macro_rules! probe_template {
    () => {
        concat!(
            ".arch_extension sme\n",
            "mrs {svcr}, svcr\n",
            "mrs {tpidr2}, tpidr2_el0\n",
            "rdsvl {svl}, #1",
        )
    };
}

/// The probe template as text, for the static template lint.
#[cfg(test)]
pub(super) const PROBE_TEMPLATE: &str = probe_template!();

/// Reads the calling thread's SME state.
///
/// # Safety
///
/// The operating system must have reported FEAT_SME for this process (see
/// [`hw_from_report`]). On a CPU without SME these instructions are undefined
/// and the process receives SIGILL.
#[inline(never)]
pub(super) unsafe fn probe_thread_state() -> ThreadState {
    let svcr: u64;
    let tpidr2: u64;
    let svl_bytes: u64;
    // SAFETY: the caller guarantees that SME is implemented and enabled for
    // this process. MRS SVCR, MRS TPIDR2_EL0 and RDSVL only read registers
    // (no memory, no stack, no flags) and are legal whether or not the thread
    // is in streaming mode. The block is deliberately not `pure`: SVCR and
    // TPIDR2_EL0 change between calls, so it must not be merged or hoisted.
    unsafe {
        core::arch::asm!(
            probe_template!(),
            svcr = out(reg) svcr,
            tpidr2 = out(reg) tpidr2,
            svl = out(reg) svl_bytes,
            options(nomem, nostack, preserves_flags),
        );
    }
    ThreadState {
        svcr,
        tpidr2,
        svl_bytes,
    }
}

// ---------------------------------------------------------------------------
// macOS: `sysctl hw.optional.arm.*`
// ---------------------------------------------------------------------------

/// Absolute path of the `sysctl` tool.
#[cfg(target_os = "macos")]
const SYSCTL_PATH: &str = "/usr/sbin/sysctl";

/// FEAT_SME (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME: &str = "hw.optional.arm.FEAT_SME";
/// FEAT_SME2 (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME2: &str = "hw.optional.arm.FEAT_SME2";
/// FEAT_SME_F64F64 (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME_F64F64: &str = "hw.optional.arm.FEAT_SME_F64F64";
/// Maximum streaming vector length in bytes (`0` without SME).
#[cfg(any(target_os = "macos", test))]
const KEY_SME_MAX_SVL_B: &str = "hw.optional.arm.sme_max_svl_b";
/// Number of performance cores.
#[cfg(any(target_os = "macos", test))]
const KEY_PERF0_PHYSICALCPU: &str = "hw.perflevel0.physicalcpu";
/// Number of performance cores sharing one L2, i.e. one cluster.
#[cfg(any(target_os = "macos", test))]
const KEY_PERF0_CPUSPERL2: &str = "hw.perflevel0.cpusperl2";

/// The keys read in the single `sysctl` call.
///
/// The two `hw.perflevel0.*` keys describe the performance cores:
/// `physicalcpu / cpusperl2` is the number of performance clusters, and an
/// Apple M4 has one SME unit per performance cluster.
#[cfg(any(target_os = "macos", test))]
const SYSCTL_KEYS: [&str; 6] = [
    KEY_FEAT_SME,
    KEY_FEAT_SME2,
    KEY_FEAT_SME_F64F64,
    KEY_SME_MAX_SVL_B,
    KEY_PERF0_PHYSICALCPU,
    KEY_PERF0_CPUSPERL2,
];

/// Runs `sysctl` once for [`SYSCTL_KEYS`] and returns its standard output.
///
/// The exit status is ignored on purpose: a key unknown to the running macOS
/// version makes `sysctl` report it on standard error and exit with status
/// 1, but it still prints the known keys. `None` when the tool cannot be
/// spawned (for example inside a sandbox); the backend then stays off.
#[cfg(target_os = "macos")]
fn sysctl_text() -> Option<String> {
    let output = std::process::Command::new(SYSCTL_PATH)
        .args(SYSCTL_KEYS)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The operating-system report on macOS.
#[cfg(target_os = "macos")]
fn os_report() -> Option<OsSme> {
    sysctl_text().map(|text| macos_report(&text))
}

/// The value of `key` in `sysctl` output made of `key: value` lines, parsed
/// as an unsigned integer.
///
/// Keys must match exactly, so `hw.optional.arm.FEAT_SME2p1` is not
/// `hw.optional.arm.FEAT_SME2`. Lines without a colon and values that are not
/// integers are skipped; the first line with the key and an integer value
/// wins.
#[cfg(any(target_os = "macos", test))]
fn sysctl_value(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim() == key {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

/// Decodes the macOS report.
///
/// A feature counts as present only when its key reads exactly `1`; a
/// missing `sme_max_svl_b` reads as 0, which fails the 512-bit requirement.
#[cfg(any(target_os = "macos", test))]
fn macos_report(text: &str) -> OsSme {
    let present = |key: &str| sysctl_value(text, key) == Some(1);
    OsSme {
        sme: present(KEY_FEAT_SME),
        sme2: present(KEY_FEAT_SME2),
        f64f64: present(KEY_FEAT_SME_F64F64),
        max_svl_bytes: Some(sysctl_value(text, KEY_SME_MAX_SVL_B).unwrap_or(0)),
    }
}

// ---------------------------------------------------------------------------
// Linux and Android: `AT_HWCAP2` from `/proc/self/auxv`
// ---------------------------------------------------------------------------
//
// The constants are the Linux arm64 uapi values. They match the toolchain's
// `std_detect` (`detect/os/linux/auxvec.rs`: `AT_HWCAP2 = 26`;
// `detect/os/linux/aarch64.rs`: `sme` is bit 23, `smef64f64` bit 25 and
// `sme2` bit 37 of `AT_HWCAP2`) and the `libc` crate (`AT_HWCAP2 = 26`;
// aarch64 Android `HWCAP2_SME = 1 << 23`, `HWCAP2_SME_F64F64 = 1 << 25`).

/// `AT_NULL`: the key that terminates the auxiliary vector.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const AT_NULL: usize = 0;
/// `AT_HWCAP2`: the key of the second hardware-capability word.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const AT_HWCAP2: usize = 26;
/// `HWCAP2_SME`: FEAT_SME.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME: u64 = 1 << 23;
/// `HWCAP2_SME_F64F64`: FEAT_SME_F64F64.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME_F64F64: u64 = 1 << 25;
/// `HWCAP2_SME2`: FEAT_SME2.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME2: u64 = 1 << 37;

/// Reads `AT_HWCAP2` from `/proc/self/auxv`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_auxv_hwcap2() -> Option<u64> {
    let bytes = std::fs::read("/proc/self/auxv").ok()?;
    auxv_hwcap2(&bytes)
}

/// The operating-system report on Linux and Android.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_report() -> Option<OsSme> {
    read_auxv_hwcap2().map(linux_report)
}

/// Finds `AT_HWCAP2` in a raw auxiliary vector: native-endian `(key, value)`
/// pairs of `usize` words, terminated by an `AT_NULL` key.
///
/// Returns `None` when no `AT_HWCAP2` entry precedes `AT_NULL`, and also when
/// the data ends before a complete `AT_NULL` pair: a truncated vector is not
/// trusted. With 32-bit words the capability bits above 31 (SME2 among them)
/// cannot occur, so SME2 reads as absent there.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn auxv_hwcap2(bytes: &[u8]) -> Option<u64> {
    const WORD: usize = core::mem::size_of::<usize>();
    let mut hwcap2 = None;
    for pair in bytes.chunks_exact(2 * WORD) {
        let (key, value) = pair.split_at(WORD);
        let key = usize::from_ne_bytes(key.try_into().ok()?);
        let value = usize::from_ne_bytes(value.try_into().ok()?);
        match key {
            AT_NULL => return hwcap2,
            AT_HWCAP2 => hwcap2 = Some(value as u64),
            _ => {}
        }
    }
    None
}

/// Decodes `AT_HWCAP2`.
///
/// Linux reports no maximum streaming vector length, so only the `RDSVL`
/// probe checks it.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn linux_report(hwcap2: u64) -> OsSme {
    OsSme {
        sme: hwcap2 & HWCAP2_SME != 0,
        sme2: hwcap2 & HWCAP2_SME2 != 0,
        f64f64: hwcap2 & HWCAP2_SME_F64F64 != 0,
        max_svl_bytes: None,
    }
}

// ---------------------------------------------------------------------------
// Other operating systems
// ---------------------------------------------------------------------------

/// Other operating systems offer no supported way to ask, so SME2 stays off.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
fn os_report() -> Option<OsSme> {
    None
}
