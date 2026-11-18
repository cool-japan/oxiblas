//! Tests for SME2 detection: the `sysctl` and auxv parsers, the order of the
//! detection steps, the environment override, the guarantee that nothing
//! is selected (and no SME instruction runs) without SME2 hardware, and the
//! detection report. The first-use check: `tests_detect_first_use.rs`.

use std::cell::Cell;
use std::ffi::OsStr;

use super::super::sme2_active;
use super::*;

/// `sysctl` output for `(key, value)` pairs, one `key: value` line each.
fn sysctl_output(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| format!("{key}: {value}\n"))
        .collect()
}

/// A raw auxiliary vector made of native-endian `usize` words.
fn auxv_bytes(pairs: &[(usize, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &(key, value) in pairs {
        bytes.extend_from_slice(&key.to_ne_bytes());
        bytes.extend_from_slice(&(value as usize).to_ne_bytes());
    }
    bytes
}

/// `AT_HWCAP`, the first capability word (its bits must not be mistaken for
/// `AT_HWCAP2` bits).
const AT_HWCAP: usize = 16;

/// A stand-in for `RDSVL` that must never run.
fn never_probe() -> u64 {
    panic!("RDSVL must not run unless the operating system reported SME2")
}

/// `sysctl` output of an Apple M4 Pro with 8 performance cores in two
/// clusters, as the keys describe it.
fn m4_like_output() -> String {
    sysctl_output(&[
        (KEY_FEAT_SME, "1"),
        (KEY_FEAT_SME2, "1"),
        (KEY_FEAT_SME_F64F64, "1"),
        (KEY_SME_MAX_SVL_B, "64"),
        (KEY_PERF0_PHYSICALCPU, "8"),
        (KEY_PERF0_CPUSPERL2, "4"),
    ])
}

#[test]
fn test_sme2_detect_parse_macos_m4_like() {
    let text = m4_like_output();
    // The synthetic output covers exactly the keys that are requested.
    for key in SYSCTL_KEYS {
        assert!(sysctl_value(&text, key).is_some(), "{key} missing");
    }

    let os = macos_report(&text);
    assert_eq!(
        os,
        OsSme {
            sme: true,
            sme2: true,
            f64f64: true,
            max_svl_bytes: Some(64),
            units: 2,
        }
    );
    assert!(os.offers_sme2());
    assert_eq!(
        hw_from_report(os, || 64),
        Some(Sme2Hw {
            f64: true,
            units: 2
        })
    );
    // A streaming vector length other than 512 bits is rejected even when the
    // operating system reports SME2.
    assert_eq!(hw_from_report(os, || 32), None);
    assert_eq!(hw_from_report(os, || 128), None);

    // The performance-cluster keys parse: 8 cores, 4 per L2 cluster.
    assert_eq!(sysctl_value(&text, KEY_PERF0_PHYSICALCPU), Some(8));
    assert_eq!(sysctl_value(&text, KEY_PERF0_CPUSPERL2), Some(4));

    // Without FEAT_SME_F64F64 only f32 may use the backend.
    let no_f64 = macos_report(&text.replace("FEAT_SME_F64F64: 1", "FEAT_SME_F64F64: 0"));
    assert_eq!(
        hw_from_report(no_f64, || 64),
        Some(Sme2Hw {
            f64: false,
            units: 2
        })
    );

    // A reported maximum SVL other than 64 bytes is rejected before RDSVL.
    let svl_256 = macos_report(&text.replace("sme_max_svl_b: 64", "sme_max_svl_b: 32"));
    assert!(!svl_256.offers_sme2());
    assert_eq!(hw_from_report(svl_256, never_probe), None);
}

#[test]
fn test_sme2_detect_parse_macos_m3() {
    // Verbatim output of `/usr/sbin/sysctl <SYSCTL_KEYS>` on an Apple M3.
    let text = "hw.optional.arm.FEAT_SME: 0\n\
                hw.optional.arm.FEAT_SME2: 0\n\
                hw.optional.arm.FEAT_SME_F64F64: 0\n\
                hw.optional.arm.sme_max_svl_b: 0\n\
                hw.perflevel0.physicalcpu: 4\n\
                hw.perflevel0.cpusperl2: 4\n";
    let os = macos_report(text);
    assert_eq!(
        os,
        OsSme {
            sme: false,
            sme2: false,
            f64f64: false,
            max_svl_bytes: Some(0),
            units: 1,
        }
    );
    assert!(!os.offers_sme2());
    assert_eq!(hw_from_report(os, never_probe), None);
    assert_eq!(sysctl_value(text, KEY_PERF0_PHYSICALCPU), Some(4));
    assert_eq!(sysctl_value(text, KEY_PERF0_CPUSPERL2), Some(4));
}

#[test]
fn test_sme2_detect_parse_macos_unknown_keys() {
    // An older macOS that knows FEAT_SME but not FEAT_SME2: `sysctl` prints
    // the known keys (the unknown one goes to standard error), plus keys
    // that merely share a prefix with the ones we need.
    let text = "hw.optional.arm.FEAT_SME: 1\n\
                hw.optional.arm.FEAT_SME2p1: 1\n\
                hw.optional.arm.FEAT_SME_F64F64: 1\n\
                hw.optional.arm.sme_max_svl_b: 64\n";
    let os = macos_report(text);
    assert!(os.sme);
    assert!(!os.sme2, "FEAT_SME2p1 must not count as FEAT_SME2");
    assert_eq!(hw_from_report(os, never_probe), None);

    // FEAT_SME_F64F64 and FEAT_SME2 must not count as FEAT_SME.
    let text = "hw.optional.arm.FEAT_SME_F64F64: 1\n\
                hw.optional.arm.FEAT_SME2: 1\n\
                hw.optional.arm.sme_max_svl_b: 64\n";
    let os = macos_report(text);
    assert!(!os.sme);
    assert!(os.sme2);
    assert_eq!(hw_from_report(os, never_probe), None);

    // Noise: an error line, a line without a colon, a `key=value` line,
    // non-integer and out-of-range values, extra whitespace.
    let text = "sysctl: unknown oid 'hw.optional.arm.FEAT_SME_F64F64'\n\
                garbage without a separator\n\
                hw.optional.arm.FEAT_SME_F64F64=1\n\
                hw.optional.arm.FEAT_SME2: yes\n\
                hw.optional.arm.FEAT_SME2: 2\n\
                \x20 hw.optional.arm.FEAT_SME :  1 \n\
                hw.optional.arm.sme_max_svl_b: 64\n";
    let os = macos_report(text);
    assert_eq!(
        os,
        OsSme {
            sme: true,
            sme2: false,
            f64f64: false,
            max_svl_bytes: Some(64),
            units: 1,
        }
    );

    // The first line with the key and an integer value wins.
    let text = "hw.optional.arm.FEAT_SME2: oops\nhw.optional.arm.FEAT_SME2: 1\n";
    assert_eq!(sysctl_value(text, KEY_FEAT_SME2), Some(1));

    // No output at all (every key unknown): nothing is offered, and the
    // missing maximum SVL reads as 0.
    let os = macos_report("");
    assert_eq!(
        os,
        OsSme {
            sme: false,
            sme2: false,
            f64f64: false,
            max_svl_bytes: Some(0),
            units: 1,
        }
    );
    // Even with the features present, a missing maximum SVL fails.
    let text = "hw.optional.arm.FEAT_SME: 1\nhw.optional.arm.FEAT_SME2: 1\n";
    assert!(!macos_report(text).offers_sme2());
}

#[test]
fn test_sme2_detect_parse_auxv_bits() {
    // The bit numbers re-verified against std_detect and the libc crate.
    assert_eq!(AT_HWCAP2, 26);
    assert_eq!(HWCAP2_SME, 1 << 23);
    assert_eq!(HWCAP2_SME_F64F64, 1 << 25);
    assert_eq!(HWCAP2_SME2, 1 << 37);

    let all = HWCAP2_SME | HWCAP2_SME2 | HWCAP2_SME_F64F64;
    let bytes = auxv_bytes(&[
        (33, 0xffff_0000), // AT_SYSINFO_EHDR
        (AT_HWCAP, 0xffff_ffff),
        (AT_HWCAP2, all | 0b11),
        (6, 4096), // AT_PAGESZ
        (AT_NULL, 0),
    ]);
    assert_eq!(auxv_hwcap2(&bytes), Some(all | 0b11));

    let os = linux_report(all);
    assert_eq!(
        os,
        OsSme {
            sme: true,
            sme2: true,
            f64f64: true,
            max_svl_bytes: None,
            units: 1,
        }
    );
    assert!(os.offers_sme2());
    assert_eq!(
        hw_from_report(os, || 64),
        Some(Sme2Hw {
            f64: true,
            units: 1
        })
    );
    // qemu-user's default streaming vector length is 32 bytes.
    assert_eq!(hw_from_report(os, || 32), None);

    // SME2 without SME_F64F64 serves f32 only.
    assert_eq!(
        hw_from_report(linux_report(HWCAP2_SME | HWCAP2_SME2), || 64),
        Some(Sme2Hw {
            f64: false,
            units: 1
        })
    );
    // An SME-only CPU (no SME2) is not offered.
    assert!(!linux_report(HWCAP2_SME | HWCAP2_SME_F64F64).offers_sme2());

    // Neighbouring bits do not count: MTE3 (22), SME_I16I64 (24),
    // SME_I8I32 (26), SVE2P1 (36), SME2P1 (38).
    let neighbours = (1 << 22) | (1 << 24) | (1 << 26) | (1 << 36) | (1 << 38);
    let os = linux_report(neighbours);
    assert_eq!(
        os,
        OsSme {
            sme: false,
            sme2: false,
            f64f64: false,
            max_svl_bytes: None,
            units: 1,
        }
    );
    assert_eq!(hw_from_report(os, never_probe), None);

    // The bits are read from AT_HWCAP2 only, never from AT_HWCAP.
    let hwcap_only = auxv_bytes(&[(AT_HWCAP, all), (AT_NULL, 0)]);
    assert_eq!(auxv_hwcap2(&hwcap_only), None);
}

#[test]
fn test_sme2_detect_parse_auxv_truncated() {
    let all = HWCAP2_SME | HWCAP2_SME2 | HWCAP2_SME_F64F64;
    let good = auxv_bytes(&[(AT_HWCAP, 1), (AT_HWCAP2, all), (AT_NULL, 0)]);
    assert_eq!(auxv_hwcap2(&good), Some(all));

    // Every proper prefix lacks a complete AT_NULL pair and is not trusted,
    // even when the AT_HWCAP2 pair itself is complete.
    for len in 0..good.len() {
        assert_eq!(auxv_hwcap2(&good[..len]), None, "prefix of {len} bytes");
    }

    // Whatever follows AT_NULL is ignored, including a partial word.
    let mut trailing = good.clone();
    trailing.extend_from_slice(&auxv_bytes(&[(AT_HWCAP2, 0)]));
    trailing.extend_from_slice(&[0xaa; 3]);
    assert_eq!(auxv_hwcap2(&trailing), Some(all));

    // An AT_HWCAP2 entry after the terminator does not count.
    let late = auxv_bytes(&[(AT_NULL, 0), (AT_HWCAP2, all)]);
    assert_eq!(auxv_hwcap2(&late), None);

    // A well-formed vector without AT_HWCAP2.
    let absent = auxv_bytes(&[(AT_HWCAP, all), (6, 4096), (AT_NULL, 0)]);
    assert_eq!(auxv_hwcap2(&absent), None);
}

#[test]
fn test_sme2_detect_env_override() {
    assert_eq!(DISABLE_ENV, "OXIBLAS_DISABLE_SME2");
    // Unset, empty and "0" leave the backend on.
    assert!(!env_disables(None));
    assert!(!env_disables(Some(OsStr::new(""))));
    assert!(!env_disables(Some(OsStr::new("0"))));
    // Anything else turns it off, whatever the spelling.
    for value in ["1", "yes", "true", "false", "00", " 0", "0 ", "off"] {
        assert!(env_disables(Some(OsStr::new(value))), "{value:?}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(env_disables(Some(OsStr::from_bytes(&[0xff, 0xfe]))));
    }
}

#[test]
fn test_sme2_detect_probe_runs_only_after_os_reports_sme2() {
    let calls = &Cell::new(0_u32);
    let probe = |svl_bytes: u64| {
        move || {
            calls.set(calls.get() + 1);
            svl_bytes
        }
    };
    let base = OsSme {
        sme: true,
        sme2: true,
        f64f64: false,
        max_svl_bytes: None,
        units: 1,
    };
    for os in [
        OsSme { sme: false, ..base },
        OsSme {
            sme2: false,
            ..base
        },
        OsSme {
            max_svl_bytes: Some(32),
            ..base
        },
        OsSme {
            max_svl_bytes: Some(0),
            ..base
        },
    ] {
        assert_eq!(hw_from_report(os, probe(64)), None, "{os:?}");
    }
    assert_eq!(calls.get(), 0, "RDSVL ran without an SME2 report");

    assert_eq!(
        hw_from_report(base, probe(64)),
        Some(Sme2Hw {
            f64: false,
            units: 1
        })
    );
    assert_eq!(calls.get(), 1);
    let macos = OsSme {
        max_svl_bytes: Some(64),
        f64f64: true,
        ..base
    };
    assert_eq!(
        hw_from_report(macos, probe(64)),
        Some(Sme2Hw {
            f64: true,
            units: 1
        })
    );
    // RDSVL has the last word, even when the report says 64 bytes.
    assert_eq!(hw_from_report(macos, probe(16)), None);
    assert_eq!(calls.get(), 3);
}

#[test]
fn test_sme2_detect_not_selected_without_hw() {
    // The expectation comes from the raw operating-system report, so the test
    // also holds on SME2 hardware and under QEMU, where only the checks after
    // this block apply.
    let offered = os_report().is_some_and(|os| os.offers_sme2());
    if !offered {
        assert_eq!(hw(), None);
        assert!(!enabled::<f32>());
        assert!(!enabled::<f64>());
        // Returns false before executing any SME instruction (on a CPU without
        // SME, reaching the probe would raise SIGILL here).
        assert!(!thread_ready());
        assert!(!sme2_active());
    }
    if !SME2_ALLOWED {
        assert!(!enabled::<f32>());
        assert!(!enabled::<f64>());
        assert!(!sme2_active());
    }
    // Types other than f32 and f64 never use the backend.
    assert!(!enabled::<i32>());
    assert!(!enabled::<u64>());
    assert!(!enabled::<num_complex::Complex<f32>>());
    assert!(!enabled::<num_complex::Complex<f64>>());
    // f64 implies f32, and an active backend implies a ready thread.
    if enabled::<f64>() {
        assert!(enabled::<f32>());
    }
    if sme2_active() {
        assert!(enabled::<f32>() && thread_ready());
    }
}

#[test]
fn test_sme2_detect_thread_state_ready_rule() {
    // The rule of `thread_ready`, on states no emulator thread is in.
    let ready = ThreadState {
        svcr: 0,
        tpidr2: 0,
        svl_bytes: 64,
    };
    assert!(ready.ready());
    // Streaming mode on (bit 0), ZA storage on (bit 1), both.
    for svcr in [1, 2, 3] {
        assert!(!ThreadState { svcr, ..ready }.ready(), "SVCR {svcr}");
    }
    // A pending lazy save of a caller's ZA contents.
    for tpidr2 in [1, 0x1000, u64::MAX] {
        assert!(
            !ThreadState { tpidr2, ..ready }.ready(),
            "TPIDR2 {tpidr2:#x}"
        );
    }
    // Another streaming vector length (Linux lets a thread change it).
    for svl_bytes in [0, 16, 32, 128, 256] {
        assert!(
            !ThreadState { svl_bytes, ..ready }.ready(),
            "SVL {svl_bytes} bytes"
        );
    }
}

#[test]
fn test_sme2_detect_report() {
    // Prints what the detection sees; never fails on hosts without SME.
    println!("SME2_ALLOWED (build does not cap SIMD): {SME2_ALLOWED}");
    println!("{DISABLE_ENV} = {:?}", std::env::var_os(DISABLE_ENV));
    #[cfg(target_os = "macos")]
    match sysctl_text() {
        Some(text) => println!("{SYSCTL_PATH} output:\n{text}"),
        None => println!("{SYSCTL_PATH} could not be spawned"),
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    match read_auxv_hwcap2() {
        Some(hwcap2) => println!(
            "AT_HWCAP2 = {hwcap2:#x} (SME {}, SME2 {}, SME_F64F64 {})",
            hwcap2 & HWCAP2_SME != 0,
            hwcap2 & HWCAP2_SME2 != 0,
            hwcap2 & HWCAP2_SME_F64F64 != 0
        ),
        None => println!("AT_HWCAP2 not available"),
    }
    println!("operating-system report: {:?}", os_report());
    // The candidate first: reading it never starts the first-use check, so
    // the line is in the log even when the check then ends the process (a
    // kernel instruction that this CPU does not implement raises SIGILL
    // inside the check); `tools/sme2_qemu.sh` tells that case from a SIGILL
    // in the register probe by this line. Then `hw()`: on this (ready)
    // thread it settles the verdict, which the next line shows.
    println!("candidate: {:?}", candidate());
    let reported = hw();
    println!("{}", verdict_line(verdict()));
    println!("hw(): {reported:?}");
    if hw().is_some() {
        // SAFETY: `hw()` is `Some` only after the operating system reported
        // FEAT_SME and the detection already executed this probe.
        let state = unsafe { probe_thread_state() };
        println!(
            "thread state: SVCR = {:#x}, TPIDR2_EL0 = {:#x}, RDSVL #1 = {} bytes",
            state.svcr, state.tpidr2, state.svl_bytes
        );
    } else {
        println!("thread state: not probed (no SME2 reported, or disabled)");
    }
    println!(
        "enabled f32: {}, f64: {}; thread_ready: {}",
        enabled::<f32>(),
        enabled::<f64>(),
        thread_ready()
    );
    println!(
        "SME2 backend: {}",
        if sme2_active() { "active" } else { "inactive" }
    );
    println!("SME units: {}", units());
}

/// The verdict line of the detection report: `first-use check: passed`,
/// `first-use check: failed: <line>` or `first-use check: open`.
fn verdict_line(verdict: Verdict<'_>) -> String {
    match verdict {
        Verdict::Passed => "first-use check: passed".to_owned(),
        Verdict::Failed(line) => format!("first-use check: failed: {line}"),
        Verdict::Open => "first-use check: open".to_owned(),
    }
}

/// The detection report prints the candidate before anything that can
/// start the first-use check: when a kernel instruction raises SIGILL
/// inside the check, the `candidate:` line is what tells
/// `tools/sme2_qemu.sh` that the probe passed and the check ended the
/// process. Read from the source (comments removed, string literals empty).
#[test]
fn test_sme2_detect_report_prints_the_candidate_before_the_check() {
    let lexed = super::super::lint_scan::lex(include_str!("tests_detect.rs"));
    let code = lexed.code.as_str();
    let Some(start) = code.find("fn test_sme2_detect_report()") else {
        panic!("tests_detect.rs: no test_sme2_detect_report");
    };
    let body = &code[start..];
    let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
    let candidate_at = body.find("candidate()");
    // Everything that reaches `hw()`, and so can start the check.
    let first_check = [
        "hw()",
        "enabled::<",
        "thread_ready()",
        "units()",
        "sme2_active()",
    ]
    .iter()
    .filter_map(|call| body.find(call).map(|at| (at, *call)))
    .min();
    match (candidate_at, first_check) {
        (Some(candidate_at), Some((check_at, call))) => assert!(
            candidate_at < check_at,
            "test_sme2_detect_report reaches {call} before it prints the candidate"
        ),
        other => panic!("test_sme2_detect_report: candidate and hw() calls: {other:?}"),
    }
}

#[test]
fn test_sme2_detect_report_verdict_lines() {
    assert_eq!(verdict_line(Verdict::Passed), "first-use check: passed");
    assert_eq!(verdict_line(Verdict::Open), "first-use check: open");
    let line = failure_line(&CheckFailure::Gemv);
    assert_eq!(
        verdict_line(Verdict::Failed(&line)),
        "first-use check: failed: oxiblas: the SME2 backend is off for this process: \
         the check of the SME2 GEMV failed; please report this line"
    );
    // None of the report's own lines may contain the `hw(): ` that
    // `tools/sme2_qemu.sh` looks for.
    for verdict in [Verdict::Passed, Verdict::Open, Verdict::Failed(&line)] {
        assert!(!verdict_line(verdict).contains("hw(): "));
    }
    assert!(!format!("candidate: {:?}", candidate()).contains("hw(): "));
}

#[test]
fn test_sme2_detect_parse_macos_units() {
    // M4 Pro-like: 8 performance cores, 4 per cluster: two SME units.
    assert_eq!(macos_report(&m4_like_output()).units, 2);
    // M4 Max-like: 12 performance cores in two clusters of 6.
    let max_like = m4_like_output()
        .replace("physicalcpu: 8", "physicalcpu: 12")
        .replace("cpusperl2: 4", "cpusperl2: 6");
    assert_eq!(macos_report(&max_like).units, 2);
    // M4-like: 4 performance cores, one cluster.
    let one = m4_like_output().replace("physicalcpu: 8", "physicalcpu: 4");
    assert_eq!(macos_report(&one).units, 1);
    // The unit count reaches `Sme2Hw`.
    let os = macos_report(&m4_like_output());
    assert_eq!(hw_from_report(os, || 64).map(|hw| hw.units), Some(2));

    // The pure function: the quotient rounds down, and every odd report
    // gives one unit.
    assert_eq!(macos_units(Some(8), Some(4)), 2);
    assert_eq!(macos_units(Some(10), Some(4)), 2);
    assert_eq!(macos_units(Some(16), Some(4)), 4);
    assert_eq!(macos_units(Some(4), Some(4)), 1);
    assert_eq!(
        macos_units(Some(3), Some(4)),
        1,
        "fewer cores than a cluster"
    );
    assert_eq!(macos_units(Some(0), Some(4)), 1, "no performance cores");
    assert_eq!(macos_units(Some(8), Some(0)), 1, "zero cores per cluster");
    assert_eq!(macos_units(None, Some(4)), 1, "physicalcpu missing");
    assert_eq!(macos_units(Some(8), None), 1, "cpusperl2 missing");
    assert_eq!(macos_units(None, None), 1);

    // Missing, unparsable or zero keys in the text.
    let text = m4_like_output().replace("hw.perflevel0.cpusperl2: 4\n", "");
    assert_eq!(macos_report(&text).units, 1);
    let text = m4_like_output().replace("cpusperl2: 4", "cpusperl2: four");
    assert_eq!(macos_report(&text).units, 1);
    let text = m4_like_output().replace("cpusperl2: 4", "cpusperl2: 0");
    assert_eq!(macos_report(&text).units, 1);
    // A key that merely starts with the right name is not the key.
    let text = m4_like_output().replace("hw.perflevel0.cpusperl2", "hw.perflevel0.cpusperl2x");
    assert_eq!(macos_report(&text).units, 1);
}

#[test]
fn test_sme2_detect_parse_linux_units() {
    let all = HWCAP2_SME | HWCAP2_SME2 | HWCAP2_SME_F64F64;
    assert_eq!(linux_report(all).units, 1);
    assert_eq!(linux_report(0).units, 1);
    assert_eq!(
        hw_from_report(linux_report(all), || 64).map(|hw| hw.units),
        Some(1)
    );
    // A report with a zero unit count (no parser makes one) still yields one
    // unit.
    let zero = OsSme {
        units: 0,
        ..linux_report(all)
    };
    assert_eq!(hw_from_report(zero, || 64).map(|hw| hw.units), Some(1));
}

#[test]
fn test_sme2_detect_units_override_nests_and_is_per_thread() {
    let detected = hw().map_or(1, |hw| hw.units);
    assert!(detected >= 1);
    assert_eq!(units(), detected);
    {
        let _two = override_units(2);
        assert_eq!(units(), 2);
        {
            let _three = override_units(3);
            assert_eq!(units(), 3);
            // Another thread sees the detected count.
            let elsewhere = std::thread::spawn(units).join();
            assert!(
                matches!(elsewhere, Ok(seen) if seen == detected),
                "{elsewhere:?}"
            );
        }
        assert_eq!(units(), 2, "the inner guard restores the outer override");
        // An override of 0 counts as 1 unit.
        let _zero = override_units(0);
        assert_eq!(units(), 1);
    }
    assert_eq!(units(), detected);
}

#[test]
fn test_sme2_detect_pretend_not_ready_is_per_thread() {
    // What `thread_ready` says on this host without the switch (false
    // without SME2, true under QEMU): a fresh thread says the same.
    // `thread_ready` never waits for a first-use check that another thread
    // runs (it answers `false` meanwhile), so the verdict is settled first
    // with the waiting form: from then on its answer is stable.
    let _ = hw();
    assert!(!pretends_not_ready());
    let natural = thread_ready();
    // Setting returns the previous value; the switch answers first.
    assert!(!set_pretend_not_ready(true));
    let (pretending, ready, active) = (pretends_not_ready(), thread_ready(), sme2_active());
    assert!(set_pretend_not_ready(true), "the second set sees the first");
    // Not visible on another thread.
    let elsewhere = std::thread::spawn(|| (pretends_not_ready(), thread_ready())).join();
    // Cleared again (before any assertion can fail); the previous value
    // comes back.
    assert!(set_pretend_not_ready(false));
    assert!(!set_pretend_not_ready(false));
    assert!(pretending);
    assert!(!ready, "thread_ready while pretending");
    assert!(!active, "sme2_active while pretending");
    assert!(
        matches!(elsewhere, Ok((false, other)) if other == natural),
        "{elsewhere:?}"
    );
    assert!(!pretends_not_ready());
    assert_eq!(thread_ready(), natural);
}

#[test]
fn test_sme2_detect_ready_countdown_counts_calls_per_thread() {
    // What `thread_ready` says on this host without a countdown (false
    // without SME2, true under QEMU). The verdict is settled first with the
    // waiting form (see the test above), so the answer is stable.
    let _ = hw();
    let natural = thread_ready();
    assert_eq!(
        set_ready_countdown(Some(2)),
        None,
        "a countdown was running"
    );
    // Two calls answer as usual, every later one `false`; each call counts.
    let answers = [
        thread_ready(),
        thread_ready(),
        thread_ready(),
        thread_ready(),
    ];
    let left = set_ready_countdown(Some(1));
    // A call that the pretend flag answers still counts down.
    let previous = set_pretend_not_ready(true);
    let pretending = (thread_ready(), sme2_active());
    set_pretend_not_ready(previous);
    let after_pretend = (set_ready_countdown(Some(3)), thread_ready());
    // Not visible on another thread, which answers as usual.
    let elsewhere = std::thread::spawn(|| (set_ready_countdown(None), thread_ready())).join();
    // Stopped again (before any assertion can fail); the calls left come
    // back.
    let stopped = set_ready_countdown(None);
    assert_eq!(answers, [natural, natural, false, false]);
    assert_eq!(left, Some(0), "the calls did not count down");
    assert_eq!(pretending, (false, false));
    assert_eq!(
        after_pretend,
        (Some(0), natural),
        "a call under the pretend flag did not count down"
    );
    assert!(
        matches!(elsewhere, Ok((None, other)) if other == natural),
        "{elsewhere:?}"
    );
    assert_eq!(stopped, Some(2));
    assert_eq!(set_ready_countdown(None), None);
    assert_eq!(thread_ready(), natural);
}
