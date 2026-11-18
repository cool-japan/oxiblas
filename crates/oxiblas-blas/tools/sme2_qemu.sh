#!/usr/bin/env bash
# sme2_qemu.sh -- runs the SME2 test suites of this checkout under QEMU, or
# on a machine that has SME2 itself (--hardware, see Hardware below).
#
# On an Apple M4 (or another machine with SME2), the one command to run, from
# the checkout's root directory:
#   crates/oxiblas-blas/tools/sme2_qemu.sh --hardware --toolchain 1.95.0
# and the one file to send back: target/sme2-qemu/sme2-qemu-results.tar.gz
# (it runs the unit tests and then the public-path stage, see below: on
# such a machine the stage's debug build is the most informative part).
# (Without rustup's toolchain 1.95.0, which `rustup toolchain install 1.95.0`
# adds, leave out `--toolchain 1.95.0`: the default toolchain is then used
# and its version recorded; one older than the workspace's rust-version is
# refused.)
#
# The `sme2` Cargo feature of oxiblas-blas (opt-in, off by default) adds GEMM
# and GEMV kernels for the Arm Scalable Matrix Extension 2 (as in Apple M4).
# Without SME2 hardware they are tested under QEMU user-mode emulation: the
# test binaries are built for aarch64-unknown-linux-gnu and cargo runs them
# through `qemu-aarch64 -cpu <CPU string>`, which emulates SME2. This script
# runs two passes over the tests of the checkout it sits in:
#
#   positive  CPU max,sme=on,sme512=on,sme_fa64=off,sme-default-vector-length=64
#             (streaming vector length 512 bits, like Apple M4), with
#             OXIBLAS_TEST_REQUIRE_SME2=1: SME2 must run, a skip is a failure.
#   negative  CPU max,sme=on,sme256=on,sme512=off,sme_fa64=off,sme-default-vector-length=32
#             (largest streaming vector length 256 bits), with
#             OXIBLAS_TEST_EXPECT_NO_SME2=1: the backend must decline SME2,
#             and nothing may fail.
# Both passes also set OXIBLAS_TEST_SME2_MARKERS=1: the tests' gates
# (test_support::hardware_or_skip and backend_or_skip, called by every test
# that executes SME2 code) then write `sme2-gate: ran <test>` or
# `sme2-gate: skipped <test>` for each decision, uncaptured, and the tests that
# prove the backend declines without SME2 (the ungated
# *_declines_without_backend tests) write `sme2-gate: declined <test>`. The
# verdict counts those lines (see Exit status).
#
# Public-path stage: after its unit tests, every pass (also --hardware's)
# runs the integration tests sme2_public_path and gemm_rayon_pool against
# the library as users build it (without cfg(test)), twice, in the pass's
# own container, copy or host, with its target, toolchain, CPU string and
# gate variables:
#   cargo test -p oxiblas-blas --features sme2,parallel --release
#         [--target aarch64-unknown-linux-gnu] --test sme2_public_path
#         --test gemm_rayon_pool --no-fail-fast
#   the same without --release (a debug build: debug assertions on)
# sme2_public_path makes public gemm and gemv calls large enough for SME2,
# sequential and parallel, against a plain reference, and writes the same
# `sme2-gate: ran|skipped <test>` lines from the public sme2_active() and the
# three gate variables (its header states the rule); gemm_rayon_pool checks
# that a GEMM leaves rayon's global pool unbuilt. The debug build is the only
# place where the debug assertion of the SME2 parallel driver runs outside
# cfg(test), and a machine with several SME units (an Apple M4; QEMU on
# Linux reports one) is the first on which that driver splits a call over
# more than one unit in a non-test build: an owner of SME2 hardware gets
# both from the one --hardware command. The stage runs whatever the unit
# tests gave, and the debug build whatever the release build gave;
# --no-fail-fast lets both binaries run when one fails. Neither --filter
# nor the extra cargo arguments after `--` reach it: they apply to the unit
# tests only, also where they change the build (a feature such as
# max-simd-128, a profile), while the stage always builds the commands
# above, so with such an argument the unit tests and the stage test
# different builds. --no-public-path leaves the stage out (a focused run);
# by default every mode runs it.
#
# SVE stays on in both (with sve=off QEMU also turns SME off). qemu-user 10.1
# or newer is required for SME2.
#
# Hosts:
#   x86_64 Linux with Docker (the main case). An x86_64 image is built; it
#     cross-compiles with the Rust target aarch64-unknown-linux-gnu and
#     gcc-aarch64-linux-gnu as the linker, and runs the test binaries with
#     `qemu-aarch64 -L <aarch64 sysroot of the cross compiler>`.
#   arm64 macOS or Linux with Docker (Docker Desktop on Apple silicon). An
#     arm64 image is built; it compiles natively and runs under qemu-aarch64.
#   --native: no Docker. A Linux host with qemu-aarch64 >= 10.1 on PATH,
#     rustup with the toolchain and its target aarch64-unknown-linux-gnu
#     (`rustup target add --toolchain 1.95.0 aarch64-unknown-linux-gnu`),
#     rsync, and on x86_64 an aarch64 cross linker (aarch64-linux-gnu-gcc,
#     e.g. Debian/Ubuntu package gcc-aarch64-linux-gnu; another linker can be
#     named with CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER).
#   --hardware: no Docker, no QEMU (see Hardware).
# The image (one Dockerfile, embedded below, for both architectures) is
# ubuntu:26.04 with its qemu-user (10.2) and Rust from rustup; building it
# needs the network once (apt, rustup), and cargo fetches crates from
# crates.io. Nothing else goes over the network and nothing is sent anywhere.
# The image is pinned only by the Rust toolchain and the text of the
# Dockerfile (both are in its tag): the base image ubuntu:26.04, the apt
# packages (qemu-user, the cross compiler) and rustup-init are whatever the
# registries serve on the day of the build, so two machines can build
# different images under one tag. The summary therefore records the image ID,
# the QEMU version and the qemu-user package version.
#
# Hardware (--hardware): for a machine that has SME2 itself, an Apple M4
# under macOS or an aarch64 Linux machine with SME2. No Docker, no QEMU and
# no cross target: the checkout is copied (rsync, as in --native) to
# target/sme2-qemu/hardware/src, so it is never written, and the host's own
# cargo (the default toolchain; with --toolchain TC, rustup's `cargo +TC`)
# builds and runs its tests (the unit tests, then the public-path stage) in
# target/sme2-qemu/hardware/target. Before
# anything is built it refuses (exit 2) a toolchain older than the
# workspace's rust-version (the key of [workspace.package] in Cargo.toml,
# which must be readable), and whatever would put a program between cargo
# and the test binaries, so that they would not run on this CPU:
# CARGO_TARGET_<host triple>_RUNNER or CARGO_BUILD_TARGET in the environment
# (set at all), --config or --target among the cargo arguments (before a
# second --), and in a cargo config file cargo reads there (.cargo/config.toml
# and .cargo/config of the copy, which are the checkout's, and of every parent
# up to /, then CARGO_HOME's) a runner for the host triple or for a cfg(...)
# expression (not evaluated), build.target, an include key, or what its scan
# cannot read; the message names the source. The summary records the
# toolchain and `target runner: none` with the files read. One pass,
# the positive one, with OXIBLAS_TEST_REQUIRE_SME2=1 and
# OXIBLAS_TEST_SME2_MARKERS=1 (and OXIBLAS_TEST_EXPECT_NO_SME2=0):
#   cargo test -p oxiblas-blas --features sme2,parallel --lib --release
#         <filter> <extra>
# (an optimised build: one of the tests is vacuous in debug builds), judged
# by the rule of the positive QEMU pass, and the public-path stage, judged
# by its positive rule (see Exit status). Then
#   cargo test -p oxiblas-blas --features sme2,parallel --lib --release
#         test_sme2_detect_report -- --nocapture
# prints what the library's detection read (log detect-report.log), and its
# line `hw(): ...` must read `hw(): Some(...)`, the report ending cleanly:
# else the verdict is FAIL, with the reason "SME2 was not detected" and the
# values read when it read another value, and "the detection report did not
# run (see detect-report.log)" with the signal or cargo's exit status when
# it never got to that line (a build failure, a SIGILL in the probe); but
# "the first-use self-check did not end" when the report printed a present
# candidate before it ended (it prints that line before it starts the
# check, so a SIGILL there comes from a kernel instruction). A report
# `hw(): Some(... f64: false ...)` (no FEAT_SME_F64F64) fails too, with the
# reason "FEAT_SME_F64F64 is absent": OXIBLAS_TEST_REQUIRE_SME2 requires the
# f64 kernels, and the f64 tests of the stage, whose gate asks the public
# sme2_active() (the f32 backend), would write `sme2-gate: ran` while their
# products run on NEON. The
# report also shows the detection's two stages: `candidate: ...` (what the
# operating system and the register probe report) and `first-use check:
# passed | failed: <line> | open` (the kernels' checks against plain Rust).
# When it reads another value than Some(...) with a present candidate, the
# reason is exact: "the first-use self-check failed on this CPU" with the
# check's line, or "the first-use self-check did not run" for an open
# verdict, instead of "SME2 was not detected". A report without those two
# lines (an older library) is judged as before. The
# summary also holds that detection report (the raw values the detection
# read, the SME units, the thread state) and the values this script reads
# itself: on macOS the sysctl keys that detect.rs names (its KEY_* constants:
# the hw.optional.arm.* features, hw.optional.arm.sme_max_svl_b, which must be
# 64, hw.perflevel0.physicalcpu and hw.perflevel0.cpusperl2), each read with
# /usr/sbin/sysctl -n, and the CPU brand string machdep.cpu.brand_string; on
# Linux the Features line of /proc/cpuinfo (with whether it lists sme, sme2
# and smef64f64) and /proc/sys/abi/sme_default_vector_length, the streaming
# vector length in bytes a process starts with (usually 32): the backend
# needs 64, and when the detection fails with another value there the reason
# says so (the script never changes it). A SIGILL of a test binary on SME2
# hardware means that an SME instruction (such as mrs tpidr2_el0 or rdsvl)
# is not usable from user space; the reason says so too. Needs: macOS or
# Linux on aarch64, cargo, and rsync (macOS has one).
#
# Usage:
#   sme2_qemu.sh [options] [-- <extra cargo test arguments>]
#
# Usual invocations (from anywhere; the checkout is the one the script is in):
#   crates/oxiblas-blas/tools/sme2_qemu.sh                 # both passes
#   crates/oxiblas-blas/tools/sme2_qemu.sh --positive      # positive pass only
#   crates/oxiblas-blas/tools/sme2_qemu.sh --filter test_sme2_smoke -- -- --nocapture
#                                     # the stage runs too; for the filtered
#                                     # unit tests alone add --no-public-path
#   crates/oxiblas-blas/tools/sme2_qemu.sh --native        # Linux, no Docker
#   crates/oxiblas-blas/tools/sme2_qemu.sh --hardware      # Apple M4 and the like
#   crates/oxiblas-blas/tools/sme2_qemu.sh --self-test     # the verdict rules
#
# Each QEMU pass runs
#   cargo +<toolchain> test -p oxiblas-blas --features sme2,parallel --lib
#         --release --target aarch64-unknown-linux-gnu <filter> <extra>
# with the filter `sme2` by default (the hardware pass the same without
# --target), then the public-path stage. Without a filter the whole lib
# suite runs, which takes well over half an hour under QEMU.
#
# Options:
#   --positive         run the positive pass
#   --negative         run the negative pass
#   --both             run both passes (the default when neither is given)
#   --filter NAME      test name filter of the unit tests (default: sme2;
#                      '' for none); the public-path stage takes none
#   --no-public-path   leave out the public-path stage (see above)
#   --toolchain TC     Rust toolchain (default 1.95.0; with --hardware the
#                      default is the host's default toolchain, see Hardware)
#   --native           run without Docker (see Hosts)
#   --hardware         run on this machine's own SME2, without Docker and
#                      QEMU (see Hardware): the positive pass only, so not
#                      with --negative, --both, --native, --platform,
#                      --image-only or --rebuild-image
#   --self-test        check the script's own logic without Docker, QEMU or
#                      cargo: the verdict rules on canned pass logs (QEMU
#                      passes, the public-path stage and --hardware), the
#                      split of a pass's output into its logs (also run
#                      twice, cut short, and by the signal handler), the
#                      container's step runner with stand-ins (both
#                      streams joined), the run lock,
#                      stopping a real
#                      process group (sh, sleep, tee), the runner checks on
#                      canned cargo config files and a scratch directory
#                      tree, the toolchain rules, and --hardware end to end:
#                      this script, copied into a skeleton checkout there
#                      (its target/ a link), run with stand-ins for uname,
#                      rustc, cargo, rsync and tar on PATH (a PASS with the
#                      stage's commands, logs and archive, a failing stage,
#                      a stage without ran lines, failing unit tests,
#                      --no-public-path, a missing or stale
#                      sme2_qemu_verdicts.sh, the
#                      toolchain floor, runner refusals, the run lock, the
#                      detection report's status, SIGTERM in the tests, in
#                      the stage and while the archive is written); it
#                      writes only in
#                      target/sme2-qemu-selftest.<pid>/ (removed at the end),
#                      exit 0 when every expectation holds, else 1; it takes
#                      no other option. The self-test is
#                      sme2_qemu_selftest.sh beside this script; without it
#                      --self-test stops with exit 2
#   --reset            first remove this checkout's build volume (or, with
#                      --native, target/sme2-qemu/native; with --hardware,
#                      target/sme2-qemu/hardware), then run
#   --rebuild-image    rebuild the Docker image even if it exists
#   --image-only       build (if needed) and check the image, print its tag
#                      and versions, and stop without running tests (a
#                      testing aid, e.g. with --platform amd64)
#   --platform ARCH    amd64 or arm64: force the container platform instead
#                      of the Docker server's own. A testing aid: on an arm64
#                      Mac, `--platform amd64` exercises the x86_64 path
#                      (cross-compiling, under Docker's x86_64 emulation, so
#                      it is very slow).
#   -h, --help         print this text
# Everything after `--` is appended to the unit tests' cargo test command
# only (not to the stage's, so a feature or profile given there changes
# the unit tests' build and not the stage's); arguments for the test binary
# itself go after a second `--`.
#
# Environment:
#   OXIBLAS_*          forwarded to the tests, except OXIBLAS_QEMU_LOCK_DIR and
#                      the three gate variables OXIBLAS_TEST_REQUIRE_SME2,
#                      OXIBLAS_TEST_EXPECT_NO_SME2 and OXIBLAS_TEST_SME2_MARKERS,
#                      which each pass sets itself (positive: 1, 0, 1;
#                      negative: 0, 1, 1) whatever the caller's environment
#                      holds. The others are passed as they are, so a variable
#                      you set can contradict a pass (e.g. OXIBLAS_DISABLE_SME2=1
#                      makes the positive pass fail).
#   RUSTFLAGS, RUST_TEST_THREADS, RUST_BACKTRACE, CARGO_BUILD_JOBS
#                      forwarded as well. (--native and --hardware pass the
#                      whole environment on, apart from the gate variables.)
#   LC_ALL             set to C for the script and all it runs (bytes, not
#                      characters: no byte of a log can stop awk or grep)
#   OXIBLAS_QEMU_LOCK_DIR
#                      optional. When set, every container run of this script
#                      first takes this lock, so that several checkouts on one
#                      machine run one at a time (a small Docker VM cannot hold
#                      several release builds at once). The lock is the
#                      directory itself (mkdir is atomic) with a file `pid`
#                      inside naming its owner; the owner removes it at the
#                      end. A lock is stale when its owner process is gone, or
#                      when it has had no `pid` file for more than a minute. A
#                      stale lock is removed only by the holder of a second,
#                      short-lived directory `<lock>.breaker`, who checks again
#                      before removing it (a breaker older than a minute is
#                      itself removed); then the waiters race for the lock
#                      with mkdir again. The development runner uses the same
#                      protocol, so both can share one lock directory.
#                      The parent directory must exist. Waiting gives up after
#                      three hours. Image builds are not serialised.
#
# Build state: the checkout is mounted read-only and copied (rsync, with
# modification times, without target/, .git/ and .claude/) to a fixed path
# inside the container, so the host tree is never written; the build output
# lives in a Docker volume of its own per checkout and platform (named from
# the checkout's path: cargo decides freshness by modification time, so two
# source trees must never share one), and the crates.io cache in the volume
# oxiblas-sme2-qemu-registry. Later runs are incremental. --native copies the
# tree to target/sme2-qemu/native/src and builds in target/sme2-qemu/native;
# --hardware likewise in target/sme2-qemu/hardware.
#
# Output, in target/sme2-qemu/ of this checkout (and nowhere else, apart
# from cargo's own crates.io cache in CARGO_HOME with --native and
# --hardware, and the scratch directory of --self-test):
#   positive.log, negative.log   the complete output of each pass's unit tests
#   hardware.log                 --hardware: the same for its pass
#   <pass>-public-release.log, <pass>-public-debug.log
#                                the public-path stage's two builds in that
#                                pass (positive, negative, hardware), each
#                                after the pass's header lines; while a pass
#                                runs, <pass>.log collects all its output,
#                                which is split at the step lines
#                                (`== step <name> [<token>]`) when it ends
#                                (a split that fails fails the pass and
#                                leaves the complete output in
#                                <pass>.log.stream, archived as well; in
#                                <pass>.log when it could not be renamed)
#   detect-report.log            --hardware: the detection report's run
#   image-build.log              when the image was built in this run
#   summary.txt                  OS, kernel and architecture of the host,
#                                mode, image tag and ID, QEMU, qemu-user
#                                package and rustc versions, CPU strings,
#                                cargo command, the names and values of the
#                                forwarded variables, the gate variables,
#                                test and gate-marker counts and verdict of
#                                each pass, the stage's commands and, per
#                                pass and build, its tests passed and failed
#                                per test binary, its sme2-gate lines, cargo's
#                                status and time (a pass's own `time`, like
#                                the seconds of its console line, covers its
#                                unit tests and the stage; each build's time
#                                is on its `public ...` line), git commit and
#                                whether the tree
#                                is dirty; with --hardware the CPU brand, the
#                                detection report and the values read (see
#                                Hardware) instead of image, QEMU and CPU
#                                strings
#   sme2-qemu-results.tar.gz     summary.txt and the logs, the file to send
#                                back; its tar headers carry user and group 0
#                                instead of your names (when tar takes the
#                                options; otherwise the summary says so)
#   .run.lock/                   the lock of the run (its process id and start
#                                time): one run per checkout in any mode, a
#                                second one stops with exit 2; the lock of a
#                                run that is gone is taken over
# The results of an earlier run are deleted at the start (after the lock), so
# after any exit the archive (if present) belongs to this run.
# The archive is meant to be attached to public issues. It holds the
# complete logs (compiler and test output, tool versions) and the values of
# the forwarded variables (OXIBLAS_*, RUSTFLAGS, ...): leave out of those
# variables whatever you would not publish. The host name is not recorded,
# paths in the summary are relative to the checkout (except, with
# --hardware, the cargo config files read, which are absolute), and in the
# archived copies the checkout's absolute path is replaced by `<checkout>`
# and then the home directory's (HOME) by `<home>`.
# Exit status: 0 when every requested pass succeeded, 1 when one failed
# (with --hardware also when SME2 was not detected or the detection report
# did not run), 2 for a usage error, a missing prerequisite (also a checkout
# path that a Docker bind mount cannot take, e.g. one containing `:`; with
# --hardware a host that is not macOS or Linux on aarch64, a toolchain older
# than the workspace's rust-version or a rust-version it cannot read, and a
# refused cargo runner; with --self-test its file sme2_qemu_selftest.sh; the
# file sme2_qemu_verdicts.sh beside this script, which every run sources,
# missing or not the version this script was written for, or without a
# function it calls: checked before anything is written) or
# another run in this checkout (.run.lock), 3 when
# the lock OXIBLAS_QEMU_LOCK_DIR timed out, 128 + the signal's number after
# INT, TERM or HUP.
# A pass succeeds when cargo exits 0, at least one test ran, none failed, the
# log has no line `uncaught target signal` (QEMU's report of a signal in the
# emulated code, such as SIGILL or an abort: it fails the pass whatever
# cargo's exit status is), and
# the gate markers prove that SME2 code was reached: in the positive pass at
# least one `sme2-gate: ran` line and no `sme2-gate: skipped` line, in the
# negative pass at least one `sme2-gate: skipped` line and no `sme2-gate: ran`
# line. A filter that selects only tests which never call the gate (for
# example test_sme2_contract_reference_equals_neon) therefore fails both
# passes: it would prove nothing about SME2. The `sme2-gate: declined` lines
# prove that the backend itself declined, not only the gate: the negative
# pass additionally needs at least one of them, and the positive pass fails
# on any (there the backend is on, so a decline test must not decline). The
# default filter `sme2` selects those tests.
# In the positive pass a test that finds no SME2 panics (the variable
# OXIBLAS_TEST_REQUIRE_SME2), so a skip shows up as a failure; with output
# not captured (`-- -- --nocapture`) a `skip:` line also fails the pass.
# QEMU's "uncaught target signal 4" in a log is SIGILL in the emulated code.
# The verdict's reason also names the signal of a test binary that cargo
# reports ended by one (`(signal: 4, SIGILL: ...)`).
# The public-path stage is judged per build by the same rules as far as
# they carry over; every rule that fails adds its reason, "public path,
# <release|debug> build: ...", which names the test binary, and makes the
# pass FAIL. The lines after cargo's `Running tests/<name>.rs` line belong
# to the binary <name>. The rules: cargo exits 0 (without its end line:
# "cargo did not end", or "the stage did not run" when the pass's job ended
# before it); sme2_public_path and gemm_rayon_pool each have a `test result:`
# line, none failed and at least one passed; no `uncaught target signal` and
# no binary ended by a signal (named); in the positive pass and --hardware
# at least one `sme2-gate: ran` line from sme2_public_path ("nothing ran on
# SME2" otherwise: a pass in which nothing ran on SME2 cannot pass) and no
# `sme2-gate: skipped`, `sme2-gate: declined` or `skip:` line; in the
# negative pass no `sme2-gate: ran` line and at least one `sme2-gate:
# skipped` line from sme2_public_path (its gate was reached). The `declined`
# rule does not carry over: the integration tests have no test that sees
# the backend decline. With --hardware a SIGILL in the stage gets the
# hint about SME instructions in user space (see Hardware), and the
# verdict of the detection report is unchanged.
# In Docker mode the container's standard error is joined to its standard
# output inside the container (Docker passes the two on as separate
# streams, in no fixed order between them), so the logs keep the order in
# which cargo and the test binaries wrote: the stage's lines are assigned
# to a test binary by that order. --native and --hardware write both to
# one descriptor anyway.
# On SIGINT, SIGTERM or SIGHUP (a closed terminal or SSH session) the running
# pass is stopped: in Docker mode its container is stopped and removed; with
# --native and --hardware its job has a process group of its own, which is
# stopped as a whole (cargo, the compiler, the test binary, with --native
# QEMU). Then tee ends, the FIFO is removed, the pass's output is split into
# its logs and each gets a last line `== interrupted by SIG...` (a split
# that fails there leaves these logs empty or incomplete and the complete
# output in <pass>.log.stream, which gets that line too; the line and the
# console note then name that file), the locks are released, and no summary
# and no archive are written (the earlier ones were removed at the start).
# Once the passes are over, signals are ignored until summary.txt and the
# archive are complete (both written under temporary names and renamed).
#---
set -euo pipefail
# Bytes, not characters: no byte of a log or a cargo config file can stop
# awk, sed or grep (macOS awk stops at an invalid UTF-8 sequence in a UTF-8
# locale). cargo and the tests print the same in any locale.
LC_ALL=C
export LC_ALL

readonly TARGET_TRIPLE="aarch64-unknown-linux-gnu"
readonly DEFAULT_TOOLCHAIN="1.95.0"
readonly DEFAULT_FILTER="sme2"
readonly CPU_POSITIVE="max,sme=on,sme512=on,sme_fa64=off,sme-default-vector-length=64"
readonly CPU_NEGATIVE="max,sme=on,sme256=on,sme512=off,sme_fa64=off,sme-default-vector-length=32"
readonly QEMU_MIN_MAJOR=10
readonly QEMU_MIN_MINOR=1
readonly IMAGE_REPO="oxiblas-sme2-qemu-sh"
readonly REGISTRY_VOLUME="oxiblas-sme2-qemu-registry"
readonly LOCK_WAIT_LIMIT=10800
readonly LOCK_STALE_AGE=60
readonly MARKERS_VAR="OXIBLAS_TEST_SME2_MARKERS"
# The version of sme2_qemu_verdicts.sh this script is written for, and the
# functions of that file it calls.
readonly VERDICTS_VERSION=2
readonly VERDICTS_FUNCTIONS="judge_pass judge_hardware judge_stage split_pass_log step_exit step_begun hw_raw_values hw_report_lines"
# What a SIGILL of a test binary means on SME2 hardware (added to a verdict's
# reason that names SIGILL; both hints are read by sme2_qemu_verdicts.sh).
# shellcheck disable=SC2034
readonly SIGILL_HINT=" (on SME2 hardware: an SME instruction such as mrs tpidr2_el0 or rdsvl is not usable from user space)"
# The hint for a SIGILL after the report printed a present candidate: the
# probe ran, so an instruction of the kernels that the first-use check runs
# is the likely one.
# shellcheck disable=SC2034
readonly SIGILL_CHECK_HINT=" (the register probe had passed: an instruction of an SME2 kernel that the first-use self-check runs is not implemented by this CPU; please report the log)"
TAB="$(printf '\t')"
# Seconds between two looks at a held lock (a plain variable, so that a test
# harness of the lock functions can shorten it).
lock_poll=5

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

die() {
  local code="$1"
  shift
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit "$code"
}

note() {
  printf '== %s\n' "$*" >&2
}

print_help() {
  # The header comment, from line 2 to the `#---` marker, without the `# `.
  awk 'NR == 1 { next } /^#---/ { exit } { sub(/^# ?/, ""); print }' "$SCRIPT_DIR/$SCRIPT_NAME"
}

# ---------------------------------------------- toolchain (--hardware)
# Whether version $1 is older than version $2 (MAJOR.MINOR[.PATCH], numbers
# compared as numbers; a missing part counts as 0).
version_older() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, ".")
    for (i = 1; i <= 3; i++) {
      p = (i <= na) ? x[i] + 0 : 0; q = (i <= nb) ? y[i] + 0 : 0
      if (p < q) exit 0
      if (p > q) exit 1
    }
    exit 1 }'
}

# The rust-version of the manifest $1 (the workspace's Cargo.toml): the key
# rust-version of its [workspace.package] table (or of [package]); empty
# when there is none or it is not MAJOR.MINOR[.PATCH].
workspace_rust_version() {
  awk '
    /^[ \t]*\[/ { t = $0; gsub(/[ \t]/, "", t); sub(/#.*/, "", t); next }
    (t == "[workspace.package]" || t == "[package]") && /^[ \t]*rust-version[ \t]*=/ {
      v = $0; sub(/^[^=]*=[ \t]*"/, "", v); sub(/".*/, "", v)
      if (v ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/) print v
      exit
    }' "$1" 2>/dev/null || true
}

# ------------------------------------------- cargo runners (--hardware)
# The parts of a cargo config file (TOML) that --hardware refuses, one per
# line: "runner<TAB><target>" (a key target.<target>.runner in any TOML
# form: tables, dotted or quoted keys, inline tables), "build.target",
# "include" (cargo would read other files), and "unreadable<TAB><what>"
# where the scan cannot be sure (a statement still open at the end or not of
# the form key = value, a key it cannot read or with a backslash escape, a
# one-line string across a line break, a failed scan): refused rather than
# guessed. It reads bytes (LC_ALL=C), skips strings (a multi-line one ends at
# the last 3 of 3 to 5 quotes), arrays and comments, also across lines (the
# parser of x86_gemm_runbook.sh's config check, the parts of a key kept apart).
cargo_config_scan() {
  if [ ! -r "$1" ]; then printf 'unreadable\tthe file cannot be read\n'; return 0; fi
  if ! cargo_config_awk "$1"; then printf 'unreadable\tthe scan of the file failed\n'; fi
  return 0
}

# The awk program of cargo_config_scan, on the file $1.
cargo_config_awk() {
  LC_ALL=C awk '
    function quote_at(s, i) {
      if (substr(s, i, 3) == "\"\"\"" || substr(s, i, 3) == "\047\047\047") return substr(s, i, 3)
      return substr(s, i, 1)
    }
    # Whether a statement (the lines so far) is complete: brackets balanced
    # and no multi-line string open, outside strings and comments.
    function complete(s,   i, n, c, d, q, e) {
      n = length(s); d = 0; q = ""
      for (i = 1; i <= n; i++) {
        c = substr(s, i, 1)
        if (q != "") {
          if (substr(s, i, length(q)) == q) { e = 0; if (length(q) == 3) while (e < 2 && substr(s, i + 3 + e, 1) == c) e++; i += length(q) - 1 + e; q = "" }
          else if (c == "\\" && substr(q, 1, 1) == "\"") i++; else if (c == "\n" && length(q) == 1) out("unreadable\ta one-line string across a line break")
        } else if (c == "\"" || c == "\047") { q = quote_at(s, i); i += length(q) - 1 }
        else if (c == "#") { while (i <= n && substr(s, i, 1) != "\n") i++ }
        else if (c == "[" || c == "{") d++
        else if (c == "]" || c == "}") d--
      }
      return d <= 0 && q == ""
    }
    function ws() {
      while (P <= L && (index(" \t\r\n", substr(S, P, 1)) || substr(S, P, 1) == "#"))
        if (substr(S, P, 1) == "#") { while (P <= L && substr(S, P, 1) != "\n") P++ } else P++
    }
    # Skips the string at P (any of the four TOML kinds); returns its text
    # and sets ESC when a basic string holds a backslash escape.
    function skip_string(   q, k, e) {
      q = quote_at(S, P); P += length(q); k = ""
      while (P <= L && substr(S, P, length(q)) != q) {
        if (substr(q, 1, 1) == "\"" && substr(S, P, 1) == "\\") { P++; ESC = 1 }
        k = k substr(S, P, 1); P++
      }
      P += length(q); e = 0; while (length(q) == 3 && e < 2 && substr(S, P, 1) == substr(q, 1, 1)) { e++; P++ }; return k
    }
    function out(line) { if (!(line in done)) { done[line] = 1; print line } }
    function key_part(   start, k) {
      ws()
      if (substr(S, P, 1) == "\"" || substr(S, P, 1) == "\047") {
        ESC = 0; k = skip_string()
        if (ESC) out("unreadable\ta quoted key with a backslash escape")
        return k
      }
      start = P; while (P <= L && substr(S, P, 1) ~ /[A-Za-z0-9_-]/) P++
      if (P == start) out("unreadable\ta key the scan cannot read")
      return substr(S, start, P - start)
    }
    function key_path(   k) {
      k = key_part(); ws()
      while (substr(S, P, 1) == ".") { P++; k = k SUBSEP key_part(); ws() }
      return k
    }
    function emit(k,   n, p) {
      n = split(k, p, SUBSEP)
      if (p[1] == "target" && n >= 3 && p[3] == "runner") out("runner\t" p[2])
      if (p[1] == "build" && n >= 2 && p[2] == "target") out("build.target")
      if (p[1] == "include") out("include")
    }
    # A value at P: an inline table is read key by key; anything else is
    # skipped up to the next comma or closing bracket of its own level.
    function value(prefix,   c, d, k, before) {
      ws()
      if (substr(S, P, 1) == "{") {
        for (P++; P <= L; ) {
          before = P; ws(); c = substr(S, P, 1)
          if (c == "}") { P++; return }
          if (c == ",") { P++; continue }
          k = key_path(); ws()
          if (substr(S, P, 1) == "=") { P++; emit(prefix SUBSEP k); value(prefix SUBSEP k) }
          else out("unreadable\ta statement that is neither a table header nor key = value")
          if (P == before) P++
        }
        return
      }
      for (d = 0; P <= L; ) {
        c = substr(S, P, 1)
        if (c == "\"" || c == "\047") { skip_string(); continue }
        if (c == "#") { while (P <= L && substr(S, P, 1) != "\n") P++; continue }
        if (c == "[" || c == "{") d++
        else if (c == "]" || c == "}") { if (d == 0) return; d-- }
        else if (c == "," && d == 0) return
        P++
      }
    }
    function statement(   k) {
      S = stmt; L = length(S); P = 1; ws()
      if (P > L) return
      if (substr(S, P, 1) == "[") { P++; if (substr(S, P, 1) == "[") P++; table = key_path(); emit(table); return }
      k = key_path(); ws()
      if (substr(S, P, 1) != "=") { out("unreadable\ta statement that is neither a table header nor key = value"); return }
      P++; if (table != "") k = table SUBSEP k
      emit(k); value(k)
    }
    { stmt = (stmt == "" ? $0 : stmt "\n" $0); if (complete(stmt)) { statement(); stmt = "" } }
    END { if (stmt != "") { out("unreadable\ta statement still open at the end of the file (a bracket or a string)"); statement() } }' "$1"
}

# The cargo config files cargo reads for a command run in directory $1, one
# path per line, nearest first, each once: .cargo/config.toml and
# .cargo/config of $1 and every parent up to /, then CARGO_HOME's (default
# ~/.cargo); parents of the real directory, as cargo's cwd (a linked target/:
# the deepest existing part of $1 resolved by pwd -P). With $2, the .cargo/
# of $1 is taken from $2 (the copy --hardware runs in is made later, from $2).
#   cargo_config_files <absolute directory> [<directory whose .cargo/ stands in>]
cargo_config_files() {
  local dir="$1" rest="" start cdir f real s known list=() seen=() home="${CARGO_HOME:-${HOME:-}/.cargo}"
  while [ ! -d "$dir" ] && [ "${dir%/*}" != "$dir" ]; do rest="/${dir##*/}$rest"; dir="${dir%/*}"; done
  start="$(cd "${dir:-/}" && pwd -P)"; start="${start%/}$rest"; dir="${start:-/}"; start="$dir"
  while :; do
    cdir="$dir/.cargo"
    if [ "$dir" = "$start" ] && [ -n "${2:-}" ]; then cdir="$2/.cargo"; fi
    list+=("$cdir/config.toml" "$cdir/config")
    if [ "$dir" = / ]; then break; fi
    dir="$(dirname "$dir")"
  done
  list+=("$home/config.toml" "$home/config")
  for f in "${list[@]}"; do
    [ -f "$f" ] || continue
    real="$(cd "$(dirname "$f")" 2>/dev/null && pwd -P)/$(basename "$f")"
    known=0
    for s in ${seen[@]+"${seen[@]}"}; do [ "$s" != "$real" ] || known=1; done
    if [ "$known" = 0 ]; then seen+=("$real"); printf '%s\n' "$f"; fi
  done
  return 0
}

# The runner variable of target $1: CARGO_TARGET_<upper case, - . as _>_RUNNER.
runner_env_name() {
  printf 'CARGO_TARGET_%s_RUNNER' "$(printf '%s' "$1" | tr 'abcdefghijklmnopqrstuvwxyz.-' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ__')"
}

# Why --hardware may not run, one reason per line (none: it may): a cargo
# runner or another target would put a program between cargo and the test
# binaries. Refused: CARGO_TARGET_<host>_RUNNER or CARGO_BUILD_TARGET set at
# all (also empty); --config or --target among the cargo arguments before a
# second `--`; in a config file of the list $2 a runner for the host triple
# $1 or for a cfg(...) expression (not evaluated here), build.target,
# include, or what the scan cannot read.
#   runner_refusals <host triple> <config files, one per line> [<cargo arguments>...]
runner_refusals() {
  local host="$1" files="$2" var f kind what arg
  shift 2
  var="$(runner_env_name "$host")"
  if [ -n "${!var+set}" ]; then printf '%s is set in the environment\n' "$var"; fi
  if [ -n "${CARGO_BUILD_TARGET+set}" ]; then printf 'CARGO_BUILD_TARGET is set in the environment (the test binaries would not be built for this host)\n'; fi
  for arg in "$@"; do
    case "$arg" in
      --) break ;;
      --config|--config=*) printf 'the cargo argument %s (a configuration of its own)\n' "$arg" ;;
      --target|--target=*) printf 'the cargo argument %s (the test binaries would not be built for this host)\n' "$arg" ;;
    esac
  done
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    while IFS="$TAB" read -r kind what; do
      case "$kind" in
        runner)
          case "$what" in
            "$host") printf '%s sets target.%s.runner\n' "$f" "$what" ;;
            cfg\(*) printf '%s sets target.%s.runner (a cfg expression, which this script does not evaluate)\n' "$f" "$what" ;;
          esac ;;
        build.target) printf '%s sets build.target (the test binaries would not be built for this host)\n' "$f" ;;
        include) printf '%s has an include key (cargo would read further files)\n' "$f" ;;
        unreadable) printf '%s cannot be read reliably here: %s\n' "$f" "$what" ;;
      esac
    done <<EOF
$(cargo_config_scan "$f")
EOF
  done <<EOF
$files
EOF
  return 0
}

# ------------------------------------------------------ processes and locks
# Modification time of a path in seconds (GNU stat, then BSD stat); the
# current time when the path is gone.
file_mtime() {
  local t
  t="$(stat -c %Y "$1" 2>/dev/null || true)"
  case "$t" in ''|*[!0-9]*) t="$(stat -f %m "$1" 2>/dev/null || true)" ;; esac
  case "$t" in ''|*[!0-9]*) date +%s ;; *) printf '%s\n' "$t" ;; esac
}

# Whether process $1 exists (`ps` also sees an owner of another user, for
# whom kill -0 fails with EPERM).
process_alive() {
  kill -0 "$1" 2>/dev/null || ps -p "$1" >/dev/null 2>&1
}

# Start time of process $1 as ps prints it, in UTC (empty when ps cannot say).
process_start() {
  TZ=UTC0 ps -p "$1" -o lstart= 2>/dev/null | awk 'NF { $1 = $1; print; exit }' || true
}

LOCK_DIR="${OXIBLAS_QEMU_LOCK_DIR:-}"
have_lock=0
current_container=""
child_pid=""
RUN_LOCK=""; child_group=0; tee_pid=""; current_fifo=""; current_log=""
# The pass whose output is being collected in its log (split by on_signal
# when a signal stops it), and the token of the step lines of this run.
split_pass=""
run_token="sq$$-$RANDOM"
public_path=1

release_lock() {
  if [ "$have_lock" = 1 ] && [ "$(cat "$LOCK_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
  have_lock=0
}

# Whether the lock directory $1 is stale: its owner is gone or is not the
# process whose start time the pid file's second line records (only the run
# lock has one: a reused id), or it has had no `pid` file for more than
# LOCK_STALE_AGE seconds. A missing lock is not stale.
lock_is_stale() {
  local dir="$1" owner start now age
  [ -d "$dir" ] || return 1
  owner="$(sed -n 1p "$dir/pid" 2>/dev/null || true)"
  start="$(sed -n 2p "$dir/pid" 2>/dev/null || true)"
  if [ -n "$owner" ]; then
    process_alive "$owner" || return 0
    if [ -n "$start" ]; then
      now="$(process_start "$owner")"
      if [ -n "$now" ] && [ "$now" != "$start" ]; then return 0; fi
    fi
    return 1
  fi
  age=$(( $(date +%s) - $(file_mtime "$dir") ))
  [ "$age" -gt "$LOCK_STALE_AGE" ]
}

# Removes the stale lock $1 (of $2, named in the message). Only the holder
# of the breaker directory <lock>.breaker may remove it, and it looks again
# before it does: without the breaker two waiters that both saw the dead
# owner could each remove the lock the other one has just taken. A breaker
# older than LOCK_STALE_AGE (its holder died) is removed; a waiter that
# finds the breaker held waits briefly and lets the caller retry.
break_stale_lock() {
  local dir="$1" breaker="$1.breaker" breaker_age
  if mkdir "$breaker" 2>/dev/null; then
    if lock_is_stale "$dir"; then
      note "lock: removing the stale lock of ${2:-nobody}"
      rm -rf "$dir"
    fi
    rmdir "$breaker" 2>/dev/null || true
  else
    breaker_age=$(( $(date +%s) - $(file_mtime "$breaker") ))
    if [ "$breaker_age" -gt "$LOCK_STALE_AGE" ]; then
      rmdir "$breaker" 2>/dev/null || true
    fi
    sleep 0.2
  fi
}

# The machine-wide lock OXIBLAS_QEMU_LOCK_DIR (see Environment), taken for
# each container run: waits while another process holds it.
acquire_lock() {
  [ -n "$LOCK_DIR" ] || return 0
  local wait_start waited=0 noted=-1 owner
  wait_start="$(date +%s)"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if lock_is_stale "$LOCK_DIR"; then
      break_stale_lock "$LOCK_DIR" "$owner"
      continue
    fi
    waited=$(( $(date +%s) - wait_start ))
    if [ "$waited" -ge "$LOCK_WAIT_LIMIT" ]; then
      printf '%s: timed out after %ss waiting for the lock %s (held by %s)\n' \
        "$SCRIPT_NAME" "$waited" "$LOCK_DIR" "${owner:-nobody}" >&2
      exit 3
    fi
    if [ $((waited / 60)) != "$noted" ]; then
      noted=$((waited / 60))
      note "lock: waiting for process ${owner:-?} (${waited}s)"
    fi
    sleep "$lock_poll"
  done
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  have_lock=1
}

# Takes the run lock $1 (target/sme2-qemu/.run.lock): one run of this script
# per checkout, whatever the mode (all write in target/sme2-qemu/). A live
# run's lock means exit 2; a stale one is taken over through the breaker.
# The pid file holds this process's id and, when ps tells, its start time.
take_run_lock() {
  local dir="$1" tries=0 owner
  while ! mkdir "$dir" 2>/dev/null; do
    [ -d "$dir" ] || die 2 "cannot create the lock $dir (is target/ writable?)"
    tries=$((tries + 1))
    [ "$tries" -le 50 ] || die 2 "cannot take the lock $dir after $tries attempts"
    owner="$(sed -n 1p "$dir/pid" 2>/dev/null || true)"
    if lock_is_stale "$dir"; then
      break_stale_lock "$dir" "process ${owner:-?}"
      continue
    fi
    if [ -n "$owner" ]; then owner="process $owner (started $(sed -n 2p "$dir/pid" 2>/dev/null || true) UTC)"; else owner="a run that is just starting"; fi
    die 2 "another run of this script uses this checkout: $owner holds the lock ${dir#"$ROOT"/}; wait until it ends or stop it (kill <its process id>), then run again; nothing was changed"
  done
  if ! { printf '%s\n%s\n' "$$" "$(process_start "$$")" > "$dir/pid.tmp" && mv -f "$dir/pid.tmp" "$dir/pid"; }; then
    rm -rf "$dir"; die 2 "cannot write the lock $dir/pid"
  fi
  RUN_LOCK="$dir"
}

# Removes the run lock if this process holds it.
# shellcheck disable=SC2329 # called from the traps (on_exit, on_signal)
release_run_lock() {
  if [ -n "$RUN_LOCK" ] && [ "$(sed -n 1p "$RUN_LOCK/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$RUN_LOCK"
  fi
  RUN_LOCK=""
}

# Whether process $1 is still running (a zombie that is not yet reaped is not).
# shellcheck disable=SC2329 # called from stop_pass (a trap)
child_running() {
  local state
  state="$(ps -o stat= -p "$1" 2>/dev/null || true)"
  case "$state" in
    ''|Z*) return 1 ;;
    *) return 0 ;;
  esac
}

# Whether a process of the process group $1 is still there.
# shellcheck disable=SC2329 # called from stop_pass (a trap)
group_alive() {
  kill -0 -- "-$1" 2>/dev/null
}

# Starts the command (the arguments after $1) in the background, output to
# $1 (a FIFO or a log), input from /dev/null, in a process group of its own
# (job control on for this start only), so that stop_pass can stop it with
# all it starts (cargo, compiler, test binary, QEMU). Sets child_pid.
start_group_job() {
  local out="$1"
  shift
  set -m
  "$@" > "$out" 2>&1 < /dev/null &
  child_pid=$!
  set +m
  child_group=1
}

# Stops the running pass: with --native and --hardware the job's whole
# process group (TERM; KILL for what is left after 10 s); in Docker mode the
# background `docker start --attach` client and the container (created in
# the foreground, see run_pass, so a container named in current_container
# exists or never will; removed once more after the client is gone, for
# good measure); then tee (stopped if still there after 2 s) and the FIFO.
# shellcheck disable=SC2329 # called from the traps
stop_pass() {
  local tries=0
  if [ -n "$child_pid" ]; then
    if [ "$child_group" = 1 ]; then
      kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    else
      kill -TERM "$child_pid" 2>/dev/null || true
    fi
  fi
  if [ -n "$current_container" ]; then
    docker rm -f "$current_container" >/dev/null 2>&1 || true
  fi
  if [ -n "$child_pid" ]; then
    while { child_running "$child_pid" || { [ "$child_group" = 1 ] && group_alive "$child_pid"; }; } && [ "$tries" -lt 50 ]; do
      sleep 0.2
      tries=$((tries + 1))
    done
    if [ "$child_group" = 1 ] && group_alive "$child_pid"; then kill -KILL -- "-$child_pid" 2>/dev/null || true; fi
    if child_running "$child_pid"; then kill -KILL "$child_pid" 2>/dev/null || true; fi
    wait "$child_pid" 2>/dev/null || true
    child_pid=""
    child_group=0
  fi
  if [ -n "$current_container" ]; then
    docker rm -f "$current_container" >/dev/null 2>&1 || true
    current_container=""
  fi
  if [ -n "$tee_pid" ]; then
    tries=0
    while child_running "$tee_pid" && [ "$tries" -lt 10 ]; do sleep 0.2; tries=$((tries + 1)); done
    kill -TERM "$tee_pid" 2>/dev/null || true
    wait "$tee_pid" 2>/dev/null || true
    tee_pid=""
  fi
  if [ -n "$current_fifo" ]; then rm -f "$current_fifo"; current_fifo=""; fi
}

# shellcheck disable=SC2329 # invoked by the EXIT trap
on_exit() {
  stop_pass
  release_lock
  release_run_lock
}

# The logs of pass $1, one per line: <pass>.log and, with the public-path
# stage, <pass>-public-release.log and <pass>-public-debug.log.
# shellcheck disable=SC2329 # called from on_signal (a trap)
pass_logs() {
  printf '%s\n' "$OUT/$1.log"
  if [ "$public_path" = 1 ]; then printf '%s\n%s\n' "$OUT/$1-public-release.log" "$OUT/$1-public-debug.log"; fi
}

# Splits the output of pass $1, collected in its log, into its logs.
split_logs() {
  split_pass_log "$OUT/$1.log" "$run_token" "$public_path" "$OUT/$1-public-release.log" "$OUT/$1-public-debug.log"
}

# The INT, TERM and HUP traps ($1 the signal's name, $2 the exit status);
# further signals are ignored meanwhile, so that the clean-up cannot be cut
# short. The output of the stopped pass is split into its logs, and each of
# them gets a last line saying so; no summary and no archive are written
# (the earlier ones were removed at the start).
# shellcheck disable=SC2329 # invoked by the INT, TERM and HUP traps
on_signal() {
  local last logs="$current_log" f whole=""
  trap '' INT TERM HUP
  stop_pass
  if [ -n "$split_pass" ]; then
    # A split that fails here (its awk) must not end the handler: a failed
    # command in a trap ends a `set -e` script with that command's status.
    # The pass's logs are then empty or incomplete and the complete output
    # is still in <pass>.log.stream: it gets the last line as well, and the
    # line and the note say where the output is.
    split_logs "$split_pass" || true
    logs="$(pass_logs "$split_pass")"
    if [ -f "$OUT/$split_pass.log.stream" ]; then
      logs="$logs
$OUT/$split_pass.log.stream"
      whole="; the split of the pass's output into its logs failed, the complete output is in $split_pass.log.stream"
    fi
  fi
  while IFS= read -r f; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    last="$(tail -c 1 "$f" 2>/dev/null || true)"
    { [ -z "$last" ] || printf '\n'
      printf '== interrupted by SIG%s: the pass was stopped; no summary and no archive were written%s\n' "$1" "$whole"; } >> "$f" 2>/dev/null || true
  done <<EOF
$logs
EOF
  printf '\n' >&2
  note "interrupted by SIG$1: the running pass was stopped; no summary and no archive were written$whole"
  release_lock
  release_run_lock
  exit "$2"
}

# ----------------------------------------------------------- inner runner
# The step runner, at the end of INNER (in the container, or on the host with
# --native) and run by bash in the copy with --hardware:
#   bash -c "$STEP_LOOP" _ <name> <count> <count cargo test arguments> [<name> <count> ...]
# For each step in turn, whatever the status of the one before, it runs
# `cargo $SQ_PLUS test <arguments>` between the step lines that
# split_pass_log reads (with the token SQ_TOKEN), and it exits with the first
# status that is not 0 (else 0).
# shellcheck disable=SC2016 # expanded by the inner bash, not here
STEP_LOOP='set -u
first_rc=0
while [ "$#" -ge 2 ]; do
  step="$1"; n="$2"; shift 2
  args=()
  while [ "$n" -gt 0 ] && [ "$#" -gt 0 ]; do args+=("$1"); shift; n=$((n - 1)); done
  echo "== step $step [$SQ_TOKEN]"
  echo "== cargo ${SQ_PLUS:+$SQ_PLUS }test ${args[*]+${args[*]}}"
  t0=$SECONDS; rc=0
  cargo ${SQ_PLUS:+"$SQ_PLUS"} test ${args[@]+"${args[@]}"} || rc=$?
  printf "\n== step %s [%s]: cargo exited with %s after %ss\n" "$step" "$SQ_TOKEN" "$rc" "$((SECONDS - t0))"
  [ "$first_rc" != 0 ] || first_rc=$rc
done
exit "$first_rc"'

# Runs inside the container (or on the host with --native) as
#   bash -c "$INNER" _ <steps, as for STEP_LOOP>
# with SQ_SRC (the checkout), SQ_WORK (where to copy it), SQ_CPU (the QEMU CPU
# string), SQ_TOOLCHAIN and SQ_TOKEN set, and CARGO_TARGET_DIR pointing at
# the build directory. On a non-aarch64 machine it cross-compiles and gives
# QEMU the cross compiler's aarch64 sysroot with -L. Its standard error goes
# to its standard output (one stream: Docker would pass two on in no fixed
# order between them, and the logs are read in the order of the writes).
# (Defined before the self-test, which runs it with stand-ins.)
# shellcheck disable=SC2016 # expanded by the inner bash, not here
INNER='set -euo pipefail
exec 2>&1
mkdir -p "$SQ_WORK"
rsync -a --delete --exclude=/target --exclude=/.git --exclude=/.claude "$SQ_SRC"/ "$SQ_WORK"/
cd "$SQ_WORK"
runner="qemu-aarch64"
machine="$(uname -m)"
if [ "$machine" != aarch64 ] && [ "$machine" != arm64 ]; then
  linker="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-aarch64-linux-gnu-gcc}"
  command -v "$linker" >/dev/null 2>&1 || { echo "error: the aarch64 cross linker $linker is missing" >&2; exit 2; }
  sysroot="$("$linker" -print-sysroot 2>/dev/null || true)"
  if [ -z "$sysroot" ] || [ ! -e "$sysroot/lib/ld-linux-aarch64.so.1" ]; then
    loader="$("$linker" -print-file-name=ld-linux-aarch64.so.1)"
    case "$loader" in
      /*/lib/ld-linux-aarch64.so.1) sysroot="${loader%/lib/ld-linux-aarch64.so.1}" ;;
      *) echo "error: $linker does not know the aarch64 loader ld-linux-aarch64.so.1 (is the aarch64 libc for it installed?)" >&2; exit 2 ;;
    esac
  fi
  sysroot="$(cd "$sysroot" && pwd -P)"
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$linker"
  export CC_aarch64_unknown_linux_gnu="$linker"
  runner="qemu-aarch64 -L $sysroot"
  echo "== cross : $machine -> aarch64, linker $linker, sysroot $sysroot"
fi
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="$runner -cpu $SQ_CPU"
echo "== qemu  : $(qemu-aarch64 --version | head -n 1)"
echo "== rustc : $(rustc "+$SQ_TOOLCHAIN" --version)"
echo "== runner: $CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER"
echo "== expect: OXIBLAS_TEST_REQUIRE_SME2=${OXIBLAS_TEST_REQUIRE_SME2:-} OXIBLAS_TEST_EXPECT_NO_SME2=${OXIBLAS_TEST_EXPECT_NO_SME2:-} OXIBLAS_TEST_SME2_MARKERS=${OXIBLAS_TEST_SME2_MARKERS:-}"
SQ_PLUS="+$SQ_TOOLCHAIN"
'"$STEP_LOOP"

# ---------------------------------------------------------------- arguments
# --help and --self-test first, before anything is touched (an option's
# value is skipped).
self_test=0
arg_index=1
while [ "$arg_index" -le "$#" ]; do
  case "${!arg_index}" in
    --) break ;;
    -h|--help) print_help; exit 0 ;;
    --self-test) self_test=1 ;;
    --filter|--toolchain|--platform) arg_index=$((arg_index + 1)) ;;
  esac
  arg_index=$((arg_index + 1))
done

# ------------------------------------------------------------------ checkout
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
[ -f "$ROOT/Cargo.toml" ] && [ -f "$ROOT/crates/oxiblas-blas/Cargo.toml" ] \
  || die 2 "cannot find the oxiblas workspace above $SCRIPT_DIR"
# The verdict rules live beside this script and are needed by every run.
VERDICTS_FILE="$SCRIPT_DIR/sme2_qemu_verdicts.sh"
[ -f "$VERDICTS_FILE" ] || die 2 "the file ${VERDICTS_FILE#"$ROOT"/} is missing (it belongs beside this script)"
# shellcheck source=sme2_qemu_verdicts.sh disable=SC1091
. "$VERDICTS_FILE" || die 2 "the file ${VERDICTS_FILE#"$ROOT"/} cannot be read"
# A copy of another version (only one of the two files updated) would fail
# late, after every pass: it must be this script's version and define every
# function this script calls.
[ "${SQ_VERDICTS_VERSION:-none}" = "$VERDICTS_VERSION" ] \
  || die 2 "the file ${VERDICTS_FILE#"$ROOT"/} is version ${SQ_VERDICTS_VERSION:-none}, this script needs version $VERDICTS_VERSION (copy both files from one checkout)"
for fn in $VERDICTS_FUNCTIONS; do
  declare -F "$fn" >/dev/null \
    || die 2 "the file ${VERDICTS_FILE#"$ROOT"/} does not match this script: it defines no function $fn (copy both files from one checkout)"
done
if [ "$self_test" = 1 ]; then
  [ "$#" = 1 ] || die 2 "--self-test takes no other option"
  # The self-test lives beside this script and is loaded only here.
  SELFTEST_FILE="$SCRIPT_DIR/sme2_qemu_selftest.sh"
  [ -f "$SELFTEST_FILE" ] || die 2 "the self-test file ${SELFTEST_FILE#"$ROOT"/} is missing (it belongs beside this script)"
  # shellcheck source=sme2_qemu_selftest.sh disable=SC1091
  . "$SELFTEST_FILE"
  run_self_test
fi
OUT="$ROOT/target/sme2-qemu"
OUT_REL="target/sme2-qemu"
mkdir -p "$OUT"
# Every exit now stops what this run started and releases its locks; one
# run per checkout: a second one stops here (exit 2), touching nothing.
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
take_run_lock "$OUT/.run.lock"
# The results of an earlier run go now, before any check that can fail, so
# that no old archive (perhaps a PASS) is left behind by an early exit.
for f in positive negative hardware; do
  rm -f "$OUT/$f-public-release.log" "$OUT/$f-public-debug.log" "$OUT/$f.log.stream"
done
rm -f "$OUT/positive.log" "$OUT/negative.log" "$OUT/hardware.log" "$OUT/detect-report.log" \
  "$OUT/summary.txt" "$OUT/image-build.log" "$OUT/sme2-qemu-results.tar.gz" \
  "$OUT/.positive.rc" "$OUT/.negative.rc" "$OUT/.positive.fifo" "$OUT/.negative.fifo" \
  "$OUT/.hardware.fifo" "$OUT/.hardware-values" "$OUT/summary.txt.tmp" "$OUT/sme2-qemu-results.tar.gz.tmp"
rm -rf "$OUT/.archive"

want_positive=0
want_negative=0
filter="$DEFAULT_FILTER"
toolchain="$DEFAULT_TOOLCHAIN"
toolchain_given=0
native=0
hardware=0
reset=0
rebuild_image=0
image_only=0
platform=""
extra_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --positive) want_positive=1; shift ;;
    --negative) want_negative=1; shift ;;
    --both) want_positive=1; want_negative=1; shift ;;
    --filter)
      [ "$#" -ge 2 ] || die 2 "--filter needs a value"
      filter="$2"; shift 2 ;;
    --filter=*) filter="${1#--filter=}"; shift ;;
    --toolchain)
      [ "$#" -ge 2 ] || die 2 "--toolchain needs a value"
      toolchain="$2"; toolchain_given=1; shift 2 ;;
    --toolchain=*) toolchain="${1#--toolchain=}"; toolchain_given=1; shift ;;
    --native) native=1; shift ;;
    --hardware) hardware=1; shift ;;
    --reset) reset=1; shift ;;
    --no-public-path) public_path=0; shift ;;
    --rebuild-image) rebuild_image=1; shift ;;
    --image-only) image_only=1; shift ;;
    --platform)
      [ "$#" -ge 2 ] || die 2 "--platform needs a value"
      platform="$2"; shift 2 ;;
    --platform=*) platform="${1#--platform=}"; shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) die 2 "unknown option: $1 (see --help; cargo arguments go after --)" ;;
  esac
done

if [ "$hardware" = 1 ]; then
  [ "$want_negative" = 0 ] || die 2 "--hardware runs the positive pass only (no --negative, no --both): this machine's SME2 cannot be turned off for a negative pass"
  [ "$native" = 0 ] || die 2 "--hardware and --native exclude each other (--native runs QEMU without Docker)"
  [ -z "$platform" ] || die 2 "--platform applies to Docker runs only, not to --hardware"
  [ "$image_only" = 0 ] || die 2 "--image-only applies to Docker runs only, not to --hardware"
  [ "$rebuild_image" = 0 ] || die 2 "--rebuild-image applies to Docker runs only, not to --hardware"
  want_positive=1
fi
if [ "$want_positive" = 0 ] && [ "$want_negative" = 0 ]; then
  want_positive=1
  want_negative=1
fi
case "$toolchain" in
  ''|*[!A-Za-z0-9._-]*) die 2 "bad --toolchain '$toolchain' (letters, digits, . _ - only)" ;;
esac
case "$platform" in
  ''|amd64|arm64) ;;
  linux/amd64|linux/arm64) platform="${platform#linux/}" ;;
  *) die 2 "bad --platform '$platform' (amd64 or arm64)" ;;
esac
if [ "$native" = 1 ] && [ -n "$platform" ]; then
  die 2 "--platform applies to Docker runs only, not to --native"
fi
if [ "$native" = 1 ] && [ "$image_only" = 1 ]; then
  die 2 "--image-only applies to Docker runs only, not to --native"
fi

# A Docker bind mount (-v SRC:DST:ro) splits at `:`, and a newline cannot be
# passed either: such a checkout path would only show up as failed passes.
if [ "$native" = 0 ] && [ "$hardware" = 0 ]; then
  case "$ROOT" in
    *:*|*"
"*) die 2 "the checkout path contains ':' or a newline, which a Docker bind mount cannot take: $ROOT (move or link the checkout elsewhere, or use --native on Linux)" ;;
  esac
fi

# The cargo test arguments shared by both passes (the hardware pass builds
# for the host: no --target).
cargo_args=(-p oxiblas-blas --features "sme2,parallel" --lib --release)
if [ "$hardware" = 0 ]; then cargo_args+=(--target "$TARGET_TRIPLE"); fi
if [ -n "$filter" ]; then cargo_args+=("$filter"); fi
if [ "${#extra_args[@]}" -gt 0 ]; then cargo_args+=("${extra_args[@]}"); fi
# The two commands of the public-path stage (no filter, no extra arguments).
target_args=()
if [ "$hardware" = 0 ]; then target_args=(--target "$TARGET_TRIPLE"); fi
stage_tail=(--test sme2_public_path --test gemm_rayon_pool --no-fail-fast)
stage_release_args=(-p oxiblas-blas --features "sme2,parallel" --release ${target_args[@]+"${target_args[@]}"} "${stage_tail[@]}")
stage_debug_args=(-p oxiblas-blas --features "sme2,parallel" ${target_args[@]+"${target_args[@]}"} "${stage_tail[@]}")

# Environment forwarded to the tests: OXIBLAS_* (except the lock and the gate
# variables, which every pass sets itself) and a few others.
forward_names=()
for name in $(env | sed -n 's/^\(OXIBLAS_[A-Za-z0-9_]*\)=.*/\1/p' | sort -u); do
  case "$name" in
    OXIBLAS_QEMU_LOCK_DIR|OXIBLAS_TEST_REQUIRE_SME2|OXIBLAS_TEST_EXPECT_NO_SME2|"$MARKERS_VAR") continue ;;
  esac
  [ -n "${!name+set}" ] && forward_names+=("$name")
done
for name in RUSTFLAGS RUST_TEST_THREADS RUST_BACKTRACE CARGO_BUILD_JOBS; do
  [ -n "${!name+set}" ] && forward_names+=("$name")
done

# ------------------------------------------------------------------- helpers
# Prints "MAJOR MINOR" of the first `qemu-aarch64 --version` line read from stdin.
qemu_major_minor() {
  sed -n 's/.*version \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -n 1
}

# Succeeds when the version line on stdin is at least QEMU_MIN_MAJOR.QEMU_MIN_MINOR.
qemu_new_enough() {
  local mm major minor
  mm="$(qemu_major_minor)"
  [ -n "$mm" ] || return 1
  major="${mm% *}"
  minor="${mm#* }"
  [ "$major" -gt "$QEMU_MIN_MAJOR" ] && return 0
  [ "$major" -eq "$QEMU_MIN_MAJOR" ] && [ "$minor" -ge "$QEMU_MIN_MINOR" ]
}

# ------------------------------------------------------------- Dockerfile
dockerfile() {
  cat <<'DOCKERFILE'
# Image of crates/oxiblas-blas/tools/sme2_qemu.sh (amd64 and arm64).
FROM ubuntu:26.04
ARG RUST_TOOLCHAIN
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eu; \
    pkgs="qemu-user gcc libc6-dev ca-certificates curl rsync"; \
    case "$(dpkg --print-architecture)" in \
      amd64) pkgs="$pkgs gcc-aarch64-linux-gnu libc6-dev-arm64-cross" ;; \
      arm64) ;; \
      *) echo "unsupported image architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends $pkgs; \
    rm -rf /var/lib/apt/lists/*; \
    qemu-aarch64 --version | head -n 1
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal \
           --default-toolchain "$RUST_TOOLCHAIN" --target aarch64-unknown-linux-gnu \
    && rustc --version && cargo --version
WORKDIR /work
DOCKERFILE
}

# ---------------------------------------------------------- environment
host_machine="$(uname -m)"
# OS, kernel and architecture only: the host name is not recorded.
host_desc="$(uname -s) $(uname -r), $host_machine"
mode=""
build_desc=""
image=""
build_volume=""
qemu_version=""
qemu_package=""
image_id=""
rustc_version=""
image_built="no"
hw_cpu=""
HW_DIR="$OUT/hardware"
# The environment of the hardware pass's cargo commands (its steps and the
# detection report), set by run_pass from the pass's gate values.
hw_env=()

if [ "$hardware" = 1 ]; then
  # The host's own cargo and rustc (with --toolchain TC: rustup's +TC).
  hw_cargo=(cargo)
  hw_rustc=(rustc)
  if [ "$toolchain_given" = 1 ]; then hw_cargo+=("+$toolchain"); hw_rustc+=("+$toolchain"); fi
  missing=()
  case "$(uname -s)" in
    Darwin|Linux) ;;
    *) missing+=("macOS or Linux: --hardware reads the operating system's SME report there (this host is $(uname -s))") ;;
  esac
  case "$host_machine" in
    aarch64|arm64) ;;
    *) missing+=("an aarch64 host: SME2 is an Arm extension (this host is $host_machine; use the QEMU passes here)") ;;
  esac
  command -v cargo >/dev/null 2>&1 || missing+=("cargo on PATH (install Rust with rustup: https://rustup.rs)")
  command -v rsync >/dev/null 2>&1 || missing+=("rsync on PATH")
  # rustc runs only on a host that passed the checks above, in the checkout,
  # where rustup picks the toolchain as for cargo in the copy.
  if [ "${#missing[@]}" = 0 ] && ! rustc_vv="$(cd "$ROOT" && "${hw_rustc[@]}" -vV 2>/dev/null)"; then
    if [ "$toolchain_given" = 1 ]; then
      missing+=("a working ${hw_rustc[*]}: install the toolchain (rustup toolchain install $toolchain), or leave out --toolchain to use the default toolchain")
    else
      missing+=("a working rustc (install Rust with rustup: https://rustup.rs)")
    fi
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    printf '%s: --hardware cannot run on this host; missing:\n' "$SCRIPT_NAME" >&2
    for item in "${missing[@]}"; do printf '  - %s\n' "$item" >&2; done
    exit 2
  fi
  rustc_version="$(printf '%s\n' "$rustc_vv" | sed -n 1p)"
  rustc_release="$(printf '%s\n' "$rustc_vv" | sed -n 's/^release: //p')"
  host_triple="$(printf '%s\n' "$rustc_vv" | sed -n 's/^host: //p')"
  if [ "$toolchain_given" = 1 ]; then toolchain_desc="--toolchain $toolchain"
  else toolchain_desc="the default toolchain (no --toolchain)"; fi
  # A toolchain older than the workspace's rust-version cannot build it (a
  # FAIL that says nothing about SME2): refused before anything is built.
  rust_floor="$(workspace_rust_version "$ROOT/Cargo.toml")"
  [ -n "$rust_floor" ] || die 2 "cannot read the workspace's rust-version (the key rust-version of [workspace.package] in Cargo.toml), so --hardware cannot check the toolchain against it; nothing was run"
  case "${rustc_release%%-*}" in
    [0-9]*.[0-9]*) ;;
    *) die 2 "cannot read the version of ${hw_rustc[*]} (no release line in its -vV output); nothing was run" ;;
  esac
  if version_older "${rustc_release%%-*}" "$rust_floor"; then
    die 2 "the toolchain ($toolchain_desc) is rustc $rustc_release, older than the workspace's rust-version $rust_floor: it cannot build this checkout, and the verdict would say nothing about SME2. Use --toolchain $DEFAULT_TOOLCHAIN (rustup toolchain install $DEFAULT_TOOLCHAIN) or update the default toolchain (rustup update); nothing was run"
  fi
  [ -n "$host_triple" ] || die 2 "${hw_rustc[*]} -vV names no host triple; nothing was run"
  # A cargo runner or another target: refused, naming the source. cargo
  # reads its configuration in the copy it runs in (whose .cargo/ is the
  # checkout's), in every parent up to /, and in CARGO_HOME.
  hw_config_files="$(cargo_config_files "$HW_DIR/src" "$ROOT")"
  runner_problems="$(runner_refusals "$host_triple" "$hw_config_files" "${cargo_args[@]}")"
  if [ -n "$runner_problems" ]; then
    printf '%s: --hardware refuses to run: a cargo runner or another target would put a program between cargo and the test binaries, which then would not run on this CPU, and the verdict would not be about this hardware:\n' "$SCRIPT_NAME" >&2
    printf '%s\n' "$runner_problems" | sed 's/^/  - /' >&2
    printf 'Unset that variable, or remove that key or argument (or run from a checkout outside that directory), and run again; nothing was run.\n' >&2
    exit 2
  fi
  hw_runner="none (cargo runs the test binaries itself: for the host $host_triple, $(runner_env_name "$host_triple") and CARGO_BUILD_TARGET are unset, no --config or --target is among the cargo arguments, and no cargo config file sets a runner for it, a cfg(...) runner, build.target or include; files read: $(printf '%s' "$hw_config_files" | awk 'NF { printf "%s%s", s, $0; s = ", " } END { if (!NR) printf "none" }'))"
  if [ "$(uname -s)" = Darwin ]; then
    hw_cpu="$(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  else
    hw_cpu="$(awk -F': *' '/^(CPU implementer|CPU part|model name)/ && !seen[$1]++ { printf "%s%s %s", s, $1, $2; s = ", " }' /proc/cpuinfo 2>/dev/null || true)"
  fi
  mode="hardware (no Docker, no QEMU)"
  build_desc="built for the host by its own cargo (no --target)"
  build_volume="$OUT_REL/hardware (directory)"
  if [ "$reset" = 1 ]; then
    note "reset: removing $OUT_REL/hardware"
    rm -rf "$HW_DIR"
  fi
elif [ "$native" = 1 ]; then
  missing=()
  [ "$(uname -s)" = Linux ] || missing+=("Linux: --native runs only on Linux (this host is $(uname -s)); use Docker mode here")
  case "$host_machine" in
    x86_64|amd64|aarch64|arm64) ;;
    *) missing+=("a supported host architecture: x86_64 or aarch64 (this host is $host_machine)") ;;
  esac
  if command -v qemu-aarch64 >/dev/null 2>&1; then
    qemu_version="$(qemu-aarch64 --version 2>/dev/null | head -n 1 || true)"
    if ! printf '%s\n' "$qemu_version" | qemu_new_enough; then
      missing+=("qemu-aarch64 >= ${QEMU_MIN_MAJOR}.${QEMU_MIN_MINOR} (found: ${qemu_version:-unknown version})")
    fi
  else
    missing+=("qemu-aarch64 >= ${QEMU_MIN_MAJOR}.${QEMU_MIN_MINOR} on PATH (not found)")
  fi
  if command -v rustup >/dev/null 2>&1; then
    if rustc_version="$(rustup run "$toolchain" rustc --version 2>/dev/null)"; then
      if ! rustup target list --toolchain "$toolchain" --installed 2>/dev/null | grep -qx "$TARGET_TRIPLE"; then
        missing+=("the Rust target $TARGET_TRIPLE for $toolchain (rustup target add --toolchain $toolchain $TARGET_TRIPLE)")
      fi
    else
      rustc_version=""
      missing+=("the Rust toolchain $toolchain (rustup toolchain install $toolchain)")
    fi
  else
    missing+=("rustup on PATH (with the toolchain $toolchain and the target $TARGET_TRIPLE)")
  fi
  command -v rsync >/dev/null 2>&1 || missing+=("rsync on PATH")
  case "$host_machine" in
    x86_64|amd64)
      linker="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-aarch64-linux-gnu-gcc}"
      command -v "$linker" >/dev/null 2>&1 \
        || missing+=("the aarch64 cross linker $linker (e.g. package gcc-aarch64-linux-gnu)")
      ;;
  esac
  if [ "${#missing[@]}" -gt 0 ]; then
    printf '%s: --native cannot run on this host; missing:\n' "$SCRIPT_NAME" >&2
    for item in "${missing[@]}"; do printf '  - %s\n' "$item" >&2; done
    exit 2
  fi
  qemu_package="$(dpkg-query -W -f='${Version}' qemu-user 2>/dev/null || true)"
  [ -n "$qemu_package" ] || qemu_package="unknown (not a dpkg package or not installed as qemu-user)"
  mode="native (no Docker)"
  case "$host_machine" in
    x86_64|amd64) build_desc="cross-compiled on $host_machine for $TARGET_TRIPLE" ;;
    *) build_desc="built natively for $TARGET_TRIPLE" ;;
  esac
  build_volume="$OUT_REL/native (directory)"
  if [ "$reset" = 1 ]; then
    note "reset: removing $OUT_REL/native"
    rm -rf "$OUT/native"
  fi
else
  command -v docker >/dev/null 2>&1 || die 2 "docker is not installed (or use --native on Linux)"
  server_arch="$(docker info --format '{{.Architecture}}' 2>/dev/null || true)"
  [ -n "$server_arch" ] || die 2 "the Docker daemon is not reachable (start Docker; on Linux your user may need to be in the docker group)"
  if [ -z "$platform" ]; then
    case "$server_arch" in
      x86_64|amd64) platform=amd64 ;;
      aarch64|arm64) platform=arm64 ;;
      *) die 2 "unsupported Docker server architecture '$server_arch' (x86_64 or aarch64)" ;;
    esac
  fi
  if [ -n "$LOCK_DIR" ]; then
    lock_parent="$(dirname "$LOCK_DIR")"
    [ -d "$lock_parent" ] || die 2 "the parent of OXIBLAS_QEMU_LOCK_DIR does not exist: $lock_parent"
  fi
  mode="docker linux/$platform (Docker server: $server_arch)"
  case "$platform" in
    amd64) build_desc="cross-compiled in an x86_64 container for $TARGET_TRIPLE" ;;
    arm64) build_desc="built natively in an arm64 container for $TARGET_TRIPLE" ;;
  esac

  # Image tag: toolchain, platform and a checksum of the Dockerfile.
  df_sum="$(dockerfile | cksum | awk '{ print $1 }')"
  image="$IMAGE_REPO:rust-$toolchain-$platform-$df_sum"
  if [ "$rebuild_image" = 1 ] || ! docker image inspect "$image" >/dev/null 2>&1; then
    note "building the image $image (needs the network: apt, rustup)"
    image_built="yes"
    if ! dockerfile | docker build --platform "linux/$platform" \
        --build-arg "RUST_TOOLCHAIN=$toolchain" -t "$image" - 2>&1 \
        | tee "$OUT/image-build.log"; then
      die 2 "the image build failed; see $OUT/image-build.log"
    fi
  fi

  # Versions inside the image; stop if its QEMU cannot emulate SME2.
  acquire_lock
  # shellcheck disable=SC2016 # ${Version} is dpkg-query's, not ours
  versions="$(docker run --rm --platform "linux/$platform" "$image" \
    sh -c 'qemu-aarch64 --version | head -n 1; rustc --version; dpkg-query -W -f="\${Version}\n" qemu-user 2>/dev/null || echo unknown' 2>&1)" \
    || die 2 "cannot run the image $image: $versions"
  release_lock
  qemu_version="$(printf '%s\n' "$versions" | sed -n 1p)"
  rustc_version="$(printf '%s\n' "$versions" | sed -n 2p)"
  qemu_package="$(printf '%s\n' "$versions" | sed -n 3p)"
  image_id="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)"
  [ -n "$image_id" ] || image_id="unknown"
  if ! printf '%s\n' "$qemu_version" | qemu_new_enough; then
    die 2 "QEMU in the image is too old for SME2: '$qemu_version' (need qemu-user >= ${QEMU_MIN_MAJOR}.${QEMU_MIN_MINOR}); the base image's qemu-user must be updated"
  fi
  note "image $image: $qemu_version; $rustc_version"
  if [ "$image_only" = 1 ]; then
    printf '%s\n%s\n%s\n%s\n%s\n' "$image" "$image_id" "$qemu_version" "qemu-user package $qemu_package" "$rustc_version"
    exit 0
  fi

  # One build volume per checkout and platform, named from the checkout's path.
  path_sum="$(printf '%s' "$ROOT" | cksum | awk '{ print $1 }')"
  base="$(printf '%s' "$(basename "$ROOT")" | tr -c 'A-Za-z0-9_.-' '_' | cut -c 1-40)"
  build_volume="oxiblas-sme2-qemu-$platform-$base-$path_sum"
  if [ "$reset" = 1 ]; then
    note "reset: removing the build volume $build_volume"
    docker volume rm -f "$build_volume" >/dev/null
  fi
fi

# ----------------------------------------------------------------- passes
# Results per pass, read by the summary.
positive_result=""
negative_result=""
hardware_result=""
report_rc="-"

# Copies the checkout for the hardware pass (rsync, as the native runner
# does), once per run: a second copy would delete the Cargo.lock that cargo
# wrote into the copy of a checkout that has none.
# shellcheck disable=SC2329 # run by hardware_job
hardware_copy() {
  mkdir -p "$HW_DIR/src"
  rsync -a --delete --exclude=/target --exclude=/.git --exclude=/.claude "$ROOT"/ "$HW_DIR/src"/
}

# Runs cargo test in the copy, with the environment of the hardware pass
# (hw_env; it replaces the calling shell: run it in a subshell); the
# arguments are cargo test's.
# shellcheck disable=SC2329 # run by start_group_job (the detection report)
hardware_cargo() {
  cd "$HW_DIR/src"
  exec env "${hw_env[@]}" "${hw_cargo[@]}" test "$@"
}

# The background job of the hardware pass: the copy, then its steps (the
# arguments, as for STEP_LOOP) in it, with the gate variables of the pass.
# shellcheck disable=SC2329 # run by start_group_job
hardware_job() {
  printf '== rustc : %s\n' "$rustc_version"
  hardware_copy
  cd "$HW_DIR/src"
  exec env "${hw_env[@]}" SQ_TOKEN="$run_token" SQ_PLUS="${hw_cargo[1]:-}" "$BASH" -c "$STEP_LOOP" _ "$@"
}

run_pass() {
  local pass="$1" cpu="$2" require="$3" expect_no="$4"
  local log="$OUT/$pass.log" fifo="$OUT/.$pass.fifo" rc=0 start elapsed create_out
  local n_results passed failed ignored skips gate_ran gate_skipped gate_declined verdict reason
  local stage_verdict stage_reason stage_counts
  local gate_env="OXIBLAS_TEST_REQUIRE_SME2=$require OXIBLAS_TEST_EXPECT_NO_SME2=$expect_no $MARKERS_VAR=1"
  local steps job_rc end build step_log step_rc stage_lines="" stage_console="" split_ok=1 whole_in
  # The steps of the pass's job: the unit tests, then the public-path stage.
  steps=(unit "${#cargo_args[@]}" "${cargo_args[@]}")
  if [ "$public_path" = 1 ]; then
    steps+=(public-release "${#stage_release_args[@]}" "${stage_release_args[@]}"
      public-debug "${#stage_debug_args[@]}" "${stage_debug_args[@]}")
  fi
  : > "$log"
  {
    printf '== pass  : %s\n' "$pass"
    printf '== mode  : %s\n' "$mode"
    printf '== cpu   : %s\n' "$cpu"
    printf '== env   : %s\n' "$gate_env"
    printf '== volume: %s\n' "$build_volume"
  } | tee -a "$log"
  start="$SECONDS"
  # The pass runs in the background with its output going through a FIFO to
  # tee: `$!` is then the docker client (or cargo) itself, which a trap can
  # stop, and `wait` returns its exit status (and, unlike a foreground job,
  # is interrupted by a signal at once). In Docker mode that status is the
  # container's (`docker start --attach` returns it), else docker's own.
  # With --native and --hardware the job has a process group of its own, so
  # that a signal stops the test binary with cargo; tee stays in this group
  # (started without job control, it ignores SIGINT; it ends with the job).
  current_log="$log"
  split_pass="$pass"
  current_fifo="$fifo"
  rm -f "$fifo"
  mkfifo "$fifo"
  tee -a "$log" < "$fifo" &
  tee_pid=$!
  if [ "$hardware" = 1 ]; then
    hw_env=(CARGO_TARGET_DIR="$HW_DIR/target" CARGO_TERM_COLOR=never
      OXIBLAS_TEST_REQUIRE_SME2="$require" OXIBLAS_TEST_EXPECT_NO_SME2="$expect_no" "$MARKERS_VAR=1")
    start_group_job "$fifo" hardware_job "${steps[@]}"
  elif [ "$native" = 1 ]; then
    start_group_job "$fifo" env SQ_SRC="$ROOT" SQ_WORK="$OUT/native/src" SQ_CPU="$cpu" SQ_TOOLCHAIN="$toolchain" \
      SQ_TOKEN="$run_token" CARGO_TARGET_DIR="$OUT/native/target" CARGO_TERM_COLOR=never \
      OXIBLAS_TEST_REQUIRE_SME2="$require" OXIBLAS_TEST_EXPECT_NO_SME2="$expect_no" \
      "$MARKERS_VAR=1" bash -c "$INNER" _ "${steps[@]}"
  else
    local docker_env=(-e CARGO_TARGET_DIR=/target -e CARGO_TERM_COLOR=never
      -e SQ_SRC=/src -e SQ_WORK=/work/src -e "SQ_CPU=$cpu" -e "SQ_TOOLCHAIN=$toolchain" -e "SQ_TOKEN=$run_token")
    local name
    if [ "${#forward_names[@]}" -gt 0 ]; then
      for name in "${forward_names[@]}"; do docker_env+=(-e "$name"); done
    fi
    docker_env+=(-e "OXIBLAS_TEST_REQUIRE_SME2=$require" -e "OXIBLAS_TEST_EXPECT_NO_SME2=$expect_no"
      -e "$MARKERS_VAR=1")
    acquire_lock
    current_container="oxiblas-sme2-qemu-$$-$pass"
    # `docker run` = create + start. The container is created in the
    # foreground: a signal that arrives meanwhile is handled only once
    # `docker create` has returned, so the trap never misses a container that
    # a create request still in flight would bring into being. It is then
    # started (and attached) in the background.
    if create_out="$(docker create --rm --init --name "$current_container" --platform "linux/$platform" \
        -v "$ROOT:/src:ro" \
        -v "$build_volume:/target" \
        -v "$REGISTRY_VOLUME:/opt/cargo/registry" \
        "${docker_env[@]}" \
        "$image" bash -c "$INNER" _ "${steps[@]}" 2>&1)"; then
      docker start --attach "$current_container" > "$fifo" 2>&1 &
      child_pid=$!
    else
      rc=$?
      printf '%s\ndocker create exited with %s\n' "$create_out" "$rc" > "$fifo"
    fi
  fi
  if [ -n "$child_pid" ]; then
    wait "$child_pid" || rc=$?
  fi
  child_pid=""
  child_group=0
  current_container=""
  wait "$tee_pid" || true
  tee_pid=""
  rm -f "$fifo"
  current_fifo=""
  release_lock
  elapsed=$((SECONDS - start))
  # The job's output, collected in the log, goes to the logs of its steps;
  # the unit tests' status is the one their end line records (the job's own
  # when there is none: the job ended before or inside them). A split that
  # failed (awk) fails the pass; the complete output stays in
  # <pass>.log.stream, which then goes into the archive. (A signal during
  # the split is handled when its command has ended: the handler splits
  # again, from the complete output.)
  split_logs "$pass" || split_ok=0
  split_pass=""
  job_rc="$rc"
  end="$(step_exit "$log" unit "$run_token")"
  if [ -n "$end" ]; then rc="${end% *}"; fi

  if [ "$hardware" = 1 ]; then
    # What the library's detection read (the same build; its own process
    # group), then the values this script reads itself.
    note "the detection report: test_sme2_detect_report --nocapture (log detect-report.log)"
    report_rc=0
    current_log="$OUT/detect-report.log"
    start_group_job "$OUT/detect-report.log" \
      hardware_cargo -p oxiblas-blas --features "sme2,parallel" --lib --release test_sme2_detect_report -- --nocapture
    wait "$child_pid" || report_rc=$?
    child_pid=""
    child_group=0
    hw_raw_values > "$OUT/.hardware-values"
    judge_hardware "$rc" "$log" "$OUT/detect-report.log" "$OUT/.hardware-values" "$report_rc"
  else
    judge_pass "$pass" "$rc" "$log"
  fi
  if [ "$split_ok" = 0 ]; then
    verdict="FAIL"
    # The complete output is in the renamed copy, or still in the log when
    # the rename itself failed.
    if [ -f "$OUT/$pass.log.stream" ]; then whole_in="$pass.log.stream"; else whole_in="$pass.log"; fi
    reason="${reason:+$reason; }the split of the pass's output into its logs failed (the complete output is in $whole_in)"
  fi
  current_log=""
  # The public-path stage, per build: its verdict joins the pass's.
  if [ "$public_path" = 1 ]; then
    for build in release debug; do
      step_log="$OUT/$pass-public-$build.log"
      end="$(step_exit "$step_log" "public-$build" "$run_token")"
      if [ -n "$end" ]; then step_rc="${end% *}"
      elif step_begun "$step_log" "public-$build" "$run_token"; then step_rc=unended
      else step_rc=unrun; fi
      judge_stage "$pass" "$build" "$step_rc" "$step_log" "$job_rc"
      if [ "$stage_verdict" != PASS ]; then verdict="FAIL"; reason="${reason:+$reason; }$stage_reason"; fi
      stage_lines="$stage_lines$(printf '\n  public %-7s: log %s; %s%s; %s%s' "$build" "${step_log##*/}" "$stage_counts" "${end:+ after ${end#* }}" \
        "$stage_verdict" "${stage_reason:+ (${stage_reason#public path, "$build" build: })}")"
      stage_console="$stage_console$(printf '\n== %s public path, %s build: %s (%s)' "$pass" "$build" "$stage_verdict" "$stage_counts")"
      printf '== public path, %s build: %s%s (%s)\n' "$build" "$stage_verdict" "${stage_reason:+ - ${stage_reason#public path, "$build" build: }}" "$stage_counts" >> "$step_log"
    done
  else
    stage_lines="$(printf '\n  public path: left out (--no-public-path)')"
  fi
  local result
  result="$(printf '%s\n' \
    "  cpu       : $cpu" \
    "  gate env  : $gate_env" \
    "  tests     : $passed passed, $failed failed, $ignored ignored ($n_results test binaries)" \
    "  gate      : $gate_ran ran SME2, $gate_skipped skipped SME2, $gate_declined backend declined (sme2-gate lines)" \
    "  skip lines: $skips" \
    "  cargo exit: $rc" \
    "  time      : ${elapsed}s")$stage_lines
  verdict   : $verdict${reason:+ ($reason)}"
  case "$pass" in
    positive) positive_result="$result" ;;
    negative) negative_result="$result" ;;
    *) hardware_result="$result" ;;
  esac
  printf '== %s pass: %s%s (%s passed, %s failed, gate %s ran / %s skipped / %s declined, %ss)\n' \
    "$pass" "$verdict" "${reason:+ - $reason}" "$passed" "$failed" "$gate_ran" "$gate_skipped" \
    "$gate_declined" "$elapsed" | tee -a "$log"
  if [ -n "$stage_console" ]; then printf '%s\n' "${stage_console#?}"; fi
  [ "$verdict" = PASS ]
}

overall=0
if [ "$hardware" = 1 ]; then
  run_pass hardware "none (this machine's own CPU, no QEMU)" 1 0 || overall=1
else
  if [ "$want_positive" = 1 ]; then
    run_pass positive "$CPU_POSITIVE" 1 0 || overall=1
  fi
  if [ "$want_negative" = 1 ]; then
    run_pass negative "$CPU_NEGATIVE" 0 1 || overall=1
  fi
fi

# ---------------------------------------------------------------- summary
# The passes are over: INT, TERM and HUP are ignored from here on (tar and
# awk inherit that), and summary.txt and the archive are written under
# temporary names and renamed when complete: none is ever half written.
trap '' INT TERM HUP
# Git is only read: `status` would otherwise refresh the stat data cached in
# the index and rewrite the index file (a write outside target/sme2-qemu/,
# racing with the user's own git commands); --no-optional-locks and
# GIT_OPTIONAL_LOCKS=0 forbid that. rev-parse and log never write.
git_commit="unknown (not a git checkout or git missing)"
git_dirty="unknown"
export GIT_OPTIONAL_LOCKS=0
if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  git_commit="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  git_subject="$(git -C "$ROOT" log -1 --format=%s 2>/dev/null || true)"
  [ -n "$git_subject" ] && git_commit="$git_commit ($git_subject)"
  if git_status="$(git --no-optional-locks -C "$ROOT" status --porcelain 2>/dev/null)"; then
    changed="$(printf '%s' "$git_status" | grep -c '' || true)"
    if [ "$changed" = 0 ]; then git_dirty="no"; else git_dirty="yes ($changed changed or untracked paths)"; fi
  fi
fi

# A cargo test command as text (the arguments), quoted where needed.
command_text() {
  local text="cargo +$toolchain test" arg
  if [ "$hardware" = 1 ]; then text="${hw_cargo[*]} test"; fi
  for arg in "$@"; do
    case "$arg" in
      ''|*[!A-Za-z0-9_.,:=/+-]*) text="$text $(printf '%q' "$arg")" ;;
      *) text="$text $arg" ;;
    esac
  done
  printf '%s\n' "$text"
}
cargo_cmd="$(command_text "${cargo_args[@]}")"
if [ "$public_path" = 1 ]; then
  stage_cmd="$(command_text "${stage_release_args[@]}")
               $(command_text "${stage_debug_args[@]}")"
else
  stage_cmd="left out (--no-public-path)"
fi
forwarded="none"
if [ "${#forward_names[@]}" -gt 0 ]; then
  forwarded=""
  for name in "${forward_names[@]}"; do forwarded="$forwarded $name=${!name}"; done
  forwarded="${forwarded# }"
fi
if [ "$overall" = 0 ]; then overall_text="PASS"; else overall_text="FAIL"; fi

# The archive's tar headers get user and group 0 without names (GNU tar or
# bsdtar), when tar takes the options; otherwise the summary says so.
mkdir "$OUT/.archive"
tar_version="$(tar --version 2>/dev/null | awk 'NR == 1' || true)"
case "$tar_version" in
  *"GNU tar"*) tar_flavor="GNU tar"; tar_opts=(--owner=0 --group=0 --numeric-owner) ;;
  *bsdtar*) tar_flavor="bsdtar"; tar_opts=(--uid 0 --gid 0 --uname "" --gname "") ;;
  *) tar_flavor="unknown tar"; tar_opts=() ;;
esac
if [ "${#tar_opts[@]}" -gt 0 ] && (cd "$OUT/.archive" && tar "${tar_opts[@]}" -cf - .) > /dev/null 2>&1; then
  tar_note="user and group 0, no names ($tar_flavor $(printf '%q ' "${tar_opts[@]}" | sed 's/ $//'))"
else
  tar_opts=()
  tar_note="WARNING: $tar_flavor took no option to leave out the user and group names; the archive's tar headers carry them"
fi

{
  if [ "$hardware" = 1 ]; then
    printf 'oxiblas SME2 on hardware: %s\n' "$overall_text"
  else
    printf 'oxiblas SME2 under QEMU: %s\n' "$overall_text"
  fi
  printf 'date (UTC)   : %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
  printf 'host         : %s\n' "$host_desc"
  if [ "$hardware" = 1 ]; then
    printf 'architecture : host %s; test binaries for the host, %s\n' "$host_machine" "$build_desc"
    printf 'cpu          : %s\n' "${hw_cpu:-unknown}"
  else
    printf 'architecture : host %s; test binaries %s, %s\n' "$host_machine" "$TARGET_TRIPLE" "$build_desc"
  fi
  printf 'mode         : %s\n' "$mode"
  if [ -n "$image" ]; then
    printf 'image        : %s (built in this run: %s)\n' "$image" "$image_built"
    printf 'image ID     : %s\n' "$image_id"
  fi
  printf 'build dir    : %s\n' "$build_volume"
  if [ "$hardware" = 0 ]; then
    printf 'qemu         : %s\n' "$qemu_version"
    printf 'qemu-user pkg: %s\n' "$qemu_package"
  fi
  printf 'rustc        : %s\n' "$rustc_version"
  if [ "$hardware" = 1 ]; then
    printf 'toolchain    : %s, release %s, host %s; the workspace'\''s rust-version %s (an older toolchain is refused)\n' \
      "$toolchain_desc" "$rustc_release" "$host_triple" "$rust_floor"
    printf 'target runner: %s\n' "$hw_runner"
  fi
  printf 'cargo command: %s\n' "$cargo_cmd"
  printf 'public path  : %s\n' "$stage_cmd"
  printf 'forwarded env: %s\n' "$forwarded"
  printf 'git commit   : %s\n' "$git_commit"
  printf 'git dirty    : %s\n' "$git_dirty"
  if [ "$hardware" = 1 ]; then
    printf 'hardware pass (SME2 must run and be detected, the rule of the positive pass; log hardware.log):\n%s\n' "$hardware_result"
    printf 'detection report (test_sme2_detect_report -- --nocapture, cargo exit %s; log detect-report.log):\n' "$report_rc"
    hw_report_lines "$OUT/detect-report.log" | awk '{ print "  " $0; n++ } END { if (!n) print "  none (the test did not run; see detect-report.log)" }'
    if [ "$(uname -s)" = Darwin ]; then
      printf 'values read by this script (/usr/sbin/sysctl -n of the keys detect.rs names, and the CPU brand):\n'
    else
      printf 'values read by this script (/proc/cpuinfo, /proc/sys/abi/sme_default_vector_length):\n'
    fi
    awk -F'\t' '{ printf "  %s: %s\n", $1, $2 }' "$OUT/.hardware-values" 2>/dev/null || true
  else
    if [ "$want_positive" = 1 ]; then
      printf 'positive pass (SVL 512, SME2 must run; log positive.log):\n%s\n' "$positive_result"
    else
      printf 'positive pass: not requested\n'
    fi
    if [ "$want_negative" = 1 ]; then
      printf 'negative pass (SVL <= 256, SME2 must be declined; log negative.log):\n%s\n' "$negative_result"
    else
      printf 'negative pass: not requested\n'
    fi
  fi
  printf 'archive      : %s\n' "$tar_note"
  printf 'overall      : %s\n' "$overall_text"
} > "$OUT/summary.txt.tmp"
mv -f "$OUT/summary.txt.tmp" "$OUT/summary.txt"
rm -f "$OUT/.hardware-values"

# The archive gets copies in which the checkout's absolute path (in native
# mode in cargo's lines, for example) reads `<checkout>`, and then the home
# directory's `<home>`. The replacement is literal: the paths come in through
# ENVIRON (awk -v would interpret backslashes) and are found with index(),
# not as regular expressions.
archive_files=(summary.txt)
for f in positive.log positive-public-release.log positive-public-debug.log \
  negative.log negative-public-release.log negative-public-debug.log \
  hardware.log hardware-public-release.log hardware-public-debug.log detect-report.log image-build.log \
  positive.log.stream negative.log.stream hardware.log.stream; do
  [ -f "$OUT/$f" ] && archive_files+=("$f")
done
home_path="${HOME:-}"
case "$home_path" in /|.) home_path="" ;; esac
for f in "${archive_files[@]}"; do
  SQ_ROOT="$ROOT" SQ_HOME="$home_path" awk '
    function swap(line, from, to,   out, i, n) {
      n = length(from); out = ""
      while (n > 0 && (i = index(line, from)) > 0) {
        out = out substr(line, 1, i - 1) to
        line = substr(line, i + n)
      }
      return out line
    }
    { print swap(swap($0, ENVIRON["SQ_ROOT"], "<checkout>"), ENVIRON["SQ_HOME"], "<home>") }' "$OUT/$f" > "$OUT/.archive/$f"
done
tar ${tar_opts[@]+"${tar_opts[@]}"} -czf "$OUT/sme2-qemu-results.tar.gz.tmp" -C "$OUT/.archive" "${archive_files[@]}"
mv -f "$OUT/sme2-qemu-results.tar.gz.tmp" "$OUT/sme2-qemu-results.tar.gz"
rm -rf "$OUT/.archive"

cat "$OUT/summary.txt"
printf '\nResults: %s\n' "$OUT/sme2-qemu-results.tar.gz"
exit "$overall"
