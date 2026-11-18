#!/usr/bin/env bash
# x86_gemm_runbook.sh -- the x86-64 Linux measurements of the 0.2.4 GEMM work.
#
# oxiblas 0.2.4 adds GEMM code paths for x86-64 (an f32 16x6 AVX2 micro-kernel,
# AVX2 small-matrix kernels, AVX2 thin-product kernels, a per-tile parallel
# packing scope, B-block caps for short, wide products). They sit behind a
# policy: the default of x86-64 builds (LEGACY) reproduces 0.2.3 (every
# result that is not a NaN has the bits of 0.2.3 and every NaN result sits
# where 0.2.3 has one; only which NaN a result carries where an operation
# meets two NaNs may differ), and the new behaviour is built with
# RUSTFLAGS="--cfg oxiblas_gemm_candidate". Each default is switched on only
# where a measurement on real hardware passes a fixed rule (on aarch64 the
# default already has the fields measured there; nothing was measured on
# x86-64, and RUSTFLAGS="--cfg oxiblas_gemm_legacy" restores 0.2.3's routes
# on every target).
# This script collects everything needed for that decision in one run on an
# x86-64 Linux machine with AVX2 and FMA, and packs it into one archive. It
# also applies the rule's last condition itself (step 5): a field is adopted
# only where both A/B passes say ADOPT and no verdict is INVALID.
#
# Usage (from anywhere; the checkout is the one the script is in):
#   crates/oxiblas-blas/tools/x86_gemm_runbook.sh [options]
# For a long unattended run, from the checkout's root directory:
#   mkdir -p target && nohup crates/oxiblas-blas/tools/x86_gemm_runbook.sh \
#     > target/x86-runbook.out 2>&1 &
# (target/x86-runbook.out is the terminal output; the next run started this
# way overwrites it. Every command's complete output is also in logs/.)
#
# What to send back: target/x86-runbook/x86-runbook-results.tar.gz
#
# Steps (each command has its own log in target/x86-runbook/logs/ and a line
# in summary.txt; a failed step does not stop the ones after it):
#   1  machine.txt: lscpu, kernel release, distribution (PRETTY_NAME), C
#      library version, kernel command line (values of root=, resume=, ip=,
#      nfsroot=, cryptdevice=, BOOTIF=, *hostname=, luks*, rd.luks*, rd.lvm*
#      read <redacted>, UUIDs <uuid>), rustc -vV, cargo -V, active toolchain,
#      CPU governor, online CPUs, physical cores, memory, OpenBLAS library,
#      the names of the inherited CARGO_* and RUST* variables (values only
#      for CARGO_PROFILE_*), the caller's OpenBLAS and OpenMP variables (with
#      values), and the cargo config files of the checkout's parent
#      directories and the cargo home with their key names (never content).
#   2  Tests under the default (LEGACY) policy:
#        cargo test -p oxiblas-blas --release [--features F] --no-fail-fast
#      for F in (none), parallel, max-simd-256, max-simd-128, force-scalar.
#      The parallel run adds `-- --show-output` (for step 4).
#   3  The same under the candidate policy (RUSTFLAGS="--cfg
#      oxiblas_gemm_candidate", CARGO_TARGET_DIR=target/cand); then
#      3-candall-parallel, the parallel feature set under RUSTFLAGS="--cfg
#      oxiblas_gemm_candidate --cfg oxiblas_gemm_bcap_all"
#      (CARGO_TARGET_DIR=target/cand-bcap, B-block caps in both drivers);
#      then the workspace tests under the candidate policy:
#        cargo test --workspace --exclude oxiblas-benchmarks --release
#      A test unit (steps 2, 3, 5) fails when cargo exits non-zero, when its
#      log has no `test result:` line, or when no test passed.
#   4  Proof that the AVX2 code ran, read from the logs of the parallel runs
#      of steps 2 and 3 (run with --show-output): with OXIBLAS_REQUIRE_AVX2
#      set, every AVX2-gated test that reaches the AVX2+FMA tier prints
#      "<test>: AVX2+FMA tier confirmed", and one that does not fails. Each
#      policy needs at least one such line, no SKIPPED line and no
#      OXIBLAS_REQUIRE_AVX2 failure. The number of passed tests whose names
#      contain "avx2" is reported for information only.
#   tiny  Information only: gemm_tiny_call_check.sh (beside this script; the
#      per-call cost of tiny GEMM products of this checkout against a build
#      of the base release the tool names, exported with git archive; its
#      header has the judged set and the limit) with the tool's own defaults
#      and --out target/x86-runbook/tiny-call (--quick passes its --quick).
#      The tool's verdict line and settings line go into summary.txt. Its
#      exit 1 (FAIL) or 3 (INCOMPLETE) is recorded as INFO, its exit 2 (no
#      git checkout, no base release tag, ...) as SKIPPED with its reason:
#      neither ever makes the run FAIL or INCOMPLETE.
#   5  The in-crate A/B harness, twice: pass1 runs the groups in the order
#        ab_seq_f32_kernel ab_seq_small_simd ab_seq_small_trans_copy
#        ab_seq_thin ab_seq_b_cap ab_seq_pack_cache
#        ab_par_pack_scope ab_par_thin ab_par_b_cap
#      and pass2 in the reverse order, each group in its own process:
#        cargo test -p oxiblas-blas --release --features parallel --lib
#          --no-fail-fast -- --ignored --test-threads=1 --nocapture
#          --exact level3::gemm_ab_tests::<group>
#      (--exact: a plain filter is a substring, and ab_seq_thin would also
#      run the information group ab_seq_thin_grid, 1530 cases that decide
#      nothing; run that one by hand with the same command when a scope is
#      to be derived), with OXIBLAS_AB_TAG=pass1 or pass2 and
#      OXIBLAS_AB_OUT=ab.tsv (one file for both passes), after a build of the
#      test binary. Each process gets OXIBLAS_AB_ROUNDS (22; --quick 4),
#      OXIBLAS_AB_QUICK (set but empty; --quick 1), OXIBLAS_AB_RETRIES (8,
#      or the caller's value when it is set and not empty: with the harness's
#      default of 2 a large group kept cases with a disturbed control, which
#      makes a run INVALID) and OXIBLAS_AB_MIN_SAMPLE_MS explicitly, so a
#      cargo config [env] table cannot fill them in. The ab_seq groups run
#      pinned with
#      `taskset -c 2` (another CPU when 2 is not allowed) and a 2 ms least
#      sample; the ab_par groups get OXIBLAS_AB_THREADS = 2, 4, 6, 8, the
#      physical and the online CPU count (deduplicated, at most the online
#      count) and 10 ms. A group process fails when it wrote no SET line,
#      when its SET and END lines differ in number (a group run did not
#      complete), when its SET and ABV lines differ in number (a group run
#      without its verdict), or, without --quick, when a SET line lacks
#      grade=decision (the harness's mark of data that may decide; the
#      summary quotes the grade, group, rounds_requested, min_sample_ms,
#      quick and debug_build of the first such line). One summary line per
#      pass and kind (sequential, parallel) counts the AB, ABR, ABV, SET and
#      END lines it added, and fails when a group process failed (each named
#      with its reason), a SET line lacks tier=Avx2Fma, SET and END (or ABV)
#      differ in number, or, without --quick, a SET line lacks grade=decision.
#      ab_seq_f32_kernel and ab_seq_small_simd exist on x86-64 only.
#      Then 5-ab-verdicts reads every verdict (ABV line; decision words
#      ADOPT, KEEP_LEGACY, NO_TARGETS, INVALID, SMOKE; the cases with an
#      invalid control: the control's paired median outside 1.5 % or one of
#      its quartiles outside 1.015², after the re-measurements) of both
#      passes and compares them candidate by
#      candidate (group, run, candidate); summary.txt shows both passes'
#      decisions in a table in every run that includes step 5. It is WARN,
#      and a run that is not limited ends INCOMPLETE (exit 3), when a verdict
#      is INVALID or a case's control was invalid (the machine was not
#      quiet), when the passes disagree on a candidate or a candidate has a
#      verdict in one pass only, or when a full
#      run (without --quick) has no verdict or SMOKE verdicts; the missing
#      parts say what to do (run step 5 again on a quiet machine). An
#      informational candidate (never adopted) whose passes disagree is named
#      in the details only and does not make the run INCOMPLETE. A verdict
#      it cannot read (an unknown decision word, a repeated or unattributable
#      one) fails it. The verdicts of a --quick run are smoke grade and are
#      not judged: 5-ab-verdicts is then INFO (with the counts, also of the
#      invalid controls), with no missing part and no advice, unless a
#      verdict cannot be read. At start-up the script checks that stats.rs
#      has exactly the decision words step 5 judges (one `Self::<Name> =>
#      "<WORD>"` arm each, no other upper-case arm), so that a word the
#      harness adds stops the run there (exit 2), not at its end. The code
#      of step 5 is x86_gemm_runbook_ab.sh beside this script: without it
#      every run (also --dry-run and --self-test; not --help) stops with
#      exit 2 before anything is written.
#   6  The OpenBLAS comparison benches (crates/oxiblas-benchmarks, bench
#      `comparison`, feature compare-openblas, 424 benchmarks): the default
#      policy saved as the criterion baseline `legacy`, then the candidate
#      policy (in target/cand) compared with `--baseline legacy`. Both builds
#      share one criterion directory, target/x86-runbook/criterion
#      (CRITERION_HOME), so the comparison finds the baseline. Both run with
#      OPENBLAS_NUM_THREADS=1 (the parallel groups set their thread count
#      themselves). The summary gives criterion's faster/slower verdicts
#      separately for the oxiblas and the openblas benchmarks (the openblas
#      ones are the same library in both runs: the noise floor). If the
#      OpenBLAS library is missing, the step is recorded as skipped.
#   7  summary.txt and the archive (always written, also after a failure or an
#      interruption).
# Compiler flags: every cargo command runs with RUSTFLAGS set explicitly:
# empty for the default policy, which keeps any rustflags of cargo config
# files and of CARGO_TARGET_<triple>_RUSTFLAGS out of the build, and the
# cfg flags above for the candidate builds; so the two sides are built with
# the same flags apart from the cfg. CARGO_ENCODED_RUSTFLAGS,
# CARGO_BUILD_RUSTFLAGS, every CARGO_TARGET_*_RUSTFLAGS, RUSTDOCFLAGS,
# CARGO_ENCODED_RUSTDOCFLAGS, CARGO_BUILD_RUSTDOCFLAGS, CARGO_TARGET_DIR,
# CARGO_BUILD_TARGET_DIR, CRITERION_HOME, RAYON_NUM_THREADS,
# OPENBLAS_NUM_THREADS, OPENBLAS_CORETYPE, OPENBLAS_VERBOSE,
# GOTO_NUM_THREADS, OMP_NUM_THREADS and every OXIBLAS_* variable of the
# caller are unset, and OXIBLAS_REQUIRE_AVX2=1 is set: an AVX2 test that
# would skip fails instead. CARGO_PROFILE_*, RUSTC_WRAPPER, RUSTUP_TOOLCHAIN
# and CARGO_INCREMENTAL are left as they are (they act on both policies
# alike) and recorded in machine.txt. Everything runs with LC_ALL=C. The
# script refuses to run when CARGO_BUILD_TARGET is set, or when a cargo
# config file (checkout's parent directories, cargo home) sets build.target
# or an [env] entry OXIBLAS_*, OPENBLAS_*, OMP_* or RAYON_* (in any form).
#
# Needs:
#   - Linux on x86-64 with AVX2 and FMA (checked in /proc/cpuinfo).
#   - A Rust toolchain (rustup; Rust 1.89 or newer; cargo and rustc must be
#     on PATH) and a C linker (Debian/Ubuntu: build-essential). cargo
#     fetches crates from crates.io.
#   - util-linux for taskset and lscpu (Debian/Ubuntu: util-linux, installed
#     by default). Without taskset step 5 runs unpinned and the run is
#     INCOMPLETE.
#   - OpenBLAS for step 6 (Debian/Ubuntu: libopenblas0). Its path is
#     OPENBLAS_LIB when set, else /usr/lib/x86_64-linux-gnu/libopenblas.so.0.
#     Without it the run is INCOMPLETE.
#   - About 12 GB of free disk space under target/, and for step 6 about 5 GB
#     of free memory (one thin bench operand is 2 GiB).
#
# Duration (full run on a current 8-core desktop; estimates, not measured on
# such a machine): step 2 about 20-40 minutes, step 3 about 40-75 minutes,
# step 4 seconds, the tiny step about 5-15 minutes (two release builds, then
# alternating timed rounds), step 5 about 2.5-3 hours (one pass of all groups
# takes about 75-90 minutes, and there are two; a re-measured case adds its
# time again), step 6 about 1.5 hours (424
# benchmarks per policy: 240 small ones of about 3 s, the rest of about
# 8-10 s, plus two release builds). Altogether about 5-7 hours. --quick
# takes about 25-50 minutes. Leave the machine otherwise idle: the tiny step
# and steps 5 and 6 are timings.
#
# Options:
#   --only LIST        run only these steps (comma-separated: the numbers 1-6
#                      and tiny, e.g. --only 5 or --only 2,3,4,tiny); step 7
#                      always runs. Step 4 reads the logs of steps 2 and 3 of
#                      the same run.
#   --skip LIST        run all steps but these
#   --quick            smoke mode: steps 2 and 3 run only the parallel feature
#                      set with the test name filters `avx2` and `gemm_plan`
#                      (no workspace run), the tiny step passes the tool's
#                      --quick, step 5 sets OXIBLAS_AB_QUICK=1 and
#                      OXIBLAS_AB_ROUNDS=4 (smoke-grade data, whose verdicts
#                      are not judged), and step 6 runs the benches in
#                      criterion's test mode (no timing)
#   --bench-filter RE  pass RE to criterion in step 6 (a regular expression
#                      over benchmark names, e.g. 'thin|short_wide'); step 6
#                      then measures only those benchmarks
#   --allow-no-avx2    a testing aid, not for the real run: run on a machine or
#                      an emulator without AVX2/FMA (or not on x86-64 Linux at
#                      all). OXIBLAS_REQUIRE_AVX2 is then not set, AVX2 tests
#                      skip, and the AVX2 checks of steps 4 and 5 report
#                      WAIVED instead of failing.
#   --dry-run          print every command of every selected step with its
#                      environment and run nothing; nothing is written (no
#                      files at all, also not in target/)
#   --self-test        check the script's own logic on built-in samples (step
#                      5 and its verdicts with a stand-in for cargo, up to the
#                      summary and exit status; the decision words of
#                      stats.rs, and the start-up stop on a new word or a
#                      missing step-5 file in a skeleton checkout; the tiny
#                      step with a stand-in tool; test counts, path
#                      scrubbing, lock, cargo config checks); writes only in
#                      target/x86-runbook-selftest.<pid>/ (removed at the
#                      end); exits 0 when every expectation holds, else 1.
#                      The self-test is x86_gemm_runbook_selftest.sh beside
#                      this script (without it, exit 2)
#   -h, --help         print this text
# --only, --skip and --bench-filter may each be given once; --only and --skip
# cannot be combined. A run with --only, --skip, --quick, --bench-filter or
# --allow-no-avx2 is a limited run, not a complete measurement: its overall
# line reads "PASS (limited run: ...)" instead of PASS.
#
# Environment:
#   OPENBLAS_LIB       the OpenBLAS shared library for step 6 (default
#                      /usr/lib/x86_64-linux-gnu/libopenblas.so.0)
#   CARGO_BUILD_JOBS   passed to cargo as it is
#
# Output, in target/x86-runbook/ of this checkout:
#   logs/<unit>.log                 the complete output of each command
#   machine.txt                     step 1
#   ab.tsv                          step 5 (lines starting AB, ABR, ABV or AB#)
#   tiny-call/                      the tiny step: the tool's summary.txt,
#                                   logs/ and its build directories
#   criterion/                      step 6 (criterion's data and reports)
#   summary.txt                     one line per command: name, verdict, exit
#                                   code, seconds, test counts or reason (a
#                                   verdict INFO is information only); with
#                                   step 5 the table of both passes' A/B
#                                   decisions; the missing parts
#   x86-runbook-results.tar.gz      the file to send back: summary.txt,
#                                   machine.txt, ab.tsv, the logs, the
#                                   criterion estimates (JSON), Cargo.lock,
#                                   and the tiny step's summary.txt with the
#                                   logs it names (up to 1 MiB each)
# Other files the run writes: cargo's build directories target/release
# (and the rest of target/), target/cand and target/cand-bcap, the lock
# target/x86-runbook.lock/, and <checkout>/Cargo.lock, which cargo creates
# when the checkout has none (it is git-ignored; a copy goes into the
# archive).
# One run per checkout: a real run first checks the options, the machine,
# the cargo configuration and the tools (an error there changes nothing),
# then takes the lock target/x86-runbook.lock/ (its process id and start
# time); while that process runs, a second start stops with exit status 2,
# and a lock left by a process that is gone (SIGKILL, a crash) is taken
# over; the lock is removed on every other exit. Earlier results are never
# deleted: the run then moves an existing target/x86-runbook/ as a whole to
# target/x86-runbook-prev/<UTC time>/ (named in the summary and the final
# message) and starts afresh. Remove old runs: rm -rf target/x86-runbook-prev
# The archive is written under a temporary name and renamed when complete,
# so the file to send back is either the complete archive of the run that
# just ended or absent. Its tar headers carry user and group 0 instead of
# your names (when tar takes the options; otherwise the summary says so).
# No host name is recorded. Paths: the logs show every command with the
# checkout's absolute path as `<checkout>` and the home directory as
# `<home>`, replaced before any shell quoting; in the archived copies every
# form of both paths (as given and physical; plain, printf %q quoted, the
# inside of a $'...' form, Debug-escaped) is replaced the same way; and a
# last check searches every archived copy for these forms: a copy that
# still holds one is left out of the archive and named in the summary.
# Nothing is sent anywhere.
#
# Exit status: 0 for PASS (every command passed, was skipped by design or
# was waived, and in a run that is not limited step 5's verdicts are a
# decision: none INVALID, no invalid control, both passes agreeing on every
# candidate; the tiny step never counts), also "PASS (limited run: ...)",
# which only says that nothing failed; 1 (FAIL) when a command
# failed, whatever else is missing (a copy left out of the archive is then
# named in the summary too, but the status stays 1); 2 for a usage error, a
# missing prerequisite (cargo, rustc, the step-5 file
# x86_gemm_runbook_ab.sh, the AVX2 gate or A/B harness texts, decision words
# in stats.rs other than the five step 5 judges, a refused cargo
# configuration, an unwritable target/), the wrong machine or another run
# here: nothing changed (a logs/ that cannot be made after the move leaves
# the earlier run moved); 3 (INCOMPLETE) when nothing failed (no command
# failed) but a part is missing: in a run that is not limited, the
# OpenBLAS comparison, the pinning of step 5, or step 5's decision (see
# 5-ab-verdicts above: then run step 5 again on a quiet machine)
# (the final message says what to install or do), and in any run an archived
# copy left out because it still held a path (or a failed path check: then
# no archive); 128 + the signal number after INT, TERM or HUP: the running
# command and its children are stopped, and summary.txt and the archive are
# written with the status INTERRUPTED (signals are ignored meanwhile).
#---
set -euo pipefail
# Bytes, not characters: a non-ASCII path cannot break awk, sed or grep.
caller_lc_all="${LC_ALL:-}"; LC_ALL=C; export LC_ALL

readonly DEFAULT_OPENBLAS_LIB="/usr/lib/x86_64-linux-gnu/libopenblas.so.0"
readonly CANDIDATE_RUSTFLAGS="--cfg oxiblas_gemm_candidate"
readonly CANDALL_RUSTFLAGS="--cfg oxiblas_gemm_candidate --cfg oxiblas_gemm_bcap_all"
readonly FEATURE_SETS="none parallel max-simd-256 max-simd-128 force-scalar"
# shellcheck disable=SC2034 # read by step 5 (x86_gemm_runbook_ab.sh), as are the next two
readonly AB_SEQ_GROUPS="ab_seq_f32_kernel ab_seq_small_simd ab_seq_small_trans_copy ab_seq_thin ab_seq_b_cap ab_seq_pack_cache"
# shellcheck disable=SC2034
readonly AB_PAR_GROUPS="ab_par_pack_scope ab_par_thin ab_par_b_cap"
# shellcheck disable=SC2034
readonly AB_X86_ONLY_GROUPS="ab_seq_f32_kernel ab_seq_small_simd"
readonly AB_THREAD_CANDIDATES="2 4 6 8"
# Rounds of a full run (the harness's default; a decision-grade run needs at
# least 21 requested, settings.rs DECISION_ROUNDS) and of --quick.
# shellcheck disable=SC2034 # read by step 5 (x86_gemm_runbook_ab.sh)
readonly AB_FULL_ROUNDS=22 AB_QUICK_ROUNDS=4
# Re-measurements of a case with a disturbed control, unless the caller sets
# OXIBLAS_AB_RETRIES (the harness's default is 2).
# shellcheck disable=SC2034 # read by step 5 (x86_gemm_runbook_ab.sh)
readonly AB_RETRIES=8
readonly QUICK_FILTERS="avx2 gemm_plan"
# The texts of the test gate avx2_fma_tier_or_skip (gemm_plan.rs) that step 4
# searches the logs for; the start-up check makes sure they still exist.
readonly SKIP_TEXT="SKIPPED (not on the AVX2+FMA tier)"
readonly REQUIRE_TEXT="OXIBLAS_REQUIRE_AVX2 is set, but"
readonly CONFIRM_TEXT="AVX2+FMA tier confirmed"
readonly WRONG_RESULT_TEXT="wrong result, not timed"
# The grade of a SET line (the key in output.rs, the words in settings.rs);
# step 5 requires the decision word in a full run.
readonly GRADE_DECISION="grade=decision"
# The decision words of an ABV verdict entry (stats.rs Decision::word), in
# the order of the counts step 5 prints; the start-up check finds them.
readonly AB_DECISION_WORDS="ADOPT KEEP_LEGACY NO_TARGETS INVALID SMOKE"
readonly OPENBLAS_ENV_NAMES="OPENBLAS_NUM_THREADS OPENBLAS_CORETYPE OPENBLAS_VERBOSE GOTO_NUM_THREADS OMP_NUM_THREADS"
TAB="$(printf '\t')"

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

die() {
  local code="$1"; shift
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit "$code"
}

note() { printf '== %s\n' "$*" >&2; }

print_help() {
  # The header comment, from line 2 to the `#---` marker, without the `# `.
  awk 'NR == 1 { next } /^#---/ { exit } { sub(/^# ?/, ""); print }' "$SCRIPT_DIR/$SCRIPT_NAME"
}

# Checks a step list ("2,3", "5" or "4,tiny") and prints it as " 2 3 ".
parse_steps() {
  local list="$1" option="$2" item out=" "
  for item in $(printf '%s' "$list" | tr ',' ' '); do
    case "$item" in
      [1-6]|tiny) out="$out$item " ;;
      7) die 2 "$option: step 7 (summary and archive) always runs" ;;
      *) die 2 "$option: '$item' is not a step (1-6 or tiny)" ;;
    esac
  done
  [ "$out" != " " ] || die 2 "$option needs a list of steps (1-6, tiny), found '$list'"
  printf '%s\n' "$out"
}

want_step() {
  case "$only_list" in '') ;; *" $1 "*) ;; *) return 1 ;; esac
  case "$skip_list" in *" $1 "*) return 1 ;; esac
  return 0
}

# ------------------------------------------------------- cargo configuration
# The dotted names of the keys a cargo config file (TOML) defines under
# build, target, profile and env, one per line (names only, never values).
# Tables, dotted keys, quoted keys and inline tables are read; strings,
# arrays and comments are skipped, also across lines.
config_keys() {
  awk '
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
          else if (c == "\\" && substr(q, 1, 1) == "\"") i++
        } else if (c == "\"" || c == "\047") { q = quote_at(s, i); i += length(q) - 1 }
        else if (c == "#") { while (i <= n && substr(s, i, 1) != "\n") i++ }
        else if (c == "[" || c == "{") d++
        else if (c == "]" || c == "}") d--
      }
      return d <= 0 && length(q) != 3
    }
    function ws() {
      while (P <= L && (index(" \t\r\n", substr(S, P, 1)) || substr(S, P, 1) == "#"))
        if (substr(S, P, 1) == "#") { while (P <= L && substr(S, P, 1) != "\n") P++ } else P++
    }
    # Skips the string at P (any of the four TOML kinds); returns its text.
    # A multi-line string ends at the last three of 3 to 5 quotes.
    function skip_string(   q, k, e) {
      q = quote_at(S, P); P += length(q); k = ""
      while (P <= L && substr(S, P, length(q)) != q) {
        if (substr(q, 1, 1) == "\"" && substr(S, P, 1) == "\\") P++
        k = k substr(S, P, 1); P++
      }
      P += length(q); e = 0; while (length(q) == 3 && e < 2 && substr(S, P, 1) == substr(q, 1, 1)) { e++; P++ }; return k
    }
    function key_part(   start) {
      ws(); if (substr(S, P, 1) == "\"" || substr(S, P, 1) == "\047") return skip_string()
      start = P; while (P <= L && substr(S, P, 1) ~ /[A-Za-z0-9_-]/) P++
      return substr(S, start, P - start)
    }
    function key_path(   k) {
      k = key_part(); ws()
      while (substr(S, P, 1) == ".") { P++; k = k "." key_part(); ws() }
      return k
    }
    function emit(k) { if (k ~ /^(build|target|profile|env)\./ && !(k in seen)) { seen[k] = 1; print k } }
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
          if (substr(S, P, 1) == "=") { P++; emit(prefix "." k); value(prefix "." k) }
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
      if (substr(S, P, 1) == "[") { P++; if (substr(S, P, 1) == "[") P++; table = key_path(); return }
      k = key_path(); ws()
      if (substr(S, P, 1) != "=") return
      P++; if (table != "") k = table "." k
      emit(k); value(k)
    }
    { stmt = (stmt == "" ? $0 : stmt "\n" $0); if (complete(stmt)) { statement(); stmt = "" } }
    END { if (stmt != "") statement() }' "$1"
}

# The keys of a cargo config file that the runbook refuses, one per line:
# build.target, and an [env] entry named OXIBLAS_*, OPENBLAS_*, OMP_* or
# RAYON_* (cargo passes [env] to the test and bench processes and fills in
# a variable the environment does not set).
config_refusals() {
  config_keys "$1" | awk '
    $0 == "build.target" || index($0, "build.target.") == 1 { print; next }
    index($0, "env.") == 1 { n = substr($0, 5); sub(/\..*/, "", n); if (n ~ /^(OXIBLAS_|OPENBLAS_|OMP_|RAYON_)/) print }'
}

# One line per cargo config file that cargo reads for a command run in the
# checkout: the parent chain of the checkout (shown relative to it) and the
# cargo home. Fields: label, path, keys (tab-separated; keys comma-separated).
config_files() {
  local dir="$ROOT" up="" cargo_home f seen=" "
  while :; do
    for f in "$dir/.cargo/config.toml" "$dir/.cargo/config"; do
      if [ -f "$f" ]; then
        printf '<checkout>%s/.cargo/%s\t%s\t%s\n' "$up" "$(basename "$f")" "$f" "$(config_keys "$f" | paste -s -d, - | sed 's/,/, /g')"
        seen="$seen$(cd "$(dirname "$f")" && pwd -P)/$(basename "$f") "
      fi
    done
    [ "$dir" = / ] && break
    dir="$(dirname "$dir")"
    up="$up/.."
  done
  cargo_home="${CARGO_HOME:-${HOME:-}/.cargo}"
  for f in "$cargo_home/config.toml" "$cargo_home/config"; do
    if [ -f "$f" ]; then
      case "$seen" in
        *" $(cd "$(dirname "$f")" && pwd -P)/$(basename "$f") "*) continue ;;
      esac
      printf '<cargo home>/%s\t%s\t%s\n' "$(basename "$f")" "$f" "$(config_keys "$f" | paste -s -d, - | sed 's/,/, /g')"
    fi
  done
}

# --------------------------------------------------------------- the machine
has_flag() {
  case " $cpu_flags " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

online_cpus() {
  local n
  n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  case "$n" in ''|*[!0-9]*) n="$(nproc 2>/dev/null || echo 1)" ;; esac
  printf '%s\n' "$n"
}

# Distinct (core, socket) pairs: lscpu, else /proc/cpuinfo, else online CPUs.
physical_cores() {
  local n=""
  if command -v lscpu >/dev/null 2>&1; then
    n="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | grep -c '' || true)"
  fi
  if { [ -z "$n" ] || [ "$n" = 0 ]; } && [ -r /proc/cpuinfo ]; then
    n="$(awk -F: '/^physical id/ { p = $2 } /^core id/ { print p ":" $2 }' /proc/cpuinfo | sort -u | grep -c '' || true)"
  fi
  case "$n" in ''|0|*[!0-9]*) n="$(online_cpus)" ;; esac
  printf '%s\n' "$n"
}

# The CPUs this process may run on, one per line (Linux), else 0..online-1.
allowed_cpus() {
  local list=""
  if [ -r /proc/self/status ]; then
    list="$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)"
  fi
  if [ -z "$list" ]; then
    awk -v n="$(online_cpus)" 'BEGIN { for (i = 0; i < n; i++) print i }'
    return
  fi
  printf '%s\n' "$list" | tr ',' '\n' | awk -F- '
    NF == 1 { print $1 }
    NF == 2 { for (i = $1; i <= $2; i++) print i }'
}

# CPU 2 if this process may use it, else the lowest allowed CPU other than 0,
# else 0.
pin_cpu() {
  local cpus
  cpus="$(allowed_cpus)"
  if printf '%s\n' "$cpus" | grep -qx 2; then
    printf '2\n'
    return
  fi
  printf '%s\n' "$cpus" | awk '$1 != 0 { print; found = 1; exit } END { if (!found) print 0 }'
}

# 2, 4, 6, 8, the physical core count and the online CPU count, deduplicated,
# at most the online CPUs, ascending and comma-separated.
ab_thread_list() {
  local online phys t
  online="$(online_cpus)"
  phys="$(physical_cores)"
  for t in $AB_THREAD_CANDIDATES "$phys" "$online"; do
    if [ "$t" -le "$online" ]; then printf '%s\n' "$t"; fi
  done | sort -n -u | paste -s -d, -
}

# ---------------------------------------------------------------------- lock
lock_dir=""

# The start time of a process as ps prints it, in UTC (a run with another TZ
# must read the same text, or a live run's lock would look stale); empty
# when ps cannot say.
process_start() {
  TZ=UTC0 ps -p "$1" -o lstart= 2>/dev/null | awk 'NF { $1 = $1; print; exit }' || true
}

# Whether the run that wrote the lock $1 still runs: its process exists and
# (when ps tells) started at the time the lock records, so a reused process
# id is not mistaken for the run. Status 0 alive, 1 gone, 2 no owner yet.
lock_owner_alive() {
  local pid start now
  pid="$(sed -n 1p "$1/pid" 2>/dev/null || true)"
  start="$(sed -n 2p "$1/pid" 2>/dev/null || true)"
  case "$pid" in ''|*[!0-9]*) return 2 ;; esac
  kill -0 "$pid" 2>/dev/null || ps -p "$pid" >/dev/null 2>&1 || return 1
  if [ -n "$start" ]; then
    now="$(process_start "$pid")"
    if [ -n "$now" ] && [ "$now" != "$start" ]; then return 1; fi
  fi
  return 0
}

# Takes the lock directory $1 for this process; a lock of a live run means
# exit 2, the lock of a run that is gone is taken over. Breakers serialise
# through $1.breaker, so two new runs cannot both remove one stale lock.
take_lock() {
  local dir="$1" tries=0 empty_checks=0 state owner
  while ! mkdir "$dir" 2>/dev/null; do
    [ -d "$dir" ] || die 2 "cannot create the lock $dir (is target/ writable?); nothing was changed"
    tries=$((tries + 1)); [ "$tries" -le 30 ] || die 2 "cannot take the lock $dir after $tries attempts; nothing was changed"
    state=0; lock_owner_alive "$dir" || state=$?
    owner="$(sed -n 1p "$dir/pid" 2>/dev/null || true) (started $(sed -n 2p "$dir/pid" 2>/dev/null || true) UTC)"
    if [ "$state" = 0 ]; then
      die 2 "another run of this script uses this checkout: process $owner holds the lock $dir. Wait until it ends (its summary is target/x86-runbook/summary.txt) or stop it (kill <its process id>); nothing was changed"
    fi
    if [ "$state" = 2 ] && [ "$empty_checks" -lt 3 ]; then
      # A run that has just made the directory writes its process id next.
      empty_checks=$((empty_checks + 1)); sleep 1; continue
    fi
    if mkdir "$dir.breaker" 2>/dev/null; then
      state=0; lock_owner_alive "$dir" || state=$?
      if [ "$state" != 0 ] && [ -d "$dir" ]; then
        note "taking over the lock of a run that no longer runs (process $owner)"; rm -rf "$dir"
      fi
    else
      sleep 1
    fi
    rmdir "$dir.breaker" 2>/dev/null || true
  done
  if ! { printf '%s\n%s\n' "$$" "$(process_start "$$")" > "$dir/pid.tmp" && mv -f "$dir/pid.tmp" "$dir/pid"; }; then
    rm -rf "$dir"; die 2 "cannot write the lock $dir/pid; nothing was changed"
  fi
  lock_dir="$dir"
}

# Removes the lock if this process holds it.
release_lock() {
  if [ -n "$lock_dir" ] && [ "$(sed -n 1p "$lock_dir/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$lock_dir"
  fi
  lock_dir=""
}

# ------------------------------------------------------------ path scrubbing
# Every form of the checkout and home paths (FORM_FROM / FORM_TO) is replaced
# in archived copies; the plain forms (RAW_FROM / RAW_TO) are also replaced
# in the command lines of the logs before quoting. A form is replaced where
# it stands as a path: not preceded and not followed by a letter, digit,
# `.`, `_` or `-` (so a checkout at /src does not touch "oxiblas-blas/src").
FORM_FROM=(); FORM_TO=(); RAW_FROM=(); RAW_TO=(); PATH_AWK_CMD=()
# shellcheck disable=SC2016 # an awk program
readonly PATH_AWK='
  function word_char(c) {
    return c != "" && index("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-", c) > 0
  }
  function swap(line, from, to,   out, i, n, before) {
    n = length(from); out = ""
    while (n > 0 && (i = index(line, from)) > 0) {
      before = (i > 1) ? substr(line, i - 1, 1) : substr(out, length(out), 1)
      if (word_char(before) || word_char(substr(line, i + n, 1))) {
        out = out substr(line, 1, i); line = substr(line, i + 1)
      } else {
        out = out substr(line, 1, i - 1) to; line = substr(line, i + n); hit = 1
      }
    }
    return out line
  }
  BEGIN {
    n = ENVIRON["RB_N"] + 0
    for (i = 0; i < n; i++) { f[i] = ENVIRON["RB_F" i]; t[i] = ENVIRON["RB_T" i] }
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (length(f[j]) > length(f[i])) {
      x = f[i]; f[i] = f[j]; f[j] = x; x = t[i]; t[i] = t[j]; t[j] = x
    }
  }
  {
    hit = 0; line = $0
    for (i = 0; i < n; i++) line = swap(line, f[i], t[i])
    if (mode == "scrub") print line
    else if (hit && !(FILENAME in shown)) { shown[FILENAME] = 1; print FILENAME }
  }'

add_form() {
  local i=0
  [ -n "$1" ] || return 0
  for ((i = 0; i < ${#FORM_FROM[@]}; i++)); do [ "${FORM_FROM[$i]}" != "$1" ] || return 0; done
  FORM_FROM+=("$1"); FORM_TO+=("$2")
}

# The forms of path $1 (shown as $2): as it is, printf %q (and the inside
# of a $'...' form), and Debug-escaped (\ and " escaped).
add_path_forms() {
  local path="$1" token="$2" quoted inner
  case "$path" in ''|/) return 0 ;; esac
  RAW_FROM+=("$path"); RAW_TO+=("$token")
  add_form "$path" "$token"
  quoted="$(printf '%q' "$path")"
  add_form "$quoted" "$token"
  case "$quoted" in
    "\$'"*"'") inner="${quoted#"\$'"}"; add_form "${inner%"'"}" "$token" ;;
  esac
  add_form "$(printf '%s' "$path" | sed 's/[\\"]/\\&/g')" "$token"
}

# Collects the forms of the checkout ($1, and as given $2) and of the home
# directory ($3, and its physical path).
#   set_path_forms <root> <root as given> <home>
set_path_forms() {
  local i=0 home="$3"
  FORM_FROM=(); FORM_TO=(); RAW_FROM=(); RAW_TO=()
  add_path_forms "$1" "<checkout>"
  add_path_forms "$2" "<checkout>"
  if [ -n "$home" ]; then
    add_path_forms "$home" "<home>"
    add_path_forms "$(cd "$home" 2>/dev/null && pwd -P || true)" "<home>"
  fi
  PATH_AWK_CMD=(env "RB_N=${#FORM_FROM[@]}")
  for ((i = 0; i < ${#FORM_FROM[@]}; i++)); do PATH_AWK_CMD+=("RB_F$i=${FORM_FROM[$i]}" "RB_T$i=${FORM_TO[$i]}"); done
  PATH_AWK_CMD+=(awk)
}

# Copies a text file with every form of the checkout and home paths
# replaced by <checkout> and <home>.
scrub_copy() {
  "${PATH_AWK_CMD[@]}" -v mode=scrub "$PATH_AWK" "$1" > "$2"
}

# Prints each file under directory $1 that still contains a form of the
# checkout or home path; status non-zero when the check itself failed.
find_leaks() {
  find "$1" -type f -exec "${PATH_AWK_CMD[@]}" -v mode=leak "$PATH_AWK" {} +
}

# Removes the files under $1 that still contain a form of the checkout or
# home path and prints their names relative to $1; status 1 when the check
# itself failed (nothing is removed then).
drop_leaking_copies() {
  local dir="$1" list f
  list="$(find_leaks "$dir")" || return 1
  [ -n "$list" ] || return 0
  printf '%s\n' "$list" | sort | while IFS= read -r f; do
    [ -n "$f" ] || continue
    rm -f "$f"
    printf '%s\n' "${f#"$dir"/}"
  done
}

# A word for display: quoted with printf %q when it needs quoting.
quote_word() {
  case "$1" in ''|*[!A-Za-z0-9_.,:=/+@%-]*) printf '%q' "$1" ;; *) printf '%s' "$1" ;; esac
}

# An argument for display: a leading checkout or home path (of the whole
# argument or of the value of NAME=VALUE) is written <checkout> or <home>
# first, and only the rest is quoted.
show_arg() {
  local arg="$1" name="" value i=0 best=-1 best_len=0 form rest
  case "$arg" in
    [A-Za-z_]*=*)
      name="${arg%%=*}="
      case "$name" in *[!A-Za-z0-9_=]*) name="" ;; esac ;;
  esac
  value="${arg#"$name"}"
  for ((i = 0; i < ${#RAW_FROM[@]}; i++)); do
    form="${RAW_FROM[$i]}"
    case "$value" in
      "$form"|"$form"/*) if [ "${#form}" -gt "$best_len" ]; then best="$i"; best_len="${#form}"; fi ;;
    esac
  done
  if [ "$best" -lt 0 ]; then quote_word "$arg"; return 0; fi
  rest="${value:$best_len}"
  printf '%s%s' "$name" "${RAW_TO[$best]}"
  if [ -n "$rest" ]; then quote_word "$rest"; fi
}

# A command for display, without the checkout and home paths.
show_cmd() {
  local out="" arg
  for arg in "$@"; do out="$out $(show_arg "$arg")"; done
  printf '%s\n' "${out# }"
}

# ------------------------------------------------------------------- helpers
child_pid=""; tee_pid=""; current_unit=""; current_agg=""; current_start=0
finalized=0; out_ready=0; interrupted=""; any_fail=0; final_rc=0; missing_parts=""
unit_rc=0; unit_secs=0; unit_verdict=PASS; bench_no_baseline=0
agg_start=0; agg_groups=0
# shellcheck disable=SC2034 # the totals step 5 (x86_gemm_runbook_ab.sh) keeps per aggregate unit
{ agg_failed=0; agg_failed_names=""; agg_skipped=0; agg_counts="0 0 0 0 0 0 0"; agg_why=""; }

# Replaces tabs and line breaks (a detail is one field of one line).
one_line() {
  printf '%s' "$1" | tr '\t\n\r' '   '
}

# Appends one result line: id, verdict, exit code, seconds, details.
record() {
  local id="$1" verdict="$2" rc="$3" secs="$4" detail
  detail="$(one_line "$5")"
  if [ "$dry_run" = 1 ]; then
    if [ "$verdict" = SKIPPED ]; then printf '\n## %s: skipped - %s\n' "$id" "$detail"; fi
    return 0
  fi
  case "$verdict" in FAIL) any_fail=1 ;; esac
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$verdict" "$rc" "$secs" "$detail" >> "$UNITS"
  printf '== %s: %s (exit %s, %ss)%s\n' "$id" "$verdict" "$rc" "$secs" "${detail:+ - $detail}" >&2
}

# Adds one line to the list of parts a full run needs but this one lacks.
add_missing() {
  missing_parts="$missing_parts$1
"
}

# "runs passed failed ignored" summed over the `test result:` lines of a log.
test_counts() {
  awk '/^test result: / {
      n++
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^passed;?$/) p += $(i - 1)
        if ($i ~ /^failed;?$/) f += $(i - 1)
        if ($i ~ /^ignored;?$/) g += $(i - 1)
      }
    } END { printf "%d %d %d %d\n", n, p, f, g }' "$1"
}

# Sets runs, passed, failed and ignored from the log $1 (0 when it cannot be
# read or has no `test result:` line).
read_counts() {
  read -r runs passed failed ignored <<EOF || true
$(test_counts "$1" 2>/dev/null || true)
EOF
  runs="${runs:-0}"; passed="${passed:-0}"; failed="${failed:-0}"; ignored="${ignored:-0}"
}

# Occurrences of a fixed text in a file, anywhere in a line.
count_text() {
  { grep -o -F -e "$1" "$2" 2>/dev/null || true; } | grep -c '' || true
}

# What went wrong in a failed cargo test log: the failed tests (the first
# three and a count), the test binaries ended by a signal (a count and the
# first signal) and the first "wrong result, not timed" line of the A/B
# harness. The failed tests come from the `test <name> ... FAILED` lines and
# from libtest's closing `failures:` list (indented names up to a blank
# line): with --nocapture the word FAILED is printed on a line of its own,
# after the test's output, so only the list names those tests.
failure_details() {
  local log="$1" names count shown signals first_signal wrong out=""
  names="$(awk '
      /^test .* \.\.\. FAILED$/ { name = $0; sub(/^test /, "", name); sub(/ \.\.\. FAILED$/, "", name); print name }
      /^failures:$/ { in_list = 1; next }
      in_list && /^    [^ ]/ { name = $0; sub(/^ +/, "", name); print name; next }
      { in_list = 0 }' "$log" | sort -u)"
  count="$(printf '%s' "$names" | grep -c '' || true)"
  if [ "$count" != 0 ]; then
    shown="$(printf '%s\n' "$names" | awk 'NR <= 3' | paste -s -d, - | sed 's/,/, /g')"
    out="$out; $count failed tests: $shown$([ "$count" -gt 3 ] && echo ', ...' || true)"
  fi
  signals="$(grep -c '(signal: ' "$log" || true)"
  if [ "$signals" != 0 ]; then
    first_signal="$(sed -n 's/.*(signal: [0-9]*, \(SIG[A-Z0-9]*\).*/\1/p' "$log" | awk 'NR == 1')"
    out="$out; $signals test binaries ended by a signal (first: ${first_signal:-unknown})"
  fi
  wrong="$(grep -F -m 1 -e "$WRONG_RESULT_TEXT" "$log" | cut -c 1-300 || true)"
  if [ -n "$wrong" ]; then
    out="$out; first wrong result: \"$(one_line "$wrong")\""
  fi
  printf '%s' "$out"
}

# Whether any process of the process group $1 is still alive.
group_alive() {
  kill -0 -- "-$1" 2>/dev/null
}

# Stops the running command and every process of its group.
stop_child() {
  local tries=0
  if [ -n "$child_pid" ]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    while group_alive "$child_pid" && [ "$tries" -lt 50 ]; do
      sleep 0.2
      tries=$((tries + 1))
    done
    kill -KILL -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=""
  fi
  if [ -n "$tee_pid" ]; then
    wait "$tee_pid" 2>/dev/null || true
    tee_pid=""
  fi
}

# Runs one command as a unit: its own log, its own process group (job
# control is on, so a signal can stop the command with all its children),
# output to the terminal and the log. Sets unit_rc and unit_secs.
#   run_cmd <id> <title> <command...>
run_cmd() {
  local id="$1" title="$2"
  shift 2
  local log="$LOG_DIR/$id.log" fifo="$OUT/.fifo"
  unit_rc=0
  unit_secs=0
  if [ "$dry_run" = 1 ]; then
    printf '\n## %s: %s\n(cd <checkout> &&\n  %s)\n' "$id" "$title" "$(show_cmd "$@")"
    return 0
  fi
  {
    printf '== step   : %s (%s)\n' "$id" "$title"
    printf '== started: %s UTC\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
    printf '== command: %s\n' "$(show_cmd "$@")"
    printf '== env    : OXIBLAS_REQUIRE_AVX2=%s CARGO_TERM_COLOR=never LC_ALL=C; run in <checkout>\n' "$require_avx2_text"
  } > "$log"
  cat "$log"
  current_unit="$id"
  current_start="$SECONDS"
  rm -f "$fifo"
  mkfifo "$fifo"
  tee -a "$log" < "$fifo" &
  tee_pid=$!
  (cd "$ROOT" && exec "$@") > "$fifo" 2>&1 < /dev/null &
  child_pid=$!
  wait "$child_pid" || unit_rc=$?
  child_pid=""
  wait "$tee_pid" || true
  tee_pid=""
  rm -f "$fifo"
  unit_secs=$((SECONDS - current_start))
  current_unit=""
  printf '== exit   : %s after %ss\n' "$unit_rc" "$unit_secs" >> "$log"
}

# Verdict of a cargo test unit from its log; prints nothing, records the
# unit and sets unit_verdict. A problem found elsewhere (step 5's checks of
# ab.tsv) fails the unit too.
#   finish_tests <id> [<reason when no test ran>] [<extra detail>] [<problem>]
finish_tests() {
  local id="$1" none_reason="${2:-no test ran (check the filter)}" extra="${3:-}" problem="${4:-}"
  local log="$LOG_DIR/$id.log" runs passed failed ignored detail
  unit_verdict=PASS
  [ "$dry_run" = 1 ] && return 0
  read_counts "$log"
  detail="tests $passed passed, $failed failed, $ignored ignored ($runs test binaries)"
  if [ "$unit_rc" != 0 ]; then
    unit_verdict=FAIL
    detail="$detail; cargo exited with $unit_rc$(failure_details "$log")"
  elif [ "$runs" = 0 ]; then
    unit_verdict=FAIL
    detail="$detail; the log has no \`test result:\` line (no test binary reported)"
  elif [ "$passed" = 0 ]; then
    unit_verdict=FAIL
    detail="$detail; $none_reason"
  fi
  if [ -n "$problem" ]; then
    unit_verdict=FAIL
    detail="$detail; $problem"
  fi
  record "$id" "$unit_verdict" "$unit_rc" "$unit_secs" "$detail$extra"
}

# The cargo arguments of a feature set ("none" = default features).
feature_args() {
  if [ "$1" != none ]; then printf -- '--features\n%s\n' "$1"; fi
}

# ------------------------------------------------------------------- steps
# The kernel command line ($1, default /proc/cmdline) with the values of keys
# that name devices, disks, addresses or the host replaced by <redacted>, and
# every UUID (a machine or VM id) by <uuid>.
kernel_cmdline() {
  if [ -r "${1:-/proc/cmdline}" ]; then
    awk 'BEGIN { h = "[0-9A-Fa-f]"; h = h h h h; u = h h "-" h "-" h "-" h "-" h h h } { for (i = 1; i <= NF; i++) if ($i ~ /^(root|resume|ip|nfsroot|cryptdevice|BOOTIF|[^=]*hostname|rd\.luks[^=]*|rd\.lvm[^=]*|luks[^=]*)=/) sub(/=.*/, "=<redacted>", $i); gsub(u, "<uuid>"); print }' "${1:-/proc/cmdline}"
  else
    printf 'unknown (no /proc/cmdline)\n'
  fi
}

step_machine() {
  local id="1-machine"
  if [ "$dry_run" = 1 ]; then
    printf '\n## %s: machine.txt\n%s\n%s\n%s\n%s\n%s\n' "$id" \
      '  lscpu; uname -srm; PRETTY_NAME of /etc/os-release; getconf GNU_LIBC_VERSION or ldd --version;' \
      '  /proc/cmdline (device, address and host values and UUIDs redacted); (cd <checkout> && rustc -vV; cargo -V; rustup show active-toolchain);' \
      '  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; getconf _NPROCESSORS_ONLN;' \
      '  lscpu -p=CORE,SOCKET; grep MemTotal /proc/meminfo; free -h; dpkg-query -W libopenblas*;' \
      '  the names of the inherited CARGO_*/RUST* variables; the cargo config files and their key names'
    return 0
  fi
  local start="$SECONDS" file="$OUT/machine.txt" rc=0 governors label keys distro libc
  distro="$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '"' | awk 'NR == 1' || true)"
  libc="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
  if [ -z "$libc" ] && command -v ldd >/dev/null 2>&1; then libc="$(ldd --version 2>&1 | awk 'NR == 1' || true)"; fi
  {
    printf 'oxiblas x86 GEMM runbook: machine\n'
    printf 'date (UTC)      : %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
    printf 'system          : %s %s %s\n' "$(uname -s)" "$(uname -r)" "$(uname -m)"
    printf 'distribution    : %s\n' "${distro:-unknown (no PRETTY_NAME in /etc/os-release)}"
    printf 'C library       : %s\n' "${libc:-unknown}"
    printf 'kernel cmdline  : %s\n' "$(kernel_cmdline)"
    printf 'online CPUs     : %s\n' "$N_ONLINE"
    printf 'physical cores  : %s\n' "$N_PHYSICAL"
    printf 'allowed CPUs    : %s\n' "$(allowed_cpus | paste -s -d, -)"
    printf 'A/B pin CPU     : %s (taskset %s)\n' "$PIN_CPU" "$([ "$have_taskset" = 1 ] && echo found || echo MISSING)"
    printf 'A/B threads     : %s\n' "$AB_THREADS"
    printf 'avx2 / fma / avx512f in /proc/cpuinfo: %s / %s / %s\n' \
      "$(has_flag avx2 && echo yes || echo no)" "$(has_flag fma && echo yes || echo no)" \
      "$(has_flag avx512f && echo yes || echo no)"
    governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | awk '{ printf "%s%s x%s", sep, $2, $1; sep = ", " }' || true)"
    printf 'freq governor   : %s\n' "${governors:-unknown (no cpufreq in /sys)}"
    printf 'boost           : %s\n' "$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null | sed 's/^/intel_pstate no_turbo=/' || echo unknown)"
    printf 'THP             : %s\n' "$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unknown)"
    printf 'memory          : %s\n' "$(awk '/^MemTotal:|^MemAvailable:/ { printf "%s%s %s %s", sep, $1, $2, $3; sep = ", " }' /proc/meminfo 2>/dev/null || echo unknown)"
    printf 'OpenBLAS library: %s (%s)\n' "$OPENBLAS_PATH" "$([ -e "$OPENBLAS_PATH" ] && echo present || echo absent)"
    # shellcheck disable=SC2016 # ${...} are dpkg-query fields
    printf 'OpenBLAS pkgs   : %s\n' "$(dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' 'libopenblas*' 2>/dev/null \
      | awk '$1 == "ii" { printf "%s%s %s", sep, $2, $3; sep = "; " }' || echo 'unknown (no dpkg)')"
    printf 'OpenBLAS/OpenMP env of the caller (unset; step 6 sets OPENBLAS_NUM_THREADS=1): %s\n' "${inherited_openblas:-none}"
    printf 'removed env     : %s\n' "${removed_env:-none}"
    printf 'OXIBLAS_REQUIRE_AVX2: %s\n' "$require_avx2_text"
    printf 'CARGO_BUILD_JOBS: %s\n' "${CARGO_BUILD_JOBS:-unset}"
    printf 'locale          : LC_ALL=C for the script and every command (the caller had LANG=%s LC_ALL=%s)\n' "${LANG:-unset}" "${caller_lc_all:-unset}"
    printf 'inherited CARGO_* and RUST* variables (names): %s\n' "${inherited_cargo_names:-none}"
    printf 'inherited CARGO_PROFILE_* values: %s\n' "${inherited_profile_values:-none}"
    printf 'rustflags       : overridden by this script for every cargo command: RUSTFLAGS="" (default policy; config rustflags and CARGO_TARGET_*_RUSTFLAGS do not apply), RUSTFLAGS="%s" (candidate), RUSTFLAGS="%s" (3-candall)\n' \
      "$CANDIDATE_RUSTFLAGS" "$CANDALL_RUSTFLAGS"
    printf 'cargo config files (parent chain of the checkout, cargo home; key names only):\n'
    if [ -z "$CONFIG_FILES" ]; then
      printf '  none\n'
    else
      while IFS="$TAB" read -r label _ keys; do
        [ -n "$label" ] || continue
        printf '  %s: %s\n' "$label" "${keys:-no build, target, profile or env key}"
      done <<EOF
$CONFIG_FILES
EOF
    fi
    printf '\n== rustc -vV, cargo -V, rustup show active-toolchain (in the checkout)\n'
    (
      cd "$ROOT" || exit 1
      rustc -vV 2>&1 || printf 'rustc failed\n'
      cargo -V 2>&1 || printf 'cargo failed\n'
      if command -v rustup >/dev/null 2>&1; then
        # A path in rustup's reason (an override file) is replaced by <path>.
        rustup show active-toolchain 2>&1 | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /\//) $i = "<path>"; print }' || true
      else
        printf 'rustup not found\n'
      fi
    )
    printf '\n== lscpu\n'
    if command -v lscpu >/dev/null 2>&1; then lscpu 2>&1; else printf 'lscpu not found (util-linux)\n'; fi
    printf '\n== free -h\n'
    free -h 2>&1 || printf 'free not found\n'
  } > "$file" 2>&1 || rc=$?
  cp "$file" "$LOG_DIR/$id.log"
  record "$id" "$([ "$rc" = 0 ] && echo PASS || echo FAIL)" "$rc" "$((SECONDS - start))" "machine.txt"
}

# Steps 2 and 3: one policy's test matrix.
#   step_tests <step> <legacy|cand>
step_tests() {
  local step="$1" policy="$2" fs id filters=() env_cmd args sets="$FEATURE_SETS" a
  if [ "$policy" = cand ]; then
    env_cmd=("${CAND_ENV[@]}")
  else
    env_cmd=("${LEGACY_ENV[@]}")
  fi
  if [ "$quick" = 1 ]; then
    sets="parallel"
    for fs in $QUICK_FILTERS; do filters+=("$fs"); done
  fi
  for fs in $sets; do
    id="$step-$policy-$fs"
    args=(cargo test -p oxiblas-blas --release)
    while IFS= read -r a; do [ -n "$a" ] && args+=("$a"); done <<EOF
$(feature_args "$fs")
EOF
    args+=(--no-fail-fast)
    if [ "$fs" = parallel ] || [ "${#filters[@]}" -gt 0 ]; then
      args+=(--)
      [ "$fs" = parallel ] && args+=(--show-output)
      [ "${#filters[@]}" -gt 0 ] && args+=("${filters[@]}")
    fi
    run_cmd "$id" "oxiblas-blas tests, $policy policy, features $fs" "${env_cmd[@]}" "${args[@]}"
    finish_tests "$id"
  done
  if [ "$policy" = cand ]; then
    id="$step-candall-parallel"
    args=(cargo test -p oxiblas-blas --release --features parallel --no-fail-fast -- --show-output)
    [ "${#filters[@]}" -gt 0 ] && args+=("${filters[@]}")
    run_cmd "$id" "oxiblas-blas tests, candidate policy with B-block caps in both drivers, features parallel" \
      "${CANDALL_ENV[@]}" "${args[@]}"
    finish_tests "$id"

    id="$step-$policy-workspace"
    if [ "$quick" = 1 ]; then
      record "$id" SKIPPED - 0 "quick mode runs no workspace tests"
    else
      run_cmd "$id" "workspace tests (without the benchmarks), candidate policy" \
        "${env_cmd[@]}" cargo test --workspace --exclude oxiblas-benchmarks --release --no-fail-fast
      finish_tests "$id"
    fi
  fi
}

# Step 4: proof in the logs of the parallel runs that the AVX2 code ran.
step_avx2() {
  local policy log id runs passed failed ignored skips requires confirmed avx2_names verdict detail rc step
  for policy in legacy cand candall; do
    id="4-avx2-$policy"
    case "$policy" in
      legacy) log="$LOG_DIR/2-legacy-parallel.log"; step=2 ;;
      cand) log="$LOG_DIR/3-cand-parallel.log"; step=3 ;;
      candall) log="$LOG_DIR/3-candall-parallel.log"; step=3 ;;
    esac
    if [ "$dry_run" = 1 ]; then
      printf '\n## %s: read %s\n' "$id" "logs/${log##*/}"
      printf "  count the lines '%s' (at least 1), '%s' and '%s' (none),\n  and the passed tests named *avx2* (information)\n" \
        "$CONFIRM_TEXT" "$SKIP_TEXT" "$REQUIRE_TEXT"
      continue
    fi
    if [ ! -f "$log" ]; then
      record "$id" SKIPPED - 0 "no log of the parallel test run of this run (run step $step with step 4)"
      continue
    fi
    read_counts "$log"
    rc="$(sed -n 's/^== exit   : \([0-9]*\) .*/\1/p' "$log" | tail -n 1)"
    confirmed="$(count_text "$CONFIRM_TEXT" "$log")"
    skips="$(count_text "$SKIP_TEXT" "$log")"
    requires="$(count_text "$REQUIRE_TEXT" "$log")"
    avx2_names="$(grep -c -E '^test [^ ]*avx2[^ ]* \.\.\. ok$' "$log" || true)"
    detail="$confirmed gated tests confirmed the AVX2+FMA tier, $skips SKIPPED lines, $requires REQUIRE_AVX2 failures (information: $avx2_names passed tests named *avx2*; run: $passed passed, $failed failed)"
    verdict=PASS
    if [ "${rc:-1}" != 0 ] || [ "$runs" = 0 ] || [ "$failed" != 0 ] || [ "$passed" = 0 ]; then
      verdict=FAIL; detail="$detail; the test run itself did not pass"
    elif [ "$confirmed" = 0 ] || [ "$skips" != 0 ] || [ "$requires" != 0 ]; then
      if [ "$allow_no_avx2" = 1 ]; then
        verdict=WAIVED; detail="$detail; allowed by --allow-no-avx2"
      else
        verdict=FAIL; detail="$detail; no proof that the AVX2+FMA code ran"
      fi
    fi
    record "$id" "$verdict" "${rc:--}" 0 "$detail"
  done
}

# The tiny-call step, information only: gemm_tiny_call_check.sh with its own
# defaults (base release, rounds, limit, judged set; --quick with --quick)
# and --out in this run's directory. Its FAIL or INCOMPLETE is recorded as
# INFO, its exit 2 (no git checkout, no base release, ...) as SKIPPED.
step_tiny() {
  local cmd=(env "RUSTFLAGS=" bash "$TINY_TOOL" --out "$TINY_OUT") why
  [ "$quick" = 1 ] && cmd+=(--quick)
  if [ ! -f "$TINY_TOOL" ]; then record tiny-call SKIPPED - 0 "information only: ${TINY_TOOL#"$ROOT"/} is missing"; return 0; fi
  run_cmd tiny-call "per-call cost of tiny GEMM products against a base release (information only)" "${cmd[@]}"
  [ "$dry_run" = 1 ] && return 0
  why="no summary.txt"
  # Its verdict line, then its first line (settings) without the tree path.
  [ ! -f "$TINY_OUT/summary.txt" ] || why="$(awk '/^verdict: / { v = $0 } NR == 1 { s = $0; sub(/^# /, "", s); sub(/, tree [^,]*,/, ",", s) }
    END { print (v == "" ? "no verdict line in its summary.txt" : v) "; " s }' "$TINY_OUT/summary.txt")"
  case "$unit_rc" in
    0|1|3) record tiny-call INFO "$unit_rc" "$unit_secs" "information only, never a failure: $why" ;;
    2) why="$(awk '/: error: / { sub(/.*: error: /, ""); w = $0 } END { print w }' "$LOG_DIR/tiny-call.log")"
      record tiny-call SKIPPED 2 "$unit_secs" "information only: the tool could not run (${why:-exit 2}); without its base release: git fetch --tags, then --only tiny" ;;
    *) record tiny-call INFO "$unit_rc" "$unit_secs" "information only: the tool ended with exit $unit_rc (see logs/tiny-call.log)" ;;
  esac
}

# Step 6: the OpenBLAS comparison benches, legacy baseline then candidate.
step_bench() {
  local bench=(cargo bench -p oxiblas-benchmarks --bench comparison --features compare-openblas)
  local common=("OPENBLAS_LIB=$OPENBLAS_PATH" "CRITERION_HOME=$CRITERION_DIR" OPENBLAS_NUM_THREADS=1)
  local crit_legacy=(-- --save-baseline legacy) crit_cand=(-- --baseline legacy) extra=()
  bench_no_baseline=0
  if [ "$quick" = 1 ]; then extra+=(--test); fi
  if [ -n "$bench_filter" ]; then extra+=("$bench_filter"); fi
  if [ ! -e "$OPENBLAS_PATH" ]; then
    if [ "$dry_run" = 1 ]; then
      note "note: no OpenBLAS library at $OPENBLAS_PATH; a real run would record step 6 as skipped"
    else
      record 6-bench SKIPPED - 0 "no OpenBLAS library at $OPENBLAS_PATH (Debian/Ubuntu: apt install libopenblas0; or set OPENBLAS_LIB to the library)"
      add_missing "the OpenBLAS comparison (step 6) did not run, because there is no OpenBLAS library at $OPENBLAS_PATH: install it (Debian/Ubuntu: sudo apt install libopenblas0; Fedora: sudo dnf install openblas, then set OPENBLAS_LIB to its libopenblas.so) and run again"
      return 0
    fi
  fi

  run_cmd 6-bench-legacy "comparison benches, legacy policy, saved as baseline legacy" \
    "${LEGACY_ENV[@]}" "${common[@]}" "${bench[@]}" "${crit_legacy[@]}" ${extra[@]+"${extra[@]}"}
  finish_bench 6-bench-legacy
  if [ "$dry_run" = 0 ] && [ "$unit_rc" != 0 ]; then
    # Without a complete baseline `--baseline legacy` would stop at the first
    # missing benchmark; save the candidate's own baseline instead.
    crit_cand=(-- --save-baseline candidate)
    bench_no_baseline=1
  fi
  run_cmd 6-bench-cand "comparison benches, candidate policy, compared with baseline legacy" \
    "${CAND_ENV[@]}" "${common[@]}" "${bench[@]}" "${crit_cand[@]}" ${extra[@]+"${extra[@]}"}
  finish_bench 6-bench-cand
}

finish_bench() {
  local id="$1" log="$LOG_DIR/$1.log" timed tested panics changes pauses verdict detail
  [ "$dry_run" = 1 ] && return 0
  # A `time:` line that follows a `change:` line is the change against the
  # baseline, not a measurement.
  timed="$(awk '/time: +\[/ && !after_change { n++ } { after_change = ($0 ~ /change:[[:space:]]*$/) } END { print n + 0 }' "$log")"
  tested="$(grep -c -x 'Success' "$log" || true)"
  panics="$(grep -c 'panicked at' "$log" || true)"
  # Criterion's change verdicts, attributed to the library of the benchmark
  # id seen last (ids are <group>/<oxiblas|openblas>/<case>).
  changes="$(awk '
    match($0, /[A-Za-z0-9_]+\/(oxiblas|openblas)\//) {
      id = substr($0, RSTART, RLENGTH); sub(/^[A-Za-z0-9_]+\//, "", id); sub(/\/$/, "", id); lib = id
    }
    /Performance has improved/ { f[lib]++ }
    /Performance has regressed/ { s[lib]++ }
    /No change in performance detected|Change within noise threshold/ { u[lib]++ }
    END {
      printf "oxiblas %d faster, %d slower, %d unchanged; openblas (noise floor) %d faster, %d slower, %d unchanged",
        f["oxiblas"], s["oxiblas"], u["oxiblas"], f["openblas"], s["openblas"], u["openblas"]
    }' "$log")"
  pauses="$(sed -n 's/.*measured at \([0-9.]*\) cores (after \([0-9]*\) pauses.*/\2 pauses, \1 cores/p' "$log" | awk 'NR == 1')"
  detail="$timed benchmarks timed, $tested tested (--test), $panics panics; OpenBLAS one-thread check: ${pauses:-no reading in the log}"
  if [ "$id" = 6-bench-cand ]; then
    if [ "$bench_no_baseline" = 1 ]; then
      detail="$detail; no baseline comparison (the legacy run failed)"
    elif [ "$quick" = 1 ]; then
      detail="$detail; no baseline comparison (test mode)"
    else
      detail="$detail; against the baseline (criterion's p < 0.05 and noise threshold): $changes"
    fi
  fi
  verdict=PASS
  if [ "$unit_rc" != 0 ]; then
    verdict=FAIL; detail="$detail; cargo exited with $unit_rc"
  elif [ $((timed + tested)) = 0 ]; then
    verdict=FAIL; detail="$detail; no benchmark ran (check --bench-filter)"
  fi
  record "$id" "$verdict" "$unit_rc" "$unit_secs" "$detail"
}

# ------------------------------------------------------- summary and archive
git_info() {
  git_commit="unknown (not a git checkout or git missing)"
  git_dirty="unknown"
  # Git is only read: `status` would otherwise refresh the stat data cached
  # in the index and rewrite the index file (a write outside
  # target/x86-runbook/); --no-optional-locks and GIT_OPTIONAL_LOCKS=0
  # forbid that. rev-parse and log never write.
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
}

# Chooses the tar options that write user and group 0 without names (GNU tar
# or bsdtar), tried on the directory $1; sets tar_opts and tar_note.
choose_tar_opts() {
  local dir="$1" version flavor
  version="$(tar --version 2>/dev/null | awk 'NR == 1' || true)"
  case "$version" in
    *"GNU tar"*) flavor="GNU tar"; tar_opts=(--owner=0 --group=0 --numeric-owner) ;;
    *bsdtar*) flavor="bsdtar"; tar_opts=(--uid 0 --gid 0 --uname "" --gname "") ;;
    *) flavor="unknown tar"; tar_opts=() ;;
  esac
  if [ "${#tar_opts[@]}" -gt 0 ] && (cd "$dir" && tar "${tar_opts[@]}" -cf - .) > /dev/null 2>&1; then
    tar_note="user and group 0, no names ($flavor $(show_cmd "${tar_opts[@]}"))"
  else
    tar_opts=()
    tar_note="WARNING: $flavor took no option to leave out the user and group names; the archive's tar headers carry them"
  fi
}

finalize() {
  local status_line="$1" overall f rel left_out="" check_ok=1 summary_left=0
  [ "$finalized" = 1 ] && return 0
  finalized=1
  # From here on a signal cannot leave a partial archive or a summary without
  # its status: INT, TERM and HUP are ignored (tar and awk inherit that).
  trap '' INT TERM HUP
  [ "$dry_run" = 1 ] && return 0
  # Only the directory this run made (never an earlier run's) is finalized.
  [ "$out_ready" = 1 ] && [ -d "$OUT" ] || return 0
  [ -f "$UNITS" ] || : > "$UNITS"
  git_info

  rm -rf "$OUT/.archive" && mkdir -p "$OUT/.archive/logs"
  tar_opts=(); choose_tar_opts "$OUT/.archive"
  for f in machine.txt ab.tsv; do
    if [ -f "$OUT/$f" ]; then scrub_copy "$OUT/$f" "$OUT/.archive/$f"; fi
  done
  if [ -f "$ROOT/Cargo.lock" ]; then scrub_copy "$ROOT/Cargo.lock" "$OUT/.archive/Cargo.lock"; fi
  for f in "$LOG_DIR"/*.log; do
    if [ -f "$f" ]; then scrub_copy "$f" "$OUT/.archive/logs/$(basename "$f")"; fi
  done
  # The tiny-call tool's summary and the logs it names (up to 1 MiB each).
  if [ -f "$TINY_OUT/summary.txt" ]; then
    mkdir -p "$OUT/.archive/tiny-call/logs" && scrub_copy "$TINY_OUT/summary.txt" "$OUT/.archive/tiny-call/summary.txt"
    for f in $(grep -o 'logs/[A-Za-z0-9_.-]*\.log' "$TINY_OUT/summary.txt" | sort -u || true); do
      if [ -f "$TINY_OUT/$f" ] && [ "$(wc -c < "$TINY_OUT/$f")" -le 1048576 ]; then scrub_copy "$TINY_OUT/$f" "$OUT/.archive/tiny-call/$f"; fi
    done
  fi
  if [ -d "$CRITERION_DIR" ]; then
    find "$CRITERION_DIR" -type f -name estimates.json | while IFS= read -r f; do
      rel="criterion/${f#"$CRITERION_DIR"/}"
      mkdir -p "$OUT/.archive/$(dirname "$rel")"
      scrub_copy "$f" "$OUT/.archive/$rel"
    done
  fi
  # The last check: a copy that still holds a checkout or home path form stays out.
  left_out="$(drop_leaking_copies "$OUT/.archive")" || check_ok=0

  if [ -n "$status_line" ]; then overall="$status_line"
  elif [ "$any_fail" = 1 ]; then overall="FAIL"; final_rc=1
  elif [ "$check_ok" = 0 ]; then
    overall="INCOMPLETE (the path check of the archived copies failed, so no archive was written; see below)"; final_rc=3
  elif [ -n "$left_out" ]; then
    overall="INCOMPLETE (files were left out of the archive because they still contain the checkout or home path; see below)"; final_rc=3
  elif [ -n "$limited_parts" ]; then overall="PASS (limited run: $limited_parts; not a complete measurement)"; final_rc=0
  elif [ -n "$missing_parts" ]; then
    overall="INCOMPLETE (nothing failed, but parts the decision needs are missing; see below)"; final_rc=3
  else overall="PASS"; final_rc=0
  fi

  {
    printf 'oxiblas x86 GEMM runbook: %s\n' "$overall"
    printf 'date (UTC)   : %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
    printf 'host         : %s %s %s (no host name recorded)\n' "$(uname -s)" "$(uname -r)" "$(uname -m)"
    printf 'cpu          : %s\n' "$(awk -F: '/^model name/ { sub(/^[ \t]+/, "", $2); print $2; exit }' /proc/cpuinfo 2>/dev/null || true)"
    printf 'cpus         : %s online, %s physical cores; A/B pin CPU %s; A/B threads %s\n' \
      "$N_ONLINE" "$N_PHYSICAL" "$PIN_CPU" "$AB_THREADS"
    printf 'mode         : %s%s%s\n' "$([ "$quick" = 1 ] && echo quick || echo full)" \
      "$([ "$allow_no_avx2" = 1 ] && echo ', --allow-no-avx2 (testing aid; not valid for decisions)' || true)" \
      "${bench_filter:+, bench filter $bench_filter}"
    printf 'steps        : %s\n' "$(for s in 1 2 3 4 tiny 5 6; do want_step "$s" && printf '%s ' "$s"; done; printf '7')"
    printf 'machine check: %s\n' "$([ "$machine_ok" = 1 ] && echo 'x86-64 Linux with avx2 and fma' || printf '%s' "$machine_problem")"
    printf 'require avx2 : OXIBLAS_REQUIRE_AVX2=%s\n' "$require_avx2_text"
    printf 'rustflags    : default policy RUSTFLAGS="" (target/), candidate RUSTFLAGS="%s" (target/cand), 3-candall RUSTFLAGS="%s" (target/cand-bcap)\n' \
      "$CANDIDATE_RUSTFLAGS" "$CANDALL_RUSTFLAGS"
    printf 'removed env  : %s\n' "${removed_env:-none}"
    printf 'openblas     : %s\n' "$OPENBLAS_PATH"
    printf 'git commit   : %s\n' "$git_commit"
    printf 'git dirty    : %s\n' "$git_dirty"
    printf 'earlier run  : %s\n' "$([ -n "$prev_rel" ] && printf 'moved to %s (remove old runs by hand: rm -rf target/x86-runbook-prev)' "$prev_rel" || echo 'none (no target/x86-runbook existed)')"
    printf 'archive      : %s\n' "$tar_note"
    printf '\n%-36s %-11s %-5s %7s  %s\n' "step" "verdict" "exit" "seconds" "details"
    awk -F'\t' '{ printf "%-36s %-11s %-5s %7s  %s\n", $1, $2, $3, $4, $5 }' "$UNITS"
    if want_step 5; then
      printf '\nA/B decisions of step 5 (from ab.tsv; a field may be switched on only where both passes say ADOPT and no verdict is INVALID):\n'
      ab_verdict_report "$AB_OUT" | awk -F'\t' '$1 == "T" { print $2 }' || true
    fi
    if [ -n "$missing_parts" ]; then
      printf '\nmissing:\n'
      printf '%s' "$missing_parts" | sed 's/^/  - /'
    fi
    if [ "$check_ok" = 0 ]; then
      printf '\nno archive: the search of the archived copies for the checkout and home paths failed (see the terminal output)\n'
    elif [ -n "$left_out" ]; then
      printf '\nleft out of the archive (they still contain a form of the checkout or home path; the originals are in target/x86-runbook/):\n'
      printf '%s\n' "$left_out" | sed 's/^/  - /'
    fi
    printf '\noverall      : %s\n' "$overall"
  } > "$OUT/summary.txt"

  # The summary itself is checked too (it cannot name itself as left out).
  scrub_copy "$OUT/summary.txt" "$OUT/.archive/summary.txt"
  if [ "$check_ok" = 1 ]; then
    if ! f="$(find_leaks "$OUT/.archive/summary.txt")"; then check_ok=0
    elif [ -n "$f" ]; then rm -f "$OUT/.archive/summary.txt"; summary_left=1
    fi
  fi
  if { [ "$check_ok" = 0 ] || [ "$summary_left" = 1 ]; } && [ -z "$status_line" ] && [ "$any_fail" = 0 ]; then final_rc=3; fi
  rm -f "$ARCHIVE.tmp"
  if [ "$check_ok" = 0 ]; then
    note "error: the path check of the archived copies failed; no archive was written"
  elif (cd "$OUT/.archive" && tar ${tar_opts[@]+"${tar_opts[@]}"} -czf "$ARCHIVE.tmp" .); then
    mv -f "$ARCHIVE.tmp" "$ARCHIVE"
  else
    rm -f "$ARCHIVE.tmp"
    note "error: tar failed; there is no archive"
  fi
  rm -rf "$OUT/.archive" "$UNITS" "$OUT/.fifo"
  cat "$OUT/summary.txt"
  if [ -f "$ARCHIVE" ]; then printf '\nSend back: %s\n' "$ARCHIVE"
  else printf '\nNo archive was written (see above); look through %s for paths before you send it.\n' "$OUT/summary.txt"
  fi
  if [ -n "$left_out" ] || [ "$summary_left" = 1 ]; then
    printf 'Left out of the archive because a form of the checkout or home path remained in it:%s\n' \
      "$([ "$summary_left" = 1 ] && printf ' summary.txt')$(printf '%s\n' "$left_out" | awk 'NF { printf " %s", $0 }')"
  fi
  if [ -n "$prev_rel" ]; then printf 'The results of the earlier run were moved to %s\n' "$ROOT/$prev_rel"; fi
  if [ -n "$missing_parts" ]; then
    printf '\nThis run lacks parts that the decision needs:\n%s\n' "$(printf '%s' "$missing_parts" | sed 's/^/  - /')"
  fi
}

on_signal() {
  local sig="$1" code="$2" id agg secs
  # A second signal must not stop the clean-up half way.
  trap '' INT TERM HUP
  interrupted="SIG$sig"
  id="$current_unit"
  agg="$current_agg"
  secs=$((SECONDS - current_start))
  stop_child
  if [ -n "$id" ]; then
    record "$id" INTERRUPTED "-" "$secs" "stopped by $interrupted"
    printf '== interrupted by %s\n' "$interrupted" >> "$LOG_DIR/$id.log" 2>/dev/null || true
  fi
  if [ -n "$agg" ] && [ "$agg" != "$id" ]; then
    record "$agg" INTERRUPTED "-" "$((SECONDS - agg_start))" "stopped by $interrupted after $agg_groups group processes"
  fi
  finalize "INTERRUPTED (by $interrupted${id:+ during $id})"
  exit "$code"
}

on_exit() {
  local rc=$?
  stop_child
  if [ "$finalized" = 0 ] && [ "$dry_run" = 0 ] && [ "$out_ready" = 1 ]; then
    finalize "ABORTED (the script stopped early with exit status $rc)"
  fi
  release_lock
}

# ============================================================== main program
# ---------------------------------------------------------------- arguments
# --help first (an option's value is skipped). Nothing is written before all
# options, the checkout and the machine have been checked.
arg_index=1
while [ "$arg_index" -le "$#" ]; do
  case "${!arg_index}" in
    -h|--help) print_help; exit 0 ;;
    --only|--skip|--bench-filter) arg_index=$((arg_index + 1)) ;;
  esac
  arg_index=$((arg_index + 1))
done

dry_run=0; quick=0; allow_no_avx2=0; self_test=0; arg_count="$#"
only_list=""; skip_list=""; bench_filter=""; only_seen=0; skip_seen=0; filter_seen=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --quick) quick=1; shift ;;
    --allow-no-avx2) allow_no_avx2=1; shift ;;
    --self-test) self_test=1; shift ;;
    --only|--only=*|--skip|--skip=*|--bench-filter|--bench-filter=*)
      opt_name="${1%%=*}"
      if [ "$1" = "$opt_name" ]; then
        [ "$#" -ge 2 ] || die 2 "$opt_name needs a value"
        opt_value="$2"
        shift 2
      else
        opt_value="${1#*=}"
        shift
      fi
      case "$opt_value" in
        --*) die 2 "$opt_name needs a value, not the option '$opt_value'" ;;
      esac
      case "$opt_name" in
        --only)
          [ "$only_seen" = 0 ] || die 2 "--only may be given only once"
          only_seen=1
          only_list="$(parse_steps "$opt_value" --only)" || exit 2 ;;
        --skip)
          [ "$skip_seen" = 0 ] || die 2 "--skip may be given only once"
          skip_seen=1
          skip_list="$(parse_steps "$opt_value" --skip)" || exit 2 ;;
        --bench-filter)
          [ "$filter_seen" = 0 ] || die 2 "--bench-filter may be given only once"
          filter_seen=1
          [ -n "$opt_value" ] || die 2 "--bench-filter needs a non-empty value"
          bench_filter="$opt_value" ;;
      esac ;;
    *) die 2 "unknown option: $1 (see --help)" ;;
  esac
done
if [ "$only_seen" = 1 ] && [ "$skip_seen" = 1 ]; then
  die 2 "--only and --skip cannot be combined"
fi
if [ "$self_test" = 1 ] && [ "$arg_count" != 1 ]; then
  die 2 "--self-test takes no other option"
fi

# ------------------------------------------------------------------ checkout
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
ROOT_LOGICAL="$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd -L)"
[ -f "$ROOT/Cargo.toml" ] && [ -f "$ROOT/crates/oxiblas-blas/Cargo.toml" ] \
  && [ -f "$ROOT/crates/oxiblas-benchmarks/Cargo.toml" ] \
  || die 2 "cannot find the oxiblas workspace above $SCRIPT_DIR"
set_path_forms "$ROOT" "$ROOT_LOGICAL" "${HOME:-}"
OUT="$ROOT/target/x86-runbook"
PREV_BASE="$ROOT/target/x86-runbook-prev"
LOCK_DIR="$ROOT/target/x86-runbook.lock"
LOG_DIR="$OUT/logs"
UNITS="$OUT/.units"
AB_OUT="$OUT/ab.tsv"
CRITERION_DIR="$OUT/criterion"
ARCHIVE="$OUT/x86-runbook-results.tar.gz"
LEGACY_TARGET="$ROOT/target"
CAND_TARGET="$ROOT/target/cand"
CANDALL_TARGET="$ROOT/target/cand-bcap"
GATE_FILE="$ROOT/crates/oxiblas-blas/src/level3/gemm_plan.rs"
HARNESS_DIR="$ROOT/crates/oxiblas-blas/src/level3/gemm_ab_tests"
TINY_TOOL="$SCRIPT_DIR/gemm_tiny_call_check.sh"
TINY_OUT="$OUT/tiny-call"
# Step 5 (the A/B harness and its verdicts) is in a file of its own beside
# this script, loaded here for every run, the dry run and the self-test.
AB_STEP_FILE="$SCRIPT_DIR/x86_gemm_runbook_ab.sh"
[ -f "$AB_STEP_FILE" ] || die 2 "the step-5 file ${AB_STEP_FILE#"$ROOT"/} is missing (it belongs beside this script); nothing was changed"
# shellcheck source=x86_gemm_runbook_ab.sh disable=SC1091
. "$AB_STEP_FILE"
if [ "$self_test" = 1 ]; then
  # The self-test lives beside this script and is loaded only here.
  SELFTEST_FILE="$SCRIPT_DIR/x86_gemm_runbook_selftest.sh"
  [ -f "$SELFTEST_FILE" ] || die 2 "the self-test file ${SELFTEST_FILE#"$ROOT"/} is missing (it belongs beside this script)"
  # shellcheck source=x86_gemm_runbook_selftest.sh disable=SC1091
  . "$SELFTEST_FILE"
  run_self_test
fi

# Step 4 relies on the texts the AVX2 test gate prints: the body of
# avx2_fma_tier_or_skip (up to its closing brace) must still contain them.
[ -f "$GATE_FILE" ] || die 2 "${GATE_FILE#"$ROOT"/} is missing"
gate_body="$(awk '!/^[ \t]*\/\// && /fn avx2_fma_tier_or_skip\(/ { on = 1 } on { print } on && /^}/ { exit }' "$GATE_FILE")"
[ -n "$gate_body" ] || die 2 "${GATE_FILE#"$ROOT"/} no longer defines avx2_fma_tier_or_skip, the AVX2 test gate step 4 relies on (update this script together with the gate)"
for gate_text in "{test}: $SKIP_TEXT" "{test}: $REQUIRE_TEXT" "{test}: $CONFIRM_TEXT"; do
  case "$gate_body" in
    *"$gate_text"*) ;;
    *) die 2 "avx2_fma_tier_or_skip in ${GATE_FILE#"$ROOT"/} no longer contains the text '$gate_text'; step 4 searches the logs for it (update this script together with the gate)" ;;
  esac
done
# Step 5 requires the harness's decision grade: the key in output.rs and the
# word in settings.rs must still be the ones this script looks for.
if [ -d "$HARNESS_DIR" ] && { ! grep -q -F -e 'Self::Decision => "decision"' "$HARNESS_DIR/settings.rs" 2>/dev/null \
    || ! grep -q -F -e '("grade", self.grade' "$HARNESS_DIR/output.rs" 2>/dev/null; }; then
  die 2 "the A/B harness (${HARNESS_DIR#"$ROOT"/}) no longer writes the SET field $GRADE_DECISION this script requires in a full run (settings.rs: Self::Decision => \"decision\"; output.rs: (\"grade\", self.grade ...); update this script together with the harness"
fi
# Step 5 compares the verdicts of the two passes: the decision words
# (stats.rs: exactly the five it judges, see ab_word_problem) and the ABV
# columns and verdict entries (output.rs) it reads.
if [ -d "$HARNESS_DIR" ]; then
  ab_missing="$(ab_word_problem "$HARNESS_DIR/stats.rs")"
  [ -z "$ab_missing" ] || ab_missing=" stats.rs: $ab_missing;"
  for ab_text in '"invalid",' '"verdicts",' '.verdicts.join(";")' '"{name}:{decision},{figures},partial={partial}"' '"geo={:.4},'; do
    grep -q -F -e "$ab_text" "$HARNESS_DIR/output.rs" 2>/dev/null || ab_missing="$ab_missing output.rs: $ab_text not found;"
  done
  [ -z "$ab_missing" ] || die 2 "the A/B harness (${HARNESS_DIR#"$ROOT"/}) no longer writes the verdicts step 5 compares as this script reads them (step 5 judges exactly the decision words $AB_DECISION_WORDS):$ab_missing update this script (x86_gemm_runbook_ab.sh) together with the harness; nothing was changed"
fi

# ------------------------------------------------------- cargo configuration
CONFIG_FILES="$(config_files)"
if [ -n "${CARGO_BUILD_TARGET:-}" ]; then
  die 2 "CARGO_BUILD_TARGET is set; the runbook measures the host's own target in target/ (unset it and run again)"
fi
refusals=""
while IFS="$TAB" read -r _ cfg_path _; do
  [ -n "$cfg_path" ] || continue
  cfg_keys="$(config_refusals "$cfg_path" | paste -s -d, - | sed 's/,/, /g')"
  if [ -n "$cfg_keys" ]; then refusals="$refusals $cfg_path sets $cfg_keys;"; fi
done <<EOF
$CONFIG_FILES
EOF
if [ -n "$refusals" ]; then
  die 2 "a cargo config file sets keys the runbook cannot allow:$refusals build.target makes cargo build for another target than the host, and an [env] entry named OXIBLAS_*, OPENBLAS_*, OMP_* or RAYON_* reaches the test and bench processes and changes the measurements. Remove these keys, or run from a checkout outside that directory; nothing was changed"
fi

# --------------------------------------------------------------- the machine
host_os="$(uname -s)"
host_arch="$(uname -m)"
cpu_flags=""
if [ -r /proc/cpuinfo ]; then
  cpu_flags="$(awk -F: '/^flags[[:space:]]*:/ { print $2; exit }' /proc/cpuinfo)"
fi
machine_ok=1
machine_problem=""
if [ "$host_os" != Linux ] || [ "$host_arch" != x86_64 ]; then
  machine_ok=0
  machine_problem="this is not x86-64 Linux (found $host_os on $host_arch); the runbook measures the x86-64 AVX2 code paths and must run on an x86-64 Linux machine with AVX2 and FMA"
elif ! has_flag avx2 || ! has_flag fma; then
  machine_ok=0
  machine_problem="this CPU does not report both avx2 and fma in /proc/cpuinfo; the runbook needs an x86-64 CPU with AVX2 and FMA (Intel Haswell / AMD Zen or newer)"
fi
if [ "$machine_ok" = 0 ]; then
  if [ "$dry_run" = 1 ]; then note "note: $machine_problem; a real run would stop here (without --allow-no-avx2)"
  elif [ "$allow_no_avx2" = 1 ]; then
    note "warning: $machine_problem; going on because of --allow-no-avx2 (a testing aid: these results cannot decide the defaults)"
  else die 2 "$machine_problem (--allow-no-avx2 overrides this check, for testing only)"
  fi
fi
for tool in cargo rustc; do
  if command -v "$tool" >/dev/null 2>&1; then continue
  elif [ "$dry_run" = 1 ]; then note "note: $tool is not on PATH; a real run would stop here"
  else die 2 "$tool is not on PATH; install Rust with rustup (https://rustup.rs) or add ~/.cargo/bin to PATH, then run again; nothing was changed"
  fi
done

N_ONLINE="$(online_cpus)"
N_PHYSICAL="$(physical_cores)"
PIN_CPU="$(pin_cpu)"
AB_THREADS="$(ab_thread_list)"
[ -n "$AB_THREADS" ] || AB_THREADS="1"
have_taskset=0
command -v taskset >/dev/null 2>&1 && have_taskset=1
OPENBLAS_PATH="${OPENBLAS_LIB:-$DEFAULT_OPENBLAS_LIB}"

# ---------------------------------------------------------- the environment
# Recorded before anything is unset: the names of the inherited CARGO_* and
# RUST* variables (values only for CARGO_PROFILE_*), and the OpenBLAS and
# OpenMP variables with their values.
inherited_cargo_names="$(compgen -e | grep -E '^(CARGO|RUST)' | sort -u | paste -s -d' ' - || true)"
inherited_profile_values=""
for name in $(compgen -e | grep -E '^CARGO_PROFILE_' | sort -u || true); do
  inherited_profile_values="$inherited_profile_values$name=${!name} "
done
inherited_openblas=""
for name in $OPENBLAS_ENV_NAMES; do
  if [ -n "${!name+set}" ]; then inherited_openblas="$inherited_openblas$name=${!name} "; fi
done

removed_env=""
for name in $(compgen -e | grep -E '^OXIBLAS_|^CARGO_TARGET_[A-Za-z0-9_]*_RUSTFLAGS$' | sort -u || true) \
    RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS RUSTDOCFLAGS \
    CARGO_ENCODED_RUSTDOCFLAGS CARGO_BUILD_RUSTDOCFLAGS CARGO_TARGET_DIR \
    CARGO_BUILD_TARGET_DIR CRITERION_HOME RAYON_NUM_THREADS $OPENBLAS_ENV_NAMES; do
  if [ -n "${!name+set}" ]; then
    removed_env="$removed_env $name"
    unset "$name"
  fi
done
removed_env="${removed_env# }"
# Set but empty: cargo then ignores build.rustflags and target.*.rustflags of
# every config file and CARGO_TARGET_<triple>_RUSTFLAGS. Each command also
# names its RUSTFLAGS explicitly (empty or the candidate's cfg flags).
export RUSTFLAGS=""
export CARGO_TERM_COLOR=never
if [ "$allow_no_avx2" = 1 ]; then require_avx2_text="not set (--allow-no-avx2)"
else export OXIBLAS_REQUIRE_AVX2=1; require_avx2_text="1"
fi
# The environment prefixes of the three build configurations.
LEGACY_ENV=(env "RUSTFLAGS=" "CARGO_TARGET_DIR=$LEGACY_TARGET")
CAND_ENV=(env "RUSTFLAGS=$CANDIDATE_RUSTFLAGS" "CARGO_TARGET_DIR=$CAND_TARGET")
CANDALL_ENV=(env "RUSTFLAGS=$CANDALL_RUSTFLAGS" "CARGO_TARGET_DIR=$CANDALL_TARGET")

limited_parts=""
[ -n "$only_list" ] && limited_parts="$limited_parts, --only$(printf '%s' "$only_list" | tr -s ' ' ',' | sed 's/,$//; s/^,/ /')"
[ -n "$skip_list" ] && limited_parts="$limited_parts, --skip$(printf '%s' "$skip_list" | tr -s ' ' ',' | sed 's/,$//; s/^,/ /')"
[ "$quick" = 1 ] && limited_parts="$limited_parts, --quick"
[ -n "$bench_filter" ] && limited_parts="$limited_parts, --bench-filter $bench_filter"
[ "$allow_no_avx2" = 1 ] && limited_parts="$limited_parts, --allow-no-avx2"
limited_parts="${limited_parts#, }"

trap on_exit EXIT
trap 'on_signal INT 130' INT; trap 'on_signal TERM 143' TERM; trap 'on_signal HUP 129' HUP

# ------------------------------------------------ lock and earlier results
# Only now, for a real run: the lock, then an existing target/x86-runbook is
# moved, in one rename, to target/x86-runbook-prev/<UTC time>[-n]/; nothing
# is deleted.
prev_rel=""
if [ "$dry_run" = 0 ]; then
  mkdir -p "$ROOT/target" 2>/dev/null || die 2 "cannot create $ROOT/target (is the checkout writable?); nothing was changed"
  take_lock "$LOCK_DIR"
  if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    mkdir -p "$PREV_BASE" 2>/dev/null \
      || die 2 "cannot create the directory $PREV_BASE for the earlier results (is there a file of that name? rename it); nothing was changed"
    prev_dest="$PREV_BASE/$stamp"; prev_n=2
    while [ -e "$prev_dest" ] || [ -L "$prev_dest" ]; do
      prev_dest="$PREV_BASE/$stamp-$prev_n"
      prev_n=$((prev_n + 1))
    done
    mv "$OUT" "$prev_dest" || die 2 "cannot move the earlier results $OUT to $prev_dest; nothing was changed"
    prev_rel="${prev_dest#"$ROOT"/}/"; note "the earlier results were moved to $prev_rel"
  fi
  if ! { mkdir -p "$LOG_DIR" 2>/dev/null && : > "$UNITS"; }; then
    die 2 "cannot create $LOG_DIR${prev_rel:+ (the earlier results are safe in $prev_rel)}"
  fi
  out_ready=1
fi

# Job control: every background command gets a process group of its own.
set -m

# -------------------------------------------------------------------- run
if [ "$dry_run" = 1 ]; then
  printf '# dry run: nothing is run and nothing is written\n'
  printf '# checkout (<checkout> below): %s\n' "$ROOT"
  printf '# unset: %s\n' "OXIBLAS_* CARGO_TARGET_*_RUSTFLAGS RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS RUSTDOCFLAGS CARGO_ENCODED_RUSTDOCFLAGS CARGO_BUILD_RUSTDOCFLAGS CARGO_TARGET_DIR CARGO_BUILD_TARGET_DIR CRITERION_HOME RAYON_NUM_THREADS $OPENBLAS_ENV_NAMES"
  printf '# would be unset here: %s\n' "${removed_env:-nothing}"
  printf '# exported for every command: RUSTFLAGS= (set, empty) CARGO_TERM_COLOR=never LC_ALL=C OXIBLAS_REQUIRE_AVX2=%s\n' "$require_avx2_text"
  printf '# each command names RUSTFLAGS and CARGO_TARGET_DIR itself (see env below)\n'
  printf '# online CPUs %s, physical cores %s, A/B pin CPU %s (taskset %s), A/B threads %s\n' \
    "$N_ONLINE" "$N_PHYSICAL" "$PIN_CPU" "$([ "$have_taskset" = 1 ] && echo found || echo missing)" "$AB_THREADS"
  printf '# a real run takes the lock %s first\n' "${LOCK_DIR#"$ROOT"/}"
  if [ -e "$OUT" ]; then printf '# a real run would then move %s to %s/<UTC time>/\n' "${OUT#"$ROOT"/}" "${PREV_BASE#"$ROOT"/}"; fi
  if [ -n "$limited_parts" ]; then printf '# a limited run (%s): not a complete measurement\n' "$limited_parts"; fi
fi

want_step 1 && step_machine
want_step 2 && step_tests 2 legacy
want_step 3 && step_tests 3 cand
want_step 4 && step_avx2
want_step tiny && step_tiny
want_step 5 && step_ab
want_step 6 && step_bench

if [ "$dry_run" = 1 ]; then
  printf '\n## 7-summary: write target/x86-runbook/summary.txt and %s (via %s.tmp)\n' "${ARCHIVE#"$ROOT"/}" "$(basename "$ARCHIVE")"
  exit 0
fi
finalize ""
[ "$final_rc" = 0 ] || exit "$final_rc"
