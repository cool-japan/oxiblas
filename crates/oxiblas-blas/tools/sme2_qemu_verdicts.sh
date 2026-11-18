# shellcheck shell=bash disable=SC2034
# sme2_qemu_verdicts.sh -- the verdict rules of sme2_qemu.sh.
#
# Not a program of its own: sme2_qemu.sh sources this file from its own
# directory at every start (without it, it stops with exit 2), before it
# reads its options. It holds the functions that judge a pass from its logs
# (judge_pass for the unit tests, judge_hardware for --hardware, judge_stage
# for the public-path stage), the split of a pass's output into its logs,
# and what --hardware reads about the CPU. They use the constants and
# variables of sme2_qemu.sh (ROOT, SIGILL_HINT, SIGILL_CHECK_HINT) and set
# variables that its run_pass and its self-test read (SC2034 is off above:
# those readers are not in this file).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  printf '%s: this file is sourced by sme2_qemu.sh; run sme2_qemu.sh --help\n' "${0##*/}" >&2
  exit 2
fi

# The version of these rules: sme2_qemu.sh stops (exit 2) when it is not the
# one the script was written for, or when a function it calls is missing.
SQ_VERDICTS_VERSION=2

# ----------------------------------------------------------------- verdicts
# Counts the occurrences of a fixed text in a file (anywhere in a line: with
# one test thread libtest prints `test <name> ... ` before the test runs, and
# a marker written meanwhile lands in the middle of that line).
count_text() {
  { grep -o -F -e "$1" "$2" 2>/dev/null || true; } | wc -l | tr -d ' '
}

# The signal that ended a test binary, as cargo reports it in the log $1
# (`(signal: 4, SIGILL: ...)` gives SIGILL); empty when there is none.
signal_of() {
  awk 'match($0, /\(signal: [0-9]+, SIG[A-Z0-9]+/) { s = substr($0, RSTART, RLENGTH); sub(/.*, /, "", s); print s; exit }' "$1" 2>/dev/null || true
}

# The verdict of a pass from cargo's exit status and the pass log (the rules
# of Exit status in the header, in this order). Sets n_results, passed,
# failed, ignored, skips, gate_ran, gate_skipped, gate_declined, verdict and
# reason (the caller's variables where it declares them local).
#   judge_pass <positive|negative> <cargo exit status> <log>
judge_pass() {
  local pass="$1" rc="$2" log="$3" counts signal
  counts="$(awk '/^test result: / {
      n++
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^passed;?$/) p += $(i - 1)
        if ($i ~ /^failed;?$/) f += $(i - 1)
        if ($i ~ /^ignored;?$/) g += $(i - 1)
      }
    } END { printf "%d %d %d %d\n", n, p, f, g }' "$log")"
  read -r n_results passed failed ignored <<EOF
$counts
EOF
  skips="$(grep -c '^skip: ' "$log" || true)"
  gate_ran="$(count_text 'sme2-gate: ran ' "$log")"
  gate_skipped="$(count_text 'sme2-gate: skipped ' "$log")"
  gate_declined="$(count_text 'sme2-gate: declined ' "$log")"
  verdict="PASS"
  reason=""
  if [ "$rc" != 0 ]; then
    verdict="FAIL"; reason="cargo exited with $rc"
  elif [ "$n_results" = 0 ] || [ "$passed" = 0 ]; then
    verdict="FAIL"; reason="no test ran (check the filter)"
  elif [ "$failed" != 0 ]; then
    verdict="FAIL"; reason="$failed failed"
  elif [ "$pass" = positive ] && [ "$skips" != 0 ]; then
    verdict="FAIL"; reason="$skips skip line(s) in the positive pass"
  elif [ "$pass" = positive ] && [ "$gate_ran" = 0 ]; then
    verdict="FAIL"; reason="no gated SME2 test ran: the filter selects only tests that do not execute SME2 code"
  elif [ "$pass" = positive ] && [ "$gate_skipped" != 0 ]; then
    verdict="FAIL"; reason="$gate_skipped gated test(s) skipped SME2 in the positive pass"
  elif [ "$pass" = positive ] && [ "$gate_declined" != 0 ]; then
    verdict="FAIL"; reason="$gate_declined test(s) saw the backend decline in the positive pass"
  elif [ "$pass" = negative ] && [ "$gate_skipped" = 0 ]; then
    verdict="FAIL"; reason="no gated SME2 test was declined: the filter selects only tests that do not execute SME2 code"
  elif [ "$pass" = negative ] && [ "$gate_ran" != 0 ]; then
    verdict="FAIL"; reason="$gate_ran gated test(s) ran SME2 in the negative pass"
  elif [ "$pass" = negative ] && [ "$gate_declined" = 0 ]; then
    verdict="FAIL"; reason="no test saw the backend decline: the filter selects no *_declines_without_backend test"
  fi
  # QEMU's report of a signal in the emulated code (SIGILL, an abort) fails
  # the pass whatever cargo's exit status says.
  if grep -q 'uncaught target signal' "$log"; then
    verdict="FAIL"
    reason="${reason:+$reason; }QEMU reported an uncaught target signal"
  fi
  # The signal that ended a test binary, as cargo reports it (for the reason).
  signal="$(signal_of "$log")"
  if [ -n "$signal" ]; then reason="${reason:+$reason; }a test binary was ended by $signal"; fi
}

# The verdict of the --hardware pass: the positive pass's rules (judge_pass),
# and the detection report (cargo log $3 of test_sme2_detect_report, cargo
# status $5) must have run and read `hw(): Some(...)`. Another value: FAIL;
# with a report line `candidate: Some(...)` and `first-use check: failed:
# <line>` the reason is that the first-use self-check failed (with the
# line), with `first-use check: open` that it did not run; otherwise (no
# candidate, or a report without those lines) "SME2 was not detected", with
# the values read ($4: hw_raw_values) and, on Linux, a default streaming
# vector length other than the 64 bytes the backend needs. No hw() line (a
# build failure, SIGILL in the detection's
# probe, no such test): FAIL "the detection report did not run", with the
# signal or cargo's status; but with a `candidate: Some(...)` line before
# the end (the report prints it before it starts the first-use check): FAIL
# "the first-use self-check did not end", and a SIGILL there gets
# SIGILL_CHECK_HINT instead of SIGILL_HINT. Some(...) without a clean end:
# FAIL. Some(...) with `f64: false` (no FEAT_SME_F64F64): FAIL, because
# OXIBLAS_TEST_REQUIRE_SME2 requires the f64 kernels too, and the f64 tests'
# `sme2-gate: ran` lines of the public-path stage (the public sme2_active()
# is the f32 backend) would not mean SME2. Any other reason that names
# SIGILL ends with SIGILL_HINT.
#   judge_hardware <cargo exit status> <pass log> <report log> <values file> <report's cargo exit status>
judge_hardware() {
  local found sig how read_values vl note="" report_status="${5:-0}" candidate check
  judge_pass positive "$1" "$2"
  found="$(awk 'i = index($0, "hw(): ") { print substr($0, i); exit }' "$3" 2>/dev/null || true)"
  candidate="$(awk 'i = index($0, "candidate: ") { print substr($0, i + 11); exit }' "$3" 2>/dev/null || true)"
  check="$(awk 'i = index($0, "first-use check: ") { print substr($0, i + 17); exit }' "$3" 2>/dev/null || true)"
  sig="$(signal_of "$3")"
  if [ -n "$sig" ]; then how="its test binary was ended by $sig"
  elif [ "$report_status" != 0 ]; then how="its cargo exited with $report_status"
  else how="its cargo exited with 0, but the log has no hw() line (test_sme2_detect_report did not run)"
  fi
  case "$candidate" in
    "Some("*) [ -n "$sig" ] || [ "$report_status" != 0 ] || how="its cargo exited with 0, but the log has no hw() line after the candidate line" ;;
  esac
  read_values="$(awk -F'\t' '$1 !~ /Features$|brand_string$/ { printf "%s%s=%s", s, $1, $2; s = ", " }' "$4" 2>/dev/null || true)"
  case "$found" in
    "hw(): Some("*)
      if [ -n "$sig" ] || [ "$report_status" != 0 ]; then
        verdict="FAIL"
        reason="the detection report read $found but did not end (see detect-report.log): $how${reason:+; $reason}"
      fi
      case "$found" in
        *"f64: false"*)
          verdict="FAIL"
          reason="${reason:+$reason; }FEAT_SME_F64F64 is absent (the library's detection reports $found): f64 products run on NEON, so the f64 tests' sme2-gate: ran lines do not mean SME2, and OXIBLAS_TEST_REQUIRE_SME2 requires it" ;;
      esac ;;
    "")
      verdict="FAIL"
      case "$candidate" in
        "Some("*)
          # The probe passed (the report prints the candidate before it
          # starts the first-use check), so the check ended the report.
          reason="the first-use self-check did not end (see detect-report.log): the report read candidate: $candidate, then $how; read: $read_values${reason:+; $reason}"
          case "$reason" in *SIGILL*) reason="$reason$SIGILL_CHECK_HINT" ;; esac
          return 0 ;;
      esac
      reason="the detection report did not run (see detect-report.log): $how; read: $read_values${reason:+; $reason}" ;;
    *)
      case "$candidate|$check" in
        "Some("*"|failed: "*)
          verdict="FAIL"
          reason="the first-use self-check failed on this CPU, so the backend is off: ${check#failed: } (the library's detection reports $found; candidate: $candidate)${reason:+; $reason}"
          case "$reason" in *SIGILL*) reason="$reason$SIGILL_HINT" ;; esac
          return 0 ;;
        "Some("*"|open")
          verdict="FAIL"
          reason="the first-use self-check did not run: its verdict is open because the reporting thread was not ready (the library's detection reports $found; candidate: $candidate)${reason:+; $reason}"
          case "$reason" in *SIGILL*) reason="$reason$SIGILL_HINT" ;; esac
          return 0 ;;
      esac
      vl="$(awk -F'\t' '$1 == "/proc/sys/abi/sme_default_vector_length" { print $2 }' "$4" 2>/dev/null || true)"
      case "$vl" in
        ''|64|missing*) ;;
        *) note="; the default streaming vector length of a process is $vl bytes (/proc/sys/abi/sme_default_vector_length) and the backend needs 64: if the CPU offers 512 bits, raise it as root (echo 64 > /proc/sys/abi/sme_default_vector_length) and run again" ;;
      esac
      verdict="FAIL"
      reason="SME2 was not detected (the library's detection reports $found; read: $read_values)$note${reason:+; $reason}" ;;
  esac
  case "$reason" in *SIGILL*) reason="$reason$SIGILL_HINT" ;; esac
}

# The values the detection reads, read by this script itself, as
# "name<TAB>value" lines: on macOS the sysctl keys of detect.rs (its KEY_*
# constants) and the CPU brand string; on Linux the Features line of
# /proc/cpuinfo, whether it lists sme, sme2 and smef64f64, and the default
# streaming vector length.
hw_raw_values() {
  local key keys v features w yn=""
  case "$(uname -s)" in
    Darwin)
      keys="$(sed -n 's/^const KEY_[A-Z0-9_]*: &str = "\([A-Za-z0-9_.]*\)";$/\1/p' "$ROOT/crates/oxiblas-blas/src/level3/sme2/detect.rs" 2>/dev/null || true)"
      [ -n "$keys" ] || printf 'detect.rs\tno KEY_* sysctl constant found (update this script together with detect.rs)\n'
      for key in $keys machdep.cpu.brand_string; do
        v="$(/usr/sbin/sysctl -n "$key" 2>/dev/null)" || v="missing (sysctl knows no such key)"
        printf '%s\t%s\n' "$key" "$v"
      done ;;
    *)
      features="$(awk -F': *' '/^Features/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true)"
      printf '/proc/cpuinfo Features\t%s\n' "${features:-none}"
      for w in sme sme2 smef64f64; do
        case " $features " in *" $w "*) yn="$yn $w=yes" ;; *) yn="$yn $w=no" ;; esac
      done
      printf '/proc/cpuinfo flags\t%s\n' "${yn# }"
      v="$(cat /proc/sys/abi/sme_default_vector_length 2>/dev/null)" || v="missing (no such file: the kernel offers no SME)"
      printf '/proc/sys/abi/sme_default_vector_length\t%s\n' "$v" ;;
  esac
}

# The lines test_sme2_detect_report printed, from its cargo log $1, without
# cargo's and libtest's own lines.
hw_report_lines() {
  awk '/^running [0-9]+ tests?$/ { on = 1; next } /^test result: / { on = 0 }
    on && NF && !/^test [^ ]+ \.\.\. (ok|FAILED|ignored)$/ { sub(/^test [^ ]+ \.\.\. /, ""); print }' "$1" 2>/dev/null || true
}

# ------------------------------------------------- steps and the stage
# The test binaries of the public-path stage, in the order of the summary,
# and the one whose gate writes the sme2-gate lines.
STAGE_BINARIES="sme2_public_path gemm_rayon_pool"
STAGE_GATED="sme2_public_path"

# Splits the output of a pass, collected in its log $1, at the step lines
# that the step runner writes with the run's token $2 (`== step <name>
# [<token>]` before a cargo command, `== step <name> [<token>]: cargo exited
# with <status> after <n>s` after it): the lines before the first step (the
# pass's header and the runner's version lines) go to every log, the steps
# `unit` to $1 itself, `public-release` to $4 and `public-debug` to $5 (with
# $3 = 1; with $3 = 0 there is no stage and $1 gets everything). A line
# like a step line but with another token is an ordinary line.
# The output is first renamed to <pass log>.stream, the only complete copy
# while the split runs, which is removed only after awk has written every
# log. So the split can be run again at any time (the signal handler does)
# and leaves complete logs: with that copy present (a split cut short, for
# example its awk stopped by a Ctrl-C) it starts again from the copy;
# without it, with the stage, and both stage logs present (they are removed
# at the start of a run, and only a split writes them before the pass's
# verdict is taken) the split has ended already and nothing changes. Returns
# 1 when the output could not be renamed, or when awk failed (the copy
# stays).
#   split_pass_log <pass log> <token> <stage 0|1> <release log> <debug log>
split_pass_log() {
  local log="$1" stream="$1.stream"
  if [ ! -f "$stream" ]; then
    [ -f "$log" ] || return 0
    if [ "$3" = 1 ] && [ -f "$4" ] && [ -f "$5" ]; then return 0; fi
    mv -f "$log" "$stream" || return 1
  fi
  : > "$log"
  if [ "$3" = 1 ]; then : > "$4"; : > "$5"; fi
  # The paths come in through ENVIRON (awk -v would interpret backslashes).
  SQ_UNIT="$log" SQ_REL="$4" SQ_DBG="$5" awk -v tok="[$2]" -v stage="$3" '
    BEGIN { unit = ENVIRON["SQ_UNIT"]; rel = ENVIRON["SQ_REL"]; dbg = ENVIRON["SQ_DBG"] }
    function flush(   i) {
      if (flushed) return
      flushed = 1
      for (i = 1; i <= nl; i++) { print lead[i] > unit; if (stage) { print lead[i] > rel; print lead[i] > dbg } }
    }
    function file_of(name) {
      if (stage && name == "public-release") return rel
      if (stage && name == "public-debug") return dbg
      return unit
    }
    NF == 4 && $1 == "==" && $2 == "step" && $4 == tok { flush(); cur = file_of($3); print > cur; next }
    { if (cur == "") lead[++nl] = $0; else print > cur }
    END { flush() }' "$stream" || return 1
  rm -f "$stream"
}

# The exit status and time ("<status> <n>s") of step $2 in log $1 (token $3),
# from its end line; empty when the step did not end (or did not start).
step_exit() {
  awk -v want="[$3]:" -v name="$2" '$1 == "==" && $2 == "step" && $3 == name && $4 == want && $5 " " $6 " " $7 == "cargo exited with" { r = $8 " " $10 } END { if (r != "") print r }' "$1" 2>/dev/null || true
}

# Whether step $2 started in log $1 (token $3).
step_begun() {
  awk -v want="[$3]" -v name="$2" 'NF == 4 && $1 == "==" && $2 == "step" && $3 == name && $4 == want { found = 1 } END { exit !found }' "$1" 2>/dev/null
}

# Adds the reason $1 to judge_stage's list (its local problems and sep).
stage_problem() {
  problems="$problems$sep$1"
  sep="; "
}

# The verdict of one build of the public-path stage (the rules of Exit
# status in the header): cargo's exit status $3 (a number; `unended` when
# the step started but wrote no end line, `unrun` when it never started,
# with the status of the pass's whole job $5), and its log $4, read per test
# binary (the lines after cargo's `Running tests/<name>.rs` line belong to
# <name>). Every rule that fails adds its reason (not only the first one).
# For pass $1 (positive, negative or hardware) and build $2 (release or
# debug) it sets stage_verdict, stage_reason ("public path, <build> build:
# ..."), stage_counts (the summary's line) and stage_ran, stage_skipped,
# stage_declined (sme2-gate lines of the whole log). A log that could not
# be judged (awk failed, no table, a test binary without its row) fails.
#   judge_stage <pass> <build> <cargo exit status|unended|unrun> <log> <job status>
judge_stage() {
  local pass="$1" build="$2" rc="$3" log="$4" job_rc="${5:-?}" table totals name seen results p f r s d q sig
  local skips tq problems="" any_q=0 sep="" table_ok=1 rows=" " bin
  if ! table="$(awk -v bins="$STAGE_BINARIES" '
    function count(line, t,   c, i) { c = 0; while ((i = index(line, t)) > 0) { c++; line = substr(line, i + length(t)) } return c }
    /^[ \t]*Running / { sec = ($2 == "unittests") ? $3 : $2; gsub(/`/, "", sec); sub(/^.*\//, "", sec); sub(/\.rs$/, "", sec); sub(/-[0-9a-f]+$/, "", sec); seen[sec] = 1; next }
    /^test result: / {
      n[sec]++
      for (i = 2; i <= NF; i++) { if ($i ~ /^passed;?$/) p[sec] += $(i - 1); if ($i ~ /^failed;?$/) f[sec] += $(i - 1) }
    }
    { c = count($0, "sme2-gate: ran "); r[sec] += c; R += c
      c = count($0, "sme2-gate: skipped "); s[sec] += c; S += c
      c = count($0, "sme2-gate: declined "); d[sec] += c; D += c }
    /^skip: / { K++ }
    index($0, "uncaught target signal") { q[sec] = 1; Q = 1 }
    match($0, /\(signal: [0-9]+, SIG[A-Z0-9]+/) && !(sec in g) { x = substr($0, RSTART, RLENGTH); sub(/.*, /, "", x); g[sec] = x }
    END {
      printf "%d %d %d %d %d\n", R, S, D, K, Q
      nb = split(bins, b, " ")
      for (k = 1; k <= nb; k++) {
        c = b[k]
        printf "%s %d %d %d %d %d %d %d %d %s\n", c, seen[c], n[c], p[c], f[c], r[c], s[c], d[c], q[c], (c in g) ? g[c] : "-"
      }
    }' "$log" 2>/dev/null)"; then
    table=""
  fi
  [ -n "$table" ] || table_ok=0
  totals="$(printf '%s\n' "$table" | sed -n 1p)"
  read -r stage_ran stage_skipped stage_declined skips tq <<EOF
${totals:-0 0 0 0 0}
EOF
  stage_counts=""
  if [ "$rc" = unrun ]; then
    stage_problem "the stage did not run (the pass's job ended with $job_rc before it started)"
  else
    [ "$table_ok" = 1 ] || stage_problem "the log ${log##*/} could not be judged (it cannot be read, or awk failed)"
    while read -r name seen results p f r s d q sig; do
      [ -n "$name" ] || continue
      rows="$rows$name "
      stage_counts="${stage_counts:+$stage_counts; }$name: $p passed, $f failed"
      if [ "$name" = "$STAGE_GATED" ]; then stage_counts="$stage_counts, sme2-gate $r ran / $s skipped / $d declined"; fi
      if [ "$results" = 0 ]; then stage_problem "$name: no test result (the binary did not run or did not end)"
      elif [ "$f" != 0 ]; then stage_problem "$name: $f failed"
      elif [ "$p" = 0 ]; then stage_problem "$name: no test passed"
      fi
      if [ "$sig" != - ]; then stage_problem "$name was ended by $sig"; fi
      if [ "$q" = 1 ]; then stage_problem "$name: QEMU reported an uncaught target signal"; any_q=1; fi
      if [ "$name" = "$STAGE_GATED" ]; then
        case "$pass" in
          negative) [ "$s" != 0 ] || stage_problem "$name wrote no sme2-gate: skipped line (its gate was not reached)" ;;
          *) [ "$r" != 0 ] || stage_problem "$name wrote no sme2-gate: ran line (nothing ran on SME2)" ;;
        esac
      fi
    done <<EOF
$(printf '%s\n' "$table" | sed 1d)
EOF
    if [ "$table_ok" = 1 ]; then
      for bin in $STAGE_BINARIES; do
        case "$rows" in *" $bin "*) ;; *) stage_problem "$bin: no row in the judgement of the log ${log##*/} (awk's output is incomplete)" ;; esac
      done
    fi
    case "$pass" in
      negative) [ "$stage_ran" = 0 ] || stage_problem "$stage_ran sme2-gate: ran line(s) in the negative pass" ;;
      *)
        [ "$stage_skipped" = 0 ] || stage_problem "$stage_skipped sme2-gate: skipped line(s)"
        [ "$stage_declined" = 0 ] || stage_problem "$stage_declined sme2-gate: declined line(s)"
        [ "$skips" = 0 ] || stage_problem "$skips skip line(s)" ;;
    esac
    if [ "$tq" = 1 ] && [ "$any_q" = 0 ]; then stage_problem "QEMU reported an uncaught target signal"; fi
    sig="$(signal_of "$log")"
    case "$problems" in *" was ended by "*) ;; *) [ -z "$sig" ] || stage_problem "a test binary was ended by $sig" ;; esac
    case "$rc" in
      0) ;;
      unended) stage_problem "cargo did not end (the pass's job ended with $job_rc)" ;;
      *) stage_problem "cargo exited with $rc" ;;
    esac
  fi
  case "$rc" in
    unrun) stage_counts="did not run" ;;
    unended) stage_counts="${stage_counts:-no test binary}; cargo did not end" ;;
    *) stage_counts="${stage_counts:-no test binary}; cargo exit $rc" ;;
  esac
  if [ -z "$problems" ]; then
    stage_verdict="PASS"; stage_reason=""
  else
    stage_verdict="FAIL"; stage_reason="public path, $build build: $problems"
    if [ "$pass" = hardware ]; then case "$stage_reason" in *SIGILL*) stage_reason="$stage_reason$SIGILL_HINT" ;; esac; fi
  fi
}
