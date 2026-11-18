# shellcheck shell=bash disable=SC2154,SC2034
# sme2_qemu_selftest.sh -- the self-test of sme2_qemu.sh.
#
# Not a program of its own: sme2_qemu.sh sources this file from its own
# directory for --self-test only, after it has defined its functions and
# found the checkout, and then calls run_self_test, which exits. Run it as
#   crates/oxiblas-blas/tools/sme2_qemu.sh --self-test
# The functions here use the functions, constants and variables of
# sme2_qemu.sh and sme2_qemu_verdicts.sh (SC2154 and SC2034 are off above: ROOT, SCRIPT_NAME,
# MARKERS_VAR, SIGILL_HINT, verdict, reason and the like are set there, and
# the variables set here, such as RUN_LOCK and current_fifo, are read by its
# functions, which shellcheck cannot see from this file alone).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  printf '%s: this file is sourced by sme2_qemu.sh; run sme2_qemu.sh --self-test\n' "${0##*/}" >&2
  exit 2
fi

# ---------------------------------------------------------------- self-test
# Every expectation prints one line; the count of "NOT OK" lines decides.
st_expect() {
  local what="$1"; shift
  if "$@"; then printf 'ok      %s\n' "$what"; else printf 'NOT OK  %s\n' "$what"; fi
}

# Writes the canned pass log $1: the script's own header lines, then
# libtest's lines with $2 `ran`, $3 `skipped` and $4 `declined` markers (the
# first `ran` marker glued to a `test <name> ... ` line, as with one test
# thread), the further arguments as lines of their own, and a `test result:`
# line with $5 passed and $6 failed tests.
#   st_log <file> <ran> <skipped> <declined> <passed> <failed> [<line>...]
st_log() {
  local f="$1" ran="$2" skipped="$3" declined="$4" passed="$5" failed="$6" i line
  shift 6
  {
    printf '== pass  : canned\n== env   : OXIBLAS_TEST_REQUIRE_SME2=1 OXIBLAS_TEST_EXPECT_NO_SME2=0 %s=1\n' "$MARKERS_VAR"
    printf 'running %s tests\n' "$((passed + failed))"
    i=0; while [ "$i" -lt "$ran" ]; do
      if [ "$i" = 0 ]; then printf 'test level3::sme2::t%s ... sme2-gate: ran level3::sme2::t%s\nok\n' "$i" "$i"
      else printf 'sme2-gate: ran level3::sme2::t%s\n' "$i"; fi
      i=$((i + 1))
    done
    i=0; while [ "$i" -lt "$skipped" ]; do printf 'sme2-gate: skipped level3::sme2::s%s\n' "$i"; i=$((i + 1)); done
    i=0; while [ "$i" -lt "$declined" ]; do printf 'sme2-gate: declined level3::sme2::d%s\n' "$i"; i=$((i + 1)); done
    for line in "$@"; do printf '%s\n' "$line"; done
    printf 'test result: %s. %s passed; %s failed; 0 ignored; 0 measured; 12 filtered out\n' \
      "$([ "$failed" = 0 ] && echo ok || echo FAILED)" "$passed" "$failed"
  } > "$f"
}

# Whether judge_pass gives verdict $2 for pass $1, log $3 and cargo exit
# status $4, with the text $5 in its reason (and, when $6 is given, the
# marker counts "ran skipped declined" $6).
# shellcheck disable=SC2329 # invoked through st_expect
st_judge() {
  local verdict reason n_results passed failed ignored skips gate_ran gate_skipped gate_declined
  judge_pass "$1" "$4" "$3"
  [ "$verdict" = "$2" ] || return 1
  case "$reason" in *"${5:-}"*) ;; *) return 1 ;; esac
  [ -z "${6:-}" ] || [ "$gate_ran $gate_skipped $gate_declined" = "$6" ]
}

# Whether judge_hardware gives verdict $1 for log $2, cargo exit status $3,
# detection report $4 (which cargo ended with status $5) and values $6, with
# the text $7 in its reason and, when $8 is given, without the text $8.
# shellcheck disable=SC2329 # invoked through st_expect
st_hw() {
  local verdict reason n_results passed failed ignored skips gate_ran gate_skipped gate_declined
  judge_hardware "$3" "$2" "$4" "$6" "$5"
  [ "$verdict" = "$1" ] || return 1
  case "$reason" in *"${7:-}"*) ;; *) return 1 ;; esac
  [ -z "${8:-}" ] || case "$reason" in *"$8"*) return 1 ;; esac
}

# How often judge_hardware's reason (arguments as st_hw's $2 to $6) holds $1.
# shellcheck disable=SC2329 # invoked through st_expect
st_hw_count() {
  local verdict reason n_results passed failed ignored skips gate_ran gate_skipped gate_declined text="$1"
  judge_hardware "$3" "$2" "$4" "$6" "$5"
  printf '%s' "$reason" | awk -v t="$text" '{ while ((i = index($0, t)) > 0) { n++; $0 = substr($0, i + length(t)) } } END { print n + 0 }'
}

# Whether the command (the arguments) fails.
# shellcheck disable=SC2329 # invoked through st_expect
st_not() { ! "$@"; }

st_qemu_rules() {
  local d="$ST"
  st_log "$d/p-ok" 3 0 0 5 0
  st_expect "positive: ran 3 (one glued to libtest's line), no skipped, no declined, cargo 0: PASS" \
    st_judge positive PASS "$d/p-ok" 0 "" "3 0 0"
  st_log "$d/p-skipped" 3 1 0 5 0
  st_expect "positive: one skipped marker: FAIL" \
    st_judge positive FAIL "$d/p-skipped" 0 "1 gated test(s) skipped SME2 in the positive pass"
  st_log "$d/p-declined" 3 0 1 5 0
  st_expect "positive: one declined marker: FAIL" \
    st_judge positive FAIL "$d/p-declined" 0 "1 test(s) saw the backend decline in the positive pass"
  st_log "$d/p-noran" 0 0 0 5 0
  st_expect "positive: zero ran: FAIL" st_judge positive FAIL "$d/p-noran" 0 "no gated SME2 test ran"
  st_log "$d/p-signal" 3 0 0 5 0 "qemu: uncaught target signal 4 (Illegal instruction) - core dumped"
  st_expect "positive: an uncaught target signal with cargo exit 0: FAIL" \
    st_judge positive FAIL "$d/p-signal" 0 "QEMU reported an uncaught target signal"
  st_log "$d/p-failed" 3 0 0 4 1
  st_expect "positive: a failed test, cargo exit 101: FAIL" st_judge positive FAIL "$d/p-failed" 101 "cargo exited with 101"
  st_expect "positive: a failed test with cargo exit 0: FAIL" st_judge positive FAIL "$d/p-failed" 0 "1 failed"
  st_log "$d/p-skipline" 3 0 0 5 0 "skip: level3::sme2::x: no SME2 (F32) on this host"
  st_expect "positive: a skip: line: FAIL" st_judge positive FAIL "$d/p-skipline" 0 "1 skip line(s) in the positive pass"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'error: could not compile `oxiblas-blas` (lib test)\n' > "$d/p-nobuild"
  st_expect "positive: no test result line, cargo exit 0: FAIL" st_judge positive FAIL "$d/p-nobuild" 0 "no test ran (check the filter)"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'running 9 tests\nerror: test failed, to rerun pass `-p oxiblas-blas --lib`\n\nCaused by:\n  process did not exit successfully: `x` (signal: 4, SIGILL: illegal instruction)\n' > "$d/p-sigill"
  st_expect "positive: a test binary ended by SIGILL: FAIL naming the signal" \
    st_judge positive FAIL "$d/p-sigill" 101 "cargo exited with 101; a test binary was ended by SIGILL"
  st_log "$d/n-ok" 0 3 2 5 0
  st_expect "negative: skipped 3, ran 0, declined 2, cargo 0: PASS" st_judge negative PASS "$d/n-ok" 0 "" "0 3 2"
  st_log "$d/n-ran" 1 3 2 5 0
  st_expect "negative: one ran marker: FAIL" st_judge negative FAIL "$d/n-ran" 0 "1 gated test(s) ran SME2 in the negative pass"
  st_log "$d/n-nodecl" 0 3 0 5 0
  st_expect "negative: no declined marker: FAIL" st_judge negative FAIL "$d/n-nodecl" 0 "no test saw the backend decline"
  st_log "$d/n-noskip" 0 0 2 5 0
  st_expect "negative: no skipped marker: FAIL" st_judge negative FAIL "$d/n-noskip" 0 "no gated SME2 test was declined"
  st_log "$d/n-signal" 0 3 2 5 0 "qemu: uncaught target signal 6 (Aborted) - core dumped"
  st_expect "negative: an uncaught target signal with cargo exit 0: FAIL" \
    st_judge negative FAIL "$d/n-signal" 0 "QEMU reported an uncaught target signal"
  st_log "$d/p-bytes" 3 0 0 5 0 "$(printf 'raw bytes \377\376 \303( of a test')"
  st_expect "positive: a log with bytes that are no UTF-8: the counts still hold, PASS" \
    st_judge positive PASS "$d/p-bytes" 0 "" "3 0 0"
}

# Writes the canned stage log $1 of one build as cargo prints it: the
# section of gemm_rayon_pool ($7 passed, $8 failed; "-": no such section),
# then the one of sme2_public_path with $2 `ran`, $3 `skipped` and $4
# `declined` markers (the first `ran` glued to a `test <name> ... ` line),
# the further arguments as lines of its own, and $5 passed, $6 failed
# ("-" for $5: no test result line).
#   st_stage_log <file> <ran> <skipped> <declined> <passed> <failed> <rayon passed> <rayon failed> [<line>...]
st_stage_log() {
  local f="$1" ran="$2" skipped="$3" declined="$4" passed="$5" failed="$6" gp="$7" gf="$8" i line
  shift 8
  {
    # shellcheck disable=SC2016 # cargo's messages hold literal backticks
    printf '== pass  : canned\n   Compiling oxiblas-blas v0.2.4\n    Finished `release` profile [optimized] target(s)\n'
    if [ "$gp" != - ]; then
      printf '     Running tests/gemm_rayon_pool.rs (/target/aarch64-unknown-linux-gnu/release/deps/gemm_rayon_pool-1d8c3b9b13b54314)\n\nrunning 1 test\n'
      printf 'test test_gemm_gated_rayon_call_leaves_the_global_pool_unbuilt ... ok\n\ntest result: %s. %s passed; %s failed; 0 ignored; 0 measured; 0 filtered out\n\n' \
        "$([ "$gf" = 0 ] && echo ok || echo FAILED)" "$gp" "$gf"
    fi
    printf '     Running tests/sme2_public_path.rs (/target/aarch64-unknown-linux-gnu/release/deps/sme2_public_path-a1661c23c8064919)\n\nrunning 4 tests\n'
    i=0; while [ "$i" -lt "$ran" ]; do
      if [ "$i" = 0 ]; then printf 'test test_sme2_public_path_gemm_f32 ... sme2-gate: ran test_sme2_public_path_t%s\nok\n' "$i"
      else printf 'sme2-gate: ran test_sme2_public_path_t%s\n' "$i"; fi
      i=$((i + 1))
    done
    i=0; while [ "$i" -lt "$skipped" ]; do printf 'sme2-gate: skipped test_sme2_public_path_s%s\n' "$i"; i=$((i + 1)); done
    i=0; while [ "$i" -lt "$declined" ]; do printf 'sme2-gate: declined test_sme2_public_path_d%s\n' "$i"; i=$((i + 1)); done
    for line in "$@"; do printf '%s\n' "$line"; done
    if [ "$passed" != - ]; then
      printf '\ntest result: %s. %s passed; %s failed; 0 ignored; 0 measured; 0 filtered out\n' \
        "$([ "$failed" = 0 ] && echo ok || echo FAILED)" "$passed" "$failed"
    fi
  } > "$f"
}

# Whether judge_stage gives verdict $5 for pass $1, build $2, cargo status
# $3 and log $4, with the text $6 in its reason, without the text $7 (when
# given), and with the pass's job status $8 (default 0).
# shellcheck disable=SC2329 # invoked through st_expect
st_stage() {
  local stage_verdict stage_reason stage_counts stage_ran stage_skipped stage_declined
  judge_stage "$1" "$2" "$3" "$4" "${8:-0}"
  [ "$stage_verdict" = "$5" ] || return 1
  case "$stage_reason" in *"$6"*) ;; *) return 1 ;; esac
  [ -z "${7:-}" ] || case "$stage_reason" in *"$7"*) return 1 ;; esac
}

# judge_stage's summary line for pass $1 and log $2 (cargo status 0).
# shellcheck disable=SC2329 # invoked through st_expect
st_stage_counts() {
  local stage_verdict stage_reason stage_counts stage_ran stage_skipped stage_declined
  judge_stage "$1" release 0 "$2" 0
  printf '%s|%s %s %s\n' "$stage_counts" "$stage_ran" "$stage_skipped" "$stage_declined"
}

# The stand-in awk, a function that shadows awk while ST_AWK_MODE is set:
# =fail makes judge_stage's awk (the one given `-v bins=...`) fail without
# output, =failafter makes it fail after its whole output, =droprow drops
# the row of gemm_rayon_pool from its output, =stop
# makes the split's awk (the one reading a `.stream` file) stop after
# ST_AWK_LINES lines of it with status 130, as if a Ctrl-C had ended it;
# any other call runs the real awk.
# shellcheck disable=SC2329 # invoked as awk while st_with_awk runs
st_awk() {
  local last="" a
  for a in "$@"; do last="$a"; done
  case "${ST_AWK_MODE:-}|$*" in
    fail\|*" bins="*) return 2 ;;
    failafter\|*" bins="*) command awk "$@"; return 2 ;;
    droprow\|*" bins="*) command awk "$@" | grep -v '^gemm_rayon_pool '; return 0 ;;
  esac
  case "${ST_AWK_MODE:-}|$last" in
    stop\|*.stream) head -n "${ST_AWK_LINES:-1}" "$last" | command awk "${@:1:$(($# - 1))}"; return 130 ;;
  esac
  command awk "$@"
}

# Runs the command (the arguments after $1) with the stand-in awk in mode
# $1 and returns its status; awk is itself again afterwards.
# shellcheck disable=SC2329 # invoked through st_expect
st_with_awk() {
  local rc=0
  ST_AWK_MODE="$1"
  shift
  awk() { st_awk "$@"; }
  "$@" || rc=$?
  unset -f awk
  ST_AWK_MODE=""
  return "$rc"
}

# The files $2... joined with `|` between their lines and `#` between files
# (a snapshot of a split's logs); $1 is the directory they are in.
# shellcheck disable=SC2329 # invoked through st_expect
st_snap() {
  local dir="$1" f
  shift
  for f in "$@"; do printf '%s#' "$(paste -s -d '|' - < "$dir/$f" 2>/dev/null)"; done
}

# The signal handler's path (on_signal) in a subshell, for the pass `cut`
# whose output is collected in $1/cut.log (this run's token, with the
# stage): the split, then the interrupted line in every log. At this point
# of the self-test no process of its own runs and no lock is held, so
# stop_pass and the lock releases do nothing.
st_on_signal() {
  (OUT="$1"; split_pass="cut"; current_log="$1/cut.log"; on_signal TERM 143) 2>/dev/null || true
}

st_stage_rules() {
  local d="$ST" rel="public path, release build: " dbg="public path, debug build: " pp="sme2_public_path"
  local whole cut i tk="$run_token"
  st_stage_log "$d/s-ok" 4 0 0 4 0 1 0
  st_expect "stage: positive, 4 ran from $pp (one glued to libtest's line), both binaries passed: PASS" \
    st_stage positive release 0 "$d/s-ok" PASS ""
  st_expect "stage: the summary line counts per binary and the markers" \
    test "$(st_stage_counts positive "$d/s-ok")" = "$pp: 4 passed, 0 failed, sme2-gate 4 ran / 0 skipped / 0 declined; gemm_rayon_pool: 1 passed, 0 failed; cargo exit 0|4 0 0"
  st_stage_log "$d/s-noran" 0 0 0 4 0 1 0
  st_expect "stage: positive, no ran line: FAIL naming the build and $pp" \
    st_stage positive release 0 "$d/s-noran" FAIL "$rel$pp wrote no sme2-gate: ran line (nothing ran on SME2)"
  st_expect "stage: hardware, no ran line: FAIL (the positive rule)" \
    st_stage hardware debug 0 "$d/s-noran" FAIL "$dbg$pp wrote no sme2-gate: ran line"
  st_stage_log "$d/s-elsewhere" 0 0 0 4 0 1 0
  awk '{ print } /^running 1 test$/ { print "sme2-gate: ran test_sme2_public_path_x" }' "$d/s-elsewhere" > "$d/s-elsewhere2"
  st_expect "stage: positive, a ran line only in gemm_rayon_pool's section: FAIL (it must come from $pp)" \
    st_stage positive release 0 "$d/s-elsewhere2" FAIL "$pp wrote no sme2-gate: ran line"
  st_stage_log "$d/s-skipped" 4 1 0 4 0 1 0
  st_expect "stage: positive, a skipped line: FAIL" st_stage positive release 0 "$d/s-skipped" FAIL "${rel}1 sme2-gate: skipped line(s)"
  st_stage_log "$d/s-declined" 4 0 1 4 0 1 0
  st_expect "stage: positive, a declined line: FAIL" st_stage positive release 0 "$d/s-declined" FAIL "${rel}1 sme2-gate: declined line(s)"
  st_stage_log "$d/s-skipline" 4 0 0 4 0 1 0 "skip: test_sme2_public_path_gemm_f32: no SME2"
  st_expect "stage: positive, a skip: line: FAIL" st_stage positive release 0 "$d/s-skipline" FAIL "${rel}1 skip line(s)"
  st_stage_log "$d/s-failed" 4 0 0 3 1 1 0
  st_expect "stage: a failed test of $pp in the debug build, cargo 101: FAIL naming the build, the binary and cargo's status" \
    st_stage positive debug 101 "$d/s-failed" FAIL "$dbg$pp: 1 failed; cargo exited with 101"
  st_stage_log "$d/s-norayon" 4 0 0 4 0 - -
  st_expect "stage: no section of gemm_rayon_pool: FAIL naming it" \
    st_stage positive release 0 "$d/s-norayon" FAIL "${rel}gemm_rayon_pool: no test result (the binary did not run or did not end)"
  st_stage_log "$d/s-rayon0" 4 0 0 4 0 0 0
  st_expect "stage: gemm_rayon_pool passed no test: FAIL" st_stage positive release 0 "$d/s-rayon0" FAIL "${rel}gemm_rayon_pool: no test passed"
  st_stage_log "$d/s-rayonf" 4 0 0 4 0 0 1
  st_expect "stage: gemm_rayon_pool failed: FAIL naming it" st_stage positive release 101 "$d/s-rayonf" FAIL "${rel}gemm_rayon_pool: 1 failed"
  st_stage_log "$d/s-qemu" 4 0 0 4 0 1 0 "qemu: uncaught target signal 4 (Illegal instruction) - core dumped"
  st_expect "stage: an uncaught target signal in $pp, cargo 0: FAIL naming the binary" \
    st_stage positive release 0 "$d/s-qemu" FAIL "$rel$pp: QEMU reported an uncaught target signal"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  st_stage_log "$d/s-sigill" 1 0 0 - - 1 0 'error: test failed, to rerun pass `-p oxiblas-blas --test sme2_public_path`' "" "Caused by:" '  process didn'\''t exit successfully: `x` (signal: 4, SIGILL: illegal instruction)'
  st_expect "stage: $pp ended by SIGILL: FAIL naming the binary, the signal and its missing result" \
    st_stage positive release 101 "$d/s-sigill" FAIL "$rel$pp: no test result (the binary did not run or did not end); $pp was ended by SIGILL"
  st_expect "stage: a SIGILL on hardware gets the hint" st_stage hardware release 101 "$d/s-sigill" FAIL "$SIGILL_HINT"
  st_expect "stage: a SIGILL under QEMU gets no hardware hint" st_stage positive release 101 "$d/s-sigill" FAIL "SIGILL" "$SIGILL_HINT"
  st_stage_log "$d/s-neg" 0 4 0 4 0 1 0
  st_expect "stage: negative, 4 skipped, no ran: PASS" st_stage negative debug 0 "$d/s-neg" PASS ""
  st_stage_log "$d/s-negran" 1 3 0 4 0 1 0
  st_expect "stage: negative, a ran line: FAIL" st_stage negative release 0 "$d/s-negran" FAIL "${rel}1 sme2-gate: ran line(s) in the negative pass"
  st_stage_log "$d/s-negnoskip" 0 0 0 4 0 1 0
  st_expect "stage: negative, no skipped line: FAIL (the gate was not reached)" \
    st_stage negative release 0 "$d/s-negnoskip" FAIL "$rel$pp wrote no sme2-gate: skipped line (its gate was not reached)"
  st_expect "stage: cargo exited with 101 although every result passed: FAIL" st_stage positive release 101 "$d/s-ok" FAIL "${rel}cargo exited with 101"
  st_expect "stage: started but no end line: FAIL, cargo did not end, with the job's status" \
    st_stage positive debug unended "$d/s-ok" FAIL "${dbg}cargo did not end (the pass's job ended with 143)" "" 143
  st_expect "stage: never started: FAIL, the stage did not run" \
    st_stage positive release unrun "$d/s-ok" FAIL "${rel}the stage did not run (the pass's job ended with 2 before it started)" "" 2
  st_stage_log "$d/s-disabled" 0 0 0 0 4 1 0
  st_expect "stage: every test of $pp failed and no ran line (SME2 off under REQUIRE): both reasons" \
    st_stage positive release 101 "$d/s-disabled" FAIL "$rel$pp: 4 failed; $pp wrote no sme2-gate: ran line (nothing ran on SME2)"
  st_stage_log "$d/s-bytes" 4 0 0 4 0 1 0 "$(printf 'raw bytes \377\376 \303( of a test')"
  st_expect "stage: a log with bytes that are no UTF-8: PASS" st_stage positive release 0 "$d/s-bytes" PASS ""
  st_expect "stage: a log that cannot be read (no such file), cargo 0: FAIL, it could not be judged" \
    st_stage positive release 0 "$d/s-no-such-log" FAIL "${rel}the log s-no-such-log could not be judged (it cannot be read, or awk failed)"
  st_expect "stage: judge_stage's awk fails on a passing log: FAIL, it could not be judged" \
    st_with_awk fail st_stage positive release 0 "$d/s-ok" FAIL "${rel}the log s-ok could not be judged"
  st_expect "stage: judge_stage's awk fails after its whole output: FAIL, it could not be judged" \
    st_with_awk failafter st_stage positive release 0 "$d/s-ok" FAIL "${rel}the log s-ok could not be judged"
  st_expect "stage: awk's output lacks the row of gemm_rayon_pool: FAIL naming it" \
    st_with_awk droprow st_stage positive release 0 "$d/s-ok" FAIL "${rel}gemm_rayon_pool: no row in the judgement of the log s-ok"
  # shellcheck disable=SC2016 # cargo's verbose lines hold literal backticks
  sed 's|^     Running tests/\([a-z0-9_]*\)\.rs (\(.*\)-\([0-9a-f]*\))$|     Running `\2-\3`|' "$d/s-ok" > "$d/s-verbose"
  st_expect "stage: cargo's verbose Running lines (the binary's path) are read too: PASS" \
    test "$(grep -c '^     Running `/target' "$d/s-verbose")|$(st_stage positive release 0 "$d/s-verbose" PASS "" && echo PASS)" = "2|PASS"
  # The split of a pass's output at the step lines of this run's token.
  printf '%s\n' "== pass  : canned" "== rustc : x" "== step unit [T1]" "== cargo test --lib" "unit line" "== step public-release [T2]" "unit line 2" "" \
    "== step unit [T1]: cargo exited with 0 after 5s" "== step public-release [T1]" "release line" "" \
    "== step public-release [T1]: cargo exited with 0 after 6s" "== step public-debug [T1]" "debug line" "" \
    "== step public-debug [T1]: cargo exited with 101 after 7s" > "$d/split.log"
  split_pass_log "$d/split.log" T1 1 "$d/split-r.log" "$d/split-d.log"
  st_expect "split: the pass log holds the header and the unit step, a step line of another token as an ordinary line" \
    test "$(paste -s -d '|' - < "$d/split.log")" = "== pass  : canned|== rustc : x|== step unit [T1]|== cargo test --lib|unit line|== step public-release [T2]|unit line 2||== step unit [T1]: cargo exited with 0 after 5s"
  st_expect "split: the release log holds the header and its step" \
    test "$(paste -s -d '|' - < "$d/split-r.log")" = "== pass  : canned|== rustc : x|== step public-release [T1]|release line||== step public-release [T1]: cargo exited with 0 after 6s"
  st_expect "split: the debug log holds the header and its step" \
    test "$(sed -n '1p;3,4p' "$d/split-d.log" | paste -s -d '|' -)" = "== pass  : canned|== step public-debug [T1]|debug line"
  st_expect "split: the end lines give each step's status and time" \
    test "$(step_exit "$d/split.log" unit T1)|$(step_exit "$d/split-d.log" public-debug T1)|$(step_exit "$d/split-r.log" public-debug T1)" = "0 5s|101 7s|"
  st_expect "split: the collected output is gone" test ! -e "$d/split.log.stream"
  printf '%s\n' "== pass  : canned" "docker create exited with 125" > "$d/split2.log"
  split_pass_log "$d/split2.log" T1 1 "$d/split2-r.log" "$d/split2-d.log"
  st_expect "split: no step line at all (the job never started): every log holds the output, no step began" \
    test "$(cat "$d/split2.log")|$(cat "$d/split2-d.log")|$(step_begun "$d/split2-r.log" public-release T1 && echo begun)" = "$(printf '== pass  : canned\ndocker create exited with 125')|$(printf '== pass  : canned\ndocker create exited with 125')|"
  printf '%s\n' "== step unit [T1]" "x" "== step unit [T1]: cargo exited with 0 after 1s" "== step public-release [T1]" "y" > "$d/split3.log"
  split_pass_log "$d/split3.log" T1 1 "$d/split3-r.log" "$d/split3-d.log"
  st_expect "split: a step that began without an end line: begun, no status" \
    test "$(step_begun "$d/split3-r.log" public-release T1 && echo begun)|$(step_exit "$d/split3-r.log" public-release T1)" = "begun|"
  printf '%s\n' "h" "== step unit [T1]" "u" > "$d/split4.log"
  split_pass_log "$d/split4.log" T1 0 "$d/split4-r.log" "$d/split4-d.log"
  st_expect "split: without the stage everything stays in the pass log, no stage log is written" \
    test "$(paste -s -d '|' - < "$d/split4.log")|$([ -e "$d/split4-r.log" ] && echo r)" = "h|== step unit [T1]|u|"
  # Run twice, cut short, and after either the signal handler's path: every
  # log complete. The pass `cut` in $d/sig*, with this run's token.
  mkdir -p "$d/sig1" "$d/sig2" "$d/sig3"
  printf '%s\n' "== pass  : canned" "== step unit [$tk]" "unit line" "" "== step unit [$tk]: cargo exited with 0 after 5s" \
    "== step public-release [$tk]" "release line" "" "== step public-release [$tk]: cargo exited with 0 after 6s" \
    "== step public-debug [$tk]" "debug line" "" "== step public-debug [$tk]: cargo exited with 0 after 7s" > "$d/sig1/cut.log"
  cp "$d/sig1/cut.log" "$d/sig2/cut.log"; cp "$d/sig1/cut.log" "$d/sig3/cut.log"
  split_pass_log "$d/sig1/cut.log" "$tk" 1 "$d/sig1/cut-public-release.log" "$d/sig1/cut-public-debug.log"
  whole="$(st_snap "$d/sig1" cut.log cut-public-release.log cut-public-debug.log)"
  split_pass_log "$d/sig1/cut.log" "$tk" 1 "$d/sig1/cut-public-release.log" "$d/sig1/cut-public-debug.log"
  st_expect "split: run twice, the second run changes nothing (every log complete, no collected output left)" \
    test "$(st_snap "$d/sig1" cut.log cut-public-release.log cut-public-debug.log)|$([ -e "$d/sig1/cut.log.stream" ] && echo stream)" = "$whole|"
  st_expect "split: the logs of a complete split hold every step" \
    test "$whole" = "== pass  : canned|== step unit [$tk]|unit line||== step unit [$tk]: cargo exited with 0 after 5s#== pass  : canned|== step public-release [$tk]|release line||== step public-release [$tk]: cargo exited with 0 after 6s#== pass  : canned|== step public-debug [$tk]|debug line||== step public-debug [$tk]: cargo exited with 0 after 7s#"
  st_on_signal "$d/sig1"
  i="|== interrupted by SIGTERM: the pass was stopped; no summary and no archive were written"
  st_expect "split: a signal right after a complete split: the handler's split changes nothing, every log gets the interrupted line" \
    test "$(st_snap "$d/sig1" cut.log cut-public-release.log cut-public-debug.log)" = "$(printf '%s' "$whole" | sed "s/#/$i#/g")"
  cut=0; ST_AWK_LINES=7
  st_with_awk stop split_pass_log "$d/sig2/cut.log" "$tk" 1 "$d/sig2/cut-public-release.log" "$d/sig2/cut-public-debug.log" || cut=$?
  st_expect "split: its awk stopped part-way: status 1, the release log cut short, the complete output kept" \
    test "$cut|$(grep -c 'cargo exited' "$d/sig2/cut-public-release.log")|$(grep -c '' "$d/sig2/cut.log.stream" 2>/dev/null)" = "1|0|13"
  split_pass_log "$d/sig2/cut.log" "$tk" 1 "$d/sig2/cut-public-release.log" "$d/sig2/cut-public-debug.log"
  st_expect "split: run again after it was cut short: it starts from the complete output, every log complete" \
    test "$(st_snap "$d/sig2" cut.log cut-public-release.log cut-public-debug.log)|$([ -e "$d/sig2/cut.log.stream" ] && echo stream)" = "$whole|"
  st_with_awk stop split_pass_log "$d/sig3/cut.log" "$tk" 1 "$d/sig3/cut-public-release.log" "$d/sig3/cut-public-debug.log" || true
  st_on_signal "$d/sig3"
  st_expect "split: cut short, then the signal handler's path: every log complete with the interrupted line, no collected output left" \
    test "$(st_snap "$d/sig3" cut.log cut-public-release.log cut-public-debug.log)|$([ -e "$d/sig3/cut.log.stream" ] && echo stream)" = "$(printf '%s' "$whole" | sed "s/#/$i#/g")|"
  # A path with a backslash: awk -v would read `\t` as a tab.
  mkdir -p "$d/back\\tslash"
  printf '%s\n' "h" "== step unit [T1]" "u" "== step public-release [T1]" "r" > "$d/back\\tslash/b.log"
  split_pass_log "$d/back\\tslash/b.log" T1 1 "$d/back\\tslash/b-r.log" "$d/back\\tslash/b-d.log"
  st_expect "split: paths with a backslash are taken as they are" \
    test "$(st_snap "$d/back\\tslash" b.log b-r.log b-d.log)" = "h|== step unit [T1]|u#h|== step public-release [T1]|r#h#"
}

st_hardware_rules() {
  local d="$ST" none="SME2 was not detected (the library's detection reports hw(): None; read: "
  printf 'running 1 test\nhw(): Some(Sme2Hw { f64: true, units: 2 })\nSME units: 2\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-some"
  printf 'running 1 test\nSME2_ALLOWED (build does not cap SIMD): true\nhw(): None\nSME units: 1\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-none"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'error: could not compile `oxiblas-blas` (lib test)\n' > "$d/r-nobuild"
  printf 'hw.optional.arm.FEAT_SME\t1\nhw.optional.arm.FEAT_SME2\t1\nhw.optional.arm.sme_max_svl_b\t64\nmachdep.cpu.brand_string\tApple M4 Pro\n' > "$d/v-m4"
  printf 'hw.optional.arm.FEAT_SME\t0\nhw.optional.arm.FEAT_SME2\t0\nhw.optional.arm.sme_max_svl_b\tmissing (sysctl knows no such key)\nmachdep.cpu.brand_string\tApple M3\n' > "$d/v-m3"
  printf '/proc/cpuinfo Features\tfp asimd sve sme sme2\n/proc/cpuinfo flags\tsme=yes sme2=yes smef64f64=no\n/proc/sys/abi/sme_default_vector_length\t32\n' > "$d/v-linux"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'running 1 test\nerror: test failed, to rerun pass `-p oxiblas-blas --lib`\n\nCaused by:\n  process did not exit successfully: `x` (signal: 4, SIGILL: illegal instruction)\n' > "$d/r-sigill"
  printf 'running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 301 filtered out\n' > "$d/r-zero"
  # Reports with the detection's two stages: a failed first-use check, an
  # open verdict, a passed one, and no candidate at all.
  printf 'running 1 test\ncandidate: Some(Sme2Hw { f64: true, units: 2 })\nfirst-use check: failed: oxiblas: the SME2 backend is off for this process: the check of the SME2 GEMV failed; please report this line\nhw(): None\nSME units: 1\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-failed"
  printf 'running 1 test\ncandidate: Some(Sme2Hw { f64: true, units: 2 })\nfirst-use check: open\nhw(): None\nSME units: 1\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-open"
  printf 'running 1 test\ncandidate: Some(Sme2Hw { f64: true, units: 2 })\nfirst-use check: passed\nhw(): Some(Sme2Hw { f64: true, units: 2 })\nSME units: 2\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-passed"
  printf 'running 1 test\ncandidate: None\nfirst-use check: open\nhw(): None\nSME units: 1\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-nocand"
  # A report cut off after the candidate line: SIGILL inside the first-use
  # check, or cargo's failure there; and one with the stage lines after the
  # hw() line.
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'running 1 test\noperating-system report: Some(..)\ncandidate: Some(Sme2Hw { f64: true, units: 2 })\nerror: test failed, to rerun pass `-p oxiblas-blas --lib`\n\nCaused by:\n  process did not exit successfully: `x` (signal: 4, SIGILL: illegal instruction)\n' > "$d/r-cut-sigill"
  # shellcheck disable=SC2016 # cargo's messages hold literal backticks
  printf 'running 1 test\ncandidate: Some(Sme2Hw { f64: true, units: 2 })\nerror: test failed, to rerun pass `-p oxiblas-blas --lib`\n' > "$d/r-cut"
  printf 'running 1 test\nhw(): None\nfirst-use check: failed: oxiblas: the SME2 backend is off for this process: the check of the SME2 few-rows GEMM failed; please report this line\ncandidate: Some(Sme2Hw { f64: false, units: 1 })\ntest level3::sme2::detect::tests::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 300 filtered out\n' > "$d/r-reordered"
  st_log "$d/h-ok" 3 0 0 5 0
  st_expect "hardware: SME2 detected, the positive rule holds: PASS" st_hw PASS "$d/h-ok" 0 "$d/r-some" 0 "$d/v-m4"
  sed 's/f64: true/f64: false/' "$d/r-some" > "$d/r-nof64"
  st_expect "hardware: SME2 without FEAT_SME_F64F64 (hw(): Some with f64: false): FAIL naming it, although every test passed" \
    st_hw FAIL "$d/h-ok" 0 "$d/r-nof64" 0 "$d/v-m4" "FEAT_SME_F64F64 is absent (the library's detection reports hw(): Some(Sme2Hw { f64: false, units: 2 })): f64 products run on NEON"
  st_log "$d/h-required" 0 0 0 2 3
  st_expect "hardware: no SME2 (hw(): None), required tests fail: FAIL, SME2 was not detected, with the values read" st_hw FAIL "$d/h-required" 101 "$d/r-none" 0 "$d/v-m3" "${none}hw.optional.arm.FEAT_SME=0, hw.optional.arm.FEAT_SME2=0, hw.optional.arm.sme_max_svl_b=missing (sysctl knows no such key)); cargo exited with 101" "did not run"
  st_expect "hardware: hw(): None although every test passed: FAIL, SME2 was not detected" st_hw FAIL "$d/h-ok" 0 "$d/r-none" 0 "$d/v-m3" "$none"
  st_expect "hardware: the detection report did not build: FAIL, it did not run (cargo's status), not 'SME2 was not detected'" st_hw FAIL "$d/h-required" 101 "$d/r-nobuild" 101 "$d/v-m4" "the detection report did not run (see detect-report.log): its cargo exited with 101; read: hw.optional.arm.FEAT_SME=1" "SME2 was not detected"
  st_expect "hardware: SIGILL in the detection's probe: FAIL, it did not run (the signal), not 'SME2 was not detected'" st_hw FAIL "$d/p-sigill" 101 "$d/r-sigill" 101 "$d/v-m4" "the detection report did not run (see detect-report.log): its test binary was ended by SIGILL; read: hw.optional.arm.FEAT_SME=1" "SME2 was not detected"
  st_expect "hardware: SIGILL in the probe and in the pass: the hint once, at the end" test "$(st_hw_count "$SIGILL_HINT" "$d/p-sigill" 101 "$d/r-sigill" 101 "$d/v-m4")" = 1
  st_expect "hardware: a detection report that ran no test: FAIL, it did not run" st_hw FAIL "$d/h-ok" 0 "$d/r-zero" 0 "$d/v-m4" "the detection report did not run (see detect-report.log): its cargo exited with 0, but the log has no hw() line" "SME2 was not detected"
  st_expect "hardware: a detection report that read Some(...) but then failed: FAIL" st_hw FAIL "$d/h-ok" 0 "$d/r-some" 101 "$d/v-m4" "the detection report read hw(): Some(Sme2Hw { f64: true, units: 2 }) but did not end (see detect-report.log): its cargo exited with 101"
  st_expect "hardware: Linux with a default streaming vector length of 32 bytes: FAIL naming it" st_hw FAIL "$d/h-required" 101 "$d/r-none" 0 "$d/v-linux" "/proc/cpuinfo flags=sme=yes sme2=yes smef64f64=no, /proc/sys/abi/sme_default_vector_length=32); the default streaming vector length of a process is 32 bytes"
  st_expect "hardware: candidate and passed first-use check, hw(): Some: PASS" st_hw PASS "$d/h-ok" 0 "$d/r-passed" 0 "$d/v-m4"
  st_expect "hardware: a failed first-use check: FAIL naming the check and its line, not 'SME2 was not detected'" st_hw FAIL "$d/h-ok" 0 "$d/r-failed" 0 "$d/v-m4" "the first-use self-check failed on this CPU, so the backend is off: oxiblas: the SME2 backend is off for this process: the check of the SME2 GEMV failed; please report this line (the library's detection reports hw(): None; candidate: Some(Sme2Hw { f64: true, units: 2 }))" "SME2 was not detected"
  st_expect "hardware: a failed first-use check and failed tests: FAIL with both reasons" st_hw FAIL "$d/h-required" 101 "$d/r-failed" 0 "$d/v-m4" "the first-use self-check failed on this CPU, so the backend is off: oxiblas: the SME2 backend is off for this process: the check of the SME2 GEMV failed; please report this line (the library's detection reports hw(): None; candidate: Some(Sme2Hw { f64: true, units: 2 })); cargo exited with 101" "SME2 was not detected"
  st_expect "hardware: an open first-use verdict: FAIL, the check did not run, not 'SME2 was not detected'" st_hw FAIL "$d/h-ok" 0 "$d/r-open" 0 "$d/v-m4" "the first-use self-check did not run: its verdict is open because the reporting thread was not ready (the library's detection reports hw(): None; candidate: Some(Sme2Hw { f64: true, units: 2 }))" "SME2 was not detected"
  st_expect "hardware: no candidate (new report format): FAIL, SME2 was not detected" st_hw FAIL "$d/h-ok" 0 "$d/r-nocand" 0 "$d/v-m3" "$none" "first-use"
  st_expect "hardware: an old-format report without the stage lines: judged as before, SME2 was not detected" st_hw FAIL "$d/h-ok" 0 "$d/r-none" 0 "$d/v-m4" "${none}hw.optional.arm.FEAT_SME=1" "first-use"
  st_expect "hardware: SIGILL after the candidate line: FAIL, the first-use self-check did not end, with the kernel hint" st_hw FAIL "$d/h-ok" 0 "$d/r-cut-sigill" 101 "$d/v-m4" "the first-use self-check did not end (see detect-report.log): the report read candidate: Some(Sme2Hw { f64: true, units: 2 }), then its test binary was ended by SIGILL; read: hw.optional.arm.FEAT_SME=1" "the detection report did not run"
  st_expect "hardware: SIGILL after the candidate line: the kernel hint, not the probe's" st_hw FAIL "$d/h-ok" 0 "$d/r-cut-sigill" 101 "$d/v-m4" "$SIGILL_CHECK_HINT" "$SIGILL_HINT"
  st_expect "hardware: SIGILL after the candidate line and in the pass: the kernel hint once" test "$(st_hw_count "$SIGILL_CHECK_HINT" "$d/p-sigill" 101 "$d/r-cut-sigill" 101 "$d/v-m4")" = 1
  st_expect "hardware: SIGILL after the candidate line and in the pass: no probe hint" test "$(st_hw_count "$SIGILL_HINT" "$d/p-sigill" 101 "$d/r-cut-sigill" 101 "$d/v-m4")" = 0
  st_expect "hardware: a report cut off after the candidate line by cargo: FAIL, the check did not end, with cargo's status" st_hw FAIL "$d/h-ok" 0 "$d/r-cut" 101 "$d/v-m4" "the first-use self-check did not end (see detect-report.log): the report read candidate: Some(Sme2Hw { f64: true, units: 2 }), then its cargo exited with 101" "SIGILL"
  st_expect "hardware: SIGILL with no candidate line keeps the probe's reason and hint" st_hw FAIL "$d/h-ok" 0 "$d/r-sigill" 101 "$d/v-m4" "the detection report did not run (see detect-report.log): its test binary was ended by SIGILL" "first-use"
  st_expect "hardware: the stage lines after the hw() line: judged by their content" st_hw FAIL "$d/h-ok" 0 "$d/r-reordered" 0 "$d/v-m4" "the first-use self-check failed on this CPU, so the backend is off: oxiblas: the SME2 backend is off for this process: the check of the SME2 few-rows GEMM failed; please report this line (the library's detection reports hw(): None; candidate: Some(Sme2Hw { f64: false, units: 1 }))" "SME2 was not detected"
  st_expect "hardware: the summary's report lines keep the stage lines" \
    test "$(hw_report_lines "$d/r-failed" | sed -n '1p;3p' | paste -s -d '|' -)" = "candidate: Some(Sme2Hw { f64: true, units: 2 })|hw(): None"
  st_expect "hardware: SME2 detected but a gated test skipped: FAIL by the positive rule" st_hw FAIL "$d/p-skipped" 0 "$d/r-some" 0 "$d/v-m4" "1 gated test(s) skipped SME2 in the positive pass"
  st_expect "hardware: a test binary ended by SIGILL: FAIL with the hint" st_hw FAIL "$d/p-sigill" 101 "$d/r-some" 0 "$d/v-m4" "SIGILL (on SME2 hardware: an SME instruction such as mrs tpidr2_el0 or rdsvl is not usable from user space)"
  st_expect "hardware: the summary's report lines leave out cargo's and libtest's lines" \
    test "$(hw_report_lines "$d/r-none" | paste -s -d '|' -)" = "SME2_ALLOWED (build does not cap SIMD): true|hw(): None|SME units: 1"
  hw_raw_values > "$d/v-here"
  if [ "$(uname -s)" = Darwin ]; then
    st_expect "hardware: the values read here name the six sysctl keys of detect.rs and the CPU brand" \
      test "$(cut -f1 "$d/v-here" | paste -s -d ' ' -)" = "hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 hw.optional.arm.FEAT_SME_F64F64 hw.optional.arm.sme_max_svl_b hw.perflevel0.physicalcpu hw.perflevel0.cpusperl2 machdep.cpu.brand_string"
  else
    st_expect "hardware: the values read here are the Features line, the flags and the vector length" \
      test "$(cut -f1 "$d/v-here" | paste -s -d '|' -)" = "/proc/cpuinfo Features|/proc/cpuinfo flags|/proc/sys/abi/sme_default_vector_length"
  fi
}

# take_run_lock's exit status on $1 (in a subshell); with $2, whether its message holds $2.
# shellcheck disable=SC2329 # invoked through st_expect
st_take() {
  local rc=0 msg
  msg="$( (take_run_lock "$1") 2>&1 >/dev/null )" || rc=$?
  if [ -n "${2:-}" ]; then case "$msg" in *"$2"*) echo yes ;; *) echo no ;; esac; else echo "$rc"; fi
}

# The run lock (here in the scratch directory) and the machine-wide lock's staleness.
st_lock_rules() {
  local l="$ST/run.lock" w="$ST/wide.lock" dead start
  st_expect "run lock: a free lock is taken" test "$(st_take "$l")" = 0
  st_expect "run lock: it holds this process id and its start time" test "$(sed -n 1p "$l/pid" 2>/dev/null)|$(sed -n 2p "$l/pid" 2>/dev/null)" = "$$|$(process_start "$$")"
  st_expect "run lock: the start time is the same in any time zone (TZ)" test "$(TZ=XYZ-9 process_start "$$")" = "$(TZ=ABC+5 process_start "$$")"
  st_expect "run lock: a second run is refused with exit 2" test "$(st_take "$l")" = 2
  st_expect "run lock: the refusal names the holder" test "$(st_take "$l" "another run of this script uses this checkout: process $$ (started ")" = yes
  st_expect "run lock: the refusal leaves the holder's lock alone" test "$(sed -n 1p "$l/pid" 2>/dev/null)" = "$$"
  (RUN_LOCK="$l"; release_run_lock)
  st_expect "run lock: released, the directory is gone" test ! -e "$l"
  (exit 0) & dead=$!
  wait "$dead" || true
  mkdir "$l" && printf '%s\nwhenever\n' "$dead" > "$l/pid"
  st_expect "run lock: the lock of a process that is gone is taken over" test "$(st_take "$l")" = 0
  rm -rf "$l"
  start="$(process_start "$$")"
  mkdir "$l" && printf '%s\nMon Jan  1 00:00:00 1990\n' "$$" > "$l/pid"
  if [ -n "$start" ]; then
    st_expect "run lock: a reused process id (another start time) is taken over" test "$(st_take "$l")" = 0
  else
    printf 'ok      run lock: a reused process id is not checked (ps gives no start time here)\n'
  fi
  rm -rf "$l" && mkdir "$l"
  st_expect "run lock: a lock without a pid file younger than a minute (a run starting) is refused" test "$(st_take "$l")" = 2
  touch -t 200001010000 "$l"
  st_expect "run lock: a lock without a pid file older than a minute is taken over" test "$(st_take "$l")" = 0
  rm -rf "$l"
  mkdir "$w" && printf '%s\n' "$$" > "$w/pid"
  st_expect "machine-wide lock: a one-line pid file of a live process is not stale" st_not lock_is_stale "$w"
  printf '%s\n' "$dead" > "$w/pid"
  st_expect "machine-wide lock: a one-line pid file of a process that is gone is stale" lock_is_stale "$w"
  rm -rf "$w"
}

# Whether process $1 is gone.
# shellcheck disable=SC2329 # invoked through st_expect
st_gone() { [ -n "$1" ] && ! kill -0 "$1" 2>/dev/null; }

# stop_pass on a job started as the --native and --hardware passes start
# theirs: a shell whose child (the test binary under cargo) outlives it.
st_signal_rules() {
  local f="$ST/sig.fifo" log="$ST/sig.log" gc="" i=0 t pgid
  rm -f "$f" "$log" && mkfifo "$f"
  tee -a "$log" < "$f" > /dev/null &
  tee_pid=$!; t="$tee_pid"; current_fifo="$f"
  start_group_job "$f" sh -c 'sleep 120 & echo "grandchild $!"; wait'
  while [ -z "$gc" ] && [ "$i" -lt 100 ]; do
    gc="$(sed -n 's/^grandchild //p' "$log" 2>/dev/null)"; [ -n "$gc" ] || sleep 0.1; i=$((i + 1))
  done
  pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d ' ')"
  st_expect "signals: the job has a process group of its own" test "$pgid" = "$child_pid"
  st_expect "signals: its grandchild is in that group" test "$(ps -o pgid= -p "${gc:-0}" 2>/dev/null | tr -d ' ')" = "$pgid"
  stop_pass
  st_expect "signals: stop_pass stops the job's whole process group, the grandchild too" st_gone "$gc"
  st_expect "signals: tee has ended" st_not child_running "$t"
  st_expect "signals: the FIFO is removed" test ! -e "$f"
  # A grandchild that survived a failed expectation is stopped here.
  if [ -n "$gc" ]; then kill -TERM "$gc" 2>/dev/null || true; fi
}

# Whether runner_refusals (host aarch64-test-host) on a config file with the
# text $3 (printf %b), the environment assignments $2 (NAME=VALUE words) and
# the cargo arguments $4... gives a reason with the text $1 ("": none at all).
# shellcheck disable=SC2329 # invoked through st_expect
st_runner() {
  local want="$1" assigns="$2" text="$3" got
  shift 3
  mkdir -p "$ST/cfg"
  printf '%b\n' "$text" > "$ST/cfg/config.toml"
  got="$(
    unset CARGO_BUILD_TARGET CARGO_TARGET_AARCH64_TEST_HOST_RUNNER
    for a in $assigns; do export "${a?}"; done
    runner_refusals aarch64-test-host "$ST/cfg/config.toml" "$@"
  )"
  if [ -z "$want" ]; then [ -z "$got" ]; else case "$got" in *"$want"*) return 0 ;; *) return 1 ;; esac; fi
}

st_runner_rules() {
  local form want w="$ST/walk" got
  while IFS='|' read -r want form; do
    st_expect "runner refused ($want): $form" st_runner "$want" "" "$form"
  done <<'EOF'
sets target.aarch64-test-host.runner|[target.aarch64-test-host]\nrunner = "qemu-aarch64 -cpu max"
sets target.aarch64-test-host.runner|target.aarch64-test-host.runner = ["qemu", "-cpu", "max"]
sets target.aarch64-test-host.runner|[target]\naarch64-test-host = { runner = "qemu" }
sets target.aarch64-test-host.runner|target = { "aarch64-test-host" = { runner = "qemu" } }
sets target.aarch64-test-host.runner|[ target . 'aarch64-test-host' ] # c\n  "runner"="x"
sets target.aarch64-test-host.runner|[build]\nrustflags = [\n  "-C", "x", # ]\n]\n[target.aarch64-test-host]\nrunner = "x"
sets target.aarch64-test-host.runner|a = """x""""\n[target.aarch64-test-host]\nrunner = "x" # "
sets target.aarch64-test-host.runner|target = { aarch64-test-host = { note = '''x'''', runner = "y" } }
cannot be read reliably here: a one-line string across a line break|a = "x\n[target.aarch64-test-host]\nrunner = "y" # "
(a cfg expression|[target.'cfg(target_os = "macos")']\nrunner = "x"
sets build.target|[build]\ntarget = "x86_64-apple-darwin"
sets build.target|build = { jobs = 2, target = ["a", "b"] }
has an include key|include = "more.toml"
cannot be read reliably here: a quoted key with a backslash escape|[target.aarch64-test-host]\n"\\u0072unner" = "x"
cannot be read reliably here: a statement still open|[target.aarch64-test-host\nrunner = "x"
cannot be read reliably here: a statement still open|x = """y\n
cannot be read reliably here: a key the scan cannot read|\0357\0273\0277target.aarch64-test-host.runner = "x"
cannot be read reliably here: a statement that is neither|[target.aarch64-test-host]\nrunner "qemu"
EOF
  while IFS= read -r form; do
    st_expect "runner accepted: $form" st_runner "" "" "$form"
  done <<'EOF'
[target.x86_64-unknown-linux-gnu]\nrunner = "x"
# [target.aarch64-test-host]\n# runner = "x"\nx = 1
[profile.dev]\ndebug = 1\n[alias]\nt = "test --config target.aarch64-test-host.runner=x"
[env]\nrunner = "x"\nCARGO_TARGET_AARCH64_TEST_HOST_RUNNER = "x"
[target.aarch64-test-host]\nlinker = "cc"\nrustflags = ["-C", "runner"]
[build]\njobs = 4\ntarget-dir = "t"
a = """x""""\nb = '''y''''\nc = 1
# \0346\0227\0245\0346\0234\0254\0350\0252\0236 comment\n[profile.dev]\nopt-level = 1 # \0303\0251
EOF
  st_expect "runner refused: the environment's CARGO_TARGET_<host>_RUNNER, also empty" st_runner "CARGO_TARGET_AARCH64_TEST_HOST_RUNNER is set in the environment" "CARGO_TARGET_AARCH64_TEST_HOST_RUNNER=" ""
  st_expect "runner refused: CARGO_BUILD_TARGET in the environment" st_runner "CARGO_BUILD_TARGET is set in the environment" "CARGO_BUILD_TARGET=x86_64-apple-darwin" ""
  st_expect "runner accepted: the environment's runner of another target" st_runner "" "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER=x" ""
  st_expect "runner refused: the cargo argument --config" st_runner "the cargo argument --config (a configuration of its own)" "" "" -p x --config 'target.aarch64-test-host.runner="x"'
  st_expect "runner refused: the cargo argument --target=..." st_runner "the cargo argument --target=x86_64-apple-darwin" "" "" --target=x86_64-apple-darwin
  st_expect "runner accepted: --config after a second -- (an argument of the test binary)" st_runner "" "" "" sme2 -- --config x
  st_expect "runner: the environment variable of a target" test "$(runner_env_name aarch64-apple-darwin.x)" = CARGO_TARGET_AARCH64_APPLE_DARWIN_X_RUNNER
  # Which files: the .cargo/ standing in for the copy's, every parent's, CARGO_HOME's.
  mkdir -p "$w/a/b/c/.cargo" "$w/a/.cargo" "$w/standin/.cargo" "$w/home"
  for f in "$w/a/b/c/.cargo/config.toml" "$w/a/.cargo/config.toml" "$w/standin/.cargo/config" "$w/home/config.toml"; do printf 'x = 1\n' > "$f"; done
  got="$(CARGO_HOME="$w/home" cargo_config_files "$w/a/b/c" "$w/standin")"
  st_expect "config files: the stand-in's .cargo/config, then the parents' (a/.cargo/config.toml)" test "$(printf '%s\n' "$got" | sed -n 1,2p | paste -s -d '|' -)" = "$w/standin/.cargo/config|$w/a/.cargo/config.toml"
  st_expect "config files: the copy's own .cargo/ is not read (the stand-in replaces it)" st_not grep -q -F -e "$w/a/b/c/.cargo" <<EOF
$got
EOF
  st_expect "config files: CARGO_HOME's config.toml comes last" test "$(printf '%s\n' "$got" | sed -n '$p')" = "$w/home/config.toml"
  got="$(CARGO_HOME="$w/a/.cargo" cargo_config_files "$w/a/b/c" "$w/standin")"
  st_expect "config files: a file that is in the walk and in CARGO_HOME is read once" test "$(printf '%s\n' "$got" | grep -c -x -F -e "$w/a/.cargo/config.toml")" = 1
  mkdir -p "$w/real/t" "$w/real/.cargo" "$w/link" && ln -s "$w/real/t" "$w/link/t" && printf 'x = 1\n' > "$w/real/.cargo/config.toml"
  st_expect "config files: through a linked directory (target/), the real one's parents" test "$(cargo_config_files "$w/link/t/sme2-qemu/hardware/src" | grep -c -x -F -e "$w/real/.cargo/config.toml")" = 1
}

st_toolchain_rules() {
  st_expect "toolchain: 1.88.0 is older than rust-version 1.89" version_older 1.88.0 1.89
  st_expect "toolchain: 1.89.0 is not older than 1.89" st_not version_older 1.89.0 1.89
  st_expect "toolchain: 1.89 is not older than 1.89.0" st_not version_older 1.89 1.89.0
  st_expect "toolchain: 1.100.0 is not older than 1.89 (numbers, not text)" st_not version_older 1.100.0 1.89
  st_expect "toolchain: 1.9.0 is older than 1.89" version_older 1.9.0 1.89
  st_expect "toolchain: 1.95.0 is older than 1.95.1" version_older 1.95.0 1.95.1
  st_expect "toolchain: 2.0.0 is not older than 1.89" st_not version_older 2.0.0 1.89
  printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "0.2.4"\nrust-version = "1.89"\n' > "$ST/ws.toml"
  printf '[package]\nname = "x"\nrust-version = "1.80.1" # pinned\n' > "$ST/pkg.toml"
  printf '[workspace.package]\nedition = "2024"\n[dependencies]\nrust-version = "9.9"\n' > "$ST/none.toml"
  printf '[workspace.package]\nrust-version = "latest"\n' > "$ST/bad.toml"
  st_expect "toolchain: the rust-version of [workspace.package]" test "$(workspace_rust_version "$ST/ws.toml")" = 1.89
  st_expect "toolchain: the rust-version of [package]" test "$(workspace_rust_version "$ST/pkg.toml")" = 1.80.1
  st_expect "toolchain: none in another table" test -z "$(workspace_rust_version "$ST/none.toml")"
  st_expect "toolchain: a rust-version that is not a version is none" test -z "$(workspace_rust_version "$ST/bad.toml")"
  st_expect "toolchain: the workspace's Cargo.toml names a rust-version" test -n "$(workspace_rust_version "$ROOT/Cargo.toml")"
}

# ------------------------------------------------------- end to end (--hardware)
# The main flow of --hardware, without a build: this script and this file,
# copied into a skeleton checkout $ST/e2e/ck (its target/ a link to
# $ST/e2e/real/target, as in checkouts whose target/ lives elsewhere), run
# with stand-ins first on PATH: uname (an aarch64 host), rustc (host
# aarch64-st-host, release ST_RUSTC_RELEASE, default 1.95.0), cargo (canned
# test output with a `sme2-gate: ran` marker, exit ST_UNIT_RC; it records
# its arguments in cargo-calls and the gate variables in cargo-env; the
# detection report prints `hw(): Some(StandIn)` and exits ST_REPORT_RC;
# ST_CARGO_SLEEP: the test binary's child sleeps that long first; the
# stage's commands print both binaries' sections, the marker on stderr
# unless ST_STAGE_MARKERS=0, a failed test of sme2_public_path in the build
# ST_STAGE_FAIL, and sleep ST_STAGE_SLEEP first), rsync (makes the copy's
# directory), tar (ST_TAR_SIGNAL: a SIGTERM to this script just before
# the archive) and qemu-aarch64 (its version line, for the container's step
# runner). Like cargo, the stand-in writes its `Running` lines and the
# marker to standard error, the rest to standard output. $ST/e2e/awkfail
# holds an awk whose split of a pass's output (the program with the
# function flush) fails; put it first on PATH to use it.
st_e2e_setup() {
  local e="$ST/e2e" b="$ST/e2e/bin" f real_uname real_tar real_awk real_mv
  real_uname="$(command -v uname)"; real_tar="$(command -v tar)"; real_awk="$(command -v awk)"; real_mv="$(command -v mv)"
  mkdir -p "$e/ck/crates/oxiblas-blas/tools" "$e/ck/crates/oxiblas-blas/src/level3/sme2" "$e/real/target" "$e/real/.cargo" "$e/home" "$b" "$e/awkfail" "$e/mvfail"
  for f in Cargo.toml crates/oxiblas-blas/Cargo.toml crates/oxiblas-blas/src/level3/sme2/detect.rs; do cp "$ROOT/$f" "$e/ck/$f"; done
  cp "$SCRIPT_DIR/$SCRIPT_NAME" "$SCRIPT_DIR/sme2_qemu_selftest.sh" "$SCRIPT_DIR/sme2_qemu_verdicts.sh" "$e/ck/crates/oxiblas-blas/tools/"
  ln -s "$e/real/target" "$e/ck/target"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '#!/bin/sh\nif [ "$1" = -m ]; then if [ "$(%s -s)" = Darwin ]; then echo arm64; else echo aarch64; fi; else exec %s "$@"; fi\n' "$real_uname" "$real_uname" > "$b/uname"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '#!/bin/sh\nr="${ST_RUSTC_RELEASE:-1.95.0}"\nprintf "rustc %%s (stand-in)\\nbinary: rustc\\nhost: aarch64-st-host\\nrelease: %%s\\n" "$r" "$r"\n' > "$b/rustc"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '%s\n' '#!/bin/sh' 'echo "cargo $*" >> "$ST_E2E/cargo-calls"' \
    'echo "${OXIBLAS_TEST_REQUIRE_SME2:-unset} ${OXIBLAS_TEST_EXPECT_NO_SME2:-unset} ${OXIBLAS_TEST_SME2_MARKERS:-unset} ${CARGO_TERM_COLOR:-unset} ${CARGO_TARGET_DIR:+set}" >> "$ST_E2E/cargo-env"' \
    'case " $* " in *" test_sme2_detect_report "*)' \
    '  printf "running 1 test\nhw(): Some(StandIn)\ntest st::test_sme2_detect_report ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"' \
    '  exit "${ST_REPORT_RC:-0}" ;; esac' \
    'case " $* " in *" --test sme2_public_path "*)' \
    '  b=debug; case " $* " in *" --release "*) b=release ;; esac' \
    '  if [ -n "${ST_STAGE_SLEEP:-}" ]; then sleep "$ST_STAGE_SLEEP" & echo "$!" > "$ST_E2E/sleep.pid"; wait; fi' \
    '  printf "     Running tests/gemm_rayon_pool.rs (t/%s/deps/gemm_rayon_pool-0a1b)\n" "$b" >&2' \
    '  printf "\nrunning 1 test\ntest g ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n\n"' \
    '  printf "     Running tests/sme2_public_path.rs (t/%s/deps/sme2_public_path-2c3d)\n" "$b" >&2' \
    '  printf "\nrunning 1 test\n"' \
    '  [ "${ST_STAGE_MARKERS:-1}" = 0 ] || echo "sme2-gate: ran test_sme2_public_path_gemm_f32" >&2' \
    '  if [ "${ST_STAGE_FAIL:-}" = "$b" ]; then printf "test p ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n\nerror: test failed\n"; exit 101; fi' \
    '  printf "test p ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"; exit 0 ;; esac' \
    'if [ -n "${ST_CARGO_SLEEP:-}" ]; then sleep "$ST_CARGO_SLEEP" & echo "$!" > "$ST_E2E/sleep.pid"; wait; fi' \
    'printf "running 1 test\nsme2-gate: ran st::t\ntest st::t ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"' \
    'exit "${ST_UNIT_RC:-0}"' > "$b/cargo"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '#!/bin/sh\nfor d in "$@"; do :; done\nmkdir -p "$d"\n' > "$b/rsync"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '#!/bin/sh\ncase " $* " in *" -czf "*) [ -z "${ST_TAR_SIGNAL:-}" ] || kill -TERM "$PPID" ;; esac\nexec %s "$@"\n' "$real_tar" > "$b/tar"
  printf '#!/bin/sh\necho "qemu-aarch64 version 10.2.0 (stand-in)"\n' > "$b/qemu-aarch64"
  # shellcheck disable=SC2016 # expanded by the stand-ins, not here
  printf '#!/bin/sh\ncase "$*" in *"function flush("*) exit 1 ;; esac\nexec %s "$@"\n' "$real_awk" > "$e/awkfail/awk"
  # A mv that cannot rename a pass's output to its `.stream` copy.
  printf '#!/bin/sh\ncase "$*" in *.log.stream) exit 1 ;; esac\nexec %s "$@"\n' "$real_mv" > "$e/mvfail/mv"
  chmod +x "$b/uname" "$b/rustc" "$b/cargo" "$b/rsync" "$b/tar" "$b/qemu-aarch64" "$e/awkfail/awk" "$e/mvfail/mv"
}

# Runs the skeleton's script with --hardware, the stand-ins, an empty
# CARGO_HOME, no git, the assignments $2... and the script's options after
# a `--`; its output goes to $ST/e2e/$1.out. Prints the exit status (with
# "&": starts it, sets st_pid).
st_e2e() {
  local name="$1" rc=0 bg=0 assigns=() opts=()
  shift
  if [ "${1:-}" = "&" ]; then bg=1; shift; fi
  while [ "$#" -gt 0 ]; do
    if [ "$1" = -- ]; then shift; opts=("$@"); break; fi
    assigns+=("$1"); shift
  done
  set -- env PATH="$ST/e2e/bin:$PATH" CARGO_HOME="$ST/e2e/home" GIT_DIR="$ST/e2e/no-git" ST_E2E="$ST/e2e" \
    ${assigns[@]+"${assigns[@]}"} "$BASH" crates/oxiblas-blas/tools/sme2_qemu.sh --hardware ${opts[@]+"${opts[@]}"}
  if [ "$bg" = 1 ]; then
    (unset CARGO_BUILD_TARGET CARGO_TARGET_AARCH64_ST_HOST_RUNNER; cd "$ST/e2e/ck" && exec "$@") > "$ST/e2e/$name.out" 2>&1 &
    st_pid=$!
    return 0
  fi
  (unset CARGO_BUILD_TARGET CARGO_TARGET_AARCH64_ST_HOST_RUNNER; cd "$ST/e2e/ck" && exec "$@") > "$ST/e2e/$name.out" 2>&1 || rc=$?
  echo "$rc"
}

st_e2e_rules() {
  local e="$ST/e2e" o="$ST/e2e/real/target/sme2-qemu" rc got sp t0 i st_pid="" vf f
  st_e2e_setup
  # Cargo config files above this checkout that --hardware refuses would stop every run here.
  got="$(unset CARGO_BUILD_TARGET; CARGO_HOME="$e/home" runner_refusals aarch64-st-host "$(CARGO_HOME="$e/home" cargo_config_files "$e/ck/target/sme2-qemu/hardware/src" "$e/ck")")"
  if [ -n "$got" ]; then
    printf 'ok      end to end: not run, --hardware refuses a cargo config file above this checkout: %s\n' "$(printf '%s\n' "$got" | sed -n 1p)"
    return 0
  fi
  rm -f "$e/cargo-calls" "$e/cargo-env"
  rc="$(st_e2e pass)"
  st_expect "end to end: every cargo command of the hardware pass (unit, stage, report) gets its gate variables" \
    test "$(grep -c '' "$e/cargo-env" 2>/dev/null)|$(sort -u "$e/cargo-env" 2>/dev/null)" = "4|1 0 1 never set"
  st_expect "end to end: with the stand-ins: overall PASS, exit 0" test "$rc|$(sed -n 's/^overall *: //p' "$o/summary.txt" 2>/dev/null)" = "0|PASS"
  st_expect "end to end: the summary says target runner: none" grep -q '^target runner: none (cargo runs the test binaries itself' "$o/summary.txt"
  st_expect "end to end: the archive is written, the run lock released" test "$([ -f "$o/sme2-qemu-results.tar.gz" ] && echo a)$([ -e "$o/.run.lock" ] && echo L)" = a
  st_expect "end to end: the unit command, then the stage's release and debug commands (no filter, no --target), then the report" \
    test "$(paste -s -d '|' - < "$e/cargo-calls" 2>/dev/null)" = "cargo test -p oxiblas-blas --features sme2,parallel --lib --release sme2|cargo test -p oxiblas-blas --features sme2,parallel --release --test sme2_public_path --test gemm_rayon_pool --no-fail-fast|cargo test -p oxiblas-blas --features sme2,parallel --test sme2_public_path --test gemm_rayon_pool --no-fail-fast|cargo test -p oxiblas-blas --features sme2,parallel --lib --release test_sme2_detect_report -- --nocapture"
  st_expect "end to end: the stage's logs hold their builds' output, the pass log none of it" \
    test "$(grep -c 'deps/sme2_public_path' "$o/hardware-public-release.log" "$o/hardware-public-debug.log" "$o/hardware.log" 2>/dev/null | paste -s -d '|' -)" = "$o/hardware-public-release.log:1|$o/hardware-public-debug.log:1|$o/hardware.log:0"
  st_expect "end to end: the summary's stage lines (commands, counts, verdicts)" \
    test "$(grep -c -e '^public path  : cargo test -p oxiblas-blas --features sme2,parallel --release --test sme2_public_path' -e '^  public release: log hardware-public-release.log; sme2_public_path: 1 passed, 0 failed, sme2-gate 1 ran / 0 skipped / 0 declined; gemm_rayon_pool: 1 passed, 0 failed; cargo exit 0 after [0-9]*s; PASS$' -e '^  public debug  : log hardware-public-debug.log; sme2_public_path: 1 passed' "$o/summary.txt")" = 3
  st_expect "end to end: the archive holds the stage's logs" \
    test "$(tar -tzf "$o/sme2-qemu-results.tar.gz" 2>/dev/null | sed 's|^\./||' | sort | paste -s -d ' ' -)" = "detect-report.log hardware-public-debug.log hardware-public-release.log hardware.log summary.txt"
  rc="$(st_e2e stagefail ST_STAGE_FAIL=debug)"
  st_expect "end to end: a failed test in the stage's debug build: FAIL, exit 1, the reason names the build and the binary" \
    test "$rc|$(grep -c '^  verdict   : FAIL (public path, debug build: sme2_public_path: 1 failed; cargo exited with 101)$' "$o/summary.txt" 2>/dev/null)" = "1|1"
  rc="$(st_e2e nomarker ST_STAGE_MARKERS=0)"
  st_expect "end to end: no ran line from the stage: FAIL in both builds, exit 1" \
    test "$rc|$(grep -o 'sme2_public_path wrote no sme2-gate: ran line' "$o/summary.txt" 2>/dev/null | grep -c '')" = "1|4"
  rm -f "$e/cargo-calls"
  rc="$(st_e2e unitfail ST_UNIT_RC=101)"
  st_expect "end to end: a failing unit command: FAIL, the stage still runs in both builds" \
    test "$rc|$(grep -c -e '--test sme2_public_path' "$e/cargo-calls" 2>/dev/null)|$(grep -c '^  public release: .*; PASS$' "$o/summary.txt" 2>/dev/null)" = "1|2|1"
  rm -f "$e/cargo-calls"
  rc="$(st_e2e nostage -- --no-public-path)"
  st_expect "end to end: --no-public-path: PASS, exit 0, no stage command, no stage log, the summary says so" \
    test "$rc|$(grep -c -e '--test' "$e/cargo-calls" 2>/dev/null)|$(set -- "$o"/*public*; if [ -e "$1" ]; then echo "$#"; else echo 0; fi)|$(grep -c '^public path  : left out (--no-public-path)$' "$o/summary.txt")" = "0|0|0|1"
  rm -f "$e/cargo-calls"
  rc="$(st_e2e old ST_RUSTC_RELEASE=1.88.0)"
  st_expect "end to end: rustc 1.88.0, older than rust-version: exit 2, cargo never runs" test "$rc|$([ -e "$e/cargo-calls" ] && echo cargo)" = "2|"
  st_expect "end to end: the toolchain refusal names both versions" grep -q -F "is rustc 1.88.0, older than the workspace's rust-version" "$e/old.out"
  rc="$(st_e2e envrunner CARGO_TARGET_AARCH64_ST_HOST_RUNNER=qemu-aarch64)"
  st_expect "end to end: CARGO_TARGET_<host>_RUNNER set: exit 2 naming it" test "$rc|$(grep -c 'CARGO_TARGET_AARCH64_ST_HOST_RUNNER is set in the environment' "$e/envrunner.out")" = "2|1"
  printf '[target.aarch64-st-host]\nrunner = "qemu-aarch64"\n' > "$e/real/.cargo/config.toml"
  rc="$(st_e2e linked)"
  rm -f "$e/real/.cargo/config.toml"
  st_expect "end to end: a runner above the real target/ (a link): exit 2 naming the file" test "$rc|$(grep -c -F "$e/real/.cargo/config.toml sets target.aarch64-st-host.runner" "$e/linked.out")" = "2|1"
  mkdir -p "$o/.run.lock" && printf '%s\n%s\n' "$$" "$(process_start "$$")" > "$o/.run.lock/pid"
  rc="$(st_e2e locked)"
  st_expect "end to end: a live run's lock: exit 2, its lock left alone" test "$rc|$(sed -n 1p "$o/.run.lock/pid" 2>/dev/null)" = "2|$$"
  rm -rf "$o/.run.lock"
  rc="$(st_e2e report ST_REPORT_RC=101)"
  st_expect "end to end: the detection report read Some(...), then its cargo failed: FAIL, exit 1" test "$rc|$(grep -c 'the detection report read hw(): Some(StandIn) but did not end' "$o/summary.txt" 2>/dev/null)" = "1|1"
  rc="$(st_e2e tarsignal ST_TAR_SIGNAL=1)"
  st_expect "end to end: SIGTERM while the archive is written is ignored: exit 0, archive written" test "$rc|$([ -f "$o/sme2-qemu-results.tar.gz" ] && echo archive)" = "0|archive"
  rc="$(st_e2e splitfail PATH="$e/awkfail:$e/bin:$PATH")"
  st_expect "end to end: the split of the pass's output fails: FAIL, exit 1, the complete output archived" \
    test "$rc|$(grep -c "the split of the pass's output into its logs failed (the complete output is in hardware.log.stream)" "$o/summary.txt" 2>/dev/null)|$(tar -tzf "$o/sme2-qemu-results.tar.gz" 2>/dev/null | grep -c 'hardware\.log\.stream$')" = "1|1|1"
  rc="$(st_e2e renamefail PATH="$e/mvfail:$e/bin:$PATH")"
  st_expect "end to end: the pass's output cannot be renamed for the split: FAIL, exit 1, the reason names the log, which holds the complete output and is archived" \
    test "$rc|$(grep -c "the split of the pass's output into its logs failed (the complete output is in hardware.log)" "$o/summary.txt" 2>/dev/null)|$(tar -tzf "$o/sme2-qemu-results.tar.gz" 2>/dev/null | grep -c 'hardware\.log$')|$(tar -tzf "$o/sme2-qemu-results.tar.gz" 2>/dev/null | grep -c 'hardware\.log\.stream$')|$(grep -c '^== step public-debug \[.*\]: cargo exited with 0' "$o/hardware.log" 2>/dev/null)" = "1|1|1|0|1"
  # SIGTERM while the test binary (its child sleeping) runs.
  rm -f "$e/sleep.pid"
  st_e2e signal "&" ST_CARGO_SLEEP=60
  i=0; while [ ! -s "$e/sleep.pid" ] && [ "$i" -lt 300 ]; do sleep 0.1; i=$((i + 1)); done
  sp="$(cat "$e/sleep.pid" 2>/dev/null || true)"; t0="$SECONDS"
  kill -TERM "$st_pid" 2>/dev/null || true
  i=0; while kill -0 "$st_pid" 2>/dev/null && [ "$i" -lt 250 ]; do sleep 0.1; i=$((i + 1)); done
  kill -0 "$st_pid" 2>/dev/null && kill -KILL "$st_pid" 2>/dev/null
  rc=0; wait "$st_pid" || rc=$?
  st_expect "end to end: SIGTERM in the tests: exit 143 within 5 s" test "$rc|$((SECONDS - t0 <= 5))" = "143|1"
  st_expect "end to end: SIGTERM: the test binary's child is gone" st_gone "$sp"
  st_expect "end to end: SIGTERM: the log ends with the interrupted line" test "$(tail -n 1 "$o/hardware.log" 2>/dev/null)" = "== interrupted by SIGTERM: the pass was stopped; no summary and no archive were written"
  st_expect "end to end: SIGTERM: no summary, archive, FIFO or run lock is left" \
    test "$(for f in summary.txt sme2-qemu-results.tar.gz .hardware.fifo .run.lock hardware.log.stream; do [ ! -e "$o/$f" ] || printf '%s ' "$f"; done)" = ""
  if [ -n "$sp" ]; then kill -TERM "$sp" 2>/dev/null || true; fi
  # SIGTERM while the stage's release build runs.
  rm -f "$e/sleep.pid"
  st_e2e stagesignal "&" ST_STAGE_SLEEP=60
  i=0; while [ ! -s "$e/sleep.pid" ] && [ "$i" -lt 300 ]; do sleep 0.1; i=$((i + 1)); done
  sp="$(cat "$e/sleep.pid" 2>/dev/null || true)"
  kill -TERM "$st_pid" 2>/dev/null || true
  i=0; while kill -0 "$st_pid" 2>/dev/null && [ "$i" -lt 250 ]; do sleep 0.1; i=$((i + 1)); done
  kill -0 "$st_pid" 2>/dev/null && kill -KILL "$st_pid" 2>/dev/null
  rc=0; wait "$st_pid" || rc=$?
  st_expect "end to end: SIGTERM in the stage: exit 143, its child gone" test "$rc|$(st_gone "$sp" && echo gone)" = "143|gone"
  st_expect "end to end: SIGTERM in the stage: the unit log is complete, every log of the pass ends with the interrupted line" \
    test "$(grep -c '^== step unit \[.*\]: cargo exited with 0' "$o/hardware.log")|$(for f in hardware.log hardware-public-release.log hardware-public-debug.log; do tail -n 1 "$o/$f" 2>/dev/null | grep -c '^== interrupted by SIGTERM'; done | tr -d '\n')" = "1|111"
  st_expect "end to end: SIGTERM in the stage: no summary, archive or collected output is left" \
    test "$(for f in summary.txt sme2-qemu-results.tar.gz hardware.log.stream .run.lock; do [ ! -e "$o/$f" ] || printf '%s ' "$f"; done)" = ""
  if [ -n "$sp" ]; then kill -TERM "$sp" 2>/dev/null || true; fi
  # SIGTERM while the stage's release build runs, and the handler's split
  # fails (its awk): the handler still ends with the signal's status.
  rm -f "$e/sleep.pid"
  st_e2e stagesplitsignal "&" ST_STAGE_SLEEP=60 PATH="$e/awkfail:$e/bin:$PATH"
  i=0; while [ ! -s "$e/sleep.pid" ] && [ "$i" -lt 300 ]; do sleep 0.1; i=$((i + 1)); done
  sp="$(cat "$e/sleep.pid" 2>/dev/null || true)"
  kill -TERM "$st_pid" 2>/dev/null || true
  i=0; while kill -0 "$st_pid" 2>/dev/null && [ "$i" -lt 250 ]; do sleep 0.1; i=$((i + 1)); done
  kill -0 "$st_pid" 2>/dev/null && kill -KILL "$st_pid" 2>/dev/null
  rc=0; wait "$st_pid" || rc=$?
  st_expect "end to end: SIGTERM in the stage while the split fails: exit 143, its child gone, the complete output kept, it and every log end with the interrupted line that names it" \
    test "$rc|$(st_gone "$sp" && echo gone)|$(for f in hardware.log hardware-public-release.log hardware-public-debug.log hardware.log.stream; do tail -n 1 "$o/$f" 2>/dev/null | grep -c '^== interrupted by SIGTERM: .*the complete output is in hardware\.log\.stream$'; done | tr -d '\n')|$(grep -c '^== step unit \[.*\]: cargo exited with 0' "$o/hardware.log.stream" 2>/dev/null)|$([ ! -e "$o/.run.lock" ] && echo unlocked)|$(grep -c 'interrupted by SIGTERM: .*the complete output is in hardware\.log\.stream' "$e/stagesplitsignal.out" 2>/dev/null)" = "143|gone|1111|1|unlocked|1"
  if [ -n "$sp" ]; then kill -TERM "$sp" 2>/dev/null || true; fi
  # sme2_qemu_verdicts.sh missing, cut off before the stage's functions (an
  # older copy), or of another version: exit 2 before anything is written.
  vf="$e/ck/crates/oxiblas-blas/tools/sme2_qemu_verdicts.sh"
  mv -f "$vf" "$e/verdicts.keep"
  rm -rf "$o"
  rc="$(st_e2e noverdicts)"
  st_expect "end to end: no sme2_qemu_verdicts.sh: exit 2 with its message, nothing written" \
    test "$rc|$(grep -c 'error: the file crates/oxiblas-blas/tools/sme2_qemu_verdicts.sh is missing' "$e/noverdicts.out")|$([ -e "$o" ] && echo written)" = "2|1|"
  sed '/^# -* steps and the stage$/,$d' "$e/verdicts.keep" > "$vf"
  rc="$(st_e2e staleverdicts)"
  st_expect "end to end: a sme2_qemu_verdicts.sh without the stage's functions: exit 2 naming the first one missing, nothing written" \
    test "$rc|$(grep -c 'does not match this script: it defines no function judge_stage' "$e/staleverdicts.out")|$([ -e "$o" ] && echo written)" = "2|1|"
  sed 's/^SQ_VERDICTS_VERSION=.*/SQ_VERDICTS_VERSION=1/' "$e/verdicts.keep" > "$vf"
  rc="$(st_e2e oldverdicts)"
  st_expect "end to end: a sme2_qemu_verdicts.sh of another version: exit 2 naming both versions, nothing written" \
    test "$rc|$(grep -c 'is version 1, this script needs version 2' "$e/oldverdicts.out")|$([ -e "$o" ] && echo written)" = "2|1|"
  mv -f "$e/verdicts.keep" "$vf"
  # The container's step runner (INNER) with the stand-ins, its standard
  # output and standard error caught apart as Docker passes them on: cargo's
  # `Running` lines (standard error) must stay in order with the test
  # results (standard output), whichever of the two is read first.
  mkdir -p "$e/inner"
  rc=0
  (cd "$e/inner" && exec env PATH="$e/bin:$PATH" ST_E2E="$e/inner" SQ_SRC="$e/ck" SQ_WORK="$e/inner/src" SQ_CPU=max \
    SQ_TOOLCHAIN=1.95.0 SQ_TOKEN=T9 CARGO_TARGET_DIR="$e/inner/target" \
    "$BASH" -c "$INNER" _ public-release 3 --release --test sme2_public_path) > "$e/inner/out" 2> "$e/inner/err" || rc=$?
  st_expect "end to end: the step runner in the container: exit 0, nothing on its standard error (joined to standard output)" \
    test "$rc|$(grep -c '' "$e/inner/err")|$(grep -c '^     Running tests/' "$e/inner/out")" = "0|0|2"
  for f in out-err err-out; do
    case "$f" in out-err) cat "$e/inner/out" "$e/inner/err" ;; *) cat "$e/inner/err" "$e/inner/out" ;; esac > "$e/inner/$f.log"
    split_pass_log "$e/inner/$f.log" T9 1 "$e/inner/$f-r.log" "$e/inner/$f-d.log"
  done
  st_expect "end to end: the step runner's lines go to their test binaries whichever stream is read first: PASS both ways" \
    test "$(st_stage_counts positive "$e/inner/out-err-r.log")|$(st_stage positive release 0 "$e/inner/out-err-r.log" PASS "" && st_stage positive release 0 "$e/inner/err-out-r.log" PASS "" && echo PASS)" = "sme2_public_path: 1 passed, 0 failed, sme2-gate 1 ran / 0 skipped / 0 declined; gemm_rayon_pool: 1 passed, 0 failed; cargo exit 0|1 0 0|PASS"
}

run_self_test() {
  local results bad total
  ST="$ROOT/target/sme2-qemu-selftest.$$"
  mkdir -p "$ST" || die 2 "cannot create $ST (is target/ writable?)"; ST="$(cd "$ST" && pwd -P)"
  trap 'rm -rf "$ST"' EXIT
  results="$ST/results.txt"
  printf '%s --self-test: canned pass logs, no Docker, no QEMU, no cargo; scratch target/%s\n' "$SCRIPT_NAME" "${ST##*/}"
  # The signal rules start real processes; bash may report the stopped ones
  # on stderr, which is left out here.
  if ! { st_qemu_rules; st_stage_rules; st_hardware_rules; st_lock_rules; st_signal_rules 2>/dev/null; st_runner_rules; st_toolchain_rules; st_e2e_rules 2>/dev/null; } | tee "$results"; then
    printf 'self-test: FAIL (it stopped early; see the lines above)\n'
    exit 1
  fi
  total="$(grep -c '' "$results" || true)"; bad="$(grep -c '^NOT OK' "$results" || true)"
  if [ "$bad" = 0 ] && [ "$total" -gt 0 ]; then printf 'self-test: PASS (%s expectations)\n' "$total"; exit 0; fi
  printf 'self-test: FAIL (%s of %s expectations do not hold)\n' "$bad" "$total"
  exit 1
}
