# shellcheck shell=bash disable=SC2034,SC2153
# x86_gemm_runbook_selftest.sh -- the self-test of x86_gemm_runbook.sh.
#
# Not a program of its own: x86_gemm_runbook.sh sources this file from its
# own directory for --self-test only, after it has defined its functions and
# found the checkout, and then calls run_self_test, which exits. Run it as
#   crates/oxiblas-blas/tools/x86_gemm_runbook.sh --self-test
# The functions here use the functions, constants and paths of the runbook
# (SC2034 and SC2153 are off above: the variables this file sets are read by
# the runbook's functions, which shellcheck cannot see from this file alone).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  printf '%s: this file is sourced by x86_gemm_runbook.sh; run x86_gemm_runbook.sh --self-test\n' "${0##*/}" >&2
  exit 2
fi

# ----------------------------------------------------------------- self-test
# Every expectation prints one line; the count of "NOT OK" lines decides.
st_expect() {
  local what="$1"; shift
  if "$@"; then printf 'ok      %s\n' "$what"; else printf 'NOT OK  %s\n' "$what"; fi
}

# Whether the units file $1 has unit $2 with verdict $3 and text $4 in its
# details.
st_unit() {
  [ -f "$1" ] && awk -F'\t' -v id="$2" -v v="$3" -v t="${4:-}" \
    '$1 == id && $2 == v && index($5, t) { found = 1 } END { exit !found }' "$1"
}

st_fake_cargo() {
  cat <<'EOF'
#!/bin/sh
# Stand-in for cargo in the self-test of x86_gemm_runbook.sh: one A/B group
# process that appends lines like the harness (settings.rs, output.rs). The
# group is the last argument (the test's full path after --exact).
for a in "$@"; do group="${a##*::}"; done
case " $* " in *" --no-run "*) echo "fake build"; exit 0 ;; esac
# What a cargo [env] table fills in when a variable is unset:
: "${OXIBLAS_AB_ROUNDS=4}" "${OXIBLAS_AB_MIN_SAMPLE_MS=1}"
grade=decision; debug=false; tier=Avx2Fma; quick=false
if [ -n "${OXIBLAS_AB_QUICK:-}" ]; then grade=smoke; quick=true; fi
if [ "$OXIBLAS_AB_ROUNDS" -lt 21 ] || [ "$OXIBLAS_AB_MIN_SAMPLE_MS" -lt 2 ]; then grade=smoke; fi
if [ "$FAKE_MODE" = debug ]; then grade=smoke; debug=true; fi
if [ "$FAKE_MODE" = sse ]; then tier=Sse42; fi
ok='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
if [ "$FAKE_MODE" = silent ] && [ "$group" = ab_seq_thin ]; then echo "$ok"; exit 0; fi
id="$$-1-0"; at="$FAKE_MODE/${OXIBLAS_AB_TAG:-}/$group"; tag="${OXIBLAS_AB_TAG:-}"
[ "$at" != notag/pass2/ab_seq_b_cap ] || tag=other
[ "$FAKE_MODE" = noheader ] || printf 'AB#\tABV\trun_id\tgroup\trun\tthreads\ttier\tgrade\tcases\tremeasured\tinvalid\tmax_abs_spread\tverdicts\n' >> "$OXIBLAS_AB_OUT"
printf 'AB#\tSET\t%s\tgroup=%s\trun=r\ttier=%s\trounds_requested=%s\tmin_sample_ms=%s\tquick=%s\tdebug_build=%s\ttag=%s\tgrade=%s\n' \
  "$id" "$group" "$tier" "$OXIBLAS_AB_ROUNDS" "$OXIBLAS_AB_MIN_SAMPLE_MS" "$quick" "$debug" "$tag" "$grade" >> "$OXIBLAS_AB_OUT"
printf 'AB\t%s\t%s\tcase\nABR\t%s\t%s\tcase\n' "$id" "$group" "$id" "$group" >> "$OXIBLAS_AB_OUT"
# The verdicts of the candidates x and y (y of ab_seq_thin is informational):
# SMOKE below decision grade, else ADOPT unless the mode changes one.
x=ADOPT; y=ADOPT; invalid=0; py=n/a
if [ "$grade" = smoke ]; then x=SMOKE; y=SMOKE; fi
if [ "$group" = ab_seq_thin ]; then py=informational; fi
case "$at" in
  invalid/pass1/ab_par_thin) invalid=1; x=INVALID; y=INVALID ;;
  disagree/pass2/ab_seq_thin) x=KEEP_LEGACY ;;
  informational/pass2/ab_seq_thin) y=KEEP_LEGACY ;;
  badword/pass1/ab_seq_thin) x=MAYBE ;;
  # Synthetic: the harness gives INVALID exactly when a case's control is
  # invalid (stats.rs, runner.rs); these two separate the signs.
  invalidword/pass2/ab_seq_thin) x=INVALID ;;
  invalidcase/pass1/ab_par_thin) invalid=1 ;;
esac
v="x:$x,geo=1.0700,min_t=1.0100,min_e=0.9900,nt=2,ne=1,noop=0,partial=n/a"
[ "$at" = dropcand/pass2/ab_seq_b_cap ] || v="$v;y:$y,geo=1.0600,min_t=1.0000,min_e=0.9800,nt=2,ne=1,noop=0,partial=$py"
[ "$at" != repeat/pass1/ab_seq_thin ] || v="$v;x:$x,geo=1.0700,min_t=1.0100,min_e=0.9900,nt=2,ne=1,noop=0,partial=n/a"
if [ "$at" != noabv/pass2/ab_par_b_cap ] && [ "$FAKE_MODE" != noabvall ]; then
  printf 'ABV\t%s\t%s\tr\t1\t%s\t%s\t3\t0\t%s\t0.0040\t%s\n' "$id" "$group" "$tier" "$grade" "$invalid" "$v" >> "$OXIBLAS_AB_OUT"
fi
if [ "$FAKE_MODE" != noend ] || [ "$group" != ab_par_thin ]; then
  printf 'AB#\tEND\t%s\tgroup=%s\tcases=1\tinvalid_cases=%s\n' "$id" "$group" "$invalid" >> "$OXIBLAS_AB_OUT"
fi
echo "$ok"
EOF
}

st_fake_taskset() {
  cat <<'EOF'
#!/bin/sh
# Stand-in for taskset in the self-test of x86_gemm_runbook.sh: drops
# "-c <cpus>" and runs the rest.
[ "$1" = -c ] && shift 2
exec "$@"
EOF
}

# A mini step 5 with the stand-in cargo; its units file is
# $ST/ab-<name>/.units. With "final", taskset is found (the stand-in) and
# finalize runs after step 5 as at the end of a run (limited by the given
# parts, or full): the units are kept in units.txt, finalize's exit status
# in rc, and its summary.txt is beside them.
#   st_ab_case <name> <quick> <allow_no_avx2> <fake mode> [final [<limited parts>]]
st_ab_case() {
  local d="$ST/ab-$1"
  mkdir -p "$d/logs"
  (
    OUT="$d"; LOG_DIR="$d/logs"; UNITS="$d/.units"; AB_OUT="$d/ab.tsv"; PATH="$ST/bin:$PATH"
    quick="$2"; allow_no_avx2="$3"; dry_run=0; have_taskset=0; PIN_CPU=2; AB_THREADS=2,4
    host_arch="$(uname -m)"; require_avx2_text=1; LEGACY_ENV=(env "RUSTFLAGS=" "CARGO_TARGET_DIR=$d/target")
    FAKE_MODE="$4"; export FAKE_MODE
    if [ "${5:-}" = final ]; then
      have_taskset=1; limited_parts="${6:-}"; out_ready=1; machine_ok=1; machine_problem=""
      CRITERION_DIR="$d/criterion"; ARCHIVE="$d/results.tar.gz"; prev_rel=""; removed_env=""
      N_ONLINE=8; N_PHYSICAL=4; OPENBLAS_PATH="$d/no-openblas"; TINY_OUT="$d/tiny-call"
    fi
    : > "$UNITS"
    step_ab
    # shellcheck disable=SC2154 # final_rc is the runbook's, set by finalize
    if [ "${5:-}" = final ]; then cp "$UNITS" "$d/units.txt"; finalize ""; printf '%s\n' "$final_rc" > "$d/rc"; fi
  ) > "$d/out.txt" 2>&1 || true
}

# Whether file $1 contains the text $2.
st_has() {
  grep -q -F -e "$2" "$1" 2>/dev/null
}

# Whether the run in directory $1 ended with exit status $2 and the overall
# line "overall      : $3" in its summary.txt.
st_overall() {
  [ "$(cat "$1/rc" 2>/dev/null)" = "$2" ] && grep -q -x -F -e "overall      : $3" "$1/summary.txt" 2>/dev/null
}

# Whether file $1 exists and does not contain the text $2.
st_lacks() {
  [ -f "$1" ] && ! grep -q -F -e "$2" "$1"
}

# Whether the A/B decision table in file $1 has a row of group $2 and
# candidate $3 whose other columns, from the fourth on and joined by single
# spaces, read $4.
st_row() {
  awk -v g="$2" -v c="$3" -v t="$4" '
    $1 == g && $3 == c { r = $4; for (i = 5; i <= NF; i++) r = r " " $i; if (r == t) found = 1 }
    END { exit !found }' "$1" 2>/dev/null
}

# Step 5 as a decision, end to end (stand-ins for cargo and taskset, then
# finalize): the verdict unit, the overall line and exit status, the missing
# parts and the decision table of summary.txt.
st_verdicts() {
  local d g n=0 inc="INCOMPLETE (nothing failed, but parts the decision needs are missing; see below)"
  # The verdicts per pass: two candidates in each group that runs here.
  for g in $AB_SEQ_GROUPS $AB_PAR_GROUPS; do
    case " $AB_X86_ONLY_GROUPS " in *" $g "*) [ "$(uname -m)" = x86_64 ] || continue ;; esac
    n=$((n + 2))
  done
  st_fake_taskset > "$ST/bin/taskset" && chmod +x "$ST/bin/taskset"
  st_ab_case v-agree 0 0 decision final; st_ab_case v-invalid 0 0 invalid final
  st_ab_case v-disagree 0 0 disagree final; st_ab_case v-dropcand 0 0 dropcand final
  st_ab_case v-noabv 0 0 noabv final; st_ab_case v-info 0 0 informational final
  st_ab_case v-badword 0 0 badword final; st_ab_case v-limited 0 0 invalid final "--only 5"
  st_expect "step 5 verdicts, earlier cases: decision grade, both passes agree: PASS" \
    st_unit "$ST/ab-decision/.units" 5-ab-verdicts PASS "0 disagree, 0 INVALID, 0 in one pass only"
  st_expect "step 5 verdicts, earlier cases: --quick: INFO, smoke verdicts not judged" \
    st_unit "$ST/ab-quick/.units" 5-ab-verdicts INFO "--quick: smoke verdicts are not judged; $n candidates: $n agree"
  st_expect "step 5 verdicts, earlier cases: SMOKE in a full run: WARN" \
    st_unit "$ST/ab-smoke/.units" 5-ab-verdicts WARN "ADOPT 0, KEEP_LEGACY 0, NO_TARGETS 0, INVALID 0, SMOKE $n)"
  d="$ST/ab-v-agree"
  st_expect "(a) both passes ADOPT and agree: 5-ab-verdicts PASS" \
    st_unit "$d/units.txt" 5-ab-verdicts PASS "$n candidates: $n agree, 0 disagree, 0 INVALID, 0 in one pass only"
  st_expect "(a) both passes ADOPT and agree: overall PASS, exit 0" st_overall "$d" 0 PASS
  st_expect "(a) both passes ADOPT and agree: the table row reads ADOPT twice and agree" \
    st_row "$d/summary.txt" ab_par_thin x "ADOPT 1.0700 n/a ADOPT 1.0700 n/a agree"
  st_expect "(a) both passes ADOPT and agree: the table counts the verdicts of a pass" st_has "$d/summary.txt" \
    "  pass2: $n verdicts (ADOPT $n, KEEP_LEGACY 0, NO_TARGETS 0, INVALID 0, SMOKE 0), 0 invalid cases in 0 of $((n / 2)) group runs"
  st_expect "(a) both passes ADOPT and agree: nothing missing" st_lacks "$d/summary.txt" "missing:"
  d="$ST/ab-v-invalid"
  st_expect "(b) one INVALID verdict: 5-ab-verdicts WARN" st_unit "$d/units.txt" 5-ab-verdicts WARN "2 INVALID"
  st_expect "(b) one INVALID verdict: overall INCOMPLETE, exit 3" st_overall "$d" 3 "$inc"
  st_expect "(b) one INVALID verdict: missing part, the machine was not quiet" st_has "$d/summary.txt" \
    "  - step 5: 2 of $((2 * n)) A/B verdicts are INVALID (pass1 2 of $n, pass2 0 of $n) and 1 cases had the control outside the limit: the machine was not quiet"
  st_expect "(b) one INVALID verdict: missing part, run step 5 again and send both archives" st_has "$d/summary.txt" \
    "run step 5 again (crates/oxiblas-blas/tools/x86_gemm_runbook.sh --only 5, about 2.5 hours); that run moves this one to target/x86-runbook-prev/, so send back both archives"
  st_expect "(b) one INVALID verdict: the table row reads INVALID" \
    st_row "$d/summary.txt" ab_par_thin y "INVALID 1.0600 n/a ADOPT 1.0600 n/a INVALID"
  d="$ST/ab-v-disagree"
  st_expect "(c) the passes disagree: 5-ab-verdicts WARN" st_unit "$d/units.txt" 5-ab-verdicts WARN "1 disagree"
  st_expect "(c) the passes disagree: overall INCOMPLETE, exit 3" st_overall "$d" 3 "$inc"
  st_expect "(c) the passes disagree: missing part names the candidate" st_has "$d/summary.txt" \
    "  - step 5: the two passes disagree on ab_seq_thin r x: pass1 ADOPT, pass2 KEEP_LEGACY"
  st_expect "(c) the passes disagree: the table row reads DISAGREE" \
    st_row "$d/summary.txt" ab_seq_thin x "ADOPT 1.0700 n/a KEEP_LEGACY 1.0700 n/a DISAGREE"
  d="$ST/ab-v-dropcand"
  st_expect "(d) a verdict missing in one pass: 5-ab-verdicts WARN" st_unit "$d/units.txt" 5-ab-verdicts WARN "1 in one pass only"
  st_expect "(d) a verdict missing in one pass: overall INCOMPLETE, exit 3" st_overall "$d" 3 "$inc"
  st_expect "(d) a verdict missing in one pass: missing part names it" st_has "$d/summary.txt" \
    "  - step 5: ab_seq_b_cap r y has a verdict in pass1 only"
  st_expect "(d) a verdict missing in one pass: the table row reads MISSING" \
    st_row "$d/summary.txt" ab_seq_b_cap y "ADOPT 1.0600 n/a - - - MISSING"
  d="$ST/ab-v-noabv"
  st_expect "(e) a group run without its ABV line: that group FAIL" \
    st_unit "$d/units.txt" 5-ab-pass2-ab_par_b_cap FAIL "SET 1 and ABV 0 lines in this process: a group run wrote no verdict"
  st_expect "(e) a group run without its ABV line: the aggregate FAIL" \
    st_unit "$d/units.txt" 5-ab-pass2-par FAIL "SET and ABV lines differ"
  st_expect "(e) a group run without its ABV line: overall FAIL, exit 1" st_overall "$d" 1 FAIL
  st_expect "(e) a group run without its ABV line: the table row reads MISSING" \
    st_row "$d/summary.txt" ab_par_b_cap x "ADOPT 1.0700 n/a - - - MISSING"
  d="$ST/ab-v-info"
  st_expect "(f) an informational candidate disagrees: no missing part, overall PASS, exit 0" st_overall "$d" 0 PASS
  st_expect "(f) an informational candidate disagrees: no missing-part line for it" st_lacks "$d/summary.txt" \
    "step 5: the two passes disagree on ab_seq_thin r y"
  st_expect "(f) an informational candidate disagrees: 5-ab-verdicts PASS and names it" st_unit "$d/units.txt" 5-ab-verdicts PASS \
    "informational candidates whose passes disagree (never adopted, not judged): ab_seq_thin r y: pass1 ADOPT, pass2 KEEP_LEGACY"
  st_expect "(f) an informational candidate disagrees: the table row labels it" \
    st_row "$d/summary.txt" ab_seq_thin y "ADOPT 1.0600 informational KEEP_LEGACY 1.0600 informational DISAGREE (informational)"
  d="$ST/ab-v-badword"
  st_expect "(g) an unknown decision word: 5-ab-verdicts FAIL" st_unit "$d/units.txt" 5-ab-verdicts FAIL "other words 1"
  st_expect "(g) an unknown decision word: overall FAIL, exit 1" st_overall "$d" 1 FAIL
  d="$ST/ab-v-limited"
  st_expect "(h) a limited run with an INVALID verdict: its limited verdict, exit 0" \
    st_overall "$d" 0 "PASS (limited run: --only 5; not a complete measurement)"
  st_expect "(h) a limited run with an INVALID verdict: the table is printed" \
    st_row "$d/summary.txt" ab_par_thin x "INVALID 1.0700 n/a ADOPT 1.0700 n/a INVALID"
  st_expect "(h) a limited run with an INVALID verdict: the missing part is printed" st_has "$d/summary.txt" \
    "the machine was not quiet"
  st_ab_case v-invalidword 0 0 invalidword final; st_ab_case v-invalidcase 0 0 invalidcase final
  st_ab_case v-repeat 0 0 repeat final; st_ab_case v-notag 0 0 notag final
  st_ab_case v-noheader 0 0 noheader final; st_ab_case v-noabvall 0 0 noabvall final
  d="$ST/ab-v-invalidword"
  st_expect "(i) an INVALID verdict without an invalid case (synthetic): 5-ab-verdicts WARN" \
    st_unit "$d/units.txt" 5-ab-verdicts WARN "1 INVALID, 0 in one pass only"
  st_expect "(i) an INVALID verdict without an invalid case (synthetic): overall INCOMPLETE, exit 3" st_overall "$d" 3 "$inc"
  st_expect "(i) an INVALID verdict without an invalid case (synthetic): missing part, the machine was not quiet" st_has "$d/summary.txt" \
    "  - step 5: 1 of $((2 * n)) A/B verdicts are INVALID (pass1 0 of $n, pass2 1 of $n) and 0 cases had the control outside the limit: the machine was not quiet"
  d="$ST/ab-v-invalidcase"
  st_expect "(j) an invalid case under ADOPT verdicts (synthetic): overall INCOMPLETE, exit 3" st_overall "$d" 3 "$inc"
  st_expect "(j) an invalid case under ADOPT verdicts (synthetic): missing part, the machine was not quiet" st_has "$d/summary.txt" \
    "  - step 5: 0 of $((2 * n)) A/B verdicts are INVALID (pass1 0 of $n, pass2 0 of $n) and 1 cases had the control outside the limit: the machine was not quiet"
  d="$ST/ab-v-repeat"
  st_expect "(k) a verdict repeated in one pass: 5-ab-verdicts FAIL" st_unit "$d/units.txt" 5-ab-verdicts FAIL "1 repeated verdicts"
  st_expect "(k) a verdict repeated in one pass: overall FAIL, exit 1" st_overall "$d" 1 FAIL
  d="$ST/ab-v-notag"
  st_expect "(l) an ABV line of a run without a pass tag: 5-ab-verdicts FAIL" \
    st_unit "$d/units.txt" 5-ab-verdicts FAIL "1 ABV lines without a pass or a column header"
  st_expect "(l) an ABV line of a run without a pass tag: overall FAIL, exit 1" st_overall "$d" 1 FAIL
  d="$ST/ab-v-noheader"
  st_expect "(m) ABV lines without the harness's column header: 5-ab-verdicts FAIL" \
    st_unit "$d/units.txt" 5-ab-verdicts FAIL "$n ABV lines without a pass or a column header"
  st_expect "(m) ABV lines without the harness's column header: overall FAIL, exit 1" st_overall "$d" 1 FAIL
  d="$ST/ab-v-noabvall"
  st_expect "(n) no ABV line at all: 5-ab-verdicts WARN with no candidate" st_unit "$d/units.txt" 5-ab-verdicts WARN "0 candidates"
  st_expect "(n) no ABV line at all: missing part, step 5 wrote no verdict" st_has "$d/summary.txt" \
    "  - step 5 wrote no A/B verdict (no ABV line in ab.tsv"
  # A --quick run on a busy machine: smoke verdicts with an invalid control
  # (the harness says SMOKE at smoke grade whatever the controls did).
  st_ab_case v-quickcase 1 0 invalidcase final "--quick"
  d="$ST/ab-v-quickcase"
  st_expect "(o) --quick with an invalid control: 5-ab-verdicts INFO, smoke verdicts not judged" st_unit "$d/units.txt" \
    5-ab-verdicts INFO "--quick: smoke verdicts are not judged; $n candidates: $n agree"
  st_expect "(o) --quick with an invalid control: the invalid case is still counted" st_unit "$d/units.txt" \
    5-ab-verdicts INFO "1 invalid cases in 1 of $((n / 2)) group runs"
  st_expect "(o) --quick with an invalid control: its limited verdict, exit 0" \
    st_overall "$d" 0 "PASS (limited run: --quick; not a complete measurement)"
  st_expect "(o) --quick with an invalid control: no missing part" st_lacks "$d/summary.txt" "missing:"
  st_expect "(o) --quick with an invalid control: no advice to run step 5 again" st_lacks "$d/summary.txt" "run step 5 again"
  st_expect "(o) --quick with an invalid control: the final message lacks nothing" st_lacks "$d/out.txt" "lacks parts"
}

# A skeleton checkout in $ST/skel-$1: the files the runbook's start-up
# checks read (the manifests, the AVX2 gate, the A/B harness's settings.rs,
# output.rs and stats.rs) and the runbook with its siblings.
st_skeleton() {
  local s="$ST/skel-$1" f
  for f in Cargo.toml crates/oxiblas-blas/Cargo.toml crates/oxiblas-benchmarks/Cargo.toml \
      crates/oxiblas-blas/src/level3/gemm_plan.rs crates/oxiblas-blas/src/level3/gemm_ab_tests/settings.rs \
      crates/oxiblas-blas/src/level3/gemm_ab_tests/output.rs crates/oxiblas-blas/src/level3/gemm_ab_tests/stats.rs; do
    mkdir -p "$s/$(dirname "$f")" && cp "$ROOT/$f" "$s/$f"
  done
  mkdir -p "$s/crates/oxiblas-blas/tools"
  for f in x86_gemm_runbook.sh x86_gemm_runbook_ab.sh x86_gemm_runbook_selftest.sh; do cp "$SCRIPT_DIR/$f" "$s/crates/oxiblas-blas/tools/$f"; done
}

# Whether the runbook of skeleton $1, run by this bash with the arguments
# $3..., exits with status $2 and says the text of file $1.want on stderr.
st_skel_run() {
  local s="$ST/skel-$1" want="$2" rc=0
  shift 2
  "$BASH" "$s/crates/oxiblas-blas/tools/x86_gemm_runbook.sh" "$@" > "$s.out" 2>&1 || rc=$?
  [ "$rc" = "$want" ] && grep -q -F -f "$s.want" "$s.out"
}

# The decision words step 5 judges: exactly five arms in stats.rs (6a),
# checked at start-up; and the step-5 file beside the runbook.
st_words() {
  local s="$ST/stats" real="$HARNESS_DIR/stats.rs" p
  mkdir -p "$s"
  st_expect "decision words: the real stats.rs has exactly the five words step 5 judges" test -z "$(ab_word_problem "$real")"
  awk '{ print } /Self::KeepLegacy => "KEEP_LEGACY",/ { sub(/KeepLegacy => "KEEP_LEGACY"/, "Maybe => \"MAYBE\""); print }' "$real" > "$s/sixth.rs"
  awk '!/Self::Smoke => "SMOKE",/' "$real" > "$s/nosmoke.rs"
  awk '{ print } /Self::Adopt => "ADOPT",/ { print }' "$real" > "$s/twice.rs"
  awk '{ sub(/Self::Adopt => "ADOPT",/, "Decision::Adopt => \"ADOPT\","); print }' "$real" > "$s/otherform.rs"
  awk '{ print } /Self::KeepLegacy => "KEEP_LEGACY",/ { print "            // Self::Maybe => \"MAYBE\"," }' "$real" > "$s/comment.rs"
  awk '{ sub(/Self::KeepLegacy => "KEEP_LEGACY",/, "Self::KeepLegacy => \"KEEP_LEGACY\""); sub(/Self::Adopt => "ADOPT",/, "Self::Adopt => \"ADOPT\", // adopted"); print }' "$real" > "$s/forms.rs"
  p="$(ab_word_problem "$s/sixth.rs")"
  st_expect "decision words: a sixth upper-case arm is refused" test "$p" = "words it does not know: MAYBE"
  st_expect "decision words: a missing word is refused" test "$(ab_word_problem "$s/nosmoke.rs")" = "words missing: SMOKE"
  st_expect "decision words: a word with two arms is refused" test "$(ab_word_problem "$s/twice.rs")" = "words with more than one arm: ADOPT"
  p="$(ab_word_problem "$s/otherform.rs")"
  st_expect "decision words: an upper-case arm of another form is refused" \
    test "$p" = "words missing: ADOPT; upper-case arms not of the form Self::<Name> => \"<WORD>\": line $(grep -n -F 'Decision::Adopt => "ADOPT",' "$s/otherform.rs" | cut -d: -f1) (ADOPT)"
  st_expect "decision words: a commented-out arm is not read" test -z "$(ab_word_problem "$s/comment.rs")"
  st_expect "decision words: an arm without its comma or with a comment after it is read" test -z "$(ab_word_problem "$s/forms.rs")"
  st_expect "decision words: a missing stats.rs is refused" test "$(ab_word_problem "$s/none.rs")" = "the file is missing"
  # End to end: a sixth word stops the runbook at start-up (a dry run, so
  # nothing is run whatever the machine), and so does a missing step-5 file.
  st_skeleton words; cp "$s/sixth.rs" "$ST/skel-words/crates/oxiblas-blas/src/level3/gemm_ab_tests/stats.rs"
  printf 'stats.rs: words it does not know: MAYBE;\n' > "$ST/skel-words.want"
  st_expect "decision words: the runbook stops at start-up (exit 2) on a sixth word" st_skel_run words 2 --dry-run
  st_skeleton noab; rm -f "$ST/skel-noab/crates/oxiblas-blas/tools/x86_gemm_runbook_ab.sh"
  printf 'error: the step-5 file crates/oxiblas-blas/tools/x86_gemm_runbook_ab.sh is missing (it belongs beside this script); nothing was changed\n' > "$ST/skel-noab.want"
  st_expect "step-5 file missing: exit 2 with the message, for a real run" st_skel_run noab 2 --only 5
  st_expect "step-5 file missing: nothing was written" test ! -e "$ST/skel-noab/target"
}

st_ab() {
  local u agg
  st_fake_cargo > "$ST/bin/cargo" && chmod +x "$ST/bin/cargo"
  st_ab_case decision 0 0 decision; st_ab_case quick 1 0 decision; st_ab_case smoke 0 0 debug
  st_ab_case silent 0 0 silent; st_ab_case sse 0 0 sse; st_ab_case sse-allowed 0 1 sse
  st_ab_case noend 0 0 noend
  for agg in 5-ab-pass1-seq 5-ab-pass1-par 5-ab-pass2-par 5-ab-pass2-seq; do
    st_expect "step 5, full run, decision grade (an [env] table filling unset variables): $agg PASS" \
      st_unit "$ST/ab-decision/.units" "$agg" PASS "$GRADE_DECISION on"
    st_expect "step 5, --quick, smoke grade: $agg PASS" st_unit "$ST/ab-quick/.units" "$agg" PASS
    st_expect "step 5, full run, smoke grade (debug build): $agg FAIL naming the grade" \
      st_unit "$ST/ab-smoke/.units" "$agg" FAIL "debug_build=true"
  done
  u="$ST/ab-silent/.units"
  st_expect "step 5, a group process that wrote nothing: that group FAIL" \
    st_unit "$u" 5-ab-pass1-ab_seq_thin FAIL "wrote no SET line"
  st_expect "step 5, a group process that wrote nothing: its aggregate FAIL naming the group" \
    st_unit "$u" 5-ab-pass1-seq FAIL "ab_seq_thin (the process wrote no SET line"
  st_expect "step 5, a group process that wrote nothing: the other aggregate PASS" \
    st_unit "$u" 5-ab-pass1-par PASS
  st_expect "step 5, tier Sse42: FAIL" st_unit "$ST/ab-sse/.units" 5-ab-pass1-par FAIL "lacks tier=Avx2Fma"
  st_expect "step 5, tier Sse42 with --allow-no-avx2: WAIVED" st_unit "$ST/ab-sse-allowed/.units" 5-ab-pass1-par WAIVED
  st_expect "step 5, a SET line without END: the group FAIL" \
    st_unit "$ST/ab-noend/.units" 5-ab-pass1-ab_par_thin FAIL "did not complete"
  st_expect "step 5, a SET line without END: the aggregate FAIL" \
    st_unit "$ST/ab-noend/.units" 5-ab-pass1-par FAIL "SET and END lines differ"
}

st_fake_tiny() {
  cat <<'EOF'
#!/bin/sh
# Stand-in for gemm_tiny_call_check.sh in the self-test of x86_gemm_runbook.sh:
# exits with FAKE_TINY (0 PASS, 1 FAIL, 2 no base release, 3 INCOMPLETE) after
# writing a summary like the tool's, which names the tree FAKE_ROOT.
out=""
while [ "$#" -gt 0 ]; do case "$1" in --out) out="$2"; shift 2 ;; *) shift ;; esac; done
if [ "$FAKE_TINY" = 2 ]; then
  echo "gemm_tiny_call_check.sh: error: the base revision 'v0.2.3' does not exist in $FAKE_ROOT" >&2
  exit 2
fi
mkdir -p "$out/logs"
printf '   Compiling oxiblas-blas v0.2.4 (%s/crates/oxiblas-blas)\n' "$FAKE_ROOT" > "$out/logs/build-tree.log"
if [ "$FAKE_TINY" = 3 ]; then
  printf '# the tree build failed; see %s/logs/build-tree.log\nverdict: INCOMPLETE\n' "$out" > "$out/summary.txt"
else
  ratio=1.010; verdict=PASS
  if [ "$FAKE_TINY" = 1 ]; then ratio=1.080; verdict=FAIL; fi
  printf '# base v0.2.3 (94f9ec4), tree %s, 5 rounds x 200 samples, limit 1.03, load 1.00\n# case base_ns tree_ns tree/base judged|info\nnn_f64_1          5.000      5.050   %s judged\nverdict: %s\n' \
    "$FAKE_ROOT" "$ratio" "$verdict" > "$out/summary.txt"
fi
cat "$out/summary.txt"
exit "$FAKE_TINY"
EOF
}

# The tiny-call step with the stand-in tool exiting with $1, then finalize
# of a run of that step (not a limited one); units.txt, rc and summary.txt
# end up in $ST/tiny-$1.
st_tiny_case() {
  local d="$ST/tiny-$1"
  mkdir -p "$d/logs"
  (
    OUT="$d"; LOG_DIR="$d/logs"; UNITS="$d/.units"; AB_OUT="$d/ab.tsv"; TINY_OUT="$d/tiny-call"; TINY_TOOL="$ST/bin/fake-tiny"
    quick=0; allow_no_avx2=0; dry_run=0; require_avx2_text=1; only_list=" tiny "; limited_parts=""; out_ready=1
    machine_ok=1; machine_problem=""; CRITERION_DIR="$d/criterion"; ARCHIVE="$d/results.tar.gz"; prev_rel=""
    removed_env=""; N_ONLINE=8; N_PHYSICAL=4; PIN_CPU=2; AB_THREADS=2,4; OPENBLAS_PATH="$d/no-openblas"
    FAKE_TINY="$1"; FAKE_ROOT="$ROOT"; export FAKE_TINY FAKE_ROOT
    : > "$UNITS"
    step_tiny
    cp "$UNITS" "$d/units.txt"; finalize ""; printf '%s\n' "$final_rc" > "$d/rc"
  ) > "$d/out.txt" 2>&1 || true
}

# Whether the archive $1 holds the member ./$2 and it contains the text $3.
st_tar_has() {
  tar -xzOf "$1" "./$2" 2>/dev/null | grep -q -F -e "$3"
}

# The tiny-call step: information only, whatever the tool says.
st_tiny() {
  local d
  st_fake_tiny > "$ST/bin/fake-tiny" && chmod +x "$ST/bin/fake-tiny"
  st_tiny_case 0; st_tiny_case 1; st_tiny_case 2; st_tiny_case 3
  d="$ST/tiny-0"
  st_expect "tiny-call, the tool exits 0: INFO with its verdict line and its settings" st_unit "$d/units.txt" tiny-call INFO \
    "information only, never a failure: verdict: PASS; base v0.2.3 (94f9ec4), 5 rounds x 200 samples, limit 1.03, load 1.00"
  st_expect "tiny-call, the tool exits 0: overall PASS, exit 0" st_overall "$d" 0 PASS
  st_expect "tiny-call, the tool exits 0: its summary is in the archive, the tree path replaced" \
    st_tar_has "$d/results.tar.gz" tiny-call/summary.txt "tree <checkout>, 5 rounds"
  d="$ST/tiny-1"
  st_expect "tiny-call, the tool exits 1 (FAIL): INFO" st_unit "$d/units.txt" tiny-call INFO "never a failure: verdict: FAIL"
  st_expect "tiny-call, the tool exits 1 (FAIL): the run stays PASS, exit 0" st_overall "$d" 0 PASS
  d="$ST/tiny-2"
  st_expect "tiny-call, the tool exits 2 (no base release): SKIPPED with the tool's reason" st_unit "$d/units.txt" tiny-call SKIPPED \
    "information only: the tool could not run (the base revision 'v0.2.3' does not exist in"
  st_expect "tiny-call, the tool exits 2: the run stays PASS, exit 0" st_overall "$d" 0 PASS
  d="$ST/tiny-3"
  st_expect "tiny-call, the tool exits 3 (INCOMPLETE): INFO with its reason" st_unit "$d/units.txt" tiny-call INFO \
    "never a failure: verdict: INCOMPLETE; the tree build failed"
  st_expect "tiny-call, the tool exits 3: the run stays PASS (not INCOMPLETE), exit 0" st_overall "$d" 0 PASS
  st_expect "tiny-call, the tool exits 3: the log it names is in the archive, the path replaced" \
    st_tar_has "$d/results.tar.gz" tiny-call/logs/build-tree.log "(<checkout>/crates/oxiblas-blas)"
  st_expect "tiny-call, the tool exits 3: no copy was left out of the archive" st_lacks "$d/summary.txt" "left out"
}

st_counts() {
  local d="$ST/counts"
  mkdir -p "$d/logs"
  (
    LOG_DIR="$d/logs"; UNITS="$d/.units"; dry_run=0; : > "$UNITS"
    printf 'running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n' > "$LOG_DIR/good.log"
    : > "$LOG_DIR/empty.log"
    printf '\377\376 garbled \303( bytes\ntest result: FAILED. many passed\n' > "$LOG_DIR/garbled.log"
    printf 'binary \200\201 and no summary line\n' > "$LOG_DIR/nosummary.log"
    for id in good empty garbled nosummary; do unit_rc=0; unit_secs=0; finish_tests "$id"; done
  ) 2>/dev/null || true
  st_expect "test counts: a normal log PASS with 3 passed" st_unit "$d/.units" good PASS "tests 3 passed"
  st_expect "test counts: an empty log FAIL" st_unit "$d/.units" empty FAIL "no \`test result:\` line"
  st_expect "test counts: a garbled log FAIL" st_unit "$d/.units" garbled FAIL "tests 0 passed"
  st_expect "test counts: a log without a summary line FAIL" st_unit "$d/.units" nosummary FAIL "no \`test result:\` line"
}

# Whether file $1 contains none of the texts $2...
st_none_of() {
  local file="$1" text
  shift
  for text in "$@"; do if [ -n "$text" ] && grep -q -F -e "$text" "$file"; then return 1; fi; done
}

st_scrub() {
  local name n=0 log root home root_q home_q root_inner
  # shellcheck disable=SC2016 # a directory name with a literal $
  for name in "space dir" "$(printf '\346\227\245\346\234\254\350\252\236')" "it's" 'cost$HOME'; do
    n=$((n + 1)); log="$ST/scrub-$n.log"
    root="$ST/scrub/$name/oxiblas"; home="$ST/home/$name user"
    root_q="$(printf '%q' "$root")"; home_q="$(printf '%q' "$home")"
    root_inner="${root_q#"\$'"}"; root_inner="${root_inner%"'"}"
    (
      set_path_forms "$root" "$root" "$home"
      {
        printf '== command: %s\n' "$(show_cmd env "CARGO_TARGET_DIR=$root/target" "OXIBLAS_AB_OUT=$home/x" cargo test)"
        printf 'old form: %s\n' "$(printf '%q' "CARGO_TARGET_DIR=$root/target")"
        printf 'raw: %s/crates/a.rs:1:2 and %s/.cargo/registry\n' "$root" "$home"
        printf 'quoted home: %s\n' "$(printf '%q' "$home/.cargo")"
        printf 'debug: "%s"\n' "$(printf '%s' "$root" | sed 's/[\\"]/\\&/g')"
      } > "$log"
      scrub_copy "$log" "$log.out"
      st_expect "scrub [$name]: the command line names <checkout> and <home> before quoting" \
        grep -q -F -e "CARGO_TARGET_DIR=<checkout>/target OXIBLAS_AB_OUT=<home>/x cargo test" "$log"
      st_expect "scrub [$name]: the check finds the paths in the raw log" test -n "$(find_leaks "$log")"
      st_expect "scrub [$name]: the check finds nothing in the scrubbed copy" test -z "$(find_leaks "$log.out")"
      st_expect "scrub [$name]: no raw, %q or \$'...' form of either path in the copy" \
        st_none_of "$log.out" "$root" "$root_q" "$root_inner" "$home" "$home_q" "$name"
      st_expect "scrub [$name]: the copy reads <checkout>/crates/a.rs:1:2 and <home>/.cargo/registry" \
        grep -q -x -F -e "raw: <checkout>/crates/a.rs:1:2 and <home>/.cargo/registry" "$log.out"
    )
  done
  mkdir -p "$ST/drop/logs"
  printf 'clean\n' > "$ST/drop/logs/a.log"
  printf 'built in %s/target\n' "$ROOT" > "$ST/drop/logs/b.log"
  st_expect "archive check: the copy with the checkout path is left out and named" \
    test "$(drop_leaking_copies "$ST/drop")" = "logs/b.log"
  st_expect "archive check: the clean copy stays" test -f "$ST/drop/logs/a.log" -a ! -e "$ST/drop/logs/b.log"
  printf 'crates/oxiblas-blas/src/lib.rs and /srcx\n' > "$ST/drop/c.log"
  st_expect "scrub: a checkout at /src leaves oxiblas-blas/src alone" \
    test -z "$(set_path_forms /src /src ""; find_leaks "$ST/drop/c.log")"
  printf 'BOOT_IMAGE=/v root=UUID=1 systemd.hostname=h BOOTIF=01-aa ip=1:2 x.VMID=e95bd103-89a0-4bd3-9b9f-6e9c6137a652\n' > "$ST/cmdline"
  st_expect "kernel cmdline: device, address and host values and UUIDs redacted" test "$(kernel_cmdline "$ST/cmdline")" = "BOOT_IMAGE=/v root=<redacted> systemd.hostname=<redacted> BOOTIF=<redacted> ip=<redacted> x.VMID=<uuid>"
}

# The exit status of take_lock on $1 in a subshell (its own exits stay
# there).
st_take() {
  if (take_lock "$1") > /dev/null 2>&1; then echo 0; else echo "$?"; fi
}

st_lock() {
  local l="$ST/lock.d" dead start
  st_expect "lock: a free lock is taken" test "$(st_take "$l")" = 0
  st_expect "lock: it holds this process id" test "$(sed -n 1p "$l/pid" 2>/dev/null)" = "$$"
  st_expect "lock: a second run is refused with exit 2" test "$(st_take "$l")" = 2
  printf '%s\n%s\n' "$$" "$(TZ=XYZ-9 process_start "$$")" > "$l/pid"
  st_expect "lock: a live run's lock is refused also when the runs have different time zones (TZ)" test "$(TZ=ABC+5 st_take "$l")" = 2
  (lock_dir="$l"; release_lock)
  st_expect "lock: released, the directory is gone" test ! -e "$l"
  (exit 0) & dead=$!
  wait "$dead" || true
  mkdir "$l" && printf '%s\n' "$dead" > "$l/pid"
  st_expect "lock: the lock of a process that is gone is taken over" test "$(st_take "$l")" = 0
  rm -rf "$l"
  start="$(process_start "$$")"
  mkdir "$l" && printf '%s\nMon Jan  1 00:00:00 1990\n' "$$" > "$l/pid"
  if [ -n "$start" ]; then
    st_expect "lock: a reused process id (other start time) is taken over" test "$(st_take "$l")" = 0
  else
    printf 'ok      lock: reused process id not checked (ps gives no start time here)\n'
  fi
  rm -rf "$l" && mkdir "$l"
  st_expect "lock: a lock without a process id (its run died at once) is taken over" test "$(st_take "$l")" = 0
  rm -rf "$l"
}

# Whether config_refusals on a file with the text $2 (printf %b) refuses
# ($1 = yes) or accepts ($1 = no).
st_cfg() {
  local refused
  printf '%b\n' "$2" > "$ST/config.toml" && refused="$(config_refusals "$ST/config.toml")"
  if [ "$1" = yes ]; then [ -n "$refused" ]; else [ -z "$refused" ]; fi
}

st_config() {
  local form
  while IFS= read -r form; do st_expect "config refused: $form" st_cfg yes "$form"; done <<'EOF'
[build]\ntarget = "x86_64-unknown-linux-musl"
build.target = "x"
build = { target = "x" }
build = {jobs = 2, "target" = ["a", "b"]}
[ build ] # c\n  target="x"
[env]\nOXIBLAS_AB_ROUNDS = "4"
env = { OXIBLAS_AB_ROUNDS = "4" }
env.RAYON_NUM_THREADS = "2"
[env.OMP_NUM_THREADS]\nvalue = "1"
[env]\n"OPENBLAS_NUM_THREADS" = { value = "1", force = true }
[build]\nrustflags = [\n  "-C", "target-cpu=native", # ]\n]\ntarget = "x"
a = ["""x"""", "["]\n[env]\nOXIBLAS_AB_ROUNDS = "4"
env = { A = '''x'''', OXIBLAS_AB_ROUNDS = "4" }
EOF
  while IFS= read -r form; do st_expect "config accepted: $form" st_cfg no "$form"; done <<'EOF'
[build]\ntarget-dir = "t"\nrustflags = [\n "--cfg", "target=x",\n]
[env]\nMY_VAR = "OXIBLAS_AB_ROUNDS"\nCC = { value = "cc" }
# build.target = "x"\n[target.x86_64-unknown-linux-gnu]\nrunner = "x"
[alias]\nb = "build --target x"\n[profile.release]\nlto = { a = 1 }
EOF
}

run_self_test() {
  local results bad total
  ST="$ROOT/target/x86-runbook-selftest.$$"
  mkdir -p "$ST/bin" || die 2 "cannot create $ST (is target/ writable?)"
  trap 'rm -rf "$ST"' EXIT
  results="$ST/results.txt"
  printf '%s --self-test: built-in sample data, no cargo; scratch target/%s\n' "$SCRIPT_NAME" "${ST##*/}"
  if ! { st_ab; st_verdicts; st_words; st_tiny; st_counts; st_scrub; st_lock; st_config; } | tee "$results"; then
    printf 'self-test: FAIL (it stopped early; see the lines above)\n'
    exit 1
  fi
  total="$(grep -c '' "$results" || true)"; bad="$(grep -c '^NOT OK' "$results" || true)"
  if [ "$bad" = 0 ] && [ "$total" -gt 0 ]; then printf 'self-test: PASS (%s expectations)\n' "$total"; exit 0; fi
  printf 'self-test: FAIL (%s of %s expectations do not hold)\n' "$bad" "$total"
  exit 1
}
