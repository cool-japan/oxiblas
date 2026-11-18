# shellcheck shell=bash disable=SC2154
# x86_gemm_runbook_ab.sh -- step 5 of x86_gemm_runbook.sh: the A/B harness
# in two passes, and the comparison of their verdicts (5-ab-verdicts).
#
# Not a program of its own: x86_gemm_runbook.sh sources this file from its
# own directory in every run, the dry run and the self-test (not for --help
# or a usage error), after it has found the checkout; without it the runbook
# stops with exit 2 before anything is written. The functions here use the
# functions, constants and paths of the runbook (SC2154 is off above: quick,
# dry_run, allow_no_avx2, host_arch, have_taskset, unit_verdict, unit_secs
# and the like are set by the runbook, which shellcheck cannot see from this
# file alone).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  printf '%s: this file is sourced by x86_gemm_runbook.sh; run x86_gemm_runbook.sh\n' "${0##*/}" >&2
  exit 2
fi

# The lines of an A/B output file between two line numbers (inclusive).
# Line 1: "AB ABR ABV SET END SET_AVX2 SET_DECISION", where SET_AVX2 counts
# the SET lines with tier=Avx2Fma and SET_DECISION those with
# grade=decision. Line 2: the first SET line without grade=decision, as its
# grade, group, rounds_requested, min_sample_ms, quick and debug_build
# fields (empty when there is none).
ab_check_slice() {
  local file="$1" first="$2" last="$3"
  if [ ! -f "$file" ]; then
    printf '0 0 0 0 0 0 0\n\n'
    return 0
  fi
  awk -F'\t' -v first="$first" -v last="$last" -v decision="$GRADE_DECISION" '
    NR < first || NR > last { next }
    $1 == "AB" { ab++ }
    $1 == "ABR" { abr++ }
    $1 == "ABV" { abv++ }
    $1 == "AB#" && $2 == "SET" {
      set++; is_avx = 0; is_dec = 0; grade = "no grade key"; desc = ""
      for (i = 3; i <= NF; i++) {
        if ($i == "tier=Avx2Fma") is_avx = 1
        if ($i == decision) is_dec = 1
        if ($i ~ /^grade=/) grade = $i
        if ($i ~ /^(group|rounds_requested|min_sample_ms|quick|debug_build)=/) desc = desc " " $i
      }
      avx += is_avx; dec += is_dec
      if (!is_dec && why == "") why = grade desc
    }
    $1 == "AB#" && $2 == "END" { end++ }
    END { printf "%d %d %d %d %d %d %d\n%s\n", ab, abr, abv, set, end, avx, dec, why }' "$file"
}

# What is wrong with the lines one group process appended (empty when
# nothing): no SET line, SET and END in different numbers, SET and ABV in
# different numbers (a completed group run without its verdict), or
# (without --quick) a SET line that is not decision grade.
#   ab_group_problem <AB> <ABR> <ABV> <SET> <END> <SET_AVX2> <SET_DECISION> <why>
ab_group_problem() {
  local abv="$3" set="$4" end="$5" dec="$7" why="$8" out=""
  if [ "$set" = 0 ]; then
    out="the process wrote no SET line to ab.tsv"
  elif [ "$set" != "$end" ]; then
    out="SET $set and END $end lines in this process: a group run did not complete"
  elif [ "$abv" != "$set" ]; then
    out="SET $set and ABV $abv lines in this process: a group run wrote no verdict (ABV line)"
  fi
  if [ "$quick" = 0 ] && [ "$dec" != "$set" ]; then
    out="${out:+$out; }$((set - dec)) of $set SET lines lack $GRADE_DECISION (first: $why), so they cannot decide"
  fi
  printf '%s' "$out"
}

# The verdict of A/B output line counts: prints "<PASS|FAIL|WAIVED>\t<detail>".
#   ab_judge <AB> <ABR> <ABV> <SET> <END> <SET_AVX2> <SET_DECISION> <allow_no_avx2: 0|1> <why>
ab_judge() {
  local ab="$1" abr="$2" abv="$3" set="$4" end="$5" avx="$6" dec="$7" allow="$8" why="$9" verdict=PASS detail
  detail="lines AB $ab, ABR $abr, ABV $abv, SET $set, END $end; tier=Avx2Fma on $avx and $GRADE_DECISION on $dec of $set SET lines"
  if [ "$set" = 0 ]; then
    verdict=FAIL; detail="$detail; no group run wrote a SET line"
  fi
  if [ "$set" != "$end" ]; then
    verdict=FAIL; detail="$detail; SET and END lines differ: a group run did not complete"
  elif [ "$abv" != "$set" ]; then
    verdict=FAIL; detail="$detail; SET and ABV lines differ: a group run wrote no verdict"
  fi
  if [ "$quick" = 0 ] && [ "$dec" != "$set" ]; then
    verdict=FAIL; detail="$detail; a SET line lacks $GRADE_DECISION, so this full run has no decision-grade data (first: $why)"
  fi
  if [ "$avx" != "$set" ]; then
    if [ "$allow" = 1 ]; then
      [ "$verdict" = PASS ] && verdict=WAIVED
      detail="$detail; a SET line lacks tier=Avx2Fma (allowed by --allow-no-avx2)"
    else
      verdict=FAIL; detail="$detail; a SET line lacks tier=Avx2Fma"
    fi
  fi
  printf '%s\t%s\n' "$verdict" "$detail"
}

ab_lines() {
  if [ -f "$AB_OUT" ]; then grep -c '' "$AB_OUT" || true; else echo 0; fi
}

# The running totals of one aggregate unit (pass and kind) of step 5.
agg_reset() {
  current_agg="$1"; agg_start="$SECONDS"; agg_groups=0; agg_failed=0; agg_failed_names=""
  agg_skipped=0; agg_counts="0 0 0 0 0 0 0"; agg_why=""
}

agg_finish() {
  local id="$current_agg" judged verdict detail
  [ -n "$id" ] || return 0
  current_agg=""
  [ "$dry_run" = 1 ] && return 0
  if [ "$agg_groups" = 0 ]; then
    record "$id" SKIPPED - 0 "no group of this kind runs here ($agg_skipped skipped)"
    return 0
  fi
  # shellcheck disable=SC2086 # seven numbers
  judged="$(ab_judge $agg_counts "$allow_no_avx2" "$agg_why")"
  verdict="${judged%%"$TAB"*}"
  detail="${judged#*"$TAB"}"
  if [ "$agg_failed" != 0 ]; then
    verdict=FAIL
    detail="$detail; $agg_failed of $agg_groups group processes failed: ${agg_failed_names#; }"
  fi
  detail="$agg_groups group processes ($agg_skipped skipped); $detail"
  record "$id" "$verdict" - "$((SECONDS - agg_start))" "$detail"
}

# Adds seven counts to the aggregate totals.
agg_add() {
  agg_counts="$(printf '%s %s\n' "$agg_counts" "$1" | awk '{ printf "%d %d %d %d %d %d %d\n", $1 + $8, $2 + $9, $3 + $10, $4 + $11, $5 + $12, $6 + $13, $7 + $14 }')"
}

# One A/B group in its own process.
#   ab_group <pass> <group> <seq|par>
ab_group() {
  local pass="$1" group="$2" kind="$3" id envs before after slice counts why problem g pin=()
  id="5-ab-$pass-$group"
  for g in $AB_X86_ONLY_GROUPS; do
    if [ "$g" = "$group" ] && [ "$host_arch" != x86_64 ] && [ "$dry_run" = 0 ]; then
      record "$id" SKIPPED - 0 "an x86-64 group (this is $host_arch)"
      agg_skipped=$((agg_skipped + 1))
      return 0
    fi
  done
  # Set explicitly (also when empty), so that a cargo [env] table cannot
  # fill them in.
  envs=("${LEGACY_ENV[@]}" "OXIBLAS_AB_OUT=$AB_OUT" "OXIBLAS_AB_TAG=$pass")
  envs+=("OXIBLAS_AB_RETRIES=${OXIBLAS_AB_RETRIES:-$AB_RETRIES}")
  if [ "$quick" = 1 ]; then
    envs+=(OXIBLAS_AB_QUICK=1 "OXIBLAS_AB_ROUNDS=$AB_QUICK_ROUNDS")
  else
    envs+=(OXIBLAS_AB_QUICK= "OXIBLAS_AB_ROUNDS=$AB_FULL_ROUNDS")
  fi
  if [ "$kind" = par ]; then
    envs+=("OXIBLAS_AB_THREADS=$AB_THREADS" OXIBLAS_AB_MIN_SAMPLE_MS=10)
  else
    envs+=(OXIBLAS_AB_MIN_SAMPLE_MS=2)
    if [ "$have_taskset" = 1 ] || [ "$dry_run" = 1 ]; then
      # A dry run shows the pinning a Linux run does, also where taskset is
      # missing.
      pin=(taskset -c "$PIN_CPU")
    fi
  fi
  before="$(ab_lines)"
  run_cmd "$id" "A/B group $group, $pass$([ "$kind" = seq ] && echo ", pinned to CPU $PIN_CPU" || echo ", threads $AB_THREADS")" \
    "${envs[@]}" ${pin[@]+"${pin[@]}"} "${AB_BASE[@]}" "${AB_RUN_ARGS[@]}" --exact "level3::gemm_ab_tests::$group"
  [ "$dry_run" = 1 ] && return 0
  after="$(ab_lines)"
  slice="$(ab_check_slice "$AB_OUT" "$((before + 1))" "$after")"
  counts="$(printf '%s\n' "$slice" | sed -n 1p)"
  why="$(printf '%s\n' "$slice" | sed -n 2p)"
  # shellcheck disable=SC2086 # seven numbers
  problem="$(ab_group_problem $counts "$why")"
  finish_tests "$id" "$AB_NONE_REASON" "; $((after - before)) lines appended to ab.tsv (AB ABR ABV SET END: $(printf '%s' "$counts" | cut -d' ' -f1-5))" "$problem"
  agg_groups=$((agg_groups + 1))
  if [ "$unit_verdict" = FAIL ]; then
    agg_failed=$((agg_failed + 1))
    agg_failed_names="$agg_failed_names; $group (${problem:-the test run failed, exit $unit_rc})"
  fi
  if [ -z "$agg_why" ] && [ -n "$why" ]; then agg_why="$why"; fi
  agg_add "$counts"
}

# One pass over a list of groups, in the given order; a new aggregate unit
# starts whenever the kind (sequential, parallel) changes.
#   ab_pass <pass> <group>...
ab_pass() {
  local pass="$1" group kind last_kind=""
  shift
  for group in "$@"; do
    case "$group" in ab_seq_*) kind=seq ;; *) kind=par ;; esac
    if [ "$kind" != "$last_kind" ]; then
      agg_finish
      agg_reset "5-ab-$pass-$kind"
      last_kind="$kind"
    fi
    ab_group "$pass" "$group" "$kind"
  done
  agg_finish
}

# Step 5: the A/B harness, two passes.
step_ab() {
  local group forward=() reverse=() i
  AB_BASE=(cargo test -p oxiblas-blas --release --features parallel --lib --no-fail-fast)
  AB_RUN_ARGS=(-- --ignored --test-threads=1 --nocapture)
  if [ -f "$HARNESS_DIR/mod.rs" ]; then
    AB_NONE_REASON="no ab_* test ran although the A/B harness exists (was the group renamed?)"
  else
    AB_NONE_REASON="no ab_* test in this checkout: the A/B harness (crates/oxiblas-blas/src/level3/gemm_ab_tests/mod.rs) is missing"
  fi

  # Built unpinned first, with the same environment as the runs, so that the
  # pinned runs compile nothing.
  run_cmd 5-ab-build "build the A/B harness (no run)" "${LEGACY_ENV[@]}" "${AB_BASE[@]}" --no-run
  record 5-ab-build "$([ "$unit_rc" = 0 ] && echo PASS || echo FAIL)" "$unit_rc" "$unit_secs" "lib test binary built"
  [ "$dry_run" = 1 ] || [ "$unit_rc" = 0 ] || return 0
  if [ "$have_taskset" = 0 ] && [ "$dry_run" = 0 ]; then
    record 5-ab-seq-pin WARN - 0 "taskset not found: the sequential groups run unpinned"
    add_missing "the sequential A/B groups of step 5 ran unpinned, because taskset was not found: install util-linux (Debian/Ubuntu: sudo apt install util-linux; Fedora: sudo dnf install util-linux) and run again"
  fi

  for group in $AB_SEQ_GROUPS $AB_PAR_GROUPS; do forward+=("$group"); done
  i=$((${#forward[@]} - 1))
  while [ "$i" -ge 0 ]; do
    reverse+=("${forward[$i]}")
    i=$((i - 1))
  done
  ab_pass pass1 "${forward[@]}"
  ab_pass pass2 "${reverse[@]}"
  if [ "$dry_run" = 1 ]; then
    printf '\n## 5-ab-verdicts: compare the ABV verdicts of pass1 and pass2 candidate by candidate\n'
  else
    ab_verdict_unit
  fi
}

# What is wrong with the decision words of the harness file $1 (stats.rs);
# empty when nothing. Step 5 judges exactly the words of AB_DECISION_WORDS,
# so the file must have one match arm `Self::<Name> => "<WORD>"` (with or
# without a comma, perhaps a // comment after it) for each of them, and no
# other arm whose string is an upper-case word: a word the harness may add
# must stop the run at start-up, not make step 5 FAIL at its end. Comment
# lines are not read.
#   ab_word_problem <stats.rs>
ab_word_problem() {
  if [ ! -f "$1" ]; then printf 'the file is missing'; return 0; fi
  awk -v words="$AB_DECISION_WORDS" '
    BEGIN { n = split(words, w, " "); for (i = 1; i <= n; i++) known[w[i]] = 1 }
    /^[[:space:]]*\/\// { next }
    /=>[[:space:]]*"[A-Z][A-Z0-9_]*"/ {
      word = $0; sub(/.*=>[[:space:]]*"/, "", word); sub(/".*/, "", word)
      if ($0 !~ /^[[:space:]]*Self::[A-Za-z0-9_]+[[:space:]]*=>[[:space:]]*"[A-Z][A-Z0-9_]*",?[[:space:]]*(\/\/.*)?$/) {
        other = other " line " NR " (" word ")"; next
      }
      seen[word]++
      if (!(word in known)) unknown = unknown " " word
    }
    END {
      for (i = 1; i <= n; i++) {
        if (!(w[i] in seen)) missing = missing " " w[i]
        else if (seen[w[i]] > 1) twice = twice " " w[i]
      }
      if (unknown != "") out = out "; words it does not know:" unknown
      if (missing != "") out = out "; words missing:" missing
      if (twice != "") out = out "; words with more than one arm:" twice
      if (other != "") out = out "; upper-case arms not of the form Self::<Name> => \"<WORD>\":" other
      printf "%s", substr(out, 3)
    }' "$1"
}

# The A/B verdicts of file $1 by candidate of a group run, tab-separated:
#   C group run candidate <pass1: decision partial geo> <pass2: the same> state
# ("-" where a pass has no verdict; state agree, DISAGREE, INVALID or
# MISSING); per pass "N pass verdicts <one count per AB_DECISION_WORDS>
# other-words group-runs runs-with-invalid-cases invalid-cases repeated";
# and "X <ABV lines without a pass or a column header>". The columns are
# found by name in the harness's `AB#<TAB>ABV` header line, the pass of a
# verdict by the tag of the SET line with its run_id.
ab_decisions() {
  local f="$1"
  [ -f "$f" ] || f=/dev/null
  awk -F'\t' -v words="$AB_DECISION_WORDS" '
    BEGIN { nw = split(words, w, " ") }
    $1 == "AB#" && $2 == "ABV" { split("", col); for (i = 3; i <= NF; i++) col[$i] = i - 1; next }
    $1 == "AB#" && $2 == "SET" { for (i = 4; i <= NF; i++) if ($i ~ /^tag=/) tag[$3] = substr($i, 5); next }
    $1 != "ABV" { next }
    !(("run_id" in col) && ("group" in col) && ("run" in col) && ("invalid" in col) && ("verdicts" in col)) { stray++; next }
    {
      p = tag[$(col["run_id"])]
      if (p != "pass1" && p != "pass2") { stray++; next }
      runs[p]++; inv = $(col["invalid"]) + 0
      if (inv > 0) { invruns[p]++; invcases[p] += inv }
      n = split($(col["verdicts"]), e, ";")
      for (j = 1; j <= n; j++) {
        name = e[j]; sub(/:.*/, "", name); m = split(substr(e[j], length(name) + 2), kv, ",")
        d = kv[1]; part = "-"; geo = "-"
        for (q = 2; q <= m; q++) {
          if (kv[q] ~ /^partial=/) part = substr(kv[q], 9)
          if (kv[q] ~ /^geo=/) geo = substr(kv[q], 5)
        }
        key = $(col["group"]) "\t" $(col["run"]) "\t" name
        if (!(key in seen)) { seen[key] = 1; order[++nk] = key }
        if ((p, key) in dec) dup[p]++
        dec[p, key] = d; par[p, key] = part; geos[p, key] = geo; total[p]++; cnt[p, d]++
      }
    }
    END {
      for (k = 1; k <= nk; k++) {
        key = order[k]; line = "C\t" key; s = ""
        for (q = 1; q <= 2; q++) {
          if (("pass" q, key) in dec) line = line "\t" dec["pass" q, key] "\t" par["pass" q, key] "\t" geos["pass" q, key]
          else { line = line "\t-\t-\t-"; s = "MISSING" }
        }
        if (s == "") { a = dec["pass1", key]; b = dec["pass2", key]; s = (a == "INVALID" || b == "INVALID") ? "INVALID" : (a == b ? "agree" : "DISAGREE") }
        print line "\t" s
      }
      for (q = 1; q <= 2; q++) {
        p = "pass" q; line = "N\t" p "\t" (total[p] + 0); known = 0
        for (i = 1; i <= nw; i++) { line = line "\t" (cnt[p, w[i]] + 0); known += cnt[p, w[i]] }
        print line "\t" (total[p] - known) "\t" (runs[p] + 0) "\t" (invruns[p] + 0) "\t" (invcases[p] + 0) "\t" (dup[p] + 0)
      }
      print "X\t" (stray + 0)
    }' "$f"
}

# Step 5 as a decision, from ab_decisions of file $1: tab-separated lines
# "V <PASS|WARN|FAIL|INFO> <detail>", "M <missing part>" and "T <summary
# line>". FAIL: a verdict that cannot be read. WARN (with missing parts, so
# a run that is not limited ends INCOMPLETE): an INVALID verdict or a case
# with an invalid control, a candidate whose two passes disagree (an
# informational one, never adopted, is only named in the detail) or that has
# a verdict in one pass only, no verdict at all, or SMOKE in a full run.
# INFO: a --quick run, whose smoke-grade verdicts are not judged (no missing
# part, no advice to run step 5 again; FAIL still for a verdict it cannot
# read).
ab_verdict_report() {
  ab_decisions "$1" | awk -F'\t' -v words="$AB_DECISION_WORDS" -v quick="$quick" \
    -v script="${SCRIPT_DIR#"$ROOT"/}/$SCRIPT_NAME" '
    function info(a, b) { return (a == "informational" || b == "informational") ? " (informational)" : "" }
    BEGIN { nw = split(words, w, " "); for (i = 1; i <= nw; i++) at[w[i]] = 3 + i }
    $1 == "C" {
      nc++; st[$11]++; i = info($6, $9)
      row[nc] = sprintf("  %-24s %-8s %-16s %-30s %-30s %s", $2, $3, $4, $5 " " $7 " " $6, $8 " " $10 " " $9, $11 i)
      # An informational candidate is never adopted: its disagreement is
      # reported, but it does not make the measurement incomplete.
      if ($11 == "DISAGREE" && i != "") { ninfo++; inote = inote (inote == "" ? "" : "; ") $2 " " $3 " " $4 ": pass1 " $5 ", pass2 " $8 }
      else if ($11 == "DISAGREE") miss[++nm] = "step 5: the two passes disagree on " $2 " " $3 " " $4 i ": pass1 " $5 ", pass2 " $8
      if ($11 == "MISSING") miss[++nm] = "step 5: " $2 " " $3 " " $4 i " has a verdict in " ($5 == "-" ? "pass2" : "pass1") " only"
    }
    $1 == "N" {
      t = $2 ": " $3 " verdicts ("; sep = ""
      for (i = 1; i <= nw; i++) { t = t sep w[i] " " $(at[w[i]]); sep = ", " }
      if ($(4 + nw) > 0) t = t ", other words " $(4 + nw)
      pass[$2] = t "), " $(7 + nw) " invalid cases in " $(6 + nw) " of " $(5 + nw) " group runs" ($(8 + nw) > 0 ? ", " $(8 + nw) " repeated verdicts" : "")
      tot += $3; vs[$2] = $3; iv[$2] = $(at["INVALID"]); invalid += iv[$2]; smoke += $(at["SMOKE"])
      cases += $(7 + nw); bad += $(4 + nw) + $(8 + nw)
    }
    $1 == "X" { bad += $2; stray = $2 }
    END {
      if (invalid > 0 || cases > 0) miss[++nm] = "step 5: " invalid " of " tot " A/B verdicts are INVALID (pass1 " iv["pass1"] " of " vs["pass1"] ", pass2 " iv["pass2"] " of " vs["pass2"] ") and " cases " cases had the control outside the limit: the machine was not quiet"
      again = (invalid > 0 || cases > 0 || st["DISAGREE"] - ninfo > 0 || st["MISSING"] > 0)
      if (nc == 0) miss[++nm] = "step 5 wrote no A/B verdict (no ABV line in ab.tsv; see the 5-ab-* lines)"
      if (!quick && smoke > 0) miss[++nm] = "step 5: " smoke " verdicts are SMOKE in a full run, which is not decision grade (see the 5-ab-* lines)"
      if (again) miss[++nm] = "step 5 is therefore not a decision: stop every other program on the machine and run step 5 again (" script " --only 5, about 2.5 hours); that run moves this one to target/x86-runbook-prev/, so send back both archives"
      detail = (nc + 0) " candidates: " (st["agree"] + 0) " agree, " (st["DISAGREE"] + 0) " disagree, " (st["INVALID"] + 0) " INVALID, " (st["MISSING"] + 0) " in one pass only; " pass["pass1"] "; " pass["pass2"]
      if (ninfo > 0) detail = detail "; informational candidates whose passes disagree (never adopted, not judged): " inote
      if (bad > 0) detail = detail "; " bad " verdicts or lines step 5 cannot read (unknown decision words, repeated verdicts, " stray " ABV lines without a pass or a column header)"
      # A --quick run is smoke grade (the harness says SMOKE whatever the
      # controls did): its verdicts decide nothing, so they are not judged.
      if (quick) { nm = 0; detail = "--quick: smoke verdicts are not judged; " detail }
      print "V\t" (bad > 0 ? "FAIL" : (quick ? "INFO" : (nm > 0 ? "WARN" : "PASS"))) "\t" detail
      for (k = 1; k <= nm; k++) print "M\t" miss[k]
      print "T\t" sprintf("  %-24s %-8s %-16s %-30s %-30s %s", "group", "run", "candidate", "pass1: decision geo partial", "pass2: decision geo partial", "both")
      for (k = 1; k <= nc; k++) print "T\t" row[k]
      if (nc == 0) print "T\t  none: ab.tsv has no A/B verdict (ABV line)"
      print "T\t  " pass["pass1"]; print "T\t  " pass["pass2"]
    }'
}

# Records 5-ab-verdicts and adds the missing parts of ab_verdict_report.
ab_verdict_unit() {
  local report line m
  report="$(ab_verdict_report "$AB_OUT")" || report="V${TAB}FAIL${TAB}the verdicts in ab.tsv could not be read"
  line="$(printf '%s\n' "$report" | awk -F'\t' '$1 == "V" { print $2 "\t" $3; exit }')"
  while IFS= read -r m; do [ -z "$m" ] || add_missing "$m"; done <<EOF
$(printf '%s\n' "$report" | awk -F'\t' '$1 == "M" { print $2 }')
EOF
  record 5-ab-verdicts "${line%%"$TAB"*}" - 0 "${line#*"$TAB"}"
}
