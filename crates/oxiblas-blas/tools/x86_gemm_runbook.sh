#!/usr/bin/env bash
# x86_gemm_runbook.sh -- the x86-64 Linux measurements of the 0.2.4 GEMM work.
#
# oxiblas 0.2.4 adds GEMM code paths for x86-64 (an f32 16x6 AVX2 micro-kernel,
# AVX2 small-matrix kernels, AVX2 thin-product kernels, a per-tile parallel
# packing scope, B-block caps for short, wide products). They sit behind a
# policy: the shipped default (LEGACY) reproduces 0.2.3, and the new behaviour
# is built with RUSTFLAGS="--cfg oxiblas_gemm_candidate". Each default is
# switched on only where a measurement on real hardware passes a fixed rule.
# This script collects everything needed for that decision in one run on an
# x86-64 Linux machine with AVX2 and FMA, and packs it into one archive.
#
# Usage (from anywhere; the checkout is the one the script is in):
#   crates/oxiblas-blas/tools/x86_gemm_runbook.sh [options]
# For a long unattended run:
#   nohup crates/oxiblas-blas/tools/x86_gemm_runbook.sh > runbook.out 2>&1 &
#
# What to send back: target/x86-runbook/x86-runbook-results.tar.gz
#
# Steps (each command has its own log in target/x86-runbook/logs/ and a line
# in summary.txt; a failed step does not stop the ones after it):
#   1  machine.txt: lscpu, kernel release, rustc -vV, cargo -V, CPU frequency
#      governor, online CPUs, physical cores, memory, OpenBLAS library, and
#      whether a cargo config file sets rustflags (no host name).
#   2  Tests under the default (LEGACY) policy:
#        cargo test -p oxiblas-blas --release [--features F]
#      for F in (none), parallel, max-simd-256, max-simd-128, force-scalar.
#      The parallel run adds `-- --show-output` (for step 4).
#   3  The same under the candidate policy (RUSTFLAGS="--cfg
#      oxiblas_gemm_candidate", CARGO_TARGET_DIR=target/cand, for these
#      commands only), then the workspace tests under the candidate policy:
#        cargo test --workspace --exclude oxiblas-benchmarks --release
#   4  AVX2 coverage of the parallel runs of steps 2 and 3, read from their
#      logs: the runs passed, no AVX2-gated test printed a SKIPPED line or hit
#      the OXIBLAS_REQUIRE_AVX2 failure, and the number of passed tests whose
#      names contain "avx2" (recorded; it must not be 0).
#   5  The in-crate A/B harness (cargo test -p oxiblas-blas --release
#      --features parallel --lib -- --ignored --test-threads=1 --nocapture
#      <groups>): built first, then the ab_seq groups pinned to one core with
#      `taskset -c 2` (another CPU when 2 is not available to the process),
#      then the ab_par groups with OXIBLAS_AB_THREADS = 2, 4, 6, 8 and the
#      number of physical cores (deduplicated, at most the online CPUs). All
#      result lines are appended to target/x86-runbook/ab.tsv (OXIBLAS_AB_OUT).
#   6  The OpenBLAS comparison benches (crates/oxiblas-benchmarks, bench
#      `comparison`, feature compare-openblas): the default policy saved as
#      the criterion baseline `legacy`, then the candidate policy (in
#      target/cand) compared with `--baseline legacy`. Both builds share one
#      criterion directory, target/x86-runbook/criterion (CRITERION_HOME), so
#      the comparison finds the baseline. If the OpenBLAS library is missing,
#      the step is recorded as skipped with the reason.
#   7  summary.txt and the archive (always written, also after a failure or an
#      interruption).
# Throughout, RUSTFLAGS (and the other rustflags variables), CARGO_TARGET_DIR,
# CRITERION_HOME, RAYON_NUM_THREADS and every OXIBLAS_* variable of the
# caller are unset, and OXIBLAS_REQUIRE_AVX2=1 is set: an AVX2 test that
# would skip fails instead.
#
# Needs:
#   - Linux on x86-64 with AVX2 and FMA (checked in /proc/cpuinfo).
#   - A Rust toolchain (rustup; Rust 1.89 or newer) and a C linker
#     (Debian/Ubuntu: build-essential). cargo fetches crates from crates.io.
#   - util-linux for taskset and lscpu (Debian/Ubuntu: util-linux, installed
#     by default). Without taskset step 5 runs unpinned (noted in the summary).
#   - Optionally OpenBLAS for step 6 (Debian/Ubuntu: libopenblas0). Its path is
#     OPENBLAS_LIB when set, else /usr/lib/x86_64-linux-gnu/libopenblas.so.0.
#   - About 10 GB of free disk space under target/, and for step 6 about 5 GB
#     of free memory (one thin bench operand is 2 GiB).
#
# Duration (full run on a current 8-core desktop; estimates, not measured on
# such a machine): step 2 about 20-40 minutes, step 3 about 30-60 minutes,
# step 4 seconds, step 5 probably 1-2 hours (the harness runs at least 21
# interleaved rounds per case), step 6 about 1.5-2 hours (312 benchmarks of
# about 8-10 s each, twice, plus two release builds). Altogether 4-6 hours.
# --quick takes about 15-30 minutes. Leave the machine otherwise idle: steps 5
# and 6 are timings.
#
# Options:
#   --only LIST        run only these steps (comma-separated numbers 1-6,
#                      e.g. --only 5 or --only 2,3,4); step 7 always runs.
#                      Step 4 reads the logs of steps 2 and 3 of the same run.
#                      Every run, a partial one too, first deletes ALL earlier
#                      results: send (or copy away) the previous archive
#                      before you rerun a step.
#   --skip LIST        run all steps but these
#   --quick            smoke mode: steps 2 and 3 run only the parallel feature
#                      set with the test name filters `avx2` and `gemm_plan`
#                      (no workspace run), step 5 sets OXIBLAS_AB_QUICK=1 and
#                      OXIBLAS_AB_ROUNDS=3, and step 6 runs the benches in
#                      criterion's test mode (each benchmark once, no timing)
#   --bench-filter RE  pass RE to criterion in step 6 (a regular expression
#                      over benchmark names, e.g. 'thin|short_wide')
#   --allow-no-avx2    a testing aid, not for the real run: run on a machine or
#                      an emulator without AVX2/FMA (or not on x86-64 Linux at
#                      all). OXIBLAS_REQUIRE_AVX2 is then not set, AVX2 tests
#                      skip, and step 4 reports WAIVED instead of failing.
#   --dry-run          print every command of every selected step with its
#                      environment and run nothing; nothing is written (no
#                      files at all, also not in target/x86-runbook/)
#   -h, --help         print this text
#
# Environment:
#   OPENBLAS_LIB       the OpenBLAS shared library for step 6 (default
#                      /usr/lib/x86_64-linux-gnu/libopenblas.so.0)
#   CARGO_BUILD_JOBS   passed to cargo as it is
#
# Output, in target/x86-runbook/ of this checkout (and nowhere else, apart
# from cargo's build directories target/ and target/cand/):
#   logs/<step>.log                 the complete output of each command
#   machine.txt                     step 1
#   ab.tsv                          step 5 (lines starting AB, AB# or ABV)
#   criterion/                      step 6 (criterion's data and reports)
#   summary.txt                     one line per command: name, verdict, exit
#                                   code, seconds, test counts or reason
#   x86-runbook-results.tar.gz      the file to send back: summary.txt,
#                                   machine.txt, ab.tsv, the logs and the
#                                   criterion estimates (JSON)
# The results of an earlier run are deleted at the start (except with
# --dry-run), so after any exit the archive, if present, belongs to this run.
# The host name is not recorded, and in the archived copies the checkout's
# absolute path reads `<checkout>` and the home directory `<home>`. Nothing is
# sent anywhere.
#
# Exit status: 0 when every command passed (or was skipped or waived), 1 when
# one failed, 2 for a usage error or the wrong kind of machine, 128 + the
# signal number after INT, TERM or HUP. On INT, TERM or HUP the running
# command and its children are stopped, and summary.txt and the archive are
# written with the status INTERRUPTED.
#---
set -euo pipefail

readonly DEFAULT_OPENBLAS_LIB="/usr/lib/x86_64-linux-gnu/libopenblas.so.0"
readonly CANDIDATE_RUSTFLAGS="--cfg oxiblas_gemm_candidate"
readonly FEATURE_SETS="none parallel max-simd-256 max-simd-128 force-scalar"
readonly AB_SEQ_GROUPS="ab_seq_f32_kernel ab_seq_small_simd ab_seq_small_trans_copy ab_seq_thin ab_seq_b_cap ab_seq_pack_cache"
readonly AB_PAR_GROUPS="ab_par_pack_scope ab_par_thin ab_par_b_cap"
readonly AB_THREAD_CANDIDATES="2 4 6 8"
readonly QUICK_FILTERS="avx2 gemm_plan"
readonly SKIP_TEXT="SKIPPED (not on the AVX2+FMA tier)"
readonly REQUIRE_TEXT="OXIBLAS_REQUIRE_AVX2 is set, but"

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

# ---------------------------------------------------------------- arguments
# --help and --dry-run first, before anything is touched (an option's value
# is skipped).
dry_run=0
arg_index=1
while [ "$arg_index" -le "$#" ]; do
  case "${!arg_index}" in
    -h|--help) print_help; exit 0 ;;
    --dry-run) dry_run=1 ;;
    --only|--skip|--bench-filter) arg_index=$((arg_index + 1)) ;;
  esac
  arg_index=$((arg_index + 1))
done

# ------------------------------------------------------------------ checkout
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
[ -f "$ROOT/Cargo.toml" ] && [ -f "$ROOT/crates/oxiblas-blas/Cargo.toml" ] \
  && [ -f "$ROOT/crates/oxiblas-benchmarks/Cargo.toml" ] \
  || die 2 "cannot find the oxiblas workspace above $SCRIPT_DIR"
OUT="$ROOT/target/x86-runbook"
LOG_DIR="$OUT/logs"
UNITS="$OUT/.units"
AB_OUT="$OUT/ab.tsv"
CRITERION_DIR="$OUT/criterion"
ARCHIVE="$OUT/x86-runbook-results.tar.gz"
CAND_TARGET="$ROOT/target/cand"

if [ "$dry_run" = 0 ]; then
  # The results of an earlier run go now, before any check that can fail (a
  # usage error included), so that no old archive is left behind by an early
  # exit.
  rm -rf "$OUT"
fi

quick=0
allow_no_avx2=0
only_list=""
skip_list=""
bench_filter=""

# Checks a step list ("2,3" or "5") and prints it as " 2 3 ".
parse_steps() {
  local list="$1" option="$2" item out=" "
  [ -n "$list" ] || die 2 "$option needs a list of step numbers (1-6)"
  for item in $(printf '%s' "$list" | tr ',' ' '); do
    case "$item" in
      [1-6]) out="$out$item " ;;
      7) die 2 "$option: step 7 (summary and archive) always runs" ;;
      *) die 2 "$option: '$item' is not a step number (1-6)" ;;
    esac
  done
  printf '%s\n' "$out"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) shift ;;
    --quick) quick=1; shift ;;
    --allow-no-avx2) allow_no_avx2=1; shift ;;
    --only)
      [ "$#" -ge 2 ] || die 2 "--only needs a value"
      only_list="$(parse_steps "$2" --only)"; shift 2 ;;
    --only=*) only_list="$(parse_steps "${1#--only=}" --only)"; shift ;;
    --skip)
      [ "$#" -ge 2 ] || die 2 "--skip needs a value"
      skip_list="$(parse_steps "$2" --skip)"; shift 2 ;;
    --skip=*) skip_list="$(parse_steps "${1#--skip=}" --skip)"; shift ;;
    --bench-filter)
      [ "$#" -ge 2 ] || die 2 "--bench-filter needs a value"
      bench_filter="$2"; shift 2 ;;
    --bench-filter=*) bench_filter="${1#--bench-filter=}"; shift ;;
    *) die 2 "unknown option: $1 (see --help)" ;;
  esac
done
if [ -n "$only_list" ] && [ -n "$skip_list" ]; then
  die 2 "--only and --skip cannot be combined"
fi

want_step() {
  case "$only_list" in
    '') ;;
    *" $1 "*) ;;
    *) return 1 ;;
  esac
  case "$skip_list" in
    *" $1 "*) return 1 ;;
  esac
  return 0
}

# --------------------------------------------------------------- the machine
host_os="$(uname -s)"
host_arch="$(uname -m)"
cpu_flags=""
if [ -r /proc/cpuinfo ]; then
  cpu_flags="$(awk -F: '/^flags[[:space:]]*:/ { print $2; exit }' /proc/cpuinfo)"
fi
has_flag() {
  case " $cpu_flags " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}
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
  if [ "$dry_run" = 1 ]; then
    note "note: $machine_problem; a real run would stop here (without --allow-no-avx2)"
  elif [ "$allow_no_avx2" = 1 ]; then
    note "warning: $machine_problem; going on because of --allow-no-avx2 (a testing aid: these results cannot decide the defaults)"
  else
    die 2 "$machine_problem (--allow-no-avx2 overrides this check, for testing only)"
  fi
fi

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

# 2, 4, 6, 8 and the physical core count, deduplicated, at most the online
# CPUs, ascending and comma-separated.
ab_thread_list() {
  local online phys t
  online="$(online_cpus)"
  phys="$(physical_cores)"
  for t in $AB_THREAD_CANDIDATES "$phys"; do
    if [ "$t" -le "$online" ]; then printf '%s\n' "$t"; fi
  done | sort -n -u | paste -s -d, -
}

N_ONLINE="$(online_cpus)"
N_PHYSICAL="$(physical_cores)"
PIN_CPU="$(pin_cpu)"
AB_THREADS="$(ab_thread_list)"
[ -n "$AB_THREADS" ] || AB_THREADS="1"
have_taskset=0
command -v taskset >/dev/null 2>&1 && have_taskset=1
OPENBLAS_PATH="${OPENBLAS_LIB:-$DEFAULT_OPENBLAS_LIB}"

# ---------------------------------------------------------- the environment
removed_env=""
for name in $(env | sed -n 's/^\(OXIBLAS_[A-Za-z0-9_]*\)=.*/\1/p' | sort -u) \
    RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS RUSTDOCFLAGS \
    CARGO_TARGET_DIR CARGO_BUILD_TARGET_DIR CRITERION_HOME RAYON_NUM_THREADS; do
  if [ -n "${!name+set}" ]; then
    removed_env="$removed_env $name"
    unset "$name"
  fi
done
removed_env="${removed_env# }"
export CARGO_TERM_COLOR=never
if [ "$allow_no_avx2" = 1 ]; then
  require_avx2_text="not set (--allow-no-avx2)"
else
  export OXIBLAS_REQUIRE_AVX2=1
  require_avx2_text="1"
fi

# ------------------------------------------------------------------- helpers
child_pid=""
tee_pid=""
current_unit=""
current_start=0
finalized=0
interrupted=""
any_fail=0

# Quotes a command for display.
show_cmd() {
  local out="" arg
  for arg in "$@"; do
    case "$arg" in
      ''|*[!A-Za-z0-9_.,:=/+@%-]*) out="$out $(printf '%q' "$arg")" ;;
      *) out="$out $arg" ;;
    esac
  done
  printf '%s\n' "${out# }"
}

# Appends one result line: id, verdict, exit code, seconds, details.
record() {
  local id="$1" verdict="$2" rc="$3" secs="$4" detail="$5"
  if [ "$dry_run" = 1 ]; then
    if [ "$verdict" = SKIPPED ]; then printf '\n## %s: skipped - %s\n' "$id" "$detail"; fi
    return 0
  fi
  case "$verdict" in FAIL) any_fail=1 ;; esac
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$verdict" "$rc" "$secs" "$detail" >> "$UNITS"
  printf '== %s: %s (exit %s, %ss)%s\n' "$id" "$verdict" "$rc" "$secs" "${detail:+ - $detail}" >&2
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

# Occurrences of a fixed text in a file, anywhere in a line.
count_text() {
  { grep -o -F -e "$1" "$2" 2>/dev/null || true; } | grep -c '' || true
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
    printf '\n## %s: %s\n(cd %s &&\n  %s)\n' "$id" "$title" "$(printf '%q' "$ROOT")" "$(show_cmd "$@")"
    return 0
  fi
  {
    printf '== step   : %s (%s)\n' "$id" "$title"
    printf '== started: %s UTC\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
    printf '== command: %s\n' "$(show_cmd "$@")"
    printf '== env    : OXIBLAS_REQUIRE_AVX2=%s CARGO_TERM_COLOR=never\n' "$require_avx2_text"
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

# Verdict of a cargo test unit from its log.
#   finish_tests <id> [<reason when no test ran>]
finish_tests() {
  local id="$1" none_reason="${2:-no test ran (check the filter)}"
  local log="$LOG_DIR/$id.log" runs passed failed ignored verdict detail
  [ "$dry_run" = 1 ] && return 0
  read -r runs passed failed ignored <<EOF
$(test_counts "$log")
EOF
  detail="tests $passed passed, $failed failed, $ignored ignored ($runs test binaries)"
  verdict=PASS
  if [ "$unit_rc" != 0 ]; then
    verdict=FAIL
    detail="$detail; cargo exited with $unit_rc"
  elif [ "$passed" = 0 ]; then
    verdict=FAIL
    detail="$detail; $none_reason"
  fi
  record "$id" "$verdict" "$unit_rc" "$unit_secs" "$detail"
}

# The cargo arguments of a feature set ("none" = default features).
feature_args() {
  if [ "$1" != none ]; then printf -- '--features\n%s\n' "$1"; fi
}

# ------------------------------------------------------------------- steps
step_machine() {
  local id="1-machine"
  if [ "$dry_run" = 1 ]; then
    printf '\n## %s: machine.txt\n' "$id"
    printf '  lscpu; uname -srm; rustc -vV; cargo -V; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor;\n'
    printf '  getconf _NPROCESSORS_ONLN; lscpu -p=CORE,SOCKET; grep MemTotal /proc/meminfo; free -h;\n'
    printf '  dpkg-query -W libopenblas*; grep rustflags in cargo config files\n'
    return 0
  fi
  local start="$SECONDS" file="$OUT/machine.txt" rc=0 governors cfg
  {
    printf 'oxiblas x86 GEMM runbook: machine\n'
    printf 'date (UTC)      : %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
    printf 'system          : %s %s %s\n' "$(uname -s)" "$(uname -r)" "$(uname -m)"
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
    printf 'removed env     : %s\n' "${removed_env:-none}"
    printf 'OXIBLAS_REQUIRE_AVX2: %s\n' "$require_avx2_text"
    printf 'CARGO_BUILD_JOBS: %s\n' "${CARGO_BUILD_JOBS:-unset}"
    # RUSTFLAGS (set for the candidate builds) replaces build.rustflags from
    # the config files, so a config that sets rustflags would make the two
    # policies' builds differ in more than the cfg.
    for cfg in "$ROOT/.cargo/config.toml" "$ROOT/.cargo/config" \
        "${CARGO_HOME:-${HOME:-}/.cargo}/config.toml" "${CARGO_HOME:-${HOME:-}/.cargo}/config"; do
      if [ -f "$cfg" ] && grep -q 'rustflags' "$cfg"; then
        printf 'WARNING         : %s sets rustflags; the candidate builds (RUSTFLAGS) ignore them, the legacy builds use them:\n' "$cfg"
        grep -n 'rustflags' "$cfg" | sed 's/^/                  /'
      fi
    done
    printf '\n== rustc -vV\n'
    rustc -vV 2>&1 || printf 'rustc failed\n'
    printf '\n== cargo -V\n'
    cargo -V 2>&1 || printf 'cargo failed\n'
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
  local step="$1" policy="$2" fs id filters=() env_cmd=(env) args sets="$FEATURE_SETS"
  if [ "$policy" = cand ]; then
    env_cmd=(env "RUSTFLAGS=$CANDIDATE_RUSTFLAGS" "CARGO_TARGET_DIR=$CAND_TARGET")
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
    if [ "$fs" = parallel ] || [ "${#filters[@]}" -gt 0 ]; then
      args+=(--)
      [ "$fs" = parallel ] && args+=(--show-output)
      [ "${#filters[@]}" -gt 0 ] && args+=("${filters[@]}")
    fi
    run_cmd "$id" "oxiblas-blas tests, $policy policy, features $fs" "${env_cmd[@]}" "${args[@]}"
    finish_tests "$id"
  done
  if [ "$policy" = cand ]; then
    id="$step-$policy-workspace"
    if [ "$quick" = 1 ]; then
      record "$id" SKIPPED - 0 "quick mode runs no workspace tests"
    else
      run_cmd "$id" "workspace tests (without the benchmarks), candidate policy" \
        "${env_cmd[@]}" cargo test --workspace --exclude oxiblas-benchmarks --release
      finish_tests "$id"
    fi
  fi
}

# Step 4: AVX2 coverage in the logs of the parallel runs.
step_avx2() {
  local policy log id runs passed failed ignored skips requires avx2_ok verdict detail rc
  for policy in legacy cand; do
    id="4-avx2-$policy"
    if [ "$policy" = legacy ]; then log="$LOG_DIR/2-legacy-parallel.log"; else log="$LOG_DIR/3-cand-parallel.log"; fi
    if [ "$dry_run" = 1 ]; then
      printf '\n## %s: read %s\n' "$id" "${log#"$ROOT"/}"
      printf "  count '%s' and '%s' lines and the passed tests named *avx2*\n" "$SKIP_TEXT" "$REQUIRE_TEXT"
      continue
    fi
    if [ ! -f "$log" ]; then
      record "$id" SKIPPED - 0 "no log of the parallel test run of this run (run step $([ "$policy" = legacy ] && echo 2 || echo 3) with step 4)"
      continue
    fi
    read -r runs passed failed ignored <<EOF
$(test_counts "$log")
EOF
    rc="$(sed -n 's/^== exit   : \([0-9]*\) .*/\1/p' "$log" | tail -n 1)"
    skips="$(count_text "$SKIP_TEXT" "$log")"
    requires="$(count_text "$REQUIRE_TEXT" "$log")"
    avx2_ok="$(grep -c -E '^test [^ ]*avx2[^ ]* \.\.\. ok$' "$log" || true)"
    detail="$avx2_ok avx2 tests passed, $skips SKIPPED lines, $requires REQUIRE_AVX2 failures (run: $passed passed, $failed failed)"
    verdict=PASS
    if [ "${rc:-1}" != 0 ] || [ "$failed" != 0 ] || [ "$passed" = 0 ]; then
      verdict=FAIL; detail="$detail; the test run itself did not pass"
    elif [ "$skips" != 0 ] || [ "$requires" != 0 ] || [ "$avx2_ok" = 0 ]; then
      if [ "$allow_no_avx2" = 1 ]; then
        verdict=WAIVED; detail="$detail; allowed by --allow-no-avx2"
      else
        verdict=FAIL
      fi
    fi
    record "$id" "$verdict" "${rc:--}" 0 "$detail"
  done
}

# Step 5: the A/B harness.
step_ab() {
  local base=(cargo test -p oxiblas-blas --release --features parallel --lib)
  local run_args=(-- --ignored --test-threads=1 --nocapture)
  local ab_env=(env "OXIBLAS_AB_OUT=$AB_OUT") group seq_groups=() par_groups=() pin=() before after
  local harness="$ROOT/crates/oxiblas-blas/src/level3/gemm_ab_tests.rs" none_reason
  for group in $AB_SEQ_GROUPS; do seq_groups+=("$group"); done
  for group in $AB_PAR_GROUPS; do par_groups+=("$group"); done
  if [ "$quick" = 1 ]; then
    ab_env+=(OXIBLAS_AB_QUICK=1 OXIBLAS_AB_ROUNDS=3)
  fi
  if [ -f "$harness" ]; then
    none_reason="no ab_* test ran although ${harness#"$ROOT"/} exists (were the groups renamed?)"
  else
    none_reason="no ab_* test in this checkout: the A/B harness (${harness#"$ROOT"/}) is missing"
  fi

  # Built unpinned first, so that the pinned run compiles nothing.
  run_cmd 5-ab-build "build the A/B harness (no run)" "${base[@]}" --no-run
  record 5-ab-build "$([ "$unit_rc" = 0 ] && echo PASS || echo FAIL)" "$unit_rc" "$unit_secs" "lib test binary built"
  [ "$dry_run" = 1 ] || [ "$unit_rc" = 0 ] || return 0

  # A dry run shows the pinning a Linux run does, also where taskset is missing.
  if [ "$have_taskset" = 1 ] || [ "$dry_run" = 1 ]; then
    pin=(taskset -c "$PIN_CPU")
  fi
  before="$(ab_lines)"
  run_cmd 5-ab-seq "A/B sequential groups, pinned to CPU $PIN_CPU" \
    "${ab_env[@]}" ${pin[@]+"${pin[@]}"} "${base[@]}" "${run_args[@]}" "${seq_groups[@]}"
  finish_ab 5-ab-seq "$before" "$none_reason"
  if [ "$have_taskset" = 0 ] && [ "$dry_run" = 0 ]; then
    record 5-ab-seq-pin WARN - 0 "taskset not found: the sequential groups ran unpinned"
  fi

  before="$(ab_lines)"
  run_cmd 5-ab-par "A/B parallel groups, threads $AB_THREADS" \
    "${ab_env[@]}" "OXIBLAS_AB_THREADS=$AB_THREADS" "${base[@]}" "${run_args[@]}" "${par_groups[@]}"
  finish_ab 5-ab-par "$before" "$none_reason"
}

ab_lines() {
  if [ -f "$AB_OUT" ]; then grep -c '' "$AB_OUT" || true; else echo 0; fi
}

finish_ab() {
  local id="$1" before="$2" none_reason="$3" after
  [ "$dry_run" = 1 ] && return 0
  after="$(ab_lines)"
  finish_tests "$id" "$none_reason"
  # Adds the number of result lines to the line just recorded.
  awk -F'\t' -v OFS='\t' -v id="$id" -v n="$((after - before))" \
    '$1 == id { $5 = $5 "; " n " lines appended to ab.tsv" } { print }' "$UNITS" > "$UNITS.tmp"
  mv "$UNITS.tmp" "$UNITS"
}

# Step 6: the OpenBLAS comparison benches, legacy baseline then candidate.
step_bench() {
  local bench=(cargo bench -p oxiblas-benchmarks --bench comparison --features compare-openblas)
  local common=(env "OPENBLAS_LIB=$OPENBLAS_PATH" "CRITERION_HOME=$CRITERION_DIR")
  local crit_legacy=(-- --save-baseline legacy) crit_cand=(-- --baseline legacy) extra=() verdict detail
  if [ "$quick" = 1 ]; then extra+=(--test); fi
  if [ -n "$bench_filter" ]; then extra+=("$bench_filter"); fi
  if [ ! -e "$OPENBLAS_PATH" ]; then
    if [ "$dry_run" = 1 ]; then
      note "note: no OpenBLAS library at $OPENBLAS_PATH; a real run would record step 6 as skipped"
    else
      record 6-bench SKIPPED - 0 "no OpenBLAS library at $OPENBLAS_PATH (Debian/Ubuntu: apt install libopenblas0; or set OPENBLAS_LIB to the library)"
      return 0
    fi
  fi
  [ "$dry_run" = 1 ] || rm -rf "$CRITERION_DIR"

  run_cmd 6-bench-legacy "comparison benches, legacy policy, saved as baseline legacy" \
    "${common[@]}" "${bench[@]}" "${crit_legacy[@]}" ${extra[@]+"${extra[@]}"}
  finish_bench 6-bench-legacy
  if [ "$dry_run" = 0 ] && [ "$unit_rc" != 0 ]; then
    # Without a complete baseline `--baseline legacy` would stop at the first
    # missing benchmark; save the candidate's own baseline instead.
    crit_cand=(-- --save-baseline candidate)
  fi
  run_cmd 6-bench-cand "comparison benches, candidate policy, compared with baseline legacy" \
    "${common[@]}" "RUSTFLAGS=$CANDIDATE_RUSTFLAGS" "CARGO_TARGET_DIR=$CAND_TARGET" \
    "${bench[@]}" "${crit_cand[@]}" ${extra[@]+"${extra[@]}"}
  finish_bench 6-bench-cand
}

finish_bench() {
  local id="$1" log="$LOG_DIR/$1.log" timed tested panics faster slower verdict detail
  [ "$dry_run" = 1 ] && return 0
  # A `time:` line that follows a `change:` line is the change against the
  # baseline, not a measurement.
  timed="$(awk '/time: +\[/ && !after_change { n++ } { after_change = ($0 ~ /change:[[:space:]]*$/) } END { print n + 0 }' "$log")"
  tested="$(grep -c -x 'Success' "$log" || true)"
  panics="$(grep -c 'panicked at' "$log" || true)"
  faster="$(grep -c 'Performance has improved' "$log" || true)"
  slower="$(grep -c 'Performance has regressed' "$log" || true)"
  detail="$timed benchmarks timed, $tested tested (--test), $panics panics"
  if [ "$id" = 6-bench-cand ]; then
    detail="$detail; against the baseline $faster faster, $slower slower (criterion's p < 0.05 and noise threshold)"
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

# Copies a text file with the checkout path replaced by <checkout> and the
# home directory by <home> (literally: the paths come in through ENVIRON and
# are found with index(), not as regular expressions).
scrub_copy() {
  RB_ROOT="$ROOT" RB_HOME="${HOME:-}" awk '
    function swap(line, from, to,   out, i, n) {
      n = length(from); out = ""
      while (n > 0 && (i = index(line, from)) > 0) {
        out = out substr(line, 1, i - 1) to
        line = substr(line, i + n)
      }
      return out line
    }
    BEGIN { root = ENVIRON["RB_ROOT"]; home = ENVIRON["RB_HOME"]; if (home == "/") home = "" }
    { line = swap($0, root, "<checkout>"); print swap(line, home, "<home>") }' "$1" > "$2"
}

finalize() {
  local status_line="$1" overall f rel
  [ "$finalized" = 1 ] && return 0
  finalized=1
  [ "$dry_run" = 1 ] && return 0
  [ -d "$OUT" ] || return 0
  git_info
  overall="PASS"
  [ "$any_fail" = 1 ] && overall="FAIL"
  [ -n "$status_line" ] && overall="$status_line"
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
    printf 'steps        : %s\n' "$(for s in 1 2 3 4 5 6; do want_step "$s" && printf '%s ' "$s"; done; printf '7')"
    printf 'machine check: %s\n' "$([ "$machine_ok" = 1 ] && echo 'x86-64 Linux with avx2 and fma' || printf '%s' "$machine_problem")"
    printf 'require avx2 : OXIBLAS_REQUIRE_AVX2=%s\n' "$require_avx2_text"
    printf 'candidate    : RUSTFLAGS="%s", CARGO_TARGET_DIR=target/cand\n' "$CANDIDATE_RUSTFLAGS"
    printf 'removed env  : %s\n' "${removed_env:-none}"
    printf 'openblas     : %s\n' "$OPENBLAS_PATH"
    printf 'git commit   : %s\n' "$git_commit"
    printf 'git dirty    : %s\n' "$git_dirty"
    printf '\n%-22s %-11s %-5s %7s  %s\n' "step" "verdict" "exit" "seconds" "details"
    awk -F'\t' '{ printf "%-22s %-11s %-5s %7s  %s\n", $1, $2, $3, $4, $5 }' "$UNITS"
    printf '\noverall      : %s\n' "$overall"
  } > "$OUT/summary.txt"

  rm -rf "$OUT/.archive"
  mkdir -p "$OUT/.archive/logs"
  scrub_copy "$OUT/summary.txt" "$OUT/.archive/summary.txt"
  for f in machine.txt ab.tsv; do
    [ -f "$OUT/$f" ] && scrub_copy "$OUT/$f" "$OUT/.archive/$f"
  done
  for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] && scrub_copy "$f" "$OUT/.archive/logs/$(basename "$f")"
  done
  if [ -d "$CRITERION_DIR" ]; then
    find "$CRITERION_DIR" -type f -name estimates.json | while IFS= read -r f; do
      rel="criterion/${f#"$CRITERION_DIR"/}"
      mkdir -p "$OUT/.archive/$(dirname "$rel")"
      scrub_copy "$f" "$OUT/.archive/$rel"
    done
  fi
  (cd "$OUT/.archive" && tar -czf "$ARCHIVE" .)
  rm -rf "$OUT/.archive" "$UNITS" "$OUT/.fifo"
  cat "$OUT/summary.txt"
  printf '\nSend back: %s\n' "$ARCHIVE"
}

on_signal() {
  local sig="$1" code="$2" id secs
  trap - INT TERM HUP
  interrupted="SIG$sig"
  id="$current_unit"
  secs=$((SECONDS - current_start))
  stop_child
  if [ -n "$id" ]; then
    record "$id" INTERRUPTED "-" "$secs" "stopped by $interrupted"
    printf '== interrupted by %s\n' "$interrupted" >> "$LOG_DIR/$id.log" 2>/dev/null || true
  fi
  finalize "INTERRUPTED (by $interrupted${id:+ during $id})"
  exit "$code"
}

on_exit() {
  local rc=$?
  stop_child
  if [ "$finalized" = 0 ] && [ "$dry_run" = 0 ] && [ -f "$UNITS" ]; then
    finalize "ABORTED (the script stopped early with exit status $rc)"
  fi
}
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

# Job control: every background command gets a process group of its own.
set -m

# -------------------------------------------------------------------- run
if [ "$dry_run" = 1 ]; then
  printf '# dry run: nothing is run and nothing is written\n'
  printf '# checkout: %s\n' "$ROOT"
  printf '# unset: RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS RUSTDOCFLAGS CARGO_TARGET_DIR CARGO_BUILD_TARGET_DIR CRITERION_HOME RAYON_NUM_THREADS OXIBLAS_*\n'
  printf '# exported for every command: CARGO_TERM_COLOR=never OXIBLAS_REQUIRE_AVX2=%s\n' "$require_avx2_text"
  printf '# online CPUs %s, physical cores %s, A/B pin CPU %s (taskset %s), A/B threads %s\n' \
    "$N_ONLINE" "$N_PHYSICAL" "$PIN_CPU" "$([ "$have_taskset" = 1 ] && echo found || echo missing)" "$AB_THREADS"
fi

if [ "$dry_run" = 0 ]; then
  mkdir -p "$LOG_DIR"
  : > "$UNITS"
fi
want_step 1 && step_machine
want_step 2 && step_tests 2 legacy
want_step 3 && step_tests 3 cand
want_step 4 && step_avx2
want_step 5 && step_ab
want_step 6 && step_bench

if [ "$dry_run" = 1 ]; then
  printf '\n## 7-summary: write target/x86-runbook/summary.txt and %s\n' "${ARCHIVE#"$ROOT"/}"
  exit 0
fi
finalize ""
[ "$any_fail" = 0 ]
