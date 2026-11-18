#!/usr/bin/env bash
# gemm_tiny_call_check.sh -- per-call cost of small GEMM products against a
# base release.
#
# What it measures: the time of one call of the public level-3 GEMM entry
# points (`gemm` for NoTrans/NoTrans, `gemm_transposed` otherwise), f64 and
# f32, on cubic products small enough that the fixed per-call work (the
# dimension checks, the plan of the call, the route decision) and the small
# loops are a visible part of the time: every cube from 1 to 16 for
# NoTrans/NoTrans (nn) and for the transposed combinations (tn, nt, tt),
# 128 cases. The tree under test is compared with a base revision (default
# v0.2.3); every judged case must stay within --limit of it. Information
# only: 48, 64 and 128 cubed (nn, one thread; the blocked route).
#
# Code alignment. Both sides are built with every function and every loop
# aligned to 64 bytes: every build gets RUSTFLAGS="<the caller's RUSTFLAGS>
# -C llvm-args=-align-all-functions=6 -C llvm-args=-align-loops=64"
# (summary.txt prints them; RUSTFLAGS is always set, so rustflags from cargo
# configuration files do not apply, and CARGO_ENCODED_RUSTFLAGS must be
# unset). Without it, where the linker happens to put a function decides how
# fast some of these cases run: the same source built at another path took
# up to 1.8 times as long (f64 tt), and a loop of 0.2.3 inlined into a 0.2.4
# function of another size ran 8 to 14 % slower only because it started at
# another offset from a 64-byte boundary (f32 tn 8 and 11 to 15 cubed). With
# every function, and every loop the compiler finds worth aligning, starting
# on a 64-byte boundary on both sides, the same code runs at nearly the same
# speed wherever it sits, and the comparison measures what the code does.
# Measured with this script in the two bench compositions of the first band
# measurement below, 8 builds per side with these flags and 8 without them:
# without them 67 of the 128 cases had a band below 1.03, with them 124 (120
# after the further measurements with the flags, see MEASURED_BANDS). The builds
# without them were measured in 3 rounds of 80 samples at load averages of
# 224 to 391, those with them in 6 rounds of 200 samples at 62 to 285, so
# part of the difference in that count is noise; the moves of single cases
# are far larger than it: f64 tt 2 to 16 cubed moved by 1.33 to 1.80
# without the flags and by 1.003 to 1.019 with them (14 cubed: 1.045, one
# 0.2.3 build), f64 tn 10 to 15 cubed by 1.08 to 1.20 and 1.002 to 1.009,
# f32 tn 8 and 11 to 15 cubed by 1.08 to 1.13 and 1.003 to 1.005; and the
# unaligned runs judged f32 tn 14 cubed at 1.081 and f64 nn 9 cubed at 1.042
# with every tree build above every base build, which the aligned builds
# measured at 0.997 and 0.999 (over the pooled builds). --no-align builds
# without these flags (to see the placement effects; such a run is a
# diagnosis, and its verdict says so).
#
# Clock. Under load the program runs at a lower clock for seconds at a time,
# or moves to a slower core: whole program runs and parts of runs were 2.5
# to 9 % slower in every case, and at load averages of 15 to 67 only 2 of 48
# runs saw the fastest clock throughout, so the minimum of a build often
# reflected a slower clock (the spreads were then about 1.04 on both sides
# and only 14 of the 128 cases were judged). The benchmark therefore times a
# clock reference before every sample of every case: a chain of dependent
# f64 adds, the same code in every build, whose time per add follows only the
# clock of the core it runs on. Each case of each run is scaled to the
# fastest clock seen in any run of either side: its minimum times R / its
# reference, R being the smallest reference of all runs and the reference of
# a case its fastest sample (so both minima come from the same stretch of
# time). A case run whose reference is over 1.15 R ran that case on a slower
# core (an efficiency core), where the scaling does not hold: it is left out
# and enters no judged figure (its raw minimum would be too slow, which on the
# base side hides a regression of the tree and on the tree side makes a false
# FAIL). A build's time is the median of the remaining scaled minima over its
# rounds: now and then the fastest case sample and the fastest reference
# sample of a run still saw different clocks, which a minimum over the rounds
# would pick up and the median does not. With an even number of remaining
# rounds the median is the mean of the two middle ones; with one remaining
# round that round is the build's time. A build of either side that has no
# run of a case left has no time for it: the case is information in that run,
# its line names the side and the build ("no run with a reference within
# 1.15 R left in base build 2 ..."), shows the fastest time of each side that
# has one, and a "-" for a spread that could not be measured; such a case
# does not by itself make the run INCOMPLETE. So:
#   - every run of a case in one build on a slower core, on one side only or
#     on both: the case is information (a case is judged only when every
#     build of both sides kept at least one run of it);
#   - some, not all, runs of a build on a slower core: the build's time is
#     the median of its other runs, and the case is judged as usual;
#   - the fastest reference of the whole run itself from a slower core (every
#     program ran on slower cores): R is that reference, no run is over
#     1.15 R, and both sides are scaled to the same clock as on fast cores;
#   - one build entirely on slower cores in every round (the threshold is R,
#     not that build's own fastest reference): it loses every case, nothing
#     is judged, and the run is INCOMPLETE as when no case is judged;
#   - --quick (one round): a run on a slower core leaves its build without a
#     time, so the case is information, not a false FAIL or a hidden
#     regression; a --quick verdict is still a smoke test only.
# The clock line of summary.txt says how many case runs were left out and how
# many cases lost a build that way. base_ns, tree_ns and the spreads are
# these times; perbuild.txt also has the raw minima (over all rounds, the
# left-out runs included). A --bench without the reference field gets raw
# minima (the minimum over the rounds; nothing is left out), and summary.txt
# says so.
#
# Which cases are judged. Alignment removes most of the placement noise, not
# all of it (absolute addresses still decide, for example, which branches
# share predictor entries). Two rules guard against the rest:
#   - the measured band: every case has a band measured with this script
#     over 12 to 16 builds of the tree and 16 to 20 of the base, in two or
#     three bench compositions (MEASURED_BANDS below, with how and under
#     which load it was measured): what moves a case between builds of the
#     same source, whether placement or load that the clock scaling does
#     not remove; a case whose band is at or above --limit is information
#     in every run;
#   - the spread of the run: each side is built --builds times (default 2),
#     each build from its own copy of the sources at its own path under
#     --out; per case
#       side time = the minimum over the builds of the side (each build's
#                   time being the median over its remaining rounds of the
#                   scaled minima, see "Clock");
#       spread    = the slowest build of a side divided by its fastest;
#     a case is judged only when the spread is below --limit on both sides;
#     otherwise it is information, printed with its spreads.
# The blocked-route cases are always information, and so is a case of which
# a build of either side has no run left (see "Clock"). With --builds 1 no
# spread can be measured: every case with a measured band below --limit is
# judged, and a case without a measured band is information.
#
# Every case line also says, as information, whether every build of the tree
# is slower than every build of the base by more than the limit
# ("every tree build above every base build"); that does not change the
# verdict.
#
# Reading the verdict. A PASS says that every judged case of this run stayed
# within --limit of the base: the judged cases only (the "# judged" line
# counts them; the information cases are not covered), against the base
# revision (not against an earlier state of the tree, so a tree that is
# faster than the base by some margin can lose part of that margin and still
# pass), and only from runs whose clock reference stayed within 1.15 R. A FAIL
# is carried by the judged lines whose ratio is above the limit; before
# believing one, look at which cases these are. The large TT cases (f64_tt_13 to 16, f32_tt_11 to 16)
# have a known two-state: each program run executes them at one of two
# speeds about 3.3 to 3.5 % apart, which one changing from one program run to
# the next, so a build's median can land in either. The spread rule usually
# turns such a case into information; a FAIL carried only by those cases with
# ratios below about 1.04 should be re-run before it is believed; a FAIL
# carried by other cases, or by ratios above that, is not explained by it.
#
# How: one benchmark source (gemm_tiny_call_bench.rs, beside this script; it
# uses only API that exists in both trees; --bench names another one) is
# copied into a generated crate per build under --out. All the generated
# crates have the same package and crate name (the file name of the
# benchmark source), so symbol names and the link order do not differ
# between the sides for that reason. Each depends by path on its build's
# copy of the sources: base-src-N is a `git archive` export of the base
# revision (kept while it is the same commit), tree-src-N a copy of what a
# build of the tree reads, its workspace Cargo.toml and crates/ without
# target/ directories and Cargo.lock files (copied again when one of these
# files changed; other files of the checkout, such as other worktrees under
# .claude/, are never copied; with --builds 1 the tree is used in place).
# All are built with the same release profile (thin LTO, one codegen unit,
# opt-level 3) and the same RUSTFLAGS. The programs then run alternately,
# --rounds times (base 1, tree 1, base 2, tree 2, ...); each program times
# every case in --samples samples (default 200) of about 0.5 ms (the calls
# per sample are calibrated per case), each after a sample of the clock
# reference of about 30 µs.
#
# Why minima over alternating process runs: on a shared or loaded machine a
# single timing, and any mean, carries the noise of everything else that
# runs. The minimum of many short samples is the run least disturbed (a
# short sample fits between two preemptions even when the machine is
# oversubscribed) and repeats to about 0.1 to 0.3 ns on tiny cases;
# alternating whole processes spreads slow phases of the machine (and
# process-level effects) evenly over all builds instead of charging them to
# one; the clock reference removes what a slower clock adds to a minimum.
#
# How long (Apple M3): each build takes 20 seconds to 2 minutes on a first
# run, depending on the load (later runs rebuild only what changed; the
# copies of the tree are rebuilt when the tree changed); each program run
# takes about 18 seconds on a quiet machine, so the default 2 builds per side
# and 5 rounds take about 6 minutes after the builds (8 to 15 minutes in all
# on a first run); at load averages of 100 and more a program run took up to
# 1.5 minutes. --quick (1 round, 20 samples) takes under a minute after the
# builds and is only a smoke test.
#
# Usage: crates/oxiblas-blas/tools/gemm_tiny_call_check.sh [options]
#   --base REV     git revision to compare with (default v0.2.3), exported
#                  with git archive from the checkout that holds this script
#   --tree DIR     the oxiblas tree to measure (default: that checkout)
#   --out DIR      output directory (default <checkout>/target/gemm_tiny_call)
#   --rounds N     alternating rounds (default 5)
#   --samples N    samples per case in each program run (default 200;
#                  fewer make a run shorter on a loaded machine and the
#                  minima less certain)
#   --builds N     builds per side, each at its own path (default 2,
#                  minimum 1; 1 judges by the measured bands alone)
#   --limit RATIO  largest allowed tree/base ratio of a judged case, and the
#                  measured band and the spread from which a case is
#                  information (default 1.03)
#   --bench FILE   another benchmark source (same output format; the
#                  generated crates are named after the file). Two are
#                  beside this script: gemm_thin_call_bench.rs (products
#                  with m == 1 or n == 1) and gemm_tall_call_bench.rs
#                  (NoTrans/Trans products of 8 to 512 rows, depths 1 to
#                  10). The measured bands below are those of the cubes,
#                  so the verdict for these shapes is information
#   --no-align     build without the code alignment (diagnosis only; the
#                  verdict says "(no-align)")
#   --quick        one round of 20 samples, whatever --rounds and --samples
#                  say (smoke test; verdict says "(quick)"; a case run on a
#                  slower core makes that case information)
#   --help         this text
#
# Output: <out>/summary.txt (also printed): lines starting with # (the run
# parameters, the RUSTFLAGS of every build, the column names, the clock
# scaling with the number of case runs left out and of cases that lost a
# build that way, and the cases missing from a run, if any), then one line per
# case
#   <case> <base ns> <tree ns> <tree/base> <judged|info> <base spread>
#   <tree spread> <why judged or not>
# ("-" in place of a time, a ratio or a spread that a case which lost a build
# does not have)
# then the line "# judged: N of M cases (...)" and a last line
# "verdict: PASS", "verdict: FAIL" or "verdict: INCOMPLETE".
# <out>/perbuild.txt has every build's time of every case (as in summary.txt:
# the median over its remaining rounds of the clock-scaled minima; "-" for a
# build with no run of the case left), the two spreads ("-" for a side with
# such a build), and every build's raw minimum over all its rounds (the
# left-out runs included). The logs of the builds and of every run are in
# <out>/logs/.
#
# Exit status: 0 PASS; 1 FAIL (a judged ratio above the limit); 2 usage error
# or missing prerequisite (git, cargo, awk, tar, cksum, the base revision,
# the tree, the benchmark source; CARGO_ENCODED_RUSTFLAGS set); 3 INCOMPLETE
# (a build or a run did not finish, or a case is missing from a run of one
# build; the summary names the case, the side, the build and the rounds that
# have it; or no case was judged, every one being information, as always
# with a --limit of 1.00 or less: a PASS needs at least one judged case; a
# case that only lost a build to slower cores is information, not missing).
#---

set -u
set -o pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

# The code alignment of every build of both sides (see the header).
ALIGN_FLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-loops=64"

# The measured band of every case (the rule in the header): the largest,
# over the two sides and over the measurements below, of the slowest build of
# a side divided by its fastest within one run of this script (each build's
# time computed as in a run of this script), rounded down to 3 decimals, so
# that "band >= limit" compares the measured value for a limit of up to 3
# decimals. Measured on an Apple M3 (rustc 1.95.0) against 0.2.3 and the
# 0.2.4 tree of 2026-11-16, with the code alignment and the clock scaling,
# each build at its own path:
#   - first measurement, 8 builds per side in 6 rounds of 200 samples: 4
#     builds with this bench (load averages 89 to 162) and 4 with a second
#     composition (the same cases in another order, without the
#     blocked-route cases, in a crate of another name; load averages 62 to
#     285), all scaled to one clock;
#   - second measurement, in 5 rounds of 200 samples: 4 builds per side with
#     this bench (load averages 21 to 91), 2 + 2 builds of 0.2.3 against
#     each other with this bench (load averages 9 to 45), and 4 builds per
#     side of a third composition (52 of the cases in another order, sizes
#     falling, in a crate of another name; load averages 9 to 45).
# 120 of the 128 cases have a band below 1.03. The other 8 are information
# in every run with the default limit:
#   f32_nn_1  1.030: the 0.2.4 builds run at one of two speeds 1.03 apart,
#             the same in every round of a build (6 of the 8 builds slow);
#   f64_nt_2  1.226: in the third composition two of the four 0.2.3 builds
#             are 1.23 times slower than the other two in every round, the
#             two builds whose f32 and f64 instances of
#             gemm_transposed_with_par lie in the other order (both
#             instances on 64-byte boundaries); in the first measurement the
#             0.2.3 builds ran at one of two speeds 1.03 apart;
#   f64_nt_4  1.107: the same two 0.2.3 builds of the third composition,
#             1.11 times slower in every round;
#   f64_tn_1  1.046: in the third composition one 0.2.4 build 1.05 times
#             faster than the other three in every round;
#   f64_tt_13 1.030, f64_tt_16 1.030: in this bench builds of both sides
#             run at one of two speeds about 1.033 apart, changing from one
#             program run to the next (a variation from run to run, not the
#             placement of a build); a build's median over 5 rounds lands
#             in either (f64_tt_14 and f64_tt_15 show the same, at 1.032 and
#             1.028 in the second measurement);
#   f64_tt_14 1.045: one 0.2.3 build 1.044 to 1.054 slower in 7 of its 9
#             rounds over both measurements of the second composition;
#   f32_tt_16 1.035: single rounds of builds of both sides 1.03 to 1.10
#             slower, and one 0.2.4 build slower in 4 of its 6 rounds (a
#             variation from round to round, not the placement of a build).
# Among the 120, the tree/base ratio was at most 1.006 over the pooled builds
# of the first measurement and at most 1.007 in the 4-build run of the second
# (1.013 with 0.2.3 on both sides). The first measurement of the second
# composition, with the same binaries in 3 rounds of 80 samples at load
# averages of 153 to 396, had 12 cases at or above 1.03; 9 of them fell
# below it with 6 rounds of 200 samples (f64_tt_1 f64_nn_5 f32_nn_5 f32_nt_6
# f32_tt_11 f32_tt_13 f32_tt_15 f64_tt_16, whose excursion had been 2 of the
# 3 rounds of one build, and f64_tt_13, 1.052 to 1.012), and only the later
# values count. The third composition found two cases at or above 1.03
# (f64_nt_4, f64_tn_1) that the first two did not: the bands are what these
# compositions and builds showed, not an upper bound for every code
# placement; re-measure them after a change of the toolchain, of this bench or
# of the code of either side that moves where the GEMM functions lie. The
# large TT cases are the nearest to the limit: after this table was set, a
# run with the default options measured f32_tt_14 at 1.031 (a base build's
# median in the slower of two speeds about 1.035 apart that change from one
# program run to the next, as for f64_tt_13 to 16 above; the spread rule made
# it information in that run), f32_tt_11 to 13 at 1.020 to 1.026, and f64_tt_15
# is at 1.027 here.
MEASURED_BANDS="f64_nn_1:1.018 f32_nn_1:1.030 f64_tn_1:1.046 f32_tn_1:1.020 f64_nt_1:1.011
f32_nt_1:1.020 f64_tt_1:1.011 f32_tt_1:1.022 f64_nn_2:1.019 f32_nn_2:1.005
f64_tn_2:1.015 f32_tn_2:1.007 f64_nt_2:1.226 f32_nt_2:1.022 f64_tt_2:1.020
f32_tt_2:1.004 f64_nn_3:1.005 f32_nn_3:1.024 f64_tn_3:1.008 f32_tn_3:1.024
f64_nt_3:1.003 f32_nt_3:1.015 f64_tt_3:1.018 f32_tt_3:1.013 f64_nn_4:1.004
f32_nn_4:1.027 f64_tn_4:1.005 f32_tn_4:1.014 f64_nt_4:1.107 f32_nt_4:1.007
f64_tt_4:1.012 f32_tt_4:1.013 f64_nn_5:1.016 f32_nn_5:1.027 f64_tn_5:1.003
f32_tn_5:1.004 f64_nt_5:1.022 f32_nt_5:1.005 f64_tt_5:1.007 f32_tt_5:1.023
f64_nn_6:1.012 f32_nn_6:1.007 f64_tn_6:1.010 f32_tn_6:1.008 f64_nt_6:1.006
f32_nt_6:1.009 f64_tt_6:1.004 f32_tt_6:1.004 f64_nn_7:1.007 f32_nn_7:1.007
f64_tn_7:1.005 f32_tn_7:1.004 f64_nt_7:1.003 f32_nt_7:1.005 f64_tt_7:1.004
f32_tt_7:1.005 f64_nn_8:1.003 f32_nn_8:1.006 f64_tn_8:1.005 f32_tn_8:1.004
f64_nt_8:1.005 f32_nt_8:1.005 f64_tt_8:1.009 f32_tt_8:1.007 f64_nn_9:1.017
f32_nn_9:1.012 f64_tn_9:1.003 f32_tn_9:1.005 f64_nt_9:1.006 f32_nt_9:1.006
f64_tt_9:1.008 f32_tt_9:1.012 f64_nn_10:1.006 f32_nn_10:1.005 f64_tn_10:1.003
f32_tn_10:1.005 f64_nt_10:1.003 f32_nt_10:1.007 f64_tt_10:1.010 f32_tt_10:1.013
f64_nn_11:1.003 f32_nn_11:1.005 f64_tn_11:1.003 f32_tn_11:1.005 f64_nt_11:1.024
f32_nt_11:1.007 f64_tt_11:1.005 f32_tt_11:1.013 f64_nn_12:1.010 f32_nn_12:1.012
f64_tn_12:1.008 f32_tn_12:1.004 f64_nt_12:1.004 f32_nt_12:1.004 f64_tt_12:1.009
f32_tt_12:1.009 f64_nn_13:1.017 f32_nn_13:1.013 f64_tn_13:1.008 f32_tn_13:1.003
f64_nt_13:1.003 f32_nt_13:1.004 f64_tt_13:1.030 f32_tt_13:1.015 f64_nn_14:1.007
f32_nn_14:1.008 f64_tn_14:1.004 f32_tn_14:1.005 f64_nt_14:1.004 f32_nt_14:1.002
f64_tt_14:1.045 f32_tt_14:1.020 f64_nn_15:1.005 f32_nn_15:1.003 f64_tn_15:1.005
f32_tn_15:1.003 f64_nt_15:1.008 f32_nt_15:1.018 f64_tt_15:1.027 f32_tt_15:1.029
f64_nn_16:1.005 f32_nn_16:1.006 f64_tn_16:1.003 f32_tn_16:1.004 f64_nt_16:1.003
f32_nt_16:1.005 f64_tt_16:1.030 f32_tt_16:1.035"

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

base_rev="v0.2.3"
tree_dir=""
out_dir=""
rounds=5
builds=2
samples=200
limit="1.03"
bench_source="$SCRIPT_DIR/gemm_tiny_call_bench.rs"
align=1
quick=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) print_help; exit 0 ;;
    --quick) quick=1; shift ;;
    --no-align) align=0; shift ;;
    --base|--tree|--out|--rounds|--builds|--limit|--bench|--samples)
      [ "$#" -ge 2 ] || die 2 "$1 needs a value"
      case "$1" in
        --base) base_rev="$2" ;;
        --tree) tree_dir="$2" ;;
        --out) out_dir="$2" ;;
        --rounds) rounds="$2" ;;
        --builds) builds="$2" ;;
        --limit) limit="$2" ;;
        --bench) bench_source="$2" ;;
        --samples) samples="$2" ;;
      esac
      shift 2 ;;
    *) die 2 "unknown option '$1' (see --help)" ;;
  esac
done

case "$rounds" in
  ''|*[!0-9]*) die 2 "--rounds: '$rounds' is not a positive whole number" ;;
esac
[ "$rounds" -ge 1 ] || die 2 "--rounds must be at least 1"
case "$builds" in
  ''|*[!0-9]*) die 2 "--builds: '$builds' is not a positive whole number" ;;
esac
[ "$builds" -ge 1 ] || die 2 "--builds must be at least 1"
case "$samples" in
  ''|*[!0-9]*) die 2 "--samples: '$samples' is not a positive whole number" ;;
esac
[ "$samples" -ge 1 ] || die 2 "--samples must be at least 1"
case "$limit" in
  ''|.|*[!0-9.]*|*.*.*) die 2 "--limit: '$limit' is not a number such as 1.03" ;;
esac

for tool in git cargo awk tar cksum; do
  command -v "$tool" > /dev/null 2>&1 || die 2 "'$tool' is not installed or not on PATH"
done
[ -f "$bench_source" ] || die 2 "the benchmark source '$bench_source' does not exist"
bench_source="$(cd "$(dirname "$bench_source")" && pwd -P)/$(basename "$bench_source")"
# The package (and crate) name of every generated crate: the file name of the
# benchmark source, as a valid package name.
package="$(basename "$bench_source" .rs | LC_ALL=C tr -c 'A-Za-z0-9_\n' '_')"
case "$package" in
  [A-Za-z]*) ;;
  *) package="bench_$package" ;;
esac

# The RUSTFLAGS of every build of both sides.
[ -z "${CARGO_ENCODED_RUSTFLAGS:-}" ] \
  || die 2 "CARGO_ENCODED_RUSTFLAGS is set: it would replace the RUSTFLAGS of the builds (put extra flags in RUSTFLAGS)"
unset CARGO_ENCODED_RUSTFLAGS
if [ "$align" = 1 ]; then
  build_rustflags="${RUSTFLAGS:+$RUSTFLAGS }$ALIGN_FLAGS"
  align_note="every function and loop aligned to 64 bytes"
else
  build_rustflags="${RUSTFLAGS:-}"
  align_note="no code alignment (--no-align: diagnosis only)"
fi

checkout="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2> /dev/null)" \
  || die 2 "$SCRIPT_DIR is not inside a git checkout"
base_commit="$(git -C "$checkout" rev-parse --verify --quiet "$base_rev^{commit}")" \
  || die 2 "the base revision '$base_rev' does not exist in $checkout"

[ -n "$tree_dir" ] || tree_dir="$checkout"
[ -d "$tree_dir" ] || die 2 "--tree: '$tree_dir' is not a directory"
tree_dir="$(cd "$tree_dir" && pwd -P)"
[ -f "$tree_dir/crates/oxiblas-blas/Cargo.toml" ] \
  || die 2 "--tree: '$tree_dir' has no crates/oxiblas-blas/Cargo.toml"
[ -f "$tree_dir/Cargo.toml" ] || die 2 "--tree: '$tree_dir' has no workspace Cargo.toml"

[ -n "$out_dir" ] || out_dir="$checkout/target/gemm_tiny_call"
# Output inside the tree goes under its target/ only (an output directory
# under crates/ would also be copied into the copies of the tree).
out_inside_tree() {
  case "$1/" in
    "$tree_dir"/target/*) return 1 ;;
    "$tree_dir"/*) return 0 ;;
  esac
  return 1
}
case "$out_dir" in
  /*) out_lexical="$out_dir" ;;
  *) out_lexical="$(pwd -P)/$out_dir" ;;
esac
out_inside_tree "$out_lexical" \
  && die 2 "--out: '$out_lexical' lies inside the tree outside its target/"
mkdir -p "$out_dir" || die 2 "cannot create the output directory '$out_dir'"
out_dir="$(cd "$out_dir" && pwd -P)"
out_inside_tree "$out_dir" \
  && die 2 "--out: '$out_dir' lies inside the tree outside its target/"
logs="$out_dir/logs"
summary="$out_dir/summary.txt"
perbuild="$out_dir/perbuild.txt"
mkdir -p "$logs" || die 2 "cannot create '$logs'"
rm -f "$summary" "$perbuild" "$logs"/run-*.log

mode=""
sample_ms=0.5
if [ "$quick" = 1 ]; then
  rounds=1 samples=20 mode="quick"
fi
if [ "$align" = 0 ]; then mode="${mode:+$mode, }no-align"; fi
[ -z "$mode" ] || mode=" ($mode)"
load_start="$(uptime 2> /dev/null | sed 's/.*load average[s]*: *//')"

# Writes the INCOMPLETE summary and stops with exit status 3.
incomplete() {
  {
    printf '# %s\n' "$@"
    printf 'verdict: INCOMPLETE%s\n' "$mode"
  } > "$summary"
  cat "$summary"
  exit 3
}

# --- The base exports (each kept while it is the same commit). -------------
export_base() {
  local dest="$1"
  if [ "$(cat "$dest/.oxiblas-base-commit" 2> /dev/null)" = "$base_commit" ]; then
    return 0
  fi
  note "exporting $base_rev ($base_commit) to $dest"
  rm -rf "$dest"
  mkdir -p "$dest" || return 1
  if ! git -C "$checkout" archive --format=tar "$base_commit" | tar -xf - -C "$dest"; then
    rm -rf "$dest"
    return 1
  fi
  printf '%s\n' "$base_commit" > "$dest/.oxiblas-base-commit"
}

# --- The copies of the tree (copied again when its build inputs changed). --
# The build inputs: the workspace Cargo.toml and crates/, without target/
# directories and Cargo.lock files (the generated crates have their own).
# A checksum of these files (names and contents), so an unchanged tree keeps
# its copies and their builds.
tree_manifest() {
  (cd "$tree_dir" \
    && find Cargo.toml crates \( -name target -o -name Cargo.lock \) -prune -o -type f -print0 \
      | LC_ALL=C sort -z | xargs -0 cksum | cksum)
}

copy_tree() {
  local dest="$1" manifest="$2"
  if [ "$(cat "$dest/.oxiblas-tree-manifest" 2> /dev/null)" = "$manifest" ]; then
    return 0
  fi
  note "copying the tree to $dest"
  rm -rf "$dest"
  mkdir -p "$dest" || return 1
  # -m: the copies get fresh modification times, so cargo rebuilds them
  # whatever times the tree's files carry.
  if ! (cd "$tree_dir" && tar --exclude=target --exclude=Cargo.lock -cf - Cargo.toml crates) \
       | tar -xmf - -C "$dest"; then
    rm -rf "$dest"
    return 1
  fi
  printf '%s\n' "$manifest" > "$dest/.oxiblas-tree-manifest"
}

if [ "$builds" -gt 1 ]; then
  manifest="$(tree_manifest)" || die 2 "cannot read the build inputs of '$tree_dir'"
fi
b=1
while [ "$b" -le "$builds" ]; do
  export_base "$out_dir/base-src-$b" || die 2 "git archive of '$base_rev' failed"
  if [ "$builds" -gt 1 ]; then
    copy_tree "$out_dir/tree-src-$b" "$manifest" \
      || die 2 "cannot copy '$tree_dir' to '$out_dir/tree-src-$b'"
  fi
  b=$((b + 1))
done

# --- The generated crates, one per build. ----------------------------------
# $1 = side (base or tree), $2 = build number, $3 = the oxiblas tree it
# depends on. Every crate has the same package name.
make_crate() {
  local src="$3" dir="$out_dir/bench-$1-$2"
  mkdir -p "$dir/src" || return 1
  cp "$bench_source" "$dir/src/main.rs" || return 1
  cat > "$dir/Cargo.toml" << EOF
[package]
name = "$package"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
oxiblas-blas = { path = "$src/crates/oxiblas-blas" }
oxiblas-matrix = { path = "$src/crates/oxiblas-matrix" }
oxiblas-core = { path = "$src/crates/oxiblas-core" }

[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3
EOF
}

b=1
while [ "$b" -le "$builds" ]; do
  if [ "$builds" -gt 1 ]; then tree_src="$out_dir/tree-src-$b"; else tree_src="$tree_dir"; fi
  make_crate base "$b" "$out_dir/base-src-$b" || die 2 "cannot write the base benchmark crate $b"
  make_crate tree "$b" "$tree_src" || die 2 "cannot write the tree benchmark crate $b"
  for side in base tree; do
    note "building $side build $b of $builds (log: $logs/build-$side-$b.log)"
    if ! (cd "$out_dir/bench-$side-$b" \
          && printf 'RUSTFLAGS=%s\n' "$build_rustflags" \
          && RUSTFLAGS="$build_rustflags" CARGO_TARGET_DIR="$out_dir/target-$side-$b" \
               cargo build --release) \
         > "$logs/build-$side-$b.log" 2>&1; then
      incomplete "the $side build $b failed; see $logs/build-$side-$b.log"
    fi
  done
  b=$((b + 1))
done

# --- Alternating runs. -----------------------------------------------------
r=1
while [ "$r" -le "$rounds" ]; do
  b=1
  while [ "$b" -le "$builds" ]; do
    for side in base tree; do
      note "round $r of $rounds: $side build $b"
      bin="$out_dir/target-$side-$b/release/$package"
      if ! "$bin" "$samples" "$sample_ms" > "$logs/run-$side-$b-$r.log" 2>&1; then
        incomplete "the run of $side build $b in round $r failed; see $logs/run-$side-$b-$r.log"
      fi
    done
    b=$((b + 1))
  done
  r=$((r + 1))
done

# --- Summary. --------------------------------------------------------------
load="$(uptime 2> /dev/null | sed 's/.*load average[s]*: *//')"
{
  printf '# base %s (%s), tree %s, %s builds per side, %s rounds x %s samples, limit %s, bench %s, load %s at start, %s at end\n' \
    "$base_rev" "$base_commit" "$tree_dir" "$builds" "$rounds" "$samples" "$limit" \
    "$(basename "$bench_source")" "${load_start:-unknown}" "${load:-unknown}"
  printf '# RUSTFLAGS of every build (both sides): %s (%s)\n' "$build_rustflags" "$align_note"
  printf '# case base_ns tree_ns tree/base judged|info base_spread tree_spread why\n'
} > "$summary"
{
  printf '# every build'"'"'s time of every case (ns, as in summary.txt: the median over its rounds of the clock-scaled minima, runs with a reference over 1.15 R left out; "-" for a build with no run of the case left), the spreads ("-" for a side with such a build), then every build'"'"'s raw minimum over all its rounds (left-out runs included)\n'
  printf '# case'
  for kind in "" raw_; do
    if [ -n "$kind" ]; then printf ' base_spread tree_spread'; fi
    b=1
    while [ "$b" -le "$builds" ]; do printf ' %sbase_%d' "$kind" "$b"; b=$((b + 1)); done
    b=1
    while [ "$b" -le "$builds" ]; do printf ' %stree_%d' "$kind" "$b"; b=$((b + 1)); done
  done
  printf '\n'
} > "$perbuild"

# Reads the "CASE<TAB>name<TAB>min_ns=X<TAB>med_ns=Y<TAB>ref_ns=C" lines of
# every run log; the side and the build are in the file name
# (run-<side>-<build>-<round>.log). Scales every case of every run to the
# fastest clock seen, X * R / C, R the smallest clock reference of all runs of
# both sides, leaves out every case run with C over 1.15 R, and takes per
# build the median over the remaining rounds (the minimum of the raw minima
# when a line has no reference). A case of which a build has no run left is
# information. Prints the "#"-lines on the clock and on missing cases, the
# case lines, the count of judged cases and then the verdict word; appends
# every build's times to perbuild.
bands="$(printf '%s\n' "$MEASURED_BANDS" | tr '\n' ' ')"
awk -v limit="$limit" -v rounds="$rounds" -v builds="$builds" \
    -v bands="$bands" -v perbuild="$perbuild" '
  # The median of the scaled values sv[k, 1..nsv[k]] of one side, build and
  # case (the mean of the two middle ones for an even count).
  function median_of(k,    m, i, j, t, a) {
    m = nsv[k]
    for (i = 1; i <= m; i++) a[i] = sv[k, i]
    for (i = 2; i <= m; i++) {
      t = a[i]
      for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]
      a[j + 1] = t
    }
    if (m % 2 == 1) return a[(m + 1) / 2]
    return (a[m / 2] + a[m / 2 + 1]) / 2
  }
  BEGIN {
    nb = split(bands, bt, / +/)
    for (i = 1; i <= nb; i++) {
      if (bt[i] == "") continue
      split(bt[i], kv, ":")
      band[kv[1]] = kv[2] + 0
    }
    lines = 0; with_ref = 0; nv = 0; refall = ""; refmax = ""
  }
  FNR == 1 {
    file = FILENAME
    sub(/.*\/run-/, "", file)
    split(file, parts, "-")
    side = parts[1]; build = parts[2]
  }
  $1 == "CASE" {
    name = $2
    split($3, kv, "=")
    v = kv[2] + 0
    clock = ""
    for (f = 4; f <= NF; f++) {
      split($f, kv, "=")
      if (kv[1] == "ref_ns") clock = kv[2] + 0
    }
    key = side SUBSEP build SUBSEP name
    if (!(key in raw) || v < raw[key]) raw[key] = v
    seen[key]++
    lines++
    if (clock != "" && clock > 0) {
      with_ref++
      if (refall == "" || clock < refall) refall = clock
      if (refmax == "" || clock > refmax) refmax = clock
      nv++; vv[nv] = v; vr[nv] = clock; vk[nv] = key; vs[nv] = side
    }
    # Cases of either side (base files first, so in the base order): a
    # case that only one side printed is then missing from the other.
    if (!(name in order_seen)) { order[++n] = name; order_seen[name] = 1 }
  }
  END {
    if (lines > 0 && with_ref == lines) {
      # Per run, the case minimum scaled to the fastest clock (X * R / C);
      # a run whose reference ran on a slower core (C over 1.15 R) is left
      # out. Per build, the median of the remaining runs; a build with no
      # run left has no time for that case (best[] has no entry).
      slow = 0
      for (i = 1; i <= nv; i++) {
        k = vk[i]
        slowrun = (vr[i] > 1.15 * refall)
        if (slowrun) slow++
        if (slowrun) continue
        nsv[k]++
        sv[k, nsv[k]] = vv[i] * refall / vr[i]
      }
      for (k in nsv) best[k] = median_of(k)
      # The cases that lost a build that way: a build of either side that
      # printed the case but has no run of it left.
      lostcases = 0
      for (i = 1; i <= n; i++) {
        name = order[i]
        for (s = 1; s <= 2; s++) {
          sd = (s == 1) ? "base" : "tree"
          for (b = 1; b <= builds; b++) {
            key = sd SUBSEP b SUBSEP name
            if ((key in seen) && !(key in best)) lostcase[name] = 1
          }
        }
        if (name in lostcase) lostcases++
      }
      printf "# clock: base_ns and tree_ns are, per build, the median over the rounds of each run'"'"'s minimum scaled to the fastest clock seen in any run of either side, runs on a slower core left out (times R = %.5f ns per add over the run'"'"'s own clock reference; the slowest reference was %.3f R; %d of %d case runs had a reference over 1.15 R, a slower core, and were left out; %d cases lost every run of a build that way and are information)\n", refall, refmax / refall, slow, lines, lostcases
    } else {
      for (key in raw) best[key] = raw[key]
      printf "# clock: %d of %d case lines carry no clock reference: base_ns and tree_ns are raw minima, not scaled to one clock\n", lines - with_ref, lines
    }
    bad = 0; missing = 0; judged = 0
    for (i = 1; i <= n; i++) {
      name = order[i]
      complete = 1
      for (s = 1; s <= 2; s++) {
        sd = (s == 1) ? "base" : "tree"
        for (b = 1; b <= builds; b++) {
          key = sd SUBSEP b SUBSEP name
          got = (key in seen) ? seen[key] : 0
          if (got != rounds || ((key in best) && best[key] <= 0)) {
            printf "# missing: %s from the %s side, build %d (in %d of %d rounds)\n", name, sd, b, got, rounds
            complete = 0
          }
        }
      }
      if (!complete) { missing++; continue }
      # Per side: the fastest and the slowest build with a time, and the
      # builds without one (every run left out, see "Clock").
      bmin = ""; bmax = ""; tmin = ""; tmax = ""; lostwhy = ""; blost = 0; tlost = 0
      line = sprintf("%-14s", name)
      for (s = 1; s <= 2; s++) {
        sd = (s == 1) ? "base" : "tree"
        for (b = 1; b <= builds; b++) {
          key = sd SUBSEP b SUBSEP name
          if (!(key in best)) {
            line = line sprintf(" %10s", "-")
            lostwhy = lostwhy sprintf("%s%s build %d", (lostwhy == "") ? "" : ", ", sd, b)
            if (s == 1) blost++; else tlost++
            continue
          }
          x = best[key]
          line = line sprintf(" %10.3f", x)
          if (s == 1) {
            if (bmin == "" || x < bmin) bmin = x
            if (bmax == "" || x > bmax) bmax = x
          } else {
            if (tmin == "" || x < tmin) tmin = x
            if (tmax == "" || x > tmax) tmax = x
          }
        }
      }
      # Spreads over the builds with a time; printed as "-" for a side that
      # lost a build (no spread was measured there).
      sb = (bmin == "") ? 0 : bmax / bmin
      st = (tmin == "") ? 0 : tmax / tmin
      sbs = (blost > 0) ? "-" : sprintf("%.3f", sb)
      sts = (tlost > 0) ? "-" : sprintf("%.3f", st)
      line = line sprintf(" %6s %6s", sbs, sts)
      for (s = 1; s <= 2; s++) {
        sd = (s == 1) ? "base" : "tree"
        for (b = 1; b <= builds; b++) line = line sprintf(" %10.3f", raw[sd SUBSEP b SUBSEP name])
      }
      print line >> perbuild
      if (lostwhy != "") {
        # A build without a time for this case: not judged.
        kind = "info"
        why = sprintf("no run with a reference within 1.15 R left in %s (every run on a slower core)", lostwhy)
        bs = (bmin == "") ? "-" : sprintf("%.3f", bmin)
        ts = (tmin == "") ? "-" : sprintf("%.3f", tmin)
        rs = (bmin == "" || tmin == "") ? "-" : sprintf("%.3f", tmin / bmin)
        printf "%-14s %10s %10s %7s %-6s %6s %6s %s\n", name, bs, ts, rs, kind, sbs, sts, why
        continue
      }
      ratio = tmin / bmin
      if (name ~ /_blk/) {
        kind = "info"; why = "blocked route, information only"
      } else if ((name in band) && band[name] >= limit + 0) {
        kind = "info"; why = sprintf("measured band %.3f >= %s", band[name], limit)
      } else if (builds == 1) {
        if (name in band) {
          kind = "judged"; why = sprintf("measured band %.3f below %s (--builds 1)", band[name], limit)
        } else {
          kind = "info"; why = "no measured band (--builds 1)"
        }
      } else if (sb < limit + 0 && st < limit + 0) {
        kind = "judged"; why = sprintf("placement spread below %s on both sides", limit)
      } else if (sb >= limit + 0 && st >= limit + 0) {
        kind = "info"; why = sprintf("placement spread base %.3f and tree %.3f >= %s", sb, st, limit)
      } else if (sb >= limit + 0) {
        kind = "info"; why = sprintf("placement spread base %.3f >= %s", sb, limit)
      } else {
        kind = "info"; why = sprintf("placement spread tree %.3f >= %s", st, limit)
      }
      if (tmin / bmax > limit + 0) why = why sprintf("; every tree build above every base build (x%.3f)", tmin / bmax)
      if (kind == "judged") judged++
      if (kind == "judged" && ratio > limit + 0) bad++
      printf "%-14s %10.3f %10.3f %7.3f %-6s %6.3f %6.3f %s\n", name, bmin, tmin, ratio, kind, sb, st, why
    }
    printf "# judged: %d of %d cases (the others are information)\n", judged, n - missing
    if (n == 0 || missing > 0) print "INCOMPLETE"
    else if (judged == 0) {
      # Nothing was judged, so nothing can pass.
      print "# no case was judged (every one is information)"
      print "INCOMPLETE"
    }
    else if (bad > 0) print "FAIL"
    else print "PASS"
  }
' "$logs"/run-base-*.log "$logs"/run-tree-*.log > "$out_dir/summary.body" \
  || incomplete "the summary could not be computed"

verdict="$(tail -n 1 "$out_dir/summary.body")"
sed '$d' "$out_dir/summary.body" >> "$summary"
rm -f "$out_dir/summary.body"
printf 'verdict: %s%s\n' "$verdict" "$mode" >> "$summary"
cat "$summary"

case "$verdict" in
  PASS) exit 0 ;;
  FAIL) exit 1 ;;
  *) exit 3 ;;
esac
