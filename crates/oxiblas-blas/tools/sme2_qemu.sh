#!/usr/bin/env bash
# sme2_qemu.sh -- runs the SME2 test suites of this checkout under QEMU.
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
# Both passes also set OXIBLAS_TEST_SME2_MARKERS=1: the tests' hardware gate
# (test_support::hardware_or_skip, called by every test that executes SME2
# code) then writes `sme2-gate: ran <test>` or `sme2-gate: skipped <test>` for
# each decision, uncaptured, and the verdict counts those lines (see Exit
# status).
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
# Usage:
#   sme2_qemu.sh [options] [-- <extra cargo test arguments>]
#
# Usual invocations (from anywhere; the checkout is the one the script is in):
#   crates/oxiblas-blas/tools/sme2_qemu.sh                 # both passes
#   crates/oxiblas-blas/tools/sme2_qemu.sh --positive      # positive pass only
#   crates/oxiblas-blas/tools/sme2_qemu.sh --filter test_sme2_smoke -- -- --nocapture
#   crates/oxiblas-blas/tools/sme2_qemu.sh --native        # Linux, no Docker
#
# Each pass runs
#   cargo +<toolchain> test -p oxiblas-blas --features sme2,parallel --lib
#         --release --target aarch64-unknown-linux-gnu <filter> <extra>
# with the filter `sme2` by default. Without a filter the whole lib suite
# runs, which takes well over half an hour under QEMU.
#
# Options:
#   --positive         run the positive pass
#   --negative         run the negative pass
#   --both             run both passes (the default when neither is given)
#   --filter NAME      test name filter (default: sme2; '' for none)
#   --toolchain TC     Rust toolchain (default 1.95.0)
#   --native           run without Docker (see Hosts)
#   --reset            first remove this checkout's build volume (or, with
#                      --native, target/sme2-qemu/native), then run
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
# Everything after `--` is appended to the cargo test command; arguments for
# the test binary itself go after a second `--`.
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
#                      forwarded as well.
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
# tree to target/sme2-qemu/native/src and builds in target/sme2-qemu/native.
#
# Output, in target/sme2-qemu/ of this checkout (and nowhere else):
#   positive.log, negative.log   the complete output of each pass
#   image-build.log              when the image was built in this run
#   summary.txt                  OS, kernel and architecture of the host,
#                                mode, image tag and ID, QEMU, qemu-user
#                                package and rustc versions, CPU strings,
#                                cargo command, the names and values of the
#                                forwarded variables, the gate variables,
#                                test and gate-marker counts and verdict of
#                                each pass, git commit and whether the tree
#                                is dirty
#   sme2-qemu-results.tar.gz     summary.txt and the logs, the file to send
#                                back
# The results of an earlier run are deleted at the start, so after any exit
# the archive (if present) belongs to this run.
# The archive is meant to be attached to public issues. It holds the
# complete logs (compiler and test output, tool versions) and the values of
# the forwarded variables (OXIBLAS_*, RUSTFLAGS, ...): leave out of those
# variables whatever you would not publish. The host name is not recorded,
# paths in the summary are relative to the checkout, and in the archived logs
# the checkout's absolute path is replaced by `<checkout>`.
# Exit status: 0 when every requested pass succeeded, 1 when one failed,
# 2 for a usage error or a missing prerequisite (also a checkout path that a
# Docker bind mount cannot take, e.g. one containing `:`), 3 when the lock
# timed out.
# A pass succeeds when cargo exits 0, at least one test ran, none failed, and
# the gate markers prove that SME2 code was reached: in the positive pass at
# least one `sme2-gate: ran` line and no `sme2-gate: skipped` line, in the
# negative pass at least one `sme2-gate: skipped` line and no `sme2-gate: ran`
# line. A filter that selects only tests which never call the gate (for
# example test_sme2_contract_reference_equals_neon) therefore fails both
# passes: it would prove nothing about SME2.
# In the positive pass a test that finds no SME2 panics (the variable
# OXIBLAS_TEST_REQUIRE_SME2), so a skip shows up as a failure; with output
# not captured (`-- -- --nocapture`) a `skip:` line also fails the pass.
# QEMU's "uncaught target signal 4" in a log is SIGILL in the emulated code.
# On SIGINT, SIGTERM or SIGHUP (a closed terminal or SSH session) the
# container of the running pass is removed and the lock is released.
#---
set -euo pipefail

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

# ---------------------------------------------------------------- arguments
# --help first, before anything is touched (an option's value is skipped).
arg_index=1
while [ "$arg_index" -le "$#" ]; do
  case "${!arg_index}" in
    --) break ;;
    -h|--help) print_help; exit 0 ;;
    --filter|--toolchain|--platform) arg_index=$((arg_index + 1)) ;;
  esac
  arg_index=$((arg_index + 1))
done

# ------------------------------------------------------------------ checkout
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
[ -f "$ROOT/Cargo.toml" ] && [ -f "$ROOT/crates/oxiblas-blas/Cargo.toml" ] \
  || die 2 "cannot find the oxiblas workspace above $SCRIPT_DIR"
OUT="$ROOT/target/sme2-qemu"
OUT_REL="target/sme2-qemu"
mkdir -p "$OUT"
# The results of an earlier run go now, before any check that can fail, so
# that no old archive (perhaps a PASS) is left behind by an early exit.
rm -f "$OUT/positive.log" "$OUT/negative.log" "$OUT/summary.txt" \
  "$OUT/image-build.log" "$OUT/sme2-qemu-results.tar.gz" \
  "$OUT/.positive.rc" "$OUT/.negative.rc" "$OUT/.positive.fifo" "$OUT/.negative.fifo"
rm -rf "$OUT/.archive"

want_positive=0
want_negative=0
filter="$DEFAULT_FILTER"
toolchain="$DEFAULT_TOOLCHAIN"
native=0
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
      toolchain="$2"; shift 2 ;;
    --toolchain=*) toolchain="${1#--toolchain=}"; shift ;;
    --native) native=1; shift ;;
    --reset) reset=1; shift ;;
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
if [ "$native" = 0 ]; then
  case "$ROOT" in
    *:*|*"
"*) die 2 "the checkout path contains ':' or a newline, which a Docker bind mount cannot take: $ROOT (move or link the checkout elsewhere, or use --native on Linux)" ;;
  esac
fi

# The cargo test arguments shared by both passes.
cargo_args=(-p oxiblas-blas --features "sme2,parallel" --lib --release --target "$TARGET_TRIPLE")
if [ -n "$filter" ]; then cargo_args+=("$filter"); fi
if [ "${#extra_args[@]}" -gt 0 ]; then cargo_args+=("${extra_args[@]}"); fi

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

LOCK_DIR="${OXIBLAS_QEMU_LOCK_DIR:-}"
have_lock=0
current_container=""
child_pid=""

release_lock() {
  if [ "$have_lock" = 1 ] && [ "$(cat "$LOCK_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
  have_lock=0
}

# Whether the lock is stale: its owner process is gone, or it has had no
# `pid` file for more than LOCK_STALE_AGE seconds. A missing lock is not stale.
lock_is_stale() {
  local owner age
  [ -d "$LOCK_DIR" ] || return 1
  owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -n "$owner" ]; then
    process_alive "$owner" && return 1
    return 0
  fi
  age=$(( $(date +%s) - $(file_mtime "$LOCK_DIR") ))
  [ "$age" -gt "$LOCK_STALE_AGE" ]
}

# Removes a stale lock. Only the holder of the breaker directory may remove
# it, and it looks again before it does: without the breaker two waiters that
# both saw the dead owner could each remove the lock the other one has just
# taken. A breaker older than LOCK_STALE_AGE (its holder died) is removed; a
# waiter that finds the breaker held waits briefly and lets the caller retry.
break_stale_lock() {
  local breaker="$LOCK_DIR.breaker" breaker_age
  if mkdir "$breaker" 2>/dev/null; then
    if lock_is_stale; then
      note "lock: removing the stale lock of ${1:-nobody}"
      rm -rf "$LOCK_DIR"
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

acquire_lock() {
  [ -n "$LOCK_DIR" ] || return 0
  local wait_start waited=0 noted=-1 owner
  wait_start="$(date +%s)"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if lock_is_stale; then
      break_stale_lock "$owner"
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

# Stops the running pass: the background `docker start --attach` client (or,
# with --native, the cargo process) and the container. The container is
# created in the foreground (see run_pass), so whenever this runs from a trap
# a container named in current_container either exists or never will; it is
# removed once more after the client is gone, for good measure.
# shellcheck disable=SC2329 # called from the traps
stop_pass() {
  local tries=0
  if [ -n "$child_pid" ]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  if [ -n "$current_container" ]; then
    docker rm -f "$current_container" >/dev/null 2>&1 || true
  fi
  if [ -n "$child_pid" ]; then
    while child_running "$child_pid" && [ "$tries" -lt 50 ]; do
      sleep 0.2
      tries=$((tries + 1))
    done
    kill -KILL "$child_pid" 2>/dev/null || true
    child_pid=""
  fi
  if [ -n "$current_container" ]; then
    docker rm -f "$current_container" >/dev/null 2>&1 || true
    current_container=""
  fi
}

# shellcheck disable=SC2329 # invoked by the EXIT trap
on_exit() {
  stop_pass
  release_lock
}

# shellcheck disable=SC2329 # invoked by the INT, TERM and HUP traps
on_signal() {
  trap - INT TERM HUP
  stop_pass
  release_lock
  exit "$1"
}
trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

# ----------------------------------------------------------- inner runner
# Runs inside the container (or on the host with --native) as
#   bash -c "$INNER" _ <cargo test arguments>
# with SQ_SRC (the checkout), SQ_WORK (where to copy it), SQ_CPU (the QEMU CPU
# string) and SQ_TOOLCHAIN set, and CARGO_TARGET_DIR pointing at the build
# directory. On a non-aarch64 machine it cross-compiles and gives QEMU the
# cross compiler's aarch64 sysroot with -L.
# shellcheck disable=SC2016 # expanded by the inner bash, not here
INNER='set -euo pipefail
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
echo "== cargo +$SQ_TOOLCHAIN test $*"
exec cargo "+$SQ_TOOLCHAIN" test "$@"'

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

if [ "$native" = 1 ]; then
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

# Counts the occurrences of a fixed text in a file (anywhere in a line: with
# one test thread libtest prints `test <name> ... ` before the test runs, and
# a marker written meanwhile lands in the middle of that line).
count_text() {
  { grep -o -F -e "$1" "$2" 2>/dev/null || true; } | wc -l | tr -d ' '
}

run_pass() {
  local pass="$1" cpu="$2" require="$3" expect_no="$4"
  local log="$OUT/$pass.log" fifo="$OUT/.$pass.fifo" rc=0 start elapsed counts tee_pid create_out
  local n_results passed failed ignored skips gate_ran gate_skipped verdict reason
  local gate_env="OXIBLAS_TEST_REQUIRE_SME2=$require OXIBLAS_TEST_EXPECT_NO_SME2=$expect_no $MARKERS_VAR=1"
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
  rm -f "$fifo"
  mkfifo "$fifo"
  tee -a "$log" < "$fifo" &
  tee_pid=$!
  if [ "$native" = 1 ]; then
    env SQ_SRC="$ROOT" SQ_WORK="$OUT/native/src" SQ_CPU="$cpu" SQ_TOOLCHAIN="$toolchain" \
      CARGO_TARGET_DIR="$OUT/native/target" CARGO_TERM_COLOR=never \
      OXIBLAS_TEST_REQUIRE_SME2="$require" OXIBLAS_TEST_EXPECT_NO_SME2="$expect_no" \
      "$MARKERS_VAR=1" bash -c "$INNER" _ "${cargo_args[@]}" > "$fifo" 2>&1 &
    child_pid=$!
  else
    local docker_env=(-e CARGO_TARGET_DIR=/target -e CARGO_TERM_COLOR=never
      -e SQ_SRC=/src -e SQ_WORK=/work/src -e "SQ_CPU=$cpu" -e "SQ_TOOLCHAIN=$toolchain")
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
        "$image" bash -c "$INNER" _ "${cargo_args[@]}" 2>&1)"; then
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
  current_container=""
  wait "$tee_pid" || true
  rm -f "$fifo"
  release_lock
  elapsed=$((SECONDS - start))

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
  elif [ "$pass" = negative ] && [ "$gate_skipped" = 0 ]; then
    verdict="FAIL"; reason="no gated SME2 test was declined: the filter selects only tests that do not execute SME2 code"
  elif [ "$pass" = negative ] && [ "$gate_ran" != 0 ]; then
    verdict="FAIL"; reason="$gate_ran gated test(s) ran SME2 in the negative pass"
  fi
  if grep -q 'uncaught target signal' "$log"; then
    reason="${reason:+$reason; }QEMU reported an uncaught target signal"
  fi
  local result
  result="$(printf '%s\n' \
    "  cpu       : $cpu" \
    "  gate env  : $gate_env" \
    "  tests     : $passed passed, $failed failed, $ignored ignored ($n_results test binaries)" \
    "  gate      : $gate_ran ran SME2, $gate_skipped skipped SME2 (sme2-gate lines)" \
    "  skip lines: $skips" \
    "  cargo exit: $rc" \
    "  time      : ${elapsed}s" \
    "  verdict   : $verdict${reason:+ ($reason)}")"
  if [ "$pass" = positive ]; then positive_result="$result"; else negative_result="$result"; fi
  printf '== %s pass: %s%s (%s passed, %s failed, gate %s ran / %s skipped, %ss)\n' \
    "$pass" "$verdict" "${reason:+ - $reason}" "$passed" "$failed" "$gate_ran" "$gate_skipped" \
    "$elapsed" | tee -a "$log"
  [ "$verdict" = PASS ]
}

overall=0
if [ "$want_positive" = 1 ]; then
  run_pass positive "$CPU_POSITIVE" 1 0 || overall=1
fi
if [ "$want_negative" = 1 ]; then
  run_pass negative "$CPU_NEGATIVE" 0 1 || overall=1
fi

# ---------------------------------------------------------------- summary
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

cargo_cmd="cargo +$toolchain test"
for arg in "${cargo_args[@]}"; do
  case "$arg" in
    ''|*[!A-Za-z0-9_.,:=/+-]*) cargo_cmd="$cargo_cmd $(printf '%q' "$arg")" ;;
    *) cargo_cmd="$cargo_cmd $arg" ;;
  esac
done
forwarded="none"
if [ "${#forward_names[@]}" -gt 0 ]; then
  forwarded=""
  for name in "${forward_names[@]}"; do forwarded="$forwarded $name=${!name}"; done
  forwarded="${forwarded# }"
fi
if [ "$overall" = 0 ]; then overall_text="PASS"; else overall_text="FAIL"; fi

{
  printf 'oxiblas SME2 under QEMU: %s\n' "$overall_text"
  printf 'date (UTC)   : %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
  printf 'host         : %s\n' "$host_desc"
  printf 'architecture : host %s; test binaries %s, %s\n' "$host_machine" "$TARGET_TRIPLE" "$build_desc"
  printf 'mode         : %s\n' "$mode"
  if [ -n "$image" ]; then
    printf 'image        : %s (built in this run: %s)\n' "$image" "$image_built"
    printf 'image ID     : %s\n' "$image_id"
  fi
  printf 'build dir    : %s\n' "$build_volume"
  printf 'qemu         : %s\n' "$qemu_version"
  printf 'qemu-user pkg: %s\n' "$qemu_package"
  printf 'rustc        : %s\n' "$rustc_version"
  printf 'cargo command: %s\n' "$cargo_cmd"
  printf 'forwarded env: %s\n' "$forwarded"
  printf 'git commit   : %s\n' "$git_commit"
  printf 'git dirty    : %s\n' "$git_dirty"
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
  printf 'overall      : %s\n' "$overall_text"
} > "$OUT/summary.txt"

# The archive gets copies in which the checkout's absolute path (in native
# mode in cargo's lines, for example) reads `<checkout>`. The replacement is
# literal: the path comes in through ENVIRON (awk -v would interpret
# backslashes) and is found with index(), not as a regular expression.
mkdir "$OUT/.archive"
archive_files=(summary.txt)
for f in positive.log negative.log image-build.log; do
  [ -f "$OUT/$f" ] && archive_files+=("$f")
done
for f in "${archive_files[@]}"; do
  SQ_ROOT="$ROOT" awk '
    BEGIN { root = ENVIRON["SQ_ROOT"]; n = length(root) }
    {
      line = $0; out = ""
      while (n > 0 && (i = index(line, root)) > 0) {
        out = out substr(line, 1, i - 1) "<checkout>"
        line = substr(line, i + n)
      }
      print out line
    }' "$OUT/$f" > "$OUT/.archive/$f"
done
tar -czf "$OUT/sme2-qemu-results.tar.gz" -C "$OUT/.archive" "${archive_files[@]}"
rm -rf "$OUT/.archive"

cat "$OUT/summary.txt"
printf '\nResults: %s\n' "$OUT/sme2-qemu-results.tar.gz"
exit "$overall"
