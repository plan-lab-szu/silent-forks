#!/usr/bin/env bash
# Two conforming execution clients, built from one source tree with different
# compiled accumulation orders, exchanging the blocks they each built.
#
# Three experiments, nine phases. Each experiment runs in both directions, so
# every verdict is reported twice with the roles swapped.
#
#   A  mandatory path, unspecified arithmetic   -> expect mutual INVALID
#      Aggregation scheduled by height as a system call, in a block with no
#      transactions, which no proposer can decline to run. Submissions are a
#      constant in the deployed contract.
#
#   A' transaction path, unspecified arithmetic -> expect mutual INVALID
#      Same submissions, now delivered as transaction calldata through a
#      contract entry point. This is the cell that makes A' vs B a comparison
#      across the specification alone rather than across the specification and
#      the delivery mechanism at once.
#
#   B  transaction path, specified arithmetic   -> expect identical roots,
#      mutual VALID. The defence control, and also the harness control: a
#      harness that only ever said INVALID would prove nothing.
#
# The accumulation order is a build tag, not a runtime flag (see
# chain/geth/core/vm/contracts_dfrl_vulnerable_order_fwd.go), so the two nodes
# are two separately compiled binaries. A file carries one node's envelope to
# the other; nothing between the runs edits it, and the receiving side
# recomputes the block hash from the payload's own fields before submitting, so
# a transport artifact fails the run instead of being reported as a verdict.
#
# Usage:  chain/tools/run_payload_exchange.sh [workdir]
# Writes: evidence/two_node_payload_exchange.log
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
geth="$repo/chain/geth"
work="${1:-$(mktemp -d)}"
log="$repo/evidence/two_node_payload_exchange.log"

FWD_TAGS="e3lite_vulnerable"
REV_TAGS="e3lite_vulnerable e3lite_reverse"

mkdir -p "$work" "$repo/evidence"
: > "$log"

# Some Windows installations run an application-control policy that refuses to
# execute freshly built, unsigned binaries -- intermittently, by binary
# reputation, so the same build can be allowed on one invocation and blocked on
# the next. `go test` and a binary produced by `go test -c` are blocked
# independently, so each phase tries both before giving up. Nothing about the
# experiment changes between the two ways of launching it.
prebuild() {
  local tags="$1" out="$2"
  ( cd "$geth" && go test -c -tags "$tags" -o "$out" ./eth/catalyst/ ) >/dev/null 2>&1
}
prebuild "$FWD_TAGS" "$work/catalyst_fwd.test.exe"
prebuild "$REV_TAGS" "$work/catalyst_rev.test.exe"

run() {
  local phase="$1" which="$2" test="$3"; shift 3
  local tags bin
  if [ "$which" = fwd ]; then tags="$FWD_TAGS"; bin="$work/catalyst_fwd.test.exe"
  else tags="$REV_TAGS"; bin="$work/catalyst_rev.test.exe"; fi

  {
    echo "########## $phase"
    echo "# build tags: $tags"
    echo "# test:       $test"
    echo "# env:        $*"
  } >> "$log"

  local tmp="$work/.phase.out" rc=1 how=""
  ( cd "$geth" && env "$@" go test -count=1 -tags "$tags" -run "$test" -v ./eth/catalyst/ ) \
    > "$tmp" 2>&1
  rc=$?
  how="go test"
  if [ $rc -ne 0 ] && grep -qi "Application Control policy" "$tmp"; then
    echo "# 'go test' launch blocked by an application-control policy; retrying the prebuilt binary" >> "$log"
    ( cd "$geth/eth/catalyst" && env "$@" "$bin" -test.run "$test" -test.v ) > "$tmp" 2>&1
    rc=$?
    how="prebuilt binary"
  fi
  cat "$tmp" >> "$log"
  { echo "# launched via: $how"; echo "# exit=$rc"; echo ""; } >> "$log"
  if [ $rc -ne 0 ] && grep -qi "Application Control policy" "$tmp"; then
    echo "# NOTE: both launch methods were blocked by the host policy. This is an" >> "$log"
    echo "#       environment obstacle, not an experimental result." >> "$log"
  fi
  return $rc
}

{
  echo "Two-node payload exchange: divergence, its transaction-path twin, and the defence control"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm)"
  echo "go:   $(cd "$geth" && go version)"
  echo "work: $work"
  echo ""
  echo "A  mandatory system call, float path      -> expect mutual INVALID"
  echo "A' transaction entry point, float path    -> expect mutual INVALID"
  echo "B  transaction entry point, integer path  -> expect equal roots, mutual VALID"
  echo ""
} >> "$log"

fail=0
A=TestDFRLTwoNodePayloadExchange
AP=TestDFRLTwoNodeTxPathDiverges
B=TestDFRLTwoNodeDeterministicPathAgrees

run "A1  seq-fwd publishes (mandatory path)"        fwd "$A"  "DFRL_PAYLOAD_OUT=$work/a_fwd.json" || fail=1
run "A2  seq-rev publishes and judges seq-fwd"      rev "$A"  "DFRL_PAYLOAD_OUT=$work/a_rev.json" "DFRL_PAYLOAD_IN=$work/a_fwd.json" || fail=1
run "A3  seq-fwd judges seq-rev"                    fwd "$A"  "DFRL_PAYLOAD_IN=$work/a_rev.json" || fail=1

run "A'1 seq-fwd publishes (tx path, float)"        fwd "$AP" "DFRL_TXV_OUT=$work/ap_fwd.json" || fail=1
run "A'2 seq-rev publishes and judges seq-fwd"      rev "$AP" "DFRL_TXV_OUT=$work/ap_rev.json" "DFRL_TXV_IN=$work/ap_fwd.json" || fail=1
run "A'3 seq-fwd judges seq-rev"                    fwd "$AP" "DFRL_TXV_IN=$work/ap_rev.json" || fail=1

run "B1  seq-fwd publishes (tx path, specified)"    fwd "$B"  "DFRL_DET_OUT=$work/b_fwd.json" || fail=1
run "B2  seq-rev publishes and judges seq-fwd"      rev "$B"  "DFRL_DET_OUT=$work/b_rev.json" "DFRL_DET_IN=$work/b_fwd.json" || fail=1
run "B3  seq-fwd judges seq-rev"                    fwd "$B"  "DFRL_DET_IN=$work/b_rev.json" || fail=1

{
  echo "########## summary"
  grep -h "compiled accumulation order\|own payload:\|control: own\|byte-identical transaction\|identical transaction, same parent\|same post-state root\|peer payload verdict\|RESULT" "$log" | sed 's/^ *//'
  echo ""
  if [ "$fail" -eq 0 ]; then
    echo "all nine phases passed:"
    echo "  A  and A' -- mutual rejection on the unspecified path, on both the"
    echo "               mandatory system call and the transaction entry point"
    echo "  B         -- identical post-state roots and mutual acceptance once"
    echo "               the arithmetic and encoding follow the specification"
  else
    echo "AT LEAST ONE PHASE FAILED -- read the phase output above."
    echo "If the failure mentions an application-control policy, it is the host"
    echo "refusing to launch the binary, not a result."
  fi
} >> "$log"

echo "wrote $log"
exit "$fail"
