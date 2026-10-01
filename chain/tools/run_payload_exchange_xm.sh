#!/usr/bin/env bash
# Cross-machine payload exchange: the same three experiments as
# run_payload_exchange.sh, but with the two nodes on two physical hosts.
#
# WHY THIS EXISTS
#
# The same-machine runs establish that two binaries differing only in compiled
# accumulation order reject each other's blocks. They cannot establish that the
# order is what caused it, because on one machine the order is the only thing
# that ever varied -- there is no cell in which two different environments run
# the same order. A reader may reasonably ask whether the divergence is really
# a property of the arithmetic or an artifact of this one host.
#
# The cross-machine grid closes that. It is a 2x2 of {seq-fwd, seq-rev} x
# {host X, host Y}, run as two cells:
#
#   --cell differ      X compiles seq-fwd, Y compiles seq-rev
#                      -> A and A' mutual INVALID, B mutual VALID
#                      The same verdicts as on one machine, now with the two
#                      binaries genuinely produced and run by different
#                      machines. Rules out anything shared by one host's
#                      process environment.
#
#   --cell same-order-observe
#                      Like same-order, but for a pair of hosts that differ in
#                      more than the order -- different architectures, say. Every
#                      guard still runs; the roots and verdicts are RECORDED
#                      rather than asserted, because there they are the
#                      measurement, not a pass/fail criterion. Characterises the
#                      pair. NOT attribution to the order.
#
#   --cell same-order  X and Y both compile seq-fwd
#                      -> agreement expected in ALL THREE experiments,
#                         including the float ones
#                      This is the load-bearing cell and the only genuinely new
#                      evidence here. If two hosts with different CPUs, core
#                      counts and operating systems, compiling the same order,
#                      agree byte for byte even on the unspecified float path,
#                      then the disagreement seen in the differ cell is
#                      attributable to the order and not to the host.
#                      Asymmetric: conclusive when it PASSES, inconclusive when
#                      it fails. Hosts that also differ in architecture make a
#                      pass stronger, since more varied and the commitment still
#                      held; but a failure then has a known alternative cause
#                      (evidence/e3lite_arm64_vs_amd64.log) and must not be read
#                      as a refutation.
#
# Run BOTH cells. Either one alone proves less than half.
#
# HOW IT RUNS
#
# Symmetric: both machines run one command and wait for each other. There is no
# phase ordering to coordinate and it does not matter who starts first.
#
#   host X:  chain/tools/run_payload_exchange_xm.sh --role fwd --cell differ
#   host Y:  chain/tools/run_payload_exchange_xm.sh --role rev --cell differ
#
#   host X:  chain/tools/run_payload_exchange_xm.sh --role fwd --cell same-order
#   host Y:  chain/tools/run_payload_exchange_xm.sh --role fwd --cell same-order
#
# Each side publishes its three envelopes into a synced directory inside the
# repository, then polls for the peer's three and judges them. Because the repo
# lives in Dropbox, the transport is the sync itself -- nothing is copied by
# hand and nothing between the two runs edits a payload. The receiving side
# recomputes the peer's block hash from the payload's own fields before
# submitting it, so a file mangled in transit fails the run rather than being
# reported as a consensus verdict.
#
# WHAT THE SECOND MACHINE NEEDS
#
#   * this repository, synced, at the same commit (the accumulation order is a
#     build tag, so both sides compile from the same source);
#   * a Go toolchain and a C compiler for cgo;
#   * nothing else. No network between the machines, no node, no keys. The
#     hosts never talk; they only read each other's files.
#
# Toolchain versions do NOT have to match, and a mismatch is recorded rather
# than rejected -- see the note this script prints when it finds one.
#
# Usage:
#   run_payload_exchange_xm.sh --role fwd|rev
#                              [--cell differ|same-order|same-order-observe]
#                              [--tag NAME] [--dir PATH] [--timeout SECS]
#
#   --tag   must be identical on both machines; it names the exchange directory
#           and keeps one attempt's files from being read by the next. Defaults
#           to the cell plus the UTC date, which is usually already identical
#           on both sides -- pass it explicitly if the two runs might straddle
#           midnight UTC.
#
# Writes: evidence/two_node_payload_exchange_xm_<cell>_<host>_<order>.log
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
geth="$repo/chain/geth"

role=""
cell="differ"
tag=""
dir="$repo/xm-exchange"
timeout=1800

while [ $# -gt 0 ]; do
  case "$1" in
    --role)    role="$2"; shift 2 ;;
    --cell)    cell="$2"; shift 2 ;;
    --tag)     tag="$2"; shift 2 ;;
    --dir)     dir="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^#\( \|$\)//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

case "$role" in
  fwd) order="seq-fwd"; tags="e3lite_vulnerable" ;;
  rev) order="seq-rev"; tags="e3lite_vulnerable e3lite_reverse" ;;
  *)   echo "--role must be fwd or rev" >&2; exit 2 ;;
esac

case "$cell" in
  differ)
    if [ "$role" = fwd ]; then peer_order="seq-rev"; else peer_order="seq-fwd"; fi
    expect=differ
    ;;
  same-order)
    peer_order="$order"
    expect=same-order
    ;;
  same-order-observe)
    peer_order="$order"
    expect=same-order-observe
    ;;
  *) echo "--cell must be differ, same-order or same-order-observe" >&2; exit 2 ;;
esac

[ -n "$tag" ] || tag="$cell-$(date -u +%Y%m%d)"
host="$(hostname)"
xdir="$dir/$tag"
log="$repo/evidence/two_node_payload_exchange_xm_${cell}_${host}_${order}.log"

mkdir -p "$xdir" "$repo/evidence"
: > "$log"

say() { echo "$*" | tee -a "$log"; }

say "Cross-machine payload exchange"
say "date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "host:  $host  ($(uname -srm))"
say "go:    $(cd "$geth" && go version)"
say "cgo:   CGO_ENABLED=${CGO_ENABLED:-<default>}"
say "cell:  $cell   (this side compiles $order, peer must be $peer_order)"
say "tag:   $tag"
say "dir:   $xdir"
say ""
if [ "$cell" = same-order ]; then
  say "Expectation: agreement in all three experiments, float path included."
  say "This cell is what attributes the differ cell's divergence to the compiled"
  say "order rather than to the host. The cell is asymmetric: conclusive when it"
  say "passes, inconclusive when it fails. Agreement between hosts that differ in"
  say "architecture and toolchain as well is stronger than a same-architecture"
  say "pass, because more varied and the commitment still did not move."
elif [ "$cell" = same-order-observe ]; then
  say "No expectation asserted. The roots and verdicts are RECORDED, because on"
  say "a pair of hosts that differ in more than the accumulation order they are"
  say "the measurement rather than a pass/fail criterion. This characterises the"
  say "pair; it does not attribute anything to the order."
else
  say "Expectation: A and A' mutual INVALID, B mutual VALID -- the same verdicts"
  say "as on one machine, now with the two nodes on two physical hosts."
fi
say ""

# Experiment table: key | test name | OUT var | IN var | label
read -r -d '' exps <<'TABLE'
a|TestDFRLTwoNodePayloadExchange|DFRL_PAYLOAD_OUT|DFRL_PAYLOAD_IN|A   mandatory system call, float path
ap|TestDFRLTwoNodeTxPathDiverges|DFRL_TXV_OUT|DFRL_TXV_IN|A'  transaction entry point, float path
b|TestDFRLTwoNodeDeterministicPathAgrees|DFRL_DET_OUT|DFRL_DET_IN|B   transaction entry point, specified path
TABLE

# Some Windows installations run an application-control policy that refuses to
# execute freshly built unsigned binaries, intermittently and by binary
# reputation, so `go test` and a `go test -c` binary are blocked independently.
# Each phase tries both. Nothing about the experiment changes between them.
bindir="$(mktemp -d)"
bin="$bindir/catalyst_${role}.test.exe"
( cd "$geth" && go test -c -tags "$tags" -o "$bin" ./eth/catalyst/ ) >/dev/null 2>&1

run() {
  local phase="$1" test="$2"; shift 2
  {
    echo "########## $phase"
    echo "# build tags: $tags"
    echo "# test:       $test"
    echo "# env:        $*"
  } >> "$log"

  local tmp; tmp="$(mktemp)"
  ( cd "$geth" && env "$@" go test -count=1 -tags "$tags" -run "$test" -v ./eth/catalyst/ ) > "$tmp" 2>&1
  local rc=$? how="go test"
  if [ $rc -ne 0 ] && grep -qi "Application Control policy" "$tmp"; then
    echo "# 'go test' launch blocked by an application-control policy; retrying the prebuilt binary" >> "$log"
    ( cd "$geth/eth/catalyst" && env "$@" "$bin" -test.run "$test" -test.v ) > "$tmp" 2>&1
    rc=$?; how="prebuilt binary"
  fi
  cat "$tmp" >> "$log"
  { echo "# launched via: $how"; echo "# exit=$rc"; echo ""; } >> "$log"
  if [ $rc -ne 0 ] && grep -qi "Application Control policy" "$tmp"; then
    say "  !! both launch methods blocked by this host's policy -- an environment"
    say "     obstacle, not a result. Allow an exception or run on Linux."
  fi
  rm -f "$tmp"
  return $rc
}

# ---------------------------------------------------------------- publish ----
# Our own files are removed first, so that a re-run under the same tag cannot
# have the peer judge a stale envelope from an earlier attempt.
fail=0
say "== publishing this host's three envelopes =="
while IFS='|' read -r key test outvar invar label; do
  [ -n "${key:-}" ] || continue
  rm -f "$xdir/${key}__${order}__${host}.json"
done <<< "$exps"
while IFS='|' read -r key test outvar invar label; do
  [ -n "${key:-}" ] || continue
  out="$xdir/${key}__${order}__${host}.json"
  if run "publish $key  ($label)" "$test" "$outvar=$out"; then
    say "  $label -> $(basename "$out")"
  else
    say "  $label -> FAILED to build its own payload (see the log)"
    fail=1
  fi
done <<< "$exps"

if [ "$fail" -ne 0 ]; then
  say ""
  say "Stopping: this host could not produce all three payloads, so there is"
  say "nothing for the peer to judge. Fix the local failure first."
  say "log: $log"
  exit 1
fi

# ------------------------------------------------------------------- wait -----
# Waits for a file this host did not write. The size-stability and parse checks
# guard against reading a file the sync client is still delivering; the block
# hash is recomputed inside the test as well, so a truncated read fails the run
# rather than producing a verdict.
wait_for_peer() {
  local key="$1" deadline=$((SECONDS + timeout)) found="" announced=0
  local mine="$xdir/${key}__${order}__${host}.json"
  while :; do
    found=""
    for f in "$xdir/${key}__${peer_order}__"*.json; do
      [ -e "$f" ] || continue
      [ "$f" != "$mine" ] || continue
      found="$f"; break
    done
    if [ -n "$found" ]; then
      local s1 s2
      s1=$(wc -c < "$found")
      sleep 2
      s2=$(wc -c < "$found")
      if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$found" >/dev/null 2>&1; then
        echo "$found"
        return 0
      fi
      continue
    fi
    if [ $SECONDS -ge $deadline ]; then
      return 1
    fi
    if [ "$announced" -eq 0 ]; then
      echo "  waiting for the peer's $key envelope ($peer_order, any host but $host)..." | tee -a "$log" >&2
      announced=1
    fi
    sleep 5
  done
}

peer_go_version() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['host'].get('goVersion','?'))" "$1" 2>/dev/null
}

say ""
say "== waiting for the peer and judging =="
fail=0
# Whether any peer envelope actually came from another machine. Rehearsing this
# script on one host is useful and the verdicts come out the same, but it is not
# cross-machine evidence and must not be logged as if it were.
peer_hosts=""
while IFS='|' read -r key test outvar invar label; do
  [ -n "${key:-}" ] || continue
  peer="$(wait_for_peer "$key")"
  if [ -z "$peer" ]; then
    say "  $label -> TIMEOUT after ${timeout}s"
    say "     No file matched ${key}__${peer_order}__*.json in the exchange"
    say "     directory from a host other than $host. Check that the other"
    say "     machine is running with --cell $cell --tag $tag, that its --role"
    say "     gives order $peer_order, and that the directory has synced."
    fail=1
    continue
  fi
  say "  $label <- $(basename "$peer")"
  peer_hosts="$peer_hosts $(python -c "import json,sys; print(json.load(open(sys.argv[1]))['host']['hostname'])" "$peer" 2>/dev/null)"

  # A differing toolchain is not a reason to stop, but it changes what a
  # disagreement in the same-order cell would mean, so it goes in the log.
  peer_go="$(peer_go_version "$peer")"
  peer_arch="$(python -c "import json,sys; h=json.load(open(sys.argv[1]))['host']; print(h['goos']+'/'+h['goarch'])" "$peer" 2>/dev/null)"
  our_go="$(cd "$geth" && go version | awk '{print $3}')"
  our_arch="$(cd "$geth" && go version | awk '{print $4}')"
  if [ -n "$peer_arch" ] && [ "$peer_arch" != "$our_arch" ]; then
    say "     peer $peer_arch vs this host $our_arch"
    if [ "$cell" = same-order ]; then
      say "     More than the order varies between these hosts, so this cell is"
      say "     asymmetric: agreement is STRONGER than a same-architecture pass,"
      say "     while a disagreement is inconclusive -- architecture alone has"
      say "     been measured to move the unspecified path. Read the ATTRIBUTION"
      say "     line on success; on failure, do not read it as a refutation."
    fi
  fi
  if [ -n "$peer_go" ] && [ "$peer_go" != "?" ] && [ "$peer_go" != "$our_go" ]; then
    say "     note: peer built with $peer_go, this host with $our_go."
    if [ "$cell" = same-order ] && [ "$peer_arch" = "$our_arch" ]; then
      say "     Agreement under differing toolchains is a STRONGER result. A"
      say "     disagreement would be inconclusive -- order and toolchain would"
      say "     both be varying -- so match the toolchains before reporting one."
    fi
  fi

  if run "judge $key  ($label)" "$test" "$invar=$peer" "DFRL_XM_EXPECT=$expect"; then
    say "     verdict as expected"
  else
    say "     VERDICT NOT AS EXPECTED -- read the phase output in the log"
    fail=1
  fi
done <<< "$exps"

# ---------------------------------------------------------------- summary ----
{
  echo "########## summary ($host, cell $cell)"
  grep -h "compiled accumulation order\|own payload:\|control: own\|  order \|byte-identical transaction\|post-state root\|peer payload verdict\|RESULT" "$log" | sed 's/^ *//'
  echo ""
} >> "$log"

rm -rf "$bindir"

# Did any envelope come from a host other than this one?
crossed=0
for h in $peer_hosts; do
  [ "$h" = "$host" ] || crossed=1
done

say ""
if [ "$fail" -eq 0 ] && [ "$crossed" -eq 0 ]; then
  say "REHEARSAL, NOT EVIDENCE: every envelope judged here was produced on"
  say "$host. The protocol works and the verdicts are the expected ones, but"
  say "both nodes were this one machine, so this run adds nothing to what"
  say "run_payload_exchange.sh already shows. Run the peer side on the second"
  say "machine and repeat."
  say ""
  say "log: $log"
  exit 0
fi

if [ "$fail" -eq 0 ]; then
  if [ "$cell" = same-order-observe ]; then
    say "observation recorded on $host. Read the OBSERVED / OBSERVATION lines in"
    say "the log. This characterises this pair of hosts; it is NOT attribution to"
    say "the accumulation order, and must not be cited as such."
  elif [ "$cell" = same-order ]; then
    say "same-order cell passed on $host: two physical hosts compiling the same"
    say "accumulation order produce identical commitments and accept each other's"
    say "blocks on all three paths, the unspecified float paths included."
    say "Together with the differ cell, the divergence is attributable to the"
    say "compiled order rather than to the machine."
  else
    say "differ cell passed on $host: mutual rejection on both unspecified paths"
    say "and mutual acceptance on the specified one, with the two nodes on two"
    say "physical hosts."
  fi
  if [ "$cell" = differ ]; then
    say "Now run the attribution cell if you have not:  --cell same-order"
    say "(two hosts of the SAME architecture; otherwise --cell same-order-observe)"
  fi
else
  say "AT LEAST ONE STEP DID NOT COME OUT AS EXPECTED on $host -- read the log."
fi
say ""
say "log: $log"
say "The peer host writes its own log; both are needed, since each side's log"
say "records only the verdicts that side reached."
exit "$fail"
