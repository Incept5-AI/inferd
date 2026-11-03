#!/bin/zsh
# release_gate.sh MODEL [DRAFT]: the hardware gate for a release candidate on one Mac (2026-11-03). Run from engine/
# after ./build.sh, with no other model process on the machine. Steps, each must pass:
#   1. host tests (grammar, json, spill, weights) and the front-end tests
#   2. decode_test check at chunks 1 and 24 vs the mlx_lm reference (top-1 ok, cos >= 0.9999 on seqs 1-3)
#   3. decode_test bench 1,16,64 at 2048 (recorded; compare with the previous release by hand)
#   4. spec_test: speculative output equals plain decode (with DRAFT)
# It writes gate-<date>.log here. The fleet and long-context gates (conc.py 16k x 8/16/32, wall.py N=1..8, mixed
# arrivals, SSD pressure) need a running server and are listed in docs/TESTING.md.
set -u
M=${1:?usage: release_gate.sh MODEL [DRAFT]}; D=${2:-}
PY=${PY:-$HOME/.venvs/inferd/bin/python}
LOG=gate-$(date +%Y%m%d-%H%M).log
step() { echo "== $1" | tee -a $LOG; shift; "$@" >> $LOG 2>&1 && echo "   ok" | tee -a $LOG || { echo "   FAILED (see $LOG)" | tee -a $LOG; exit 1; }; }
[ -x build/inferd ] || { echo "run ./build.sh first"; exit 1; }
step "host: grammar" ./build/grammar_test
step "host: json" ./build/json_test
step "host: spill" ./build/spill_test
step "host: weights" ./build/weights_test
for t in test_structured test_tools test_stream; do step "front end: $t" zsh -c "cd frontend && $PY $t.py"; done
T=$(mktemp -d)
step "reference tokens" $PY tools/ref_logits.py make "$M" $T/toks.txt 4 24
for c in 1 24; do
  step "logits, chunk $c" ./build/decode_test check "$M" $T/toks.txt $T/eng.bin $c
  step "compare, chunk $c" $PY tools/ref_logits.py compare "$M" $T/toks.txt $T/eng.bin
done
step "bench" ./build/decode_test bench "$M" 1,16,64 2048
[ -n "$D" ] && step "speculative = plain" ./build/spec_test "$M" "$D" $T/toks.txt 128
rm -rf $T
echo "gate passed: $LOG"
