#!/bin/zsh
# release_gate.sh MODEL [DRAFT]: the hardware gate for a release candidate on one Mac (2026-11-03; 2026-11-04 Codex
# review R04/R08). Run from engine/ after ./build.sh, with no other model process on the machine. Steps, each must pass:
#   1. host tests (grammar, json, spill, weights), spec_sample_test (the GPU speculative sampler's top-K and accept /
#      resample distribution vs src/sampling.h, no model), the front-end tests (hardening, structured output, tools, streaming,
#      image decoder, vision) and the logits-gate fixtures (compare must FAIL on deliberately bad logits)
#   2. decode_test check at chunks 1, 24, 256 and 1536 on 4 prompts of 2,048 tokens vs the mlx_lm reference
#      (ref_logits.py compare: finite logits, top-1 (explicit bf16 near-ties only), cos >= 0.995 and KL <= 0.03 on the
#      real-text sequences 1-3 at this length (0.999 / 0.01 for <= 48-token prompts), cos >= 0.99 / KL <= 0.05 on the
#      repetitive sequence 0). 256 and 1536 rows reach the
#      large-M prefill kernels the macOS 26 target broke (build.sh); 24-token prompts never did.
#   3. decode_test bench 1,16,64 at 2048 (recorded; compare with the previous release by hand). Skipped, with a note,
#      when Metal's GPU budget is under 40 GiB (64 x 2048 does not fit there).
#   4. spec_test: speculative output equals plain decode (with DRAFT), on the short prompts
# It writes gate-<date>.log here, headed by the machine, OS, build (metallib sha-256) and model identities. The fleet and
# long-context gates (conc.py 16k x 8/16/32, wall.py N=1..8, mixed arrivals, SSD pressure) need a running server and are
# listed in docs/TESTING.md.
set -u
M=${1:?usage: release_gate.sh MODEL [DRAFT]}; D=${2:-}
PY=${PY:-$HOME/.venvs/inferd/bin/python}
LOG=gate-$(date +%Y%m%d-%H%M).log
step() { echo "== $1" | tee -a $LOG; shift; "$@" >> $LOG 2>&1 && echo "   ok" | tee -a $LOG || { echo "   FAILED (see $LOG)" | tee -a $LOG; exit 1; }; }
note() { echo "== $1" | tee -a $LOG; }
[ -x build/inferd ] || { echo "run ./build.sh first"; exit 1; }
sha() { shasum -a 256 "$1" 2>/dev/null | cut -c1-16; }
{
  echo "# release gate $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(( $(sysctl -n hw.memsize) >> 30 )) GiB"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(uname -r)"
  echo "xcode: $(xcrun metal --version 2>/dev/null | head -1)"
  echo "metallib: build/core.metallib sha256 $(sha build/core.metallib)"
  echo "inferd: build/inferd sha256 $(sha build/inferd)"
  echo "source: $(git rev-parse HEAD 2>/dev/null || echo 'no git checkout')$(git status --porcelain 2>/dev/null | grep -q . && echo ' (dirty)')"
  echo "model: $M (config sha256 $(sha $M/config.json))"
  [ -n "$D" ] && echo "draft: $D (config sha256 $(sha $D/config.json))"
  echo "python: $PY ($($PY -c 'import sys, PIL; print(sys.version.split()[0], "Pillow", PIL.__version__)' 2>&1))"
} | tee -a $LOG
step "host: grammar" ./build/grammar_test
step "host: json" ./build/json_test
step "host: spill" ./build/spill_test
step "host: weights" ./build/weights_test
step "gpu: speculative sampler = the sampling contract" ./build/spec_sample_test
for t in test_hardening test_structured test_tools test_stream test_image_decode test_vision; do
  step "front end: $t" zsh -c "cd frontend && $PY $t.py"
done
step "logits gate fails on bad logits (fixtures)" $PY tools/test_ref_logits.py
T=$(mktemp -d)
step "reference tokens: 4 x 2048 (seq 0 repetitive, 1-3 real text)" $PY tools/ref_logits.py make "$M" $T/toks.txt 4 2048
for c in 1 24 256 1536; do
  step "logits, chunk $c" ./build/decode_test check "$M" $T/toks.txt $T/eng.bin $c
  step "compare, chunk $c" $PY tools/ref_logits.py compare "$M" $T/toks.txt $T/eng.bin
done
BUDGET=$(./build/decode_test budget 2>/dev/null || echo 0)
echo "gpu budget: $BUDGET GiB" >> $LOG
if (( BUDGET >= 40 )); then
  step "bench" ./build/decode_test bench "$M" 1,16,64 2048
else
  note "bench SKIPPED: GPU budget $BUDGET GiB < 40 (64 x 2048 does not fit); run 'decode_test bench $M 1,16 2048' by hand"
fi
if [ -n "$D" ]; then
  step "reference tokens: 4 x 24 (speculative check)" $PY tools/ref_logits.py make "$M" $T/short.txt 4 24
  step "speculative = plain" ./build/spec_test "$M" "$D" $T/short.txt 128
fi
rm -rf $T
echo "gate passed: $LOG" | tee -a $LOG
