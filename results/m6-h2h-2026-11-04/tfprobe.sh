#!/bin/zsh
# tfprobe.sh (2026-11-04): which TensorFold 1.2.2 configs load Qwen3.8-27B 4-bit on the 32 GB M6. Most capable first,
# with and without the DFlash2 drafter; each success gets one tiny request, then is stopped by pid.
zmodload zsh/datetime
O=~/inference/results/m6-h2h; mkdir -p $O/tfprobe; L=$O/tfprobe/probe.log
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/* | head -1)
DRAFT=$(ls -d ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/* | head -1)
TF=~/exh2h/tf-1.2.2/bin/tensorfold-native
try() {  # label flags...
  local lab=$1; shift
  nohup $TF serve $MODEL --port 8091 --name q27 --no-thinking --no-update-check "$@" > $O/tfprobe/$lab.log 2>&1 < /dev/null &
  local p=$! ok=0
  for i in $(seq 240); do curl -sf -m 5 localhost:8091/v1/models >/dev/null && { ok=1; break; }; kill -0 $p 2>/dev/null || break; sleep 1; done
  if [ $ok = 1 ]; then
    r=$(curl -s -m 300 localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"q27","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}' | head -c 300)
    echo "$(date +%T) LOADS $lab ($*): $r" | tee -a $L
    kill $p; for i in $(seq 60); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; sleep 10
    return 0
  fi
  kill $p 2>/dev/null; sleep 2
  echo "$(date +%T) FAILS $lab ($*): $(tail -2 $O/tfprobe/$lab.log | tr '\n' ' ' | cut -c1-300)" | tee -a $L; return 1
}
for d in draft nodraft; do
  [ $d = draft ] && DF=(--drafter $DRAFT) || DF=(--no-drafts)
  for p in 4 2 1; do for c in 16384 8192 4096; do
    try $d-p$p-c$c $DF --parallel $p --context $c && break 2
  done; done
done
# the default (no --parallel) with the drafter, for the single-agent arm
try draft-default --drafter $DRAFT --context 16384 || try draft-default8k --drafter $DRAFT --context 8192
echo "$(date +%T) probe done" | tee -a $L; touch $O/tfprobe/DONE
