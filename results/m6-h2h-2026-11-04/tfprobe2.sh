#!/bin/zsh
# tfprobe.sh (2026-11-04): which TensorFold 1.2.2 configs load Qwen3.8-27B 4-bit on the 32 GB M6. Most capable first,
# with and without the DFlash2 drafter; each success gets one tiny request, then is stopped by pid.
zmodload zsh/datetime
O=~/inference/results/m6-h2h-2; mkdir -p $O/tfprobe; L=$O/tfprobe/probe.log
MODEL=~/models/Qwen3.8-27B-MLX-4bit
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
# mini-m6-2 (macOS 27.2): below 4k context, and --profile stable (smallest configs that failed on mini-m6)
try nodraft-p1-c2048 --no-drafts --parallel 1 --context 2048 || try nodraft-p1-c1024 --no-drafts --parallel 1 --context 1024
try nodraft-p1-c4096-stable --no-drafts --parallel 1 --context 4096 --profile stable
try draft-p1-c2048 --drafter $DRAFT --parallel 1 --context 2048
try nodraft-p1-c4096 --no-drafts --parallel 1 --context 4096
echo "$(date +%T) probe done" | tee -a $L; touch $O/tfprobe/DONE
