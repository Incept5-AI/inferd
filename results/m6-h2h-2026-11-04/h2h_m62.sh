#!/bin/zsh
# h2h_m62.sh (2026-11-04, mini-m6-2, macOS 27.2): (1) inferd v0.2.0 27B control cells to compare with mini-m6 (27.0);
# (2) TensorFold 1.2.2 cannot load the 27B on 32 GB (tfprobe), so: Qwen3.5-9B 4-bit on TensorFold 1.2.2 vs inferd v0.2.0,
# same checkpoint, single agent + follow-ups. One model process at a time.
zmodload zsh/datetime
O=~/inference/results/m6-h2h-2; mkdir -p $O/logs; L=$O/progress.log; PY=~/.venvs/mlxvlm/bin/python
M27=~/models/Qwen3.8-27B-MLX-4bit; M9=~/models/Qwen3.5-9B-MLX-4bit
DRAFT=$(ls -d ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/* | head -1)
E=~/inference/rel020/engine; CONC=~/inference/conc.py; TF=~/exh2h/tf-1.2.2/bin/tensorfold-native
say() { echo "$(date +%H:%M:%S) $*" | tee -a $L; }
port_pid() { lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null | head -1; }
inferd_up() { local lab=$1 m=$2; shift 2; rm -rf /tmp/h2hspill; mkdir -p /tmp/h2hspill
  (cd $E && MODEL=$m PY=$PY SPILL_DIR=/tmp/h2hspill SPILL_GIB=96 "$@" ./run.sh 8320 16 1024 0 8 32768 > $O/logs/run-$lab.txt 2>&1) || { say "$lab: inferd failed"; return 1; }
  say "$lab up: $(grep -E 'memory:|KV blocks' ~/inference/logs/inferd-8320.log | tail -2 | tr '\n' ' ')"; }
inferd_down() { cp ~/inference/logs/inferd-8320.log $O/logs/inferd-$1.log; (cd $E && ./run.sh stop 8320 >> $O/logs/run-$1.txt 2>&1); sleep 15; rm -rf /tmp/h2hspill; }
single() { for r in 1 2; do python3 $CONC --url $1 --model $2 --label $3-r$r --mode cold --ctx 4k,16k --n 1 --out $O/single.jsonl > $O/single-$3-r$r.txt 2>&1; done
  say "$3 single: $(grep -h -E '^ +(4k|16k) +1 ' $O/single-$3-r*.txt | tr -s ' ' | cut -d' ' -f1-8 | tr '\n' ';')"; }
follow() { python3 $CONC --url $1 --model $2 --label $3 --mode followup --ctx 4k,16k --n 1,4,8,16 --out $O/followup.jsonl > $O/followup-$3.txt 2>&1
  say "$3 follow-ups: $(grep -E '^ +(4k|16k) ' $O/followup-$3.txt | tr -s ' ' | cut -d' ' -f1-7 | tr '\n' ';')"; }
say "##### h2h m6-2 start"
# (1) 27B control, the same server line and harness as mini-m6
env DRAFT=$DRAFT DRAFT_BITS=4 true
inferd_up inferd27-ctl $M27 env DRAFT=$DRAFT DRAFT_BITS=4 && {
  single http://127.0.0.1:8320 qwen3.8-27b inferd-m62
  python3 $CONC --url http://127.0.0.1:8320 --model qwen3.8-27b --label inferd-m62-ctl --mode followup --ctx 4k --n 8 --out $O/control.jsonl > $O/control-inferd-m62-4k.txt 2>&1
  python3 $CONC --url http://127.0.0.1:8320 --model qwen3.8-27b --label inferd-m62-ctl --mode followup --ctx 16k --n 4 --out $O/control.jsonl > $O/control-inferd-m62-16k.txt 2>&1
  say "inferd control m6-2: $(tail -1 $O/control-inferd-m62-4k.txt | tr -s ' ') | $(tail -1 $O/control-inferd-m62-16k.txt | tr -s ' ')"
  inferd_down inferd27-ctl; }
# (2) Qwen3.5-9B: wait for the checkpoint copy
while [ ! -f $M9/.COPIED ]; do sleep 20; done
nohup $TF serve $M9 --port 8091 --name q9 --no-thinking --no-update-check > $O/logs/tf-q9.log 2>&1 < /dev/null &
TP=$!; ok=0; for i in $(seq 600); do curl -sf -m 5 localhost:8091/v1/models >/dev/null && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
if [ $ok = 1 ]; then say "tf q9 up: $(grep -i -E 'parallel|stream|context|GiB' $O/logs/tf-q9.log | head -4 | tr '\n' ' ' | cut -c1-400)"
  single http://127.0.0.1:8091 q9 tf122-q9; follow http://127.0.0.1:8091 q9 tf122-q9
else say "tf q9 failed: $(tail -2 $O/logs/tf-q9.log)"; fi
kill $TP 2>/dev/null; for i in $(seq 60); do kill -0 $TP 2>/dev/null || break; sleep 1; done; sleep 15
inferd_up inferd-q9 $M9 env && { single http://127.0.0.1:8320 qwen3.5-9b inferd-q9; follow http://127.0.0.1:8320 qwen3.5-9b inferd-q9; inferd_down inferd-q9; }
say "##### h2h m6-2 done"; touch $O/DONE
