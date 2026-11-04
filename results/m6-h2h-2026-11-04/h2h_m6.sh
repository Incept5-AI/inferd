#!/bin/zsh
# h2h_m6.sh (2026-11-04, mini-m6, macOS 27.0): inferd v0.2.0 and Splash 1.5.0 (latest) on the same weights + DFlash2.
#  single: conc.py --mode cold --ctx 4k,16k --n 1, 2 reps    follow-ups: conc.py --mode followup (Splash; inferd control cells)
#  fleet: agents.py 8k, cells 8:8,16:16,32:32, 2 rounds (the 1 Nov night harness)    scenarios: agents_scenarios.py b,c
# One model process at a time; servers stopped by pid / listening port.
zmodload zsh/datetime
export PATH=/opt/homebrew/bin:$PATH
O=~/inference/results/m6-h2h; mkdir -p $O/logs; L=$O/progress.log; PY=~/.venvs/mlxvlm/bin/python
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/* | head -1)
DRAFT=$(ls -d ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/* | head -1)
E=~/inference/rel020/engine; CONC=~/inference/conc.py; SC=~/inference/agents_scenarios.py
say() { echo "$(date +%H:%M:%S) $*" | tee -a $L; }
port_pid() { lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null | head -1; }
stop_port() { for p in $(lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null); do kill $p; done
  for i in $(seq 90); do [ -z "$(port_pid $1)" ] && break; sleep 1; done; sleep 15; }
inferd_up() {  # label env... -- run.sh args
  local lab=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=$1; shift; done; shift
  rm -rf /tmp/h2hspill; mkdir -p /tmp/h2hspill
  (cd $E && env $envs MODEL=$MODEL DRAFT=$DRAFT DRAFT_BITS=4 PY=$PY SPILL_DIR=/tmp/h2hspill SPILL_GIB=96 ./run.sh 8300 "$@" > $O/logs/run-$lab.txt 2>&1) || { say "$lab: inferd failed"; return 1; }
  cp ~/inference/logs/inferd-8300.log $O/logs/inferd-$lab-start.log
  IPID=$(cd $E && ./run.sh status 8300 2>/dev/null | grep -o -E 'inferd pid [0-9]+' | grep -o -E '[0-9]+$')
  MS=$(nohup ~/m6/memsample.sh $O/mem-$lab.tsv ${IPID:-0} > /dev/null 2>&1 < /dev/null & echo $!)
  say "$lab up: $(grep -E 'memory:|KV blocks' ~/inference/logs/inferd-8300.log | tail -2 | tr '\n' ' ')"
}
inferd_down() { kill $MS 2>/dev/null; cp ~/inference/logs/inferd-8300.log $O/logs/inferd-$1.log; (cd $E && ./run.sh stop 8300 >> $O/logs/run-$1.txt 2>&1); sleep 15; rm -rf /tmp/h2hspill; }
single() {  # url model label
  for r in 1 2; do python3 $CONC --url $1 --model $2 --label $3-r$r --mode cold --ctx 4k,16k --n 1 --out $O/single.jsonl > $O/single-$3-r$r.txt 2>&1; done
  say "$3 single: $(grep -E '^ +(4k|16k) +1 ' $O/single-$3-r*.txt | tr -s ' ' | cut -d' ' -f1-8 | tr '\n' ';')"
}
fleet() { $PY ~/m6/agents.py --url $1 --model $2 --label $3 --ctx 8k --agents 64 --cells 8:8,16:16,32:32 --rounds 2 --cell-budget 3600 \
  ${4:+--server-log=$4} --out $O/fleet.jsonl > $O/fleet-$3.txt 2>&1; say "$3 fleet: $(grep -o -E '"n": [0-9]+.{0,120}turns_per_min": [0-9.]+' $O/fleet.jsonl | grep -A0 . | tail -3 | tr '\n' ';' | cut -c1-500)"; }
scen() { python3 -u $SC --url $1/v1 --model $2 --mode $3 --preset m6 --scenario b,c --label $4 --out $O/scen.jsonl ${5:+--pid $5} > $O/scen-$4.txt 2>&1; say "$4 scen done: $(grep -E 'n=' $O/scen-$4.txt | cut -c1-110 | tr '\n' ';')"; }
say "##### h2h m6 start"
# A: inferd v0.2.0, the 32 GB server line from run.sh's header
inferd_up inferd-main -- 16 1024 0 8 32768 && {
  single http://127.0.0.1:8300 qwen3.8-27b inferd-m6
  python3 $CONC --url http://127.0.0.1:8300 --model qwen3.8-27b --label inferd-m6-ctl --mode followup --ctx 4k --n 8 --out $O/control.jsonl > $O/control-inferd-m6-4k.txt 2>&1
  python3 $CONC --url http://127.0.0.1:8300 --model qwen3.8-27b --label inferd-m6-ctl --mode followup --ctx 16k --n 4 --out $O/control.jsonl > $O/control-inferd-m6-16k.txt 2>&1
  say "inferd control m6: $(tail -1 $O/control-inferd-m6-4k.txt | tr -s ' ') | $(tail -1 $O/control-inferd-m6-16k.txt | tr -s ' ')"
  scen http://127.0.0.1:8300 qwen3.8-27b api inferd-m6 $IPID
  inferd_down inferd-main; }
# B/C: inferd fleets, the night settings (8 running, 8 snapshots, 12k window)
inferd_up inferd-fleet -- 8 1024 0 8 12288 && { fleet http://127.0.0.1:8300 qwen3.8-27b inferd-m6-8k ~/inference/logs/inferd-8300.log; inferd_down inferd-fleet; }
inferd_up inferd-fleet-ane ANE_UNITS=24 ANE_MIN_ROWS=1024 -- 8 1024 0 8 12288 && { fleet http://127.0.0.1:8300 qwen3.8-27b inferdane24-m6-8k ~/inference/logs/inferd-8300.log; inferd_down inferd-fleet-ane; }
# D: Splash 1.5.0 (latest stable), the night flags
splash_up() {  # label window
  rm -rf /tmp/h2hkv; mkdir -p /tmp/h2hkv
  nohup splash serve --model mlx-community/Qwen3.8-27B-4bit --draft-model $DRAFT --offline --port 8100 --host 127.0.0.1 --no-webui --max-context $2 \
    --default-reasoning-effort none --idle-release off --persistent-cache --cache-dir /tmp/h2hkv --max-cache-disk 96G > $O/logs/splash-$1.log 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 360); do curl -sf -m 5 localhost:8100/v1/models >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  SMID=$(curl -sf -m 5 localhost:8100/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -z "$SMID" ] && { say "splash $1 failed: $(tail -2 $O/logs/splash-$1.log)"; return 1; }
  MS=$(nohup ~/m6/memsample.sh $O/mem-splash-$1.tsv $(port_pid 8100) > /dev/null 2>&1 < /dev/null & echo $!)
  say "splash $1 up: $SMID"
}
splash_down() { kill $MS 2>/dev/null; stop_port 8100; kill $SP 2>/dev/null; rm -rf /tmp/h2hkv; }
splash_up main 32768 && {
  single http://127.0.0.1:8100 $SMID splash-m6
  python3 $CONC --url http://127.0.0.1:8100 --model $SMID --label splash-m6 --mode followup --ctx 4k,16k --n 1,4,8,16 --out $O/followup.jsonl > $O/followup-splash-m6.txt 2>&1
  say "splash follow-ups: $(grep -E '^ +(4k|16k) ' $O/followup-splash-m6.txt | tr -s ' ' | cut -d' ' -f1-7 | tr '\n' ';')"
  scen http://127.0.0.1:8100 $SMID plain splash-m6 $(port_pid 8100)
  splash_down; }
splash_up fleet 12288 && { fleet http://127.0.0.1:8100 $SMID splash-m6-8k $O/logs/splash-fleet.log; splash_down; }
say "##### h2h m6 done"; touch $O/DONE
