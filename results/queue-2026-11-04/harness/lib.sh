# lib.sh: helpers for queue jobs (source it). Host paths come from ~/inference/queue/host.env:
#   M27 (Qwen3.8-27B 4-bit), DRAFT (DFlash2), M9 (Qwen3.5-9B 4-bit), M9_8 (8-bit), E (inferd engine dir),
#   PY (front-end python), TF (tensorfold-native), AG (agents.py), MEMS (memsample.sh)
zmodload zsh/datetime
export PATH=/opt/homebrew/bin:$PATH
source ~/inference/queue/host.env
B=~/inference/queue/bench; CONC=$B/conc.py; SC=$B/agents_scenarios.py   # one harness on every box: Incept5-AI/inferd bench/
O=~/inference/results/queue/${JOB:?set JOB}; mkdir -p $O/logs; L=$O/progress.log
say() { echo "$(date '+%F %T') $*" | tee -a $L >> ~/inference/queue/summary.log; }
# a measurement whose client crashed marks the job failed (the runner moves it to failed/) instead of passing silently
check() { if grep -l -E '^Traceback|Error:' "$@" >/dev/null 2>&1; then say "$JOB CLIENT ERROR in $(grep -l -E '^Traceback|Error:' "$@" | xargs -n1 basename | tr '\n' ' '): $(grep -h -E 'Error' "$@" | head -1)"; FAILED=1; fi; }
trap '[ -n "$FAILED" ] && exit 1' EXIT
port_pid() { lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null | head -1; }
stop_port() { for p in $(lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null); do kill $p; done
  for i in $(seq 90); do [ -z "$(port_pid $1)" ] && break; sleep 1; done; sleep 15; }
# inferd_up LABEL MODEL [VAR=value ...] -- run.sh args   (port 8320; SSD tier on)
inferd_up() { local lab=$1 m=$2; shift 2; local envs=(); while [ "$1" != "--" ]; do envs+=$1; shift; done; shift
  rm -rf /tmp/qspill; mkdir -p /tmp/qspill
  (cd $E && env $envs MODEL=$m PY=$PY SPILL_DIR=/tmp/qspill SPILL_GIB=96 ./run.sh 8320 "$@" > $O/logs/run-$lab.txt 2>&1) || { say "$JOB $lab: inferd failed: $(tail -2 $O/logs/run-$lab.txt | tr '\n' ' ')"; return 1; }
  IPID=$(cd $E && ./run.sh status 8320 2>/dev/null | awk '$1=="inferd"{print $2; exit}')
  MS=$(nohup $MEMS $O/mem-$lab.tsv ${IPID:-0} > /dev/null 2>&1 < /dev/null & echo $!)
  URL=http://127.0.0.1:8320; MID=inferd
  say "$JOB $lab up: $(grep -E 'memory:|KV blocks' ~/inference/logs/inferd-8320.log | tail -2 | tr '\n' ' ' | cut -c1-300)"; }
inferd_down() { kill $MS 2>/dev/null; cp ~/inference/logs/inferd-8320.log $O/logs/inferd-$1.log; (cd $E && ./run.sh stop 8320 >> $O/logs/run-$1.txt 2>&1); sleep 15; rm -rf /tmp/qspill; }
# tf_up LABEL MODEL [tensorfold serve args...]   (port 8091)
tf_up() { local lab=$1 m=$2; shift 2
  nohup $TF serve $m --port 8091 --name tf --no-thinking --no-update-check "$@" > $O/logs/tf-$lab.log 2>&1 < /dev/null &
  TP=$!; for i in $(seq 900); do curl -sf -m 5 localhost:8091/v1/models >/dev/null && break; kill -0 $TP 2>/dev/null || { say "$JOB $lab: tensorfold refused: $(tail -2 $O/logs/tf-$lab.log | tr '\n' ' ' | cut -c1-300)"; return 1; }; sleep 1; done
  MS=$(nohup $MEMS $O/mem-$lab.tsv $TP > /dev/null 2>&1 < /dev/null & echo $!); URL=http://127.0.0.1:8091; MID=tf
  say "$JOB $lab up (tensorfold): $(grep -i -E 'parallel|context|GiB' $O/logs/tf-$lab.log | head -3 | tr '\n' ' ' | cut -c1-300)"; }
tf_down() { kill $MS 2>/dev/null; kill $TP 2>/dev/null; for i in $(seq 60); do kill -0 $TP 2>/dev/null || break; sleep 1; done; stop_port 8091; }
# measurements: each writes a .txt per label and appends rows to a shared jsonl in $O
single() { for r in ${=REPS:-1 2}; do python3 $CONC --url $URL --model $MID --label $1-r$r --mode cold --ctx ${CTX:-4k,16k} --n 1 --out $O/single.jsonl > $O/single-$1-r$r.txt 2>&1; done
  check $O/single-$1-r*.txt; say "$JOB $1 single: $(grep -h -E '^ +[0-9]+k +1 ' $O/single-$1-r*.txt | tr -s ' ' | cut -d' ' -f1-8 | tr '\n' ';')"; }
follow() { python3 $CONC --url $URL --model $MID --label $1 --mode followup --ctx ${CTX:-4k,16k} --n ${NS:-1,4,8,16} --out $O/followup.jsonl > $O/followup-$1.txt 2>&1
  check $O/followup-$1.txt; say "$JOB $1 follow-ups: $(grep -E '^ +[0-9]+k ' $O/followup-$1.txt | tr -s ' ' | cut -d' ' -f1-7 | tr '\n' ';')"; }
fleet() { $PY $AG --url $URL --model $MID --label $1 --ctx ${FCTX:-8k} --agents 64 --cells ${CELLS:-8:8,16:16,32:32} --rounds 2 --cell-budget 3600 \
  --out $O/fleet.jsonl > $O/fleet-$1.txt 2>&1; check $O/fleet-$1.txt; say "$JOB $1 fleet: $(grep -o -E 'turns_per_min": [0-9.]+' $O/fleet.jsonl | tail -3 | tr '\n' ';')"; }
scen() { python3 -u $SC --url $URL/v1 --model $MID --mode $2 --preset m6 --scenario ${SCEN:-b,c} --label $1 --out $O/scen.jsonl ${3:+--pid} ${3:-} > $O/scen-$1.txt 2>&1
  check $O/scen-$1.txt; say "$JOB $1 scen: $(grep -E 'n=' $O/scen-$1.txt | cut -c1-110 | tr '\n' ';')"; }
# splash_up LABEL MODEL WINDOW [draft dir]   (port 8100; the 1 Nov night flags; MODEL = HF id or local dir)
splash_up() { local lab=$1 m=$2 w=$3 d=${4:-}; rm -rf /tmp/qkv; mkdir -p /tmp/qkv
  nohup splash serve --model $m ${d:+--draft-model} ${d:-} --offline --port 8100 --host 127.0.0.1 --no-webui --max-context $w \
    --default-reasoning-effort none --idle-release off --persistent-cache --cache-dir /tmp/qkv --max-cache-disk 96G > $O/logs/splash-$lab.log 2>&1 < /dev/null &
  SP=$!; for i in $(seq 360); do curl -sf -m 5 localhost:8100/v1/models >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  MID=$(curl -sf -m 5 localhost:8100/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -z "$MID" ] && { say "$JOB $lab: splash failed: $(tail -2 $O/logs/splash-$lab.log | tr '\n' ' ' | cut -c1-300)"; kill $SP 2>/dev/null; return 1; }
  MS=$(nohup $MEMS $O/mem-$lab.tsv $(port_pid 8100) > /dev/null 2>&1 < /dev/null & echo $!); URL=http://127.0.0.1:8100
  say "$JOB $lab up (splash $(splash --version 2>&1 | tail -1)): $MID"; }
splash_down() { kill $MS 2>/dev/null; stop_port 8100; kill $SP 2>/dev/null; rm -rf /tmp/qkv; }
