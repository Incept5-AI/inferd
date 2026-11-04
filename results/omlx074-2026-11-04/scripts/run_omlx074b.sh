#!/bin/zsh
# run_omlx074.sh (2026-11-04): inferd 0.2.0 (~/inference/rel020/engine, prebuilt, gated) vs oMLX 0.7.4.dev1
# (~/.venvs/omlx074) and 0.7.3 (~/.venvs/omlx073) on the Studio, same weights and harnesses as RESULTS (b1)/(h).
# One model process at a time: every arm takes the shared lock (re-sourced per acquire) and releases it.
# oMLX homes live under <models>/omlx074/<arm> and are only deleted when they carry our marker file.
# Servers are stopped by pid and port, never by command-line pattern.
#   usage: run_omlx074.sh [arm ...]   (default: all arms in order)
unsetopt BG_NICE
zmodload zsh/datetime
O=~/inference/results/omlx074; S=~/inference/omlx074; B=~/inference/bench; mkdir -p $O/raw $O/logs $O/mem
L=$O/logs/driver.log
ENG=~/inference/rel020/engine; PY=~/.venvs/mlxvlm/bin/python
Q27=$(ls -d <models>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/* | head -1)
Q8=$(ls -d <models>/models--mlx-community--Qwen3.8-27B-8bit/snapshots/* | head -1)
FN=$(ls -d <models>/models--Jundot--Qwen3.8-Flash-Next-oQ4e-mtp/snapshots/* | head -1)
# incoai snapshot dir was empty on 2026-11-04 (blobs only); z-lab is byte-identical (same blob hashes)
DI=$(ls -d <models>/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/* | head -1)
H0=<models>/omlx074; P=8374
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a $L; }
qlog() { echo "$(date '+%Y-%m-%d %H:%M:%S') omlx074: $*" >> $O/logs/queue.log; }
lock() { source ~/inference/studio_lock.sh; studio_lock_acquire; say "== lock"; }
unlock() { studio_lock_release; trap - EXIT INT TERM HUP; say "== unlock"; }
up() { for i in $(seq 1200); do curl -sf -m 5 localhost:$1/v1/models > /dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }
stop_port() { for p in $(lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null); do kill $p 2>/dev/null; done
  for i in $(seq 60); do [ -z "$(lsof -ti tcp:$1 -sTCP:LISTEN 2>/dev/null)" ] && break; sleep 1; done; sleep 10; }
memstart() { nohup zsh ~/inference/pfwork/memsample.sh $O/mem/$1.tsv $2 > /dev/null 2>&1 < /dev/null &; }

# ---- tests (url model label) ----
t_warm()  { python3 $B/wall.py --url http://127.0.0.1:$P --model $1 --label $2-warm --n 1 --prompt 2048 --gen 64 --out $O/raw/warm.jsonl >> $O/logs/$2.txt 2>&1; }
t_conc()  { python3 $B/conc.py --url http://127.0.0.1:$P --model $1 --label $2 --mode followup --ctx 16k --n 1,8,32 --out $O/raw/conc.jsonl >> $O/logs/$2.txt 2>&1; say "$2 conc rc $?"; }
t_wall()  { python3 $B/wall.py --url http://127.0.0.1:$P --model $1 --label $2 --n ${WALLN:-1,4,8} --trials ${TRIALS:-3} --out $O/raw/wall.jsonl >> $O/logs/$2.txt 2>&1; say "$2 wall rc $?"; }
t_speed() { python3 $B/speed.py --url http://127.0.0.1:$P --model $1 --label $2 --sizes ${SIZES:-1k,8k,16k} --reps 2 --out $O/raw/speed.jsonl >> $O/logs/$2.txt 2>&1; say "$2 speed rc $?"; }
t_b2b()   { python3 $S/b2b.py --url http://127.0.0.1:$P --model $1 --label $2 --rounds 2 --out $O/raw/b2b.jsonl >> $O/logs/$2.txt 2>&1; say "$2 b2b rc $?"; }
t_six()   { python3 $B/wall.py --url http://127.0.0.1:$P --model $1 --label $2-six --n 6 --prompt 1024 --gen 1024 --trials 3 --out $O/raw/six.jsonl >> $O/logs/$2.txt 2>&1
            python3 $B/wall.py --url http://127.0.0.1:$P --model $1 --label $2-one --n 1 --prompt 1024 --gen 1024 --trials 2 --out $O/raw/six.jsonl >> $O/logs/$2.txt 2>&1; say "$2 six rc $?"; }

inferd_arm() {  # label model "run.sh args" "env" tests...
  local lab=$1 m=$2 args=$3 ev=$4; shift 4
  lock; say "## inferd $lab ($args; $ev)"; stop_port $P
  (cd $ENG && env ${=ev} MODEL=$m PY=$PY ./run.sh $P ${=args} > $O/logs/$lab-start.txt 2>&1)
  local pid=$(lsof -ti tcp:$P -sTCP:LISTEN | head -1)
  if [ -n "$pid" ] && up $P $pid; then
    memstart $lab $pid; say "$lab up: $(grep -E 'memory:|KV blocks' ~/inference/logs/inferd-$P.log | tr '\n' ' ' | cut -c1-300)"
    t_warm qwen $lab; for t in $@; do t_$t qwen $lab; done
  else say "$lab failed to start: $(tail -3 $O/logs/$lab-start.txt | tr '\n' ' ')"; fi
  cp ~/inference/logs/inferd-$P.log $O/logs/$lab-inferd.log 2>/dev/null
  (cd $ENG && ./run.sh stop $P >> $O/logs/$lab-start.txt 2>&1); stop_port $P; unlock
}

omlx_arm() {  # label venv model id settings-json "extra serve args" tests...
  local lab=$1 venv=$2 m=$3 id=$4 ms=$5 extra=$6; shift 6
  local H=$H0/$lab
  if [ -e $H ]; then [ -f $H/.omlx074-marker ] || { say "refusing to delete $H (no marker)"; return 1; }; rm -rf $H; fi
  mkdir -p $H/models $H/cache; touch $H/.omlx074-marker; ln -s $m $H/models/$id; print -r -- "$ms" > $H/model_settings.json
  lock; say "## omlx $lab ($($venv/bin/python -c 'import importlib.metadata as m; print(m.version("omlx"))'); $extra; $ms)"; stop_port $P
  nohup $venv/bin/omlx serve --base-path $H --model-dir $H/models --port $P --no-hf-cache --max-concurrent-requests 32 \
    --paged-ssd-cache-dir $H/cache --paged-ssd-cache-max-size 96GB --hot-cache-max-size 48GB ${=extra} \
    > $O/logs/$lab-serve.log 2>&1 < /dev/null &
  local pid=$!
  if up $P $pid; then
    memstart $lab $pid; say "$lab up (pid $pid)"
    t_warm $id $lab; say "$lab warm: $(tail -1 $O/raw/warm.jsonl | cut -c1-250)"; for t in $@; do t_$t $id $lab; done
  else say "$lab failed to start: $(tail -3 $O/logs/$lab-serve.log | tr '\n' ' ')"; fi
  kill $pid 2>/dev/null; for i in $(seq 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
  stop_port $P; unlock
}

MS27='{"version":1,"models":{"q27":{"enable_thinking":false,"mtp_enabled":false,"dflash_enabled":true,"dflash_draft_model":"'$DI'"}}}'
MSFN='{"version":1,"models":{"fn":{"enable_thinking":false,"mtp_enabled":true,"dflash_enabled":false}}}'
MSFN0='{"version":1,"models":{"fn":{"enable_thinking":false,"mtp_enabled":false,"dflash_enabled":false}}}'
MS8='{"version":1,"models":{"q8":{"enable_thinking":false,"mtp_enabled":false,"dflash_enabled":false}}}'
MS8D='{"version":1,"models":{"q8":{"enable_thinking":false,"mtp_enabled":false,"dflash_enabled":true,"dflash_draft_model":"'$DI'"}}}'
V4=~/.venvs/omlx074; V3=~/.venvs/omlx073

arm() {
  case $1 in
  i27pf) inferd_arm i27pf $Q27 "32 2048 0 96 32768" "DRAFT=$DI DRAFT_BITS=4" conc wall speed ;;
  o4_27mtp) QJ=$(ls -d <models>/models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/* | head -1); WALLN=8 TRIALS=1 SIZES=1k,8k,16k omlx_arm o4_27mtp $V4 $QJ q27j '{"version":1,"models":{"q27j":{"enable_thinking":false,"mtp_enabled":true,"dflash_enabled":false}}}' "" speed wall ;;
  o4_27dw) WALLN=1,4,8 TRIALS=1 omlx_arm o4_27dw $V4 $Q27 q27 "$MS27" "" wall ;;
  ifn8k) WALLN=8 TRIALS=1 inferd_arm ifn8k $FN "32 8192 64 32 65536" "MTP=1" b2b wall ;;
  ifn4k) WALLN=8 TRIALS=1 inferd_arm ifn4k $FN "32 4096 64 32 65536" "MTP=1" b2b wall ;;
  i27_8k) WALLN=8 TRIALS=1 inferd_arm i27_8k $Q27 "32 8192 0 96 32768" "DRAFT=$DI DRAFT_BITS=4" wall ;;
  i27)   inferd_arm i27   $Q27 "64 2048 140 96 262144" "DRAFT=$DI DRAFT_BITS=4" conc wall speed ;;
  o4_27) omlx_arm  o4_27 $V4 $Q27 q27 "$MS27" "" conc wall speed ;;
  ifn)   inferd_arm ifn   $FN  "32 2048 64 32 65536" "MTP=1" wall b2b ;;
  o4_fn) omlx_arm  o4_fn $V4 $FN fn "$MSFN" "" wall b2b ;;
  o4_fn_g) omlx_arm o4_fn_g $V4 $FN fn "$MSFN" "--memory-guard-gb 118" b2b ;;
  o3_fn_g) omlx_arm o3_fn_g $V3 $FN fn "$MSFN" "--memory-guard-gb 118" b2b ;;
  o3_fn) WALLN=8 TRIALS=1 omlx_arm o3_fn $V3 $FN fn "$MSFN" "" b2b wall ;;
  i8)    inferd_arm i8   $Q8 "64 2048 96 48 32768" "DRAFT=$DI DRAFT_BITS=4" six ;;
  i8nd)  inferd_arm i8nd $Q8 "64 2048 96 48 32768" "" six ;;
  o4_8)  omlx_arm o4_8 $V4 $Q8 q8 "$MS8" "" six ;;
  o3_8)  omlx_arm o3_8 $V3 $Q8 q8 "$MS8" "" six ;;
  o4_8d) omlx_arm o4_8d $V4 $Q8 q8 "$MS8D" "" six ;;
  o3_27) WALLN=8 TRIALS=1 omlx_arm o3_27 $V3 $Q27 q27 "$MS27" "" conc wall ;;
  *) say "unknown arm $1" ;;
  esac
}
ARMS=($@); [ $#ARMS = 0 ] && ARMS=(i27 o4_27 ifn o4_fn o4_fn_g o3_fn_g i8 i8nd o4_8 o3_8 o4_8d o3_fn o3_27)
for v in Q27 Q8 FN DI; do [ -d ${(P)v} ] || { say "missing $v"; exit 1; }; done
say "##### start: arms $ARMS; inferd $ENG; models $Q27 | $FN | $Q8; draft $DI"; qlog "driver start, arms $ARMS"
for a in $ARMS; do t0=$EPOCHREALTIME; arm $a; say "arm $a done in $(printf %.0f $(( EPOCHREALTIME - t0 ))) s"; qlog "arm $a done"; done
say "##### done"; qlog "driver done"
