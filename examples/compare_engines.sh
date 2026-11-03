#!/bin/zsh
# compare_engines.sh (2 Nov 2026): runs the three head-to-head examples (agent-swarm, agent-followups, users-at-once)
# against inferd and other Mac engines, ONE server at a time, on the same Qwen3.8-27B 4-bit weights (mlx-community).
# Run it on the machine under test (clients are cheap); the examples sit next to it in the same layout as here.
#   ENGINES="inferd splash omlx mlxlm tf tfpar" ./compare_engines.sh m5pro
# Rows -> results/h2h-<MACHINE>/<example>.jsonl, console output -> results/h2h-<MACHINE>/<engine>-<example>.txt.
# Paths below are the Mac mini M5 Pro's; override MODEL, DRAFT, INFERD, PY, TF, OMLX_HOME as needed.
# Lab orchestration: it stops whatever listens on its ports (8300, 8090-8092, 8100) and recreates OMLX_HOME, so it asks
# for consent first (COMPARE_OK=1 skips the question) and only deletes an OMLX_HOME it created (marker file).
zmodload zsh/datetime
cd "$(dirname $0)"; export PATH=/opt/homebrew/bin:$PATH
if [ "${COMPARE_OK:-}" != 1 ]; then
  echo "compare_engines.sh stops any process listening on ports 8300, 8090-8092 and 8100 and recreates OMLX_HOME."
  read -q "?Continue? [y/N] " || { echo; exit 1; }; echo
fi
M=${1:-m5pro}; R=results/h2h-$M; mkdir -p $R logs
MODEL=${MODEL:-$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/* | head -1)}
DRAFT=${DRAFT:-$(ls -d ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/* | head -1)}
INFERD=${INFERD:-~/inference/engine-m2}; PY=${PY:-~/.venvs/mlxvlm/bin/python}
TF=${TF:-~/exh2h/tf-1.2.1/bin/tensorfold-native}; OMLX_HOME=${OMLX_HOME:-~/exh2h/omlx-home}
ENGINES=(${=ENGINES:-inferd splash omlx mlxlm tf})
typeset -A EXARGS  # per-example arguments (defaults: 8 x 16k shared, 8 x 8k own, users 1,4 x 8k + 1,024)
EXARGS=(agent-swarm "${SWARM_ARGS:-}" agent-followups "${FOLLOW_ARGS:-}" users-at-once "${USERS_ARGS:-}")
say() { echo "$(date +%H:%M:%S) $*" | tee -a $R/progress.log; }

stop_port() {  # stop inferd through its run.sh (pid files, graceful), then whatever else listens on the port
  [ -x $INFERD/run.sh ] && (cd $INFERD && ./run.sh stop $1 > /dev/null 2>&1)
  for p in $(lsof -ti tcp:$1 -sTCP:LISTEN) $(lsof -t /tmp/inferd-$1.sock 2>/dev/null); do kill $p 2>/dev/null; done
  for i in $(seq 60); do [ -z "$(lsof -ti tcp:$1 -sTCP:LISTEN)" ] && break; sleep 1; done
  sleep 10  # let the GPU memory go back
}
wait_up() {  # port pid
  for i in $(seq 900); do curl -sf -m 5 localhost:$1/v1/models > /dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done
  return 1
}

run_examples() {  # engine-label url model
  local url=$2/v1
  # warm-up (untimed): load kernels, page the weights in
  curl -s -m 600 $url/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$3\",\"max_tokens\":16,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" > $R/$1-warmup.json
  for ex in "agent-swarm agent-swarm/agent_swarm.py" "agent-followups agent-followups/agent_followups.py" \
            "users-at-once users-at-once/users_at_once.py"; do set -- $1 $2 $3 ${=ex}
    local t0=$EPOCHREALTIME
    python3 $5 --url $url --model $3 --label $1 --out $R/$4.jsonl ${=EXARGS[$4]} > $R/$1-$4.txt 2>&1
    local rc=$?
    say "$1 $4: $(printf %.0f $(( EPOCHREALTIME - t0 ))) s, exit $rc; $(memory_pressure | tail -1)"
  done
}

for e in $ENGINES; do
  case $e in
  inferd|inferd-s64)  # 16 agents at once, 1,024-token steps, KV auto-sized, 32k max context; SSD tier off.
    # inferd keeps at most `snapshots` cached prefixes (GDN state snapshots, ROADMAP P0-7): 32 is the examples'
    # standard setting; inferd-s64 gives 64, enough for 16 agents with their own contexts.
    stop_port 8300
    snaps=32; [ $e = inferd-s64 ] && snaps=64
    (cd $INFERD && MODEL=$MODEL DRAFT=$DRAFT DRAFT_BITS=4 PY=$PY ./run.sh 8300 16 1024 0 $snaps 32768 > /dev/null 2>&1)
    pid=$(lsof -ti tcp:8300 -sTCP:LISTEN | head -1)  # the front end run.sh started (any socket location)
    wait_up 8300 $pid && { say "$e up: $(grep -E 'memory:|KV blocks' ~/inference/logs/inferd-8300.log | tr '\n' ' ')"; run_examples $e http://127.0.0.1:8300 qwen3.8-27b; } || say "$e failed"
    stop_port 8300 ;;
  splash)  # Splash 1.5.0 as in the overnight fleets: same weights and DFlash2 drafter, effort none, 32k context
    stop_port 8100
    nohup splash serve --model mlx-community/Qwen3.8-27B-4bit --draft-model $DRAFT --offline --port 8100 --host 127.0.0.1 \
      --no-webui --max-context 32K --default-reasoning-effort none --idle-release off > logs/splash.log 2>&1 < /dev/null &
    pid=$!
    if wait_up 8100 $pid; then
      mid=$(curl -sf localhost:8100/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
      say "splash up ($mid)"; run_examples splash http://127.0.0.1:8100 $mid
    else say "splash failed"; fi
    stop_port 8100 ;;
  omlx)  # oMLX 0.7.1 in an isolated home (its SSD cache survives restarts: wiped here, and prompts carry nonces)
    stop_port 8092; if [ -e $OMLX_HOME ]; then [ -f $OMLX_HOME/.compare-engines ] || { say "refusing to delete $OMLX_HOME (not created by this script)"; continue; }; rm -rf $OMLX_HOME; fi; mkdir -p $OMLX_HOME/models; touch $OMLX_HOME/.compare-engines; ln -s $MODEL $OMLX_HOME/models/qwen3.8-27b
    nohup omlx serve --base-path $OMLX_HOME --model-dir $OMLX_HOME/models --port 8092 --no-hf-cache > logs/omlx.log 2>&1 < /dev/null &
    pid=$!
    wait_up 8092 $pid && { say "omlx up"; run_examples omlx http://127.0.0.1:8092 qwen3.8-27b; } || say "omlx failed"
    stop_port 8092 ;;
  mlxlm|mlxlm-pc2)  # mlx_lm.server (continuous batching in recent mlx-lm), defaults apart from thinking off.
    # Its defaults prefill up to 8 prompts at once: at 8 x 16k that ran the 64 GB M5 Pro's GPU out of memory, so
    # mlxlm-pc2 limits prompt processing to 2 at a time.
    stop_port 8090
    extra=(); [ $e = mlxlm-pc2 ] && extra=(--prompt-concurrency 2)
    nohup ${PY:h}/mlx_lm.server --model $MODEL --port 8090 --chat-template-args '{"enable_thinking":false}' $extra > logs/$e.log 2>&1 < /dev/null &
    pid=$!
    wait_up 8090 $pid && { say "$e up"; run_examples $e http://127.0.0.1:8090 $MODEL; } || say "$e failed"
    stop_port 8090 ;;
  tf|tfpar)  # TensorFold 1.2.1, same weights + DFlash2 (z-lab = incoai, byte-identical); tfpar = --parallel 8
    stop_port 8091
    extra=(); [ $e = tfpar ] && extra=(--parallel 8 --context 32768)
    nohup $TF serve $MODEL --drafter $DRAFT --port 8091 --name q27 --no-thinking --no-update-check $extra > logs/$e.log 2>&1 < /dev/null &
    pid=$!
    wait_up 8091 $pid && { say "$e up"; run_examples $e http://127.0.0.1:8091 q27; } || say "$e failed: $(tail -3 logs/$e.log)"
    stop_port 8091 ;;
  esac
done
say "##### done"
