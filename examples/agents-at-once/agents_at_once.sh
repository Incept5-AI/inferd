#!/bin/zsh
# agents_at_once.sh: N agents on their own contexts send their next turn at the same moment (bench/quick64.py); every
# answer is checked and the run prints PASS/FAIL. Env: AGENTS (8), CTX (4k), ROUNDS (2), POWER_CMD (optional, prints
# watts), INFERD_URL (http://127.0.0.1:8300/v1, as run_all.sh), OUT (agents_at_once.jsonl).
cd "$(dirname $0)"
url=${INFERD_URL:-http://127.0.0.1:8300/v1}; url=${url%/v1}
args=(--url $url --agents ${AGENTS:-8} --ctx ${CTX:-4k} --rounds ${ROUNDS:-2} --out ${OUT:-agents_at_once.jsonl})
[[ -n ${POWER_CMD:-} ]] && args+=(--power-cmd "$POWER_CMD")
exec ${PY:-python3} ../../bench/quick64.py $args
