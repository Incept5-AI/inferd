#!/bin/zsh
# agents_at_once.sh: N agents on their own contexts send their next turn at the same moment (bench/quick64.py); every
# answer is checked and the run prints PASS/FAIL. Env: AGENTS (8), CTX (4k), ROUNDS (2), POWER_CMD (optional, prints
# watts), INFERD_URL (http://127.0.0.1:8300/v1, as run_all.sh), OUT (default
# ${TMPDIR:-/tmp}/agents_at_once.jsonl, outside the source tree; one JSON line appended per run).
PWD0=$PWD  # a relative OUT is the caller's
cd "$(dirname $0)"
url=${INFERD_URL:-http://127.0.0.1:8300/v1}; url=${url%/v1}
out=${OUT:-${${TMPDIR:-/tmp}%/}/agents_at_once.jsonl}  # outside the source tree unless OUT says otherwise
[[ $out == /* ]] || out=$PWD0/$out
print -r -- "agents_at_once.sh: results are appended to $out"
args=(--url $url --agents ${AGENTS:-8} --ctx ${CTX:-4k} --rounds ${ROUNDS:-2} --out $out)
[[ -n ${POWER_CMD:-} ]] && args+=(--power-cmd "$POWER_CMD")
exec ${PY:-python3} ../../bench/quick64.py $args
