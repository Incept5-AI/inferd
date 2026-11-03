#!/bin/zsh
# Coding smoke test: the Python and TypeScript half of languages-benchmark v1 (8 challenges x 2 languages = 16
# tests) against an inferd server. The model writes a program, the harness builds and runs it and checks the output;
# one feedback round on failure (1.0 first try, 0.5 after feedback).
#
#   [LANGBENCH=<checkout>] ./coding_smoke.sh [effort=medium] [workers=8]
#
# Needs: python3, git, node + tsx for TypeScript (`brew install node && npm i -g tsx`). The benchmark repo
# (github.com/Incept5-AI/languages-benchmark, public) is used as is; without LANGBENCH it is cloned once into
# ~/.cache/inferd/languages-benchmark.
set -e
LANGBENCH=${LANGBENCH:-~/.cache/inferd/languages-benchmark}
if [ ! -d $LANGBENCH ]; then
  mkdir -p ${LANGBENCH:h} && git clone --depth 1 https://github.com/Incept5-AI/languages-benchmark $LANGBENCH
fi
URL=${INFERD_URL:-http://127.0.0.1:8300/v1}
EFFORT=${1:-medium}
WORKERS=${2:-8}
# The token cap must leave room for the prompt inside the server's max context (run.sh's last argument, 32768 here):
# a request whose prompt + max_tokens is larger is refused ("context too long").
MAX_TOKENS=${MAX_TOKENS:-30000}
OUT=$(mktemp -d)/smoke-$EFFORT.json

MODEL=$(curl -sf $URL/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
cd $LANGBENCH
python3 bench.py doctor | grep '^v1'
if [ $EFFORT = none ]; then THINK=(--no-thinking); else THINK=(--effort $EFFORT); fi
START=$(date +%s)
python3 bench.py run --suite v1 --langs python,typescript --base-url $URL --model $MODEL $THINK \
  --max-tokens $MAX_TOKENS --workers $WORKERS --engine inferd --hardware "$(sysctl -n machdep.cpu.brand_string)" \
  --out $OUT
echo "wall time: $(( $(date +%s) - START ))s"
python3 bench.py report $OUT
echo "result file: $OUT"
