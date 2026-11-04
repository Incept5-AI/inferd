#!/bin/zsh
# qa8_gate.sh MODEL OUTDIR [BASE_ENGINE_DIR]: QA8 logits gates. decode_test check vs tools/ref_logits.py (mlx_lm) at chunks
# 1/24 (4 x 24 tokens) and 256/1536 (2 x 1536 real-text tokens: pass them as TOKS1536=<file>, "2 1536" then 2 lines;
# ref_logits.py make has only short texts), QA8=0 and QA8=1 with PF8=0; with BASE, also QA8=0
# against an unmodified build byte for byte (cmp) at every chunk. PY = a python with mlx_lm.
set -u
M=$1; T=$2; BASE=${3:-}; PY=${PY:-$HOME/.venvs/mlxvlm/bin/python}
mkdir -p $T; cd "$(dirname $0)/.."
[ -s $T/toks24.txt ] || $PY -I tools/ref_logits.py make "$M" $T/toks24.txt 4 24 > $T/make24.txt 2>&1
[ -s $T/toks1536.txt ] || { [ -n "${TOKS1536:-}" ] && cp $TOKS1536 $T/toks1536.txt; } || $PY -I tools/ref_logits.py make "$M" $T/toks1536.txt 2 1536 > $T/make1536.txt 2>&1
for c in ${=CHUNKS:-1 24 256 1536}; do
  tk=$T/toks24.txt; [ $c -ge 256 ] && tk=$T/toks1536.txt
  for q in 0 1; do
    PF8=0 QA8=$q ./build/decode_test check "$M" $tk $T/eng-c$c-q$q.bin $c > $T/run-c$c-q$q.txt 2>&1
    echo "== chunk $c QA8=$q" >> $T/gate.txt
    $PY -I tools/ref_logits.py compare "$M" $tk $T/eng-c$c-q$q.bin >> $T/gate.txt 2>&1
  done
  if [ -n "$BASE" ]; then
    (cd $BASE && PF8=0 ./build/decode_test check "$M" $tk $T/base-c$c.bin $c > $T/run-c$c-base.txt 2>&1)
    cmp -s $T/base-c$c.bin $T/eng-c$c-q0.bin && echo "== chunk $c: QA8=0 byte-identical to base" >> $T/gate.txt || echo "== chunk $c: QA8=0 DIFFERS from base" >> $T/gate.txt
    cmp -s $T/eng-c$c-q0.bin $T/eng-c$c-q1.bin && echo "   (QA8=1 also byte-identical: path not taken at this chunk)" >> $T/gate.txt
  fi
done
echo DONE >> $T/gate.txt
