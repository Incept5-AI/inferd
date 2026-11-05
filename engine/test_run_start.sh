#!/bin/zsh
# run.sh's named start (`run.sh start --model ...`): option parsing, the RAM-based sizing (INFERD_RAM_GB), --dry-run, and
# where a named model is found (INFERD_MODELS, the Hugging Face cache, LM Studio: fake trees under HOME / HF_HUB_CACHE /
# LMSTUDIO_MODELS), install.sh's use of a copy it finds (a stub `hf` records any download), and
# a check that the positional form, stop and status behave exactly as in the run.sh before the named form
# (BASE, default 19805b9: main just before the named start, after status learnt the start's log dir; read with git show). It runs copies of run.sh against a fake build/inferd and a fake front end
# in a temporary directory: no model, no GPU, no network. Ports 18397 and 18398 must be free. Cleans up only what it
# started. Every case that could reach a real start names one of those ports: never the default 8300, where stop_all
# would find (and stop) a real server's front end by its command line.
#   ./test_run_start.sh
emulate -L zsh
setopt no_unset
cd "${0:A:h}" || exit 1
BASE=${BASE:-19805b9}
P1=18398 P2=18397
for p in $P1 $P2; do lsof -nP -t -iTCP:$p -sTCP:LISTEN >/dev/null && { print "port $p is in use"; exit 1; }; done
unset MODEL DRAFT PY SPILL_DIR SPILL_GIB MTP INFERD_RAM_GB INFERD_MACOS VISION INFERD_EQUIVALENT 2>/dev/null
unset HF_HUB_CACHE HF_HOME XDG_CACHE_HOME LMSTUDIO_MODELS 2>/dev/null  # the fake HOME below holds the default places
T=$(mktemp -d /tmp/inferd-st.XXXXXX) || exit 1
T=${T:A}  # /private/tmp: run.sh resolves its own directory
E=$T/new O=$T/old M=$T/models REC=$T/rec
export INFERD_MODELS=$M INFERD_VENV=$T/venv INFERD_RUN_DIR=$T/run INFERD_LOGS=$T/logs STOP_WAIT=1 REC
pass=0 fail=0 mine=()
check() {  # description, then a command that must succeed
  local what=$1; shift
  if "$@"; then pass=$((pass + 1)); print "ok   $what"; else fail=$((fail + 1)); print "FAIL $what"; print -r -- "     output: $out"; fi
}
has() { [[ $out == *"$1"* ]] }
line() { print -r -- ${${(M)${(f)out}:#$1*}#$1} }  # the rest of the dry-run line that starts with $1
alive() { kill -0 $1 2>/dev/null }
gone() { local i; for i in {1..40}; do alive $1 || return 0; sleep 0.1; done; return 1 }
cleanup() {
  local p
  for f in $T/run/*.pid(N); do p=$(<$f); [[ $p == <-> ]] && kill -KILL $p 2>/dev/null; done
  for p in $mine; do kill -KILL $p 2>/dev/null; done
  rm -rf -- $T
}
trap cleanup EXIT

# Two engine directories: this run.sh and the one at BASE, each with the same fake build
mkdir -p $E/build $E/tools $O/build $M/{Qwen3.8-27B-4bit,Qwen3.8-27B-8bit,Qwen3.8-27B-DFlash2,Qwen3.8-Flash-Next-oQ4e-mtp} \
  "$T/my ckpt" $T/mydraft $T/venv/bin $REC
cp run.sh install.sh requirements.txt $E/ && cp tools/models.zsh $E/tools/ || exit 1
git show "${BASE}:engine/run.sh" > $O/run.sh 2>/dev/null || { print "cannot read run.sh at $BASE (git show)"; exit 1; }
chmod +x $O/run.sh
export HOME=$T/home  # after git show: no real HF cache or LM Studio is ever seen
mkdir -p $HOME
F4="Qwen3_5ForConditionalGeneration 64 5120 4 64" F8="Qwen3_5ForConditionalGeneration 64 5120 8 64"
FD="DFlash2DraftModel 5 5120 - -" FN="Qwen4ExpForConditionalGeneration 48 2560 4 64"
mkckpt() {  # dir "arch layers hidden bits group" [blobs dir: files become symlinks into it, as in the HF cache]
  local d=$1 f=(${=2}) b=${3:-} q= n
  mkdir -p $d
  [[ ${f[4]} != - ]] && q=', "quantization": {"group_size": '${f[5]}', "bits": '${f[4]}', "mode": "affine"}'
  print -r -- '{"architectures": ["'${f[1]}'"], "text_config": {"num_hidden_layers": '${f[2]}', "hidden_size": '${f[3]}', "rope_scaling": null}'$q'}' > $d/config.json
  print -r -- '{"metadata": {}, "weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors", "c.weight": "model-00001-of-00002.safetensors"}}' > $d/model.safetensors.index.json
  print w > $d/model-00001-of-00002.safetensors; print w > $d/model-00002-of-00002.safetensors
  if [[ -n $b ]]; then  # move every file into blobs/ and link it back, relative, as huggingface_hub does
    mkdir -p $b
    for n in $d/*(.); do mv $n $b/${d:t}-${n:t}; ln -s ../../blobs/${d:t}-${n:t} $n; done
  fi
}
mkckpt $M/Qwen3.8-27B-4bit "$F4"; mkckpt $M/Qwen3.8-27B-8bit "$F8"; mkckpt $M/Qwen3.8-27B-DFlash2 "$FD"
mkckpt $M/Qwen3.8-Flash-Next-oQ4e-mtp "$FN"
for d in "$T/my ckpt" $T/mydraft; do print '{}' > "$d/config.json"; done
for d in $E $O; do
  print 'metallib' > $d/build/core.metallib
  # fake engine: records its arguments and environment, then listens on the socket and prints what run.sh waits for
  cat > $d/build/inferd <<'EOF'
#!/bin/zsh
print -r -- "argv $*" > $REC/inferd-${2:t}.txt
print -r -- "env DRAFT=${DRAFT-unset} MTP=${MTP-unset} SPILL_DIR=${SPILL_DIR-unset} SPILL_GIB=${SPILL_GIB-unset}" >> $REC/inferd-${2:t}.txt
exec python3 -c 'import socket, sys, time
s = socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); s.listen(1); print("listening on", sys.argv[1], flush=True)
time.sleep(600)' "$2"
EOF
  chmod +x $d/build/inferd
done
# fake front-end python: passes run.sh's package check, records its arguments, serves /health on --port
cat > $T/venv/bin/python <<'EOF'
#!/bin/zsh
[[ $1 == -c ]] && exit 0
print -r -- "$*" > $REC/serve-${@[-1]}.txt
exec python3 -c 'import sys, http.server
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s): s.send_response(200); s.end_headers(); s.wfile.write(b"ok")
    def log_message(s, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()' "$@"
EOF
chmod +x $T/venv/bin/python
R=$E/run.sh SOCK1=$T/run/inferd-$P1.sock

# --- the sizing from memory (docs/MACHINES.md), each tier and its edges
for ram want spill in 32 "16 1024 0 8 32768" 64 47 "16 1024 0 8 32768" 64 48 "16 1024 0 30 147456" 96 \
                      64 "16 1024 0 30 147456" 96 127 "16 1024 0 30 147456" 96 128 "16 1024 24 30 147456" 96 \
                      255 "16 1024 24 30 147456" 96 256 "64 2048 140 192 262144" off 512 "64 2048 140 192 262144" off; do
  out=$(INFERD_RAM_GB=$ram $R start --port $P1 --model 27b --dry-run 2>&1); rc=$?
  check "$ram GB: dry run exits 0" test $rc -eq 0
  check "$ram GB: engine gets $want" eval '[[ $(line "engine    ") == *" ./build/inferd $M/Qwen3.8-27B-4bit $SOCK1 $want" ]]'
  if [[ $spill == off ]]; then check "$ram GB: SSD tier off" eval '[[ $(line "SSD tier  ") == off && $(line "engine    ") != *SPILL_DIR* ]]'
  else check "$ram GB: SSD tier $spill GiB in a per-port directory" eval \
    '[[ $(line "engine    ") == *" SPILL_DIR=$HOME/Library/Caches/inferd/spill-$P1 SPILL_GIB=$spill "* ]]'
  fi
done
out=$(INFERD_RAM_GB=32 $R start --port $P1 --model 27b --dry-run 2>&1)
check "27b resolves the drafter beside it" eval '[[ $(line "drafter   ") == $M/Qwen3.8-27B-DFlash2 && $(line "engine    ") == *" DRAFT=$M/Qwen3.8-27B-DFlash2 "* ]]'
check "PY defaults to the install.sh venv" eval '[[ $(line "front end ") == "$T/venv/bin/python -I frontend/serve.py --model $M/Qwen3.8-27B-4bit --socket $SOCK1 --host 127.0.0.1 --port $P1" ]]'
check "a clean dry run warns about nothing" eval '[[ $out != *warning* ]]'
check "dry run created no log or run directory" eval '[[ ! -e $T/logs && ! -e $T/run ]]'

# --- options override the sizing
out=$(INFERD_RAM_GB=256 $R start --model 27b --port $P2 --agents 4 --context 65536 --kv 3 --snapshots 6 --dry-run 2>&1)
check "--port/--agents/--context/--kv/--snapshots" eval '[[ $(line "engine    ") == *" $T/run/inferd-$P2.sock 4 2048 3 6 65536" && $(line "front end ") == *"--port $P2" ]]'
out=$(INFERD_RAM_GB=256 $R start --model=27b --port=$P2 --kv=7.5 --dry-run 2>&1)
check "--opt=value forms" eval '[[ $(line "engine    ") == *" $T/run/inferd-$P2.sock 64 2048 7.5 192 262144" ]]'
out=$(INFERD_RAM_GB=32 $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "--draft none: no speculation" eval '[[ $(line "drafter   ") == none && $(line "engine    ") == *" DRAFT='"''"' "* ]]'
out=$(INFERD_RAM_GB=32 $R start --port $P1 --model 27b --draft $T/mydraft --dry-run 2>&1)
check "--draft DIR" eval '[[ $(line "engine    ") == *" DRAFT=$T/mydraft "* ]]'
out=$(INFERD_RAM_GB=32 DRAFT=$T/mydraft $R start --port $P1 --model 27b --dry-run 2>&1)
check "DRAFT from the environment is kept" eval '[[ $(line "engine    ") == *" DRAFT=$T/mydraft "* ]]'
out=$(INFERD_RAM_GB=32 $R start --port $P1 --model 27b --spill off --dry-run 2>&1)
check "--spill off" eval '[[ $(line "SSD tier  ") == off ]]'
out=$(INFERD_RAM_GB=32 $R start --port $P1 --model 27b --spill $T/sp --dry-run 2>&1)
check "--spill DIR keeps the tier's size" eval '[[ $(line "engine    ") == *" SPILL_DIR=$T/sp SPILL_GIB=64 "* ]]'
out=$(INFERD_RAM_GB=256 SPILL_DIR=$T/sp SPILL_GIB=12 $R start --port $P1 --model 27b --dry-run 2>&1)
check "SPILL_DIR/SPILL_GIB from the environment are kept" eval '[[ $(line "engine    ") == *" SPILL_DIR=$T/sp SPILL_GIB=12 "* ]]'
out=$(INFERD_RAM_GB=32 PY=/usr/bin/true $R start --port $P1 --model 27b --dry-run 2>&1)
check "PY from the environment wins over the venv" eval '[[ $(line "front end ") == "/usr/bin/true -I "* ]]'
out=$(INFERD_RAM_GB=32 INFERD_VENV=$T/none $R start --port $P1 --model 27b --dry-run 2>&1)
check "no venv: python3, as before" eval '[[ $(line "front end ") == "python3 -I "* ]]'

# --- other models
out=$(INFERD_RAM_GB=256 $R start --port $P1 --model 27b-8bit --dry-run 2>&1)
check "27b-8bit: its directory and the same drafter" eval '[[ $(line "engine    ") == *" DRAFT=$M/Qwen3.8-27B-DFlash2 "*" $M/Qwen3.8-27B-8bit "* ]]'
out=$(INFERD_RAM_GB=32 $R start --port $P2 --model 27b-8bit 2>&1); rc=$?
check "27b-8bit refuses a 32 GB Mac" eval '(( rc != 0 )) && has "needs a Mac with at least 64 GB"'
out=$(INFERD_RAM_GB=256 $R start --port $P1 --model flash-next --dry-run 2>&1)
check "flash-next: MTP, no drafter, the Studio setting" eval '[[ $(line "engine    ") == *" DRAFT='"''"' MTP=1 ./build/inferd $M/Qwen3.8-Flash-Next-oQ4e-mtp $SOCK1 64 2048 64 128 65536" ]]'
out=$(INFERD_RAM_GB=256 $R start --port $P1 --model flash-next --draft none --dry-run 2>&1)
check "flash-next --draft none: MTP=0" eval '[[ $(line "engine    ") == *" MTP=0 "* ]]'
out=$(INFERD_RAM_GB=128 $R start --port $P2 --model flash-next 2>&1); rc=$?
check "flash-next refuses 128 GB" eval '(( rc != 0 )) && has "needs a Mac with at least 256 GB"'
out=$(INFERD_RAM_GB=128 $R start --port $P1 --model flash-next --dry-run 2>&1); rc=$?
check "... and the dry run warns instead" eval '(( rc == 0 )) && has "warning (dry run): flash-next needs"'
out=$(INFERD_RAM_GB=64 $R start --port $P1 --model "$T/my ckpt" --dry-run 2>&1)
check "a directory (with a space): no drafter, sized from memory" eval '[[ $(line "model     ") == "$T/my ckpt" && $(line "drafter   ") == none && $(line "engine    ") == *" 16 1024 0 30 147456" ]]'
out=$(cd $T && INFERD_RAM_GB=64 $R start --port $P1 --model "./my ckpt" --draft mydraft --spill sp --dry-run 2>&1)
check "relative paths are the caller's" eval '[[ $(line "model     ") == "$T/my ckpt" && $(line "engine    ") == *" DRAFT=$T/mydraft SPILL_DIR=$T/sp "* ]]'
out=$(INFERD_RAM_GB=64 MODEL=$T/mydraft $R start --port $P1 --dry-run 2>&1)
check "MODEL from the environment when --model is absent" eval '[[ $(line "model     ") == $T/mydraft ]]'

# --- errors (a start that cannot work stops before anything runs)
# (the error checks pin the memory: a small CI runner, 14 GB, would otherwise stop at "27b needs 32 GB" first)
out=$(INFERD_RAM_GB=64 $R start --port $P2 2>&1); rc=$?;                              check "start without a model" eval '(( rc != 0 )) && has "start needs --model"'
out=$(INFERD_RAM_GB=64 $R start --port $P2 --model 27b --bogus 2>&1); rc=$?;          check "unknown option" eval '(( rc != 0 )) && has "unknown option '"'--bogus'"'"'
out=$(INFERD_RAM_GB=64 $R start --port $P2 --model 2>&1); rc=$?;                      check "missing value" eval '(( rc != 0 )) && has "--model needs a value"'
out=$(INFERD_RAM_GB=64 $R start --port $P2 --model 70b 2>&1); rc=$?;                  check "unknown model name" eval '(( rc != 0 )) && has "unknown model '"'70b'"'"'
out=$(INFERD_RAM_GB=64 $R start --port $P2 --model 27b --agents x 2>&1); rc=$?;       check "bad --agents" eval '(( rc != 0 )) && has "max_running must be an integer"'
out=$(INFERD_RAM_GB=64 $R start --model 27b --port 0 2>&1); rc=$?;         check "bad --port" eval '(( rc != 0 )) && has "port must be an integer"'
out=$(INFERD_RAM_GB=x $R start --port $P2 --model 27b 2>&1); rc=$?;  check "bad INFERD_RAM_GB" eval '(( rc != 0 )) && has "INFERD_RAM_GB must be"'
out=$(INFERD_RAM_GB=64 INFERD_MODELS=$T/empty $R start --port $P2 --model 27b 2>&1); rc=$?
check "a missing checkpoint names install.sh" eval '(( rc != 0 )) && has "run ./install.sh --model 27b"'
print 'macos_target 26' > $E/build/BUILD-INFO
out=$(INFERD_RAM_GB=64 INFERD_MACOS=27 $R start --port $P2 --model 27b 2>&1); rc=$?
check "a release build for another macOS refuses to start" eval '(( rc != 0 )) && has "this build is for macOS 26 and this Mac runs macOS 27"'
print 'macos_target 27' > $E/build/BUILD-INFO
out=$(INFERD_MACOS=27 INFERD_RAM_GB=32 $R start --port $P1 --model 27b --dry-run 2>&1)
check "... and the matching one passes" eval '[[ $out != *warning* ]]'
rm -f $E/build/BUILD-INFO
check "no errors created a log or run directory" eval '[[ ! -e $T/logs && ! -e $T/run ]]'

# --- where a named model is found (tools/models.zsh): INFERD_MODELS, the Hugging Face cache, LM Studio
H=$T/hub L=$T/lms X=$T/empty
hfsnap() {  # repo rev facts: a snapshot of REPO in the fake cache H, files linked into blobs/
  local r=$H/models--${1//\//--}
  mkdir -p $r/snapshots $r/refs
  mkckpt $r/snapshots/$2 "$3" $r/blobs
}
HM=$H/models--mlx-community--Qwen3.8-27B-4bit
hfsnap mlx-community/Qwen3.8-27B-4bit aaa "$F4"; touch -t 202601010000 $HM/snapshots/aaa
hfsnap mlx-community/Qwen3.8-27B-4bit bbb "$F4"; touch -t 202602010000 $HM/snapshots/bbb
hfsnap mlx-community/Qwen3.8-27B-4bit ccc "$F4"  # the newest, but one shard's blob is empty
: > $HM/blobs/ccc-model-00002-of-00002.safetensors
print aaa > $HM/refs/main
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$H $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "HF cache: the exact repo, the snapshot refs/main points to" eval '[[ $(line "model     ") == $HM/snapshots/aaa ]] && has "(mlx-community/Qwen3.8-27B-4bit, in the Hugging Face cache (the snapshot refs/main points to))"'
check "... the engine and front end get the snapshot (links into blobs/ followed)" eval '[[ $(line "engine    ") == *" ./build/inferd $HM/snapshots/aaa "* && $(line "front end ") == *"--model $HM/snapshots/aaa "* ]]'
print ccc > $HM/refs/main
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$H $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "HF cache: refs/main incomplete (an empty shard) -> the newest complete snapshot" eval '[[ $(line "model     ") == $HM/snapshots/bbb ]] && has "(snapshot bbb)"'
check "... and says why it skipped refs/main's" eval 'has "note: skipped $HM/snapshots/ccc (mlx-community/Qwen3.8-27B-4bit): incomplete: model-00002-of-00002.safetensors is missing or empty"'
rm $HM/blobs/bbb-model-00001-of-00002.safetensors  # now a dangling link
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$H $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "HF cache: a dangling shard link is incomplete too -> the older complete one" eval '[[ $(line "model     ") == $HM/snapshots/aaa ]] && has "skipped $HM/snapshots/bbb"'
out=$(INFERD_RAM_GB=32 HF_HUB_CACHE=$H $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "precedence: INFERD_MODELS before the HF cache" eval '[[ $(line "model     ") == $M/Qwen3.8-27B-4bit ]] && has "(mlx-community/Qwen3.8-27B-4bit, in INFERD_MODELS)"'
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HOME=$T/hfhome $R start --port $P1 --model 27b --draft none --dry-run 2>&1); rc=$?
check "HF_HOME/hub replaces the default cache (empty here: not found)" eval '(( rc == 0 )) && has "warning (dry run): no complete copy of mlx-community/Qwen3.8-27B-4bit"'
mkdir -p $T/hfhome; ln -s $H $T/hfhome/hub
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HOME=$T/hfhome $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "... and found there" eval '[[ $(line "model     ") == $T/hfhome/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/aaa ]]'

# LM Studio: LMSTUDIO_MODELS/<org>/<name> (default ~/.lmstudio/models), and the older ~/.cache/lm-studio/models
mkckpt $L/mlx-community/Qwen3.8-27B-8bit "$F8"
out=$(INFERD_RAM_GB=64 INFERD_MODELS=$X HF_HUB_CACHE=$X LMSTUDIO_MODELS=$L $R start --port $P1 --model 27b-8bit --draft none --dry-run 2>&1)
check "LM Studio: LMSTUDIO_MODELS/<org>/<name>" eval '[[ $(line "model     ") == $L/mlx-community/Qwen3.8-27B-8bit ]] && has "in LM Studio"'
mkckpt $HOME/.lmstudio/models/mlx-community/Qwen3.8-27B-8bit "$F8"
out=$(INFERD_RAM_GB=64 INFERD_MODELS=$X HF_HUB_CACHE=$X $R start --port $P1 --model 27b-8bit --draft none --dry-run 2>&1)
check "LM Studio: ~/.lmstudio/models by default" eval '[[ $(line "model     ") == $HOME/.lmstudio/models/mlx-community/Qwen3.8-27B-8bit ]]'
mkckpt $HOME/.cache/lm-studio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp "$FN"
out=$(INFERD_RAM_GB=256 INFERD_MODELS=$X HF_HUB_CACHE=$X $R start --port $P1 --model flash-next --dry-run 2>&1)
check "LM Studio: the older ~/.cache/lm-studio/models layout" eval '[[ $(line "model     ") == $HOME/.cache/lm-studio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp && $(line "engine    ") == *" MTP=1 "* ]]'

# the same-weights rule: exact repo ids first, then (by default) the listed equivalent builds when config.json matches;
# --exact takes only the exact repo ids
mkckpt $L/Qwen/Qwen3.8-27B-MLX-4bit "$F4"
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$X LMSTUDIO_MODELS=$L $R start --port $P2 --model 27b --draft none --exact 2>&1); rc=$?
check "--exact: an equivalent build is not used, but named" eval '(( rc != 0 )) && has "note: found Qwen/Qwen3.8-27B-MLX-4bit (an equivalent build of mlx-community/Qwen3.8-27B-4bit) is $L/Qwen/Qwen3.8-27B-MLX-4bit: --exact refuses it" && has "run ./install.sh --model 27b"'
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$X LMSTUDIO_MODELS=$L $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "default: the equivalent build, with a note" eval '[[ $(line "model     ") == $L/Qwen/Qwen3.8-27B-MLX-4bit ]] && has "note: an equivalent build of mlx-community/Qwen3.8-27B-4bit, not the exact repo our benchmarks use (--exact refuses it): Qwen3_5ForConditionalGeneration, 64 layers x 5120, 4-bit, group size 64"'
out=$(INFERD_RAM_GB=32 INFERD_EQUIVALENT=1 INFERD_MODELS=$X HF_HUB_CACHE=$H LMSTUDIO_MODELS=$L $R start --port $P1 --model 27b --draft none --dry-run 2>&1)
check "INFERD_EQUIVALENT=1 still prefers the exact repo" eval '[[ $(line "model     ") == $HM/snapshots/aaa ]]'
rm -rf $L/Qwen/Qwen3.8-27B-MLX-4bit; mkckpt $L/lmstudio-community/Qwen3.8-27B-MLX-4bit "$F8"  # mislabelled: 8-bit
mkckpt $T/lms8/mlx-community/Qwen3.8-27B-4bit "$F8"
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$X LMSTUDIO_MODELS=$L $R start --port $P2 --model 27b --draft none --equivalent 2>&1); rc=$?
check "an 8-bit build is refused for 27b" eval '(( rc != 0 )) && has "skipped $L/lmstudio-community/Qwen3.8-27B-MLX-4bit (lmstudio-community/Qwen3.8-27B-MLX-4bit): config.json says Qwen3_5ForConditionalGeneration, 64 layers x 5120, 8-bit, group size 64, not Qwen3_5ForConditionalGeneration, 64 layers x 5120, 4-bit, group size 64" && has "no complete copy of mlx-community/Qwen3.8-27B-4bit"'
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$X LMSTUDIO_MODELS=$T/lms8 $R start --port $P2 --model 27b --draft none 2>&1); rc=$?
check "... even under the exact repo id" eval '(( rc != 0 )) && has "skipped $T/lms8/mlx-community/Qwen3.8-27B-4bit (mlx-community/Qwen3.8-27B-4bit): config.json says Qwen3_5ForConditionalGeneration, 64 layers x 5120, 8-bit"'

# the drafter: in the default HF cache (~/.cache/huggingface/hub), and its byte-identical z-lab twin
D=$HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2
mkdir -p $D/refs; mkckpt $D/snapshots/d1 "$FD" $D/blobs; print d1 > $D/refs/main
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$H $R start --port $P1 --model 27b --dry-run 2>&1)
check "HF_HUB_CACHE replaces the default cache" eval '[[ $(line "drafter   ") != $D/snapshots/d1 ]]'
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X $R start --port $P1 --model 27b --dry-run 2>&1)
check "the drafter in the default HF cache: found, and the engine gets it" eval '[[ $(line "drafter   ") == $D/snapshots/d1 && $(line "engine    ") == *" DRAFT=$D/snapshots/d1 "* ]] && has "(incoai/Qwen3.8-27B-DFlash2, in the Hugging Face cache (the snapshot refs/main points to))"'
rm -rf $D; Z=$H/models--z-lab--Qwen3.8-27B-DFlash2; mkdir -p $Z/refs; mkckpt $Z/snapshots/z1 "$FD" $Z/blobs; print z1 > $Z/refs/main
out=$(INFERD_RAM_GB=32 HF_HUB_CACHE=$H INFERD_MODELS=$X $R start --port $P1 --model 27b --equivalent --dry-run 2>&1)
check "--equivalent (the default): the z-lab drafter" eval '[[ $(line "drafter   ") == $Z/snapshots/z1 ]] && has "z-lab/Qwen3.8-27B-DFlash2, in the Hugging Face cache"'
out=$(INFERD_RAM_GB=32 INFERD_MODELS=$X HF_HUB_CACHE=$H $R start --port $P2 --model 27b --exact 2>&1); rc=$?
check "... and with --exact, refused with the hint" eval '(( rc != 0 )) && has "found z-lab/Qwen3.8-27B-DFlash2" && has "no complete copy of the drafter incoai/Qwen3.8-27B-DFlash2"'
out=$(INFERD_MODELS=$X HF_HUB_CACHE=$H LMSTUDIO_MODELS=$L $R models --equivalent 2>&1); rc=$?
check "run.sh models lists each name with its path" eval '(( rc == 0 )) && has "27b: $HM/snapshots/aaa" && has "27b drafter: $Z/snapshots/z1" && has "flash-next: $HOME/.cache/lm-studio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"'
check "no model search created a log or run directory" eval '[[ ! -e $T/logs && ! -e $T/run ]]'

# --- install.sh: uses a copy it finds, downloads (a stub hf on PATH) only what it does not find
mkdir -p $T/bin $T/ivenv/bin
print -r -- '#!/bin/zsh
[[ $* == *version_info* ]] && { print 3.12; exit 0 }
exit 0' > $T/ivenv/bin/python
shasum -a 256 < $E/requirements.txt | cut -d' ' -f1 > $T/ivenv/.inferd-requirements.sha256
typeset -f mkckpt > $T/mkckpt.zsh
print -r -- '#!/bin/zsh
print -r -- "$*" >> '$REC'/hf-calls.txt
source '$T'/mkckpt.zsh
case $2 in *DFlash2) f="'$FD'" ;; *8bit) f="'$F8'" ;; *) f="'$F4'" ;; esac
mkckpt ${@[-1]} "$f"' > $T/bin/hf
chmod +x $T/ivenv/bin/python $T/bin/hf
I=$E/install.sh
inst() { env PATH=$T/bin:$PATH INFERD_VENV=$T/ivenv INFERD_RAM_GB=64 INFERD_MODELS=$1 HF_HUB_CACHE=$H LMSTUDIO_MODELS=$L "${@:2}" }
rm -f $REC/hf-calls.txt
out=$(inst $X $I --model 27b 2>&1); rc=$?
check "install.sh: both found, no download" eval '(( rc == 0 )) && [[ ! -e $REC/hf-calls.txt ]] && has "mlx-community/Qwen3.8-27B-4bit is already on this Mac: $HM/snapshots/aaa" && has "incoai/Qwen3.8-27B-DFlash2 is already on this Mac: $Z/snapshots/z1"'
check "... and its start line needs no flag" eval 'has "./run.sh start --model 27b" && ! has "./run.sh start --model 27b --exact"'
out=$(inst $T/dl $I --model 27b --exact 2>&1); rc=$?
check "install.sh --exact: only the missing drafter is downloaded, into the models directory" eval '(( rc == 0 )) && [[ $(<$REC/hf-calls.txt) == "download incoai/Qwen3.8-27B-DFlash2 --local-dir $T/dl/Qwen3.8-27B-DFlash2" ]] && has "found z-lab/Qwen3.8-27B-DFlash2"'
rm -f $REC/hf-calls.txt
out=$(inst $T/dl $I --model 27b --exact 2>&1); rc=$?
check "install.sh again: nothing to download" eval '(( rc == 0 )) && [[ ! -e $REC/hf-calls.txt ]] && has "is already on this Mac: $T/dl/Qwen3.8-27B-DFlash2 (incoai/Qwen3.8-27B-DFlash2, in INFERD_MODELS)"'
out=$(inst $T/dl $I --model 27b --download 2>&1); rc=$?
want="download mlx-community/Qwen3.8-27B-4bit --local-dir $T/dl/Qwen3.8-27B-4bit
download incoai/Qwen3.8-27B-DFlash2 --local-dir $T/dl/Qwen3.8-27B-DFlash2"
check "install.sh --download: fetches both into the models directory" eval '(( rc == 0 )) && [[ $(<$REC/hf-calls.txt) == $want ]]'
out=$(inst $T/dl $I --list 2>&1); rc=$?
check "install.sh --list" eval '(( rc == 0 )) && has "27b: $T/dl/Qwen3.8-27B-4bit" && has "27b drafter: $T/dl/Qwen3.8-27B-DFlash2" && has "27b-8bit: $L/mlx-community/Qwen3.8-27B-8bit"'
out=$(inst $T/dl $I --list --model 27b 2>&1); rc=$?
check "install.sh --list with --model is refused" eval '(( rc != 0 )) && has "drop --model"'

# --- a real start through the fakes: the named start runs what its dry run printed
out=$(INFERD_RAM_GB=32 $R start --model 27b --port $P1 --dry-run 2>&1); dry=$(line "engine    ")
out=$(INFERD_RAM_GB=32 $R start --model 27b --port $P1 --spill off 2>&1); rc=$?
check "named start comes up" eval '(( rc == 0 )) && has "up on 127.0.0.1:$P1"'
check "named start reports its sizing" eval 'has "sized for a 32 GB Mac: agents 16, KV auto, snapshots 8, max context 32768"'
rec=$(<$REC/inferd-inferd-$P1.sock.txt 2>/dev/null)
check "engine got the dry run's arguments" eval '[[ ${${(f)rec}[1]} == "argv $M/Qwen3.8-27B-4bit $SOCK1 16 1024 0 8 32768" && $dry == *" ./build/inferd $M/Qwen3.8-27B-4bit $SOCK1 16 1024 0 8 32768" ]]'
check "engine got the drafter and no SSD tier" eval '[[ ${${(f)rec}[2]} == "env DRAFT=$M/Qwen3.8-27B-DFlash2 MTP=unset SPILL_DIR= SPILL_GIB=unset" ]]'
check "front end got the model, socket and port" eval '[[ $(<$REC/serve-$P1.txt) == "-I frontend/serve.py --model $M/Qwen3.8-27B-4bit --socket $SOCK1 --host 127.0.0.1 --port $P1" ]]'
out=$($R status $P1 2>&1); rc=$?;  check "status after a named start exits 0" test $rc -eq 0
fe=$(<$T/run/serve-$P1.pid) eng=$(<$T/run/inferd-$P1.pid)
out=$($R stop $P1 2>&1); rc=$?
check "stop after a named start" eval '(( rc == 0 )) && has "stopped 2 process"'
check "both processes gone" eval 'gone $fe && gone $eng'

# --- the positional form, stop and status: the same as run.sh at BASE
cmp_run() {  # description, env assignments..., --, args...: both run.shs, same exit code and output
  local what=$1; shift
  local envs=() a o n ro rn
  while [[ $1 != -- ]]; do envs+=($1); shift; done; shift
  o=$(env $envs $O/run.sh "$@" 2>&1); ro=$?
  n=$(env $envs $E/run.sh "$@" 2>&1); rn=$?
  out="old($ro): $o | new($rn): $n"
  check "unchanged: $what" eval '(( ro == rn )) && [[ $o == $n ]]'
}
cmp_run "stop with nothing running" -- stop $P1
cmp_run "status with nothing running" -- status $P1
cmp_run "stop rejects a bad port" -- stop abc
cmp_run "stop rejects extra arguments" -- stop 1 2
cmp_run "status rejects extra arguments" -- status 1 2
cmp_run "start rejects port 0" -- 0
cmp_run "start rejects max_running abc" -- $P1 abc
cmp_run "start rejects kv_gib x" -- $P1 16 1024 x
cmp_run "start rejects max_ctx 1" -- $P1 16 1024 0 8 1
cmp_run "start requires MODEL" MODEL= -- $P1
cmp_run "start rejects a model without config.json" MODEL=$T/run -- $P1
cmp_run "start rejects a bad DRAFT" MODEL=$M/Qwen3.8-27B-4bit DRAFT=$T -- $P1 16 1024 0 8 32768
cmp_run "PY without the packages" MODEL=$M/Qwen3.8-27B-4bit PY=/usr/bin/false -- $P1
for d in old new; do
  [[ $d == old ]] && r=$O/run.sh || r=$E/run.sh
  out=$(MODEL=$M/Qwen3.8-27B-4bit DRAFT=$M/Qwen3.8-27B-DFlash2 PY=$T/venv/bin/python $r $P1 16 1024 0 8 32768 2>&1); rc=$?
  check "$d positional start comes up" eval '(( rc == 0 )) && has "up on 127.0.0.1:$P1"'
  cp $REC/inferd-inferd-$P1.sock.txt $REC/pos-$d-inferd.txt; cp $REC/serve-$P1.txt $REC/pos-$d-serve.txt
  st=$($r status $P1 2>&1); print -r -- "${st//<->/N}" > $REC/pos-$d-status.txt  # pids differ
  out=$($r stop $P1 2>&1); rc=$?
  print -r -- "$rc $out" > $REC/pos-$d-stop.txt
done
check "unchanged: positional start's engine arguments and environment" cmp -s $REC/pos-old-inferd.txt $REC/pos-new-inferd.txt
check "unchanged: positional start's front-end arguments" cmp -s $REC/pos-old-serve.txt $REC/pos-new-serve.txt
check "unchanged: status while running (pids masked)" cmp -s $REC/pos-old-status.txt $REC/pos-new-status.txt
check "unchanged: stop while running" cmp -s $REC/pos-old-stop.txt $REC/pos-new-stop.txt
out=$(<$REC/pos-new-inferd.txt)
check "positional start passes the six numbers through" eval 'has "argv $M/Qwen3.8-27B-4bit $SOCK1 16 1024 0 8 32768"'

print "\n$pass passed, $fail failed"
(( fail == 0 ))
