#!/bin/zsh
# Set up inferd's front end, and optionally download a model. Safe to run again: it reuses what is already there.
#   ./install.sh [--model 27b|27b-8bit|flash-next]... [--venv DIR] [--python PATH] [--models-dir DIR]
#   --model       download the checkpoint and its drafter with `hf download` into the models directory, unless they are
#                 already complete there (27b: mlx-community/Qwen3.8-27B-4bit + incoai/Qwen3.8-27B-DFlash2; 27b-8bit:
#                 mlx-community/Qwen3.8-27B-8bit + the same drafter; flash-next: Jundot/Qwen3.8-Flash-Next-oQ4e-mtp,
#                 which drafts with its own MTP head). Repeat for more than one.
#   --venv        the virtual environment (default INFERD_VENV or ~/.venvs/inferd, where run.sh looks for it)
#   --python      the interpreter to build it from (default: the first Python 3.12 found)
#   --models-dir  where models go (default INFERD_MODELS or ~/models, where `run.sh start --model` looks)
# Works in a checkout's engine/ and in an unpacked release tarball (tools/package.sh) alike.
emulate -L zsh
setopt no_unset pipe_fail
cd "${0:A:h}" || exit 1

die() { print -u2 -r -- "install.sh: $*"; exit 1; }
say() { print -r -- "install.sh: $*"; }
VENV=${INFERD_VENV:-$HOME/.venvs/inferd} MROOT=${INFERD_MODELS:-$HOME/models} PYTHON= models=()
while (( $# )); do
  [[ $1 == --*=* ]] && set -- "${1%%=*}" "${1#*=}" "${@:2}"
  case $1 in
    -h|--help) sed -n '2,/^emulate/p' "${0:A}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --model|--venv|--python|--models-dir) (( $# >= 2 )) && [[ -n $2 ]] || die "$1 needs a value" ;;
    *) die "unknown option '$1' (./install.sh --help)" ;;
  esac
  v=$2; [[ $v == '~'/* ]] && v=$HOME/${v#'~'/}
  case $1 in
    --model) models+=($v) ;; --venv) VENV=$v ;; --python) PYTHON=$v ;; --models-dir) MROOT=$v ;;
  esac
  shift 2
done
source tools/models.zsh || die "tools/models.zsh is missing"
for m in $models; do
  model_info $m || die "unknown model '$m': use ${(j:, :)INFERD_MODEL_NAMES}"
done
[[ -f requirements.txt ]] || die "requirements.txt is missing next to install.sh"

# The machine: Apple Silicon, and (for a release tarball) the macOS its kernels were built for
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || die "inferd needs a Mac with Apple Silicon (arm64 macOS)"
host=${INFERD_MACOS:-${$(sw_vers -productVersion)%%.*}}
if [[ -f build/BUILD-INFO ]]; then
  built=$(sed -n 's/^macos_target //p' build/BUILD-INFO)
  [[ -z $built || $built == $host ]] ||
    die "this release is built for macOS $built and this Mac runs macOS $host: download the macos$host file (or build from source)"
elif [[ ! -x build/inferd ]]; then
  say "note: no build/inferd here yet: run ./build.sh (needs Xcode's Metal toolchain) before ./run.sh"
fi
ram=${INFERD_RAM_GB:-$(( $(/usr/sbin/sysctl -n hw.memsize) / 1073741824 ))}

# Python 3.12: the version the front end's pinned requirements are tested with
py_minor() { "$1" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null }
if [[ -n $PYTHON ]]; then
  [[ -x $PYTHON || -n ${commands[$PYTHON]:-} ]] || die "--python $PYTHON is not an executable"
  [[ $(py_minor $PYTHON) == 3.12 ]] || say "warning: $PYTHON is Python $(py_minor $PYTHON); inferd is tested with 3.12"
elif [[ -x $VENV/bin/python ]]; then
  PYTHON=$VENV/bin/python  # reuse the existing venv's interpreter
else
  cands=(${commands[python3.12]:-} /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12
         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3)
  (( ${+commands[uv]} )) && cands+=($(uv python find 3.12 2>/dev/null))
  cands+=(${commands[python3]:-})
  for c in $cands; do [[ -x $c && $(py_minor $c) == 3.12 ]] && { PYTHON=$c; break }; done
  [[ -n $PYTHON ]] || die "inferd's front end needs Python 3.12, and none was found (tried python3.12, Homebrew,
  python.org, uv and python3). Install it (e.g. \`brew install python@3.12\` or \`uv python install 3.12\`) and run
  ./install.sh again, or pass --python /path/to/python3.12"
fi

# The venv and the front end's packages (skipped when requirements.txt is unchanged since the last install)
if [[ -x $VENV/bin/python ]]; then
  vv=$(py_minor $VENV/bin/python)
  [[ $vv == 3.12 || -n ${INFERD_ANY_PYTHON:-} ]] ||
    die "$VENV already exists with Python $vv (not 3.12): move it away or use --venv DIR"
  say "using the venv $VENV (Python $vv)"
else
  say "creating the venv $VENV with $PYTHON ($(py_minor $PYTHON))"
  mkdir -p -- "${VENV:h}" || die "cannot create ${VENV:h}"
  "$PYTHON" -m venv "$VENV" || die "python -m venv $VENV failed"
fi
stamp=$VENV/.inferd-requirements.sha256 want=$(shasum -a 256 < requirements.txt | cut -d' ' -f1)
if [[ -f $stamp && $(<$stamp) == $want ]] &&
   "$VENV/bin/python" -I -c 'import fastapi, uvicorn, transformers, jinja2, mlx' 2>/dev/null; then
  say "the front end's packages are installed (requirements.txt unchanged)"
else
  say "installing the front end's packages (requirements.txt)"
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q -r requirements.txt ||
    die "pip install -r requirements.txt failed"
  "$VENV/bin/python" -I -c 'import fastapi, uvicorn, transformers, jinja2, mlx' 2>/dev/null ||
    { "$VENV/bin/python" -I -c 'import fastapi, uvicorn, transformers, jinja2, mlx'; die "the packages installed but do not import"; }
  print -r -- $want > $stamp
fi

# Models
have_weights() {  # dir: config.json and every weight file (the shards a safetensors index names, or one *.safetensors)
  [[ -f $1/config.json ]] || return 1
  if [[ -f $1/model.safetensors.index.json ]]; then
    "$VENV/bin/python" -I -c 'import json, os, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
sys.exit(0 if all(os.path.isfile(os.path.join(d, f)) for f in set(m.values())) else 1)' "$1"
  else
    local w=($1/*.safetensors(N))
    (( $#w ))
  fi
}
fetch() {  # repo dir
  if have_weights $2; then say "$1 is already in $2"; return 0; fi
  [[ -x $VENV/bin/hf ]] || die "$VENV/bin/hf is missing (it comes with huggingface_hub): pip install huggingface_hub"
  say "downloading $1 into $2 (resumes if interrupted; run ./install.sh again)"
  mkdir -p -- "$2" || die "cannot create $2"
  "$VENV/bin/hf" download "$1" --local-dir "$2" || die "hf download $1 failed"
  have_weights $2 || die "$2 is still incomplete after the download"
}
for m in $models; do
  model_info $m
  (( ram >= MI_MIN_GB )) || say "warning: $m needs a Mac with at least $MI_MIN_GB GB of memory; this one has $ram GB"
  fetch $MI_REPO $MROOT/$MI_DIR
  [[ -n $MI_DRAFT_REPO ]] && fetch $MI_DRAFT_REPO $MROOT/$MI_DRAFT_DIR
done

# What next
[[ $VENV == ${INFERD_VENV:-$HOME/.venvs/inferd} ]] || pyenv="PY=${(q-)VENV}/bin/python "
[[ $MROOT == ${INFERD_MODELS:-$HOME/models} ]] || menv="INFERD_MODELS=${(q-)MROOT} "
print
if (( $#models )); then
  say "done. Start the server with:"
  print -r -- "  cd ${(q-)PWD} && ${pyenv:-}${menv:-}./run.sh start --model ${models[1]}"
else
  say "done. Download a model with ./install.sh --model 27b, or start one you have with:"
  print -r -- "  cd ${(q-)PWD} && ${pyenv:-}./run.sh start --model /path/to/checkpoint [--draft /path/to/drafter]"
fi
