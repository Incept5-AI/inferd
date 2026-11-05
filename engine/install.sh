#!/bin/zsh
# Set up inferd's front end, and optionally download a model. Safe to run again: it reuses what is already there.
#   ./install.sh [--model 27b|27b-8bit|flash-next]... [--download] [--exact] [--venv DIR] [--python PATH]
#                [--models-dir DIR]
#   ./install.sh --list [--exact] [--models-dir DIR]
#   --model       the checkpoint and its drafter (27b: mlx-community/Qwen3.8-27B-4bit + incoai/Qwen3.8-27B-DFlash2;
#                 27b-8bit: mlx-community/Qwen3.8-27B-8bit + the same drafter; flash-next:
#                 Jundot/Qwen3.8-Flash-Next-oQ4e-mtp, which drafts with its own MTP head). Repeat for more than one.
#                 A complete copy already on this Mac is used where it is, with no download and no copy: the models
#                 directory first, then the Hugging Face cache, then LM Studio (`run.sh start --model` looks in the same
#                 places; tools/models.zsh). Otherwise `hf download` puts it in the models directory.
#   --download    download into the models directory even when a copy exists elsewhere (hf download resumes or
#                 checks what is already there)
#   --exact       use only the exact repo ids (as the benchmarks do). By default an equivalent build of the same weights
#                 already on the Mac (e.g. 27b: Qwen/ or lmstudio-community/Qwen3.8-27B-MLX-4bit; the z-lab drafter) is used,
#                 with a note, when its config.json matches. Start with `run.sh start --exact` too.
#   --list        show what was found for every model name, and where, then exit
#   --venv        the virtual environment (default INFERD_VENV or ~/.venvs/inferd, where run.sh looks for it)
#   --python      the interpreter to build it from (default: the first Python 3.12 found)
#   --models-dir  where models go (default INFERD_MODELS or ~/models, where `run.sh start --model` looks)
# Works in a checkout's engine/ and in an unpacked release tarball (tools/package.sh) alike.
emulate -L zsh
setopt no_unset pipe_fail
cd "${0:A:h}" || exit 1
ORIG_MODELS=${INFERD_MODELS:-}

die() { print -u2 -r -- "install.sh: $*"; exit 1; }
say() { print -r -- "install.sh: $*"; }
VENV=${INFERD_VENV:-$HOME/.venvs/inferd} MROOT=${INFERD_MODELS:-$HOME/models} PYTHON= models=() LIST=0 DOWNLOAD=0
while (( $# )); do
  [[ $1 == --*=* ]] && set -- "${1%%=*}" "${1#*=}" "${@:2}"
  case $1 in
    -h|--help) sed -n '2,/^emulate/p' "${0:A}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --list) LIST=1; shift; continue ;;
    --download) DOWNLOAD=1; shift; continue ;;
    --equivalent) INFERD_EQUIVALENT=1; shift; continue ;;
    --exact) INFERD_EQUIVALENT=0; shift; continue ;;
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
INFERD_MODELS=$MROOT  # where model_find looks first
if (( LIST )); then
  (( $#models == 0 )) || die "--list shows every model; drop --model"
  models_list
  exit 0
fi
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

# Models: a complete copy anywhere model_find looks is used where it is; otherwise hf download into MROOT
notes() { local n; for n in $FOUND_NOTES; do say "note: $n"; done; [[ -z $FOUND_HINT ]] || say "note: found $FOUND_HINT" }
fetch() {  # model|draft repo dir facts
  if (( ! DOWNLOAD )); then
    if model_find $1; then
      notes; say "$2 is already on this Mac: $FOUND ($FOUND_WHY); no download"; return 0
    fi
    notes
  fi
  local hf=$VENV/bin/hf
  [[ -x $hf ]] || hf=${commands[hf]:-}
  [[ -n $hf ]] || die "$VENV/bin/hf is missing (it comes with huggingface_hub): pip install huggingface_hub"
  say "downloading $2 into $3 (resumes if interrupted; run ./install.sh again)"
  mkdir -p -- "$3" || die "cannot create $3"
  "$hf" download "$2" --local-dir "$3" || die "hf download $2 failed"
  ckpt_check $3 $4 || die "$3 is not a complete $2 after the download: $REPLY"
}
for m in $models; do
  model_info $m
  (( ram >= MI_MIN_GB )) || say "warning: $m needs a Mac with at least $MI_MIN_GB GB of memory; this one has $ram GB"
  fetch model $MI_REPO $MROOT/$MI_DIR "$MI_FACTS"
  [[ -n $MI_DRAFT_REPO ]] && fetch draft $MI_DRAFT_REPO $MROOT/$MI_DRAFT_DIR "$MI_DRAFT_FACTS"
done

# What next
[[ $VENV == ${INFERD_VENV:-$HOME/.venvs/inferd} ]] || pyenv="PY=${(q-)VENV}/bin/python "
[[ ${INFERD_EQUIVALENT:-1} == 0 ]] && eqv=" --exact"
[[ $MROOT == ${ORIG_MODELS:-$HOME/models} ]] || menv="INFERD_MODELS=${(q-)MROOT} "
print
if (( $#models )); then
  say "done. Start the server with:"
  print -r -- "  cd ${(q-)PWD} && ${pyenv:-}${menv:-}./run.sh start --model ${models[1]}${eqv:-}"
else
  say "done. Download a model with ./install.sh --model 27b, or start one you have with:"
  print -r -- "  cd ${(q-)PWD} && ${pyenv:-}./run.sh start --model /path/to/checkpoint [--draft /path/to/drafter]"
fi
