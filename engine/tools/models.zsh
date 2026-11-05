# The named models of `install.sh --model` and `run.sh start --model`, and where to find them (sourced by both; zsh).
#   model_info NAME   sets MI_REPO, MI_DIR (under INFERD_MODELS, default ~/models), MI_DRAFT_REPO / MI_DRAFT_DIR (empty:
#                     no separate drafter), MI_MTP (1: the checkpoint's own MTP head drafts) and MI_MIN_GB (the smallest
#                     Mac it runs on); MI_FACTS / MI_DRAFT_FACTS (what config.json must say, below) and MI_EQUIV /
#                     MI_DRAFT_EQUIV (known equivalent builds). Returns 1 for a name it does not know.
#   model_find model|draft   after model_info: the first complete, matching copy, in this order:
#                     1. $INFERD_MODELS/<dir> (where install.sh downloads; default ~/models)
#                     2. the Hugging Face cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub):
#                        models--<org>--<name>/snapshots/<rev>, the snapshot refs/main points to, else the newest complete
#                     3. LM Studio: $LMSTUDIO_MODELS/<org>/<name> (default ~/.lmstudio/models), then the older
#                        ~/.cache/lm-studio/models/<org>/<name>
#                     It sets FOUND (the directory, empty if none), FOUND_WHY (which copy and why), FOUND_NOTES (the
#                     copies it skipped, and why) and FOUND_HINT (an equivalent build it found but was not allowed to use).
# Same weights: by default only the exact repo id counts. INFERD_EQUIVALENT=1 (install.sh / run.sh start --equivalent)
# also accepts the builds listed in MI_EQUIV / MI_DRAFT_EQUIV: the same model at the same quantization from another
# publisher. Every candidate, exact or not, must be complete (config.json, and every shard model.safetensors.index.json
# lists present and non-empty, or a non-empty model.safetensors) and its config.json must match MI_FACTS: architecture,
# layers, hidden size, quantization bits and group size ("- -": unquantized). So an 8-bit build never stands in for 27b.
# Repo ids: README / CONTRIBUTING "Getting a model"; the Flash-Next oQ4e build is the one every Flash-Next result in
# docs/RESULTS.md used. Memory: the 27B 4-bit runs on the 32 GB M6, the 8-bit on the 64 GB M5 Pro, Flash-Next needs
# ~170 GiB (docs/FEATURES.md: Mac Studio only). z-lab/Qwen3.8-27B-DFlash2 is byte-identical to the incoai drafter (same
# sha256 for model.safetensors and the same config.json).
INFERD_MODEL_NAMES=(27b 27b-8bit flash-next)
model_info() {
  MI_DRAFT_REPO=incoai/Qwen3.8-27B-DFlash2 MI_DRAFT_DIR=Qwen3.8-27B-DFlash2 MI_MTP=0
  MI_DRAFT_FACTS="DFlash2DraftModel 5 5120 - -" MI_DRAFT_EQUIV=(z-lab/Qwen3.8-27B-DFlash2)
  case $1 in
    27b)        MI_REPO=mlx-community/Qwen3.8-27B-4bit MI_DIR=Qwen3.8-27B-4bit MI_MIN_GB=32
                MI_FACTS="Qwen3_5ForConditionalGeneration 64 5120 4 64"
                MI_EQUIV=(lmstudio-community/Qwen3.8-27B-MLX-4bit Qwen/Qwen3.8-27B-MLX-4bit) ;;
    27b-8bit)   MI_REPO=mlx-community/Qwen3.8-27B-8bit MI_DIR=Qwen3.8-27B-8bit MI_MIN_GB=64
                MI_FACTS="Qwen3_5ForConditionalGeneration 64 5120 8 64"
                MI_EQUIV=(lmstudio-community/Qwen3.8-27B-MLX-8bit Qwen/Qwen3.8-27B-MLX-8bit) ;;
    flash-next) MI_REPO=Jundot/Qwen3.8-Flash-Next-oQ4e-mtp MI_DIR=Qwen3.8-Flash-Next-oQ4e-mtp MI_MIN_GB=256
                MI_FACTS="Qwen4ExpForConditionalGeneration 48 2560 4 64" MI_EQUIV=()
                MI_DRAFT_REPO= MI_DRAFT_DIR= MI_DRAFT_FACTS= MI_DRAFT_EQUIV=() MI_MTP=1 ;;
    *) return 1 ;;
  esac
}

# ckpt_scan DIR: REPLY = "arch layers hidden bits group" from config.json ('?' unknown, '-' unquantized), reply = the
# shard files model.safetensors.index.json names (empty without an index). Returns 1 without a readable config.json,
# 2 for an unreadable index. JavaScript for Automation reads the JSON: on every Mac, no Python needed.
ckpt_scan() {
  local r
  r=$(/usr/bin/osascript -l JavaScript -e 'function run(argv) {
  ObjC.import("Foundation");
  function rd(p) {
    var s = $.NSString.stringWithContentsOfFileEncodingError(p, 4, null);
    if (s.isNil()) return undefined;
    try { return JSON.parse(s.js); } catch (e) { return null; }
  }
  var c = rd(argv[0] + "/config.json");
  if (!c || typeof c !== "object") return "noconfig";
  var t = c.text_config || c, q = c.quantization || c.quantization_config, a = c.architectures;
  function v(x) { return x === undefined || x === null ? "?" : String(x).replace(/\s/g, "_"); }
  var out = [Array.isArray(a) && a.length ? v(a[0]) : "?", v(t.num_hidden_layers), v(t.hidden_size),
             q ? v(q.bits) : "-", q ? v(q.group_size) : "-"].join(" ");
  var x = rd(argv[0] + "/model.safetensors.index.json");
  if (x === undefined) return out + "\nnoindex";
  if (!x || typeof x.weight_map !== "object") return out + "\nbadindex";
  var s = {};
  for (var k in x.weight_map) s[x.weight_map[k]] = 1;
  return [out, "index"].concat(Object.keys(s)).join("\n");
}' "$1" 2>/dev/null) || return 1
  local lines=("${(@f)r}")
  [[ ${lines[1]:-noconfig} == noconfig ]] && return 1
  REPLY=${lines[1]} reply=()
  [[ ${lines[2]:-} == badindex ]] && return 2
  [[ ${lines[2]:-} == index ]] && reply=("${(@)lines[3,-1]}")
  return 0
}
# ckpt_check DIR FACTS: 0 when DIR is a complete checkpoint whose config.json matches FACTS (empty FACTS: completeness
# only); otherwise 1 with the reason in REPLY. Symlinks (the HF cache's snapshots point into blobs/) are followed.
ckpt_check() {
  local d=$1 want=${2:-} f
  [[ -f $d/config.json ]] || { REPLY="no config.json"; return 1 }
  ckpt_scan $d; case $? in
    1) REPLY="config.json is not readable JSON"; return 1 ;;
    2) REPLY="model.safetensors.index.json is not readable JSON"; return 1 ;;
  esac
  local facts=$REPLY
  if [[ -f $d/model.safetensors.index.json ]]; then
    (( $#reply )) || { REPLY="model.safetensors.index.json lists no shards"; return 1 }
    for f in $reply; do
      [[ -s $d/$f ]] || { REPLY="incomplete: $f is missing or empty"; return 1 }
    done
  else
    [[ -s $d/model.safetensors ]] || { REPLY="incomplete: no weights (model.safetensors)"; return 1 }
  fi
  if [[ -n $want && $facts != $want ]]; then
    REPLY="config.json says $(facts_text ${=facts}), not $(facts_text ${=want})"; return 1
  fi
  return 0
}
facts_text() {  # "arch layers hidden bits group" -> words
  local q="unquantized"; [[ $4 != - ]] && q="$4-bit, group size $5"
  print -r -- "$1, $2 layers x $3, $q"
}

hf_hub_dir() { print -r -- ${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub} }
lmstudio_dirs() {
  local a=${LMSTUDIO_MODELS:-$HOME/.lmstudio/models} b=$HOME/.cache/lm-studio/models
  print -r -- $a; [[ ${b:A} == ${a:A} ]] || print -r -- $b
}
# model_candidates REPO DIRNAME: "dir<TAB>where" lines, in search order (directories that exist)
model_candidates() {
  local repo=$1 dn=$2 hub=$(hf_hub_dir) h s ref l
  [[ -d ${INFERD_MODELS:-$HOME/models}/$dn ]] && print -r -- "${INFERD_MODELS:-$HOME/models}/$dn	in INFERD_MODELS"
  h=$hub/models--${repo//\//--}
  if [[ -d $h/snapshots ]]; then
    ref=
    [[ -f $h/refs/main ]] && ref=$(<$h/refs/main) && [[ -d $h/snapshots/$ref ]] &&
      print -r -- "$h/snapshots/$ref	in the Hugging Face cache (the snapshot refs/main points to)"
    for s in $h/snapshots/*(N/om); do  # newest first
      [[ ${s:t} == $ref ]] || print -r -- "$s	in the Hugging Face cache (snapshot ${s:t})"
    done
  fi
  for l in ${(f)"$(lmstudio_dirs)"}; do
    [[ -d $l/$repo ]] && print -r -- "$l/$repo	in LM Studio's models"
  done
}
model_find() {  # model|draft
  local repo dn facts equiv=() r d why c seen=()
  if [[ $1 == draft ]]; then repo=$MI_DRAFT_REPO dn=$MI_DRAFT_DIR facts=$MI_DRAFT_FACTS equiv=($MI_DRAFT_EQUIV)
  else repo=$MI_REPO dn=$MI_DIR facts=$MI_FACTS equiv=($MI_EQUIV)
  fi
  FOUND= FOUND_WHY= FOUND_NOTES=() FOUND_HINT=
  for r in $repo $equiv; do
    if [[ $r != $repo && ${INFERD_EQUIVALENT:-0} != 1 ]]; then  # look, to say so, but do not use it
      [[ -n $FOUND_HINT ]] && continue
      for c in ${(f)"$(model_candidates $r ${r:t})"}; do
        d=${c%%	*}
        ckpt_check $d $facts && { FOUND_HINT="$r (an equivalent build of $repo) is $d: use it with --equivalent (INFERD_EQUIVALENT=1)"; break }
      done
      continue
    fi
    for c in ${(f)"$(model_candidates $r $([[ $r == $repo ]] && print -r -- $dn || print -r -- ${r:t}))"}; do
      d=${c%%	*} why=${c#*	}
      (( ${seen[(Ie)${d:A}]} )) && continue
      seen+=(${d:A})
      if ckpt_check $d $facts; then
        FOUND=$d FOUND_WHY="$r, $why"
        [[ $r != $repo ]] && FOUND_WHY+="; an equivalent build of $repo (--equivalent): $(facts_text ${=facts})"
        return 0
      fi
      FOUND_NOTES+=("skipped $d ($r): $REPLY")
    done
  done
  return 1
}
# models_list: what model_find finds for every name (install.sh --list, run.sh models)
models_list() {
  local n role n_ok=0 note
  for n in $INFERD_MODEL_NAMES; do
    model_info $n
    for role in model draft; do
      [[ $role == draft && -z $MI_DRAFT_REPO ]] && { print -r -- "$n drafter: none (its own MTP head drafts)"; continue }
      local label=$n repo=$MI_REPO; [[ $role == draft ]] && label="$n drafter" repo=$MI_DRAFT_REPO
      if model_find $role; then print -r -- "$label: $FOUND"; print -r -- "    ($FOUND_WHY)"
      else print -r -- "$label: not found ($repo)"
      fi
      for note in $FOUND_NOTES; do print -r -- "    $note"; done
      [[ -n $FOUND_HINT ]] && print -r -- "    found $FOUND_HINT"
    done
  done
}
