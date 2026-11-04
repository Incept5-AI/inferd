# The named models of `install.sh --model` and `run.sh start --model` (sourced by both; zsh).
#   model_info NAME   sets MI_REPO, MI_DIR (under INFERD_MODELS, default ~/models), MI_DRAFT_REPO / MI_DRAFT_DIR (empty:
#                     no separate drafter), MI_MTP (1: the checkpoint's own MTP head drafts) and MI_MIN_GB (the smallest
#                     Mac it runs on). Returns 1 for a name it does not know.
# Repo ids: README / CONTRIBUTING "Getting a model"; the Flash-Next oQ4e build is the one every Flash-Next result in
# docs/RESULTS.md used. Memory: the 27B 4-bit runs on the 32 GB M6, the 8-bit on the 64 GB M5 Pro, Flash-Next needs
# ~170 GiB (docs/FEATURES.md: Mac Studio only).
INFERD_MODEL_NAMES=(27b 27b-8bit flash-next)
model_info() {
  MI_DRAFT_REPO=incoai/Qwen3.8-27B-DFlash2 MI_DRAFT_DIR=Qwen3.8-27B-DFlash2 MI_MTP=0
  case $1 in
    27b)        MI_REPO=mlx-community/Qwen3.8-27B-4bit MI_DIR=Qwen3.8-27B-4bit MI_MIN_GB=32 ;;
    27b-8bit)   MI_REPO=mlx-community/Qwen3.8-27B-8bit MI_DIR=Qwen3.8-27B-8bit MI_MIN_GB=64 ;;
    flash-next) MI_REPO=Jundot/Qwen3.8-Flash-Next-oQ4e-mtp MI_DIR=Qwen3.8-Flash-Next-oQ4e-mtp MI_MIN_GB=256
                MI_DRAFT_REPO= MI_DRAFT_DIR= MI_MTP=1 ;;
    *) return 1 ;;
  esac
}
