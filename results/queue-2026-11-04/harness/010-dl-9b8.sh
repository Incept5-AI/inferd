JOB=010-dl-9b8; source ~/inference/queue/lib.sh
# Qwen3.5-9B 8-bit (mlx-community, the 4-bit's sibling) into ~/models
[ -f $M9_8/config.json ] || $(dirname $PY)/hf download mlx-community/Qwen3.5-9B-MLX-8bit --local-dir $M9_8 > $O/dl.log 2>&1
say "$JOB: $(du -sh $M9_8 | cut -f1) $(python3 -c "import json;print(json.load(open('$M9_8/config.json'))['quantization'])")"
