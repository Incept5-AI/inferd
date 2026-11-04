JOB=030-9b-fleet; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-/8-bit agent fleet (agents.py, 8k contexts, 8/16/32 agents, turns per minute): inferd vs TensorFold 1.2.2
for q in 4 8; do m=$M9; [ $q = 8 ] && m=$M9_8
  inferd_up inferd-q9b$q-fleet $m -- 8 1024 0 8 12288 && { fleet inferd-q9b$q; inferd_down inferd-q9b$q-fleet; }
  tf_up tf122-q9b$q-fleet $m && { fleet tf122-q9b$q; tf_down; }
done
