JOB=020-9b-single-follow; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-bit and 8-bit, one agent + follow-ups 4k/16k x 1-16: inferd v0.2.0 (no drafter) vs TensorFold 1.2.2
for q in 4 8; do m=$M9; [ $q = 8 ] && m=$M9_8
  inferd_up inferd-q9b$q $m -- 16 1024 0 8 32768 && { single inferd-q9b$q; follow inferd-q9b$q; inferd_down inferd-q9b$q; }
  tf_up tf122-q9b$q $m && { single tf122-q9b$q; follow tf122-q9b$q; tf_down; }
done
