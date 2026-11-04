JOB=040-9b-scen; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-/8-bit agent scenarios b, c, e (forked agents, resumes, mixed fleet): inferd (api) vs TensorFold (plain)
export SCEN=b,c,e
for q in 4 8; do m=$M9; [ $q = 8 ] && m=$M9_8
  inferd_up inferd-q9b$q-scen $m -- 16 1024 0 8 32768 && { scen inferd-q9b$q api $IPID; inferd_down inferd-q9b$q-scen; }
  tf_up tf122-q9b$q-scen $m && { scen tf122-q9b$q plain $TP; tf_down; }
done
