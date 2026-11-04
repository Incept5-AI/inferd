JOB=050-27b-scen-ade; source ~/inference/queue/lib.sh
# Qwen3.8-27B 4-bit + DFlash2, agent scenarios a, d, e (many small agents, one-offs beside long-lived, mixed fleet), inferd api
export SCEN=a,d,e
inferd_up inferd27-scen $M27 DRAFT=$DRAFT DRAFT_BITS=4 -- 16 1024 0 8 32768 && { scen inferd27-ade api $IPID; inferd_down inferd27-scen; }
