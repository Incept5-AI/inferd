JOB=090-9b-scen-ade; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-bit agent scenarios a, d, e (many small agents, one-offs beside long-lived, mixed fleet): inferd (api) vs TensorFold 1.2.2 (plain)
export SCEN=a,d,e
inferd_up inferd-q9b4-ade $M9 -- 16 1024 0 8 32768 && { scen inferd-q9b4-ade api $IPID; inferd_down inferd-q9b4-ade; }
tf_up tf122-q9b4-ade $M9 && { scen tf122-q9b4-ade plain $TP; tf_down; }
