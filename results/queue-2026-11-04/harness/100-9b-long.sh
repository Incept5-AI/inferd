JOB=100-9b-long; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-bit long contexts: one agent cold at 32k and 64k, follow-ups 32k x 1/2/4: inferd vs TensorFold 1.2.2
export CTX=32k,64k REPS=1
inferd_up inferd-q9b4-long $M9 -- 4 1024 0 4 70000 && { single inferd-q9b4-long; CTX=32k NS=1,2,4 follow inferd-q9b4-long; inferd_down inferd-q9b4-long; }
tf_up tf122-q9b4-long $M9 --context 70000 && { single tf122-q9b4-long; CTX=32k NS=1,2,4 follow tf122-q9b4-long; tf_down; }
