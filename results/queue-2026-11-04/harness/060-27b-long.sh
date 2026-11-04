JOB=060-27b-long; source ~/inference/queue/lib.sh
# Qwen3.8-27B one agent at 32k and 64k (cold prompt + 512 out), SSD tier on: how long a context a 32 GB Mac serves
export CTX=32k,64k REPS=1
inferd_up inferd27-long $M27 DRAFT=$DRAFT DRAFT_BITS=4 -- 4 1024 0 4 70000 && { single inferd27-long; NS=1,2,4 follow inferd27-long; inferd_down inferd27-long; }
