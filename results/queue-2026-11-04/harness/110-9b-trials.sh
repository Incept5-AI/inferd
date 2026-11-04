JOB=110-9b-trials; source ~/inference/queue/lib.sh
# Qwen3.5-9B 4-bit follow-ups 4k/16k x 4/8/16, two more trials of each engine (noise bands for the report)
export NS=4,8,16
inferd_up inferd-q9b4-trials $M9 -- 16 1024 0 8 32768 && { follow inferd-q9b4-t2; follow inferd-q9b4-t3; inferd_down inferd-q9b4-trials; }
tf_up tf122-q9b4-trials $M9 && { follow tf122-q9b4-t2; follow tf122-q9b4-t3; tf_down; }
