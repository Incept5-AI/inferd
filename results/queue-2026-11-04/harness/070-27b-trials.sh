JOB=070-27b-trials; source ~/inference/queue/lib.sh
# Qwen3.8-27B follow-ups 4k/16k x 1/4/8/16, 3 trials (noise bands for the M6 report)
inferd_up inferd27-trials $M27 DRAFT=$DRAFT DRAFT_BITS=4 -- 16 1024 0 8 32768 && { for t in 1 2 3; do follow inferd27-t$t; done; inferd_down inferd27-trials; }
