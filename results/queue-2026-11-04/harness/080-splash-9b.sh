JOB=080-splash-9b; source ~/inference/queue/lib.sh
# Splash (latest installed) on Qwen3.5-9B 4-/8-bit: single, follow-ups 4k/16k x 1-16, 8k fleet; same harness as inferd/TF
command -v splash >/dev/null || { say "$JOB: no splash on this box"; exit 0; }
for q in 4 8; do m=$M9; [ $q = 8 ] && m=$M9_8
  splash_up splash-q9b$q $m 32768 && { single splash-q9b$q; follow splash-q9b$q; splash_down; }
  splash_up splash-q9b$q-fleet $m 12288 && { fleet splash-q9b$q; splash_down; }
done
