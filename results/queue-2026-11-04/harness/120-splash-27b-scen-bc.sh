JOB=120-splash-27b-scen-bc; source ~/inference/queue/lib.sh
# mini-m6 only: Splash 1.5.0 on Qwen3.8-27B 4-bit + DFlash2, agent scenarios b,c (plain mode, preset m6): the cell the
# 4 Nov M6 head-to-head lost to a quoting bug (results/m6-h2h-2026-11-04). Same window as its follow-up arm (32k).
command -v splash >/dev/null || { say "$JOB: no splash on this box"; exit 0; }
splash_up splash-27b-scen mlx-community/Qwen3.8-27B-4bit 32768 $DRAFT && { scen splash-27b-m6 plain $(port_pid 8100); splash_down; }
