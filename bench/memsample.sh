#!/bin/zsh
# memsample.sh <out.tsv> <pid>  -- every 2 s: time, wired GiB, free-ish GiB (free+inactive+purgeable), server RSS GiB,
# memory_pressure free %. Runs until the pid exits. Wired is where MLX/Metal buffers live on Apple Silicon.
out=$1; pid=$2
echo -e "t\twired_gib\tavail_gib\trss_gib\tfree_pct" > $out
while kill -0 $pid 2>/dev/null; do
  vm=$(vm_stat)
  pg() { echo "$vm" | awk -v k="$1" '$0 ~ k {gsub("\\.","",$NF); print $NF}'; }
  w=$(pg "Pages wired down"); f=$(pg "Pages free"); i=$(pg "Pages inactive"); p=$(pg "Pages purgeable")
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  fp=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2}' | tr -d '%')
  printf "%s\t%.1f\t%.1f\t%.1f\t%s\n" $(date +%s) $(( w*16384/1073741824.0 )) $(( (f+i+p)*16384/1073741824.0 )) $(( ${rss:-0}/1048576.0 )) "$fp" >> $out
  sleep 2
done
