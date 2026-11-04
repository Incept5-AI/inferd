#!/bin/zsh
# run_all.sh MACHINE [URL]: runs every example against one inferd server as a self-test of that machine, keeps each
# output with its wall time in results/MACHINE/, and ends with a PASS/FAIL table (exit code, plus the PASS/FAIL lines
# a demo printed). Exits 1 if any demo failed. The server must be up first (see README.md, "Start the server");
# clients can run on another Mac (e.g. through `ssh -N -L 18300:127.0.0.1:8300 <host>` and URL
# http://127.0.0.1:18300/v1). ONLY=name1,name2 re-runs a subset into the same results (the summary then covers every
# demo recorded there, each with its latest run).
zmodload zsh/datetime
cd "$(dirname $0)"
M=$1; export INFERD_URL=${2:-http://127.0.0.1:8300/v1}
[[ -z $M ]] && { echo "usage: run_all.sh MACHINE [URL]"; exit 2; }
PY=${PY:-python3}; R=results/$M; mkdir -p $R
# LANGBENCH (a languages-benchmark checkout) passes through to coding-smoke, which clones one into
# ~/.cache/inferd/languages-benchmark when it is unset.
run() {  # name command...
  [[ -n $ONLY && ",$ONLY," != *",$1,"* ]] && return
  local t0=$EPOCHREALTIME
  "${@:2}" > $R/$1.txt 2>&1
  local rc=$?
  local dt=$(printf "%.1f" $(( EPOCHREALTIME - t0 )))
  echo "$1	$dt s	exit $rc" | tee -a $R/timings.tsv
}
if [[ -z $ONLY ]]; then
  echo "# $M  $(date '+%Y-%m-%d %H:%M')  $INFERD_URL" > $R/timings.tsv
else
  echo "# re-run $ONLY  $(date '+%Y-%m-%d %H:%M')  $INFERD_URL" >> $R/timings.tsv
fi
run quickstart           $PY quickstart/quickstart.py
run agent-budgets        $PY agent-budgets/agent_budgets.py
run tools-web-search     $PY tools-web-search/web_search_agent.py
run tools-shell          $PY tools-shell/shell_agent.py
run structured-output    $PY structured-output/structured_output.py
run agent-contexts       $PY agent-contexts/agent_contexts.py --url $INFERD_URL
run agent-handoff        $PY agent-handoff/agent_handoff.py --url $INFERD_URL
run agent-team           $PY agent-team/agent_team.py --url $INFERD_URL
run ocr-agents           $PY ocr-agents/ocr_agents.py
run ocr-agents-compare   $PY ocr-agents/ocr_agents.py --compare 3
run coding-smoke-none    ./coding-smoke/coding_smoke.sh none 8
run coding-smoke-medium  ./coding-smoke/coding_smoke.sh medium 8
run agent-swarm          $PY agent-swarm/agent_swarm.py --url $INFERD_URL
run agent-followups      $PY agent-followups/agent_followups.py --url $INFERD_URL
run users-at-once        $PY users-at-once/users_at_once.py --url $INFERD_URL

# Summary: a demo passes when it exits 0 and printed no FAIL line.
{
  echo
  printf "%-20s %-6s %8s %5s %8s\n" demo result time exit "checks"
  bad=0
  NAMES=(${(u)$(awk -F'\t' '!/^#/ {print $1}' $R/timings.tsv)})
  for n in $NAMES; do
    line=$(awk -F'\t' -v n=$n '$1 == n' $R/timings.tsv | tail -1)
    t=$(echo "$line" | cut -f2); rc=${${line##*exit }:-?}
    p=$(grep -cE '^ *PASS ' $R/$n.txt); f=$(grep -cE '^ *FAIL ' $R/$n.txt)
    res=PASS; [[ $rc != 0 || $f != 0 ]] && { res=FAIL; bad=$((bad + 1)); }
    c=-; (( p + f )) && c="$p/$((p + f))"
    printf "%-20s %-6s %8s %5s %8s\n" $n $res "$t" $rc $c
    grep -E '^ *(FAIL|SKIP) ' $R/$n.txt | sed 's/^ */    /'
    if [[ $n == coding-smoke-* && $rc == 2 ]]; then  # coding_smoke.sh exits 2 when a toolchain is missing
      why=$(grep -m1 '^coding-smoke needs' $R/$n.txt)
      [[ $why == *tsx* ]] && echo "    tsx missing: $why" || echo "    toolchain missing: ${why:-exit 2}"
    fi
  done
  echo; echo "$((${#NAMES} - bad))/${#NAMES} demos passed on $M"
} > $R/summary.txt
cat $R/summary.txt
[[ $bad == 0 ]]

