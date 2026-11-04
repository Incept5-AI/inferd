#!/bin/zsh
# runner.sh: a job queue for one benchmark machine (2026-11-04). Keeps the box busy: runs the oldest
# ~/inference/queue/pending/*.sh, one at a time, logging to logs/<job>.log, then moves it to done/ (exit 0) or failed/.
# Before each job it waits until no model process runs (an agent's run or a stray server), so two models never load
# at once. Stop it with: touch ~/inference/queue/STOP (it exits between jobs).
Q=~/inference/queue; mkdir -p $Q/{pending,running,done,failed,logs}
echo $$ > $Q/runner.pid
models() { ps -axo comm= | grep -c -E '(^|/)(inferd|tensorfold-native|splash)$'; }
while [ ! -f $Q/STOP ]; do
  j=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then sleep 60; continue; fi
  while [ "$(models)" -gt 0 ]; do sleep 30; done
  n=${j:t:r}; mv $j $Q/running/
  echo "$(date '+%F %T') start $n" >> $Q/queue.log
  zsh $Q/running/$n.sh > $Q/logs/$n.log 2>&1; rc=$?
  [ $rc = 0 ] && mv $Q/running/$n.sh $Q/done/ || mv $Q/running/$n.sh $Q/failed/
  echo "$(date '+%F %T') end $n rc=$rc" >> $Q/queue.log
  while [ "$(models)" -gt 0 ]; do sleep 10; done   # a job must leave no server behind
done
rm -f $Q/runner.pid $Q/STOP
