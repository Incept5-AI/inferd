#!/bin/zsh
# run.sh's stop / status / argument checks, with fake processes in a temporary INFERD_RUN_DIR (no model, no build, no
# GPU). Port 18399 must be free. Cleans up only the pids it started.
#   ./test_run.sh
emulate -L zsh
setopt no_unset
cd "${0:A:h}" || exit 1
PORT=18399
export INFERD_RUN_DIR=$(mktemp -d /tmp/inferd-test.XXXXXX) INFERD_LOGS=$(mktemp -d /tmp/inferd-logs.XXXXXX) STOP_WAIT=1
chmod 700 $INFERD_RUN_DIR
SOCK=$INFERD_RUN_DIR/inferd-$PORT.sock LEGACY=/tmp/inferd-$PORT.sock
pass=0 fail=0 mine=()
check() {  # description, then a command that must succeed
  local what=$1; shift
  if "$@"; then pass=$((pass + 1)); print "ok   $what"; else fail=$((fail + 1)); print "FAIL $what"; fi
}
alive() { kill -0 $1 2>/dev/null }
gone() { local i; for i in {1..40}; do alive $1 || return 0; sleep 0.1; done; return 1 }
spawn() {  # command... -> pid in $REPLY (recorded for cleanup)
  "$@" >/dev/null 2>&1 </dev/null &
  REPLY=$!; mine+=($REPLY); sleep 0.3
}
cleanup() { local p; for p in $mine; do kill -KILL $p 2>/dev/null; done; rm -rf $INFERD_RUN_DIR $INFERD_LOGS; [[ -S $LEGACY && -O $LEGACY ]] && rm -f $LEGACY }
trap cleanup EXIT
lsof -nP -t -iTCP:$PORT -sTCP:LISTEN >/dev/null && { print "port $PORT is in use"; exit 1; }

# argument checks (start validates before needing MODEL, stop/status need neither MODEL nor a build)
out=$(./run.sh stop abc 2>&1); check "stop rejects a bad port" test $? -ne 0
out=$(./run.sh stop 1 2 2>&1); check "stop rejects extra arguments" test $? -ne 0
out=$(./run.sh 0 2>&1); check "start rejects port 0" test $? -ne 0
out=$(./run.sh $PORT abc 2>&1); check "start rejects max_running abc" test $? -ne 0
out=$(./run.sh $PORT 16 1024 x 2>&1); check "start rejects kv_gib x" test $? -ne 0
out=$(MODEL= ./run.sh $PORT 2>&1); check "start requires MODEL" eval '[[ $out == *"set MODEL"* ]]'

# nothing running
out=$(./run.sh stop $PORT 2>&1); check "stop with nothing running exits 0" test $? -eq 0
out=$(./run.sh status $PORT 2>&1); rc=$?
check "status with nothing running exits 3" test $rc -eq 3
check "status names the socket and logs" eval '[[ $out == *$SOCK* && $out == *serve-$PORT.log* && $out == *"not running"* ]]'
print -r -- $INFERD_LOGS > $INFERD_RUN_DIR/logs-$PORT   # what a start records
out=$(INFERD_LOGS=/nonexistent ./run.sh status $PORT 2>&1)
check "status reports the log directory the start used" eval '[[ $out == *"$INFERD_LOGS/inferd-$PORT.log"* && $out != *nonexistent* ]]'
rm -f $INFERD_RUN_DIR/logs-$PORT

# a running pair: front end (matches frontend/serve.py --port) and an inferd that ignores SIGTERM (SIGKILL after STOP_WAIT)
spawn perl -e 'sleep 600' frontend/serve.py --model m --socket $SOCK --host 127.0.0.1 --port $PORT; fe=$REPLY
spawn perl -e '$SIG{TERM}="IGNORE"; sleep 600' ./build/inferd m $SOCK 16; eng=$REPLY
print $fe > $INFERD_RUN_DIR/serve-$PORT.pid; print $eng > $INFERD_RUN_DIR/inferd-$PORT.pid
out=$(./run.sh status $PORT 2>&1); rc=$?
check "status with both running exits 0" test $rc -eq 0
check "status prints both pids" eval '[[ $out == *"front end $fe "* && $out == *"inferd    $eng "* ]]'
out=$(./run.sh stop $PORT 2>&1); rc=$?
check "stop exits 0" test $rc -eq 0
check "front end stopped" gone $fe
check "SIGTERM-ignoring inferd stopped (SIGKILL after STOP_WAIT)" gone $eng
check "stop reports the escalation" eval '[[ $out == *SIGKILL* && $out == *"stopped 2 process"* ]]'
check "pid files removed" eval '[[ ! -e $INFERD_RUN_DIR/serve-$PORT.pid && ! -e $INFERD_RUN_DIR/inferd-$PORT.pid ]]'

# a recycled pid: the pid file names a process that is not ours to stop
spawn perl -e 'sleep 600'; other=$REPLY
print $other > $INFERD_RUN_DIR/serve-$PORT.pid; print $other > $INFERD_RUN_DIR/inferd-$PORT.pid
out=$(./run.sh stop $PORT 2>&1); rc=$?
check "stop exits 0 with a stale pid file" test $rc -eq 0
check "an unrelated process named by a pid file survives" alive $other

# an engine started by an older run.sh (socket in /tmp, no pid file) is found through its socket and stopped
if [[ -e $LEGACY ]]; then
  print "skip legacy-socket case: $LEGACY exists"
else
  spawn python3 -c 'import socket, sys, time
s = socket.socket(socket.AF_UNIX); s.bind(sys.argv[3]); s.listen(1); time.sleep(600)' build/inferd m $LEGACY 16
  legacy=$REPLY
  out=$(./run.sh status $PORT 2>&1)
  check "status lists the legacy holder" eval '[[ $out == *"other     pid $legacy"* ]]'
  out=$(./run.sh stop $PORT 2>&1); rc=$?
  check "stop exits 0 (legacy)" test $rc -eq 0
  check "legacy engine on /tmp socket stopped" gone $legacy
  check "its stale socket removed" eval '[[ ! -e $LEGACY ]]'
fi

# a foreign listener on the port is reported and left alone; stop still exits 0
spawn python3 -c 'import socket, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", '$PORT')); s.listen(1); time.sleep(600)'
foreign=$REPLY
out=$(./run.sh stop $PORT 2>&1); rc=$?
check "stop exits 0 beside a foreign listener" test $rc -eq 0
check "foreign listener reported" eval '[[ $out == *"not ours, left alone"* ]]'
check "foreign listener survives" alive $foreign

print "\n$pass passed, $fail failed"
(( fail == 0 ))
