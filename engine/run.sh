#!/bin/zsh
# Start, stop or inspect inferd + the OpenAI front end on one port.
#   run.sh [port=8300] [max_running=64] [step_tokens=2048] [kv_gib=96] [snapshots=96] [max_ctx=131072]   start
#   run.sh stop [port=8300]     stop exactly what run.sh started on that port (pid files: SIGTERM, wait up to
#                               STOP_WAIT s, then SIGKILL); exit 0 when nothing is running
#   run.sh status [port=8300]   pids, socket and log paths; exit 0 when both processes run, 3 otherwise
# The engine socket is no longer /tmp/inferd-<port>.sock: use `run.sh stop` / `run.sh status`, not lsof on a path.
# Env: MODEL (checkpoint dir, required to start), DRAFT (DFlash2 dir; speculation off without it), DRAFT_BITS, PY (python
# with the front end's packages, default python3), SPILL_DIR/SPILL_GIB (SSD tier), VISION, plus the engine knobs in
# README.md. INFERD_LOGS (default ~/inference/logs), INFERD_RUN_DIR (owner-only directory for the engine socket and pid
# files, default $TMPDIR/inferd-<uid>), INFERD_HOST (front end bind address, default 127.0.0.1; anything else needs
# INFERD_API_KEY; not HOST, which zsh sets to the machine name),
# INFERD_API_KEY (bearer token for every HTTP endpoint), STOP_WAIT (seconds to wait for a graceful stop, 60).
# Trust model: one trusted local user (README "Security and limits"); `python3 frontend/serve.py --help` lists the
# front end's limits.
# VISION=1 when the server will take images: auto KV sizing then leaves ~1 GiB (budgets <= 64 GiB) for the front end's
# MLX vision tower; without it, image work on a full 32-64 GB machine can hit a GPU out-of-memory step failure.
# Small machines: kv_gib 0 sizes the pool from what is left. 32 GB (M6): `run.sh 8300 16 1024 0 8 32768` with DRAFT
# set and SPILL_DIR on (the drafter wins there: 1 stream 39 vs 9.5 tok/s, 16 agents x 8k 89 vs 64; README "Sizing").
# M3/M4 (no neural accelerators): leave DRAFT unset for fleets of 16+ agents.
# A start first stops what this script started on the same port (as `run.sh stop`); it never kills by command-line
# pattern, and refuses if a process it does not recognise holds the port.
emulate -L zsh
setopt no_unset pipe_fail
cd "${0:A:h}" || exit 1

die() { print -u2 -r -- "run.sh: $*"; exit 1; }
int_in() {  # name value lo hi
  [[ $2 == <-> ]] && (( $2 >= $3 && $2 <= $4 )) || die "$1 must be an integer in $3..$4 (got '$2')"
}
CMD=start
if [[ ${1:-} == (stop|status) ]]; then
  CMD=$1
  (( $# <= 2 )) || die "usage: run.sh $CMD [port]"
  PORT=${2:-8300}
else
  PORT=${1:-8300} RUNNING=${2:-64} STEP=${3:-2048} KV=${4:-96} SNAPS=${5:-96} MAXCTX=${6:-131072}
fi
int_in port "$PORT" 1 65535
LOGS=${INFERD_LOGS:-$HOME/inference/logs}
STOP_WAIT=${STOP_WAIT:-60}
int_in STOP_WAIT "$STOP_WAIT" 1 3600

# Engine socket and pid files: an owner-only directory (Codex review R08), never a predictable path in shared /tmp.
RUN=${INFERD_RUN_DIR:-${${TMPDIR:-/tmp}%/}/inferd-$UID}
[[ -L $RUN ]] && die "$RUN is a symlink; refusing"
if [[ -e $RUN ]]; then
  [[ -d $RUN && -O $RUN ]] || die "$RUN is not a directory owned by $USER"
fi
SOCK=$RUN/inferd-$PORT.sock IPID=$RUN/inferd-$PORT.pid SPID=$RUN/serve-$PORT.pid
LEGACY=/tmp/inferd-$PORT.sock  # where run.sh put the socket before 2026-11-03

alive() { kill -0 "$1" 2>/dev/null }
stop_pid() {  # pid what: SIGTERM, wait up to STOP_WAIT s (inferd flushes the SSD tier on SIGTERM), then SIGKILL
  local pid=$1 i
  alive $pid || return 0
  [[ $(ps -o uid= -p $pid | tr -d ' ') == $UID ]] || die "$2 (pid $pid) is not ours; refusing to stop it"
  kill -TERM $pid 2>/dev/null
  for i in {1..$(( STOP_WAIT * 4 ))}; do alive $pid || return 0; sleep 0.25; done
  print -u2 "run.sh: $2 (pid $pid) did not stop in ${STOP_WAIT}s; SIGKILL"
  kill -KILL $pid 2>/dev/null
  for i in {1..20}; do alive $pid || return 0; sleep 0.25; done
  die "$2 (pid $pid) survived SIGKILL"
}
file_pid() {  # pidfile expected-command-substring -> the pid, if it is alive and still runs what we started
  [[ -f $1 ]] || return 1
  local pid=$(<$1)
  [[ $pid == <-> ]] && alive $pid && [[ $(ps -o command= -p $pid) == *$2* ]] && print -r -- $pid
}
holders() {  # pids listening on the port or holding the socket (current or legacy path)
  local p
  for p in ${(f)"$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null)"} \
           ${(f)"$( [[ -S $SOCK ]] && lsof -t "$SOCK" 2>/dev/null)"} \
           ${(f)"$( [[ -S $LEGACY ]] && lsof -t "$LEGACY" 2>/dev/null)"}; do
    [[ $p == <-> ]] && print -r -- $p
  done | sort -un
}
ours() {  # pid: our front end or engine for this port (by command line), never anything else
  local cmd=$(ps -o command= -p $1)
  [[ $cmd == *frontend/serve.py*--port\ ${PORT}(|\ *) || $cmd == *build/inferd\ *\ ($SOCK|$LEGACY)(|\ *) ]]
}
stop_all() {  # -> 0, after stopping our processes on PORT; a holder we do not recognise is reported (start: fatal)
  local pid n=0 foreign=0
  pid=$(file_pid "$SPID" "frontend/serve.py") && { stop_pid $pid "front end"; n=$((n + 1)); }
  rm -f -- "$SPID"
  pid=$(file_pid "$IPID" "$SOCK") && { stop_pid $pid "inferd"; n=$((n + 1)); }
  rm -f -- "$IPID"
  for pid in $(holders); do  # started before the pid files existed, or by an older run.sh (socket in /tmp)
    if ours $pid; then stop_pid $pid "pid $pid on port $PORT"; n=$((n + 1))
    else print -u2 -r -- "run.sh: port $PORT / its socket is held by pid $pid ($(ps -o command= -p $pid)); not ours, left alone"; foreign=1
    fi
  done
  for s in $SOCK $LEGACY; do  # stale sockets of ours
    [[ -S $s && -O $s && ! -L $s ]] && [[ -z $(lsof -t "$s" 2>/dev/null) ]] && rm -f -- "$s"
  done
  [[ $CMD == stop ]] && print "run.sh: stopped $n process(es) on port $PORT"
  (( foreign == 0 )) || [[ $CMD == stop ]] || die "port $PORT is in use by a process run.sh did not start"
  return 0
}

if [[ $CMD == stop ]]; then
  stop_all
  exit 0
fi
if [[ $CMD == status ]]; then
  spid=$(file_pid "$SPID" "frontend/serve.py") || spid=
  ipid=$(file_pid "$IPID" "$SOCK") || ipid=
  print -r -- "port      $PORT"
  print -r -- "front end ${spid:-not running}${spid:+  ($(ps -o command= -p $spid))}"
  print -r -- "inferd    ${ipid:-not running}${ipid:+  ($(ps -o command= -p $ipid))}"
  print -r -- "socket    $SOCK$([[ -S $SOCK ]] && print ' (present)' || print ' (absent)')"
  print -r -- "pid files $SPID $IPID"
  print -r -- "logs      $LOGS/inferd-$PORT.log $LOGS/serve-$PORT.log"
  for pid in $(holders); do
    [[ $pid == ($spid|$ipid) ]] || print -r -- "other     pid $pid holds the port or a socket: $(ps -o command= -p $pid)"
  done
  [[ -n $spid && -n $ipid ]] && exit 0 || exit 3
fi

int_in max_running "$RUNNING" 1 4096
int_in step_tokens "$STEP" 1 1048576
[[ $KV == <->(|.<->) ]] && (( KV <= 65536 )) || die "kv_gib must be a number in 0..65536 (got '$KV'; 0 = auto)"
int_in snapshots "$SNAPS" 0 1048576
int_in max_ctx "$MAXCTX" 256 16777216

[[ -n ${MODEL:-} ]] || die "set MODEL to the checkpoint directory (e.g. MODEL=~/models/Qwen3.8-27B-4bit)"
[[ -d $MODEL && -f $MODEL/config.json ]] || die "MODEL=$MODEL is not a checkpoint directory (no config.json)"
[[ -z ${DRAFT:-} || -f $DRAFT/config.json ]] || die "DRAFT=$DRAFT is not a drafter directory (no config.json)"
[[ -x build/inferd && -f build/core.metallib ]] || die "build/inferd or its metallib is missing: run ./build.sh"
PY=${PY:-python3}
"$PY" -c 'import fastapi, uvicorn, transformers, jinja2' 2>/dev/null ||
  die "PY=$PY lacks the front end's packages (pip install -r requirements.txt)"
HOST=${INFERD_HOST:-127.0.0.1}
mkdir -p -- "$LOGS" || die "cannot create the log directory $LOGS"
mkdir -p -m 700 -- "$RUN" || die "cannot create $RUN"
[[ -d $RUN && -O $RUN && ! -L $RUN ]] || die "$RUN is not a directory owned by $USER"
chmod 700 "$RUN" || die "cannot set $RUN to mode 700"
[[ $(stat -f %Lp "$RUN") == 700 ]] || die "$RUN is not mode 700 after chmod"
(( ${#SOCK} < 104 )) || die "socket path $SOCK is too long for a Unix socket; set INFERD_RUN_DIR to a shorter path"

stop_all
if [[ -e $SOCK || -L $SOCK ]]; then  # a stale socket would let the front end start before inferd listens
  [[ -S $SOCK && -O $SOCK && ! -L $SOCK ]] || die "$SOCK exists and is not a socket owned by $USER; refusing"
  rm -f -- "$SOCK" || die "cannot remove the stale socket $SOCK"
fi

INFERD_HOME=$PWD DRAFT=${DRAFT:-} nohup ./build/inferd "$MODEL" "$SOCK" "$RUNNING" "$STEP" "$KV" "$SNAPS" "$MAXCTX" \
  > "$LOGS/inferd-$PORT.log" 2>&1 < /dev/null &
print -r -- $! > "$IPID"
ipid=$!
for i in {1..720}; do
  grep -q "listening on" "$LOGS/inferd-$PORT.log" 2>/dev/null && break
  alive $ipid || { print -u2 'inferd died:'; tail -5 "$LOGS/inferd-$PORT.log"; rm -f -- "$IPID"; exit 1; }
  sleep 0.5
done
grep -q "listening on" "$LOGS/inferd-$PORT.log" || { stop_pid $ipid inferd; die "inferd did not start in 360 s"; }
[[ -S $SOCK && -O $SOCK ]] || { stop_pid $ipid inferd; die "inferd's socket $SOCK is missing or not ours"; }
[[ $(stat -f %Lp "$SOCK") == 600 ]] || print -u2 "run.sh: warning: $SOCK is not mode 600 (its directory is 700)"

nohup "$PY" -I frontend/serve.py --model "$MODEL" --socket "$SOCK" --host "$HOST" --port "$PORT" \
  > "$LOGS/serve-$PORT.log" 2>&1 < /dev/null &
print -r -- $! > "$SPID"
spid=$!
[[ $HOST == (0.0.0.0|::|\[::\]) ]] && CHECK=127.0.0.1 || CHECK=$HOST
[[ $CHECK == *:* ]] && CHECK="[$CHECK]"
health() {  # the key goes to curl on stdin, not on its command line
  if [[ -n ${INFERD_API_KEY:-} ]]; then
    print -r -- "header = \"Authorization: Bearer $INFERD_API_KEY\"" | curl -sf -K - "http://$CHECK:$PORT/health" >/dev/null
  else
    curl -sf "http://$CHECK:$PORT/health" >/dev/null
  fi
}
for i in {1..240}; do health && break; alive $spid || break; sleep 0.5; done
health || { print -u2 'front end failed:'; tail -5 "$LOGS/serve-$PORT.log"; stop_pid $spid "front end"
             stop_pid $ipid inferd; rm -f -- "$SPID" "$IPID"; exit 1; }
print "up on $HOST:$PORT (inferd pid $ipid, front end pid $spid, socket $SOCK)"
cat -- "$LOGS/inferd-$PORT.log"
