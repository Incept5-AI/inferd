#!/usr/bin/env python3
"""An agent with a `shell` tool restricted to read-only commands inside a sandbox directory.
Tight budget: max_context 4096, reasoning effort none (thinking off).

    python3 shell_agent.py ["your task"]     # server on http://127.0.0.1:8300 (INFERD_URL to change)
Python standard library only. Needs inferd's front end with tool-call parsing (engine/frontend/serve.py, Nov 2026).

The guard, not the model, is the security boundary: commands are parsed with shlex (no shell), the program must be on
the allow-list, shell metacharacters are refused, and every path argument must resolve inside the sandbox.
"""
import json, os, re, shlex, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")
MAX_CONTEXT, MAX_STEPS = 4096, 6
TASK = sys.argv[1] if len(sys.argv) > 1 else (
    "Look around the sandbox: list the files, tell me how many lines data.csv has, show the first 3 lines of "
    "notes.txt, and how much disk space is free. Also show me /etc/hosts.")

ALLOW = {"ls": True, "wc": True, "head": True, "pwd": False, "date": False, "uname": False, "df": False}  # paths ok?
FORBIDDEN = set(";|&><`$(){}\n\\*?")

# A throwaway sandbox with a few files to look at.
SANDBOX = os.path.realpath(tempfile.mkdtemp(prefix="inferd-sandbox-"))
os.makedirs(f"{SANDBOX}/src")
open(f"{SANDBOX}/notes.txt", "w").write("Ship examples on Monday.\nReview tool calling.\nStop the server after.\n"
                                        "Nothing else.\n")
open(f"{SANDBOX}/data.csv", "w").write("id,name,score\n" + "".join(f"{i},agent-{i},{i * 7 % 100}\n" for i in range(41)))
open(f"{SANDBOX}/src/app.py", "w").write("print('hello from the sandbox')\n")


def guard(command):
    """-> argv if the command is allowed, else raises ValueError with the reason."""
    if any(c in FORBIDDEN for c in command):
        raise ValueError("shell metacharacters are not allowed")
    argv = shlex.split(command)
    if not argv or argv[0] not in ALLOW:
        raise ValueError(f"'{argv[0] if argv else ''}' is not on the allow-list {sorted(ALLOW)}")
    for a in argv[1:]:
        if a.startswith("-"):
            if not re.fullmatch(r"-[A-Za-z0-9]+", a):  # short flags only: no --files0-from=/etc/passwd
                raise ValueError(f"flag '{a}' is not allowed")
            continue
        if not ALLOW[argv[0]]:
            raise ValueError(f"{argv[0]} takes no path arguments here")
        p = os.path.realpath(os.path.join(SANDBOX, a))
        if p != SANDBOX and not p.startswith(SANDBOX + os.sep):
            raise ValueError(f"path '{a}' is outside the sandbox")
    return argv


def shell(command):
    try:
        argv = guard(command)
    except ValueError as e:
        return f"REFUSED: {e}"
    try:
        p = subprocess.run(argv, cwd=SANDBOX, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "ERROR: timed out"
    return ((p.stdout + p.stderr).replace(SANDBOX, ".") or f"(exit {p.returncode}, no output)")[:2000]


TOOLS = [{"type": "function", "function": {
    "name": "shell",
    "description": "Run one read-only command in the sandbox directory. Allowed: ls, wc, head (paths relative to the "
                   "sandbox), pwd, date, uname, df. No pipes, redirects or other shell syntax.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def chat(messages):
    body = {"model": "qwen3.8-27b", "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 1024,
            "max_context": MAX_CONTEXT, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {json.load(e)['error']['message']}")


messages = [{"role": "system", "content": "You inspect a sandbox directory with the shell tool. Run one command per "
                                          "call; you may make several calls at once. Then answer briefly."},
            {"role": "user", "content": TASK}]
print(f"task: {TASK}\n(max_context {MAX_CONTEXT}, effort none, sandbox {SANDBOX})\n")
t0 = time.time()
for step in range(1, MAX_STEPS + 1):
    r = chat(messages)
    msg, usage = r["choices"][0]["message"], r["usage"]
    if not msg.get("tool_calls"):
        print(f"step {step}: final answer [context {usage['total_tokens']}/{MAX_CONTEXT}]\n\n{msg['content']}")
        break
    print(f"step {step}: {len(msg['tool_calls'])} tool call(s) [context {usage['total_tokens']}/{MAX_CONTEXT}]")
    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
    for call in msg["tool_calls"]:
        cmd = json.loads(call["function"]["arguments"]).get("command", "")
        out = shell(cmd)
        print(f"  $ {cmd}\n    " + out.strip().replace("\n", "\n    "))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
else:
    print(f"stopped after {MAX_STEPS} steps")
print(f"\n{time.time() - t0:.1f}s total")
shutil.rmtree(SANDBOX)

print("\nThe guard on its own (what a misbehaving model would get back):")
for bad in ["rm -rf notes.txt", "head /etc/passwd", "ls ..", "ls; date", "cat notes.txt", "df /",
            "wc --files0-from=/etc/passwd"]:
    print(f"  {bad!r:22} -> {shell(bad)}")
