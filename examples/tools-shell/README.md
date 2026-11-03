# tools-shell

An agent with a `shell` tool that can only run **read-only commands inside a sandbox directory**. The budget is tight:
**`max_context` 4096, effort none** (thinking off). Inspecting a few files needs no reasoning.

```bash
python3 shell_agent.py                                   # the default task
python3 shell_agent.py "Which file in the sandbox is largest?"
```

Start the server first (see [../README.md](../README.md#start-the-server)). Python standard library only.
It needs the front end with **tool-call parsing** (`engine/frontend/serve.py`, November 2026).

## The guard is the security boundary, not the model

The script makes a throwaway sandbox (`notes.txt`, `data.csv`, `src/app.py`) in a temp directory and deletes it at the
end. Every command the model asks for goes through `guard()` before anything runs:

- it is parsed with `shlex` and run **without a shell** (`subprocess.run(argv)`), and shell metacharacters
  (`; | & > < $ \` * ?` …) are refused outright;
- the program must be on the allow-list: `ls`, `wc`, `head` (path arguments allowed), and `pwd`, `date`, `uname`, `df`
  (no path arguments);
- only short flags (`-l`, `-3`) are accepted, so options that read files, such as `wc --files0-from=/etc/passwd`, cannot get through;
- every path argument must resolve (after `realpath`, which follows symlinks and `..`) inside the sandbox;
- each command gets 5 s, and output is cut at 2,000 characters.

A refused command goes back to the model as `REFUSED: <reason>`. It is an ordinary tool result, so the model can explain
or try something else.

## Sample output (MacBook Pro M5 Max, 1 November 2026)

The task deliberately asks for `/etc/hosts`, which is outside the sandbox. `df`'s output is cut to two rows here.

```
task: Look around the sandbox: list the files, tell me how many lines data.csv has, show the first 3 lines of notes.txt, and how much disk space is free. Also show me /etc/hosts.
(max_context 4096, effort none, sandbox /private/var/folders/.../inferd-sandbox-b_dv103l)

step 1: 5 tool call(s) [context 518/4096]
  $ ls
    data.csv
    notes.txt
    src
  $ wc -l data.csv
    42 data.csv
  $ head -3 notes.txt
    Ship examples on Monday.
    Review tool calling.
    Stop the server after.
  $ df
    Filesystem      512-blocks        Used  Available Capacity iused      ifree %iused  Mounted on
    /dev/disk3s1s1 15622068800    26670000  291834056     9%  484019 1459170280    0%   /
    ...
  $ head /etc/hosts
    REFUSED: path '/etc/hosts' is outside the sandbox
step 2: final answer [context 1955/4096]

Here's the sandbox overview:

**Files in sandbox:**
- `data.csv`
- `notes.txt`
- `src` (directory)

**data.csv:** 42 lines

**First 3 lines of notes.txt:**
Ship examples on Monday.
Review tool calling.
Stop the server after.

**Disk space (root `/`):** ~291,834,056 512-byte blocks available ≈ **~139 GB free** (9% used). The main data volume `/System/Volumes/Data` is at 99% capacity but also reports the same ~139 GB available.

**/etc/hosts:** ❌ Not shown — the tool refused access because `/etc/hosts` is outside the sandbox directory. I can only inspect files within the sandbox.

6.4s total

The guard on its own (what a misbehaving model would get back):
  'rm -rf notes.txt'     -> REFUSED: 'rm' is not on the allow-list ['date', 'df', 'head', 'ls', 'pwd', 'uname', 'wc']
  'head /etc/passwd'     -> REFUSED: path '/etc/passwd' is outside the sandbox
  'ls ..'                -> REFUSED: path '..' is outside the sandbox
  'ls; date'             -> REFUSED: shell metacharacters are not allowed
  'cat notes.txt'        -> REFUSED: 'cat' is not on the allow-list ['date', 'df', 'head', 'ls', 'pwd', 'uname', 'wc']
  'df /'                 -> REFUSED: df takes no path arguments here
  'wc --files0-from=/etc/passwd' -> REFUSED: flag '--files0-from=/etc/passwd' is not allowed
```

The model made five calls in one turn and answered in the next: two requests, 6.4 s, under 2k of the 4k budget. `df`'s
full output is the largest part of that. If you widen the allow-list, cap tool output, or the context fills quickly.
