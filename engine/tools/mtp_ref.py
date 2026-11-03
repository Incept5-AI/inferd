# Reference for inferd's MTP drafter (Flash-Next, MTP=1): the mlx_vlm Qwen4-Exp MTP head's drafts after a prompt.
#   mtp_ref.py ref <model> <prompts.json> [block]   prompts: JSON list of token-id lists. Prints, per prompt, the
#       target's greedy next token (the bonus) and the drafter's first block - 1 drafts
#       (prefill_from_target_hidden + draft_block, greedy).
#   mtp_ref.py engine <socket> <prompts.json>      sends the prompts to a running inferd (MTP=1 INFERD_DEBUG=1, no
#       front end attached) with max_tokens 8; the first commit line per request in inferd's log holds its drafts.
import json, socket, sys

if sys.argv[1] == "engine":
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sys.argv[2])
    prompts = json.load(open(sys.argv[3]))
    f = s.makefile("rw")
    for i, ids in enumerate(prompts):  # one at a time, so each prompt's first cycle is its own step
        f.write(json.dumps({"id": i + 1, "tokens": ids, "max_tokens": 8, "temperature": 0}) + "\n")
        f.flush()
        out = []
        while True:
            ev = json.loads(f.readline())
            if "t" in ev:
                out += ev["t"]
            if ev.get("done"):
                break
        print(json.dumps({"id": i + 1, "tokens": out}))
    sys.exit(0)

import mlx.core as mx
from mlx_vlm import load

path, prompts = sys.argv[2], json.load(open(sys.argv[3]))
block = int(sys.argv[4]) if len(sys.argv) > 4 else 4
model, _ = load(path)


def load_mtp(path):
    # load_drafter(path) drops the checkpoint's per-tensor quantization (mixed 4/5/6/8-bit: keys "mtp.layers.0...."
    # while the drafter's modules are "layers.0....", so q_proj loads at the default 4 bits and fails). Remap the keys.
    from pathlib import Path
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
    from mlx_vlm.utils import get_model_and_args, load_model_weights
    src = json.load(open(Path(path) / "config.json"))
    sp = get_mtp_splitter(source_config=src)
    config, weights = sp.prepare(Path(path), source_config=src)
    for key in ("quantization", "quantization_config"):
        q = dict(src.get(key) or {})
        for k, v in list(q.items()):
            for pre in ("language_model.mtp.", "model.mtp.", "mtp."):
                if isinstance(v, dict) and k.startswith(pre):
                    q[k[len(pre):]] = v
        if q:
            config[key] = q
    arch, _ = get_model_and_args(config)
    m = arch.Model(arch.ModelConfig.from_dict(config))
    return load_model_weights(m, config, weights, Path(path))


drafter = load_mtp(path)
lm = model.language_model
for ids in prompts:
    out = lm(mx.array([ids]), return_hidden=True)
    hidden = out.hidden_states[0]
    bonus = int(mx.argmax(out.logits[0, -1]).item())
    drafter.reset(model)
    drafter.prefill_from_target_hidden(mx.array([ids]), hidden, bonus, None, greedy=True)
    d = drafter.draft_block(bonus, hidden[:, -1:], None, block, None, greedy=True)
    print(json.dumps({"n": len(ids), "bonus": bonus, "drafts": [int(x) for x in d.reshape(-1).tolist()]}), flush=True)
