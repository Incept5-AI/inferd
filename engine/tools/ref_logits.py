# Reference for decode_test: makes a token file, runs mlx_lm on each sequence, and compares logits.
#   ref_logits.py make <model> <tokens.txt> <B> <T>     random prompts drawn from real text tokens
#   ref_logits.py compare <model> <tokens.txt> <engine_logits.bin>
import sys
import numpy as np
import mlx.core as mx
cmd, path, tok_file = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from mlx_lm import load
    model, tok = load(path)
except Exception:  # architectures only mlx-vlm has (e.g. Flash-Next's qwen4_exp): its language model
    from mlx_vlm import load as vlm_load
    _m, _proc = vlm_load(path)
    tok = getattr(_proc, "tokenizer", _proc)
    _lm = _m.language_model
    def model(ids):
        out = _lm(ids)
        return getattr(out, "logits", out)
if cmd == "make":
    B, T = int(sys.argv[4]), int(sys.argv[5])
    texts = ["The quick brown fox jumps over the lazy dog. " * 8,
             "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n" * 4,
             "In 1905 Albert Einstein published four papers that changed physics: " * 6,
             "Paris est la capitale de la France et la plus grande ville du pays. " * 6]
    with open(tok_file, "w") as f:
        f.write(f"{B} {T}\n")
        for b in range(B):
            ids = tok.encode(texts[b % len(texts)])[b // len(texts):][:T]
            assert len(ids) == T, len(ids)
            f.write(" ".join(map(str, ids)) + "\n")
    sys.exit(0)

lines = open(tok_file).read().split("\n")
B, T = map(int, lines[0].split())
seqs = [list(map(int, l.split())) for l in lines[1:1 + B]]
eng = np.fromfile(sys.argv[4], dtype=np.float32).reshape(B, -1)
bad = 0  # exit status: top-1 must match everywhere (Codex review: a failed check used to exit 0)
for b, s in enumerate(seqs):
    ref = np.array(model(mx.array([s]))[0, -1].astype(mx.float32))
    e = eng[b]
    cos = float(np.dot(ref, e) / np.linalg.norm(ref) / np.linalg.norm(e))
    top_r, top_e = np.argsort(-ref)[:5], np.argsort(-e)[:5]
    lr = ref - ref.max(); pr = np.exp(lr) / np.exp(lr).sum()
    le = e - e.max(); pe = np.exp(le) / np.exp(le).sum()
    kl = float(np.sum(pr * (np.log(pr + 1e-12) - np.log(pe + 1e-12))))
    print(f"seq {b}: cos {cos:.5f}  max|diff| {np.abs(ref - e).max():.3f}  KL {kl:.5f}  "
          f"top1 {'ok' if top_r[0] == top_e[0] else 'DIFF'} ({tok.decode([int(top_r[0])])!r})  top5 overlap {len(set(top_r) & set(top_e))}/5")
    bad += top_r[0] != top_e[0]
sys.exit(1 if bad else 0)
