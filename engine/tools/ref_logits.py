# Reference for decode_test: makes a token file, runs mlx_lm on each sequence, and compares logits.
#   ref_logits.py make <model> <tokens.txt> <B> <T>     B prompts of T tokens: sequence 0 is the repetitive "quick brown
#                                                       fox" text, the others real text (CORPUS: the repository's docs
#                                                       and the scheduler's source, a different file each; T > 48)
#   ref_logits.py compare <model> <tokens.txt> <engine_logits.bin> [options]
#   ref_logits.py compare-bin <tokens.txt> <ref_logits.bin> <engine_logits.bin> [options]   (no model: fixtures)
# compare options (Codex review 2026-11-04, R04: it used to fail only on a top-1 difference):
#   --cos-min C       cosine floor for normal sequences (1..B-1): 0.999 for short prompts (T <= 48), 0.995 for long
#   --cos-min-seq0 C  floor for sequence 0, the deliberately repetitive prompt: 0.99 (docs/TESTING.md: ~0.996 at short
#                     chunks since day 1, as the reference's prefill path differs; 0.9948-0.9979 at 2,048 tokens)
#   --kl-max K        KL(ref || engine) ceiling for normal sequences: 0.01 short, 0.03 long (seq 0: 0.05, --kl-max-seq0)
# Long-prompt floors are from the Studio (27B 4-bit, macOS 27.0.1, 2026-11-04, 4 x 2,048 at chunks 1/24/256/1536,
# 8-bit and bf16 KV alike): real-text sequences measured cos 0.99719-0.99963 and KL <= 0.0148; the code sequence (the
# scheduler source) is the lowest at every chunk. The gap grows with length (docs/TESTING.md: ~0.998 at ~3k tokens).
# Every sequence must have finite logits on both sides and the same top-1, except an explicit near-tie: the engine's
# top-1 differs but its reference logit is within one bf16 rounding step of the reference's top-1 (the engine's
# logits are bf16), reported as "TIE". Exit status 1 on any failure.
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPEAT = "The quick brown fox jumps over the lazy dog. "
SHORT = [REPEAT * 8,
         "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n" * 4,
         "In 1905 Albert Einstein published four papers that changed physics: " * 6,
         "Paris est la capitale de la France et la plus grande ville du pays. " * 6]


CORPUS = ("docs/ARCHITECTURE.md", "docs/RESULTS.md", "engine/src/scheduler.h", "docs/EXPERIMENTS.md")


def corpus():
    """Real long texts shipped with the repository, fixed: prose (docs) and code (the scheduler). Run from a checkout."""
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    files = [os.path.join(root, f) for f in CORPUS]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f"ref_logits: long prompts need the repository's texts; missing {', '.join(missing)}")
    return files


def make_ids(tok, B, T):
    """B prompts of exactly T tokens. T <= 48 keeps the historical short prompts (results stay comparable)."""
    out = []
    long_texts = corpus() if T > 48 else []
    for b in range(B):
        if T <= 48:
            ids = tok.encode(SHORT[b % len(SHORT)])[b // len(SHORT):][:T]
        elif b == 0:  # the repetitive one, as long as asked
            ids = tok.encode(REPEAT * (T // 8 + 16))[:T]
        else:
            text = open(long_texts[(b - 1) % len(long_texts)], encoding="utf-8").read()
            ids = tok.encode(text)[(b - 1) // len(long_texts) * T:][:T]
        assert len(ids) == T, (b, len(ids), T)
        out.append(ids)
    return out


def bf16_step(x):
    """One bf16 rounding step (8 significant bits) at magnitude |x|."""
    x = abs(float(x))
    return 2.0 ** (np.floor(np.log2(x)) - 7) if x > 0 else 2.0 ** -133


def check_row(b, ref, e, cos_min, cos_min0, kl_max, kl_max0):
    """-> (ok, report line) for sequence b's reference and engine logits."""
    ref, e = np.asarray(ref, np.float64), np.asarray(e, np.float64)
    if ref.shape != e.shape:
        return False, f"seq {b}: FAIL shape {e.shape} vs reference {ref.shape}"
    bad_e, bad_r = int((~np.isfinite(e)).sum()), int((~np.isfinite(ref)).sum())
    if bad_e or bad_r:
        return False, f"seq {b}: FAIL non-finite logits (engine {bad_e}, reference {bad_r})"
    ne, nr = np.linalg.norm(e), np.linalg.norm(ref)
    cos = float(np.dot(ref, e) / nr / ne) if ne > 0 and nr > 0 else 0.0
    lr = ref - ref.max(); pr = np.exp(lr) / np.exp(lr).sum()
    le = e - e.max(); pe = np.exp(le) / np.exp(le).sum()
    kl = float(np.sum(pr * (np.log(pr + 1e-12) - np.log(pe + 1e-12))))
    top_r, top_e = np.argsort(-ref, kind="stable")[:5], np.argsort(-e, kind="stable")[:5]
    margin = float(ref[top_r[0]] - ref[top_e[0]])
    if top_r[0] == top_e[0]:
        top = "ok"
    elif margin <= bf16_step(ref[top_r[0]]):
        top = f"TIE (reference margin {margin:.4f} <= one bf16 step)"
    else:
        top = f"DIFF (reference margin {margin:.4f})"
    floor, kmax = (cos_min0, kl_max0) if b == 0 else (cos_min, kl_max)
    fails = []
    if top.startswith("DIFF"):
        fails.append("top-1")
    if cos < floor:
        fails.append(f"cos < {floor}")
    if kl > kmax:
        fails.append(f"KL > {kmax}")
    line = (f"seq {b}: cos {cos:.5f}  max|diff| {np.abs(ref - e).max():.3f}  KL {kl:.5f}  top1 {top}  "
            f"top5 overlap {len(set(top_r) & set(top_e))}/5" + (f"  FAIL: {', '.join(fails)}" if fails else ""))
    return not fails, line


def read_tokens(path):
    lines = open(path).read().split("\n")
    B, T = map(int, lines[0].split())
    return B, T, [list(map(int, l.split())) for l in lines[1:1 + B]]


def compare(refs, eng, args, decode=None):
    bad = 0
    for b, ref in enumerate(refs):
        ok, line = check_row(b, ref, eng[b], args.cos_min, args.cos_min_seq0, args.kl_max, args.kl_max_seq0)
        if decode is not None:
            line += f"  ({decode(int(np.argmax(ref)))!r})"
        print(line)
        bad += not ok
    print(f"compare: {len(refs) - bad}/{len(refs)} sequences pass")
    return 1 if bad else 0


def load_model(path):
    import mlx.core as mx
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
    return mx, model, tok


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "compare", "compare-bin"))
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c", nargs="?")
    ap.add_argument("d", nargs="?")
    ap.add_argument("--cos-min", type=float, default=None)
    ap.add_argument("--cos-min-seq0", type=float, default=0.99)
    ap.add_argument("--kl-max", type=float, default=None)
    ap.add_argument("--kl-max-seq0", type=float, default=0.05)
    args = ap.parse_args(argv)
    if args.cmd != "make":  # length-dependent defaults (see the header)
        T = read_tokens(args.a if args.cmd == "compare-bin" else args.b)[1]
        long = T > 48
        args.cos_min = args.cos_min if args.cos_min is not None else 0.995 if long else 0.999
        args.kl_max = args.kl_max if args.kl_max is not None else 0.03 if long else 0.01
        print(f"compare: T {T}: cos >= {args.cos_min} / KL <= {args.kl_max} (seq 0: {args.cos_min_seq0} / "
              f"{args.kl_max_seq0})")
    if args.cmd == "make":
        _, _, tok = load_model(args.a)
        B, T = int(args.c), int(args.d)
        with open(args.b, "w") as f:
            f.write(f"{B} {T}\n")
            for ids in make_ids(tok, B, T):
                f.write(" ".join(map(str, ids)) + "\n")
        return 0
    if args.cmd == "compare-bin":  # tokens, reference logits, engine logits (float32 [B, V] each)
        B, _, _ = read_tokens(args.a)
        ref = np.fromfile(args.b, dtype=np.float32).reshape(B, -1)
        eng = np.fromfile(args.c, dtype=np.float32).reshape(B, -1)
        return compare(list(ref), eng, args)
    mx, model, tok = load_model(args.a)
    B, _, seqs = read_tokens(args.b)
    eng = np.fromfile(args.c, dtype=np.float32).reshape(B, -1)
    refs = [np.array(model(mx.array([s]))[0, -1].astype(mx.float32)) for s in seqs]
    return compare(refs, eng, args, decode=lambda t: tok.decode([t]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
