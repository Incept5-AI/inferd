# tie_check.py MODEL SPEC_OUT [--max-gap G] [--ignore-seqs 0,..]
# For each spec/plain divergence in SPEC_OUT (spec_test's outputs: plain rows first, then spec rows; spec_test prints
# the path as "outputs: <path>", or set SPEC_OUT for it), report the reference (mlx_lm) logits of the two competing
# tokens at the first differing position, given the shared prefix. A divergence is a near-tie when that gap is at
# most --max-gap (default 0.5, docs/TESTING.md section 2.2); exit status 1 when any divergence on a sequence not in
# --ignore-seqs is wider, 0 otherwise (also when nothing diverged). tools/release_gate.sh runs it with --ignore-seqs 0:
# sequence 0 is the repetitive "quick brown fox" prompt, whose choice at one position swings by 1-5 logits with any
# change of precision or kernel shape (see the gate's header), so it is reported but not judged.
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_logits import load_model  # noqa: E402  (mlx_lm, or mlx-vlm's language model)


def first_diff(p, s):
    return next((i for i in range(min(len(p), len(s))) if p[i] != s[i]), None)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("spec_out", nargs="?", default=os.environ.get("SPEC_OUT"))
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--ignore-seqs", default="", help="comma-separated sequences reported but not judged")
    args = ap.parse_args(argv)
    if not args.spec_out:
        sys.exit("tie_check: give spec_test's output file (its 'outputs:' line) or set SPEC_OUT")
    ignore = {int(x) for x in args.ignore_seqs.split(",") if x.strip()}
    rows = [list(map(int, l.split())) for l in open(args.spec_out) if l.strip()]
    if not rows or len(rows) % 2:
        sys.exit(f"tie_check: {args.spec_out}: expected plain rows then spec rows, got {len(rows)} rows")
    n = len(rows) // 2
    todo = [(b, k) for b in range(n) if (k := first_diff(rows[b], rows[n + b])) is not None]
    if not todo:
        print(f"tie_check: no divergence in {n} sequences")
        return 0
    mx, model, _tok = load_model(args.model)
    bad = 0
    for b, k in todo:
        p, s = rows[b], rows[n + b]
        lg = np.array(model(mx.array([p[:k]]))[0, -1].astype(mx.float32))
        top = np.argsort(-lg)[:3]
        gap = abs(float(lg[p[k]]) - float(lg[s[k]]))
        if b in ignore:
            verdict = "ignored (not judged)"
        elif gap <= args.max_gap:
            verdict = f"near-tie (<= {args.max_gap})"
        else:
            verdict = f"FAIL: gap > {args.max_gap}"
            bad += 1
        print(f"seq {b} @ {k}: plain {p[k]} ({lg[p[k]]:.3f})  spec {s[k]} ({lg[s[k]]:.3f})  gap {gap:.3f}  "
              f"ref top3 {[(int(t), round(float(lg[t]), 3)) for t in top]}  {verdict}")
    judged = sum(b not in ignore for b, _ in todo)
    print(f"tie_check: {len(todo)} diverged, {judged} judged, {bad} above the {args.max_gap} gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
