# The logits gate must fail on bad logits (Codex review 2026-11-04, R04). No model: ref_logits.py compare-bin on
# fixtures. numpy only.
#   python3 tools/test_ref_logits.py
import os, subprocess, sys, tempfile, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_logits as R  # noqa: E402

V, B = 4096, 4


def good_ref(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 3, (B, V)).astype(np.float32)
    ref[:, 7] += 12  # a clear top-1
    return ref


def bf16(x):  # what the engine's logits are: bf16, round to nearest even
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


class Gate(unittest.TestCase):
    def run_bin(self, ref, eng, *opts, T=3):
        d = tempfile.mkdtemp()
        toks, rb, eb = (os.path.join(d, n) for n in ("toks.txt", "ref.bin", "eng.bin"))
        with open(toks, "w") as f:
            f.write(f"{B} {T}\n" + (" ".join(["1"] * T) + "\n") * B)
        np.asarray(ref, np.float32).tofile(rb)
        np.asarray(eng, np.float32).tofile(eb)
        p = subprocess.run([sys.executable, os.path.join(HERE, "ref_logits.py"), "compare-bin", toks, rb, eb, *opts],
                           capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_bf16_rounding_passes(self):
        ref = good_ref()
        code, out = self.run_bin(ref, bf16(ref))
        self.assertEqual(code, 0, out)

    def test_same_argmax_corrupted_fails(self):
        # The review's point: a distribution can be badly wrong and keep its argmax; top-1 alone passed it.
        ref = good_ref()
        eng = ref + np.random.default_rng(1).normal(0, 2.0, ref.shape).astype(np.float32)
        eng[:, 7] = ref[:, 7] + 20  # top-1 preserved
        self.assertTrue((eng.argmax(1) == ref.argmax(1)).all())
        code, out = self.run_bin(ref, eng)
        self.assertEqual(code, 1, out)
        self.assertIn("cos <", out)

    def test_non_finite_fails(self):
        ref = good_ref()
        for bad in (np.nan, np.inf, -np.inf):
            eng = ref.copy()
            eng[2, 100] = bad
            code, out = self.run_bin(ref, eng)
            self.assertEqual(code, 1, out)
            self.assertIn("non-finite", out)

    def test_top1_diff_fails(self):
        ref = good_ref()
        eng = ref.copy()
        eng[1, 8] = eng[1, 7] + 5
        code, out = self.run_bin(ref, eng)
        self.assertEqual(code, 1, out)
        self.assertIn("DIFF", out)

    def test_near_tie_is_explicit(self):
        ref = good_ref()
        ref[3, 9] = ref[3, 7] - R.bf16_step(ref[3, 7]) / 2  # within one bf16 step
        eng = ref.copy()
        eng[3, 9], eng[3, 7] = ref[3, 7], ref[3, 9]  # the engine flips the near-tie
        code, out = self.run_bin(ref, eng)
        self.assertEqual(code, 0, out)
        self.assertIn("TIE", out)
        ref[3, 9] = ref[3, 7] - 4 * R.bf16_step(ref[3, 7])  # not a tie
        eng[3, 9], eng[3, 7] = ref[3, 7], ref[3, 9]
        code, out = self.run_bin(ref, eng)
        self.assertEqual(code, 1, out)

    def test_seq0_allowance_only_for_seq0(self):
        ref = good_ref()
        eng = ref.copy()
        noise = np.random.default_rng(2).normal(0, 0.3, V).astype(np.float32)
        noise[7] = 1.0
        eng[0] += noise  # cos ~0.995: inside seq 0's allowance (0.99), outside the normal floor (0.999)
        c0 = R.check_row(0, ref[0], eng[0], 0.999, 0.99, 0.01, 0.05)
        c1 = R.check_row(1, ref[0], eng[0], 0.999, 0.99, 0.01, 0.05)
        self.assertTrue(c0[0], c0[1])
        self.assertFalse(c1[0], c1[1])

    def test_long_prompt_floor(self):  # T > 48: cos >= 0.995 / KL <= 0.03 (the Studio's 2,048-token measurements)
        ref = good_ref()
        rng = np.random.default_rng(3)
        for scale, want in ((0.2, 0), (0.6, 1)):  # cos ~0.998 passes; ~0.98 fails
            eng = ref + rng.normal(0, scale, ref.shape).astype(np.float32)
            eng[:, 7] = ref[:, 7]
            code, out = self.run_bin(ref, eng, T=2048)
            self.assertEqual(code, want, out)
            code_short, _ = self.run_bin(ref, eng, T=24)
            self.assertEqual(code_short, 1, "the short-prompt floor (0.999) is stricter")

    def test_shape_mismatch_fails(self):
        ok, line = R.check_row(1, np.zeros(10), np.zeros(9), 0.999, 0.99, 0.01, 0.05)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main(verbosity=1)
