// kv_check <model_dir> <tokens.txt> <out.bin> <tail N> [rows R] [prefill chunk]
//   Long-context accuracy (8-bit vs bf16 KV, W4A8, ...): each of the B sequences in tokens.txt ("B T" then B lines of
//   T ids) is prefilled up to T - N in chunks, then its last N tokens are fed teacher-forced in steps of R rows
//   (R = 1: the decode path; R = 5: the verify-attention path), keeping the last row's logits of every step.
//   Writes float32 [B, N / R, vocab]; tools/kv_ref.py compares it with mlx_lm at the same positions.
#include <chrono>
#include <fstream>
#include "../src/qwen35.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void run(Gpu& g, Qwen35& m, int T, int S) {
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  m.encode_step(cb, T, S);
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "step: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 5) { fprintf(stderr, "usage: see header\n"); return 1; }
  const int N = atoi(argv[4]), R = argc > 5 ? atoi(argv[5]) : 1, chunk = argc > 6 ? atoi(argv[6]) : 2048;
  if (N % R) { fprintf(stderr, "N must be a multiple of R\n"); return 1; }
  @autoreleasepool {
    Gpu g("build/core.metallib");
    double t0 = now();
    Checkpoint ck(argv[1]);
    std::ifstream f(argv[2]);
    int B, T;
    f >> B >> T;
    std::vector<std::vector<int>> toks(B, std::vector<int>(T));
    for (auto& r : toks) for (auto& v : r) f >> v;
    const int maxb = (T + KV_BLOCK - 1) / KV_BLOCK;
    Qwen35 m(g, ck, B * std::max(chunk, R), B, B, B * maxb, maxb);
    printf("loaded in %.1f s (KV %s)\n", now() - t0, m.kv4 ? "4-bit" : m.kv8 ? "8-bit" : "bf16");
    std::vector<std::vector<int>> blocks(B);
    for (int b = 0; b < B; b++) {
      m.reset_slot(b);
      for (int j = 0; j < maxb; j++) blocks[b].push_back(b * maxb + j);
    }
    auto step = [&](int t, int n) {
      std::vector<Qwen35::SeqWork> w;
      for (int b = 0; b < B; b++) w.push_back({b, &blocks[b], t, toks[b].data() + t, n});
      run(g, m, m.set_batch(w), B);
    };
    for (int t = 0; t < T - N; t += chunk) step(t, std::min(chunk, T - N - t));
    const size_t V = m.cfg.vocab, S = N / R;
    std::vector<float> out((size_t)B * S * V);
    for (size_t s = 0; s < S; s++) {
      step(T - N + (int)s * R, R);
      for (int b = 0; b < B; b++)
        for (size_t i = 0; i < V; i++) {
          uint32_t u = (uint32_t)m.logits.ptr<uint16_t>()[(size_t)b * V + i] << 16;
          memcpy(&out[((size_t)b * S + s) * V + i], &u, 4);
        }
    }
    std::ofstream o(argv[3], std::ios::binary);
    o.write((const char*)out.data(), out.size() * 4);
    printf("wrote %d x %zu x %zu logits (T %d, tail %d, rows %d) in %.1f s\n", B, S, V, T, N, R, now() - t0);
  }
}
