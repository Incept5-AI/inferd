// qsa_stats <model_dir> <tokens.txt> <L> [chunk 2048]
//   Flash-Next prefill of real text (the first sequence of tokens.txt, L tokens, in chunks): GPU ms per chunk and
//   tok/s (decode_test prefill uses synthetic ids, whose MoE routing is unrealistic). Then, for the last chunk's
//   sparse rows, the QSA selections of the last attention layer (qsa_sel): the union size of runs of 1/4/16/64
//   consecutive rows, counted in 4-token groups and in 32-token KV blocks (how much a larger query tile would read).
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include "../src/qwen35.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 4) { fprintf(stderr, "usage: see header\n"); return 1; }
  const int L = atoi(argv[3]), chunk = argc > 4 ? atoi(argv[4]) : 2048;
  @autoreleasepool {
    Gpu g("build/core.metallib");
    Checkpoint ck(argv[1]);
    std::ifstream f(argv[2]);
    int nseq, T;
    f >> nseq >> T;
    if (T < L) { fprintf(stderr, "tokens.txt has %d tokens per sequence, need %d\n", T, L); return 1; }
    std::vector<int> toks(T);
    for (auto& v : toks) f >> v;
    const int maxb = (L + KV_BLOCK - 1) / KV_BLOCK;
    Qwen35 m(g, ck, chunk, 1, 1, maxb, maxb);
    std::vector<int> blocks;
    for (int j = 0; j < maxb; j++) blocks.push_back(j);
    for (int rep = 0; rep < 2; rep++) {  // the first pass compiles pipelines
      m.reset_slot(0);
      double gsum = 0, w0 = now();
      for (int t = 0; t < L; t += chunk) {
        const int n = std::min(chunk, L - t);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        const int Tn = m.set_batch({{0, &blocks, t, toks.data() + t, n}});
        m.encode_step(cb, Tn, 1);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "step: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        const double gt = cb.GPUEndTime - cb.GPUStartTime;
        gsum += gt;
        if (rep == 1 && getenv("QSA_STATS_STEPS")) printf("chunk at %6d: %7.1f ms\n", t, gt * 1e3);
      }
      if (rep == 1) printf("prefill L=%d chunk %d: GPU %.3f s = %.0f tok/s (wall %.3f s)\n", L, chunk, gsum, L / gsum, now() - w0);
    }
    if (Enc::prof) {  // PROF_KERNELS=1: GPU ms per kernel over both passes (each dispatch alone: overstates small ones)
      std::vector<std::pair<double, std::string>> v;
      double tot = 0;
      for (auto& [k, ms] : *Enc::prof) { v.push_back({ms / 2, k}); tot += ms / 2; }
      std::sort(v.rbegin(), v.rend());
      printf("PROF per prefill: %.1f ms\n", tot);
      for (size_t i = 0; i < v.size() && i < 40; i++) printf("%9.2f ms %5.1f%%  %s\n", v[i].first, 100 * v[i].first / tot, v[i].second.c_str());
    }
    if (!m.qsa_on || !m.qsa_nsp) return 0;
    const int K = m.cfg.qsa_topk, n = m.qsa_nsp;
    const int* sel = m.qsa_sel.ptr<int>();
    const int* rw = m.qsa_rows.ptr<int>();  // sparse row i -> step row (sel is indexed by step row)
    printf("last chunk: %d sparse rows, top-%d groups each\n", n, K);
    for (int tile : {1, 4, 16, 64}) {
      double sg = 0, sb = 0;
      int cnt = 0;
      for (int r0 = 0; r0 + tile <= n; r0 += tile) {
        std::set<int> gs, bs;
        for (int r = r0; r < r0 + tile; r++)
          for (int k = 0; k < K; k++) { gs.insert(sel[(size_t)rw[r] * K + k]); bs.insert(sel[(size_t)rw[r] * K + k] * m.cfg.qsa_ratio / KV_BLOCK); }
        sg += gs.size();
        sb += bs.size();
        cnt++;
      }
      printf("tile %2d rows: union %7.1f groups (%.2fx one row), %6.1f KV blocks of 32\n", tile, sg / cnt, sg / cnt / K, sb / cnt);
    }
  }
  return 0;
}
