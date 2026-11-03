// gap_check <model_dir> <cases.txt> [chunk 2048]
//   Exactness of speculative decoding at its first divergence from plain greedy decoding (engine-pf, 2026-11-03).
//   cases.txt: one case per line, "<tag> <tok_a> <tok_b> <n> <n token ids>": the prefix up to (not including) the
//   divergent position, the token plain decoding committed (a) and the one speculation committed (b). Each prefix is
//   prefilled alone (chunks of `chunk`) and the target's last-row logits give: logit(a) - logit(b), the top-1 margin
//   (top-1 minus top-2), and the ranks of a and b. A divergence is a near-tie when |logit(a) - logit(b)| is small; a
//   bug shows as b far below the top-1. One model load for every case.
#include <algorithm>
#include <fstream>
#include <sstream>
#include "../src/qwen35.h"

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) { fprintf(stderr, "usage: see header\n"); return 1; }
  const int chunk = argc > 3 ? atoi(argv[3]) : 2048;
  @autoreleasepool {
    Gpu g("build/core.metallib");
    Checkpoint ck(argv[1]);
    std::ifstream f(argv[2]);
    struct Case { std::string tag; int a, b; std::vector<int> t; };
    std::vector<Case> cs;
    int maxT = 0;
    for (std::string line; std::getline(f, line);) {
      if (line.empty()) continue;
      std::istringstream is(line);
      Case c;
      int n;
      is >> c.tag >> c.a >> c.b >> n;
      c.t.resize(n);
      for (auto& v : c.t) is >> v;
      maxT = std::max(maxT, n);
      cs.push_back(std::move(c));
    }
    const int maxb = (maxT + KV_BLOCK - 1) / KV_BLOCK;
    Qwen35 m(g, ck, chunk, 1, 1, maxb, maxb);
    std::vector<int> blocks;
    for (int j = 0; j < maxb; j++) blocks.push_back(j);
    const int V = m.cfg.vocab;
    printf("%-14s %8s %8s %9s %9s %9s %6s %6s\n", "case", "tok_a", "tok_b", "la - lb", "top1 gap", "pos", "rank_a", "rank_b");
    for (auto& c : cs) {
      m.reset_slot(0);
      const int T = (int)c.t.size();
      for (int t = 0; t < T; t += chunk) {
        const int n = std::min(chunk, T - t);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        const int Tn = m.set_batch({{0, &blocks, t, c.t.data() + t, n}});
        m.encode_step(cb, Tn, 1);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "step: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
      }
      std::vector<float> l(V);
      for (int i = 0; i < V; i++) {
        uint32_t u = (uint32_t)m.logits.ptr<uint16_t>()[i] << 16;
        memcpy(&l[i], &u, 4);
      }
      std::vector<float> s(l);
      std::nth_element(s.begin(), s.begin() + 1, s.end(), std::greater<float>());
      const float t1 = std::max(s[0], s[1]), t2 = std::min(s[0], s[1]);
      int ra = 0, rb = 0;
      for (float v : l) { ra += v > l[c.a]; rb += v > l[c.b]; }
      printf("%-14s %8d %8d %9.3f %9.3f %9d %6d %6d\n", c.tag.c_str(), c.a, c.b, l[c.a] - l[c.b], t1 - t2, T, ra + 1, rb + 1);
    }
  }
  return 0;
}
