// qsa_bench <model_dir> <tokens.txt> <L> <B list> [rows R]
//   Step timing on a real long context (Flash-Next QSA: the selection's cost depends on real index scores, which
//   decode_test bench's empty cache cannot give). Prefills the first sequence of tokens.txt up to L - R, then B
//   sequences share that prefix (each with a private copy of the blocks it writes and of the recurrent state) and
//   feed R rows ending at L - 1, as decode (R = 1) or a verify block (R = 4) does. Median of 10 steps.
//   BENCH_VARIANTS="QSA_DENSE=0;QSA_DENSE=1" times several knob settings on one load; PROF_KERNELS=1 lists GPU ms
//   per kernel (each dispatch alone: overstates small kernels).
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include "../src/qwen35.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double run(Gpu& g, Qwen35& m, int T, int S) {
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  m.encode_step(cb, T, S);
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "step: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
  return cb.GPUEndTime - cb.GPUStartTime;
}

// Copy KV block src to dst in every KV plane (all layers' K, V and scales, the QSA index planes included).
static void copy_block(Gpu& g, Qwen35& m, int src, int dst) {
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
  const size_t ns = m.kv_ns();
  for (size_t l = 0; l < m.kc.size(); l++) {
    const int H = m.cfg.hkv;
    const size_t sb = (size_t)m.cfg.dh * m.kv_elem();  // every pool in kc/vc has the target's geometry here
    for (int h = 0; h < H; h++) {
      const size_t s0 = h * ns + (size_t)src * KV_BLOCK, d0 = h * ns + (size_t)dst * KV_BLOCK;
      for (auto* v : {&m.kc, &m.vc})
        [b copyFromBuffer:(*v)[l].buf sourceOffset:(*v)[l].off + s0 * sb toBuffer:(*v)[l].buf
            destinationOffset:(*v)[l].off + d0 * sb size:KV_BLOCK * sb];
      if (m.kv8)
        for (auto* v : {&m.ks, &m.vs})
          [b copyFromBuffer:(*v)[l].buf sourceOffset:(*v)[l].off + s0 * 4 toBuffer:(*v)[l].buf
              destinationOffset:(*v)[l].off + d0 * 4 size:KV_BLOCK * 4];
    }
  }
  [b endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 5) { fprintf(stderr, "usage: see header\n"); return 1; }
  const int L = atoi(argv[3]), R = argc > 5 ? atoi(argv[5]) : 1;
  std::vector<int> Bs;
  {
    std::stringstream ss(argv[4]);
    for (std::string t; std::getline(ss, t, ',');) Bs.push_back(std::stoi(t));
  }
  const int Bmax = *std::max_element(Bs.begin(), Bs.end());
  @autoreleasepool {
    Gpu g("build/core.metallib");
    double t0 = now();
    Checkpoint ck(argv[1]);
    std::ifstream f(argv[2]);
    int nseq, T;
    f >> nseq >> T;
    if (T < L) { fprintf(stderr, "tokens.txt has %d tokens per sequence, need %d\n", T, L); return 1; }
    std::vector<int> toks(T);
    for (auto& v : toks) f >> v;
    const int maxb = (L + KV_BLOCK - 1) / KV_BLOCK, first = (L - R) / KV_BLOCK, own = maxb - first;
    const int chunk = 2048;
    Qwen35 m(g, ck, std::max(chunk, Bmax * R), Bmax, Bmax, maxb + Bmax * own, maxb);
    printf("loaded in %.1f s (KV %s, QSA %s)\n", now() - t0, m.kv8 ? "8-bit" : "bf16", m.qsa_on ? "on" : "off");
    std::vector<std::vector<int>> blocks(Bmax);
    for (int b = 0; b < Bmax; b++)
      for (int j = 0; j < maxb; j++) blocks[b].push_back(j < first ? j : b == 0 ? j : maxb + (b - 1) * own + (j - first));
    m.reset_slot(0);
    double tp = now();
    for (int t = 0; t < L - R; t += chunk) {
      const int n = std::min(chunk, L - R - t);
      run(g, m, m.set_batch({{0, &blocks[0], t, toks.data() + t, n}}), 1);
    }
    printf("prefill %d tokens: %.2f s\n", L - R, now() - tp);
    if (Enc::prof) Enc::prof->clear();  // profile the timed steps only
    for (int b = 1; b < Bmax; b++) {  // private copies of the written blocks and of the state
      for (int j = first; j < maxb; j++) copy_block(g, m, blocks[0][j], blocks[b][j]);
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      m.encode_copy_slot(bl, 0, b);
      [bl endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
    std::vector<std::string> variants{""};
    if (const char* bv = getenv("BENCH_VARIANTS")) {
      variants.clear();
      std::stringstream vs(bv);
      for (std::string v; std::getline(vs, v, ';');) variants.push_back(v);
    }
    for (auto& var : variants) {
      if (!var.empty()) {
        std::stringstream es(var);
        for (std::string kv; std::getline(es, kv, ',');)
          if (auto eq = kv.find('='); eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
        m.skip = getenv("SKIP") ? getenv("SKIP") : "";
        m.knobs();
        printf("variant [%s]\n", var.c_str());
      }
      for (int B : Bs) {
        std::vector<Qwen35::SeqWork> w;
        for (int b = 0; b < B; b++) w.push_back({b, &blocks[b], L - R, toks.data() + L - R, R});
        std::vector<double> gpu, wall;
        for (int i = 0; i < 13; i++) {
          double w0 = now();
          int Tn = m.set_batch(w);
          double gt = run(g, m, Tn, B);
          if (i >= 3) { gpu.push_back(gt); wall.push_back(now() - w0); }
        }
        std::sort(gpu.begin(), gpu.end());
        std::sort(wall.begin(), wall.end());
        const double s = wall[wall.size() / 2];
        printf("B=%3d L=%6d R=%d  step %7.2f ms (GPU %7.2f)  = %7.1f rows/s total\n", B, L, R, s * 1e3,
               gpu[gpu.size() / 2] * 1e3, B * R / s);
      }
    }
  }
  if (Enc::prof) {
    std::vector<std::pair<double, std::string>> v;
    double tot = 0;
    for (auto& [k, ms] : *Enc::prof) { v.push_back({ms, k}); tot += ms; }
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 25; i++) printf("%9.2f ms %5.1f%%  %s\n", v[i].first, 100 * v[i].first / tot, v[i].second.c_str());
  }
  return 0;
}
