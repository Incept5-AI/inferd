// draft_dbg <target> <draft> <tokens.txt> <reps>: prefill prompt 0, append drafter context, run the draft block
// `reps` times on identical inputs; print draft tokens and buffer checksums each time.
#include <fstream>
#include "../src/dflash.h"
static uint64_t ck(const BufRef& b, size_t n) {
  const uint16_t* p = b.ptr<uint16_t>(); uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    Checkpoint tck(argv[1]), dck(argv[2]);
    std::ifstream f(argv[3]); int B0, T; f >> B0 >> T;
    std::vector<int> pr(T); for (auto& v : pr) f >> v;
    int reps = atoi(argv[4]);
    int maxb = (T + 64) / KV_BLOCK + 2;
    Qwen35 m(g, tck, std::max(T, kMaxBlock), 1, 1, maxb, maxb);
    Drafter d(g, m, dck);
    m.enable_spec(d.target_layers, d.block);
    m.verify = true;
    std::vector<int> blocks; for (int j = 0; j < maxb; j++) blocks.push_back(j);
    m.reset_slot(0);
    std::vector<Qwen35::SeqWork> w{{0, &blocks, 0, pr.data(), T, false}};
    int TT = m.set_batch(w);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    m.encode_step(cb, TT, 1); [cb commit]; [cb waitUntilCompleted];
    int staged = m.pred.ptr<int>()[0];
    printf("staged %d\n", staged);
    std::vector<Drafter::CtxRow> ctx; for (int j = 0; j < T; j++) ctx.push_back({j, j, 0});
    cb = [g.queue commandBuffer]; { Enc e(g, cb); d.encode_context(e, ctx, 0); e.end(); } [cb commit]; [cb waitUntilCompleted];
    {
      auto bf = [](uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; };
      auto nan = [&](const BufRef& b, size_t n) { size_t c = 0; for (size_t i = 0; i < n; i++) { float v = bf(b.ptr<uint16_t>()[i]); c += !std::isfinite(v); } return c; };
      int K = d.fc.K, N = d.fc.N;
      printf("cx %016llx nan %zu | h %016llx nan %zu | K %d N %d bits %d flipped %d\n", (unsigned long long)ck(d.cx, (size_t)T * K), nan(d.cx, (size_t)T * K),
             (unsigned long long)ck(d.h, (size_t)T * N), nan(d.h, (size_t)T * N), K, N, d.fc.bits, (int)d.fc.flipped);
      // CPU reference for rows 0 and T-1 of h = cx * dequant(fc)^T (8-bit, flipped: q ^ 0x80)
      for (int r : {0, T - 1}) {
        double err = 0, mag = 0;
        for (int n = 0; n < N; n += 7) {
          double acc = 0;
          for (int k = 0; k < K; k++) {
            uint8_t q = d.fc.wq.ptr<uint8_t>()[(size_t)n * K + k]; if (d.fc.flipped) q ^= 0x80;
            float sc = bf(d.fc.s.ptr<uint16_t>()[(size_t)n * (K / 64) + k / 64]), bb = bf(d.fc.b.ptr<uint16_t>()[(size_t)n * (K / 64) + k / 64]);
            acc += (double)(sc * q + bb) * bf(d.cx.ptr<uint16_t>()[(size_t)r * K + k]);
          }
          double gv = bf(d.h.ptr<uint16_t>()[(size_t)r * N + n]);
          err = std::max(err, fabs(gv - acc)); mag = std::max(mag, fabs(acc));
        }
        printf("fc row %d: max|err| %.4f max|ref| %.4f\n", r, err, mag);
      }
    }
    printf("ring0 %016llx kv %016llx cxn %016llx\n", (unsigned long long)ck(d.kr[0], (size_t)d.ring * d.Hkv * d.D),
           (unsigned long long)ck(d.kv, (size_t)T * 2 * d.Hkv * d.D), (unsigned long long)ck(d.cxn, (size_t)T * d.H));
    int L = d.block;
    for (int r = 0; r < reps; r++) {
      std::vector<int> vt(L, 0); vt[0] = staged;
      std::vector<Qwen35::SeqWork> vw{{0, &blocks, T, vt.data(), L, true}};
      m.set_batch(vw);
      cb = [g.queue commandBuffer]; { Enc e(g, cb); d.encode_block(e, {{0, T, staged, 0}}); e.end(); } [cb commit]; [cb waitUntilCompleted];
      { size_t c = 0; for (size_t i = 0; i < (size_t)L * d.H; i++) { uint32_t u = (uint32_t)d.x.ptr<uint16_t>()[i] << 16; float v; memcpy(&v, &u, 4); c += !std::isfinite(v); } printf("x nonfinite %zu  ", c); }
      printf("rep %d draft:", r); for (int j = 0; j < L; j++) printf(" %d", m.tok.ptr<int>()[j]);
      printf("  x %016llx q %016llx attn %016llx hp %016llx cand %016llx\n", (unsigned long long)ck(d.x, (size_t)L * d.H),
             (unsigned long long)ck(d.q, (size_t)L * d.Hq * d.D), (unsigned long long)ck(d.attn, (size_t)L * d.Hq * d.D),
             (unsigned long long)ck(d.hp, (size_t)L * 256), (unsigned long long)ck(d.cand, (size_t)L * 32));
    }
  }
}
