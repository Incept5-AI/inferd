// moe_bench <kernel> <T> <unique> [layers=16] [reps=40]
//   Times one 35B-A3B-shaped MoE gate/up (moe_gu2_4_4_t*) or down (moe_dn2_4_4) launch on random
//   4-bit weights: E = 256 experts, I = 512, H = 2048, k = 8, T rows whose picks cover `unique` distinct experts
//   (rows spread over them round-robin). `layers` weight copies are cycled (more than the SLC holds) and `reps`
//   launches go into one command buffer, so the time per launch includes the dispatch boundaries a real step has.
#include <chrono>
#include <random>
#include "../src/gpu.h"

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: moe_bench <kernel> <T> <unique> [layers] [reps]\n"); return 1; }
  @autoreleasepool {
    Gpu g("build/core.metallib");
    const std::string kn = argv[1];
    const int T = atoi(argv[2]), U = atoi(argv[3]), NL = argc > 4 ? atoi(argv[4]) : 16, reps = argc > 5 ? atoi(argv[5]) : 40;
    const int E = 256, I = 512, H = 2048, k = 8, NU = T * k;
    const bool down = kn.rfind("moe_dn", 0) == 0;
    const int N = down ? H : 2 * I, K = down ? I : H;  // per expert [N][K]
    const size_t wb = (size_t)E * N * K / 2, sb = (size_t)E * N * (K / 64) * 2;
    std::mt19937 rng(1);
    std::vector<BufRef> W, Sc, Bi;
    for (int l = 0; l < NL; l++) {
      W.push_back(g.alloc(wb)); Sc.push_back(g.alloc(sb)); Bi.push_back(g.alloc(sb));
      uint32_t* w = W.back().ptr<uint32_t>();
      for (size_t i = 0; i < wb / 4; i++) w[i] = rng();
      uint16_t* s = Sc.back().ptr<uint16_t>(); uint16_t* b = Bi.back().ptr<uint16_t>();
      for (size_t i = 0; i < sb / 2; i++) s[i] = 0x3c00 - (rng() & 255), b[i] = 0xbc00 + (rng() & 255);  // ~ +-0.01 bf16
    }
    // shared expert (4-bit, unflipped here): [2I][H] for gate/up, [H][I] for down
    BufRef SW = g.alloc((size_t)N * K / 2), SS = g.alloc((size_t)N * (K / 64) * 2), SB = g.alloc((size_t)N * (K / 64) * 2);
    BufRef x = g.alloc((size_t)8 * (k + 1) * std::max(H, I) * 2), act = g.alloc((size_t)8 * (k + 1) * std::max(H, I) * 2);
    for (size_t i = 0; i < x.buf.length / 2; i++) x.ptr<uint16_t>()[i] = 0x3f80 - (rng() & 127);
    // plan: U unique experts, row t's pick j -> expert uexp[(t * k + j) % U] (distinct within a row when U >= k)
    std::vector<int> ex(E);
    for (int i = 0; i < E; i++) ex[i] = i;
    std::shuffle(ex.begin(), ex.end(), rng);
    BufRef plan = g.alloc((size_t)(1 + NU * 9) * 4), ridx = g.alloc(NU * 4), rw = g.alloc(NU * 4), rsg = g.alloc(8 * 4);
    int* pl = plan.ptr<int>();
    pl[0] = U;
    for (int i = 0; i < NU * 8; i++) pl[1 + NU + i] = -1;
    for (int u = 0; u < U; u++) pl[1 + u] = ex[u];
    for (int t = 0; t < T; t++)
      for (int j = 0; j < k; j++) {
        const int u = (t * k + j) % U;
        pl[1 + NU + u * 8 + t] = j;
        ridx.ptr<int>()[t * k + j] = ex[u];
        rw.ptr<float>()[t * k + j] = 0.125f;
      }
    for (int t = 0; t < 8; t++) rsg.ptr<float>()[t] = 0.5f;
    const int tb = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
    std::vector<double> ts;
    for (int it = 0; it < 7; it++) {
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      Enc e(g, cb);
      for (int r = 0; r < reps; r++) {
        const int l = r % NL;
        if (!down) {
          struct { int k, I, K, xr, T; } ga{k, I, H, 0, T};
          e.k(kn.c_str()).buf(0, x).buf(1, plan).buf(2, W[l]).buf(3, Sc[l]).buf(4, Bi[l]).buf(5, SW).buf(6, SS).buf(7, SB)
              .buf(8, act).args(9, ga);
          e.groups(S3(I / 8, T * k + T), S3(256));
        } else {
          struct { int k, I, H, xr, assign; } da{k, I, H, 0, 0};
          e.k(kn.c_str()).buf(0, x).buf(1, ridx).buf(2, rw).buf(3, rsg).buf(4, W[l]).buf(5, Sc[l]).buf(6, Bi[l])
              .buf(7, SW).buf(8, SS).buf(9, SB).buf(10, act).args(11, da);
          e.groups(S3(H / 8, T), S3(256));
        }
      }
      e.end();
      [cb commit];
      [cb waitUntilCompleted];
      if (it >= 2) ts.push_back((cb.GPUEndTime - cb.GPUStartTime) / reps);
    }
    std::sort(ts.begin(), ts.end());
    const double s = ts[ts.size() / 2];
    const double bytes = down ? (double)T * k * H * I / 2 * 1.0625 : (double)U * 2 * I * H / 2 * 1.0625;
    printf("%s T=%d unique=%d: %.1f us per launch, %.0f GB/s of routed weights\n", kn.c_str(), T, U, s * 1e6, bytes / s / 1e9);
    (void)tb;
  }
  return 0;
}
