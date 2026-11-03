// fn_gemm <model_dir> <M list>
//   Flash-Next prefill GEMM classes at M rows, every layer's own weights in turn (as a prefill chunk reads them):
//   GDN / attention in_proj, QSA index projection, out / o_proj, shared expert, hyper-connection down / up, PLE kv,
//   and the router (moe_qmv16f as the engine runs it, against a bf16 GEMM). Per class: ms per call and per chunk.
//   FN_GEMM_VARIANTS="W4A8_MAX=256;W4A8_MAX=100000" re-times every class with the engine's w4a8_max set from each
//   variant (one load).
#include <chrono>
#include <sstream>
#include "../src/qwen35.h"

static std::vector<int> int_list(const char* s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stoi(t));
  return v;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) { fprintf(stderr, "usage: see header\n"); return 1; }
  @autoreleasepool {
    Gpu g("build/core.metallib");
    Checkpoint ck(argv[1]);
    std::vector<int> Ms = int_list(argv[2]);
    int Mmax = *std::max_element(Ms.begin(), Ms.end());
    Qwen35 m(g, ck, Mmax, 1, 1, 64, 64);
    struct Sh { const char* name; std::vector<QLinear*> ws; };
    std::vector<Sh> shapes = {{"gdn.in_proj", {}}, {"attn.in_proj", {}}, {"qsa.iproj", {}}, {"gdn.out_proj", {}},
                              {"attn.o_proj", {}}, {"shared.gate_up", {}}, {"shared.down", {}}, {"hyper.dn", {}},
                              {"hyper.up", {}}, {"ple.kv", {}}};
    for (auto& l : m.L) {
      shapes[l.linear ? 0 : 1].ws.push_back(&l.in_proj);
      if (l.has_qsa) shapes[2].ws.push_back(&l.iproj);
      shapes[l.linear ? 3 : 4].ws.push_back(&l.out_proj);
      shapes[5].ws.push_back(&l.gate_up);
      shapes[6].ws.push_back(&l.down);
      for (auto* h : {&l.ahc, &l.mhc}) { shapes[7].ws.push_back(&h->dn); shapes[8].ws.push_back(&h->up); }
      if (l.ple) shapes[9].ws.push_back(&l.ple_kv);
    }
    // Routers as bf16 QLinear views (same buffer: [N, K] bf16 rows)
    std::vector<QLinear> routers;
    for (auto& l : m.L) {
      QLinear q;
      q.wq = l.router.wq; q.s = l.router.wq; q.b = l.router.wq;
      q.N = l.router.N; q.K = l.router.K; q.bits = 16;
      routers.push_back(q);
    }
    BufRef A = g.alloc((size_t)Mmax * 17408 * 2), C = g.alloc((size_t)Mmax * 20480 * 4);
    memset(A.ptr<char>(), 0x3c, (size_t)Mmax * 17408 * 2);  // ~1.1 in bf16
    auto timeit = [&](auto&& body, int reps) {
      std::vector<double> t;
      for (int i = 0; i < 5; i++) {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        Enc e(g, cb);
        body(e);
        e.end();
        [cb commit];
        [cb waitUntilCompleted];
        if (i >= 1) t.push_back((cb.GPUEndTime - cb.GPUStartTime) / reps);
      }
      std::sort(t.begin(), t.end());
      return t[t.size() / 2];
    };
    std::vector<std::string> variants{""};
    if (const char* bv = getenv("FN_GEMM_VARIANTS")) {
      variants.clear();
      std::stringstream vs(bv);
      for (std::string v; std::getline(vs, v, ';');) variants.push_back(v);
    }
    for (auto& var : variants) {
      if (auto eq = var.find('='); eq != std::string::npos && var.substr(0, eq) == "W4A8_MAX") m.w4a8_max = std::stoi(var.substr(eq + 1));
      printf("variant [%s] w4a8_max %d\n", var.c_str(), m.w4a8_max);
      for (int M : Ms) {
        double total = 0;
        for (auto& sh : shapes) {
          if (sh.ws.empty()) continue;
          const int n = (int)sh.ws.size();
          double s = timeit([&](Enc& e) { for (auto* w : sh.ws) { m.a8_ready = false; m.gemm(e, A, *w, C, M, 0); m.flush_split(e); } }, n);
          total += s * n;
          const QLinear& w0 = *sh.ws[0];
          printf("M=%5d %-15s N=%6d K=%6d bits %d%s  %8.3f ms x%-3d = %7.2f ms  %6.1f TFLOPS\n", M, sh.name, w0.N, w0.K,
                 w0.bits, w0.flipped ? "f" : " ", s * 1e3, n, s * n * 1e3, 2.0 * M * w0.N * w0.K / s / 1e12);
        }
        {
          const int n = (int)routers.size();
          double s = timeit([&](Enc& e) { for (auto& l : m.L) m.moe_mv(e, A, l.router, m.rlog, M, 1, false, true); }, n);
          printf("M=%5d %-15s N=%6d K=%6d moe_qmv16f %8.3f ms x%-3d = %7.2f ms\n", M, "router", routers[0].N, routers[0].K, s * 1e3, n, s * n * 1e3);
          double s2 = timeit([&](Enc& e) { for (auto& q : routers) m.gemm(e, A, q, C, M, 0); }, n);
          printf("M=%5d %-15s N=%6d K=%6d bf16 gemm  %8.3f ms x%-3d = %7.2f ms\n", M, "router", routers[0].N, routers[0].K, s2 * 1e3, n, s2 * n * 1e3);
          total += s * n;
        }
        printf("M=%5d total %.1f ms per chunk (router as moe_qmv16f)\n", M, total * 1e3);
      }
    }
  }
  return 0;
}
