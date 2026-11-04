// decode_test check <model_dir> <tokens.txt> <out.bin> [chunk]
//   tokens.txt: "B T" then B lines of T token ids. All B sequences advance together in ragged steps of up to
//   `chunk` tokens each (1 = pure decode; larger exercises prefill). Writes the final logits, float32 [B, vocab].
// decode_test bench <model_dir> <B list, e.g. 1,8,16,32,64> <context L> [rows]
//   Times decode steps with every sequence at position L-1 (cache contents are irrelevant to speed). With rows > 1
//   each sequence feeds a block of that many rows ending at L-1, as a speculative verify step does.
// decode_test prefill <model_dir> <L> <chunk list, e.g. 512,1024,2048>
//   Times prefilling one sequence of L tokens in chunks.
// decode_test gemm <model_dir> <M list>
//   Times every projection shape of layers 0 (GDN) and 3 (attention).
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include "../src/qwen35.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> int_list(const char* s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stoi(t));
  return v;
}

static double run(Gpu& g, Qwen35& m, int T, int S) {
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  const uint64_t d0 = Enc::dispatches;
  m.encode_step(cb, T, S);
  if (getenv("COUNT_DISPATCHES")) fprintf(stderr, "dispatches per step: %llu\n", (unsigned long long)(Enc::dispatches - d0));
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "step: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
  return cb.GPUEndTime - cb.GPUStartTime;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 4) { fprintf(stderr, "usage: see header\n"); return 1; }
  std::string mode = argv[1];
  @autoreleasepool {
    Gpu g("build/core.metallib");
    double t0 = now();
    Checkpoint ck(argv[2]);
    if (mode == "check") {
      std::ifstream f(argv[3]);
      int B, T;
      f >> B >> T;
      std::vector<std::vector<int>> toks(B, std::vector<int>(T));
      for (auto& r : toks) for (auto& v : r) f >> v;
      // chunk "a" or "a,b": first step a tokens, then b per step
      int first = argc > 5 ? atoi(argv[5]) : 1, chunk = first;
      if (argc > 5 && strchr(argv[5], ',')) chunk = atoi(strchr(argv[5], ',') + 1);
      int maxb = (T + KV_BLOCK - 1) / KV_BLOCK;
      Qwen35 m(g, ck, B * std::max(first, chunk), B, B, B * maxb, maxb);
      printf("loaded in %.1f s\n", now() - t0);
      std::vector<std::vector<int>> blocks(B);
      for (int b = 0; b < B; b++) {
        m.reset_slot(b);
        for (int j = 0; j < maxb; j++) blocks[b].push_back(b * maxb + j);
      }
      for (int t = 0; t < T; t += (t == 0 ? first : chunk)) {
        int n = std::min(t == 0 ? first : chunk, T - t);
        std::vector<Qwen35::SeqWork> w;
        for (int b = 0; b < B; b++) w.push_back({b, &blocks[b], t, toks[b].data() + t, n});
        run(g, m, m.set_batch(w), B);
      }
      std::vector<float> lf((size_t)B * m.cfg.vocab);
      for (size_t i = 0; i < lf.size(); i++) {
        uint32_t u = (uint32_t)m.logits.ptr<uint16_t>()[i] << 16;
        memcpy(&lf[i], &u, 4);
      }
      std::ofstream o(argv[4], std::ios::binary);
      o.write((const char*)lf.data(), lf.size() * 4);
      printf("wrote %d x %d logits (chunk %d)\n", B, m.cfg.vocab, chunk);
    } else if (mode == "bench") {
      std::vector<int> Bs = int_list(argv[3]);
      int L = std::stoi(argv[4]), R = argc > 5 ? atoi(argv[5]) : 1;
      int Bmax = *std::max_element(Bs.begin(), Bs.end());
      int maxb = (L + KV_BLOCK - 1) / KV_BLOCK;
      Qwen35 m(g, ck, Bmax * R, Bmax, Bmax, Bmax * maxb, maxb);
      printf("loaded in %.1f s\n", now() - t0);
      std::vector<std::vector<int>> blocks(Bmax);
      std::vector<int> tk(Bmax * R);
      for (int b = 0; b < Bmax; b++) {
        for (int r = 0; r < R; r++) tk[b * R + r] = 1000 + b + r;
        for (int j = 0; j < maxb; j++) blocks[b].push_back(b * maxb + j);
        // BENCH_SHARED=C: every sequence's first C tokens use sequence 0's KV blocks (a shared prefix: cascade).
        if (getenv("BENCH_SHARED"))
          for (int j = 0; j < atoi(getenv("BENCH_SHARED")) / 32 && j < maxb; j++) blocks[b][j] = j;
      }
      // BENCH_SHUFFLE=1: scatter every sequence's blocks across the pool (a fragmented KV pool after long runs).
      if (getenv("BENCH_SHUFFLE")) {
        std::vector<int> perm(Bmax * maxb);
        for (int i = 0; i < (int)perm.size(); i++) perm[i] = i;
        std::mt19937 rng(7);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (auto& bl : blocks) for (auto& x : bl) x = perm[x];
      }
      // BENCH_VARIANTS="SKIP=moe;HC_V2=0,MOE_V2=0;..." (timing only): every variant's env settings applied in turn
      // (m.skip and Qwen35::knobs() re-read) and the B list run again, so one load times several configurations.
      std::vector<std::string> variants{""};
      if (const char* bv = getenv("BENCH_VARIANTS")) {
        variants.clear();
        std::stringstream vs(bv);
        for (std::string v; std::getline(vs, v, ';');) variants.push_back(v);
      }
      for (auto& var : variants) {
      if (!var.empty() || variants.size() > 1) {
        std::stringstream es(var);
        for (std::string kv; std::getline(es, kv, ',');)
          if (auto eq = kv.find('='); eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
        m.skip = getenv("SKIP") ? getenv("SKIP") : "";
        m.knobs();
        printf("variant [%s]\n", var.c_str());
      }
      for (int B : Bs) {
        std::vector<Qwen35::SeqWork> w;
        for (int b = 0; b < B; b++) w.push_back({b, &blocks[b], L - R, &tk[b * R], R});
        std::vector<double> gpu, wall;
        for (int i = 0; i < 13; i++) {
          double w0 = now();
          int T = m.set_batch(w);
          double gt = run(g, m, T, B);
          if (i >= 3) { gpu.push_back(gt); wall.push_back(now() - w0); }
        }
        std::sort(gpu.begin(), gpu.end());
        std::sort(wall.begin(), wall.end());
        double s = wall[wall.size() / 2];
        printf("B=%3d L=%6d R=%d  step %7.2f ms (GPU %7.2f)  = %7.1f rows/s total, %6.1f/stream\n", B, L, R, s * 1e3,
               gpu[gpu.size() / 2] * 1e3, B * R / s, R / s);
      }
      }
    } else if (mode == "prefill") {
      int L = std::stoi(argv[3]);
      std::vector<int> chunks = int_list(argv[4]);
      int cmax = *std::max_element(chunks.begin(), chunks.end());
      int maxb = (L + KV_BLOCK - 1) / KV_BLOCK;
      Qwen35 m(g, ck, cmax, 1, 1, maxb, maxb);
      printf("loaded in %.1f s\n", now() - t0);
      std::vector<int> blocks, tk(L);
      for (int j = 0; j < maxb; j++) blocks.push_back(j);
      for (int i = 0; i < L; i++) tk[i] = 1000 + (i * 7919) % 100000;
      for (int c : chunks) {
        m.reset_slot(0);
        double w0 = now(), gpu = 0;
        for (int t = 0; t < L; t += c) {
          int n = std::min(c, L - t);
          gpu += run(g, m, m.set_batch({{0, &blocks, t, tk.data() + t, n}}), 1);
        }
        double s = now() - w0;
        printf("prefill L=%6d chunk %5d: %7.2f s  = %7.0f tok/s (GPU %.2f s)\n", L, c, s, L / s, gpu);
      }
    } else if (mode == "followup") {
      // followup <model> <L> <B list> <chunk list>: one step of B sequences each adding a chunk of n rows after L
      // positions of history (an agent's next turn on a cached context). The KV contents are whatever the pool
      // holds: this times the step, it does not check it.
      int L = std::stoi(argv[3]);
      std::vector<int> Bs = int_list(argv[4]), chunks = int_list(argv[5]);
      int Bmax = *std::max_element(Bs.begin(), Bs.end()), cmax = *std::max_element(chunks.begin(), chunks.end());
      int maxb = (L + cmax + KV_BLOCK - 1) / KV_BLOCK;
      Qwen35 m(g, ck, Bmax * cmax, Bmax, Bmax, Bmax * maxb, maxb);
      printf("loaded in %.1f s\n", now() - t0);
      std::vector<std::vector<int>> blocks(Bmax);
      for (int b = 0; b < Bmax; b++)
        for (int j = 0; j < maxb; j++) blocks[b].push_back(b * maxb + j);
      std::vector<int> tk(cmax);
      for (int i = 0; i < cmax; i++) tk[i] = 1000 + (i * 7919) % 100000;
      for (int B : Bs)
        for (int c : chunks) {
          std::vector<Qwen35::SeqWork> w;
          for (int b = 0; b < B; b++) w.push_back({b, &blocks[b], L, tk.data(), c});
          std::vector<double> t;
          for (int i = 0; i < 5; i++) {
            double w0 = now();
            run(g, m, m.set_batch(w), B);
            t.push_back(now() - w0);
          }
          std::sort(t.begin(), t.end());
          printf("followup L=%6d B=%2d chunk %4d: %8.1f ms per step (median of 5)\n", L, B, c, t[2] * 1e3);
        }
    } else if (mode == "gemmcmp") {
      // gemmcmp <model> <M list>: QA8 vs the default route on each 4-bit shape (first layer's weights), random
      // activations (normal plus a few outlier channels): cosine and max abs error of the bf16 outputs, then timing of
      // both routes over every layer's weights of the shape (mode 0, bf16 out).
      std::vector<int> Ms = int_list(argv[3]);
      int Mmax = *std::max_element(Ms.begin(), Ms.end());
      Qwen35 m(g, ck, Mmax, Mmax, 1, 1, 1);
      if (m.qa8_kmax == 0) { m.qa8_on = 1; m.qa8_prepare(); }
      struct Sh { const char* name; std::vector<QLinear*> ws; };
      std::vector<Sh> shapes = {{"gdn.in_proj", {}}, {"gdn.out_proj", {}}, {"attn.in_proj", {}}, {"attn.o_proj", {}},
                                {"mlp.gate_up", {}}, {"mlp.down", {}}};
      for (auto& l : m.L) {
        shapes[l.linear ? 0 : 2].ws.push_back(&l.in_proj);
        shapes[l.linear ? 1 : 3].ws.push_back(&l.out_proj);
        shapes[4].ws.push_back(&l.gate_up);
        shapes[5].ws.push_back(&l.down);
      }
      int Kmax = 0;
      for (auto& sh : shapes) for (auto* q : sh.ws) Kmax = std::max(Kmax, q->K);
      size_t Nmax = 0;
      for (auto& sh : shapes) for (auto* q : sh.ws) Nmax = std::max<size_t>(Nmax, q->N);
      BufRef A = g.alloc((size_t)Mmax * Kmax * 2), C0 = g.alloc((size_t)Mmax * Nmax * 4), C1 = g.alloc((size_t)Mmax * Nmax * 4);
      {
        uint32_t h = 12345;
        auto rnd = [&]() { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h & 0xffffff) / 16777216.0; };
        uint16_t* a = A.ptr<uint16_t>();
        for (size_t i = 0; i < (size_t)Mmax * Kmax; i++) {
          double u1 = std::max(rnd(), 1e-9), u2 = rnd();
          float v = (float)(sqrt(-2 * log(u1)) * cos(6.2831853 * u2));
          if ((i % Kmax) % 997 == 3) v *= 30;  // outlier channels
          uint32_t b; memcpy(&b, &v, 4);
          a[i] = (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
        }
      }
      const int q_on = m.qa8_on;
      auto run1 = [&](const QLinear& W, const BufRef& C, int M, int mode, bool qa8, int reps, const std::vector<QLinear*>* ws) {
        m.qa8_on = qa8 ? 1 : 0;
        int pf = m.pf8_on;
        if (qa8) m.pf8_on = 0;
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        Enc e(g, cb);
        for (int r = 0; r < reps; r++) m.gemm(e, A, ws ? *(*ws)[r % ws->size()] : W, C, M, mode);
        e.end();
        [cb commit];
        [cb waitUntilCompleted];
        m.pf8_on = pf;
        return (cb.GPUEndTime - cb.GPUStartTime) / reps;
      };
      for (int M : Ms) {
        double t0s = 0, t1s = 0;
        for (auto& sh : shapes) {
          if (sh.ws.empty()) continue;
          const QLinear& W = *sh.ws[0];
          if (!m.qa8_route(W, M) && !(m.qa8_on = 1, m.qa8_route(W, M))) { printf("M=%4d %-12s N=%6d K=%6d: not routable\n", M, sh.name, W.N, W.K); continue; }
          run1(W, C0, M, 0, false, 1, nullptr);  // bf16 outputs (MLX's qmm has no float epilogue)
          run1(W, C1, M, 0, true, 1, nullptr);
          auto bfv = [](uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; };
          const uint16_t *c0 = C0.ptr<uint16_t>(), *c1 = C1.ptr<uint16_t>();
          double dot = 0, n0 = 0, n1 = 0, mx = 0, ref = 0;
          for (size_t i = 0; i < (size_t)M * W.N; i++) {
            const double a0 = bfv(c0[i]), a1 = bfv(c1[i]);
            dot += a0 * a1, n0 += a0 * a0, n1 += a1 * a1;
            mx = std::max(mx, fabs(a0 - a1)), ref = std::max(ref, fabs(a0));
          }
          std::vector<double> ta, tb;
          int reps = std::max<int>(sh.ws.size(), 4);
          for (int i = 0; i < 5; i++) {
            ta.push_back(run1(W, C0, M, 0, false, reps, &sh.ws));
            tb.push_back(run1(W, C1, M, 0, true, reps, &sh.ws));
          }
          std::sort(ta.begin(), ta.end());
          std::sort(tb.begin(), tb.end());
          const double a = ta[2], b = tb[2];
          t0s += a * sh.ws.size(), t1s += b * sh.ws.size();
          printf("M=%4d %-12s N=%6d K=%6d  cos %.6f  maxerr %.3g (max |ref| %.3g)  default %7.3f ms (%5.1f TOPS)  QA8 %7.3f ms (%5.1f TOPS)  x%zu\n",
                 M, sh.name, W.N, W.K, dot / sqrt(n0 * n1 + 1e-30), mx, ref, a * 1e3, 2.0 * M * W.N * W.K / a / 1e12, b * 1e3,
                 2.0 * M * W.N * W.K / b / 1e12, sh.ws.size());
        }
        printf("M=%4d all 4-bit linears: default %.1f ms, QA8 %.1f ms per step (%.2fx)\n", M, t0s * 1e3, t1s * 1e3, t0s / t1s);
      }
      m.qa8_on = q_on;
    } else if (mode == "gemm") {
      std::vector<int> Ms = int_list(argv[3]);
      int Mmax = *std::max_element(Ms.begin(), Ms.end());
      Qwen35 m(g, ck, Mmax, Mmax, 1, 1, 1);
      // Every layer's own weights in turn (as a real step reads them), so nothing is served from cache.
      struct Sh { const char* name; std::vector<QLinear*> ws; };
      std::vector<Sh> shapes = {{"gdn.in_proj", {}}, {"gdn.out_proj", {}}, {"attn.in_proj", {}}, {"attn.o_proj", {}},
                                {"mlp.gate_up", {}}, {"mlp.down", {}}, {"lm_head", {&m.lm_head}}};
      for (auto& l : m.L) {
        shapes[l.linear ? 0 : 2].ws.push_back(&l.in_proj);
        shapes[l.linear ? 1 : 3].ws.push_back(&l.out_proj);
        shapes[4].ws.push_back(&l.gate_up);
        shapes[5].ws.push_back(&l.down);
      }
      BufRef A = g.alloc((size_t)Mmax * 17408 * 2), C = g.alloc((size_t)Mmax * m.cfg.vocab * 2);
      for (int M : Ms) {
        double total = 0, tbytes = 0;
        for (auto& sh : shapes) {
          std::vector<double> t;
          const QLinear& w0 = *sh.ws[0];
          for (int i = 0; i < 6; i++) {
            id<MTLCommandBuffer> cb = [g.queue commandBuffer];
            Enc e(g, cb);
            int reps = std::max<int>(sh.ws.size(), 4);
            for (int r = 0; r < reps; r++) m.gemm(e, A, *sh.ws[r % sh.ws.size()], C, M, 0);
            e.end();
            [cb commit];
            [cb waitUntilCompleted];
            if (i >= 2) t.push_back((cb.GPUEndTime - cb.GPUStartTime) / reps);
          }
          std::sort(t.begin(), t.end());
          double s = t[t.size() / 2];
          double bytes = (double)w0.N * w0.K * w0.bits / 8 * (1 + (w0.bits == 16 ? 0 : 32.0 / w0.bits / 64 * 2));
          int per = sh.ws.size();
          total += s * per;
          tbytes += bytes * per;
          printf("M=%4d %-13s N=%6d K=%6d %7.3f ms %6.0f GB/s %6.1f TFLOPS  x%-2d = %6.2f ms\n", M, sh.name, w0.N,
                 w0.K, s * 1e3, bytes / s / 1e9, 2.0 * M * w0.N * w0.K / s / 1e12, per, s * per * 1e3);
        }
        printf("M=%4d all linear layers: %.1f ms per step (%.1f GB, %.0f GB/s; bandwidth floor %.1f ms)\n", M, total * 1e3,
               tbytes / 1e9, tbytes / total / 1e9, tbytes / 1.14e12 * 1e3);
      }
    }
  }
  if (Enc::prof) {  // PROF_KERNELS=1: GPU ms per kernel name over the whole run, largest first
    std::vector<std::pair<double, std::string>> v;
    double tot = 0;
    for (auto& [k, ms] : *Enc::prof) { v.push_back({ms, k}); tot += ms; }
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 25; i++) printf("%9.2f ms %5.1f%%  %s\n", v[i].first, 100 * v[i].first / tot, v[i].second.c_str());
  }
  return 0;
}
