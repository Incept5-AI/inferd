// gqmm_test [T=256] [kernel...]: the sorted-MoE prefill chain (moe_sort -> moe_gather -> gathered qmm) per op, on
// random 4-bit weights of the 35B-A3B's shapes (E = 256 experts, k = 8, H = 2048, I = 512), against the per-row GEMV
// the unsorted path runs (moe_qmv4) and a CPU fp64 reference on every 16th output column.
//   - routing: T tokens, k distinct experts each, Zipf-skewed (a few experts get many rows, as real routing does);
//   - moe_sort and moe_gather are checked exactly (permutation, offsets, copied rows);
//   - each kernel runs the gate/up shape (N 1024, K 2048) and the down shape (N 2048, K 512).
// Kernels: "nax16", "nax64" (mlx_gqmm_b4_bm16 / _bm64, MLX's NAX gathered qmm, what Qwen35::gqmm ran everywhere
// before 21 Nov), "sg" (mlx_gqmm_sg_b4, MLX's simdgroup affine_gather_qmm_rhs: BM 16, BN 32, BK 32, 1 x 2
// simdgroups; src/ops/gather.h), and the MXFP4 pair "mxnax" (mxfp4 fp_gather_qmm_rhs_nax, bm 32) / "mxsg"
// (fp_gather_qmm_rhs, MLX's non-NAX tiles; src/mxq.h mx_gqmm). Default: all five.
// Prints, per op: max |err| / rms(ref) and the share of outputs off by more than 2^-7 relative (bf16 rounding is 2^-9).
// On a no-NAX GPU Gpu refuses the NAX kernels; NAX_ON_NONAX=1 runs them anyway (they are reported, not judged).
// Exit status (release gate, tools/release_gate.sh): 1 if moe_sort / moe_gather are not exact, or if a kernel the
// engine routes to on this GPU (sg, mxsg on M1-M4; nax16, nax64, mxnax on M5+) has rel-rms > 1e-2 or > 0.1% bad
// outputs. The NAX kernels on a no-NAX GPU are reported (expected garbage) but do not fail the run.
#include <cmath>
#include <random>
#include <vector>
#include "../src/gpu.h"

static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

struct Stats { double maxe = 0, se = 0, sr = 0; long bad = 0, n = 0; };
static void acc(Stats& s, double ref, double got, double scale) {
  const double e = std::fabs(got - ref);
  s.maxe = std::max(s.maxe, e), s.se += e * e, s.sr += ref * ref, s.n++;
  if (e > std::ldexp(1.0, -7) * std::max(std::fabs(ref), scale)) s.bad++;
}
static void print(const char* what, const Stats& s) {
  const double rms = std::sqrt(s.sr / std::max<long>(s.n, 1));
  printf("  %-34s max|err|/rms %.3e  rel-rms %.3e  bad %ld / %ld (%.2f%%)\n", what, s.maxe / rms,
         std::sqrt(s.se / std::max(s.sr, 1e-30)), s.bad, s.n, 100.0 * s.bad / std::max<long>(s.n, 1));
}

int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    const int T = argc > 1 ? atoi(argv[1]) : 256;
    std::vector<std::string> kns;
    for (int i = 2; i < argc; i++) kns.push_back(argv[i]);
    if (kns.empty()) kns = {"nax16", "nax64", "sg", "mxnax", "mxsg"};
    int fails = 0;
    // A kernel the engine routes to on this GPU must match fp64 to bf16 noise; a NAX kernel on a no-NAX GPU is reported only.
    auto verdict = [&](const std::string& kn, bool nax_kernel, const Stats& s, long nan) {
      const bool routed = nax_kernel == g.nax;
      const double rr = std::sqrt(s.se / std::max(s.sr, 1e-30));
      const bool ok = nan == 0 && rr < 1e-2 && s.bad * 1000 <= s.n;
      printf("  -> %s: %s%s\n", kn.c_str(), ok ? "ok" : "WRONG", routed ? "" : " (not routed on this GPU: informational)");
      if (routed && !ok) fails++;
    };
    const int E = 256, k = 8, H = 2048, I = 512, R = T * k;
    printf("gpu nax=%d cores=%d  T=%d R=%d\n", (int)g.nax, g.cores, T, R);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // routing: Zipf-ish weights over a shuffled expert order, k distinct per token
    std::vector<int> order(E);
    for (int e = 0; e < E; e++) order[e] = e;
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<double> wz(E);
    for (int e = 0; e < E; e++) wz[e] = 1.0 / std::pow(e + 1.0, 0.8);
    std::discrete_distribution<int> pick(wz.begin(), wz.end());
    BufRef ridx = g.alloc((size_t)R * 4);
    for (int t = 0; t < T; t++)
      for (int j = 0; j < k; j++) {
        int e;
        bool dup;
        do {
          e = order[pick(rng)], dup = false;
          for (int q = 0; q < j; q++) dup |= ridx.ptr<int>()[t * k + q] == e;
        } while (dup);
        ridx.ptr<int>()[t * k + j] = e;
      }
    std::vector<int> cnt(E, 0);
    for (int r = 0; r < R; r++) cnt[ridx.ptr<int>()[r]]++;
    int used = 0, mx = 0;
    for (int c : cnt) used += c > 0, mx = std::max(mx, c);
    printf("routing: %d experts used, max %d rows on one expert\n", used, mx);

    BufRef x = g.alloc((size_t)T * H * 2), moffs = g.alloc((size_t)E * 4 + 16), mperm = g.alloc((size_t)R * 4),
           minv = g.alloc((size_t)R * 4), xs = g.alloc((size_t)R * H * 2);
    for (size_t i = 0; i < (size_t)T * H; i++) x.ptr<uint16_t>()[i] = tobf(nd(rng));

    // op 1-2: moe_sort, moe_gather
    {
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      Enc e(g, cb);
      struct { int R, E; } sa{R, E};
      e.k("moe_sort").buf(0, ridx).buf(1, moffs).buf(2, mperm).buf(3, minv).args(4, sa);
      e.groups(S3(1), S3(1024));
      struct { int K, div; } ga{H, k};
      e.k("moe_gather").buf(0, x).buf(1, mperm).buf(2, xs).args(3, ga);
      e.threads(S3(H, R), S3(256));
      e.end();
      [cb commit];
      [cb waitUntilCompleted];
    }
    {
      const int *off = moffs.ptr<int>(), *perm = mperm.ptr<int>(), *inv = minv.ptr<int>(), *id = ridx.ptr<int>();
      long errs = 0;
      int s = 0;
      for (int e = 0; e < E; e++) errs += off[e] != s, s += cnt[e];
      std::vector<int> seen(R, 0);
      for (int q = 0; q < R; q++) {
        const int r = perm[q];
        if (r < 0 || r >= R) { errs++; continue; }
        seen[r]++;
        errs += inv[r] != q;
        errs += q > 0 && id[perm[q - 1]] > id[r];
      }
      for (int r = 0; r < R; r++) errs += seen[r] != 1;
      printf("moe_sort: %s (%ld errors)\n", errs ? "WRONG" : "ok", errs);
      fails += errs != 0;
      long gerr = 0;
      for (int q = 0; q < R; q++)
        gerr += memcmp(xs.ptr<uint16_t>() + (size_t)q * H, x.ptr<uint16_t>() + (size_t)(perm[q] / k) * H, H * 2) != 0;
      printf("moe_gather: %s (%ld rows differ)\n", gerr ? "WRONG" : "ok", gerr);
      fails += gerr != 0;
    }

    for (int shape = 0; shape < 2; shape++) {
      const int N = shape == 0 ? 2 * I : H, K = shape == 0 ? H : I;
      printf("%s shape: N %d, K %d\n", shape == 0 ? "gate/up" : "down", N, K);
      const int words = K / 8, groups = K / 64;
      BufRef W = g.alloc((size_t)E * N * words * 4), S = g.alloc((size_t)E * N * groups * 2),
             B = g.alloc((size_t)E * N * groups * 2);
      for (size_t i = 0; i < (size_t)E * N * words; i++) W.ptr<uint32_t>()[i] = rng();
      for (size_t i = 0; i < (size_t)E * N * groups; i++) {
        const float sc = 0.004f + 0.004f * (rng() % 1000) / 1000.f;
        S.ptr<uint16_t>()[i] = tobf(sc);
        B.ptr<uint16_t>()[i] = tobf(-7.5f * sc + 0.002f * nd(rng));
      }
      // the activation: x (gate/up) or a random [R, I] (down), sorted copy for the gathered kernels
      BufRef A = shape == 0 ? x : g.alloc((size_t)R * K * 2), As = shape == 0 ? xs : g.alloc((size_t)R * K * 2);
      const int div = shape == 0 ? k : 1;
      if (shape == 1) {
        for (size_t i = 0; i < (size_t)R * K; i++) A.ptr<uint16_t>()[i] = tobf(nd(rng));
        for (int q = 0; q < R; q++)
          memcpy(As.ptr<uint16_t>() + (size_t)q * K, A.ptr<uint16_t>() + (size_t)mperm.ptr<int>()[q] * K, K * 2);
      }
      // CPU reference on every 16th column, in unsorted row order r
      std::vector<double> ref((size_t)R * (N / 16));
      for (int r = 0; r < R; r++) {
        const int e = ridx.ptr<int>()[r];
        const uint16_t* a = A.ptr<uint16_t>() + (size_t)(r / div) * K;
        for (int c = 0; c < N / 16; c++) {
          const int n = c * 16;
          const size_t row = (size_t)e * N + n;
          double s = 0;
          for (int kk = 0; kk < K; kk++) {
            const uint32_t u = W.ptr<uint32_t>()[row * words + kk / 8];
            const double q = (u >> (4 * (kk % 8))) & 15;
            const double w = bf(S.ptr<uint16_t>()[row * groups + kk / 64]) * q + bf(B.ptr<uint16_t>()[row * groups + kk / 64]);
            s += w * bf(a[kk]);
          }
          ref[(size_t)r * (N / 16) + c] = s;
        }
      }
      double rs = 0;
      for (double v : ref) rs += v * v;
      const double scale = std::sqrt(rs / ref.size());
      // unsorted: moe_qmv4 per row
      BufRef Yu = g.alloc((size_t)R * N * 2);
      {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        Enc e(g, cb);
        struct { int N, K, div, use_idx; } a{N, K, div, 1};
        e.k("moe_qmv4").buf(0, A).buf(1, W).buf(2, S).buf(3, B).buf(4, ridx).buf(5, Yu).args(6, a);
        e.groups(S3((N + 7) / 8, R), S3(256));
        e.end();
        [cb commit];
        [cb waitUntilCompleted];
      }
      Stats su;
      for (int r = 0; r < R; r++)
        for (int c = 0; c < N / 16; c++) acc(su, ref[(size_t)r * (N / 16) + c], bf(Yu.ptr<uint16_t>()[(size_t)r * N + c * 16]), scale);
      print("moe_qmv4 (unsorted) vs fp64", su);
      for (const auto& kn : kns) {
        if (kn.rfind("mx", 0) == 0) continue;  // MXFP4: below
        int bm, bn, tg;
        std::string name;
        std::vector<std::pair<int, bool>> fc;
        if (kn == "nax16" || kn == "nax64") {
          bm = kn == "nax16" ? 16 : 64, bn = 64, tg = bm == 64 ? 128 : 64;
          name = "mlx_gqmm_b4_bm" + std::to_string(bm);
          fc = {{40, false}, {200, false}, {201, N % 64 == 0}, {202, K % 64 == 0}};
        } else if (kn == "sg") {
          bm = 16, bn = 32, tg = 64;
          name = "mlx_gqmm_sg_b4";
          fc = {{40, false}, {201, N % 32 == 0}, {202, K % 32 == 0}};
        } else { fprintf(stderr, "unknown kernel %s\n", kn.c_str()); return 1; }
        if (![g.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]]) {
          printf("  %s: not in the metallib, skipped\n", name.c_str());
          continue;
        }
        if (!g.nax && Gpu::mlx_nax_kernel(name) && !(getenv("NAX_ON_NONAX") && atoi(getenv("NAX_ON_NONAX")))) {
          printf("  %s: MLX-NAX kernel, refused on this no-NAX GPU (NAX_ON_NONAX=1 runs it for the record)\n", name.c_str());
          continue;
        }
        BufRef Ys = g.alloc((size_t)R * N * 2);
        memset(Ys.ptr<char>(), 0xff, (size_t)R * N * 2);  // NaN fill: unwritten outputs show up
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        Enc e(g, cb);
        e.k(g.pso_fc(name, fc)).buf(0, As).buf(1, W).buf(2, S).buf(3, B).buf(4, moffs).buf(5, Ys).args(6, R).args(7, N)
            .args(8, K).args(9, E);
        e.groups(S3((N + bn - 1) / bn, (R + bm - 1) / bm + std::min(E, R)), S3(tg));
        e.end();
        [cb commit];
        [cb waitUntilCompleted];
        Stats sr, sv;
        long nan = 0;
        for (int r = 0; r < R; r++) {
          const int q = minv.ptr<int>()[r];
          for (int c = 0; c < N / 16; c++)
            acc(sr, ref[(size_t)r * (N / 16) + c], bf(Ys.ptr<uint16_t>()[(size_t)q * N + c * 16]), scale);
          for (int n = 0; n < N; n++) {
            const float a = bf(Ys.ptr<uint16_t>()[(size_t)q * N + n]), b = bf(Yu.ptr<uint16_t>()[(size_t)r * N + n]);
            if (std::isnan(a)) { nan++; continue; }
            acc(sv, b, a, scale);
          }
        }
        char lab[96];
        snprintf(lab, sizeof lab, "%s vs fp64", name.c_str());
        print(lab, sr);
        snprintf(lab, sizeof lab, "%s vs moe_qmv4 (all)", name.c_str());
        print(lab, sv);
        if (nan) printf("  %s: %ld outputs never written\n", name.c_str(), nan);
        verdict(kn, kn != "sg", sr, nan);
      }
      // MXFP4 experts (DeepSeek V4.1's layout): w uint32 [E, N, K/8] (low nibble first, E2M1), scales uint8 [E, N, K/32]
      // (E8M0), group 32. Reference in fp64 on every 16th column.
      BufRef MW = g.alloc((size_t)E * N * K / 2), MS = g.alloc((size_t)E * N * (K / 32));
      for (size_t i = 0; i < (size_t)E * N * K / 8; i++) MW.ptr<uint32_t>()[i] = rng();
      for (size_t i = 0; i < (size_t)E * N * (K / 32); i++) MS.ptr<uint8_t>()[i] = 127 - 6 + rng() % 4;
      static const double e2m1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
      std::vector<double> mref((size_t)R * (N / 16));
      for (int r = 0; r < R; r++) {
        const int e = ridx.ptr<int>()[r];
        const uint16_t* a = A.ptr<uint16_t>() + (size_t)(r / div) * K;
        for (int c = 0; c < N / 16; c++) {
          const size_t row = (size_t)e * N + c * 16;
          double s = 0;
          for (int kk = 0; kk < K; kk++) {
            const uint32_t q = (MW.ptr<uint32_t>()[row * (K / 8) + kk / 8] >> (4 * (kk % 8))) & 15;
            const double w = (q & 8 ? -1 : 1) * e2m1[q & 7] * std::ldexp(1.0, (int)MS.ptr<uint8_t>()[row * (K / 32) + kk / 32] - 127);
            s += w * bf(a[kk]);
          }
          mref[(size_t)r * (N / 16) + c] = s;
        }
      }
      double ms2 = 0;
      for (double v : mref) ms2 += v * v;
      const double mscale = std::sqrt(ms2 / mref.size());
      for (const auto& kn : kns) {
        if (kn != "mxnax" && kn != "mxsg") continue;
        const bool nx = kn == "mxnax";
        const std::string name = nx ? "mxfp4_gather_qmm_rhs_nax_nt_bfloat16_t_gs_32_b_4_bm_32_bn_64_bk_64_wm_2_wn_2"
                                    : "mxfp4_gather_qmm_rhs_nt_bfloat16_t_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2";
        if (![g.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]]) {
          printf("  %s: not in the metallib, skipped\n", name.c_str());
          continue;
        }
        if (!g.nax && Gpu::mlx_nax_kernel(name) && !(getenv("NAX_ON_NONAX") && atoi(getenv("NAX_ON_NONAX")))) {
          printf("  %s: MLX-NAX kernel, refused on this no-NAX GPU (NAX_ON_NONAX=1 runs it for the record)\n", name.c_str());
          continue;
        }
        const int bm = nx ? 32 : 16, bn = nx ? 64 : 32;
        std::vector<std::pair<int, bool>> fc = {{201, N % bn == 0}, {202, K % bn == 0}};
        if (nx) fc.insert(fc.begin(), {200, false});
        BufRef Ys = g.alloc((size_t)R * N * 2);
        memset(Ys.ptr<char>(), 0xff, (size_t)R * N * 2);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        Enc e(g, cb);
        e.k(g.pso_fc(name, fc)).buf(0, As).buf(1, MW).buf(2, MS).buf(4, moffs).buf(5, Ys).args(6, R).args(7, N).args(8, K)
            .args(9, E);
        e.groups(MTLSizeMake((N + bn - 1) / bn, std::min(R, (R + bm - 1) / bm + E - 1), 1), MTLSizeMake(32, 2, nx ? 2 : 1));
        e.end();
        [cb commit];
        [cb waitUntilCompleted];
        Stats sr;
        long nan = 0;
        for (int r = 0; r < R; r++) {
          const int q = minv.ptr<int>()[r];
          for (int c = 0; c < N / 16; c++) {
            const float v = bf(Ys.ptr<uint16_t>()[(size_t)q * N + c * 16]);
            if (std::isnan(v)) { nan++; continue; }
            acc(sr, mref[(size_t)r * (N / 16) + c], v, mscale);
          }
        }
        char lab[96];
        snprintf(lab, sizeof lab, "%s vs fp64", nx ? "mxfp4 gather NAX bm32" : "mxfp4 gather sg");
        print(lab, sr);
        if (nan) printf("  %s: %ld sampled outputs never written\n", kn.c_str(), nan);
        verdict(kn, nx, sr, nan);
      }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
  }
  return 0;
}
