// qa8_dbg <M> <N> <K> [WM WN] [rf]: random flipped 4-bit weights + bf16 activations; qa8_quant + qa8 GEMM (float out);
// compare to a CPU reference computed from the same int8 activations (exact up to fp32 rounding) and to bf16 math.
#include "../src/gpu.h"
#include <random>
#include <vector>
static float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t tobf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16); }
int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    int WM = argc > 5 ? atoi(argv[4]) : 2, WN = argc > 5 ? atoi(argv[5]) : 4, rf = argc > 6 ? atoi(argv[6]) : 1;
    int kg = K / 64;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1, 1);
    BufRef A(g.alloc((size_t)M * K * 2)), W(g.alloc((size_t)N * K / 2)), Sc(g.alloc((size_t)N * kg * 2)), Bz(g.alloc((size_t)N * kg * 2));
    BufRef u(g.alloc((size_t)M * K)), xs(g.alloc((size_t)M * kg * 8)), C(g.alloc((size_t)M * N * 4));
    for (size_t i = 0; i < (size_t)M * K; i++) A.ptr<uint16_t>()[i] = tobf(U(rng));
    std::vector<uint8_t> q((size_t)N * K);
    for (auto& v : q) v = rng() & 15;
    for (size_t i = 0; i < (size_t)N * kg; i++) { Sc.ptr<uint16_t>()[i] = tobf(0.01f + 0.01f * fabsf(U(rng))); Bz.ptr<uint16_t>()[i] = tobf(0.05f * U(rng)); }
    uint8_t* w = W.ptr<uint8_t>();
    for (size_t i = 0; i < (size_t)N * K; i++) {
      uint8_t f = q[i] ^ 8;
      if (i % 2 == 0) w[i / 2] = f; else w[i / 2] |= f << 4;
    }
    memset(C.ptr<char>(), 0xff, (size_t)M * N * 4);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    Enc e(g, cb);
    struct { int M, K, lda; } qa{M, K, K};
    e.k("qa8_quant").buf(0, A).buf(1, u).buf(2, xs).args(3, qa);
    e.threads(S3(kg * 32, M), S3(256));
    const int BM = 32 * WM, BN = 32 * WN, tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
    struct { int M, N, K; uint32_t xr; int tiles_x; } ga{M, N, K, 0x88888888u, tm};
    char nm[40];
    snprintf(nm, sizeof nm, "qa8_%dx%d_m2_rf%d", WM, WN, rf);
    e.k(nm).buf(0, u).buf(1, W).buf(2, Sc).buf(3, Bz).buf(4, xs).buf(5, C).args(6, ga);
    if (rf) e.groups(S3(tm * tn), S3(32 * WM * WN));
    else e.groups(S3(tn, tm), S3(32 * WM * WN));
    e.end(); [cb commit]; [cb waitUntilCompleted];
    // Un-permute the int8 activations: slot 16c + 4t + j holds k = 16c + 8(t>>1) + 2j + (t&1)
    std::vector<int> aq((size_t)M * K);
    for (int m = 0; m < M; m++)
      for (int k = 0; k < K; k++) {
        int r = k & 15, c = (k & 63) >> 4, t = 2 * (r >> 3) + (r & 1), j = (r & 7) >> 1;
        aq[(size_t)m * K + k] = u.ptr<int8_t>()[(size_t)m * K + (k & ~63) + 16 * c + 4 * t + j];
      }
    double worst = 0, worstbf = 0, mag = 0;
    int bad = 0;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        double ref = 0, refbf = 0;
        for (int gi = 0; gi < kg; gi++) {
          const float* x2 = (const float*)(xs.ptr<char>() + ((size_t)gi * M + m) * 8);
          double s = bf(Sc.ptr<uint16_t>()[(size_t)n * kg + gi]), b = bf(Bz.ptr<uint16_t>()[(size_t)n * kg + gi]);
          long dq = 0, sq = 0;
          for (int k = gi * 64; k < gi * 64 + 64; k++) { dq += (long)q[(size_t)n * K + k] * aq[(size_t)m * K + k]; sq += aq[(size_t)m * K + k]; }
          ref += x2[0] * (s * dq + b * sq);
          for (int k = gi * 64; k < gi * 64 + 64; k++) refbf += (s * q[(size_t)n * K + k] + b) * bf(A.ptr<uint16_t>()[(size_t)m * K + k]);
        }
        double gv = C.ptr<float>()[(size_t)m * N + n];
        double d = std::isfinite(gv) ? fabs(gv - ref) : 1e30;
        if (d > 1e-3 * (fabs(ref) + 1)) { if (bad < 8) printf("bad m=%d n=%d gpu %.5f ref %.5f\n", m, n, gv, ref); bad++; }
        worst = std::max(worst, d), worstbf = std::max(worstbf, fabs(gv - refbf)), mag = std::max(mag, fabs(ref));
      }
    printf("%s M=%d N=%d K=%d: max|err| vs int8 ref %.3g, vs bf16 %.3g (max|ref| %.3g), bad %d/%d\n", nm, M, N, K, worst, worstbf, mag, bad, M * N);
  }
}
