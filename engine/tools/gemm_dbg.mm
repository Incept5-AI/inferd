// gemm_dbg <bits> <M> <N> <K> <TM> [S] [NSG]: random flipped weights + bf16 activations; a8_quant + w{bits}a8 kernel; compare to CPU.
#include "../src/gpu.h"
#include <random>
static float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t tobf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16); }
struct A8Args { int M, K; };
struct W4A8Args { int M, N, K, gps, split; };
int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    int bits = atoi(argv[1]), M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), TM = atoi(argv[5]);
    int S = argc > 6 ? atoi(argv[6]) : 1, NSG = argc > 7 ? atoi(argv[7]) : 4;
    int TN = 128, kg = K / 64;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1, 1);
    BufRef A(g.alloc((size_t)M * K * 2)), W(g.alloc((size_t)N * K * bits / 8)), Sc(g.alloc((size_t)N * kg * 2)), Bz(g.alloc((size_t)N * kg * 2));
    BufRef u(g.alloc((size_t)M * K)), xs(g.alloc((size_t)M * kg * 8)), C(g.alloc((size_t)S * M * N * 4));
    for (size_t i = 0; i < (size_t)M * K; i++) A.ptr<uint16_t>()[i] = tobf(U(rng));
    std::vector<uint8_t> q((size_t)N * K);
    for (auto& v : q) v = rng() & ((1 << bits) - 1);
    for (size_t i = 0; i < (size_t)N * kg; i++) { Sc.ptr<uint16_t>()[i] = tobf(0.01f + 0.01f * fabsf(U(rng))); Bz.ptr<uint16_t>()[i] = tobf(0.05f * U(rng)); }
    uint8_t* w = W.ptr<uint8_t>();
    for (size_t i = 0; i < (size_t)N * K; i++) {
      uint8_t f = bits == 8 ? q[i] ^ 0x80 : q[i] ^ 8;
      if (bits == 8) w[i] = f; else if (i % 2 == 0) w[i / 2] = f; else w[i / 2] |= f << 4;
    }
    memset(C.ptr<char>(), 0xff, (size_t)S * M * N * 4);  // poison
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    Enc e(g, cb);
    A8Args qa{M, K};
    e.k("a8_quant").buf(0, A).buf(1, u).buf(2, xs).args(3, qa);
    e.threads(S3(kg * 32, M), S3(256));
    W4A8Args a{M, N, K, kg / S, S};
    char name[48];
    if (NSG != 4) snprintf(name, sizeof name, "w4a8_%dx%ds%d_m%d", TM, TN, NSG, S > 1 ? 3 : 2);
    else snprintf(name, sizeof name, "w%da8_%dx%d_m%d", bits, TM, TN, S > 1 ? 3 : 2);
    e.k(name).buf(0, u).buf(1, W).buf(2, Sc).buf(3, Bz).buf(5, xs).buf(6, C).args(7, a);
    e.groups(S3((M + TM - 1) / TM, (N + TN - 1) / TN, S), S3(32 * NSG));
    e.end(); [cb commit]; [cb waitUntilCompleted];
    double worst = 0, mag = 0; int badrows = 0;
    for (int m = 0; m < M; m++) {
      double rw = 0;
      for (int n = 0; n < N; n++) {
        double ref = 0;
        for (int k = 0; k < K; k++) ref += (bf(Sc.ptr<uint16_t>()[(size_t)n * kg + k / 64]) * q[(size_t)n * K + k] + bf(Bz.ptr<uint16_t>()[(size_t)n * kg + k / 64])) * bf(A.ptr<uint16_t>()[(size_t)m * K + k]);
        double gv = 0; for (int z = 0; z < S; z++) gv += C.ptr<float>()[(size_t)z * M * N + (size_t)m * N + n];
        double d = std::isfinite(gv) ? fabs(gv - ref) : 1e30;
        rw = std::max(rw, d); mag = std::max(mag, fabs(ref));
      }
      if (rw > 0.05 * mag + 0.05) badrows++;
      worst = std::max(worst, rw);
      if (rw > 0.05 * mag + 0.05 && badrows <= 40) printf("%d ", m);
    }
    printf("\n%s M=%d N=%d K=%d S=%d: max|err| %.3g (max|ref| %.3g), bad rows %d/%d\n", name, M, N, K, S, worst, mag, badrows, M);
  }
}
