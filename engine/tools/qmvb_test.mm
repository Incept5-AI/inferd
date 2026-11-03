// qmvb_test <bits> <M> <N> <K>: random MLX-packed <bits>-bit weights (group 64) and bf16 activations through the
// qmv<bits> kernel, compared with a CPU reference. Checks the 5/6-bit GEMV (kernels/qmvb.metal).
#include <random>
#include "../src/gpu.h"
static float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t tobf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16); }
struct GemmArgs { int M, N, K, xr; };
int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    const int bits = atoi(argv[1]), M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), KG = K / 64;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(0, (1 << bits) - 1);
    std::uniform_real_distribution<float> fd(-1, 1);
    const size_t rowb = (size_t)K * bits / 8;
    BufRef W = g.alloc(N * rowb), S = g.alloc((size_t)N * KG * 2), B = g.alloc((size_t)N * KG * 2);
    BufRef A = g.alloc((size_t)M * K * 2), C = g.alloc((size_t)M * N * 2);
    memset(W.ptr<char>(), 0, N * rowb);
    std::vector<int> q((size_t)N * K);
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++) {
        int v = q[(size_t)n * K + k] = qd(rng);
        for (int b = 0; b < bits; b++)
          if ((v >> b) & 1) W.ptr<uint8_t>()[n * rowb + ((size_t)k * bits + b) / 8] |= 1u << (((size_t)k * bits + b) % 8);
      }
    for (size_t i = 0; i < (size_t)N * KG; i++) { S.ptr<uint16_t>()[i] = tobf(fd(rng) * 0.02f); B.ptr<uint16_t>()[i] = tobf(fd(rng) * 0.1f); }
    for (size_t i = 0; i < (size_t)M * K; i++) A.ptr<uint16_t>()[i] = tobf(fd(rng));
    char name[32];
    snprintf(name, sizeof name, "qmv%d_r%d_m0", bits, M <= 1 ? 1 : 2);
    GemmArgs a{M, N, K, 0};
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    Enc e(g, cb);
    e.k(name).buf(0, A).buf(1, W).buf(2, S).buf(3, B).buf(4, C).args(5, a);
    e.groups(S3((M + 1) / 2 * (M <= 1 ? 1 : 1), (N + 7) / 8), S3(64));
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
    double maxe = 0, maxr = 0;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        double ref = 0;
        for (int k = 0; k < K; k++)
          ref += bf(A.ptr<uint16_t>()[m * K + k]) *
                 (q[(size_t)n * K + k] * bf(S.ptr<uint16_t>()[n * KG + k / 64]) + bf(B.ptr<uint16_t>()[n * KG + k / 64]));
        maxe = std::max(maxe, std::fabs(ref - bf(C.ptr<uint16_t>()[m * N + n])));
        maxr = std::max(maxr, std::fabs(ref));
      }
    printf("%s M=%d N=%d K=%d: max |err| %.4f (max |ref| %.2f) %s\n", name, M, N, K, maxe, maxr, maxe < 0.01 * maxr + 0.05 ? "OK" : "FAIL");
  }
}
