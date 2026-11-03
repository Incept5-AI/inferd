// 4-bit GEMM kernels: correctness against a CPU reference, then timing at decode shapes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pso(const std::string& name) {
  NSError* err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!f) { fprintf(stderr, "no function %s\n", name.c_str()); exit(1); }
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
  if (!p) { fprintf(stderr, "pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
  return p;
}

static id<MTLBuffer> buf(size_t bytes) { return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; }

struct Problem {
  int M, N, K;
  id<MTLBuffer> A, Wq, S, Bz, C, Asum;
  Problem(int M, int N, int K, uint32_t seed) : M(M), N(N), K(K) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ua(-1, 1), us(0.001f, 0.02f);
    A = buf((size_t)M * K * 2); Wq = buf((size_t)N * K / 2); S = buf((size_t)N * K / 64 * 2);
    Bz = buf((size_t)N * K / 64 * 2); C = buf((size_t)M * N * 2); Asum = buf((size_t)M * K / 64 * 4);
    auto a = (__fp16*)A.contents; auto s = (__fp16*)S.contents; auto b = (__fp16*)Bz.contents;
    auto w = (uint32_t*)Wq.contents; auto as = (float*)Asum.contents;
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = ua(rng);
    for (size_t i = 0; i < (size_t)N * K / 8; i++) w[i] = rng();
    for (size_t i = 0; i < (size_t)N * K / 64; i++) { s[i] = us(rng); b[i] = -8 * (float)s[i]; }
    for (int m = 0; m < M; m++)
      for (int g = 0; g < K / 64; g++) {
        float t = 0;
        for (int k = 0; k < 64; k++) t += a[(size_t)m * K + g * 64 + k];
        as[(size_t)m * (K / 64) + g] = t;
      }
  }
  float ref(int m, int n) const {
    auto a = (const __fp16*)A.contents; auto s = (const __fp16*)S.contents; auto b = (const __fp16*)Bz.contents;
    auto w = (const uint32_t*)Wq.contents;
    double t = 0;
    for (int k = 0; k < K; k++) {
      uint32_t q4 = (w[(size_t)n * K / 8 + k / 8] >> (4 * (k % 8))) & 15;
      float wv = (float)s[(size_t)n * K / 64 + k / 64] * q4 + (float)b[(size_t)n * K / 64 + k / 64];
      t += (double)(float)a[(size_t)m * K + k] * wv;
    }
    return t;
  }
};

static double run(const std::string& name, Problem& P, int tm, int tn, int iters, int reps) {
  if (getenv("QTRACE")) fprintf(stderr, "run %s M=%d\n", name.c_str(), P.M);
  auto p = pso(name);
  if (getenv("QTRACE")) fprintf(stderr, "  pso ok\n");
  int mnk[4] = {P.M, P.N, P.K, 0};
  MTLSize grid = MTLSizeMake((P.N + tn - 1) / tn, (P.M + tm - 1) / tm, 1);
  std::vector<double> t;
  for (int i = 0; i < iters + 1; i++) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int r = 0; r < reps; r++) {
      [e setComputePipelineState:p];
      [e setBuffer:P.A offset:0 atIndex:0];
      [e setBuffer:P.Wq offset:0 atIndex:1];
      [e setBuffer:P.S offset:0 atIndex:2];
      [e setBuffer:P.Bz offset:0 atIndex:3];
      [e setBuffer:P.C offset:0 atIndex:4];
      [e setBytes:mnk length:16 atIndex:5];
      [e setBuffer:P.Asum offset:0 atIndex:6];
      [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "%s: %s\n", name.c_str(), cb.error.localizedDescription.UTF8String); exit(1); }
    if (i > 0) t.push_back((cb.GPUEndTime - cb.GPUStartTime) / reps);
  }
  if (t.empty()) return 0;
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

struct K { const char* fam; int tm, tn; };
static const K kernels[] = {
  {"qgemm_tg", 16, 64}, {"qgemm_tg", 32, 64}, {"qgemm_tg", 64, 64}, {"qgemm_tg", 32, 128}, {"qgemm_tg", 64, 128},
  {"qgemm_hw", 16, 64}, {"qgemm_hw", 32, 64}, {"qgemm_hw", 64, 64}, {"qgemm_hw", 32, 128}, {"qgemm_hw", 64, 128},
};
static std::string kname(const K& k) { return std::string(k.fam) + "_" + std::to_string(k.tm) + "x" + std::to_string(k.tn); }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice();
    q = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"qgemm.metallib"] error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    // Correctness: odd M and N not a multiple of the tile, so edges are exercised.
    Problem small(37, 200, 512, 1);
    bool ok = true;
    for (auto& k : kernels) {
      memset(small.C.contents, 0, small.M * small.N * 2);
      run(kname(k), small, k.tm, k.tn, 0, 1);
      auto c = (const __fp16*)small.C.contents;
      double maxerr = 0, maxref = 0;
      for (int m = 0; m < small.M; m++)
        for (int n = 0; n < small.N; n++) {
          double r = small.ref(m, n);
          maxerr = std::max(maxerr, std::fabs(r - (float)c[m * small.N + n]));
          maxref = std::max(maxref, std::fabs(r));
        }
      if (getenv("QDUMP")) for (int n = 0; n < 6; n++) printf("  m0 n%d ref %.4f got %.4f | m1 ref %.4f got %.4f\n", n, small.ref(0, n), (float)c[n], small.ref(1, n), (float)c[small.N + n]);
      bool pass = maxerr <= 2e-3 * maxref + 1e-2;
      ok &= pass;
      printf("check %-18s maxerr %.4f (max |ref| %.2f) %s\n", kname(k).c_str(), maxerr, maxref, pass ? "ok" : "FAIL");
    }
    if (!ok && !(argc > 1 && std::string(argv[1]) == "force")) return 1;
    // Timing: 27B MLP gate/up projection, 5120 -> 17408.
    const int N = 17408, Kd = 5120;
    for (int M : {1, 16, 32, 64, 128, 256, 1024}) {
      Problem P(M, N, Kd, 2);
      for (auto& k : kernels) {
        if (k.tm > std::max(M, 16) * 2) continue;
        double s = run(kname(k), P, k.tm, k.tn, 5, 10);
        double bytes = (double)N * Kd / 2 + (double)N * Kd / 64 * 4 + (double)M * Kd * 2 + (double)M * N * 2;
        printf("M=%5d %-18s %8.4f ms  %6.1f TFLOPS  %6.1f GB/s\n", M, kname(k).c_str(), s * 1e3,
               2.0 * M * N * Kd / s / 1e12, bytes / s / 1e9);
      }
    }
  }
  return 0;
}
