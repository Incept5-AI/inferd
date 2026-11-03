// w4a8: times the W4A8 prototypes (w4a8.metal) on the 27B projection shapes. Weights cycle through 8 copies
// (> the GPU's caches) so the weight stream comes from DRAM.
#import <Metal/Metal.h>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
static id<MTLDevice> dev; static id<MTLCommandQueue> q; static id<MTLLibrary> lib;
static id<MTLComputePipelineState> pso(const char* n) {
  NSError* e = nil;
  id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:n]];
  if (!f) { fprintf(stderr, "no %s\n", n); exit(1); }
  auto p = [dev newComputePipelineStateWithFunction:f error:&e];
  if (!p) { fprintf(stderr, "%s: %s\n", n, e.localizedDescription.UTF8String); exit(1); }
  return p;
}
static bool RANDOM = getenv("RANDOM_DATA") != nullptr;
static id<MTLBuffer> buf(size_t n, int fill) {
  auto b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
  memset(b.contents, fill, n);
  if (RANDOM && fill) { uint32_t x = 12345; uint8_t* p = (uint8_t*)b.contents; for (size_t i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; p[i] = (x >> 24); } }
  return b;
}
int main() {
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice(); q = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"w4a8.metallib"] error:&err];
    if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
    struct Sh { int N, K; } shapes[] = {{34816, 5120}};
    const int COPIES = 8;
    for (auto sh : shapes) {
      int N = sh.N, K = sh.K, kg = K / 64;
      std::vector<id<MTLBuffer>> W, S;
      for (int c = 0; c < COPIES; c++) { W.push_back(buf((size_t)N * K / 2, 0x35)); S.push_back(buf((size_t)N * kg * 8, 0x3c)); }
      auto X = buf((size_t)256 * K, 3), Sx = buf((size_t)256 * kg * 8, 0), Xs = buf((size_t)256 * kg * 4, 0);
      auto C = buf((size_t)256 * N * 4, 0);
      double wbytes = (double)N * K * (0.5 + 4.0 / 64);
      for (int M : {40, 80}) {
        for (const char* v : {"v1", "eng", "engA", "engB"})
          for (auto t : std::vector<std::pair<int, int>>{{32, 128}}) {
            if (t.first > 2 * std::max(M, 32)) continue;
            char name[64];
            bool eng = !strncmp(v, "eng", 3);
            if (eng) snprintf(name, sizeof name, "%s_%dx%d_m0", v, t.first, t.second);
            else snprintf(name, sizeof name, "w4a8_%s_%dx%d", v, t.first, t.second);
            if ((!strcmp(v, "v0") || !strcmp(v, "v1")) && (t.first == 16 || t.second == 256)) continue;
            auto p = pso(name);
            int mnk[8] = {M, N, K, eng ? K / 64 : 0, 1, 0, 0, 0};
            std::vector<double> ts;
            for (int it = 0; it < 6; it++) {
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
              for (int r = 0; r < 16; r++) {
                [e setComputePipelineState:p];
                [e setBuffer:X offset:0 atIndex:0];
                [e setBuffer:W[r % COPIES] offset:0 atIndex:1];
                [e setBuffer:S[r % COPIES] offset:0 atIndex:2];
                [e setBuffer:S[r % COPIES] offset:0 atIndex:3];
                [e setBuffer:Sx offset:0 atIndex:4];
                [e setBuffer:Xs offset:0 atIndex:5];
                [e setBuffer:C offset:0 atIndex:6];
                [e setBytes:mnk length:32 atIndex:7];
                [e dispatchThreadgroups:eng ? MTLSizeMake((M + t.first - 1) / t.first, (N + t.second - 1) / t.second, 1) : MTLSizeMake((N + t.second - 1) / t.second, (M + t.first - 1) / t.first, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
              }
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              if (it >= 2) ts.push_back((cb.GPUEndTime - cb.GPUStartTime) / 16);
            }
            std::sort(ts.begin(), ts.end());
            double s = ts[ts.size() / 2];
            printf("N=%6d K=%6d M=%4d %-18s %8.4f ms %6.0f GB/s %6.1f TOPS\n", N, K, M, name, s * 1e3, wbytes / s / 1e9,
                   2.0 * M * N * K / s / 1e12);
          }
      }
    }
  }
  return 0;
}
