// Microbenchmarks for the M5 Ultra: memory bandwidth and MPP fp16 GEMM at decode shapes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pso(const char* name) {
  NSError* err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!f) { fprintf(stderr, "no function %s\n", name); exit(1); }
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
  if (!p) { fprintf(stderr, "pso %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
  return p;
}

// Median GPU time in seconds of `iters` command buffers, each holding `reps` dispatches.
template <class F>
static double time_gpu(F encode, int iters = 7, int reps = 10) {
  std::vector<double> t;
  for (int i = 0; i < iters + 1; i++) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int r = 0; r < reps; r++) encode(e);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (i > 0) t.push_back((cb.GPUEndTime - cb.GPUStartTime) / reps);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static id<MTLBuffer> buf(size_t bytes, bool fill = true) {
  id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (fill) {
    uint16_t* p = (uint16_t*)b.contents;
    for (size_t i = 0; i < bytes / 2; i++) p[i] = 0x3000 + (i % 251);  // small fp16 values
  }
  return b;
}

static void bench_bw() {
  auto p = pso("bw_read");
  for (size_t gb : {1, 4, 16}) {
    size_t bytes = gb << 30;
    auto src = buf(bytes, false);
    memset(src.contents, 0, bytes);
    auto out = buf(4 << 20, false);
    uint n4 = (uint)(bytes / 16);
    NSUInteger threads = 80 * 1024 * 4;  // ~4 waves of threads per core
    double s = time_gpu([&](id<MTLComputeCommandEncoder> e) {
      [e setComputePipelineState:p];
      [e setBuffer:src offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e setBytes:&n4 length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }, 5, 4);
    printf("bw_read %2zu GiB: %7.1f GB/s\n", gb, bytes / s / 1e9);
  }
}

struct Tile { int tm, tn; bool nt; };

static void bench_gemm(int M, int N, int K, Tile t) {
  char name[64];
  snprintf(name, sizeof name, "gemm_f16_%dx%d_%s_sg4", t.tm, t.tn, t.nt ? "true" : "false");
  auto p = pso(name);
  auto A = buf((size_t)M * K * 2), W = buf((size_t)N * K * 2), C = buf((size_t)M * N * 2, false);
  int mnk[4] = {M, N, K, 0};
  MTLSize grid = MTLSizeMake((N + t.tn - 1) / t.tn, (M + t.tm - 1) / t.tm, 1);
  NSUInteger w = p.threadExecutionWidth;
  double s = time_gpu([&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:p];
    [e setBuffer:A offset:0 atIndex:0];
    [e setBuffer:W offset:0 atIndex:1];
    [e setBuffer:C offset:0 atIndex:2];
    [e setBytes:mnk length:16 atIndex:3];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(w * 4, 1, 1)];
  });
  double flops = 2.0 * M * N * K, bytes = 2.0 * ((double)N * K + (double)M * K + (double)M * N);
  printf("gemm %-26s M=%5d N=%5d K=%5d  %8.3f ms  %6.1f TFLOPS  %6.1f GB/s\n", name, M, N, K, s * 1e3,
         flops / s / 1e12, bytes / s / 1e9);
}

static void bench_ty(const char* ty, int asz, int wsz, int osz, int tm, int tn, int M, int N, int K) {
  char name[64];
  snprintf(name, sizeof name, "gemm_%s_%dx%d", ty, tm, tn);
  auto p = pso(name);
  auto A = buf((size_t)M * K * asz, false), W = buf((size_t)N * K * wsz, false), C = buf((size_t)M * N * osz, false);
  memset(A.contents, 0x30, (size_t)M * K * asz); memset(W.contents, 0x30, (size_t)N * K * wsz);
  int mnk[4] = {M, N, K, 0};
  MTLSize grid = MTLSizeMake((N + tn - 1) / tn, (M + tm - 1) / tm, 1);
  double s = time_gpu([&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:p];
    [e setBuffer:A offset:0 atIndex:0];
    [e setBuffer:W offset:0 atIndex:1];
    [e setBuffer:C offset:0 atIndex:2];
    [e setBytes:mnk length:16 atIndex:3];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(p.threadExecutionWidth * 4, 1, 1)];
  });
  printf("gemm %-18s M=%5d N=%5d K=%5d  %8.3f ms  %6.1f T(FL)OPS\n", name, M, N, K, s * 1e3, 2.0 * M * N * K / s / 1e12);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice();
    q = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"kernels.metallib"] error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("%s, maxWorkingSet %.1f GiB\n", dev.name.UTF8String, dev.recommendedMaxWorkingSetSize / 1073741824.0);
    std::string what = argc > 1 ? argv[1] : "all";
    if (what == "all" || what == "bw") bench_bw();
    if (what == "ty") {
      for (int M : {4096, 128, 64}) {
        int N = M == 4096 ? 4096 : 34816, K = M == 4096 ? 4096 : 5120;
        for (int t : {128, 64}) {
          bench_ty("h", 2, 2, 2, t, 64, M, N, K);
          bench_ty("bf", 2, 2, 2, t, 64, M, N, K);
          bench_ty("i8", 1, 1, 4, t, 64, M, N, K);
          bench_ty("bfi8", 2, 1, 4, t, 64, M, N, K);
        }
      }
    }
    if (what == "all" || what == "gemm") {
      bench_gemm(4096, 4096, 4096, {64, 32, false});
      bench_gemm(4096, 4096, 4096, {64, 32, true});
      bench_gemm(4096, 4096, 4096, {64, 64, true});
      bench_gemm(4096, 4096, 4096, {128, 64, true});
      // 27B MLP up/gate projection: K=5120 -> N=17408, at decode batch sizes
      for (int M : {1, 8, 16, 32, 64, 128, 256, 512, 2048})
        for (Tile t : {Tile{16, 64, true}, Tile{16, 128, true}, Tile{32, 64, true}, Tile{32, 128, true}, Tile{64, 64, true}})
          if (t.tm <= std::max(M, 16) * 2) bench_gemm(M, 17408, 5120, t);
    }
  }
  return 0;
}
