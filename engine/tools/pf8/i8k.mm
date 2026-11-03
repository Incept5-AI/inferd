// i8k: time i8k_<tile>_k<KS>_s<NSG> at M x N x K. usage: i8k <metallib> M N K [reps]
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 10;
    id<MTLBuffer> A = [dev newBufferWithLength:(size_t)M * K options:0], W = [dev newBufferWithLength:(size_t)N * K options:0];
    id<MTLBuffer> sx = [dev newBufferWithLength:(size_t)M * (K / 64) * 4 options:0], sw = [dev newBufferWithLength:(size_t)N * 4 options:0];
    id<MTLBuffer> C = [dev newBufferWithLength:(size_t)M * N * 2 options:0];
    for (size_t i = 0; i < (size_t)M * K; i++) ((int8_t*)A.contents)[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < (size_t)N * K; i++) ((int8_t*)W.contents)[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < (size_t)M * (K / 64); i++) ((float*)sx.contents)[i] = 1e-3f;
    for (int i = 0; i < N; i++) ((float*)sw.contents)[i] = 1e-3f;
    struct T { int tm, tn, nsg; } tiles[] = {{64, 64, 4}, {128, 64, 4}, {64, 128, 4}, {128, 128, 4}, {128, 128, 8}, {64, 128, 8}};
    int ks[] = {64, 128, 256, 512, 0};
    int mnk[3] = {M, N, K};
    for (auto t : tiles) {
      printf("%3dx%-3d s%d:", t.tm, t.tn, t.nsg);
      for (int k : ks) {
        std::string nm = "i8k_" + std::to_string(t.tm) + "x" + std::to_string(t.tn) + "_k" + std::to_string(k) + "_s" + std::to_string(t.nsg);
        id<MTLFunction> f = [lib newFunctionWithName:@(nm.c_str())];
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
        if (!p) { printf(" %s fail", nm.c_str()); continue; }
        double best = 1e9;
        for (int r = 0; r < reps; r++) {
          id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:p];
          [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:sx offset:0 atIndex:2];
          [e setBuffer:sw offset:0 atIndex:3]; [e setBuffer:C offset:0 atIndex:4]; [e setBytes:mnk length:12 atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake((M + t.tm - 1) / t.tm, (N + t.tn - 1) / t.tn, 1) threadsPerThreadgroup:MTLSizeMake(32 * t.nsg, 1, 1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
        }
        printf("  k%-3d %6.1f", k, 2.0 * M * N * K / best / 1e12);
      }
      printf("  TOPS\n");
    }
  }
}
