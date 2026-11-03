// disp_test: GPU cost of a chain of N tiny dependent dispatches in one compute encoder (how much each kernel
// boundary costs when a step is ~1,400 small kernels): serial encoder vs a concurrent one with a buffer barrier
// between dispatches. usage: disp_test [N=1000]
#include "../src/gpu.h"
#include <chrono>
int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 1000;
  Gpu g(std::string(getenv("INFERD_HOME") ? getenv("INFERD_HOME") : ".") + "/build/core.metallib");
  BufRef b = g.alloc(1 << 20);
  struct { uint32_t n32, mask; } a{256, 1};
  id<MTLComputePipelineState> pso = g.pso("w_flip");
  for (int conc = 0; conc < 2; conc++)
    for (int rep = 0; rep < 3; rep++) {
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      id<MTLComputeCommandEncoder> e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                            : [cb computeCommandEncoder];
      for (int i = 0; i < N; i++) {
        [e setComputePipelineState:pso];
        [e setBuffer:b.buf offset:0 atIndex:0];
        [e setBytes:&a length:sizeof(a) atIndex:1];
        [e dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (conc) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      printf("%s: %d dependent tiny dispatches: %.2f ms GPU = %.2f us each\n", conc ? "concurrent+barrier" : "serial",
             N, gpu, gpu * 1e3 / N);
    }
}
