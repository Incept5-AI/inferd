// disp_lat <n>: time n dependent tiny dispatches (scale_rows on 2,560 values) in one command buffer, serial vs
// concurrent encoder: the fixed cost per dispatch that bounds small-model decode steps.
#include "../src/gpu.h"
int main(int argc, char** argv) {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    const int n = argc > 1 ? atoi(argv[1]) : 1000;
    BufRef x = g.alloc(2560 * 2);
    memset(x.ptr<char>(), 0, 2560 * 2);
    struct { int n; float s; } a{2560, 1.0f};
    for (int conc = 0; conc < 2; conc++)
      for (int rep = 0; rep < 3; rep++) {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = conc ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                              : [cb computeCommandEncoder];
        id<MTLComputePipelineState> p = g.pso("scale_rows");
        [e setComputePipelineState:p];
        [e setBuffer:x.buf offset:0 atIndex:0];
        [e setBytes:&a length:sizeof a atIndex:1];
        for (int i = 0; i < n; i++) [e dispatchThreads:MTLSizeMake(2560, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (rep == 2) printf("%s: %d dispatches %.2f ms = %.2f us each\n", conc ? "concurrent" : "serial", n,
                             (cb.GPUEndTime - cb.GPUStartTime) * 1e3, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n);
      }
  }
}
