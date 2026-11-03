// fma_rate: ALU throughput of float vs half FMA chains on this GPU (is fp16 SIMD math faster than fp32?).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
static const char* src = R"(
#include <metal_stdlib>
using namespace metal;
inline float sumv(float x) { return x; } inline float sumv(half x) { return x; }
inline float sumv(float2 x) { return x.x + x.y; } inline float sumv(half2 x) { return x.x + x.y; }
template <typename T> kernel void fma_chain(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  T a0 = T(gid * 1e-7), a1 = a0 + T(0.1), a2 = a0 + T(0.2), a3 = a0 + T(0.3), a4 = a0 + T(0.4), a5 = a0 + T(0.5), a6 = a0 + T(0.6), a7 = a0 + T(0.7);
  const T m = T(0.999), c = T(0.0001);
  for (int i = 0; i < 4096; i++) {
    a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
    a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
  }
  T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7; out[gid] = sumv(s);
}
template [[host_name("f32")]] kernel void fma_chain<float>(device float*, uint);
template [[host_name("f16")]] kernel void fma_chain<half>(device float*, uint);
template [[host_name("f32x2")]] kernel void fma_chain<float2>(device float*, uint);
template [[host_name("f16x2")]] kernel void fma_chain<half2>(device float*, uint);
)";
int main() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
  if (!lib) { printf("%s\n", err.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [dev newCommandQueue];
  const int N = 1 << 22;
  id<MTLBuffer> out = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
  for (NSString* name in @[@"f32", @"f16", @"f32x2", @"f16x2"]) {
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    int lanes = [name hasSuffix:@"x2"] ? 2 : 1;
    for (int rep = 0; rep < 2; rep++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:out offset:0 atIndex:0];
      [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      double s = cb.GPUEndTime - cb.GPUStartTime;
      if (rep) printf("%-6s %.1f TFLOPS (FMA = 2 flops)\n", name.UTF8String, (double)N * 4096 * 8 * 2 * lanes / s / 1e12);
    }
  }
}
