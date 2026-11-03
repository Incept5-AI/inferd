// Thin Metal wrapper: device, queue, pipeline cache, and dispatch helpers.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <string>
#include <chrono>

struct Gpu {
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> lib;
  std::map<std::string, id<MTLComputePipelineState>> psos;
  id<MTLResidencySet> resident = nil;  // every buffer, attached to the queue (RESIDENCY=1; default none)
  int cores = 80;  // GPU cores (IORegistry AGXAccelerator "gpu-core-count"); M6 12, M5 Pro 20, M5 Ultra 80
  // GPU neural accelerators (Apple10 family: M5 and later). Without them (M4 Pro, 2026-11-01) MLX's NAX kernels we
  // vendor (mlx_qmm, mlx_attn) compute wrong results at prefill sizes, so callers route around them. NAX_GPU=0/1
  // overrides the detection.
  bool nax = true;

  explicit Gpu(const std::string& metallib) {
    dev = MTLCreateSystemDefaultDevice();
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
    if (svc) {
      CFTypeRef v = IORegistryEntryCreateCFProperty(svc, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (v) { CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &cores); CFRelease(v); }
      IOObjectRelease(svc);
    }
    nax = getenv("NAX_GPU") ? atoi(getenv("NAX_GPU")) != 0 : [dev supportsFamily:MTLGPUFamilyApple10];
    queue = [dev newCommandQueue];
    NSError* err = nil;
    // Without a residency set each command buffer makes every buffer it binds resident on submission: at 16
    // sequences x 5-row speculative blocks that cost 81 ms per cycle before the GPU started (2026-10-30).
    // On by default with the Neural Engine split (ANE_UNITS): its per-layer hand-offs leave the GPU idle between
    // command buffers, and on the 32 GB M6 macOS then unwired the weights between layers (wired 8-15 GiB of ~23).
    const char* rs = getenv("RESIDENCY");
    const char* au = getenv("ANE_UNITS");
    // Also on by default without neural accelerators: on the M3 Ultra (2026-11-01) a 140 GiB KV pool without it made
    // every server step pay for the pool (4k follow-up 2.8 tok/s; 100 with the set; 77 with a 64 GiB pool and no set).
    if (rs ? atoi(rs) : (au && atoi(au) > 0) || !nax) {  // otherwise opt-in on M5: no gain (16 x 6k spec, 2026-10-30)
      MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
      resident = [dev newResidencySetWithDescriptor:d error:&err];
      if (resident) { [resident requestResidency]; [queue addResidencySet:resident]; }
    }
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
    if (!lib) { fprintf(stderr, "metallib %s: %s\n", metallib.c_str(), err.localizedDescription.UTF8String); exit(1); }
  }

  // Pipeline creations so far and the time spent in them (the warm-up reports these; PSO_LOG=1 logs each one).
  int pso_created = 0;
  double pso_ms = 0;
  static inline bool pso_log = getenv("PSO_LOG") && atoi(getenv("PSO_LOG"));
  void pso_note(const std::string& key, std::chrono::steady_clock::time_point t0) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pso_created++;
    pso_ms += ms;
    if (pso_log) fprintf(stderr, "pso %s %.1f ms\n", key.c_str(), ms);
  }

  id<MTLComputePipelineState> pso(const std::string& name) {
    auto it = psos.find(name);
    if (it != psos.end()) return it->second;
    const auto t0 = std::chrono::steady_clock::now();
    NSError* err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!f) { fprintf(stderr, "no kernel %s\n", name.c_str()); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
    psos[name] = p;
    pso_note(name, t0);
    return p;
  }

  // Pipeline for a kernel specialised by boolean function constants {index, value}.
  id<MTLComputePipelineState> pso_fc(const std::string& name, const std::vector<std::pair<int, bool>>& fcs) {
    std::string key = name;
    for (auto& [i, v] : fcs) key += "_" + std::to_string(i) + (v ? "t" : "f");
    auto it = psos.find(key);
    if (it != psos.end()) return it->second;
    const auto t0 = std::chrono::steady_clock::now();
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    for (auto& [i, v] : fcs) {
      bool b = v;
      [cv setConstantValue:&b type:MTLDataTypeBool atIndex:i];
    }
    NSError* err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()] constantValues:cv error:&err];
    if (!f) { fprintf(stderr, "kernel %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
    psos[key] = p;
    pso_names[(__bridge void*)p] = name;  // PROF_KERNELS names specialised dispatches by their kernel
    pso_note(key, t0);
    return p;
  }
  std::unordered_map<void*, std::string> pso_names;

  id<MTLBuffer> alloc(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    if (!b) { fprintf(stderr, "alloc %zu bytes failed\n", bytes); exit(1); }
    if (getenv("ALLOC_LOG") && bytes >= (1ull << 30)) fprintf(stderr, "alloc %.2f GiB\n", bytes / 1073741824.0);
    if (resident) { [resident addAllocation:b]; [resident commit]; }
    return b;
  }
};

// A buffer plus a byte offset: the unit every kernel argument is bound as.
struct BufRef {
  id<MTLBuffer> buf = nil;
  size_t off = 0;
  BufRef() = default;
  BufRef(id<MTLBuffer> b, size_t o = 0) : buf(b), off(o) {}
  BufRef at(size_t bytes) const { return BufRef(buf, off + bytes); }
  template <class T> T* ptr() const { return (T*)((char*)buf.contents + off); }
};

// Encodes kernels into one compute encoder. Dispatches run in order (serial dispatch type).
struct Enc {
  Gpu& g;
  id<MTLComputeCommandEncoder> e;
  // PROF_KERNELS=1 (profiling only): every dispatch runs alone in its own command buffer and its GPU time is added
  // to prof[name]. Ordering against the caller's own command buffer is not kept, so results may be wrong.
  static inline std::map<std::string, double>* prof = getenv("PROF_KERNELS") ? new std::map<std::string, double> : nullptr;
  id<MTLCommandBuffer> pcb = nil;
  std::string pname;
  Enc(Gpu& g, id<MTLCommandBuffer> cb) : g(g) {
    if (prof) { pcb = [g.queue commandBuffer]; e = [pcb computeCommandEncoder]; }
    else e = [cb computeCommandEncoder];
  }
  void prof_flush() {
    [e endEncoding];
    [pcb commit];
    [pcb waitUntilCompleted];
    (*prof)[pname] += (pcb.GPUEndTime - pcb.GPUStartTime) * 1e3;
    pcb = [g.queue commandBuffer];
    e = [pcb computeCommandEncoder];
  }
  static inline std::map<std::string, uint64_t>* hist = getenv("DISPATCH_HIST") ? new std::map<std::string, uint64_t> : nullptr;
  Enc& k(const char* name) {
    if (hist) (*hist)[name]++;
    if (prof) pname = name;
    [e setComputePipelineState:g.pso(name)];
    return *this;
  }
  Enc& k(id<MTLComputePipelineState> p) {
    if (prof) { auto it = g.pso_names.find((__bridge void*)p); pname = it != g.pso_names.end() ? it->second : "(pipeline)"; }
    [e setComputePipelineState:p];
    return *this;
  }
  Enc& buf(int i, const BufRef& r) { [e setBuffer:r.buf offset:r.off atIndex:i]; return *this; }
  template <class T> Enc& args(int i, const T& v) { [e setBytes:&v length:sizeof(T) atIndex:i]; return *this; }
  static inline uint64_t dispatches = 0;  // running count of kernel dispatches (profiling)
  void groups(MTLSize grid, MTLSize tg) { dispatches++; [e dispatchThreadgroups:grid threadsPerThreadgroup:tg]; if (prof) prof_flush(); }
  void threads(MTLSize grid, MTLSize tg) { dispatches++; [e dispatchThreads:grid threadsPerThreadgroup:tg]; if (prof) prof_flush(); }
  void end() { [e endEncoding]; }
  // Continue on another command buffer (after the caller committed the current one).
  void restart(id<MTLCommandBuffer> cb) { if (!prof) e = [cb computeCommandEncoder]; }
};

inline MTLSize S3(NSUInteger x, NSUInteger y = 1, NSUInteger z = 1) { return MTLSizeMake(x, y, z); }
