// ane_probe: can the Neural Engine take part of a prefill chunk's FFN? (2026-11-01)
// Compiles one MIL program through the private AppleNeuralEngine client (the service behind Core ML), with the FFN's
// weights as INPUTS (int8 rows + per-row fp16 scales in IOSurfaces the GPU could write a layer ahead), so one program
// serves every layer. Computes y = down(silu(gate x) * up x) for C intermediate channels over R tokens, checks a few
// outputs against the CPU, and times evaluations. Design after Splash's AneFfn (github.com/incoai/splash, Apache-2.0,
// runtime/ane/Program.mm, runtime/ops/AneFfn.cpp); this is our own minimal client.
//   ane_probe <R> <C> [reps]      R tokens (multiple of 64), C channels (multiple of 512); hidden 5120
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

@protocol PAneModel
+ (id)modelAtURL:(NSURL*)url key:(NSString*)key;
- (NSDictionary*)modelAttributes;
@end
@protocol PAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary*)options request:(id)request qos:(unsigned)qos
                    error:(NSError**)error;
@end
@protocol PAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol PAneRequest
+ (id)requestWithInputs:(NSArray*)inputs inputIndices:(NSArray*)inputIndices outputs:(NSArray*)outputs
          outputIndices:(NSArray*)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber*)procedure sharedEvents:(id)events transactionHandle:(NSNumber*)transaction;
@end

static const int H = 5120, SEG = 2560, DSEG = 2560;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// A surface of `rows` rows of `width` elements (1 = int8, 2 = fp16), rows padded to 64 bytes.
struct Surf {
  IOSurfaceRef s = nullptr;
  int rows = 0, width = 0, eb = 2, stride = 0;
  Surf() = default;
  Surf(int rows, int width, int eb) : rows(rows), width(width), eb(eb) {
    stride = (width * eb + 63) / 64 * 64;
    size_t size = ((size_t)stride * rows + 16383) / 16384 * 16384;
    NSDictionary* p = @{(id)kIOSurfaceWidth : @(width), (id)kIOSurfaceHeight : @(rows), (id)kIOSurfaceBytesPerElement : @(eb),
                        (id)kIOSurfaceBytesPerRow : @(stride), (id)kIOSurfaceAllocSize : @(size),
                        (id)kIOSurfacePixelFormat : @(eb == 1 ? 0x4c303038 : 0x4c303068)};
    s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
    if (!s || IOSurfaceGetBytesPerRow(s) != (size_t)stride) { fprintf(stderr, "IOSurface failed\n"); exit(1); }
  }
  char* row(int r) const { return (char*)IOSurfaceGetBaseAddress(s) + (size_t)r * stride; }
  std::string type(int r, int w) const {  // MIL tensor_buffer type of the leading r rows x w elements
    const size_t st = stride / eb, plane = (size_t)r * st;
    return std::string("tensor_buffer<") + (eb == 1 ? "int8" : "fp16") + ", shape=[1, 1, " + std::to_string(r) + ", " +
           std::to_string(w) + "], strides=[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " +
           std::to_string(st) + ", 1], interleave_factors=[1, 1, 1, 1]>";
  }
};

static std::string dims(int r, int w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
static std::string t16(int r, int w) { return "tensor<fp16, " + dims(r, w) + ">"; }

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: ane_probe <R> <C> [reps]\n"); return 1; }
    const int R = atoi(argv[1]), C = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 10;
    const bool a8 = getenv("A8") && atoi(getenv("A8"));  // int8 activations (dequantized in the program)
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      fprintf(stderr, "AppleNeuralEngine: %s\n", dlerror()); return 1;
    }
    Class cModel = NSClassFromString(@"_ANEModel"), cClient = NSClassFromString(@"_ANEClient");
    Class cSurf = NSClassFromString(@"_ANEIOSurfaceObject"), cReq = NSClassFromString(@"_ANERequest");
    if (!cModel || !cClient || !cSurf || !cReq) { fprintf(stderr, "ANE classes missing\n"); return 1; }
    id<PAneClient> client = [(Class<PAneClient>)cClient sharedConnection];

    // Surfaces: x (fp16 [H, R] as two segments of SEG channels, token-minor), weights int8 [C, SEG] x 2 for gate and
    // up, down int8 [H, DSEG] per DSEG channels of C, scales fp16 [C, 1] / [H, 1], output fp16 [H, R].
    const int nseg = H / SEG, ndown = C / DSEG + (C % DSEG ? 1 : 0);
    std::vector<Surf> xs, wg, wu, wd;
    std::vector<int> dw;
    for (int k = 0; k < nseg; k++) { xs.emplace_back(SEG, R, a8 ? 1 : 2); wg.emplace_back(C, SEG, 1); wu.emplace_back(C, SEG, 1); }
    for (int i = 0, b = 0; i < ndown; i++, b += DSEG) { dw.push_back(std::min(DSEG, C - b)); wd.emplace_back(H, dw.back(), 1); }
    Surf sg(C, 1, 2), su(C, 1, 2), sd(H, 1, 2), y(H, R, 2);

    // MIL: dequantize int8 weights by 1/128 (scales carry the rest), fp16 matmuls, silu as g/2 (1 + tanh(g/2)).
    std::string par, body;
    auto line = [&](const std::string& s) { body += "        " + s + ";\n"; };
    std::vector<std::pair<std::string, Surf*>> ins;
    for (int k = 0; k < nseg; k++) ins.push_back({"x" + std::to_string(k), &xs[k]});
    for (int k = 0; k < nseg; k++) { ins.push_back({"wg" + std::to_string(k), &wg[k]}); ins.push_back({"wu" + std::to_string(k), &wu[k]}); }
    ins.push_back({"sg", &sg}); ins.push_back({"su", &su});
    for (int i = 0; i < ndown; i++) ins.push_back({"wd" + std::to_string(i), &wd[i]});
    ins.push_back({"sd", &sd});
    for (auto& [n, s] : ins) {
      par += (par.empty() ? "" : ", ") + s->type(s->rows, s->width) + " " + n;
      line(std::string("tensor<") + (s->eb == 1 ? "int8" : "fp16") + ", " + dims(s->rows, s->width) + "> " + n +
           "_t = tensor_buffer_to_tensor<ios17>(input = " + n + ")");
    }
    const std::string unit = "fp16(0x1p-7)";
    auto mm = [&](const std::string& out, const std::string& w, int hgt, int wid, const std::string& x) {
      line(t16(hgt, wid) + " " + w + "_d = dequantize(input = " + w + "_t, scale = " + unit + ")");
      line(t16(hgt, R) + " " + out + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + w + "_d, y = " + x + ")");
    };
    if (a8) for (int k = 0; k < nseg; k++) line(t16(SEG, R) + " x" + std::to_string(k) + "_d = dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
    for (const char* p : {"g", "u"}) {
      std::string P = p;
      for (int k = 0; k < nseg; k++) mm(P + "m" + std::to_string(k), "w" + P + std::to_string(k), C, SEG, "x" + std::to_string(k) + (a8 ? "_d" : "_t"));
      std::string tot = P + "m0";
      for (int k = 1; k < nseg; k++) { line(t16(C, R) + " " + P + "a" + std::to_string(k) + " = add(x = " + tot + ", y = " + P + "m" + std::to_string(k) + ")"); tot = P + "a" + std::to_string(k); }
      line(t16(C, R) + " " + P + "s = mul(x = " + tot + ", y = s" + P + "_t)");
    }
    line(t16(C, R) + " gh = mul(x = gs, y = fp16(0x1p-1))");
    line(t16(C, R) + " th = tanh(x = gh)");
    line(t16(C, R) + " tp = add(x = th, y = fp16(0x1p+0))");
    line(t16(C, R) + " si = mul(x = gh, y = tp)");
    line(t16(C, R) + " h = mul(x = si, y = us)");
    for (int i = 0, b = 0; i < ndown; i++) {
      std::string I = std::to_string(i);
      line(t16(dw[i], R) + " h" + I + " = slice_by_size(x = h, begin = tensor<int32, [4]>([0, 0, " + std::to_string(b) +
           ", 0]), size = tensor<int32, [4]>(" + dims(dw[i], R) + "))");
      mm("dm" + I, "wd" + I, H, dw[i], "h" + I);
      b += dw[i];
    }
    std::string tot = "dm0";
    for (int i = 1; i < ndown; i++) { line(t16(H, R) + " da" + std::to_string(i) + " = add(x = " + tot + ", y = dm" + std::to_string(i) + ")"); tot = "da" + std::to_string(i); }
    line(t16(H, R) + " yt = mul(x = " + tot + ", y = sd_t)");
    line(y.type(H, R) + " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([" +
         std::to_string((size_t)H * y.stride / 2) + ", " + std::to_string((size_t)H * y.stride / 2) + ", " + std::to_string(y.stride / 2) + ", 1]))");
    std::string mil = "program(1.3)\n{\n    func main<ios18>(" + par + ") {\n" + body + "    } -> (y);\n}\n";

    // The model directory: model.mil + an (empty) weights.bin.
    char key[32];
    snprintf(key, sizeof key, "%016zx", std::hash<std::string>{}(mil));
    std::filesystem::path dir = std::filesystem::temp_directory_path() / ("inferd-ane-" + std::string(key));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "model.mil") << mil;
    { std::vector<char> blob(128, 0); uint32_t one = 1, two = 2; memcpy(blob.data(), &one, 4); memcpy(blob.data() + 4, &two, 4);
      std::ofstream(dir / "weights.bin", std::ios::binary).write(blob.data(), blob.size()); }
    id model = [(Class<PAneModel>)cModel modelAtURL:[NSURL fileURLWithPath:@(dir.c_str()) isDirectory:YES] key:@(key)];
    NSError* err = nil;
    double t0 = now();
    if (![client compiledModelExistsFor:model] &&
        ![client compileModel:model options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"} qos:21 error:&err]) {
      fprintf(stderr, "compile failed: %s\n%s", err.description.UTF8String, mil.substr(0, 3000).c_str()); return 1;
    }
    double t1 = now();
    if (![client loadModel:model options:@{} qos:21 error:&err]) { fprintf(stderr, "load failed: %s\n", err.description.UTF8String); return 1; }
    printf("compiled in %.2f s, loaded in %.2f s\n", t1 - t0, now() - t1);

    // Symbol indices from the model description.
    NSDictionary* desc = [model modelAttributes][@"ANEFModelDescription"];
    NSArray* symbols = desc[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary* proc = [desc[@"ANEFModelProcedures"] firstObject];
    NSArray* inIdx = proc[@"ANEFModelInputSymbolIndexArray"];
    NSNumber* outIdx = [proc[@"ANEFModelOutputSymbolIndexArray"] firstObject];
    NSMutableArray *objs = [NSMutableArray array], *idxs = [NSMutableArray array];
    for (NSNumber* sym in inIdx) {
      NSString* name = symbols[sym.unsignedIntValue];
      Surf* found = nullptr;
      for (auto& [n, s] : ins) if ([name isEqualToString:@(n.c_str())]) found = s;
      if (!found) { fprintf(stderr, "unknown input %s\n", name.UTF8String); return 1; }
      [objs addObject:[(Class<PAneSurface>)cSurf objectWithIOSurface:found->s]];
      [idxs addObject:sym];
    }
    id req = [(Class<PAneRequest>)cReq requestWithInputs:objs inputIndices:idxs outputs:@[ [(Class<PAneSurface>)cSurf objectWithIOSurface:y.s] ]
                                          outputIndices:@[ outIdx ] weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:nil transactionHandle:@0];

    // Random data: x ~ N(0, 1) fp16; int8 weights in [-127, 127]; scales so values stay O(1).
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    std::uniform_int_distribution<int> q(-127, 127);
    std::vector<float> X((size_t)H * R);
    for (int k = 0; k < nseg; k++)
      for (int c = 0; c < SEG; c++) for (int r = 0; r < R; r++) { float v = nd(rng);
        if (a8) { int8_t qv = (int8_t)std::max(-127.f, std::min(127.f, rintf(v * 32))); ((int8_t*)xs[k].row(c))[r] = qv; X[(size_t)(k * SEG + c) * R + r] = qv / 128.f; }
        else { X[(size_t)(k * SEG + c) * R + r] = (float)(_Float16)v; ((_Float16*)xs[k].row(c))[r] = v; } }
    auto fillw = [&](Surf& s) { for (int r = 0; r < s.rows; r++) for (int c = 0; c < s.width; c++) ((int8_t*)s.row(r))[c] = q(rng); };
    for (auto& s : wg) fillw(s); for (auto& s : wu) fillw(s); for (auto& s : wd) fillw(s);
    for (int c = 0; c < C; c++) { ((_Float16*)sg.row(c))[0] = 0.02f; ((_Float16*)su.row(c))[0] = 0.02f; }
    for (int c = 0; c < H; c++) ((_Float16*)sd.row(c))[0] = 0.002f;

    auto eval = [&] {
      NSError* e = nil;
      if (![client evaluateWithModel:model options:@{} request:req qos:21 error:&e]) { fprintf(stderr, "eval failed: %s\n", e.description.UTF8String); exit(1); }
    };
    eval();
    // CPU reference for a few (output, token) pairs.
    double maxrel = 0;
    for (int t : {0, R / 2, R - 1}) {
      std::vector<float> hh(C);
      for (int c = 0; c < C; c++) {
        double g = 0, u = 0;
        for (int k = 0; k < nseg; k++)
          for (int j = 0; j < SEG; j++) {
            float x = X[(size_t)(k * SEG + j) * R + t];
            g += ((int8_t*)wg[k].row(c))[j] / 128.0 * x;
            u += ((int8_t*)wu[k].row(c))[j] / 128.0 * x;
          }
        g *= 0.02; u *= 0.02;
        hh[c] = (float)(g / (1 + exp(-g)) * u);
      }
      for (int o : {0, 1, H / 2, H - 1}) {
        double acc = 0;
        for (int i = 0, b = 0; i < ndown; i++) { for (int j = 0; j < dw[i]; j++) acc += ((int8_t*)wd[i].row(o))[j] / 128.0 * hh[b + j]; b += dw[i]; }
        acc *= 0.002;
        double got = ((_Float16*)y.row(o))[t];
        maxrel = std::max(maxrel, fabs(got - acc) / (fabs(acc) + 1e-2));
        if (o == 0 && t == 0) printf("y[0,0] ane %.5f cpu %.5f\n", got, acc);
      }
    }
    printf("max rel err (12 samples) %.4f\n", maxrel);
    double best = 1e9, sum = 0;
    for (int i = 0; i < reps; i++) { double a = now(); eval(); double d = now() - a; best = std::min(best, d); sum += d; }
    const double flops = 2.0 * R * (3.0 * H * C);
    printf("R=%d C=%d (%d units of 512): best %.2f ms, mean %.2f ms, %.2f ms per unit, %.1f TFLOPS\n", R, C, C / 512,
           best * 1e3, sum / reps * 1e3, best * 1e3 / (C / 512.0), flops / best / 1e12);
  }
  return 0;
}
