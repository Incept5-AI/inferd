// ane_rot_test: does the ANE program's block-Hadamard rotation (grouped 1x1 conv with a BLOBFILE constant) compute
// what ane_req_rot_cols assumes? (2026-11-01, debugging ANE_A8D.)   ane_rot_test <C> <R>
#include "../src/ane.h"
#include <cmath>
int main(int argc, char** argv) {
  @autoreleasepool {
    const int C = atoi(argv[1]), R = atoi(argv[2]);
    Gpu g("build/core.metallib");
    AneSurf h(g, C, R, 2), y(g, C + 1, R, 2), wd(g, 256, C, 1), z(g, 256, R, 2);
    std::string mil = "program(1.3)\n{\n    func main<ios18>(" + h.type(C, R) + " h, " + wd.type(256, C) + " wd) {\n";
    auto L = [&](const std::string& s) { mil += "        " + s + ";\n"; };
    const std::string c = std::to_string(C), r = std::to_string(R);
    L("tensor<fp16, [1, 1, " + c + ", " + r + "]> h_t = tensor_buffer_to_tensor<ios17>(input = h)");
    L("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h_t, shape = tensor<int32, [4]>([1, " + c + ", 1, " + r + "]))");
    L("tensor<fp16, [" + c + ", 512, 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", 512, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
    L("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" + std::to_string(C / 512) + "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)");
    L("tensor<fp16, [1, 1, " + c + ", " + r + "]> hr = reshape(x = hr4, shape = tensor<int32, [4]>([1, 1, " + c + ", " + r + "]))");
    L("tensor<fp16, [1, 1, " + c + ", " + r + "]> habs = abs(x = hr)");
    L("tensor<fp16, [1, 1, 1, " + r + "]> peak = reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    L("tensor<fp16, [1, 1, 1, " + r + "]> floor = maximum(x = peak, y = fp16(0x1p-9))");
    L("tensor<fp16, [1, 1, 1, " + r + "]> inverse = real_div(x = fp16(0x1.fcp+6), y = floor)");
    L("tensor<fp16, [1, 1, " + c + ", " + r + "]> hs = mul(x = hr, y = inverse)");
    L("tensor<int8, [1, 1, " + c + ", " + r + "]> hq = quantize(input = hs, scale = fp16(0x1p+0), output_dtype = string(\"int8\"))");
    L("tensor<fp16, [1, 1, " + c + ", " + r + "]> hd = dequantize(input = hq, scale = fp16(0x1p-7))");
    L("tensor<fp16, [1, 1, 1, " + r + "]> hscale = mul(x = floor, y = fp16(0x1.0204p+0))");
    L("tensor<int8, [1, 1, 256, " + c + "]> wd_t = tensor_buffer_to_tensor<ios17>(input = wd)");
    L("tensor<fp16, [1, 1, 256, " + c + "]> wd_d = dequantize(input = wd_t, scale = fp16(0x1p-7))");
    L("tensor<fp16, [1, 1, 256, " + r + "]> zt = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd_d, y = hd)");
    const size_t st = y.stride / 2;
    const size_t zs = z.stride / 2;
    L(z.type(256, R) + " y = tensor_to_tensor_buffer<ios17>(input = zt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([" + std::to_string(256 * zs) + ", " + std::to_string(256 * zs) + ", " + std::to_string(zs) + ", 1]))");
    mil += "    } -> (y);\n}\n";
    // blob: R[o][i] = H[o][i] / sqrt(512) (no signs)
    std::vector<char> blob(128 + (size_t)C * 512 * 2, 0);
    const uint32_t one = 1, two = 2, sen = 0xdeadbeef; const uint64_t bytes = (size_t)C * 512 * 2, data = 128;
    memcpy(blob.data(), &one, 4); memcpy(blob.data() + 4, &two, 4); memcpy(blob.data() + 64, &sen, 4); memcpy(blob.data() + 68, &one, 4);
    memcpy(blob.data() + 72, &bytes, 8); memcpy(blob.data() + 80, &data, 8);
    _Float16* rw = (_Float16*)(blob.data() + 128);
    for (int b = 0; b < C / 512; b++) for (int o = 0; o < 512; o++) for (int i = 0; i < 512; i++)
      rw[((size_t)b * 512 + o) * 512 + i] = (_Float16)((__builtin_popcount(o & i) & 1 ? -1.f : 1.f) * 0.044194174f);
    char key[32]; snprintf(key, sizeof key, "rot%016zx", std::hash<std::string>{}(mil));
    std::filesystem::path dir = std::filesystem::path(getenv("TMPDIR")) / key;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "model.mil") << mil;
    std::ofstream(dir / "weights.bin", std::ios::binary).write(blob.data(), blob.size());
    Class cModel = NSClassFromString(@"_ANEModel"), cClient = NSClassFromString(@"_ANEClient");
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    cModel = NSClassFromString(@"_ANEModel"); cClient = NSClassFromString(@"_ANEClient");
    id client = [(Class<AneClientP>)cClient sharedConnection];
    id model = [(Class<AneModelP>)cModel modelAtURL:[NSURL fileURLWithPath:@(dir.c_str()) isDirectory:YES] key:@(key)];
    NSError* err = nil;
    if (![client compileModel:model options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"} qos:21 error:&err]) { printf("compile: %s\n", err.description.UTF8String); return 1; }
    if (![client loadModel:model options:@{} qos:21 error:&err]) { printf("load: %s\n", err.description.UTF8String); return 1; }
    NSDictionary* desc = [model modelAttributes][@"ANEFModelDescription"];
    NSDictionary* proc = [desc[@"ANEFModelProcedures"] firstObject];
    Class cS = NSClassFromString(@"_ANEIOSurfaceObject"), cR = NSClassFromString(@"_ANERequest");
    NSArray* syms = desc[@"kANEFModelInputSymbolsArrayKey"];
    NSMutableArray* objs = [NSMutableArray array];
    for (NSNumber* sy in proc[@"ANEFModelInputSymbolIndexArray"])
      [objs addObject:[(Class<AneSurfaceP>)cS objectWithIOSurface:[syms[sy.unsignedIntValue] isEqualToString:@"h"] ? h.s : wd.s]];
    id req = [(Class<AneRequestP>)cR requestWithInputs:objs inputIndices:proc[@"ANEFModelInputSymbolIndexArray"]
        outputs:@[ [(Class<AneSurfaceP>)cS objectWithIOSurface:z.s] ] outputIndices:proc[@"ANEFModelOutputSymbolIndexArray"] weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:nil transactionHandle:@0];
    // h[c][t] = c == 3 ? 1 : 0 (a unit vector per token): hr[o][t] = H[o][3] / sqrt(512) = +-0.0442 within block 0
    _Float16* hp = h.buf.ptr<_Float16>();
    for (int cc = 0; cc < C; cc++) for (int t = 0; t < R; t++) hp[(size_t)cc * (h.stride / 2) + t] = (cc == 3 || cc == 512 + 5) ? 3.f : 0.f;
    for (int o = 0; o < 256; o++) for (int i = 0; i < C; i++) wd.buf.ptr<int8_t>()[(size_t)o * wd.stride + i] = (int8_t)(((o * 7 + i * 13) % 255) - 127);
    if (![client evaluateWithModel:model options:@{} request:req qos:21 error:&err]) { printf("eval: %s\n", err.description.UTF8String); return 1; }
    // CPU: hd[i] = sign of H[i%512][3 or 5] * 127/128 for the two blocks; z[o] = sum_i w[o][i] / 128 * hd[i]
    const _Float16* zp = z.buf.ptr<_Float16>();
    for (int o : {0, 1, 100}) {
      double acc = 0;
      for (int i = 0; i < C; i++) {
        const int b = i / 512, ii = i % 512;
        double hd = b == 0 ? ((__builtin_popcount(ii & 3) & 1) ? -1 : 1) * 127.0 / 128 : b == 1 ? ((__builtin_popcount(ii & 5) & 1) ? -1 : 1) * 127.0 / 128 : 0;
        acc += wd.buf.ptr<int8_t>()[(size_t)o * wd.stride + i] / 128.0 * hd;
      }
      printf("z[%d][t0] ane %.4f cpu %.4f\n", o, (float)zp[(size_t)o * zs + 0], acc);
    }
  }
}
