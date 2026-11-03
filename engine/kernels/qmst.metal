// qmst: small-batch (Rows = 8 or 16) 4-bit GEMM, the staged-tile design of Splash's decode kernels
// (github.com/incoai/splash, runtime/metal/kernels/common/gguf_staged_tile.h, Apache-2.0), for MLX affine 4-bit g64:
//  - weights repacked at load (Qwen35::repack): for each 32-column tile and 32-input step, the 32 columns' 16-byte
//    nibble payloads are contiguous (one coalesced 512-byte load per simdgroup per step), and so are their (s, b)
//    per 64-group;
//  - each simdgroup owns 32 columns alone (execution_simdgroups<1>): lane l dequantizes column l's 32 inputs of the
//    step to fp16 *with the scale applied* (s q + b) into a double-buffered stage, so the matmul accumulates straight
//    through K with no per-group epilogue;
//  - lane l loads the next step's payload while the step's Rows x 32 x 32 matmul runs (bf16 activations x fp16
//    weights -> fp32), overlapping the DRAM round trip;
//  - each tile starts its walk over its K partition at its own step (golden-ratio spread), so tiles do not all wait
//    on the same freshly written slice of the activations.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;

struct QmstArgs { int M, N, K, steps, split, spread; };

template <int Rows, int NSG, int MODE>
kernel void qmst(device const bfloat* A [[buffer(0)]], device const uint4* W2 [[buffer(1)]],
                 device const bfloat2* SB [[buffer(2)]], device void* Cv [[buffer(4)]],
                 constant QmstArgs& p [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KS = 32, C = 32;
  const int K = p.K, N = p.N, KJ = K / KS, KG = K / 64;
  const int tile = int(tg.x) * NSG + int(sg);
  if (tile * C >= N) return;  // simdgroups are independent: no threadgroup barrier below
  threadgroup half stage[NSG][2][C * KS];
  const int jb = int(tg.z) * p.steps, je = min(KJ, jb + p.steps), nsteps = je - jb;
  if (nsteps <= 0) return;
  const int jf = p.spread ? jb + int((ulong((uint(tile) * 40503u) & 0xFFFFu) * uint(nsteps)) >> 16) : jb;

  auto a = tensor<device bfloat, DT2, tensor_inline>((device bfloat*)A, DT2(K, Rows));
  constexpr auto desc = matmul2d_descriptor(Rows, C, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto bt0 = tensor<threadgroup half, DT2, tensor_inline>(stage[sg][0], DT2(KS, C));
  auto bt1 = tensor<threadgroup half, DT2, tensor_inline>(stage[sg][1], DT2(KS, C));
  using AT = decltype(a.slice(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<AT, decltype(bt0), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;

  device const uint4* wcol = W2 + (size_t)tile * KJ * C + lane;     // step j: + j * 32
  device const bfloat2* mcol = SB + (size_t)tile * KG * C + lane;   // group g: + g * 32
  uint4 pk = wcol[(size_t)jf * C];
  float2 mt = float2(mcol[(size_t)(jf / 2) * C]);

  int j = jf;
  for (int walked = 0; walked < nsteps; walked++) {
    const int buf = walked & 1;
    // Dequantize column `lane`'s 32 inputs of step j: word e holds inputs 8e..8e+7 (nibble k at bits 4k).
    threadgroup half2* dst = (threadgroup half2*)(stage[sg][buf] + lane * KS);
    for (int e = 0; e < 4; e++)
      for (int b = 0; b < 4; b++) {
        const uint t = (pk[e] >> (8 * b)) & 0xFFu;
        const float2 code = float2(as_type<half2>(((t & 0xFu) | ((t & 0xF0u) << 12)) | 0x64006400u) - half2(1024.0h));
        dst[e * 4 + b] = half2(fma(code, float2(mt.x), float2(mt.y)));
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int jn = j + 1 < je ? j + 1 : jb;  // the walk wraps from je back to jb
    if (walked + 1 < nsteps) {
      pk = wcol[(size_t)jn * C];
      if (jn / 2 != j / 2) mt = float2(mcol[(size_t)(jn / 2) * C]);
    }
    auto as = a.slice(j * KS, 0);
    if (buf) op.run(as, bt1, acc); else op.run(as, bt0, acc);
    j = jn;
  }

  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto ix = acc.get_multidimensional_index(i);
    const int n = tile * C + ix[0], m = ix[1];
    if (n >= N || m >= p.M) continue;
    const size_t o = (size_t)m * N + n;
    if (MODE == 3) ((device float*)Cv)[(size_t)tg.z * p.M * N + o] = acc[i];
    else if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define QMST(R, NSG, MODE)                                                                                       \
  template [[host_name("qmst_r" #R "_s" #NSG "_m" #MODE)]] kernel void qmst<R, NSG, MODE>(device const bfloat*,   \
      device const uint4*, device const bfloat2*, device void*, constant QmstArgs&, uint3, uint, uint);
#define QMST_MODES(R, NSG) QMST(R, NSG, 0) QMST(R, NSG, 1) QMST(R, NSG, 2) QMST(R, NSG, 3)
QMST_MODES(8, 2) QMST_MODES(16, 2) QMST_MODES(8, 1) QMST_MODES(8, 4)
