// W4A8: 8-bit activations x the checkpoint's 4-bit weights on the tensor units' integer path (uint8 x uint4 ->
// int32 runs at ~1.8x the bf16 rate on the M5's neural accelerators). Weights stay in MLX's affine layout
// (Wq uint32[N, K/8], S/B bfloat[N, K/64], w = S*q + B); nothing is re-quantized.
// Activations: one int8 scale per row per 64-group (KL vs bf16 activations ~0.002 on real text, about the error
// of 8-bit weights; one scale per row is ~20x worse). The weights' nibbles are flipped in place at load
// (q ^ 8 = q - 8 as signed int4; signed int8 x int4 is ~15-20% faster than the unsigned pair), and every
// other kernel that reads them (qmv4) flips back. Per group g:
//   sum_k (S q + B)(sx xq) = S sx sum (q - 8) xq + (B + 8 S) sx sum xq
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;
constant constexpr int G = 64;

// In-place flip of a quantized weight matrix to signed values: 4-bit q ^ 8, 8-bit q ^ 0x80 (mask per word).
// grid: n32 threads.
struct FlipArgs { uint n32, mask; };
kernel void w_flip(device uint* Wq [[buffer(0)]], constant FlipArgs& p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid < p.n32) Wq[gid] ^= p.mask;
}

// Row x group quantization. One simdgroup per (row, group): 2 values per lane.
// U int8[M, K] = xq, XS float2[M, K/64] = (sx, sx * sum xq). grid (K/64 * 32, M), threadgroups of 256.
struct A8Args { int M, K; };
kernel void a8_quant(device const bfloat* A [[buffer(0)]], device char* U [[buffer(1)]],
                     device float2* XS [[buffer(2)]], constant A8Args& p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const int m = gid.y, g = gid.x / 32, kg = p.K / G;
  if (m >= p.M || g >= kg) return;
  const size_t o = (size_t)m * p.K + g * G + lane * 2;
  float2 v = float2(A[o], A[o + 1]);
  float amax = simd_max(max(fabs(v.x), fabs(v.y)));
  float sx = amax > 0 ? amax / 127.f : 1.f;
  float2 q = rint(v / sx);
  U[o] = char(q.x);
  U[o + 1] = char(q.y);
  float qs = simd_sum(q.x + q.y);
  if (lane == 0) XS[(size_t)m * kg + g] = float2(sx, sx * qs);
}

// C[M, N] (op)= dequant(A) * W^T. Tile TM x TN, 4 simdgroups. grid (ceil(M/TM), ceil(N/TN), S): the row tiles
// that share a weight tile are dispatched next to each other, so it is read from DRAM once. Split z takes
// groups [z * gps, (z + 1) * gps). S > 1 writes float partials [S, M, N] for splitk_reduce.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float, 3: split-K float partial at [z, M, N].
// (The mode must be a template parameter: a runtime branch in the store made the whole kernel 1.7x slower.)
struct W4A8Args { int M, N, K, gps, split; };
// BITS 4: int8 x int4 (W4A8); BITS 8: int8 x int8 (W8A8, MLX affine 8-bit, flipped q ^ 0x80 = q - 128).
template <int TM, int TN, int MODE, int BITS, int NSG = 4>
kernel void w4a8_gemm(device const char* U [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                      device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                      device const float2* XS [[buffer(5)]],
                      device void* Cv [[buffer(6)]], constant W4A8Args& p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K, kg = K / G;
  const int n0 = tg.y * TN, m0 = tg.x * TM;
  const int g0 = tg.z * p.gps, g1 = min(kg, g0 + p.gps);
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)U, DT2(K, M));
  using WE = metal::conditional_t<BITS == 4, int4b_format, int8_t>;
  using WP = metal::conditional_t<BITS == 4, uchar, int8_t>;  // int4b_format tensors take a byte pointer
  auto tW = tensor<device WE, DT2, tensor_inline>((device WP*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  constexpr float kOff = BITS == 4 ? 8.f : 128.f;
  auto part = op.template get_destination_cooperative_tensor<AT, WT, int>();
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  const int cap = acc.get_capacity();
  int nn[64], mm[64];
  for (uint16_t i = 0; i < cap; ++i) {
    acc[i] = 0;
    auto ix = acc.get_multidimensional_index(i);
    nn[i] = min(n0 + ix[0], N - 1) * kg;
    mm[i] = min(m0 + ix[1], M - 1) * kg;
  }
  for (int g = g0; g < g1; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      const float sw = float(S[nn[i] + g]), bw = float(B[nn[i] + g]) + kOff * sw;
      const float2 x = XS[mm[i] + g];
      acc[i] += sw * x.x * float(part[i]) + bw * x.y;
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (!acc.is_valid_element(i) || n >= N || m >= M) continue;
    const size_t o = (size_t)m * N + n;
    if (MODE == 3) ((device float*)Cv)[(size_t)tg.z * M * N + o] = acc[i];
    else if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define W4A8(TM, TN, MODE, BITS)                                                                            \
  template [[host_name("w" #BITS "a8_" #TM "x" #TN "_m" #MODE)]] kernel void w4a8_gemm<TM, TN, MODE, BITS>(  \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*, \
      device void*, constant W4A8Args&, uint3);
W4A8(32, 128, 0, 4) W4A8(32, 128, 1, 4) W4A8(32, 128, 2, 4) W4A8(32, 128, 3, 4)
W4A8(32, 128, 0, 8) W4A8(32, 128, 1, 8) W4A8(32, 128, 2, 8) W4A8(32, 128, 3, 8)
W4A8(48, 128, 0, 4) W4A8(48, 128, 1, 4) W4A8(48, 128, 2, 4) W4A8(48, 128, 3, 4)
W4A8(16, 128, 0, 4) W4A8(16, 128, 1, 4) W4A8(16, 128, 2, 4) W4A8(16, 128, 3, 4)
W4A8(8, 128, 0, 4) W4A8(8, 128, 1, 4) W4A8(8, 128, 2, 4) W4A8(8, 128, 3, 4)
W4A8(16, 128, 0, 8) W4A8(16, 128, 1, 8) W4A8(16, 128, 2, 8) W4A8(16, 128, 3, 8)
W4A8(16, 64, 0, 4) W4A8(16, 64, 1, 4) W4A8(16, 64, 2, 4) W4A8(16, 64, 3, 4)
W4A8(16, 256, 0, 4) W4A8(16, 256, 1, 4) W4A8(16, 256, 2, 4) W4A8(16, 256, 3, 4)
W4A8(40, 128, 0, 4) W4A8(40, 128, 1, 4) W4A8(40, 128, 2, 4) W4A8(40, 128, 3, 4)

#define W4A8G(TM, TN, MODE, NSG)                                                                              \
  template [[host_name("w4a8_" #TM "x" #TN "s" #NSG "_m" #MODE)]] kernel void w4a8_gemm<TM, TN, MODE, 4, NSG>(  \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*, \
      device void*, constant W4A8Args&, uint3);
W4A8G(16, 32, 0, 1) W4A8G(16, 32, 1, 1) W4A8G(16, 32, 2, 1) W4A8G(16, 32, 3, 1)
W4A8G(16, 64, 0, 1) W4A8G(16, 64, 1, 1) W4A8G(16, 64, 2, 1) W4A8G(16, 64, 3, 1)
W4A8G(16, 64, 0, 2) W4A8G(16, 64, 1, 2) W4A8G(16, 64, 2, 2) W4A8G(16, 64, 3, 2)
W4A8G(32, 128, 0, 8) W4A8G(32, 128, 1, 8) W4A8G(32, 128, 2, 8) W4A8G(32, 128, 3, 8)
W4A8G(64, 128, 0, 8) W4A8G(64, 128, 1, 8) W4A8G(64, 128, 2, 8) W4A8G(64, 128, 3, 8)

