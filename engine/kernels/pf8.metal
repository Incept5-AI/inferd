// PF8 (engine-pf, 2026-11-02): a group-free int8 GEMM for large prefill steps.
// The integer tensor path runs int8 x int8 at ~1.8x the bf16 rate only when matmul2d consumes K in long runs (laptop
// M5 Max, 4096 x 6144 x 2560: K run 64 -> 32-47 TOPS, 256 -> 72, 512 -> 83, all of K -> 87-99; tools/pf8/i8k.metal).
// W4A8's per-row-per-64-group activation scales force 64-wide runs, so it loses to MLX's bf16 qmm at prefill sizes.
// Group-free scaling needs one activation scale per row, which plain int8 cannot carry (KL 0.049); a random-sign
// block Hadamard rotation (512) spreads the outliers first (QuaRot-style):  x W^T = (x D H)(W D H)^T.
//   pf8_wrot (load time, once per linear): dequantize a row of W (MLX affine 4/8-bit, group 64, possibly flipped),
//     rotate each 512-block, requantize to int8 with one scale per output row.
//   pf8_act: rotate each 512-block of an activation row, one int8 scale per row.
//   pf8_gemm: C[M,N] = (sum_k A8 W8) * sx[m] * sw[n], one matmul2d run over all of K.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;

// The rotation of one 512-block held by a simdgroup: lane L owns elements 16L..16L+15 (already multiplied by the
// signs D). Unnormalised Walsh-Hadamard (Sylvester order): 4 in-lane stages, 5 cross-lane stages.
inline void wht512(thread float (&v)[16], uint lane) {
  for (int h = 1; h < 16; h <<= 1)
    for (int i = 0; i < 16; i++)
      if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
  for (int s = 0; s < 5; s++) {
    const bool hi = (lane >> s) & 1;
    for (int i = 0; i < 16; i++) {
      const float a = v[i], b = simd_shuffle_xor(a, ushort(1 << s));
      v[i] = hi ? b - a : a + b;
    }
  }
}

// Load-time weight rotation + requantization. One threadgroup per output row, 32 x NSG threads; simdgroup s handles
// blocks s, s + NSG, ... in two passes (row absmax, then quantize). sgn: 512 signs (+-1).
struct Pf8WArgs { int K, bits, xr; };  // xr: the flip mask of the stored nibbles/bytes (0 if not flipped)
kernel void pf8_wrot(device const uint8_t* Wq [[buffer(0)]], device const bfloat* S [[buffer(1)]],
                     device const bfloat* B [[buffer(2)]], device const float* sgn [[buffer(3)]],
                     device int8_t* out [[buffer(4)]], device float* osc [[buffer(5)]], constant Pf8WArgs& p [[buffer(6)]],
                     uint row [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float red[32];
  const int K = p.K, nb = K / 512, kg = K / 64;
  const size_t rb = (size_t)K * p.bits / 8;
  auto load = [&](int blk, thread float (&v)[16]) {
    const int k0 = blk * 512 + lane * 16, g = k0 / 64;
    const float s = S[(size_t)row * kg + g], b = B[(size_t)row * kg + g];
    for (int i = 0; i < 16; i++) {
      const int k = k0 + i;
      int q;
      if (p.bits == 4) q = ((Wq[(size_t)row * rb + k / 2] >> (4 * (k & 1))) & 15) ^ (p.xr & 15);
      else q = Wq[(size_t)row * rb + k] ^ (p.xr & 255);
      v[i] = (s * q + b) * sgn[lane * 16 + i];
    }
    wht512(v, lane);
  };
  float m = 0;
  for (int blk = sg; blk < nb; blk += nsg) {
    float v[16];
    load(blk, v);
    for (int i = 0; i < 16; i++) m = max(m, fabs(v[i]));
  }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = 0;
  for (uint i = 0; i < nsg; i++) m = max(m, red[i]);
  const float sc = m > 0 ? m / 127.f : 1.f;
  for (int blk = sg; blk < nb; blk += nsg) {
    float v[16];
    load(blk, v);
    for (int i = 0; i < 16; i++) out[(size_t)row * K + blk * 512 + lane * 16 + i] = int8_t(clamp(rint(v[i] / sc), -127.f, 127.f));
  }
  // the rotation is unnormalised on both sides: fold 1/512 into the weight scale
  if (sg == 0 && lane == 0) osc[row] = sc / 512.f;
}

// Activations: one threadgroup per row, NSG simdgroups; simdgroup s holds blocks s and s + NSG in registers (K <= 2 x
// 512 x NSG), row absmax over the threadgroup, then int8. grid (M) x (32 NSG).
struct Pf8AArgs { int K, lda; };
kernel void pf8_act(device const bfloat* A [[buffer(0)]], device const float* sgn [[buffer(1)]],
                    device int8_t* out [[buffer(2)]], device float* sx [[buffer(3)]], constant Pf8AArgs& p [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float red[32];
  const int K = p.K, nb = K / 512;
  float v[2][16];
  float d[16];
  for (int i = 0; i < 16; i++) d[i] = sgn[lane * 16 + i];
  float m = 0;
  for (int j = 0; j < 2; j++) {
    const int blk = sg + j * nsg;
    if (blk >= nb) { for (int i = 0; i < 16; i++) v[j][i] = 0; continue; }
    device const bfloat* a = A + (size_t)row * p.lda + blk * 512 + lane * 16;
    for (int i = 0; i < 16; i++) v[j][i] = float(a[i]) * d[i];
    wht512(v[j], lane);
    for (int i = 0; i < 16; i++) m = max(m, fabs(v[j][i]));
  }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = 0;
  for (uint i = 0; i < nsg; i++) m = max(m, red[i]);
  const float sc = m > 0 ? m / 127.f : 1.f;
  for (int j = 0; j < 2; j++) {
    const int blk = sg + j * nsg;
    if (blk >= nb) continue;
    device char4* o = (device char4*)(out + (size_t)row * K + blk * 512 + lane * 16);
    for (int i = 0; i < 4; i++)
      o[i] = char4(int8_t(clamp(rint(v[j][4 * i] / sc), -127.f, 127.f)), int8_t(clamp(rint(v[j][4 * i + 1] / sc), -127.f, 127.f)),
                   int8_t(clamp(rint(v[j][4 * i + 2] / sc), -127.f, 127.f)), int8_t(clamp(rint(v[j][4 * i + 3] / sc), -127.f, 127.f)));
  }
  if (sg == 0 && lane == 0) sx[row] = sc;
}

// C[M, N] (op)= (A8 . W8^T) sx[m] sw[n]. Tile TM x TN on NSG simdgroups, one matmul2d run over all of K.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float. grid (ceil(M / TM), ceil(N / TN)).
struct Pf8GArgs { int M, N, K, ldc; };
template <int TM, int TN, int NSG, int MODE>
kernel void pf8_gemm(device const int8_t* A [[buffer(0)]], device const int8_t* W [[buffer(1)]],
                     device const float* sx [[buffer(2)]], device const float* sw [[buffer(3)]],
                     device void* Cv [[buffer(4)]], constant Pf8GArgs& p [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K;
  const int m0 = tg.x * TM, n0 = tg.y * TN;
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)A, DT2(K, M));
  auto tW = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)W, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto mA = tA.slice(0, m0);
  auto mW = tW.slice(0, n0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), int>();
  op.run(mA, mW, acc);
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (n >= N || m >= M) continue;
    const float v = float(acc[i]) * sx[m] * sw[n];
    const size_t o = (size_t)m * p.ldc + n;
    if (MODE == 2) ((device float*)Cv)[o] = v;
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(v);
  }
}
#define PF8G(TM, TN, NSG, MODE)                                                                                     \
  template [[host_name("pf8_" #TM "x" #TN "s" #NSG "_m" #MODE)]] kernel void pf8_gemm<TM, TN, NSG, MODE>(           \
      device const int8_t*, device const int8_t*, device const float*, device const float*, device void*,          \
      constant Pf8GArgs&, uint3);
PF8G(128, 128, 8, 0) PF8G(128, 128, 8, 1) PF8G(128, 128, 8, 2)
PF8G(64, 128, 4, 0) PF8G(64, 128, 4, 1) PF8G(64, 128, 4, 2)
PF8G(128, 64, 4, 0) PF8G(128, 64, 4, 1) PF8G(128, 64, 4, 2)
