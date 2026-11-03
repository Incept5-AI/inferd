// W4A8 prototypes: int8 activations x signed int4 weights (MLX nibble layout, nibbles XOR 8) -> int32 on the
// tensor units, with the 64-group affine epilogue. Synthetic data; speed only.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;
constant constexpr int G = 64;

// v0: no epilogue, full-K int MMA (the ceiling for this data path).
template <int TM, int TN>
kernel void w4a8_v0(device const int8_t* X [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                    device int* C [[buffer(6)]], constant int3& mnk [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)X, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  auto tC = tensor<device int, DT2, tensor_inline>(C, DT2(N, M));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tg.y * TM);
  auto mW = tW.slice(0, tg.x * TN);
  auto mC = tC.slice(tg.x * TN, tg.y * TM);
  op.run(mA, mW, mC);
}

// v1: per (row, group) activation scale sx and group sums xs; acc += sw*sx*I + b'*xs per group.
// v2 (SHARED): one activation scale per group for the whole batch, folded into c = sw*sx per (n, g); the bias
// term (b' * xs) is left to a separate small GEMM. Epilogue = one FMA per element per group.
template <int TM, int TN, bool SHARED>
kernel void w4a8_g(device const int8_t* X [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                   device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                   device const float* Sx [[buffer(4)]], device const float* Xs [[buffer(5)]],
                   device bfloat* C [[buffer(6)]], constant int3& mnk [[buffer(7)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z, kg = K / G;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)X, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  auto part = op.template get_destination_cooperative_tensor<AT, WT, int>();
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  constexpr int CAP = 64;  // >= acc.get_capacity() for these tiles (checked below)
  int nn[CAP], mm[CAP];
  const int cap = acc.get_capacity();
  for (uint16_t i = 0; i < cap; ++i) {
    acc[i] = 0;
    auto ix = acc.get_multidimensional_index(i);
    nn[i] = min(n0 + ix[0], N - 1);
    mm[i] = min(m0 + ix[1], M - 1);
  }
  for (int g = 0; g < kg; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      if (SHARED) {
        acc[i] += float(S[nn[i] * kg + g]) * float(part[i]);
      } else {
        float sw = float(S[nn[i] * kg + g]), bw = float(B[nn[i] * kg + g]) + 8.f * sw;
        acc[i] += sw * Sx[mm[i] * kg + g] * float(part[i]) + bw * Xs[mm[i] * kg + g];
      }
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    int n = n0 + ix[0], m = m0 + ix[1];
    if (acc.is_valid_element(i) && n < N && m < M) C[m * N + n] = bfloat(acc[i]);
  }
}

#define V0(TM, TN) template [[host_name("w4a8_v0_" #TM "x" #TN)]] kernel void w4a8_v0<TM, TN>( \
    device const int8_t*, device const uint*, device int*, constant int3&, uint2);
#define VG(NM, TM, TN, SH) template [[host_name("w4a8_" #NM "_" #TM "x" #TN)]] kernel void w4a8_g<TM, TN, SH>( \
    device const int8_t*, device const uint*, device const bfloat*, device const bfloat*, device const float*, \
    device const float*, device bfloat*, constant int3&, uint2);
V0(32, 64) V0(64, 64) V0(128, 64) V0(64, 128) V0(32, 128)
VG(v1, 32, 64, false) VG(v1, 64, 64, false) VG(v1, 128, 64, false) VG(v1, 64, 128, false) VG(v1, 32, 128, false)
VG(v2, 32, 64, true) VG(v2, 64, 64, true) VG(v2, 128, 64, true) VG(v2, 64, 128, true) VG(v2, 32, 128, true)

// v3: per-group loop with no epilogue loads (isolates the cost of 64-wide run() calls).
// v4: v1 with packed operands: WSB float2[N, kg] = (sw, bw + 8 sw), XSB float2[M, kg] = (sx, sum of x in group).
template <int TM, int TN, int V>
kernel void w4a8_p(device const int8_t* X [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                   device const float2* WSB [[buffer(2)]], device const float2* XSB [[buffer(4)]],
                   device bfloat* C [[buffer(6)]], constant int3& mnk [[buffer(7)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z, kg = K / G;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)X, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  auto part = op.template get_destination_cooperative_tensor<AT, WT, int>();
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  const int cap = acc.get_capacity();
  int on[64], om[64];
  for (uint16_t i = 0; i < cap; ++i) {
    acc[i] = 0;
    auto ix = acc.get_multidimensional_index(i);
    on[i] = min(n0 + ix[0], N - 1) * kg;
    om[i] = min(m0 + ix[1], M - 1) * kg;
  }
  for (int g = 0; g < kg; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      if (V == 3) acc[i] += float(part[i]);
      else {
        float2 w = WSB[on[i] + g], x = XSB[om[i] + g];
        acc[i] += w.x * x.x * float(part[i]) + w.y * x.y;
      }
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    int n = n0 + ix[0], m = m0 + ix[1];
    if (acc.is_valid_element(i) && n < N && m < M) C[m * N + n] = bfloat(acc[i]);
  }
}
#define VP(NM, TM, TN, V) template [[host_name("w4a8_" #NM "_" #TM "x" #TN)]] kernel void w4a8_p<TM, TN, V>( \
    device const int8_t*, device const uint*, device const float2*, device const float2*, device bfloat*, \
    constant int3&, uint2);
VP(v3, 32, 128, 3) VP(v3, 64, 64, 3) VP(v3, 32, 64, 3) VP(v3, 16, 128, 3) VP(v3, 32, 256, 3)
VP(v4, 32, 128, 4) VP(v4, 64, 64, 4) VP(v4, 32, 64, 4) VP(v4, 16, 128, 4) VP(v4, 32, 256, 4)

// unsigned variant of v0 / v1 (uint8 x uint4)
template <int TM, int TN, bool EPI>
kernel void w4a8_u(device const uint8_t* X [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                   device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                   device const float* Sx [[buffer(4)]], device const float* Xs [[buffer(5)]],
                   device bfloat* C [[buffer(6)]], constant int3& mnk [[buffer(7)]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z, kg = K / G;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device uint8_t, DT2, tensor_inline>((device uint8_t*)X, DT2(K, M));
  auto tW = tensor<device uint4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  auto part = op.template get_destination_cooperative_tensor<AT, WT, int>();
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  constexpr int CAP = 64;
  int nn[CAP], mm[CAP];
  const int cap = acc.get_capacity();
  for (uint16_t i = 0; i < cap; ++i) {
    acc[i] = 0;
    auto ix = acc.get_multidimensional_index(i);
    nn[i] = min(n0 + ix[0], N - 1);
    mm[i] = min(m0 + ix[1], M - 1);
  }
  for (int g = 0; g < kg; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      if (!EPI) { acc[i] += float(part[i]); continue; }
      float sw = float(S[nn[i] * kg + g]), bw = float(B[nn[i] * kg + g]) + 8.f * sw;
      acc[i] += sw * Sx[mm[i] * kg + g] * float(part[i]) + bw * Xs[mm[i] * kg + g];
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    int n = n0 + ix[0], m = m0 + ix[1];
    if (acc.is_valid_element(i) && n < N && m < M) C[m * N + n] = bfloat(acc[i]);
  }
}
#define VU(NM, TM, TN, E) template [[host_name("w4a8_" #NM "_" #TM "x" #TN)]] kernel void w4a8_u<TM, TN, E>( \
    device const uint8_t*, device const uint*, device const bfloat*, device const bfloat*, device const float*, \
    device const float*, device bfloat*, constant int3&, uint2);
VU(u0, 32, 128, false) VU(u1, 32, 128, true)

// --- engine kernel copy
namespace eng {
// C[M, N] (op)= dequant(A) * W^T. Tile TM x TN, 4 simdgroups. grid (ceil(M/TM), ceil(N/TN), S): the row tiles
// that share a weight tile are dispatched next to each other, so it is read from DRAM once. Split z takes
// groups [z * gps, (z + 1) * gps). S > 1 writes float partials [S, M, N] for splitk_reduce.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float.
struct W4A8Args { int M, N, K, gps, split; };
template <int TM, int TN, int MODE>
kernel void w4a8_gemm(device const char* U [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                      device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                      device const float2* XS [[buffer(5)]],
                      device void* Cv [[buffer(6)]], constant W4A8Args& p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K, kg = K / G;
  const int n0 = tg.y * TN, m0 = tg.x * TM;
  const int g0 = tg.z * p.gps, g1 = min(kg, g0 + p.gps);
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)U, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
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
      const float sw = float(S[nn[i] + g]), bw = float(B[nn[i] + g]) + 8.f * sw;
      const float2 x = XS[mm[i] + g];
      acc[i] += sw * x.x * float(part[i]) + bw * x.y;
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (!acc.is_valid_element(i) || n >= N || m >= M) continue;
    const size_t o = (size_t)m * N + n;
    if (p.split > 1) ((device float*)Cv)[(size_t)tg.z * M * N + o] = acc[i];
    else if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define EW4A8(TM, TN, MODE)                                                                                  \
  template [[host_name("eng_" #TM "x" #TN "_m" #MODE)]] kernel void w4a8_gemm<TM, TN, MODE>(              \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*, \
      device void*, constant W4A8Args&, uint3);
EW4A8(32, 128, 0)
}

namespace engA {
// C[M, N] (op)= dequant(A) * W^T. Tile TM x TN, 4 simdgroups. grid (ceil(M/TM), ceil(N/TN), S): the row tiles
// that share a weight tile are dispatched next to each other, so it is read from DRAM once. Split z takes
// groups [z * gps, (z + 1) * gps). S > 1 writes float partials [S, M, N] for splitk_reduce.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float.
struct W4A8Args { int M, N, K, gps, split; };
template <int TM, int TN, int MODE>
kernel void w4a8_gemm(device const char* U [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                      device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                      device const float2* XS [[buffer(5)]],
                      device void* Cv [[buffer(6)]], constant W4A8Args& p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K, kg = K / G;
  const int n0 = tg.y * TN, m0 = tg.x * TM;
  const int g0 = tg.z * p.gps, g1 = min(kg, g0 + p.gps);
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)U, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
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
  for (int g = 0; g < kg; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      const float sw = float(S[nn[i] + g]), bw = float(B[nn[i] + g]) + 8.f * sw;
      const float2 x = XS[mm[i] + g];
      acc[i] += sw * x.x * float(part[i]) + bw * x.y;
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (!acc.is_valid_element(i) || n >= N || m >= M) continue;
    const size_t o = (size_t)m * N + n;
    if (p.split > 1) ((device float*)Cv)[(size_t)tg.z * M * N + o] = acc[i];
    else if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define EW4A8(TM, TN, MODE)                                                                                  \
  template [[host_name("engA_" #TM "x" #TN "_m" #MODE)]] kernel void w4a8_gemm<TM, TN, MODE>(              \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*, \
      device void*, constant W4A8Args&, uint3);
EW4A8(32, 128, 0)
}

namespace engB {
// C[M, N] (op)= dequant(A) * W^T. Tile TM x TN, 4 simdgroups. grid (ceil(M/TM), ceil(N/TN), S): the row tiles
// that share a weight tile are dispatched next to each other, so it is read from DRAM once. Split z takes
// groups [z * gps, (z + 1) * gps). S > 1 writes float partials [S, M, N] for splitk_reduce.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float.
struct W4A8Args { int M, N, K, gps, split; };
template <int TM, int TN, int MODE>
kernel void w4a8_gemm(device const char* U [[buffer(0)]], device const uint* Wq [[buffer(1)]],
                      device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                      device const float2* XS [[buffer(5)]],
                      device void* Cv [[buffer(6)]], constant W4A8Args& p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K, kg = K / G;
  const int n0 = tg.y * TN, m0 = tg.x * TM;
  const int g0 = tg.z * p.gps, g1 = min(kg, g0 + p.gps);
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)U, DT2(K, M));
  auto tW = tensor<device int4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
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
  for (int g = 0; g < kg; g++) {
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
    for (uint16_t i = 0; i < cap; ++i) {
      const float sw = float(S[nn[i] + g]), bw = float(B[nn[i] + g]) + 8.f * sw;
      const float2 x = XS[mm[i] + g];
      acc[i] += sw * x.x * float(part[i]) + bw * x.y;
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (!acc.is_valid_element(i) || n >= N || m >= M) continue;
    const size_t o = (size_t)m * N + n;
    if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define EW4A8(TM, TN, MODE)                                                                                  \
  template [[host_name("engB_" #TM "x" #TN "_m" #MODE)]] kernel void w4a8_gemm<TM, TN, MODE>(              \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*, \
      device void*, constant W4A8Args&, uint3);
EW4A8(32, 128, 0)
}
