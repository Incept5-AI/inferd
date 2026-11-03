#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// MLX affine 4-bit layout: Wq uint32[N, K/8] (element j of a word at bits 4j),
// S, Bz half[N, K/64], w = S*q + Bz.  C[M,N] = A[M,K] * W^T.
constant constexpr int G = 64;

using DT2 = dextents<int32_t, 2>;

// (a) Dequantise a TN x 64 weight tile into threadgroup memory, fp16 matmul2d into a float accumulator.
template <int TM, int TN>
kernel void qgemm_tg(device const half* A [[buffer(0)]],
                     device const uint* Wq [[buffer(1)]],
                     device const half* S [[buffer(2)]],
                     device const half* Bz [[buffer(3)]],
                     device half* C [[buffer(4)]],
                     constant int3& mnk [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = G;
  threadgroup half Wt[TN * TK];
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device half, DT2, tensor_inline>((device half*)A, DT2(K, M));
  auto tW = tensor<threadgroup half, DT2, tensor_inline>(Wt, DT2(TK, TN));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, true,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<AT, decltype(tW), float>();
#pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0;
  const int kw = K / 8, kg = K / G;
  for (int k = 0; k < K; k += TK) {
    for (int w = tid; w < TN * TK / 8; w += 128) {
      int row = w / (TK / 8), c8 = w % (TK / 8);
      int n = n0 + row;
      uint word = n < N ? Wq[n * kw + k / 8 + c8] : 0;
      half s = n < N ? S[n * kg + k / G] : 0, b = n < N ? Bz[n * kg + k / G] : 0;
#pragma unroll
      for (int j = 0; j < 8; j++) Wt[row * TK + c8 * 8 + j] = s * half((word >> (4 * j)) & 15) + b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The default mode overwrites the destination; multiply_accumulate adds into it.
    auto mA = tA.slice(k, m0);
    op.run(mA, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto ix = acc.get_multidimensional_index(i);
      int n = n0 + ix[0], m = m0 + ix[1];
      if (n < N && m < M) C[m * N + n] = half(acc[i]);
    }
  }
}

// (b) Hardware 4-bit: matmul2d on the raw uint4b weights one 64-group at a time,
// then acc += S[n,g]*partial + Bz[n,g]*rowsum(A[m, group g]).  Asum is float[M, K/64].
template <int TM, int TN>
kernel void qgemm_hw(device const half* A [[buffer(0)]],
                     device const uint* Wq [[buffer(1)]],
                     device const half* S [[buffer(2)]],
                     device const half* Bz [[buffer(3)]],
                     device half* C [[buffer(4)]],
                     constant int3& mnk [[buffer(5)]],
                     device const float* Asum [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device half, DT2, tensor_inline>((device half*)A, DT2(K, M));
  auto tW = tensor<device uint4b_format, DT2, tensor_inline>((device uchar*)Wq, DT2(K, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  auto part = op.template get_destination_cooperative_tensor<AT, WT, float>();
#pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0;
  const int kg = K / G;
  for (int g = 0; g < kg; g++) {
#pragma unroll
    for (uint16_t i = 0; i < part.get_capacity(); ++i)
      if (part.is_valid_element(i)) part[i] = 0;
    auto mA = tA.slice(g * G, m0);
    auto mW = tW.slice(g * G, n0);
    op.run(mA, mW, part);
#pragma unroll
    for (uint16_t i = 0; i < part.get_capacity(); ++i) {
      if (part.is_valid_element(i)) {
        auto ix = part.get_multidimensional_index(i);
        int n = n0 + ix[0], m = m0 + ix[1];
        if (n < N && m < M)
          acc[i] += float(S[n * kg + g]) * part[i] + float(Bz[n * kg + g]) * Asum[m * kg + g];
      }
    }
  }
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto ix = acc.get_multidimensional_index(i);
      int n = n0 + ix[0], m = m0 + ix[1];
      if (n < N && m < M) C[m * N + n] = half(acc[i]);
    }
  }
}

#define QINST(KN, TM, TN) \
  template [[host_name(#KN "_" #TM "x" #TN)]] kernel void KN<TM, TN>( \
      device const half*, device const uint*, device const half*, device const half*, device half*, \
      constant int3&, uint2, uint);
#define HINST(TM, TN) \
  template [[host_name("qgemm_hw_" #TM "x" #TN)]] kernel void qgemm_hw<TM, TN>( \
      device const half*, device const uint*, device const half*, device const half*, device half*, \
      constant int3&, device const float*, uint2);
QINST(qgemm_tg, 16, 64)
QINST(qgemm_tg, 32, 64)
QINST(qgemm_tg, 64, 64)
QINST(qgemm_tg, 32, 128)
QINST(qgemm_tg, 64, 128)
HINST(16, 64)
HINST(32, 64)
HINST(64, 64)
HINST(32, 128)
HINST(64, 128)
