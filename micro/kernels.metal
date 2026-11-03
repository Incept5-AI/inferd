#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Streaming read: each thread sums float4s over a grid-stride loop.
kernel void bw_read(device const float4* src [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& n4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
  float4 acc = 0;
  for (uint i = gid; i < n4; i += gsz) acc += src[i];
  if (acc.x == 1234.5f) out[gid] = acc.y + acc.z + acc.w;  // defeat DCE, never true
}

// C[M,N] = A[M,K] * W, W either [K,N] (NN) or [N,K] (NT, the usual weight layout).
// Tensor extents are (innermost, outer) = (cols, rows).
template <int TM, int TN, bool NT, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]],
                     device half* W [[buffer(1)]],
                     device half* C [[buffer(2)]],
                     constant int3& mnk [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, M));
  auto tW = NT ? tensor<device half, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(K, N))
               : tensor<device half, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(N, K));
  auto tC = tensor<device half, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, NT, false);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mA = tA.slice(0, tg.y * TM);
  auto mW = NT ? tW.slice(0, tg.x * TN) : tW.slice(tg.x * TN, 0);
  auto mC = tC.slice(tg.x * TN, tg.y * TM);
  op.run(mA, mW, mC);
}

#define INST(TM, TN, NT, SG) \
  template [[host_name("gemm_f16_" #TM "x" #TN "_" #NT "_sg" #SG)]] kernel void gemm_f16<TM, TN, NT, SG>( \
      device half*, device half*, device half*, constant int3&, uint2);
INST(64, 32, false, 4)
INST(64, 32, true, 4)
INST(32, 64, true, 4)
INST(16, 64, true, 4)
INST(16, 128, true, 4)
INST(32, 128, true, 4)
INST(64, 64, true, 4)
INST(128, 64, true, 4)

// Element-type sweep (NT layout): does the tensor path run int8 / fp8 / mixed faster than fp16?
template <typename TA, typename TW, typename TO, int TM, int TN>
kernel void gemm_ty(device TA* A [[buffer(0)]],
                    device TW* W [[buffer(1)]],
                    device TO* C [[buffer(2)]],
                    constant int3& mnk [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device TA, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, M));
  auto tW = tensor<device TW, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(K, N));
  auto tC = tensor<device TO, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tg.y * TM);
  auto mW = tW.slice(0, tg.x * TN);
  auto mC = tC.slice(tg.x * TN, tg.y * TM);
  op.run(mA, mW, mC);
}
#define INSTY(NM, TA, TW, TO, TM, TN) \
  template [[host_name("gemm_" #NM "_" #TM "x" #TN)]] kernel void gemm_ty<TA, TW, TO, TM, TN>( \
      device TA*, device TW*, device TO*, constant int3&, uint2);
INSTY(h, half, half, half, 128, 64)
INSTY(h, half, half, half, 64, 64)
INSTY(bf, bfloat, bfloat, bfloat, 128, 64)
INSTY(bf, bfloat, bfloat, bfloat, 64, 64)
INSTY(i8, int8_t, int8_t, int32_t, 128, 64)
INSTY(i8, int8_t, int8_t, int32_t, 64, 64)
INSTY(bfi8, bfloat, int8_t, float, 128, 64)
INSTY(bfi8, bfloat, int8_t, float, 64, 64)
