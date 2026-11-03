// i8k.metal (engine-pf, 2026-11-02): int8 x int8 GEMM rate vs the K run length of one matmul2d call.
// C[M,N] = sum_g sx[m,g] * (A_g . W_g^T)[m,n] * sw[n]: activations int8 with one scale per row per KS-wide group,
// weights int8 with one scale per output row. KS = 0: one run over all of K (per-row activation scale).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;
template <int TM, int TN, int KS, int NSG>
kernel void i8k(device const int8_t* A [[buffer(0)]], device const int8_t* W [[buffer(1)]],
                device const float* sx [[buffer(2)]], device const float* sw [[buffer(3)]],
                device bfloat* C [[buffer(4)]], constant int3& mnk [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = mnk.x, N = mnk.y, K = mnk.z;
  const int m0 = tg.x * TM, n0 = tg.y * TN;
  auto tA = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)A, DT2(K, M));
  auto tW = tensor<device int8_t, DT2, tensor_inline>((device int8_t*)W, DT2(K, N));
  constexpr int KD = KS == 0 ? static_cast<int>(dynamic_extent) : KS;
  constexpr auto desc = matmul2d_descriptor(TM, TN, KD, false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using AT = decltype(tA.slice(0, 0));
  using WT = decltype(tW.slice(0, 0));
  auto part = op.template get_destination_cooperative_tensor<AT, WT, int>();
  auto acc = op.template get_destination_cooperative_tensor<AT, WT, float>();
  const int cap = acc.get_capacity();
  const int G = KS == 0 ? 1 : K / KS;
  for (uint16_t i = 0; i < cap; ++i) acc[i] = 0;
  if (KS == 0) {
    auto a0 = tA.slice(0, m0); auto w0 = tW.slice(0, n0);
    op.run(a0, w0, part);
    for (uint16_t i = 0; i < cap; ++i) acc[i] = float(part[i]);
  } else {
    int mm[64];
    for (uint16_t i = 0; i < cap; ++i) mm[i] = min(m0 + acc.get_multidimensional_index(i)[1], M - 1) * G;
    for (int g = 0; g < G; g++) {
      auto ag = tA.slice(g * KS, m0); auto wg = tW.slice(g * KS, n0);
      op.run(ag, wg, part);
      for (uint16_t i = 0; i < cap; ++i) acc[i] += float(part[i]) * sx[mm[i] + g];
    }
  }
  for (uint16_t i = 0; i < cap; ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (n >= N || m >= M) continue;
    C[(size_t)m * N + n] = bfloat(acc[i] * sw[n] * (KS == 0 ? sx[m] : 1.f));
  }
}
#define I(TM, TN, KS, NSG) template [[host_name("i8k_" #TM "x" #TN "_k" #KS "_s" #NSG)]] kernel void i8k<TM, TN, KS, NSG>( \
  device const int8_t*, device const int8_t*, device const float*, device const float*, device bfloat*, constant int3&, uint3);
#define ALLK(TM, TN, NSG) I(TM, TN, 64, NSG) I(TM, TN, 128, NSG) I(TM, TN, 256, NSG) I(TM, TN, 512, NSG) I(TM, TN, 0, NSG)
ALLK(64, 64, 4) ALLK(128, 64, 4) ALLK(64, 128, 4) ALLK(128, 128, 4) ALLK(128, 128, 8) ALLK(64, 128, 8)
