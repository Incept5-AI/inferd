// lane4: small-batch (M <= 16) 4-bit GEMM after TensorFold's tf_lane (mlx.fast challenge, 2026-10-31).
// Weights repacked at load so one 32-column tile's codes for one 64-input group are one contiguous 1 KB block
// (LP[tile][g][32 cols][32 bytes], flipped: signed int4 q - 8), and scale/bias pairs group-major (SB[g][N] = (s, b) bf16).
// Each simdgroup runs a 16 x 32 x 64 matmul2d per group in multiply mode on bf16 activations straight from device
// memory, then applies the affine per group: C += s * P + (b + 8 s) * sum(x over the group). A threadgroup owns one tile;
// its SK simdgroups split the groups and reduce through threadgroup memory.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;

// Per row and 64-group sums of the activations: XS[m * KG + g]. grid (KG * 32, M).
struct XSumArgs { int M, K; };
kernel void lane_xsum(device const bfloat* X [[buffer(0)]], device float* XS [[buffer(1)]],
                      constant XSumArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const int m = gid.y, g = gid.x / 32, kg = p.K / 64;
  if (m >= p.M || g >= kg) return;
  const size_t o = (size_t)m * p.K + g * 64 + lane * 2;
  const float s = simd_sum(float(X[o]) + float(X[o + 1]));
  if (lane == 0) XS[(size_t)m * kg + g] = s;
}

struct LaneArgs { int M, N, K, SBN; };  // SBN: the (s, b) table's row stride
// MODE 0: C = bf16, 1: C += bf16, 2: C = float. RT row tiles of 16 per simdgroup share each weight block (RT 2 and
// 4 measured slower than W4A8 above 16 rows on the M5 Max: the bf16 product is compute-bound there, so only RT 1 is built).
template <int SK, int MODE, int RT>
kernel void lane4(device const bfloat* X [[buffer(0)]], device const uchar* LP [[buffer(1)]],
                  device const uint* SB [[buffer(2)]], device const float* XS [[buffer(3)]],
                  device void* Cv [[buffer(4)]], constant LaneArgs& p [[buffer(5)]],
                  uint2 tg2 [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  // grid (row blocks of 16 * RT, column tiles): a weight tile's row blocks run next to each other (one DRAM read).
  const int M = p.M, N = p.N, K = p.K, KG = K / 64, tile = tg2.y, n0 = tile * 32, m0 = tg2.x * 16 * RT;
  const int g0 = (int)sg * KG / SK, g1 = ((int)sg + 1) * KG / SK;
  constexpr auto desc = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto tA = tensor<device bfloat, DT2, tensor_inline>((device bfloat*)X, DT2(K, M));
  using AT = decltype(tA.slice(0, 0));
  using BT = tensor<device int4b_format, DT2, tensor_inline>;
  auto P = op.template get_destination_cooperative_tensor<AT, BT, float>();
  constexpr int CAP = 16;
  float C[RT][CAP];
  int cn[CAP], cm[RT][CAP];
  for (int i = 0; i < CAP; i++) {
    auto ix = P.get_multidimensional_index(i);
    cn[i] = min(n0 + ix[0], N - 1);
    for (int t = 0; t < RT; t++) { C[t][i] = 0.f; cm[t][i] = min(m0 + t * 16 + (int)ix[1], M - 1); }
  }
  uint sbn[CAP];
  for (int i = 0; i < CAP; i++) sbn[i] = SB[(size_t)g0 * p.SBN + cn[i]];
  for (int g = g0; g < g1; g++) {
    float sc[CAP], bc[CAP];
    for (int i = 0; i < CAP; i++) { const float2 sb = float2(as_type<vec<bfloat, 2>>(sbn[i])); sc[i] = sb.x; bc[i] = sb.y + 8.f * sb.x; }
    const int gn = min(g + 1, g1 - 1);
    for (int i = 0; i < CAP; i++) sbn[i] = SB[(size_t)gn * p.SBN + cn[i]];
    BT b((device uchar*)LP + ((size_t)tile * KG + g) * 1024, DT2(64, 32));
    for (int t = 0; t < RT; t++) {
      if (RT > 1 && m0 + t * 16 >= M) break;
      auto a = tA.slice(g * 64, m0 + t * 16);
      op.run(a, b, P);
      for (int i = 0; i < CAP; i++) C[t][i] = fma(sc[i], P[i], fma(bc[i], XS[(size_t)cm[t][i] * KG + g], C[t][i]));
    }
  }
  for (int t = 0; t < RT; t++) {
    if (RT > 1 && m0 + t * 16 >= M) break;
    if (SK > 1) {
      threadgroup float part[(SK > 1 ? SK - 1 : 1) * CAP * 32];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg > 0) for (int i = 0; i < CAP; i++) part[((sg - 1) * CAP + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) for (int s = 1; s < SK; s++) for (int i = 0; i < CAP; i++) C[t][i] += part[((s - 1) * CAP + i) * 32 + lane];
    }
    if (sg != 0) continue;
    for (int i = 0; i < CAP; i++) {
      auto ix = P.get_multidimensional_index(i);
      const int n = n0 + ix[0], m = m0 + t * 16 + ix[1];
      if (!P.is_valid_element(i) || n >= N || m >= M) continue;
      const size_t o = (size_t)m * N + n;
      if (MODE == 2) ((device float*)Cv)[o] = C[t][i];
      else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(C[t][i] + float(((device bfloat*)Cv)[o]));
      else ((device bfloat*)Cv)[o] = bfloat(C[t][i]);
    }
  }
}

#define LANE(SK, MODE, RT)                                                                                     \
  template [[host_name("lane4_s" #SK "_m" #MODE "_r" #RT)]] kernel void lane4<SK, MODE, RT>(                  \
      device const bfloat*, device const uchar*, device const uint*, device const float*, device void*,        \
      constant LaneArgs&, uint2, uint, uint);
#define LANES(SK, RT) LANE(SK, 0, RT) LANE(SK, 1, RT) LANE(SK, 2, RT)
LANES(1, 1) LANES(2, 1) LANES(4, 1) LANES(8, 1)

