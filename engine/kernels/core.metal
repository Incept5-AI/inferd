// Core decode kernels for Qwen3.5/3.8 hybrid models (Gated DeltaNet + gated full attention).
// Activations are bfloat; reductions and recurrent state are float.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

using DT2 = dextents<int32_t, 2>;
constant constexpr int G = 64;  // quantization group size

// ---------------------------------------------------------------------------------------------
// Quantized GEMM: C[M,N] (op)= A[M,K] * W^T, W in MLX affine 4-bit layout:
// Wq uint32[N, K/8] (element j of a word at bits 4j), S/Bz bfloat[N, K/64], w = S*q + Bz.
// MODE 0: C = result (bfloat). MODE 1: C += result (bfloat, residual add). MODE 2: C = result (float).
struct GemmArgs { int M, N, K, xr; };  // xr: XOR mask for flipped 4-bit nibbles (W4A8)

template <int TM, int TN, int MODE>
kernel void qgemm4(device const bfloat* A [[buffer(0)]],
                   device const uint* Wq [[buffer(1)]],
                   device const bfloat* S [[buffer(2)]],
                   device const bfloat* Bz [[buffer(3)]],
                   device void* Cv [[buffer(4)]],
                   constant GemmArgs& p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = G;
  threadgroup bfloat Wt[TN * TK];
  const int M = p.M, N = p.N, K = p.K;
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  auto tA = tensor<device bfloat, DT2, tensor_inline>((device bfloat*)A, DT2(K, M));
  auto tW = tensor<threadgroup bfloat, DT2, tensor_inline>(Wt, DT2(TK, TN));
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
      uint word = 0;
      float s = 0, b = 0;
      if (n < N) {
        word = Wq[(size_t)n * kw + k / 8 + c8];
        s = S[(size_t)n * kg + k / G];
        b = Bz[(size_t)n * kg + k / G];
      }
#pragma unroll
      for (int j = 0; j < 8; j++) Wt[row * TK + c8 * 8 + j] = bfloat(s * float((word >> (4 * j)) & 15) + b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA = tA.slice(k, m0);
    op.run(mA, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto ix = acc.get_multidimensional_index(i);
    int n = n0 + ix[0], m = m0 + ix[1];
    if (n >= N || m >= M) continue;
    size_t o = (size_t)m * N + n;
    if (MODE == 2) ((device float*)Cv)[o] = acc[i];
    else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(acc[i] + float(((device bfloat*)Cv)[o]));
    else ((device bfloat*)Cv)[o] = bfloat(acc[i]);
  }
}

#define QG(TM, TN, MODE)                                                                          \
  template [[host_name("qgemm4_" #TM "x" #TN "_m" #MODE)]] kernel void qgemm4<TM, TN, MODE>(      \
      device const bfloat*, device const uint*, device const bfloat*, device const bfloat*,      \
      device void*, constant GemmArgs&, uint2, uint);
QG(32, 64, 0)
QG(32, 64, 1)
QG(32, 64, 2)

// Small-batch quantized matvec (decode, M <= 8): each simdgroup computes 4 output rows for R input rows,
// so a weight read is reused R times. Lanes hold 16 consecutive K values; x is pre-scaled by 16^-j so the
// four nibbles of a uint16 can be masked in place (MLX's qmv trick). Requires K % 512 == 0.
// grid: threadgroups (ceil(M/R), ceil(N/8)), 64 threads (2 simdgroups).
template <int R, int MODE>
kernel void qmv4(device const bfloat* A [[buffer(0)]],
                 device const uint* Wq [[buffer(1)]],
                 device const bfloat* S [[buffer(2)]],
                 device const bfloat* Bz [[buffer(3)]],
                 device void* Cv [[buffer(4)]],
                 constant GemmArgs& p [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPT = 16, BLOCK = 32 * VPT;
  const int M = p.M, N = p.N, K = p.K;
  const int row0 = tg.y * 8 + sg * ROWS, m0 = tg.x * R;
  if (row0 >= N) return;
  const int kq = K / 4, kg = K / G;  // uint16 per row, groups per row
  device const ushort* ws = (device const ushort*)Wq + (size_t)row0 * kq + lane * 4;
  device const bfloat* sp = S + (size_t)row0 * kg + lane / 4;
  device const bfloat* bp = Bz + (size_t)row0 * kg + lane / 4;
  device const bfloat* xp = A + (size_t)m0 * K + lane * VPT;
  float res[R][ROWS];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) res[r][j] = 0;
  for (int k = 0; k < K; k += BLOCK) {
    // K need only be a multiple of 64: lanes whose 16 values lie past the row end skip the tail block
    if (k + (int)lane * VPT >= K) { ws += BLOCK / 4; sp += BLOCK / G; bp += BLOCK / G; xp += BLOCK; continue; }
    float xt[R][VPT], xs[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      xs[r] = 0;
      bool ok = m0 + r < M;
#pragma unroll
      for (int i = 0; i < VPT; i += 4) {
        float a0 = ok ? float(xp[r * K + i]) : 0.f, a1 = ok ? float(xp[r * K + i + 1]) : 0.f;
        float a2 = ok ? float(xp[r * K + i + 2]) : 0.f, a3 = ok ? float(xp[r * K + i + 3]) : 0.f;
        xs[r] += a0 + a1 + a2 + a3;
        xt[r][i] = a0;
        xt[r][i + 1] = a1 / 16.f;
        xt[r][i + 2] = a2 / 256.f;
        xt[r][i + 3] = a3 / 4096.f;
      }
    }
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      if (row0 + j >= N) break;
      device const ushort* w = ws + (size_t)j * kq;
      const ushort xr = ushort(p.xr);
      ushort w0 = w[0] ^ xr, w1 = w[1] ^ xr, w2 = w[2] ^ xr, w3 = w[3] ^ xr;
      float s = sp[(size_t)j * kg], b = bp[(size_t)j * kg];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float acc = xt[r][0] * (w0 & 0xf) + xt[r][1] * (w0 & 0xf0) + xt[r][2] * (w0 & 0xf00) + xt[r][3] * (w0 & 0xf000) +
                    xt[r][4] * (w1 & 0xf) + xt[r][5] * (w1 & 0xf0) + xt[r][6] * (w1 & 0xf00) + xt[r][7] * (w1 & 0xf000) +
                    xt[r][8] * (w2 & 0xf) + xt[r][9] * (w2 & 0xf0) + xt[r][10] * (w2 & 0xf00) + xt[r][11] * (w2 & 0xf000) +
                    xt[r][12] * (w3 & 0xf) + xt[r][13] * (w3 & 0xf0) + xt[r][14] * (w3 & 0xf00) + xt[r][15] * (w3 & 0xf000);
        res[r][j] += s * acc + b * xs[r];
      }
    }
    ws += BLOCK / 4;
    sp += BLOCK / G;
    bp += BLOCK / G;
    xp += BLOCK;
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      float v = simd_sum(res[r][j]);
      int m = m0 + r, n = row0 + j;
      if (lane != 0 || m >= M || n >= N) continue;
      size_t o = (size_t)m * N + n;
      if (MODE == 2) ((device float*)Cv)[o] = v;
      else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
      else ((device bfloat*)Cv)[o] = bfloat(v);
    }
  }
}

#define QV(R, MODE)                                                                              \
  template [[host_name("qmv4_r" #R "_m" #MODE)]] kernel void qmv4<R, MODE>(                       \
      device const bfloat*, device const uint*, device const bfloat*, device const bfloat*,      \
      device void*, constant GemmArgs&, uint2, uint, uint);
QV(1, 0) QV(1, 1) QV(1, 2)
QV(2, 0) QV(2, 1) QV(2, 2)
QV(4, 0) QV(4, 1) QV(4, 2)
QV(8, 0) QV(8, 1) QV(8, 2)

// 8-bit variant of qmv4 (MLX affine 8-bit, group 64: byte i of a row is element i). Same tiling: each lane takes
// 16 consecutive values (one uint4) of a 512-value block for 4 output rows.
template <int R, int MODE>
kernel void qmv8(device const bfloat* A [[buffer(0)]],
                 device const uint* Wq [[buffer(1)]],
                 device const bfloat* S [[buffer(2)]],
                 device const bfloat* Bz [[buffer(3)]],
                 device void* Cv [[buffer(4)]],
                 constant GemmArgs& p [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPT = 16, BLOCK = 32 * VPT;
  const int M = p.M, N = p.N, K = p.K;
  const int row0 = tg.y * 8 + sg * ROWS, m0 = tg.x * R;
  if (row0 >= N) return;
  const int kg = K / G;
  device const uchar* ws = (device const uchar*)Wq + (size_t)row0 * K + lane * VPT;
  device const bfloat* sp = S + (size_t)row0 * kg + lane / 4;
  device const bfloat* bp = Bz + (size_t)row0 * kg + lane / 4;
  device const bfloat* xp = A + (size_t)m0 * K + lane * VPT;
  float res[R][ROWS];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) res[r][j] = 0;
  for (int k = 0; k < K; k += BLOCK) {
    // K need only be a multiple of 64: lanes whose 16 values lie past the row end skip the tail block
    if (k + (int)lane * VPT >= K) { ws += BLOCK / 1; sp += BLOCK / G; bp += BLOCK / G; xp += BLOCK; continue; }
    float xt[R][VPT], xs[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      xs[r] = 0;
      bool ok = m0 + r < M;
#pragma unroll
      for (int i = 0; i < VPT; i++) {
        xt[r][i] = ok ? float(xp[r * K + i]) : 0.f;
        xs[r] += xt[r][i];
      }
    }
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      if (row0 + j >= N) break;
      uint4 w = *(device const uint4*)(ws + (size_t)j * K);
      float s = sp[(size_t)j * kg], b = bp[(size_t)j * kg];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float acc = 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
          uint u = w[q] ^ uint(p.xr);
          acc += xt[r][4 * q] * float(u & 0xff) + xt[r][4 * q + 1] * float((u >> 8) & 0xff) +
                 xt[r][4 * q + 2] * float((u >> 16) & 0xff) + xt[r][4 * q + 3] * float(u >> 24);
        }
        res[r][j] += s * acc + b * xs[r];
      }
    }
    ws += BLOCK;
    sp += BLOCK / G;
    bp += BLOCK / G;
    xp += BLOCK;
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      float v = simd_sum(res[r][j]);
      int m = m0 + r, n = row0 + j;
      if (lane != 0 || m >= M || n >= N) continue;
      size_t o = (size_t)m * N + n;
      if (MODE == 2) ((device float*)Cv)[o] = v;
      else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
      else ((device bfloat*)Cv)[o] = bfloat(v);
    }
  }
}

// bf16 weights, M <= 2 (decode): each lane takes 16 consecutive values (two uint4) of a 512-value block for 4 rows.
template <int R, int MODE>
kernel void gemv16(device const bfloat* A [[buffer(0)]],
                   device const bfloat* W [[buffer(1)]],
                   device void* Cv [[buffer(4)]],
                   constant GemmArgs& p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPT = 16, BLOCK = 32 * VPT;
  const int M = p.M, N = p.N, K = p.K;
  const int row0 = tg.y * 8 + sg * ROWS, m0 = tg.x * R;
  if (row0 >= N) return;
  device const bfloat* ws = W + (size_t)row0 * K + lane * VPT;
  device const bfloat* xp = A + (size_t)m0 * K + lane * VPT;
  float res[R][ROWS];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) res[r][j] = 0;
  for (int k = 0; k < K; k += BLOCK) {
    float xt[R][VPT];
#pragma unroll
    for (int r = 0; r < R; r++) {
      bool ok = m0 + r < M;
#pragma unroll
      for (int i = 0; i < VPT; i++) xt[r][i] = ok ? float(xp[r * K + i]) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      if (row0 + j >= N) break;
      device const bfloat4* w4 = (device const bfloat4*)(ws + (size_t)j * K);
      float4 w[4] = {float4(w4[0]), float4(w4[1]), float4(w4[2]), float4(w4[3])};
#pragma unroll
      for (int r = 0; r < R; r++) {
        float acc = 0;
#pragma unroll
        for (int q = 0; q < 4; q++)
          acc += dot(w[q], float4(xt[r][4 * q], xt[r][4 * q + 1], xt[r][4 * q + 2], xt[r][4 * q + 3]));
        res[r][j] += acc;
      }
    }
    ws += BLOCK;
    xp += BLOCK;
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      float v = simd_sum(res[r][j]);
      int m = m0 + r, n = row0 + j;
      if (lane != 0 || m >= M || n >= N) continue;
      size_t o = (size_t)m * N + n;
      if (MODE == 2) ((device float*)Cv)[o] = v;
      else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
      else ((device bfloat*)Cv)[o] = bfloat(v);
    }
  }
}

#define GV16(R, MODE)                                                                            \
  template [[host_name("gemv16_r" #R "_m" #MODE)]] kernel void gemv16<R, MODE>(                   \
      device const bfloat*, device const bfloat*, device void*, constant GemmArgs&, uint2, uint, uint);
GV16(1, 0) GV16(1, 1) GV16(2, 0) GV16(2, 1)

#define QV8(R, MODE)                                                                             \
  template [[host_name("qmv8_r" #R "_m" #MODE)]] kernel void qmv8<R, MODE>(                       \
      device const bfloat*, device const uint*, device const bfloat*, device const bfloat*,      \
      device void*, constant GemmArgs&, uint2, uint, uint);
QV8(1, 0) QV8(1, 1) QV8(1, 2)
QV8(2, 0) QV8(2, 1) QV8(2, 2)

// Small-M quantized matmul (3 <= M <= R, e.g. speculative verify blocks): like qmv4, but the R activation rows
// of each 512-wide K block are staged in threadgroup memory, and each lane dequantizes its 16 weights once and
// reuses them for all R rows. Keeps the weight stream at bandwidth without qmv4's register spill.
// grid: threadgroups (ceil(M/R), ceil(N/8)), 64 threads (2 simdgroups x 4 output rows).
template <int R, int MODE>
kernel void qmvt4(device const bfloat* A [[buffer(0)]],
                  device const uint* Wq [[buffer(1)]],
                  device const bfloat* S [[buffer(2)]],
                  device const bfloat* Bz [[buffer(3)]],
                  device void* Cv [[buffer(4)]],
                  constant GemmArgs& p [[buffer(5)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPT = 16, BLOCK = 32 * VPT;
  threadgroup bfloat xs[R][BLOCK];
  const int M = p.M, N = p.N, K = p.K;
  const int row0 = tg.y * 8 + sg * ROWS, m0 = tg.x * R;
  const int kq = K / 4, kg = K / G;
  float res[R][ROWS];
  for (int r = 0; r < R; r++)
    for (int j = 0; j < ROWS; j++) res[r][j] = 0;
  for (int k = 0; k < K; k += BLOCK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < R * BLOCK / 4; i += 64) {
      int r = i / (BLOCK / 4), c = (i % (BLOCK / 4)) * 4;
      bfloat4 v = m0 + r < M ? *(device const bfloat4*)(A + (size_t)(m0 + r) * K + k + c) : bfloat4(0);
      *(threadgroup bfloat4*)&xs[r][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= N) continue;
    // Dequantize this lane's 16 weights of each of the 4 output rows once, then reuse them for all R rows,
    // reading each row's 16 activations from threadgroup memory once.
    float wv[ROWS][VPT];
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      int n = min(row0 + j, N - 1);
      device const ushort* w = (device const ushort*)Wq + (size_t)n * kq + k / 4 + lane * 4;
      float s = S[(size_t)n * kg + (k + lane * VPT) / G], b = Bz[(size_t)n * kg + (k + lane * VPT) / G];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        ushort u = w[q] ^ ushort(p.xr);  // flipped weights (W4A8) back to MLX's unsigned nibbles
        wv[j][4 * q] = s * float(u & 0xf) + b;
        wv[j][4 * q + 1] = s * float((u >> 4) & 0xf) + b;
        wv[j][4 * q + 2] = s * float((u >> 8) & 0xf) + b;
        wv[j][4 * q + 3] = s * float((u >> 12) & 0xf) + b;
      }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
      threadgroup const bfloat4* xr = (threadgroup const bfloat4*)&xs[r][lane * VPT];
      float4 x0 = float4(xr[0]), x1 = float4(xr[1]), x2 = float4(xr[2]), x3 = float4(xr[3]);
#pragma unroll
      for (int j = 0; j < ROWS; j++) {
        res[r][j] += wv[j][0] * x0.x + wv[j][1] * x0.y + wv[j][2] * x0.z + wv[j][3] * x0.w +
                     wv[j][4] * x1.x + wv[j][5] * x1.y + wv[j][6] * x1.z + wv[j][7] * x1.w +
                     wv[j][8] * x2.x + wv[j][9] * x2.y + wv[j][10] * x2.z + wv[j][11] * x2.w +
                     wv[j][12] * x3.x + wv[j][13] * x3.y + wv[j][14] * x3.z + wv[j][15] * x3.w;
      }
    }
  }
  if (row0 >= N) return;
  for (int r = 0; r < R; r++) {
    for (int j = 0; j < ROWS; j++) {
      float v = simd_sum(res[r][j]);
      int m = m0 + r, n = row0 + j;
      if (lane != 0 || m >= M || n >= N) continue;
      size_t o = (size_t)m * N + n;
      if (MODE == 2) ((device float*)Cv)[o] = v;
      else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
      else ((device bfloat*)Cv)[o] = bfloat(v);
    }
  }
}

#define QVT(R, MODE)                                                                             \
  template [[host_name("qmvt4_r" #R "_m" #MODE)]] kernel void qmvt4<R, MODE>(                     \
      device const bfloat*, device const uint*, device const bfloat*, device const bfloat*,      \
      device void*, constant GemmArgs&, uint2, uint, uint, uint);
QVT(8, 0) QVT(8, 1) QVT(16, 0) QVT(16, 1)

// ---------------------------------------------------------------------------------------------
// Embedding lookup from a 4-bit table. grid (D/256 groups... ) : one thread per output element.
struct EmbedArgs { int D; };
kernel void embed4(device const int* tok [[buffer(0)]],
                   device const uint* Wq [[buffer(1)]],
                   device const bfloat* S [[buffer(2)]],
                   device const bfloat* Bz [[buffer(3)]],
                   device bfloat* out [[buffer(4)]],
                   constant EmbedArgs& p [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x, b = gid.y;
  if (d >= p.D) return;
  size_t row = (size_t)tok[b];
  uint word = Wq[row * (p.D / 8) + d / 8];
  float s = S[row * (p.D / G) + d / G], z = Bz[row * (p.D / G) + d / G];
  out[(size_t)b * p.D + d] = bfloat(s * float((word >> (4 * (d % 8))) & 15) + z);
}

// Multimodal rows: overwrite the embeddings of rows flagged in ext with rows of xin (image features).
kernel void embed_ext(device const int* ext [[buffer(0)]],
                      device const bfloat* xin [[buffer(1)]],
                      device bfloat* out [[buffer(4)]],
                      constant EmbedArgs& p [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x, b = gid.y;
  if (d >= p.D || !ext[b]) return;
  out[(size_t)b * p.D + d] = xin[(size_t)b * p.D + d];
}

// Embedding lookup from a bf16 table.
kernel void embed16(device const int* tok [[buffer(0)]],
                    device const bfloat* W [[buffer(1)]],
                    device bfloat* out [[buffer(4)]],
                    constant EmbedArgs& p [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x, b = gid.y;
  if (d >= p.D) return;
  out[(size_t)b * p.D + d] = W[(size_t)tok[b] * p.D + d];
}

// Embedding lookup from an 8-bit table (one byte per element).
kernel void embed8(device const int* tok [[buffer(0)]],
                   device const uchar* Wq [[buffer(1)]],
                   device const bfloat* S [[buffer(2)]],
                   device const bfloat* Bz [[buffer(3)]],
                   device bfloat* out [[buffer(4)]],
                   constant EmbedArgs& p [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x, b = gid.y;
  if (d >= p.D) return;
  size_t row = (size_t)tok[b];
  float s = S[row * (p.D / G) + d / G], z = Bz[row * (p.D / G) + d / G];
  out[(size_t)b * p.D + d] = bfloat(s * float(Wq[row * p.D + d]) + z);
}

// ---------------------------------------------------------------------------------------------
// Threadgroup-wide sum for a 1-D threadgroup of up to 1024 threads.
inline float tg_sum(float v, threadgroup float* scratch, uint tid, uint ntg) {
  v = simd_sum(v);
  uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? scratch[lane] : 0.f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

inline float tg_max(float v, threadgroup float* scratch, uint tid, uint ntg) {
  v = simd_max(v);
  uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? scratch[lane] : 0.f;
  t = simd_max(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// RMSNorm over rows of length D: out = x * rsqrt(mean(x^2) + eps) * w. One threadgroup per row.
// x and out rows have strides in_stride / out_stride (elements); w may be shared by all rows.
struct NormArgs { int D, in_stride, out_stride; float eps; };
kernel void rmsnorm(device const bfloat* x [[buffer(0)]],
                    device const bfloat* w [[buffer(1)]],
                    device bfloat* out [[buffer(2)]],
                    constant NormArgs& p [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device const bfloat* xr = x + (size_t)row * p.in_stride;
  float ss = 0;
  for (int i = tid; i < p.D; i += ntg) { float v = xr[i]; ss += v * v; }
  float r = rsqrt(tg_sum(ss, scratch, tid, ntg) / p.D + p.eps);
  device bfloat* o = out + (size_t)row * p.out_stride;
  for (int i = tid; i < p.D; i += ntg) o[i] = bfloat(float(xr[i]) * r * float(w[i]));
}

// rmsnorm fused with the W4A8 activation quantization of the next GEMM (a8_quant in w4a8.metal): also writes
// U int8[rows, D] and XS float2[rows, D/64] = (sx, sx * sum q) from the bf16-rounded output, so the GEMM skips its
// a8_quant dispatch. Bit-identical to rmsnorm + a8_quant. One threadgroup per row; simdgroups take 64-wide groups.
kernel void rmsnorm_a8(device const bfloat* x [[buffer(0)]],
                       device const bfloat* w [[buffer(1)]],
                       device bfloat* out [[buffer(2)]],
                       constant NormArgs& p [[buffer(3)]],
                       device char* U [[buffer(4)]],
                       device float2* XS [[buffer(5)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint ntg [[threads_per_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const bfloat* xr = x + (size_t)row * p.in_stride;
  float ss = 0;
  for (int i = tid; i < p.D; i += ntg) { float v = xr[i]; ss += v * v; }
  float r = rsqrt(tg_sum(ss, scratch, tid, ntg) / p.D + p.eps);
  device bfloat* o = out + (size_t)row * p.out_stride;
  const int kg = p.D / 64, nsg = ntg / 32;
  for (int g = sg; g < kg; g += nsg) {
    const int i = g * 64 + lane * 2;
    bfloat b0 = bfloat(float(xr[i]) * r * float(w[i])), b1 = bfloat(float(xr[i + 1]) * r * float(w[i + 1]));
    o[i] = b0;
    o[i + 1] = b1;
    float2 v = float2(float(b0), float(b1));
    float amax = simd_max(max(fabs(v.x), fabs(v.y)));
    float sx = amax > 0 ? amax / 127.f : 1.f;
    float2 q = rint(v / sx);
    U[(size_t)row * p.D + i] = char(q.x);
    U[(size_t)row * p.D + i + 1] = char(q.y);
    float qs = simd_sum(q.x + q.y);
    if (lane == 0) XS[(size_t)row * kg + g] = float2(sx, sx * qs);
  }
}

// A deferred split-K residual reduce fused into the next rmsnorm (and, with A8, its W4A8 activation quantization):
// x[row] = bf16(sum_s part[s, row] + x[row]) exactly as splitk_reduce with add, then rmsnorm (+ a8_quant) of the
// updated row exactly as rmsnorm / rmsnorm_a8. Saves the splitk_reduce dispatch of out_proj and down.
struct SplitRed { int M, N, S; };
template <bool A8>
kernel void splitk_res_norm(device bfloat* x [[buffer(0)]],
                            device const bfloat* w [[buffer(1)]],
                            device bfloat* out [[buffer(2)]],
                            constant NormArgs& p [[buffer(3)]],
                            device char* U [[buffer(4)]],
                            device float2* XS [[buffer(5)]],
                            device const float* part [[buffer(6)]],
                            constant SplitRed& sr [[buffer(7)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device bfloat* xr = x + (size_t)row * p.in_stride;
  const size_t mn = (size_t)sr.M * sr.N, ro = (size_t)row * sr.N;
  float ss = 0;
  for (int i = tid; i < p.D; i += ntg) {
    float acc = 0;
    for (int z = 0; z < sr.S; z++) acc += part[z * mn + ro + i];
    const bfloat v = bfloat(acc + float(xr[i]));
    xr[i] = v;
    ss += float(v) * float(v);
  }
  threadgroup_barrier(mem_flags::mem_device);
  float r = rsqrt(tg_sum(ss, scratch, tid, ntg) / p.D + p.eps);
  device bfloat* o = out + (size_t)row * p.out_stride;
  if (!A8) {
    for (int i = tid; i < p.D; i += ntg) o[i] = bfloat(float(xr[i]) * r * float(w[i]));
    return;
  }
  const int kg = p.D / 64, nsg = ntg / 32;
  for (int g = sg; g < kg; g += nsg) {
    const int i = g * 64 + lane * 2;
    bfloat b0 = bfloat(float(xr[i]) * r * float(w[i])), b1 = bfloat(float(xr[i + 1]) * r * float(w[i + 1]));
    o[i] = b0;
    o[i + 1] = b1;
    float2 v = float2(float(b0), float(b1));
    float amax = simd_max(max(fabs(v.x), fabs(v.y)));
    float sx = amax > 0 ? amax / 127.f : 1.f;
    float2 q = rint(v / sx);
    U[(size_t)row * p.D + i] = char(q.x);
    U[(size_t)row * p.D + i + 1] = char(q.y);
    float qs = simd_sum(q.x + q.y);
    if (lane == 0) XS[(size_t)row * kg + g] = float2(sx, sx * qs);
  }
}
#define SPLITK_RES_NORM(A8, NAME)                                                                               \
  template [[host_name(NAME)]] kernel void splitk_res_norm<A8>(device bfloat*, device const bfloat*, device bfloat*, \
      constant NormArgs&, device char*, device float2*, device const float*, constant SplitRed&, uint, uint, uint,  \
      uint, uint);
SPLITK_RES_NORM(false, "splitk_res_norm") SPLITK_RES_NORM(true, "splitk_res_norm_a8")

// Same, reading row rows[i] of x for output row i.
kernel void rmsnorm_gather(device const bfloat* x [[buffer(0)]],
                           device const bfloat* w [[buffer(1)]],
                           device bfloat* out [[buffer(2)]],
                           constant NormArgs& p [[buffer(3)]],
                           device const int* rows [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device const bfloat* xr = x + (size_t)rows[row] * p.in_stride;
  float ss = 0;
  for (int i = tid; i < p.D; i += ntg) { float v = xr[i]; ss += v * v; }
  float r = rsqrt(tg_sum(ss, scratch, tid, ntg) / p.D + p.eps);
  device bfloat* o = out + (size_t)row * p.out_stride;
  for (int i = tid; i < p.D; i += ntg) o[i] = bfloat(float(xr[i]) * r * float(w[i]));
}

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet pieces.
// The fused input projection writes, per token, [qkv (2*Hk*Dk + Hv*Dv) | z (Hv*Dv) | b (Hv) | a (Hv)].
struct GdnArgs {
  int S, Hk, Hv, Dk, Dv;  // sequences in the step; heads and head dims
  int conv_dim;           // 2*Hk*Dk + Hv*Dv
  int proj_stride;        // row stride of the fused projection output
  float eps;
  int verify;             // verify step: speculative sequences keep their state; inputs go to the tape
  int sig;                // output gate: sigmoid (Flash-Next) instead of SiLU
};

// Depthwise causal conv (kernel 4) over [state0, state1, state2, x...], then SiLU, for each sequence's run
// of tokens in order; leaves the last 3 inputs in the state.
// conv_state: [nslots, 3, conv_dim] bfloat; w: [conv_dim, 4]. grid (conv_dim, S).
kernel void gdn_conv(device const bfloat* proj [[buffer(0)]],
                     device bfloat* conv_state [[buffer(1)]],
                     device const bfloat* w [[buffer(2)]],
                     device bfloat* out [[buffer(3)]],
                     device const int* slot [[buffer(4)]],
                     constant GdnArgs& p [[buffer(5)]],
                     device const int* seq_start [[buffer(6)]],
                     device const int* seq_len [[buffer(7)]],
                     device bfloat* tape_x [[buffer(8)]],
                     device const int* spec [[buffer(9)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int c = gid.x, b = gid.y;
  if (c >= p.conv_dim) return;
  const bool keep = p.verify && spec[b];
  device bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  float s0 = st[c], s1 = st[p.conv_dim + c], s2 = st[2 * p.conv_dim + c];
  device const bfloat* wc = w + (size_t)c * 4;
  float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
  int t0 = seq_start[b], n = seq_len[b];
  for (int j = 0; j < n; j++) {
    int t = t0 + j;
    float x = proj[(size_t)t * p.proj_stride + c];
    if (p.verify) tape_x[(size_t)t * p.conv_dim + c] = bfloat(x);
    float y = float(bfloat(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x));
    out[(size_t)t * p.conv_dim + c] = bfloat(y / (1.f + exp(-y)));
    s0 = s1; s1 = s2; s2 = x;
  }
  if (keep) return;
  st[c] = bfloat(s0);
  st[p.conv_dim + c] = bfloat(s1);
  st[2 * p.conv_dim + c] = bfloat(s2);
}

// L2-normalise q and k per head (q also scaled by 1/Dk, k by 1/sqrt(Dk), matching mlx_lm normalize_qk),
// and compute beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a + dt_bias)).
// grid: threadgroups (2*Hk + 1, B), Dk threads each. Group 2*Hk does the Hv gates.
kernel void gdn_prep(device bfloat* conv_out [[buffer(0)]],
                     device const bfloat* proj [[buffer(1)]],
                     device const bfloat* A_log [[buffer(2)]],
                     device const bfloat* dt_bias [[buffer(3)]],
                     device float* g_out [[buffer(4)]],
                     device float* beta_out [[buffer(5)]],
                     constant GdnArgs& p [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float scratch[32];
  int h = tg.x, b = tg.y;
  if (h < 2 * p.Hk) {
    device bfloat* v = conv_out + (size_t)b * p.conv_dim + h * p.Dk;
    float x = tid < (uint)p.Dk ? float(v[tid]) : 0.f;
    float inv = rsqrt(float(p.Dk));
    float rms_eps = 1e-6f * inv * inv;
    float r = rsqrt(tg_sum(x * x, scratch, tid, ntg) / p.Dk + rms_eps);
    float scale = h < p.Hk ? inv * inv : inv;
    if (tid < (uint)p.Dk) v[tid] = bfloat(scale * float(bfloat(x * r)));
  } else {
    for (int i = tid; i < p.Hv; i += ntg) {
      device const bfloat* row = proj + (size_t)b * p.proj_stride + p.conv_dim + p.Hv * p.Dv;
      float bb = row[i], aa = row[p.Hv + i];
      beta_out[b * p.Hv + i] = float(bfloat(1.f / (1.f + exp(-bb))));
      float sp = float(bfloat(aa + float(dt_bias[i])));
      sp = sp > 20.f ? sp : log(1.f + exp(sp));
      g_out[b * p.Hv + i] = exp(-exp(float(A_log[i])) * sp);
    }
  }
}

// Recurrence over each sequence's tokens, per (sequence, value head, value row). The state row stays in
// registers across the run. state: [nslots, Hv, Dv, Dk] float.
// grid: threads (32, Dv, S*Hv), threadgroup (32, 4, 1). Each simdgroup owns one Dv row.
kernel void gdn_step(device const bfloat* conv_out [[buffer(0)]],
                     device const float* g [[buffer(1)]],
                     device const float* beta [[buffer(2)]],
                     device float* state [[buffer(3)]],
                     device bfloat* y [[buffer(4)]],
                     device const int* slot [[buffer(5)]],
                     constant GdnArgs& p [[buffer(6)]],
                     device const int* seq_start [[buffer(7)]],
                     device const int* seq_len [[buffer(8)]],
                     device const int* spec [[buffer(9)]],
                     uint3 gid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const int n = gid.z, b = n / p.Hv, hv = n % p.Hv, hk = hv / (p.Hv / p.Hk);
  const int dv = gid.y;
  constexpr int NPT = 4;  // Dk / 32 for Dk = 128
  device float* st = state + (((size_t)slot[b] * p.Hv + hv) * p.Dv + dv) * p.Dk;
  float s[NPT];
#pragma unroll
  for (int i = 0; i < NPT; i++) s[i] = st[NPT * lane + i];
  int t0 = seq_start[b], nt = seq_len[b];
  for (int j = 0; j < nt; j++) {
    int t = t0 + j;
    device const bfloat* row = conv_out + (size_t)t * p.conv_dim;
    device const bfloat* q = row + hk * p.Dk;
    device const bfloat* k = row + p.Hk * p.Dk + hk * p.Dk;
    float v = row[2 * p.Hk * p.Dk + hv * p.Dv + dv];
    float gg = g[t * p.Hv + hv], bt = beta[t * p.Hv + hv];
    float kk[NPT], kv = 0;
#pragma unroll
    for (int i = 0; i < NPT; i++) {
      kk[i] = k[NPT * lane + i];
      s[i] *= gg;
      kv += s[i] * kk[i];
    }
    kv = simd_sum(kv);
    float delta = (v - kv) * bt, o = 0;
#pragma unroll
    for (int i = 0; i < NPT; i++) {
      s[i] += kk[i] * delta;
      o += s[i] * float(q[NPT * lane + i]);
    }
    o = simd_sum(o);
    if (lane == 0) y[((size_t)t * p.Hv + hv) * p.Dv + dv] = bfloat(o);
  }
  if (p.verify && spec[b]) return;
#pragma unroll
  for (int i = 0; i < NPT; i++) st[NPT * lane + i] = s[i];
}

// out = rmsnorm(y_head) * w * silu(z), per (sequence, value head). Dv threads per threadgroup.
kernel void gdn_out_norm(device const bfloat* y [[buffer(0)]],
                         device const bfloat* proj [[buffer(1)]],
                         device const bfloat* w [[buffer(2)]],
                         device bfloat* out [[buffer(3)]],
                         constant GdnArgs& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float scratch[32];
  int hv = tg.x, b = tg.y;
  size_t base = ((size_t)b * p.Hv + hv) * p.Dv;
  float x = tid < (uint)p.Dv ? float(y[base + tid]) : 0.f;
  float r = rsqrt(tg_sum(x * x, scratch, tid, ntg) / p.Dv + p.eps);
  if (tid >= (uint)p.Dv) return;
  float nx = float(bfloat(x * r * float(w[tid])));
  float z = proj[(size_t)b * p.proj_stride + p.conv_dim + hv * p.Dv + tid];
  out[base + tid] = bfloat(nx * (p.sig ? float(bfloat(1.f / (1.f + exp(-z)))) : z / (1.f + exp(-z))));
}

// ---------------------------------------------------------------------------------------------
// MLP activation: h = silu(gate) * up, where gu = [gate (I) | up (I)] per row.
struct ActArgs { int I; };
kernel void silu_mul(device const bfloat* gu [[buffer(0)]],
                     device bfloat* h [[buffer(1)]],
                     constant ActArgs& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int i = gid.x, b = gid.y;
  if (i >= p.I) return;
  float g = gu[(size_t)b * 2 * p.I + i], u = gu[(size_t)b * 2 * p.I + p.I + i];
  h[(size_t)b * p.I + i] = bfloat(float(bfloat(g / (1.f + exp(-g)))) * u);
}

// silu_mul fused with the W4A8 activation quantization of the down projection: U int8[rows, I], XS float2[rows,
// I/64] from the bf16-rounded activation (bit-identical to silu_mul + a8_quant; the bf16 activation is not
// written). grid (I/64 * 32, rows), threadgroups of 256: one simdgroup per 64-wide group, 2 values per lane.
kernel void silu_mul_a8(device const bfloat* gu [[buffer(0)]],
                        device char* U [[buffer(1)]],
                        device float2* XS [[buffer(2)]],
                        constant ActArgs& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const int g = gid.x / 32, b = gid.y, kg = p.I / 64;
  if (g >= kg) return;
  const int i = g * 64 + lane * 2;
  device const bfloat* row = gu + (size_t)b * 2 * p.I;
  float2 v;
  for (int j = 0; j < 2; j++) {
    float gg = row[i + j], u = row[p.I + i + j];
    v[j] = float(bfloat(float(bfloat(gg / (1.f + exp(-gg)))) * u));
  }
  float amax = simd_max(max(fabs(v.x), fabs(v.y)));
  float sx = amax > 0 ? amax / 127.f : 1.f;
  float2 q = rint(v / sx);
  U[(size_t)b * p.I + i] = char(q.x);
  U[(size_t)b * p.I + i + 1] = char(q.y);
  float qs = simd_sum(q.x + q.y);
  if (lane == 0) XS[(size_t)b * kg + g] = float2(sx, sx * qs);
}

// x += y, rows of length I. grid (I, M).
kernel void add_inplace(device bfloat* x [[buffer(0)]],
                        device const bfloat* y [[buffer(1)]],
                        constant ActArgs& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.I) return;
  size_t o = (size_t)gid.y * p.I + gid.x;
  x[o] = bfloat(float(x[o]) + float(y[o]));
}

// ---------------------------------------------------------------------------------------------
// Gated full attention, decode. The fused projection row is [q+gate (Hq * 2*Dh) | k (Hkv*Dh) | v (Hkv*Dh)],
// with each q head laid out as [q (Dh) | gate (Dh)].
// KV pool per layer, in head planes: [Hkv][ns][Dh] for K and for V, ns = slots (nblocks * BLK), slot = blk * BLK + t,
// so one head's block (and a run of consecutive blocks) is contiguous. block_table: [B, max_blocks].
struct AttnArgs {
  int T, Hq, Hkv, Dh, rot;  // token rows in the step; rot = rotary dims (partial rotary)
  int proj_stride, max_blocks;
  float eps, theta, scale;
  int ns;
  int qs, koff, voff, rpair, rden, vnorm;  // q head stride in proj, k/v offsets, rope pairing, V norm (Gemma)  // slots per head plane
};
constant constexpr int BLK = 32;

// 8-bit KV cache (function constant 20): K/V stored as int8 with one float scale per (kv head, slot) at
// kh * ns + slot, absmax / 127 over the head's Dh values. The same element offsets address either precision.
constant bool kv8_fc [[function_constant(20)]];
constant bool kv8 = is_function_constant_defined(kv8_fc) && kv8_fc;
// 4-bit KV (function constant 21, with 20 also set: every scale path is KV8's): two signed nibbles a byte, the even
// element in the low nibble (Metal's int4b_format order), one float scale per (kv head, slot), absmax / 7 of the
// Hadamard-rotated head (KV4_ROT, below). Element offsets are KV8's; the byte is offset / 2.
constant bool kv4_fc [[function_constant(21)]];
constant bool kv4 = is_function_constant_defined(kv4_fc) && kv4_fc;
// KV4 rotations (function constants 22: K and the queries, 23: V and the attention output): a 256-point Hadamard
// transform before quantizing spreads a head's outlier channels over all dims, so one per-token scale fits. Scores
// are unchanged (q and k rotate alike); a rotated V makes the attention output rotated, and the combine / gate
// kernels rotate it back (the transform is its own inverse) before the gate.
constant bool kv4_rk_fc [[function_constant(22)]];
constant bool kv4_rv_fc [[function_constant(23)]];
constant bool kv4_rk = kv4 && is_function_constant_defined(kv4_rk_fc) && kv4_rk_fc;
constant bool kv4_rv = kv4 && is_function_constant_defined(kv4_rv_fc) && kv4_rv_fc;
inline float4 kv4_unpack(uint w) {  // 4 nibbles (low first) of a 16-bit word, sign-extended
  return float4(float(int(w << 28) >> 28), float(int(w << 24) >> 28), float(int(w << 20) >> 28), float(int(w << 16) >> 28));
}
inline float4 kv_load4(device const bfloat* base, size_t off) {
  if (kv4) return kv4_unpack(uint(*(device const ushort*)((device const uchar*)base + (off >> 1))));
  if (kv8) return float4(*(device const char4*)((device const char*)base + off));
  return float4(*(device const bfloat4*)(base + off));
}
inline float kv_load1(device const bfloat* base, size_t off) {
  if (kv4) {
    const int b = int(((device const uchar*)base)[off >> 1]);
    return float((off & 1) ? (b << 24) >> 28 : (b << 28) >> 28);
  }
  if (kv8) return float(((device const char*)base)[off]);
  return float(base[off]);
}
// In-place 256-point (any power of two up to the threadgroup size) orthonormal Hadamard transform of x[0, n), one
// element per thread tid < n; every thread of the threadgroup must call it (barriers).
inline float tg_hadamard(threadgroup float* x, float v, uint tid, int n) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < (uint)n) x[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int h = 1; h < n; h <<= 1) {
    float a = 0, b = 0;
    if (tid < (uint)n) { a = x[tid]; b = x[tid ^ h]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)n) x[tid] = (tid & h) ? b - a : a + b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return tid < (uint)n ? x[tid] * rsqrt(float(n)) : 0.f;
}

// q/k RMSNorm + RoPE on the first `rot` dims (non-traditional halves), write k/v into the cache,
// q into q_out [B, Hq, Dh]. grid: threadgroups (Hq + Hkv, B), Dh threads.
kernel void attn_prep(device const bfloat* proj [[buffer(0)]],
                      device const bfloat* qn_w [[buffer(1)]],
                      device const bfloat* kn_w [[buffer(2)]],
                      device bfloat* q_out [[buffer(3)]],
                      device bfloat* Kc [[buffer(4)]],
                      device bfloat* Vc [[buffer(5)]],
                      device const int* block_table [[buffer(6)]],
                      device const int* pos [[buffer(7)]],
                      constant AttnArgs& p [[buffer(8)]],
                      device const int* tseq [[buffer(9)]],
                      device float* Ks [[buffer(10)]],
                      device float* Vs [[buffer(11)]],
                      device const int* rpos [[buffer(12)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup float scratch[32];
  threadgroup float xn[512];
  int h = tg.x, b = tg.y;
  bool isq = h < p.Hq;
  int kh = h - p.Hq;
  device const bfloat* row = proj + (size_t)b * p.proj_stride;
  device const bfloat* src = isq ? row + h * p.qs : row + p.koff + kh * p.Dh;
  float x = tid < (uint)p.Dh ? float(src[tid]) : 0.f;
  float r = rsqrt(tg_sum(x * x, scratch, tid, ntg) / p.Dh + p.eps);
  device const bfloat* w = isq ? qn_w : kn_w;
  if (tid < (uint)p.Dh) xn[tid] = float(bfloat(x * r * float(w[tid])));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid >= (uint)p.Dh && !kv8) return;  // (with kv8 every thread joins the absmax reductions; ntg == Dh)
  int ps = pos[b];
  float val = xn[tid];
  // Rotate pairs (i, i + rpair) for i < rot / 2, frequency theta^(-2i / rden): rpair = rot / 2, rden = rot for the
  // usual partial rope; Gemma-4's proportional rope pairs across the whole head (rpair = Dh / 2, rden = Dh).
  const int rpair = p.rpair;
  if (tid < (uint)(2 * rpair) && (int)(tid % rpair) < p.rot / 2) {
    int i = tid % rpair;
    float freq = pow(p.theta, -2.f * i / p.rden);
    // Interleaved M-RoPE: frequency i rotates by position component i % 3 (t, h, w); text rows have t = h = w.
    float ang = float(rpos[b * 3 + i % 3]) * freq, c = cos(ang), s = sin(ang);
    val = tid < (uint)rpair ? xn[tid] * c - xn[tid + rpair] * s : xn[tid] * c + xn[tid - rpair] * s;
  }
  // KV4 rotation of q and k (uniform per threadgroup: every thread takes part)
  if (kv4_rk) val = tg_hadamard(xn, val, tid, p.Dh);
  if (isq) {
    q_out[((size_t)b * p.Hq + h) * p.Dh + tid] = bfloat(val);
  } else {
    int blk = block_table[tseq[b] * p.max_blocks + ps / BLK], off = ps % BLK;
    const size_t si = (size_t)kh * p.ns + (size_t)blk * BLK + off, o = si * p.Dh + tid;
    bfloat v = row[p.voff + kh * p.Dh + tid];
    if (p.vnorm) {  // scaleless rms norm over the head (Gemma-4)
      float vf = tid < (uint)p.Dh ? float(v) : 0.f;
      float vr = rsqrt(tg_sum(vf * vf, scratch, tid, ntg) / p.Dh + p.eps);
      v = bfloat(vf * vr);
    }
    if (kv4) {
      float kq = val, vq = float(v);
      if (kv4_rv) vq = tg_hadamard(xn, vq, tid, p.Dh);
      float ka = tg_max(fabs(kq), scratch, tid, ntg), va = tg_max(fabs(vq), scratch, tid, ntg);
      // clipped scales (0.9 x absmax / 7): rotated heads are near-Gaussian, and clipping the rare extremes lowers the
      // error of everything else (fake-quant on the 27B, 8k: KL 0.0097 -> 0.0075 vs bf16 KV; 0.8: 0.0095)
      float ksc = max(ka, 1e-20f) * (0.9f / 7.f), vsc = max(va, 1e-20f) * (0.9f / 7.f);
      const int kq4 = int(clamp(rint(kq / ksc), -7.f, 7.f)), vq4 = int(clamp(rint(vq / vsc), -7.f, 7.f));
      const int kh4 = simd_shuffle_down(kq4, (ushort)1), vh4 = simd_shuffle_down(vq4, (ushort)1);
      if (!(tid & 1)) {  // the even element's thread writes the byte: low nibble even, high nibble odd
        ((device uchar*)Kc)[o >> 1] = uchar((kq4 & 15) | ((kh4 & 15) << 4));
        ((device uchar*)Vc)[o >> 1] = uchar((vq4 & 15) | ((vh4 & 15) << 4));
      }
      if (tid == 0) {
        Ks[si] = ksc;
        Vs[si] = vsc;
      }
    } else if (kv8) {
      float kq = float(bfloat(val)), vq = float(v);
      float ka = tg_max(fabs(kq), scratch, tid, ntg), va = tg_max(fabs(vq), scratch, tid, ntg);
      float ksc = max(ka, 1e-20f) / 127.f, vsc = max(va, 1e-20f) / 127.f;
      ((device char*)Kc)[o] = char(clamp(rint(kq / ksc), -127.f, 127.f));
      ((device char*)Vc)[o] = char(clamp(rint(vq / vsc), -127.f, 127.f));
      if (tid == 0) {
        Ks[si] = ksc;
        Vs[si] = vsc;
      }
    } else {
      Kc[o] = bfloat(val);
      Vc[o] = v;
    }
  }
}

// Decode attention over positions [0, pos] for each (sequence, q head), flash-decoding style:
// 8 simdgroups split the positions; each lane holds Dh/32 dims. Output is multiplied by sigmoid(gate).
// grid: threadgroups (Hq, B), 256 threads.
kernel void attn_decode(device const bfloat* q [[buffer(0)]],
                        device const bfloat* Kc [[buffer(1)]],
                        device const bfloat* Vc [[buffer(2)]],
                        device const int* block_table [[buffer(3)]],
                        device const int* pos [[buffer(4)]],
                        device const bfloat* proj [[buffer(5)]],
                        device bfloat* out [[buffer(6)]],
                        constant AttnArgs& p [[buffer(7)]],
                        device const int* tseq [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NSG = 8, DPL = 8;  // Dh = 256 = 32 lanes * 8
  threadgroup float tm[NSG], tl[NSG];
  threadgroup float tacc[NSG][256];
  int h = tg.x, b = tg.y, kh = h / (p.Hq / p.Hkv);
  int L = pos[b] + 1, sq = tseq[b];
  float qv[DPL];
  device const bfloat* qp = q + ((size_t)b * p.Hq + h) * p.Dh + lane * DPL;
#pragma unroll
  for (int i = 0; i < DPL; i++) qv[i] = float(qp[i]) * p.scale;
  float m = -INFINITY, l = 0, acc[DPL] = {0};
  for (int t = sg; t < L; t += NSG) {
    int blk = block_table[sq * p.max_blocks + t / BLK];
    size_t o = ((size_t)kh * p.ns + (size_t)blk * BLK + t % BLK) * p.Dh + lane * DPL;
    float sc = 0;
#pragma unroll
    for (int i = 0; i < DPL; i++) sc += qv[i] * float(Kc[o + i]);
    sc = simd_sum(sc);
    float mn = max(m, sc), corr = exp(m - mn), e = exp(sc - mn);
    l = l * corr + e;
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = acc[i] * corr + e * float(Vc[o + i]);
    m = mn;
  }
  if (lane == 0) { tm[sg] = m; tl[sg] = l; }
#pragma unroll
  for (int i = 0; i < DPL; i++) tacc[sg][lane * DPL + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M = -INFINITY;
  for (int s = 0; s < NSG; s++) M = max(M, tm[s]);
  float den = 0, num = 0;
  for (int s = 0; s < NSG; s++) {
    float f = tm[s] == -INFINITY ? 0.f : exp(tm[s] - M);
    den += tl[s] * f;
    num += tacc[s][tid] * f;
  }
  float gate = proj[(size_t)b * p.proj_stride + h * 2 * p.Dh + p.Dh + tid];
  float o = float(bfloat(num / den));
  out[((size_t)b * p.Hq + h) * p.Dh + tid] = bfloat(o * (1.f / (1.f + exp(-gate))));
}

// GQA-grouped attention over [0, pos] for every token row, split across positions (flash-decoding).
// One threadgroup per (kv head, sequence, split). Its simdgroups take the sequence's rows (row r, r+8, ...) and
// all walk the same positions, so K/V comes from DRAM once and is shared through the cache: a verify block of
// R rows costs about one row's bandwidth. Within a simdgroup the GQ = Hq/Hkv query heads sharing the kv head are
// handled together, one position per lane: scores need no cross-lane traffic, the softmax update costs one
// simd_max/simd_sum per head per 32 positions, and V is accumulated with lanes over dims using broadcast
// weights. Writes unnormalised partials [T, Hq, splits] x {acc[Dh], m, l} for attn_combine.
// grid: threadgroups (Hkv, S, splits), threads (32, 8). Requires Dh = 256 and chunk % 32 == 0.
struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles, own_splits, ns, win, gated; };
// GQ query heads per kv head, D head dim; GROWS rows per threadgroup keep qs within 24 KB.
template <int GQ, int D>
kernel void attn_gqa(device const bfloat* q [[buffer(0)]],
                     device const bfloat* Kc [[buffer(1)]],
                     device const bfloat* Vc [[buffer(2)]],
                     device const int* block_table [[buffer(3)]],
                     device const int* pos [[buffer(4)]],
                     device const int* seq_start [[buffer(5)]],
                     device float* part [[buffer(6)]],
                     constant GqaArgs& p [[buffer(7)]],
                     device const int* rflag [[buffer(8)]],
                     device const int* seq_len [[buffer(9)]],
                     device const float* Ks [[buffer(10)]],
                     device const float* Vs [[buffer(11)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr int DPL = D / 32, GROWS = GQ * D <= 1536 ? 8 : GQ * D <= 3072 ? 4 : 2;
  threadgroup bfloat qs[GROWS][GQ][D];
  const int kh = tg.x, b = tg.y, z = tg.z;
  const int r0 = seq_start[b], nrows = seq_len[b];
  if (rflag[r0] != 2) return;  // single-row (attn_gqa_row) and flash sequences are handled elsewhere
  const size_t rs = D;  // consecutive slots of a head plane
  for (int r = sg; r < nrows; r += GROWS) {
    const int t = r0 + r, L = pos[t] + 1;
    const int p0 = z * p.chunk, p1 = min(L, p0 + p.chunk);
    for (int i = lane; i < GQ * D; i += 32)
      qs[sg][i / D][i % D] = bfloat(float(q[((size_t)t * p.Hq + kh * GQ + i / D) * D + i % D]) * p.scale);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float acc[GQ][DPL], m[GQ], l[GQ];
#pragma unroll
    for (int j = 0; j < GQ; j++) {
      m[j] = -INFINITY;
      l[j] = 0;
#pragma unroll
      for (int i = 0; i < DPL; i++) acc[j][i] = 0;
    }
    const int ws = p.win > 0 ? max(0, L - p.win) : 0;  // sliding window: first visible position
    for (int base = p0; base < p1; base += BLK) {
      if (base + BLK <= ws) continue;
      const int blk = block_table[b * p.max_blocks + base / BLK];
      const bool valid = base + (int)lane < p1 && base + (int)lane >= ws;
      const size_t kofs = ((size_t)kh * p.ns + (size_t)blk * BLK) * D + lane * rs;
      float sc[GQ];
#pragma unroll
      for (int j = 0; j < GQ; j++) sc[j] = 0;
      float vsc = 1.f;
      if (valid) {
        for (int d4 = 0; d4 < D / 4; d4++) {
          float4 k4 = kv_load4(Kc, kofs + 4 * d4);
#pragma unroll
          for (int j = 0; j < GQ; j++) {
            float4 q4 = float4(*(threadgroup const bfloat4*)&qs[sg][j][4 * d4]);
            sc[j] += dot(k4, q4);
          }
        }
        if (kv8) {
          const size_t si = (size_t)kh * p.ns + (size_t)blk * BLK + lane;
          float ksc = Ks[si];
          vsc = Vs[si];
#pragma unroll
          for (int j = 0; j < GQ; j++) sc[j] *= ksc;
        }
      }
      float e[GQ];
#pragma unroll
      for (int j = 0; j < GQ; j++) {
        float s = valid ? sc[j] : -INFINITY;
        float mn = max(m[j], simd_max(s));
        float corr = exp(m[j] - mn);
        e[j] = valid ? exp(s - mn) : 0.f;
        l[j] = l[j] * corr + simd_sum(e[j]);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[j][i] *= corr;
        m[j] = mn;
      }
      const size_t vofs = ((size_t)kh * p.ns + (size_t)blk * BLK) * D + lane * DPL;
      const int n = min(BLK, p1 - base);
      for (int rr = 0; rr < n; rr++) {
        const float vs_r = kv8 ? simd_shuffle(vsc, (ushort)rr) : 1.f;
        if constexpr (DPL == 8) {  // two named float4s: an indexed float4 array made 16-64 rows 15-23% slower
          float4 v0 = kv_load4(Vc, vofs + rr * rs), v1 = kv_load4(Vc, vofs + rr * rs + 4);
#pragma unroll
          for (int j = 0; j < GQ; j++) {
            float w = simd_shuffle(e[j], (ushort)rr) * vs_r;
            acc[j][0] += w * v0.x; acc[j][1] += w * v0.y; acc[j][2] += w * v0.z; acc[j][3] += w * v0.w;
            acc[j][4] += w * v1.x; acc[j][5] += w * v1.y; acc[j][6] += w * v1.z; acc[j][7] += w * v1.w;
          }
        } else {
          float4 v[DPL / 4];
#pragma unroll
          for (int c = 0; c < DPL / 4; c++) v[c] = kv_load4(Vc, vofs + rr * rs + 4 * c);
#pragma unroll
          for (int j = 0; j < GQ; j++) {
            float w = simd_shuffle(e[j], (ushort)rr) * vs_r;
#pragma unroll
            for (int c = 0; c < DPL / 4; c++) {
              acc[j][4 * c] += w * v[c].x; acc[j][4 * c + 1] += w * v[c].y; acc[j][4 * c + 2] += w * v[c].z;
              acc[j][4 * c + 3] += w * v[c].w;
            }
          }
        }
      }
    }
    const int stride = D + 2;
#pragma unroll
    for (int j = 0; j < GQ; j++) {
      device float* out = part + (((size_t)t * p.Hq + kh * GQ + j) * p.splits + z) * stride;
#pragma unroll
      for (int i = 0; i < DPL; i++) out[lane * DPL + i] = acc[j][i];
      if (lane == 0) { out[D] = m[j]; out[D + 1] = l[j]; }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Single-row variant (decode): one threadgroup per (kv head, token row, split); 4 simdgroups split the
// positions, lanes over positions as in attn_gqa, merged through threadgroup memory. Rows whose mode is not 0
// (flash or multi-row) are skipped. grid: threadgroups (Hkv, T, splits), 128 threads.
template <int GQ, int D>
kernel void attn_gqa_row(device const bfloat* q [[buffer(0)]],
                         device const bfloat* Kc [[buffer(1)]],
                         device const bfloat* Vc [[buffer(2)]],
                         device const int* block_table [[buffer(3)]],
                         device const int* pos [[buffer(4)]],
                         device const int* tseq [[buffer(5)]],
                         device float* part [[buffer(6)]],
                         constant GqaArgs& p [[buffer(7)]],
                         device const int* rflag [[buffer(8)]],
                         device const float* Ks [[buffer(9)]],
                         device const float* Vs [[buffer(10)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int DPL = D / 32, GSG = 4, TS = GQ * D <= 1536 ? 4 : GQ * D <= 2048 ? 2 : 1;  // TS: simdgroups merged in tacc
  using QT = metal::conditional_t<(GQ * D > 2048), bfloat, float>;  // keeps qs + tacc within 32 KB
  threadgroup QT qs[GQ][D];
  threadgroup float tm[GSG][GQ], tl[GSG][GQ];
  threadgroup float tacc[TS][GQ][D];
  const int kh = tg.x, t = tg.y, z = tg.z;
  if (rflag[t] != 0) return;
  const int L = pos[t] + 1, sq = tseq[t];
  const int p0 = z * p.chunk, p1 = min(L, p0 + p.chunk);
  for (int i = tid; i < GQ * D; i += GSG * 32)
    qs[i / D][i % D] = QT(float(q[((size_t)t * p.Hq + kh * GQ + i / D) * D + i % D]) * p.scale);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const size_t rs = D;  // consecutive slots of a head plane
  float acc[GQ][DPL], m[GQ], l[GQ];
#pragma unroll
  for (int j = 0; j < GQ; j++) {
    m[j] = -INFINITY;
    l[j] = 0;
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[j][i] = 0;
  }
  const int ws = p.win > 0 ? max(0, L - p.win) : 0;  // sliding window: first visible position
  for (int base = p0 + (int)sg * BLK; base < p1; base += GSG * BLK) {
    if (base + BLK <= ws) continue;
    const int blk = block_table[sq * p.max_blocks + base / BLK];
    const bool valid = base + (int)lane < p1 && base + (int)lane >= ws;
    const size_t kofs = ((size_t)kh * p.ns + (size_t)blk * BLK) * D + lane * rs;
    float sc[GQ];
#pragma unroll
    for (int j = 0; j < GQ; j++) sc[j] = 0;
    float vsc = 1.f;
    if (valid) {
      for (int d4 = 0; d4 < D / 4; d4++) {
        float4 k4 = kv_load4(Kc, kofs + 4 * d4);
#pragma unroll
        for (int j = 0; j < GQ; j++)
          sc[j] += k4.x * float(qs[j][4 * d4]) + k4.y * float(qs[j][4 * d4 + 1]) + k4.z * float(qs[j][4 * d4 + 2]) +
                   k4.w * float(qs[j][4 * d4 + 3]);
      }
      if (kv8) {
        const size_t si = (size_t)kh * p.ns + (size_t)blk * BLK + lane;
        float ksc = Ks[si];
        vsc = Vs[si];
#pragma unroll
        for (int j = 0; j < GQ; j++) sc[j] *= ksc;
      }
    }
    float e[GQ];
#pragma unroll
    for (int j = 0; j < GQ; j++) {
      float s = valid ? sc[j] : -INFINITY;
      float mn = max(m[j], simd_max(s));
      float corr = exp(m[j] - mn);
      e[j] = valid ? exp(s - mn) : 0.f;
      l[j] = l[j] * corr + simd_sum(e[j]);
#pragma unroll
      for (int i = 0; i < DPL; i++) acc[j][i] *= corr;
      m[j] = mn;
    }
    const size_t vofs = ((size_t)kh * p.ns + (size_t)blk * BLK) * D + lane * DPL;
    const int n = min(BLK, p1 - base);
    for (int r = 0; r < n; r++) {
      const float vs_r = kv8 ? simd_shuffle(vsc, (ushort)r) : 1.f;
      if constexpr (DPL == 8) {  // two named float4s: an indexed float4 array made 16-64 rows 15-23% slower
        float4 v0 = kv_load4(Vc, vofs + r * rs), v1 = kv_load4(Vc, vofs + r * rs + 4);
#pragma unroll
        for (int j = 0; j < GQ; j++) {
          float w = simd_shuffle(e[j], (ushort)r) * vs_r;
          acc[j][0] += w * v0.x; acc[j][1] += w * v0.y; acc[j][2] += w * v0.z; acc[j][3] += w * v0.w;
          acc[j][4] += w * v1.x; acc[j][5] += w * v1.y; acc[j][6] += w * v1.z; acc[j][7] += w * v1.w;
        }
      } else {
        float4 v[DPL / 4];
#pragma unroll
        for (int c = 0; c < DPL / 4; c++) v[c] = kv_load4(Vc, vofs + r * rs + 4 * c);
#pragma unroll
        for (int j = 0; j < GQ; j++) {
          float w = simd_shuffle(e[j], (ushort)r) * vs_r;
#pragma unroll
          for (int c = 0; c < DPL / 4; c++) {
            acc[j][4 * c] += w * v[c].x; acc[j][4 * c + 1] += w * v[c].y; acc[j][4 * c + 2] += w * v[c].z;
            acc[j][4 * c + 3] += w * v[c].w;
          }
        }
      }
    }
  }
  if constexpr (TS == GSG) {  // every simdgroup's accumulator fits: the original single-pass merge
#pragma unroll
    for (int j = 0; j < GQ; j++) {
      if (lane == 0) { tm[sg][j] = m[j]; tl[sg][j] = l[j]; }
#pragma unroll
      for (int i = 0; i < DPL; i++) tacc[sg][j][lane * DPL + i] = acc[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {  // wide heads: rescale to the common max in registers, then sum into tacc TS simdgroups at a time
#pragma unroll
    for (int j = 0; j < GQ; j++)
      if (lane == 0) { tm[sg][j] = m[j]; tl[sg][j] = l[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int j = 0; j < GQ; j++) {
      float M = -INFINITY;
      for (int s = 0; s < GSG; s++) M = max(M, tm[s][j]);
      const float f = m[j] == -INFINITY ? 0.f : exp(m[j] - M);
#pragma unroll
      for (int i = 0; i < DPL; i++) acc[j][i] *= f;
    }
    for (int s0 = 0; s0 < GSG; s0 += TS) {
      if ((int)sg >= s0 && (int)sg < s0 + TS) {
#pragma unroll
        for (int j = 0; j < GQ; j++)
#pragma unroll
          for (int i = 0; i < DPL; i++) {
            threadgroup float& d = tacc[sg - s0][j][lane * DPL + i];
            d = s0 == 0 ? acc[j][i] : d + acc[j][i];
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const int stride = D + 2;
  for (int j = 0; j < GQ; j++) {
    float M = -INFINITY;
    for (int s = 0; s < GSG; s++) M = max(M, tm[s][j]);
    device float* out = part + (((size_t)t * p.Hq + kh * GQ + j) * p.splits + z) * stride;
    for (int d = tid; d < D; d += GSG * 32) {
      float num = 0;
      if constexpr (TS == GSG) {
        for (int s = 0; s < GSG; s++) num += tm[s][j] == -INFINITY ? 0.f : tacc[s][j][d] * exp(tm[s][j] - M);
      } else {
        for (int s = 0; s < TS; s++) num += tacc[s][j][d];
      }
      out[d] = num;
    }
    if (tid == 0) {
      float den = 0;
      for (int s = 0; s < GSG; s++) den += tm[s][j] == -INFINITY ? 0.f : tl[s][j] * exp(tm[s][j] - M);
      out[D] = M;
      out[D + 1] = den;
    }
  }
}

#define GQA_INST(G, D) \
  template [[host_name("attn_gqa_g" #G "_d" #D)]] kernel void attn_gqa<G, D>(device const bfloat*, device const bfloat*, \
      device const bfloat*, device const int*, device const int*, device const int*, device float*, constant GqaArgs&, \
      device const int*, device const int*, device const float*, device const float*, uint3, uint, uint); \
  template [[host_name("attn_gqa_row_g" #G "_d" #D)]] kernel void attn_gqa_row<G, D>(device const bfloat*, \
      device const bfloat*, device const bfloat*, device const int*, device const int*, device const int*, device float*, \
      constant GqaArgs&, device const int*, device const float*, device const float*, uint3, uint, uint, uint);
GQA_INST(6, 256)
GQA_INST(4, 256)
GQA_INST(8, 256)
GQA_INST(2, 256)
GQA_INST(12, 256)
GQA_INST(16, 128)
GQA_INST(8, 512)

// Merge attn_gqa's split partials and apply sigmoid(gate). grid: threadgroups (Hq, T), Dh threads.
kernel void attn_combine(device const float* part [[buffer(0)]],
                         device const bfloat* proj [[buffer(1)]],
                         device bfloat* out [[buffer(2)]],
                         constant GqaArgs& p [[buffer(3)]],
                         device const int* rflag [[buffer(4)]],
                         device const int* shrow [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  const int h = tg.x, t = tg.y, stride = p.Dh + 2;
  if (rflag[t] == 1) return;  // flash rows write their output directly
  device const float* pp = part + ((size_t)t * p.Hq + h) * p.splits * stride;
  // Cascade steps: rows of a shared-prefix group also have slots [own_splits, splits) from attn_shared.
  const int nz = p.own_splits > 0 && !shrow[t] ? p.own_splits : p.splits;
  float M = -INFINITY;
  for (int z = 0; z < nz; z++) M = max(M, pp[z * stride + p.Dh]);
  float num = 0, den = 0;
  for (int z = 0; z < nz; z++) {
    float mz = pp[z * stride + p.Dh];
    if (mz == -INFINITY) continue;
    float f = exp(mz - M);
    num += pp[z * stride + tid] * f;
    den += pp[z * stride + p.Dh + 1] * f;
  }
  const float gate = p.gated ? float(proj[(size_t)t * p.proj_stride + h * 2 * p.Dh + p.Dh + tid]) : INFINITY;
  float o;
  if (kv4_rv) {  // KV4 rotated V: rotate the output back
    threadgroup float hb[512];
    o = float(bfloat(tg_hadamard(hb, num / den, tid, p.Dh)));
  } else {
    o = float(bfloat(num / den));
  }
  out[((size_t)t * p.Hq + h) * p.Dh + tid] = bfloat(o * (1.f / (1.f + exp(-gate))));
}

// Copy one sequence's paged K/V (positions [0, n)) into contiguous [n, Hkv, Dh] scratch for the flash kernel.
// grid (Hkv * Dh, n).
struct GatherKvArgs { int HD, n, max_blocks, seq, Dh, ns; };
kernel void kv_gather(device const bfloat* Kc [[buffer(0)]],
                      device const bfloat* Vc [[buffer(1)]],
                      device const int* block_table [[buffer(2)]],
                      device bfloat* Kt [[buffer(3)]],
                      device bfloat* Vt [[buffer(4)]],
                      constant GatherKvArgs& p [[buffer(5)]],
                      device const float* Ks [[buffer(6)]],
                      device const float* Vs [[buffer(7)]],
                      uint2 gid [[thread_position_in_grid]]) {
  int i = gid.x, ps = gid.y;
  if (i >= p.HD || ps >= p.n) return;
  int blk = block_table[p.seq * p.max_blocks + ps / BLK];
  const size_t si = (size_t)(i / p.Dh) * p.ns + (size_t)blk * BLK + ps % BLK;
  size_t src = si * p.Dh + i % p.Dh, dst = (size_t)ps * p.HD + i;
  if (kv8) {  // (and kv4: kv_load1 reads the nibble)
    Kt[dst] = bfloat(kv_load1(Kc, src) * Ks[si]);
    Vt[dst] = bfloat(kv_load1(Vc, src) * Vs[si]);
  } else {
    Kt[dst] = Kc[src];
    Vt[dst] = Vc[src];
  }
}

// out[t, h*Dh + d] *= sigmoid(gate) for rows [row0, row0 + n). grid (Hq * Dh, n).
struct GateArgs { int Hq, Dh, row0, proj_stride, gated; };
// KV4 rotated V (kv4_rv): the flash output is rotated; each 256-thread threadgroup is one head (Dh = 256) of one row
// and rotates it back first. Ungated layers (p.gated 0) only rotate.
kernel void attn_gate(device bfloat* out [[buffer(0)]],
                      device const bfloat* proj [[buffer(1)]],
                      constant GateArgs& p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  int i = gid.x, t = p.row0 + gid.y;
  if (kv4_rv) {
    threadgroup float hb[256];
    size_t o = (size_t)t * p.Hq * p.Dh + i;
    float x = tg_hadamard(hb, float(out[o]), tid, 256);
    int h = i / p.Dh, d = i % p.Dh;
    float gate = p.gated ? float(proj[(size_t)t * p.proj_stride + h * 2 * p.Dh + p.Dh + d]) : INFINITY;
    out[o] = bfloat(float(bfloat(x)) * (1.f / (1.f + exp(-gate))));
    return;
  }
  if (i >= p.Hq * p.Dh) return;
  int h = i / p.Dh, d = i % p.Dh;
  float gate = proj[(size_t)t * p.proj_stride + h * 2 * p.Dh + p.Dh + d];
  size_t o = (size_t)t * p.Hq * p.Dh + i;
  out[o] = bfloat(float(out[o]) * (1.f / (1.f + exp(-gate))));
}

// Gemma-4 MLP: gelu_tanh(gate) * up over [rows, 2I] (gate first) -> [rows, I].
kernel void gelu_mul(device const bfloat* gu [[buffer(0)]],
                     device bfloat* h [[buffer(1)]],
                     constant ActArgs& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int i = gid.x, b = gid.y;
  if (i >= p.I) return;
  float g = gu[(size_t)b * 2 * p.I + i], u = gu[(size_t)b * 2 * p.I + p.I + i];
  float ge = 0.5f * g * (1.f + precise::tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
  h[(size_t)b * p.I + i] = bfloat(float(bfloat(ge)) * u);
}

// x[row, :] *= s. grid (n, rows).
struct ScaleArgs { int n; float s; };
kernel void scale_rows(device bfloat* x [[buffer(0)]], constant ScaleArgs& p [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.n) return;
  device bfloat& v = x[(size_t)gid.y * p.n + gid.x];
  v = bfloat(float(v) * p.s);
}
