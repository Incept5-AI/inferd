// 5- and 6-bit affine GEMV for M <= 2 (decode), MLX layout (values packed as a little-endian bit stream, group 64),
// unflipped. Same shape as qmv8: each lane takes 16 consecutive values of a 512-value block for 4 rows; 16 values of
// BITS bits are 2 * BITS bytes, so every lane's chunk starts on a 2-byte boundary and is read as BITS ushorts.
// Flash-Next's oQ4e checkpoint stores its attention / GDN / hyper-connection projections at 5 and 6 bits; widening
// them to 8 bits made these GEMVs ~half of a decode step.
// grid: threadgroups (ceil(M/R), ceil(N/8)), 64 threads (2 simdgroups x 4 output rows).
#include <metal_stdlib>
using namespace metal;

struct GemmArgs { int M, N, K, xr; };
constant constexpr int G = 64;

template <int BITS, int R, int MODE>
kernel void qmvb(device const bfloat* A [[buffer(0)]],
                 device const uchar* Wq [[buffer(1)]],
                 device const bfloat* S [[buffer(2)]],
                 device const bfloat* Bz [[buffer(3)]],
                 device void* Cv [[buffer(4)]],
                 constant GemmArgs& p [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPT = 16, BLOCK = 32 * VPT, CHUNK = VPT * BITS / 8;  // bytes per lane per block
  constexpr uint MASK = (1u << BITS) - 1;
  const int M = p.M, N = p.N, K = p.K;
  const int row0 = tg.y * 8 + sg * ROWS, m0 = tg.x * R;
  if (row0 >= N) return;
  const int kg = K / G;
  const size_t rowb = (size_t)K * BITS / 8;  // bytes per weight row
  device const uchar* ws = Wq + (size_t)row0 * rowb + lane * CHUNK;
  device const bfloat* sp = S + (size_t)row0 * kg + lane / 4;
  device const bfloat* bp = Bz + (size_t)row0 * kg + lane / 4;
  device const bfloat* xp = A + (size_t)m0 * K + lane * VPT;
  float res[R][ROWS];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) res[r][j] = 0;
  for (int k = 0; k < K; k += BLOCK) {
    if (k + (int)lane * VPT >= K) { ws += BLOCK * BITS / 8; sp += BLOCK / G; bp += BLOCK / G; xp += BLOCK; continue; }
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
      device const ushort* wp = (device const ushort*)(ws + (size_t)j * rowb);
      ulong lo = 0, hi = 0;  // the chunk's 16 * BITS bits (80 or 96)
#pragma unroll
      for (int u = 0; u < BITS; u++) {
        const ulong v = wp[u];
        if (u < 4) lo |= v << (16 * u);
        else hi |= v << (16 * (u - 4));
      }
      float q[VPT];
#pragma unroll
      for (int i = 0; i < VPT; i++) {
        const int b = i * BITS;
        uint v;
        if (b + BITS <= 64) v = uint(lo >> b) & MASK;
        else if (b >= 64) v = uint(hi >> (b - 64)) & MASK;
        else v = uint((lo >> b) | (hi << (64 - b))) & MASK;
        q[i] = float(v);
      }
      const float s = sp[(size_t)j * kg], bb = bp[(size_t)j * kg];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float acc = 0;
#pragma unroll
        for (int i = 0; i < VPT; i++) acc += xt[r][i] * q[i];
        res[r][j] += s * acc + bb * xs[r];
      }
    }
    ws += BLOCK * BITS / 8;
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

#define QVB(BITS, R, MODE)                                                                       \
  template [[host_name("qmv" #BITS "_r" #R "_m" #MODE)]] kernel void qmvb<BITS, R, MODE>(          \
      device const bfloat*, device const uchar*, device const bfloat*, device const bfloat*,     \
      device void*, constant GemmArgs&, uint2, uint, uint);
QVB(5, 1, 0) QVB(5, 1, 1) QVB(5, 1, 2) QVB(5, 2, 0) QVB(5, 2, 1) QVB(5, 2, 2)
QVB(6, 1, 0) QVB(6, 1, 1) QVB(6, 1, 2) QVB(6, 2, 0) QVB(6, 2, 1) QVB(6, 2, 2)
