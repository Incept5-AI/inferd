// Kernels for the Neural Engine FFN split (src/ane.h): activations into the ANE's fp16 surfaces (channel-major,
// token-minor), the layer's 4-bit weights re-quantized to int8 rows with per-row scales, and the join.
#include <metal_stdlib>
using namespace metal;

// xn bf16 [T, H] -> out fp16 [SEG channels][st] for channels [k0, k0 + SEG), tokens [0, R) (zero past T).
// grid (R, SEG).
struct AnePackArgs { int T, H, k0, st; };
kernel void ane_pack_x(device const bfloat* xn [[buffer(0)]], device half* out [[buffer(1)]],
                       constant AnePackArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const int t = gid.x, c = gid.y;
  out[(size_t)c * p.st + t] = t < p.T ? half(xn[(size_t)t * p.H + p.k0 + c]) : half(0);
}

// Rows [n0, n0 + rows) of a 4-bit affine matrix (MLX layout: Wq uint32[N, K/8], S/B bfloat[N, K/64]; flip = 8 when
// its nibbles were flipped for W4A8), columns [c0, c0 + cn). ane_rowscale: one threadgroup (256) per row writes
// sc[row * scst] = 128 * max|w| / 127 (the program dequantizes int8 by 1/128). ane_quant: one thread per 8 columns
// writes round(w * 127 / max|w|) into out[row * ost + (column - c0 - q0)] for columns [c0 + q0, c0 + q0 + qn).
struct AneWArgs { int K, n0, c0, cn, flip, scst, ost, q0, qn; };
inline float ane_w(device const uint* Wq, device const bfloat* S, device const bfloat* B, int K, int n, int k,
                   int flip, thread uint& word, thread int& wk) {
  if (k / 8 != wk) { wk = k / 8; word = Wq[(size_t)n * (K / 8) + wk]; }
  const int q = int((word >> (4 * (k % 8))) & 15u) ^ flip;
  const size_t gi = (size_t)n * (K / 64) + k / 64;
  return float(S[gi]) * float(q) + float(B[gi]);
}
kernel void ane_rowscale(device const uint* Wq [[buffer(0)]], device const bfloat* S [[buffer(1)]],
                         device const bfloat* B [[buffer(2)]], device half* sc [[buffer(3)]],
                         constant AneWArgs& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8];
  const int n = p.n0 + row;
  float m = 0;
  uint word = 0;
  int wk = -1;
  for (int k = p.c0 + tid * 8; k < p.c0 + p.cn; k += 256 * 8)
    for (int j = 0; j < 8; j++) m = max(m, fabs(ane_w(Wq, S, B, p.K, n, k + j, p.flip, word, wk)));
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (int i = 1; i < 8; i++) m = max(m, red[i]);
    sc[(size_t)row * p.scst] = half(m > 0 ? m * 128.f / 127.f : 1.f);
  }
}
kernel void ane_quant(device const uint* Wq [[buffer(0)]], device const bfloat* S [[buffer(1)]],
                      device const bfloat* B [[buffer(2)]], device const half* sc [[buffer(3)]],
                      device char* out [[buffer(5)]], constant AneWArgs& p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int row = gid.y, k = p.c0 + p.q0 + gid.x * 8;
  if (gid.x * 8 >= (uint)p.qn) return;
  const int n = p.n0 + row;
  const float inv = 128.f / float(sc[(size_t)row * p.scst]);
  uint word = 0;
  int wk = -1;
  device char* o = out + (size_t)row * p.ost + gid.x * 8;
  for (int j = 0; j < 8; j++) o[j] = char(clamp(rint(ane_w(Wq, S, B, p.K, n, k + j, p.flip, word, wk) * inv), -127.f, 127.f));
}

// x[t, h] += y[h][t] for t < T (the ANE's partial down projection into the residual). grid (H, T).
struct AneJoinArgs { int H, st; };
kernel void ane_join(device bfloat* x [[buffer(0)]], device const half* y [[buffer(1)]],
                     constant AneJoinArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const size_t o = (size_t)gid.y * p.H + gid.x;
  x[o] = bfloat(float(x[o]) + float(y[(size_t)gid.x * p.st + gid.y]));
}

// act[t, c] = silu(g[t, c]) * u[t, c] over the GPU's channels (gate and up computed into buffers of their own).
// grid (Cg, T).
struct AneSiluArgs { int C; };
kernel void silu_mul2(device const bfloat* g [[buffer(0)]], device const bfloat* u [[buffer(1)]],
                      device bfloat* act [[buffer(2)]], constant AneSiluArgs& p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const size_t o = (size_t)gid.y * p.C + gid.x;
  const float a = float(g[o]);
  act[o] = bfloat(a / (1.f + exp(-a)) * float(u[o]));
}

// ---- W8A8 gate/up (ANE_A8): random-sign 512-point Walsh-Hadamard rotation of every input row (activations per
// token, weights per output row) so per-row int8 scales survive outlier channels; R = H D is orthonormal and
// symmetric, so (W R^T)(R x) = W x with the same transform on both sides.
inline float ane_sign(int k) { uint h = uint(k) * 2654435761u; h ^= h >> 15; return (h & 1u) ? -1.f : 1.f; }
// In-place orthonormal FWHT of n = 512 * nb values in threadgroup memory (256 threads), per 512-block.
inline void ane_fwht(threadgroup float* v, int nb, uint tid) {
  for (int b = 0; b < nb; b++) {
    threadgroup float* w = v + b * 512;
    for (int h = 1; h < 512; h *= 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const int i = (tid / h) * 2 * h + tid % h;  // 256 butterflies per stage
      const float a = w[i], c = w[i + h];
      w[i] = a + c;
      w[i + h] = a - c;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline float ane_tg_max(float m, threadgroup float* red, uint tid, uint lane, uint sg) {
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (int i = 1; i < 8; i++) m = max(m, red[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return m;
}
// Token t's channels [k0, k0 + 2560) rotated, quantized to int8 into out[c * st + t], scale 128 * max / 127 into
// tx[t]. One threadgroup (256) per (token, segment). grid groups (R, 1).
struct AnePack8Args { int T, H, k0, st; };
kernel void ane_pack_x8(device const bfloat* xn [[buffer(0)]], device char* out [[buffer(1)]],
                        device half* tx [[buffer(2)]], constant AnePack8Args& p [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float v[2560];
  threadgroup float red[8];
  for (int c = tid; c < 2560; c += 256)
    v[c] = (int)t < p.T ? float(xn[(size_t)t * p.H + p.k0 + c]) * ane_sign(p.k0 + c) : 0.f;
  ane_fwht(v, 5, tid);
  float m = 0;
  for (int c = tid; c < 2560; c += 256) m = max(m, fabs(v[c]));
  m = ane_tg_max(m, red, tid, lane, sg) * 0.044194174f;  // 1 / sqrt(512)
  const float inv = m > 0 ? 127.f / m * 0.044194174f : 0.f;
  for (int c = tid; c < 2560; c += 256) out[(size_t)c * p.st + t] = char(clamp(rint(v[c] * inv), -127.f, 127.f));
  if (tid == 0) tx[t] = half(m * 128.f / 127.f);
}
// Rows [n0, n0 + rows) of a 4-bit affine matrix over all K (= 2 x 2560), rotated, int8 per row: segment k's columns
// into outk[row * ost + c], scale 128 * max / 127 into sc[row * scst]. One threadgroup (256) per row.
kernel void ane_req_rot(device const uint* Wq [[buffer(0)]], device const bfloat* S [[buffer(1)]],
                        device const bfloat* B [[buffer(2)]], device half* sc [[buffer(3)]],
                        constant AneWArgs& p [[buffer(4)]], device char* out0 [[buffer(5)]], device char* out1 [[buffer(6)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float v[5120];
  threadgroup float red[8];
  const int n = p.n0 + row;
  for (int w = tid; w < p.K / 8; w += 256) {
    const uint word = Wq[(size_t)n * (p.K / 8) + w];
    const size_t gi = (size_t)n * (p.K / 64) + w / 8;
    const float s = float(S[gi]), b = float(B[gi]);
    for (int j = 0; j < 8; j++) v[w * 8 + j] = (s * float(int((word >> (4 * j)) & 15u) ^ p.flip) + b) * ane_sign(w * 8 + j);
  }
  ane_fwht(v, p.K / 512, tid);
  float m = 0;
  for (int c = tid; c < p.K; c += 256) m = max(m, fabs(v[c]));
  m = ane_tg_max(m, red, tid, lane, sg) * 0.044194174f;
  const float inv = m > 0 ? 127.f / m * 0.044194174f : 0.f;
  for (int c = tid; c < p.K; c += 256) {
    const char q = char(clamp(rint(v[c] * inv), -127.f, 127.f));
    if (c < 2560) out0[(size_t)row * p.ost + c] = q; else out1[(size_t)row * p.ost + c - 2560] = q;
  }
  if (tid == 0) sc[(size_t)row * p.scst] = half(m > 0 ? m * 128.f / 127.f : 1.f);
}

// Down's rows over input columns [c0, c0 + cn) (cn a multiple of 512), rotated per 512-block exactly as the ANE
// program rotates the intermediate (the block Hadamard with the channel signs), int8 per row: block b's columns go to
// segment surface (512 b) / 2560 at column (512 b) % 2560 + i. One threadgroup (256) per row; two passes (max, then
// write) recompute the rotation instead of holding the row.
kernel void ane_req_rot_cols(device const uint* Wq [[buffer(0)]], device const bfloat* S [[buffer(1)]],
                             device const bfloat* B [[buffer(2)]], device half* sc [[buffer(3)]],
                             constant AneWArgs& p [[buffer(4)]], device char* o0 [[buffer(5)]], device char* o1 [[buffer(6)]],
                             device char* o2 [[buffer(7)]], device char* o3 [[buffer(8)]], device char* o4 [[buffer(9)]],
                             device char* o5 [[buffer(10)]], device char* o6 [[buffer(11)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float v[512];
  threadgroup float red[8];
  const int n = p.n0 + row;
  float m = 0, inv = 0;
  for (int pass = 0; pass < 2; pass++) {
    for (int b = 0; b < p.cn / 512; b++) {
      for (int i = tid; i < 512; i += 256) {
        const int k = p.c0 + b * 512 + i;
        const uint word = Wq[(size_t)n * (p.K / 8) + k / 8];
        const size_t gi = (size_t)n * (p.K / 64) + k / 64;
        v[i] = (float(S[gi]) * float(int((word >> (4 * (k % 8))) & 15u) ^ p.flip) + float(B[gi])) * ane_sign(k);
      }
      ane_fwht(v, 1, tid);
      if (pass == 0) {
        float bm = 0;
        for (int i = tid; i < 512; i += 256) bm = max(bm, fabs(v[i]));
        m = max(m, bm);
      } else {
        const int col = b * 512, s = col / 2560;
        device char* o = s == 0 ? o0 : s == 1 ? o1 : s == 2 ? o2 : s == 3 ? o3 : s == 4 ? o4 : s == 5 ? o5 : o6;
        for (int i = tid; i < 512; i += 256)
          o[(size_t)row * p.ost + col % 2560 + i] = char(clamp(rint(v[i] * inv), -127.f, 127.f));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (pass == 0) {
      m = ane_tg_max(m, red, tid, lane, sg) * 0.044194174f;
      inv = m > 0 ? 127.f / m * 0.044194174f : 0.f;
      if (tid == 0) sc[(size_t)row * p.scst] = half(m > 0 ? m * 128.f / 127.f : 1.f);
    }
  }
}

// The join when the ANE quantized the intermediate per token: y holds H rows before each token's intermediate scale,
// and that scale in row H; x[t, h] += y[h][t] * y[H][t] in fp32 (at full scale the product could overflow fp16).
kernel void ane_join_s(device bfloat* x [[buffer(0)]], device const half* y [[buffer(1)]],
                       constant AneJoinArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const size_t o = (size_t)gid.y * p.H + gid.x;
  x[o] = bfloat(float(x[o]) + float(y[(size_t)gid.x * p.st + gid.y]) * float(y[(size_t)p.H * p.st + gid.y]));
}
