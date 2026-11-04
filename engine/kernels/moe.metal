// Mixture-of-experts FFN (Qwen3.5/3.6 MoE): router softmax + top-k, a gathered quantized GEMV over each routed
// (token, expert) row, and the weighted combine with the sigmoid-gated shared expert.
#include <metal_stdlib>
using namespace metal;

// y[r, n] = sum_k W[e(r)][n][k] * x[r / div][k] for MLX affine weights (group 64, `bits` 4 or 8, unflipped):
// Wq uint32 [E, N, K*bits/32], S/B bfloat [E, N, K/64]. e(r) = idx[r] (idx null: expert 0).
// grid: threadgroups (ceil(N / 8), R), 256 threads: one simdgroup per output column, lanes over K.
// OUTF: write float (router logits) instead of bfloat.
struct MoeArgs { int N, K, div, use_idx; };
template <int BITS, bool OUTF>
kernel void moe_qmv(device const bfloat* x [[buffer(0)]],
                    device const uint* Wq [[buffer(1)]],
                    device const bfloat* S [[buffer(2)]],
                    device const bfloat* B [[buffer(3)]],
                    device const int* idx [[buffer(4)]],
                    device void* y [[buffer(5)]],
                    constant MoeArgs& p [[buffer(6)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const int n = tg.x * 8 + sg, r = tg.y;
  if (n >= p.N) return;
  const int e = p.use_idx ? idx[r] : 0;
  constexpr int PER = 32 / BITS;  // values per uint32
  const int words = p.K / PER, groups = p.K / 64;
  const size_t row = (size_t)e * p.N + n;
  device const uint* w = Wq + row * words;
  device const bfloat* sc = S + row * groups;
  device const bfloat* bi = B + row * groups;
  device const bfloat* xr = x + (size_t)(r / p.div) * p.K;
  float acc = 0;
  for (int j = lane; j < words; j += 32) {
    const uint u = w[j];
    const int k0 = j * PER, gi = k0 / 64;
    float qx = 0, sx = 0;
#pragma unroll
    for (int i = 0; i < PER; i++) {
      const float xv = float(xr[k0 + i]);
      qx += float((u >> (BITS * i)) & ((1u << BITS) - 1)) * xv;
      sx += xv;
    }
    acc += float(sc[gi]) * qx + float(bi[gi]) * sx;
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (OUTF) ((device float*)y)[(size_t)r * p.N + n] = acc;
    else ((device bfloat*)y)[(size_t)r * p.N + n] = bfloat(acc);
  }
}
template [[host_name("moe_qmv4")]] kernel void moe_qmv<4, false>(device const bfloat*, device const uint*,
    device const bfloat*, device const bfloat*, device const int*, device void*, constant MoeArgs&, uint2, uint, uint);
template [[host_name("moe_qmv8")]] kernel void moe_qmv<8, false>(device const bfloat*, device const uint*,
    device const bfloat*, device const bfloat*, device const int*, device void*, constant MoeArgs&, uint2, uint, uint);
template [[host_name("moe_qmv8f")]] kernel void moe_qmv<8, true>(device const bfloat*, device const uint*,
    device const bfloat*, device const bfloat*, device const int*, device void*, constant MoeArgs&, uint2, uint, uint);
template [[host_name("moe_qmv4f")]] kernel void moe_qmv<4, true>(device const bfloat*, device const uint*,
    device const bfloat*, device const bfloat*, device const int*, device void*, constant MoeArgs&, uint2, uint, uint);

// Router: logits float [T, E + 1] (column E: the shared expert's gate). Softmax over E (fp32), top-k, renormalised
// (norm_topk_prob). Writes idx [T, k], w [T, k], sg [T] = sigmoid(shared gate). One simdgroup per token, E <= 512.
struct RouteArgs { int E, k, stride, norm; };
kernel void moe_route(device const float* logits [[buffer(0)]],
                      device int* idx [[buffer(1)]],
                      device float* wt [[buffer(2)]],
                      device float* sgate [[buffer(3)]],
                      constant RouteArgs& p [[buffer(4)]],
                      uint t [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  device const float* l = logits + (size_t)t * p.stride;
  constexpr int MAXV = 16;  // E / 32 values per lane
  float v[MAXV];
  const int per = (p.E + 31) / 32;
  float mx = -INFINITY;
  for (int i = 0; i < per; i++) {
    const int c = i * 32 + lane;
    v[i] = c < p.E ? l[c] : -INFINITY;
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0;
  for (int i = 0; i < per; i++) { v[i] = exp(v[i] - mx); sum += v[i]; }
  sum = simd_sum(sum);
  float tot = 0;
  for (int j = 0; j < p.k; j++) {
    float best = -1;
    int bi = 0;
    for (int i = 0; i < per; i++)
      if (v[i] > best) { best = v[i]; bi = i * 32 + lane; }
    const float m = simd_max(best);
    const int who = simd_min(best == m ? bi : 1 << 30);
    if ((who & 31) == (int)lane) v[who / 32] = -1;  // take it out
    if (lane == 0) { idx[t * p.k + j] = who; wt[t * p.k + j] = m / sum; }
    tot += m / sum;
  }
  if (lane == 0) {
    if (p.norm)
      for (int j = 0; j < p.k; j++) wt[t * p.k + j] /= tot;
    sgate[t] = 1.f / (1.f + exp(-l[p.E]));
  }
}

// silu(gate) * up over rows of [R, 2I] (gate first) -> [R, I].
struct MoeActArgs { int I; };
kernel void moe_silu_mul(device const bfloat* gu [[buffer(0)]], device bfloat* h [[buffer(1)]],
                         constant MoeActArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x, r = gid.y;
  if (i >= p.I) return;
  const float g = gu[(size_t)r * 2 * p.I + i], u = gu[(size_t)r * 2 * p.I + p.I + i];
  h[(size_t)r * p.I + i] = bfloat(float(bfloat(g / (1.f + exp(-g)))) * u);
}

// x[t, h] += sum_j w[t, j] * yd[t * k + j, h] + sg[t] * ys[t, h]. grid (H, T).
struct CombArgs { int H, k, assign; };  // assign: out = acc (Flash-Next), else x += acc
kernel void moe_combine(device bfloat* x [[buffer(0)]],
                        device const bfloat* yd [[buffer(1)]],
                        device const float* wt [[buffer(2)]],
                        device const bfloat* ys [[buffer(3)]],
                        device const float* sgate [[buffer(4)]],
                        constant CombArgs& p [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int h = gid.x, t = gid.y;
  if (h >= p.H) return;
  float acc = 0;
  for (int j = 0; j < p.k; j++) acc += wt[t * p.k + j] * float(yd[((size_t)t * p.k + j) * p.H + h]);
  acc += sgate[t] * float(ys[(size_t)t * p.H + h]);
  x[(size_t)t * p.H + h] = p.assign ? bfloat(acc) : bfloat(float(x[(size_t)t * p.H + h]) + float(bfloat(acc)));
}

// Counting sort of the routed rows r = t * k + j by expert: offsets [E] (start row of each expert), perm [R] (sorted
// position -> r), inv [R] (r -> sorted position). One threadgroup of 1024 threads, E <= 1024.
struct SortArgs { int R, E; };
kernel void moe_sort(device const int* idx [[buffer(0)]],
                     device int* offsets [[buffer(1)]],
                     device int* perm [[buffer(2)]],
                     device int* inv [[buffer(3)]],
                     constant SortArgs& p [[buffer(4)]],
                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_int cnt[1024];
  threadgroup int start[1024];
  for (int e = tid; e < p.E; e += 1024) atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = tid; r < p.R; r += 1024) atomic_fetch_add_explicit(&cnt[idx[r]], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    int s = 0;
    for (int e = 0; e < p.E; e++) {
      start[e] = s;
      offsets[e] = s;
      s += atomic_load_explicit(&cnt[e], memory_order_relaxed);
      atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = tid; r < p.R; r += 1024) {
    const int e = idx[r];
    const int q = start[e] + atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed);
    perm[q] = r;
    inv[r] = q;
  }
}

// xs[q, :] = x[perm[q] / div, :]. grid (K, R).
struct GatherArgs { int K, div; };
kernel void moe_gather(device const bfloat* x [[buffer(0)]], device const int* perm [[buffer(1)]],
                       device bfloat* xs [[buffer(2)]], constant GatherArgs& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.K) return;
  xs[(size_t)gid.y * p.K + gid.x] = x[(size_t)(perm[gid.y] / p.div) * p.K + gid.x];
}

// moe_combine over sorted expert outputs: x[t, h] += sum_j w[t, j] * yd[inv[t * k + j], h] + sg[t] * ys[t, h].
kernel void moe_combine_sorted(device bfloat* x [[buffer(0)]],
                               device const bfloat* yd [[buffer(1)]],
                               device const float* wt [[buffer(2)]],
                               device const bfloat* ys [[buffer(3)]],
                               device const float* sgate [[buffer(4)]],
                               constant CombArgs& p [[buffer(5)]],
                               device const int* inv [[buffer(6)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const int h = gid.x, t = gid.y;
  if (h >= p.H) return;
  float acc = 0;
  for (int j = 0; j < p.k; j++) acc += wt[t * p.k + j] * float(yd[(size_t)inv[t * p.k + j] * p.H + h]);
  acc += sgate[t] * float(ys[(size_t)t * p.H + h]);
  x[(size_t)t * p.H + h] = p.assign ? bfloat(acc) : bfloat(float(x[(size_t)t * p.H + h]) + float(bfloat(acc)));
}

// Glimmer's logits: l = cap * tanh(l * mult / cap), in place over [rows, n]. grid (n, rows).
struct CapArgs2 { int n; float mult, cap; };
kernel void logit_softcap(device bfloat* l [[buffer(0)]], constant CapArgs2& p [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.n) return;
  device bfloat& v = l[(size_t)gid.y * p.n + gid.x];
  v = bfloat(p.cap * precise::tanh(float(v) * p.mult / p.cap));
}

// Router logits from a bf16 matrix W [N, K] (Flash-Next's unquantized router + the shared gate row): y float [R, N].
// grid: threadgroups (ceil(N / 8), R), 256 threads.
kernel void moe_qmv16f(device const bfloat* x [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                       device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                       device const int* idx [[buffer(4)]], device float* y [[buffer(5)]],
                       constant MoeArgs& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int n = tg.x * 8 + sg, r = tg.y;
  if (n >= p.N) return;
  device const bfloat* w = W + (size_t)n * p.K;
  device const bfloat* xr = x + (size_t)r * p.K;
  float acc = 0;
  for (int k = lane; k < p.K; k += 32) acc += float(w[k]) * float(xr[k]);
  acc = simd_sum(acc);
  if (lane == 0) y[(size_t)r * p.N + n] = acc;
}

// ---- Three-launch MoE for decode-sized steps (MOE_V2; Qwen35::encode_moe) ----
// 8 consecutive affine weights (row bytes w, column k, BITS 4 or 8, xr = flip mask) dotted with x[0..8).
template <int BITS, typename XP>
inline float moe_dot8(device const uchar* w, int k, XP x, int xr, thread float& sx) {
  float q = 0, s = 0;
  if (BITS == 8) {
    const uint2 u = *(device const uint2*)(w + k) ^ uint2(uint(xr) * 0x01010101u);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float a = x[i], b = x[4 + i];
      q += float((u.x >> (8 * i)) & 0xffu) * a + float((u.y >> (8 * i)) & 0xffu) * b;
      s += a + b;
    }
  } else {
    const uint u = *(device const uint*)(w + k / 2) ^ (uint(xr & 15) * 0x11111111u);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float a = x[i];
      q += float((u >> (4 * i)) & 15u) * a;
      s += a;
    }
  }
  sx = s;
  return q;
}

// 16 consecutive affine weights at column k of row w (bytes; BITS 4 or 8, xr = flip mask) dotted with x[0..16).
template <int BITS>
inline float moe_dot16(device const uchar* w, int k, thread const float* x, int xr) {
  float q = 0;
  if (BITS == 8) {
    const uint4 u = *(device const uint4*)(w + k) ^ uint4(uint(xr) * 0x01010101u);
    const uint v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int i = 0; i < 4; i++) q += float((v[j] >> (8 * i)) & 0xffu) * x[4 * j + i];
  } else {
    const uint2 u = *(device const uint2*)(w + k / 2) ^ uint2(uint(xr & 15) * 0x11111111u);
#pragma unroll
    for (int i = 0; i < 8; i++) q += float((u.x >> (4 * i)) & 15u) * x[i] + float((u.y >> (4 * i)) & 15u) * x[8 + i];
  }
  return q;
}
template <typename P>
inline float ld16(P p, thread float* x) {
  float s = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) { x[i] = p[i]; s += x[i]; }
  return s;
}

// Launch 1: router logits from a bf16 matrix W [N, K] (experts + the shared gate row) for T <= 8 rows at once (the
// weights read once): y float [T, N]. Two simdgroups per output row (halves of K). grid: threadgroups ceil(N / 4).
struct RouterArgs { int N, K, T; };
kernel void moe_router16(device const bfloat* x [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                         device float* y [[buffer(2)]], constant RouterArgs& p [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8][8];
  const int n = tg * 4 + sg / 2, hk = sg & 1, K2 = p.K / 2;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (n < p.N) {
    device const bfloat* w = W + (size_t)n * p.K + hk * K2;
    for (int k = lane * 4; k < K2; k += 128) {
      const float w0 = w[k], w1 = w[k + 1], w2 = w[k + 2], w3 = w[k + 3];
      for (int t = 0; t < p.T; t++) {
        device const bfloat* xr = x + (size_t)t * p.K + hk * K2 + k;
        acc[t] += w0 * float(xr[0]) + w1 * float(xr[1]) + w2 * float(xr[2]) + w3 * float(xr[3]);
      }
    }
  }
  for (int t = 0; t < p.T; t++) {
    const float v = simd_sum(acc[t]);
    if (lane == 0) part[sg][t] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (hk == 0 && lane < (uint)p.T && n < p.N) y[(size_t)lane * p.N + n] = part[sg][lane] + part[sg + 1][lane];
}

// Softmax over E logits (fp32) and the top-k in descending order, renormalised: one simdgroup (as moe_route).
inline void moe_topk(device const float* l, int E, int k, uint lane, thread int* idx, thread float* wt) {
  constexpr int MAXV = 16;
  float v[MAXV];
  const int per = (E + 31) / 32;
  float mx = -INFINITY;
  for (int i = 0; i < per; i++) {
    const int c = i * 32 + lane;
    v[i] = c < E ? l[c] : -INFINITY;
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0;
  for (int i = 0; i < per; i++) { v[i] = exp(v[i] - mx); sum += v[i]; }
  sum = simd_sum(sum);
  float tot = 0;
  for (int j = 0; j < k; j++) {
    float best = -1;
    int bi = 0;
    for (int i = 0; i < per; i++)
      if (v[i] > best) { best = v[i]; bi = i * 32 + lane; }
    const float m = simd_max(best);
    const int who = simd_min(best == m ? bi : 1 << 30);
    if ((who & 31) == (int)lane) v[who / 32] = -1;
    idx[j] = who;
    wt[j] = m / sum;
    tot += m / sum;
  }
  for (int j = 0; j < k; j++) wt[j] /= tot;
}

// Launch 2a (moe_plan, one threadgroup): routing of every row (simdgroup t: softmax + top-k of row t from the router
// logits) and the step's unique routed experts in pick order: idx / weights / shared gate per row for launch 3, and
// plan = [nuniq, uexp[T * k], upick[T * k][8]] (upick: row t's pick index j of unique expert u, or -1).
// With dedup = 0 every pick is its own unique slot. T <= 8, k <= 16, E <= 512.
struct MoePlanArgs { int E, k, stride, T, dedup; };
kernel void moe_plan(device const float* rlog [[buffer(0)]], device int* ridx [[buffer(1)]],
                     device float* rw [[buffer(2)]], device float* rsg [[buffer(3)]], device int* plan [[buffer(4)]],
                     constant MoePlanArgs& p [[buffer(5)]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int tidx[8 * 16];
  threadgroup short mark[512];
  const int T = p.T, kk = p.k, NU = T * kk;
  for (int i = tid; i < p.E; i += 256) mark[i] = -1;
  device int* upick = plan + 1 + NU;
  for (int i = tid; i < NU * 8; i += 256) upick[i] = -1;
  if ((int)sg < T) {
    int idx[16];
    float wt[16];
    device const float* l = rlog + (size_t)sg * p.stride;
    moe_topk(l, p.E, kk, lane, idx, wt);
    if (lane == 0) {
      for (int j = 0; j < kk; j++) {
        tidx[sg * kk + j] = idx[j];
        ridx[sg * kk + j] = idx[j];
        rw[sg * kk + j] = wt[j];
      }
      rsg[sg] = 1.f / (1.f + exp(-l[p.E]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (tid != 0) return;
  int c = 0;
  for (int q = 0; q < NU; q++) {
    const int e = tidx[q];
    int u;
    if (!p.dedup) u = c++;
    else if (mark[e] < 0) { mark[e] = short(c); u = c++; }
    else u = mark[e];
    plan[1 + u] = e;
    upick[u * 8 + q / kk] = q % kk;
  }
  plan[0] = c;
}

// Launch 2b: the gate/up SwiGLU of every pick, each unique expert's weights read once for every row that picked it.
// Slots (grid y): [0, T * k) the plan's unique experts (threadgroups past the unique count exit), [T * k, T * k + T)
// the shared expert of row y - T * k. Output columns n = tg.x * 8 + simdgroup: act[(t * (k + 1) + j) * I + n] =
// bf16(silu(bf16(gate))) * bf16(up) (as moe_qmv + moe_silu_mul / gemm + silu_mul; j = k: the shared expert).
// Routed gate|up: MoeW [E][2I][K] (EB bits, unflipped); shared: QLinear [2I][K] (SB bits, xr).
// grid: threadgroups (I / 8, T * k + T), 256 threads.
struct MoeGuArgs { int k, I, K, xr, T; };
template <int EB, int SB, int TM>  // TM: rows the per-row arrays hold (T <= TM; compile-time so they stay in registers)
kernel void moe_gu2(device const bfloat* x [[buffer(0)]], device const int* plan [[buffer(1)]],
                    device const uchar* EW [[buffer(2)]], device const bfloat* ES [[buffer(3)]],
                    device const bfloat* EBi [[buffer(4)]], device const uchar* SW [[buffer(5)]],
                    device const bfloat* SS [[buffer(6)]], device const bfloat* SBi [[buffer(7)]],
                    device bfloat* act [[buffer(8)]], constant MoeGuArgs& p [[buffer(9)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const int u = tg.y, T = p.T, kk = p.k, NU = T * kk, I = p.I, K = p.K, G = K / 64;
  const int n = tg.x * 8 + sg;
  if (n >= I) return;
  if (u >= NU) {  // the shared expert of row t
    const int t = u - NU;
    device const bfloat* xr = x + (size_t)t * K;
    const size_t rg = n, ru = n + I;
    device const uchar* wg = SW + rg * K * SB / 8;
    device const uchar* wu = SW + ru * K * SB / 8;
    float ag = 0, au = 0;
    for (int k = lane * 16; k < K; k += 512) {
      float xv[16];
      const float sx = ld16(xr + k, xv);
      ag += float(SS[rg * G + k / 64]) * moe_dot16<SB>(wg, k, xv, p.xr) + float(SBi[rg * G + k / 64]) * sx;
      au += float(SS[ru * G + k / 64]) * moe_dot16<SB>(wu, k, xv, p.xr) + float(SBi[ru * G + k / 64]) * sx;
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) {
      const float g = float(bfloat(ag)), uu = float(bfloat(au));
      act[((size_t)t * (kk + 1) + kk) * I + n] = bfloat(float(bfloat(g / (1.f + exp(-g)))) * uu);
    }
    return;
  }
  if (u >= plan[0]) return;
  const int e = plan[1 + u];
  device const int* upick = plan + 1 + NU + u * 8;
  const size_t rg = (size_t)e * 2 * I + n, ru = rg + I;
  device const uchar* wg = EW + rg * K * EB / 8;
  device const uchar* wu = EW + ru * K * EB / 8;
  if (TM == 1) {  // one row (decode): no per-row bookkeeping
    const int t = 0, j = upick[0];
    device const bfloat* xr = x + (size_t)t * K;
    float ag = 0, au = 0;
    for (int k = lane * 16; k < K; k += 512) {
      float xv[16];
      const float sx = ld16(xr + k, xv);
      ag += float(ES[rg * G + k / 64]) * moe_dot16<EB>(wg, k, xv, 0) + float(EBi[rg * G + k / 64]) * sx;
      au += float(ES[ru * G + k / 64]) * moe_dot16<EB>(wu, k, xv, 0) + float(EBi[ru * G + k / 64]) * sx;
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) {
      const float g = float(bfloat(ag)), uu = float(bfloat(au));
      act[((size_t)t * (kk + 1) + j) * I + n] = bfloat(float(bfloat(g / (1.f + exp(-g)))) * uu);
    }
    return;
  }
  int pick[TM];
#pragma unroll
  for (int t = 0; t < TM; t++) pick[t] = t < T ? upick[t] : -1;
  float ag[TM], au[TM];
#pragma unroll
  for (int t = 0; t < TM; t++) ag[t] = au[t] = 0;
  for (int k = lane * 16; k < K; k += 512) {
    const float sg_ = ES[rg * G + k / 64], bg_ = EBi[rg * G + k / 64];
    const float su_ = ES[ru * G + k / 64], bu_ = EBi[ru * G + k / 64];
    const uint2 a = *(device const uint2*)(wg + k / 2), b = *(device const uint2*)(wu + k / 2);  // weights once
#pragma unroll
    for (int t = 0; t < TM; t++) {  // every row that picked this expert
      if (pick[t] < 0) continue;
      float xv[16];
      const float sx = ld16(x + (size_t)t * K + k, xv);
      float dg = 0, du = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        dg += float((a.x >> (4 * i)) & 15u) * xv[i] + float((a.y >> (4 * i)) & 15u) * xv[8 + i];
        du += float((b.x >> (4 * i)) & 15u) * xv[i] + float((b.y >> (4 * i)) & 15u) * xv[8 + i];
      }
      ag[t] += sg_ * dg + bg_ * sx;
      au[t] += su_ * du + bu_ * sx;
    }
  }
#pragma unroll
  for (int t = 0; t < TM; t++) {
    if (pick[t] < 0) continue;
    const float g = float(bfloat(simd_sum(ag[t]))), uu = float(bfloat(simd_sum(au[t])));
    if (lane == 0) act[((size_t)t * (kk + 1) + pick[t]) * I + n] = bfloat(float(bfloat(g / (1.f + exp(-g)))) * uu);
  }
}
template [[host_name("moe_gu2_4_8_t1")]] kernel void moe_gu2<4, 8, 1>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_8_t2")]] kernel void moe_gu2<4, 8, 2>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_8_t4")]] kernel void moe_gu2<4, 8, 4>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_8_t8")]] kernel void moe_gu2<4, 8, 8>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_4_t1")]] kernel void moe_gu2<4, 4, 1>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_4_t2")]] kernel void moe_gu2<4, 4, 2>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_4_t4")]] kernel void moe_gu2<4, 4, 4>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);
template [[host_name("moe_gu2_4_4_t8")]] kernel void moe_gu2<4, 4, 8>(device const bfloat*,
    device const int*, device const uchar*, device const bfloat*, device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device bfloat*, constant MoeGuArgs&, uint2, uint, uint);

// Launch 3: out[t, h] = bf16(sum_j w[t, j] * bf16(<down_e(j)[h], act_j>) + sg[t] * bf16(<shared_down[h], act_k>))
// (moe_qmv + moe_combine with assign; x += for !assign). One simdgroup per h; its four 8-lane groups take slots
// j = group, group + 4, ... (j = k: the shared expert), 16 columns per lane per 128-column block (I % 128 == 0).
// Routed down: MoeW [E][H][I] (EB bits); shared down: QLinear [H][I] (SB bits, xr). (One slot after another per
// simdgroup with the activations staged in threadgroup memory was ~2x slower at 4 rows.)
// grid: threadgroups (H / 8, T), 256 threads.
struct MoeDnArgs { int k, I, H, xr, assign; };
template <int EB, int SB>
kernel void moe_dn2(device const bfloat* act [[buffer(0)]], device const int* ridx [[buffer(1)]],
                    device const float* rw [[buffer(2)]], device const float* rsg [[buffer(3)]],
                    device const uchar* EW [[buffer(4)]], device const bfloat* ES [[buffer(5)]],
                    device const bfloat* EBi [[buffer(6)]], device const uchar* SW [[buffer(7)]],
                    device const bfloat* SS [[buffer(8)]], device const bfloat* SBi [[buffer(9)]],
                    device bfloat* out [[buffer(10)]], constant MoeDnArgs& p [[buffer(11)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const int t = tg.y, I = p.I, H = p.H, R = (p.k + 1) * I;
  const int h = tg.x * 8 + sg;
  if (h >= H) return;
  const int G = I / 64, grp = lane / 8, sub = lane % 8;
  device const bfloat* ar = act + (size_t)t * R;
  float acc = 0;
  for (int j = grp; j <= p.k; j += 4) {
    float d = 0;
    device const bfloat* xj = ar + j * I;
    if (j < p.k) {
      const size_t r = (size_t)ridx[t * p.k + j] * H + h;
      device const uchar* w = EW + r * I * EB / 8;
      for (int k = sub * 16; k < I; k += 128) {
        float xv[16];
        const float sx = ld16(xj + k, xv);
        d += float(ES[r * G + k / 64]) * moe_dot16<EB>(w, k, xv, 0) + float(EBi[r * G + k / 64]) * sx;
      }
    } else {
      device const uchar* w = SW + (size_t)h * I * SB / 8;
      for (int k = sub * 16; k < I; k += 128) {
        float xv[16];
        const float sx = ld16(xj + k, xv);
        d += float(SS[(size_t)h * G + k / 64]) * moe_dot16<SB>(w, k, xv, p.xr) + float(SBi[(size_t)h * G + k / 64]) * sx;
      }
    }
    d += simd_shuffle_xor(d, 4);
    d += simd_shuffle_xor(d, 2);
    d += simd_shuffle_xor(d, 1);
    acc += (j < p.k ? rw[t * p.k + j] : rsg[t]) * float(bfloat(d));
  }
  // groups' sums (every lane of a group holds the same value)
  acc += simd_shuffle_xor(acc, 8);
  acc += simd_shuffle_xor(acc, 16);
  if (lane == 0) {
    device bfloat& o = out[(size_t)t * H + h];
    o = p.assign ? bfloat(acc) : bfloat(float(o) + float(bfloat(acc)));
  }
}
template [[host_name("moe_dn2_4_8")]] kernel void moe_dn2<4, 8>(device const bfloat*, device const int*,
    device const float*, device const float*, device const uchar*, device const bfloat*, device const bfloat*,
    device const uchar*, device const bfloat*, device const bfloat*, device bfloat*, constant MoeDnArgs&, uint2, uint,
    uint);
template [[host_name("moe_dn2_4_4")]] kernel void moe_dn2<4, 4>(device const bfloat*, device const int*,
    device const float*, device const float*, device const uchar*, device const bfloat*, device const bfloat*,
    device const uchar*, device const bfloat*, device const bfloat*, device bfloat*, constant MoeDnArgs&, uint2, uint,
    uint);

// Router logits from the 35B-A3B's 8-bit router (MoeW [1][N][K], group 64, unflipped; the shared gate is row N - 1)
// for T <= TM rows, the weights read once: y float [T, N]. One simdgroup per output row, 16 columns per lane per
// 512-column step; each row's sum is formed the same way whatever T (TM only bounds the rows), so verify rows route
// as single-row decode does. grid: threadgroups ceil(N / 8), 256 threads.
template <int TM>
kernel void moe_router8(device const bfloat* x [[buffer(0)]], device const uchar* W [[buffer(1)]],
                        device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                        device float* y [[buffer(4)]], constant RouterArgs& p [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const int n = tg * 8 + sg, K = p.K, G = K / 64;
  if (n >= p.N) return;
  device const uchar* w = W + (size_t)n * K;
  float acc[TM];
#pragma unroll
  for (int t = 0; t < TM; t++) acc[t] = 0;
  for (int k = lane * 16; k < K; k += 512) {
    const float s = S[(size_t)n * G + k / 64], b = B[(size_t)n * G + k / 64];
    const uint4 u = *(device const uint4*)(w + k);
    const uint v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int t = 0; t < TM; t++) {
      if (t >= p.T) break;
      float xv[16];
      const float sx = ld16(x + (size_t)t * K + k, xv);
      float q = 0;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) q += float((v[j] >> (8 * i)) & 0xffu) * xv[4 * j + i];
      acc[t] += s * q + b * sx;
    }
  }
#pragma unroll
  for (int t = 0; t < TM; t++) {
    if (t >= p.T) break;
    const float r = simd_sum(acc[t]);
    if (lane == 0) y[(size_t)t * p.N + n] = r;
  }
}
template [[host_name("moe_router8_t1")]] kernel void moe_router8<1>(device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device float*, constant RouterArgs&, uint, uint, uint);
template [[host_name("moe_router8_t2")]] kernel void moe_router8<2>(device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device float*, constant RouterArgs&, uint, uint, uint);
template [[host_name("moe_router8_t4")]] kernel void moe_router8<4>(device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device float*, constant RouterArgs&, uint, uint, uint);
template [[host_name("moe_router8_t8")]] kernel void moe_router8<8>(device const bfloat*, device const uchar*,
    device const bfloat*, device const bfloat*, device float*, constant RouterArgs&, uint, uint, uint);


