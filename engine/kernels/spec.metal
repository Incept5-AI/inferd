// Speculative decoding (DFlash2 drafter + target verify) kernels.
#include <metal_stdlib>
using namespace metal;

inline float sp_tg_sum(float v, threadgroup float* scratch, uint tid, uint ntg) {
  v = simd_sum(v);
  uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? scratch[lane] : 0.f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// ---------------------------------------------------------------------------------------------
// Target side.

// hcap[t, col0 + i] = x[t, i]: capture a layer's hidden state for the drafter. grid (H, T).
struct CapArgs { int H, stride, col0; };
kernel void capture_cols(device const bfloat* x [[buffer(0)]],
                         device bfloat* hcap [[buffer(1)]],
                         constant CapArgs& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.H) return;
  hcap[(size_t)gid.y * p.stride + p.col0 + gid.x] = x[(size_t)gid.y * p.H + gid.x];
}

// pred[r] = argmax of logits row r (bfloat [R, V]). One threadgroup of 1024 per row.
kernel void row_argmax(device const bfloat* logits [[buffer(0)]],
                       device int* pred [[buffer(1)]],
                       constant int& V [[buffer(2)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[32];
  threadgroup int bi[32];
  device const bfloat* l = logits + (size_t)row * V;
  float best = -INFINITY;
  int idx = 0;
  for (int i = tid; i < V; i += 1024) {
    float v = l[i];
    if (v > best) { best = v; idx = i; }
  }
  for (ushort o = 16; o > 0; o /= 2) {
    float ov = simd_shuffle_down(best, o);
    int oi = simd_shuffle_down(idx, o);
    if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
  }
  if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    best = bv[lane];
    idx = bi[lane];
    for (ushort o = 16; o > 0; o /= 2) {
      float ov = simd_shuffle_down(best, o);
      int oi = simd_shuffle_down(idx, o);
      if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) pred[row] = idx;
  }
}

// Greedy acceptance for each speculative sequence b: its rows are [staged, d1..d_{L-1}] (tokens in tok) and
// pred[out_base[b] + i] is the target's argmax after row i. a = length of the matching draft prefix.
// commit[b] = a + 1 rows are kept; emit[b] = d1..da then pred after row a (the bonus token).
// Non-speculative sequences keep all their rows (commit = seq_len) and emit nothing.
constant constexpr int MAX_BLOCK = 16;  // = kMaxBlock (src/qwen35.h): emit stride per sequence
kernel void spec_accept(device const int* tok [[buffer(0)]],
                        device const int* pred [[buffer(1)]],
                        device const int* seq_start [[buffer(2)]],
                        device const int* seq_len [[buffer(3)]],
                        device const int* out_base [[buffer(4)]],
                        device const int* spec [[buffer(5)]],
                        device int* commit [[buffer(6)]],
                        device int* emit [[buffer(7)]],
                        constant int& S [[buffer(8)]],
                        uint b [[thread_position_in_grid]]) {
  if ((int)b >= S) return;
  int r0 = seq_start[b], L = seq_len[b];
  if (!spec[b]) { commit[b] = L; return; }
  int ob = out_base[b], a = 0;
  while (a < L - 1 && tok[r0 + 1 + a] == pred[ob + a]) a++;
  commit[b] = a + 1;
  for (int i = 0; i < a; i++) emit[b * MAX_BLOCK + i] = tok[r0 + 1 + i];
  emit[b * MAX_BLOCK + a] = pred[ob + a];
}

// Verify steps with more rows than the tapes hold (a prefill chunk beside the verify blocks) run the GDN kernels on
// the ordinary scratch; this copies rows [0, n) (the verify blocks, which the scheduler puts first) into the layer's
// tapes: conv inputs, normalised conv outputs, g and beta. grid (conv_dim, n).
struct TapeArgs { int n, conv_dim, hv; };
kernel void tape_save(device const bfloat* tx [[buffer(0)]], device const bfloat* co [[buffer(1)]],
                      device const float* gb [[buffer(2)]], device const float* bb [[buffer(3)]],
                      device bfloat* tape_x [[buffer(4)]], device bfloat* tape_c [[buffer(5)]],
                      device float* tape_g [[buffer(6)]], device float* tape_b [[buffer(7)]],
                      constant TapeArgs& p [[buffer(8)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, t = gid.y;
  if (t >= p.n || c >= p.conv_dim) return;
  const size_t i = (size_t)t * p.conv_dim + c;
  tape_x[i] = tx[i];
  tape_c[i] = co[i];
  if (c < p.hv) {
    const size_t k = (size_t)t * p.hv + c;
    tape_g[k] = gb[k];
    tape_b[k] = bb[k];
  }
}

// Replay the conv state of speculative sequences over their committed rows only (the verify step left the
// state untouched for them and saved its inputs in tape_x [T, conv_dim]). grid (conv_dim, S).
struct ReplayArgs { int conv_dim, Hk, Hv, Dk, Dv; };
kernel void gdn_conv_replay(device const bfloat* tape_x [[buffer(0)]],
                            device bfloat* conv_state [[buffer(1)]],
                            device const int* slot [[buffer(2)]],
                            device const int* seq_start [[buffer(3)]],
                            device const int* commit [[buffer(4)]],
                            device const int* spec [[buffer(5)]],
                            constant ReplayArgs& p [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
  int c = gid.x, b = gid.y;
  if (c >= p.conv_dim || !spec[b]) return;
  device bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  float s0 = st[c], s1 = st[p.conv_dim + c], s2 = st[2 * p.conv_dim + c];
  int t0 = seq_start[b], n = commit[b];
  for (int j = 0; j < n; j++) {
    s0 = s1; s1 = s2;
    s2 = tape_x[(size_t)(t0 + j) * p.conv_dim + c];
  }
  st[c] = bfloat(s0);
  st[p.conv_dim + c] = bfloat(s1);
  st[2 * p.conv_dim + c] = bfloat(s2);
}

// Replay the recurrence of speculative sequences over their committed rows, from the per-layer tape of
// normalised q/k/v (conv_out), g and beta. Same arithmetic as gdn_step. grid (32, Dv, S*Hv), tg (32, 4, 1).
kernel void gdn_replay(device const bfloat* conv_out [[buffer(0)]],
                       device const float* g [[buffer(1)]],
                       device const float* beta [[buffer(2)]],
                       device float* state [[buffer(3)]],
                       device const int* slot [[buffer(4)]],
                       device const int* seq_start [[buffer(5)]],
                       device const int* commit [[buffer(6)]],
                       device const int* spec [[buffer(7)]],
                       constant ReplayArgs& p [[buffer(8)]],
                       uint3 gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const int n = gid.z, b = n / p.Hv, hv = n % p.Hv, hk = hv / (p.Hv / p.Hk);
  if (!spec[b]) return;
  const int dv = gid.y;
  constexpr int NPT = 4;
  device float* st = state + (((size_t)slot[b] * p.Hv + hv) * p.Dv + dv) * p.Dk;
  float s[NPT];
  for (int i = 0; i < NPT; i++) s[i] = st[NPT * lane + i];
  int t0 = seq_start[b], nt = commit[b];
  for (int j = 0; j < nt; j++) {
    int t = t0 + j;
    device const bfloat* row = conv_out + (size_t)t * p.conv_dim;
    device const bfloat* k = row + p.Hk * p.Dk + hk * p.Dk;
    float v = row[2 * p.Hk * p.Dk + hv * p.Dv + dv];
    float gg = g[t * p.Hv + hv], bt = beta[t * p.Hv + hv];
    float kk[NPT], kv = 0;
    for (int i = 0; i < NPT; i++) {
      kk[i] = k[NPT * lane + i];
      s[i] *= gg;
      kv += s[i] * kk[i];
    }
    kv = simd_sum(kv);
    float delta = (v - kv) * bt;
    for (int i = 0; i < NPT; i++) s[i] += kk[i] * delta;
  }
  for (int i = 0; i < NPT; i++) st[NPT * lane + i] = s[i];
}

// dst[i, :] = src[rows[i], :]. grid (cols, n).
struct GatherArgs { int cols; };
kernel void gather_rows(device const bfloat* src [[buffer(0)]],
                        device const int* rows [[buffer(1)]],
                        device bfloat* dst [[buffer(2)]],
                        constant GatherArgs& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.cols) return;
  dst[(size_t)gid.y * p.cols + gid.x] = src[(size_t)rows[gid.y] * p.cols + gid.x];
}

// ---------------------------------------------------------------------------------------------
// Drafter (DFlash2): 5 Qwen3-style layers, non-causal within the block, attending to a sliding window of
// projected target hidden states. Rows are grouped per sequence (seq_start/seq_len); j = row - seq_start.

// Grouped dynamic causal conv over the block (kernel 2): for channel i in group gi,
//   out[t, i] = res[t, i] + sum_{o in 0,1} (base[o, i] + kp[t, half*2*G + o*G + gi]) * in[t - o, i]
// where t - o must stay inside the same block (else 0) and res is optional. grid (H, rows).
struct DynConvArgs { int H, G, group, half_, kp_stride, has_res; };
kernel void dyn_conv(device const bfloat* in [[buffer(0)]],
                     device const bfloat* kp [[buffer(1)]],
                     device const bfloat* base [[buffer(2)]],
                     device bfloat* out [[buffer(3)]],
                     device const bfloat* res [[buffer(4)]],
                     device const int* tseq [[buffer(5)]],
                     device const int* seq_start [[buffer(6)]],
                     constant DynConvArgs& p [[buffer(7)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int i = gid.x, t = gid.y;
  if (i >= p.H) return;
  int gi = i / p.group, j = t - seq_start[tseq[t]];
  device const bfloat* k = kp + (size_t)t * p.kp_stride + p.half_ * 2 * p.G;
  float y = (float(base[i]) + float(k[gi])) * float(in[(size_t)t * p.H + i]);
  if (j > 0) y += (float(base[p.H + i]) + float(k[p.G + gi])) * float(in[(size_t)(t - 1) * p.H + i]);
  y = float(bfloat(y));
  if (p.has_res) y += float(res[(size_t)t * p.H + i]);
  out[(size_t)t * p.H + i] = bfloat(y);
}

// q/k RMSNorm + RoPE (full head dim, non-traditional halves), then write k/v into the drafter's per-slot
// ring cache at pos % RING; q (if wanted) to q_out [rows, Hq, D]. The input row is [q (Hq*D) | k | v]
// (with_q) or [k | v]. grid: threadgroups (Hq + Hkv, rows) (Hkv only when !with_q), D threads.
struct DqkArgs { int Hq, Hkv, D, ring, in_stride, with_q; float eps, theta; };
kernel void draft_qk(device const bfloat* in [[buffer(0)]],
                     device const bfloat* qn_w [[buffer(1)]],
                     device const bfloat* kn_w [[buffer(2)]],
                     device bfloat* q_out [[buffer(3)]],
                     device bfloat* Kr [[buffer(4)]],
                     device bfloat* Vr [[buffer(5)]],
                     device const int* pos [[buffer(6)]],
                     device const int* tseq [[buffer(7)]],
                     device const int* slot [[buffer(8)]],
                     constant DqkArgs& p [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint2 ntg2 [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float xn[256];
  const uint ntg = ntg2.x;
  int h = tg.x, t = tg.y;
  int nq = p.with_q ? p.Hq : 0;
  bool isq = h < nq;
  int kh = h - nq;
  device const bfloat* row = in + (size_t)t * p.in_stride;
  device const bfloat* src = isq ? row + h * p.D : row + nq * p.D + kh * p.D;
  float x = src[tid];
  float r = rsqrt(sp_tg_sum(x * x, scratch, tid, ntg) / p.D + p.eps);
  xn[tid] = float(bfloat(x * r * float((isq ? qn_w : kn_w)[tid])));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int ps = pos[t], half_d = p.D / 2, i = tid % half_d;
  float ang = ps * pow(p.theta, -2.f * i / p.D), c = cos(ang), s = sin(ang);
  float val = (int)tid < half_d ? xn[tid] * c - xn[tid + half_d] * s : xn[tid] * c + xn[tid - half_d] * s;
  if (isq) {
    q_out[((size_t)t * p.Hq + h) * p.D + tid] = bfloat(val);
  } else {
    size_t o = (((size_t)slot[tseq[t]] * p.ring + ps % p.ring) * p.Hkv + kh) * p.D + tid;
    Kr[o] = bfloat(val);
    Vr[o] = row[nq * p.D + p.Hkv * p.D + kh * p.D + tid];
  }
}

// Drafter attention: each block row (query position q0 + j) attends to ring keys with
// key >= q0 (the block itself, non-causal) or q0 + j - key < window (context). One simdgroup per
// (kv head, sequence, split of key positions) handles all rows x GQ heads together, lanes over keys.
// Writes partials [S, Hkv, splits, R*GQ] x {acc[D], m, l}; draft_attn_combine merges them.
// seg: rows j < seg attend only to the block's first seg keys, so a block wider than the drafter's trained size (8)
// leaves the first seg drafts exactly as at block seg; rows >= seg see the whole block (seg >= block: plain DFlash2).
struct DattnArgs { int Hq, Hkv, D, ring, window, splits, chunk; float scale; int seg; };
// DMAXR rows per block: 5, 8 (DFlash2's trained block size), 12 or 16 (wide blocks for one or a few streams). The
// per-thread accumulators scale with RG, so each size gets its own instance; 12 and 16 only as groups of 4 rows.
constant constexpr int DGQ = 4, DD = 128;
template <int DMAXR, int RG = DMAXR>
kernel void draft_attn(device const bfloat* q [[buffer(0)]],
                       device const bfloat* Kr [[buffer(1)]],
                       device const bfloat* Vr [[buffer(2)]],
                       device const int* pos [[buffer(3)]],
                       device const int* seq_start [[buffer(4)]],
                       device const int* seq_len [[buffer(5)]],
                       device const int* slot [[buffer(6)]],
                       device float* part [[buffer(7)]],
                       constant DattnArgs& p [[buffer(8)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  // RG < DMAXR: the block's rows are split into DMAXR / RG groups of RG rows across grid z (z = split * groups +
  // group), so each simdgroup keeps RG rows of accumulators (8 rows in one simdgroup spills).
  constexpr int NQ = RG * DGQ, NQT = DMAXR * DGQ, DPL = DD / 32;
  threadgroup float qs[NQ][DD];
  const int kh = tg.x, b = tg.y, z = tg.z / (DMAXR / RG), rg = tg.z % (DMAXR / RG);
  const int R = max(0, min(seq_len[b], DMAXR) - rg * RG), r0 = seq_start[b] + rg * RG, q0 = pos[seq_start[b]];
  const int hi = q0 + min(seq_len[b], DMAXR), lo = max(0, q0 - p.window + 1);
  const int p0 = lo + z * p.chunk, p1 = min(hi, p0 + p.chunk);
  for (int i = lane; i < NQ * DD; i += 32) {
    int v = i / DD, r = v / DGQ, hh = v % DGQ;
    qs[v][i % DD] = r < R ? float(q[((size_t)(r0 + r) * p.Hq + kh * DGQ + hh) * DD + i % DD]) * p.scale : 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[NQ][DPL], m[NQ], l[NQ];
  for (int v = 0; v < NQ; v++) {
    m[v] = -INFINITY;
    l[v] = 0;
    for (int i = 0; i < DPL; i++) acc[v][i] = 0;
  }
  const size_t rs = (size_t)p.Hkv * DD;
  device const bfloat* kbase = Kr + ((size_t)slot[b] * p.ring * p.Hkv + kh) * DD;
  device const bfloat* vbase = Vr + ((size_t)slot[b] * p.ring * p.Hkv + kh) * DD;
  for (int base = p0; base < p1; base += 32) {
    int key = base + lane;
    bool inr = key < p1;
    float sc[NQ];
    for (int v = 0; v < NQ; v++) sc[v] = 0;
    if (inr) {
      device const bfloat4* kr = (device const bfloat4*)(kbase + (size_t)(key % p.ring) * rs);
      for (int d4 = 0; d4 < DD / 4; d4++) {
        float4 k4 = float4(kr[d4]);
        for (int v = 0; v < NQ; v++)
          sc[v] += k4.x * qs[v][4 * d4] + k4.y * qs[v][4 * d4 + 1] + k4.z * qs[v][4 * d4 + 2] + k4.w * qs[v][4 * d4 + 3];
      }
    }
    float e[NQ];
    for (int v = 0; v < NQ; v++) {
      int qp = q0 + rg * RG + v / DGQ;
      bool ok = inr && v / DGQ < R && (key >= q0 ? (qp - q0 >= p.seg || key < q0 + p.seg) : qp - key < p.window);
      float s = ok ? sc[v] : -INFINITY;
      float mx = simd_max(s);
      float mn = max(m[v], mx);
      float corr = mn == -INFINITY ? 1.f : exp(m[v] - mn);
      e[v] = ok ? exp(s - mn) : 0.f;
      l[v] = l[v] * corr + simd_sum(e[v]);
      for (int i = 0; i < DPL; i++) acc[v][i] *= corr;
      m[v] = mn;
    }
    int n = min(32, p1 - base);
    for (int r = 0; r < n; r++) {
      device const bfloat4* vr = (device const bfloat4*)(vbase + (size_t)((base + r) % p.ring) * rs + lane * DPL);
      float4 v4 = float4(vr[0]);
      for (int v = 0; v < NQ; v++) {
        float w = simd_shuffle(e[v], (ushort)r);
        acc[v][0] += w * v4.x; acc[v][1] += w * v4.y; acc[v][2] += w * v4.z; acc[v][3] += w * v4.w;
      }
    }
  }
  const int stride = DD + 2;
  for (int v = 0; v < NQ; v++) {
    device float* out = part + ((((size_t)b * p.Hkv + kh) * p.splits + z) * NQT + rg * NQ + v) * stride;
    for (int i = 0; i < DPL; i++) out[lane * DPL + i] = acc[v][i];
    if (lane == 0) { out[DD] = m[v]; out[DD + 1] = l[v]; }
  }
}

// out[row, h*D + d] = merged attention. grid: threadgroups (Hq, rows), D threads.
template <int DMAXR>
kernel void draft_attn_combine(device const float* part [[buffer(0)]],
                               device bfloat* out [[buffer(1)]],
                               device const int* tseq [[buffer(2)]],
                               device const int* seq_start [[buffer(3)]],
                               constant DattnArgs& p [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  constexpr int NQ = DMAXR * DGQ;
  const int h = tg.x, t = tg.y, b = tseq[t], r = t - seq_start[b];
  if (r >= DMAXR) return;
  const int kh = h / DGQ, v = r * DGQ + h % DGQ, stride = DD + 2;
  float M = -INFINITY;
  for (int z = 0; z < p.splits; z++)
    M = max(M, part[((((size_t)b * p.Hkv + kh) * p.splits + z) * NQ + v) * stride + DD]);
  float num = 0, den = 0;
  for (int z = 0; z < p.splits; z++) {
    device const float* pp = part + ((((size_t)b * p.Hkv + kh) * p.splits + z) * NQ + v) * stride;
    if (pp[DD] == -INFINITY) continue;
    float f = exp(pp[DD] - M);
    num += pp[tid] * f;
    den += pp[DD + 1] * f;
  }
  out[((size_t)t * p.Hq + h) * DD + tid] = bfloat(num / den);
}

#define DRAFT_ATTN(R)                                                                                          \
  template [[host_name("draft_attn" #R)]] kernel void draft_attn<R>(device const bfloat*, device const bfloat*,      \
      device const bfloat*, device const int*, device const int*, device const int*, device const int*, device float*, \
      constant DattnArgs&, uint3, uint);                                                                           \
  template [[host_name("draft_attn_combine" #R)]] kernel void draft_attn_combine<R>(device const float*,             \
      device bfloat*, device const int*, device const int*, constant DattnArgs&, uint2, uint);
DRAFT_ATTN(5) DRAFT_ATTN(8)
#define DRAFT_ATTN_G(R, NAME)                                                                                  \
  template [[host_name(NAME)]] kernel void draft_attn<R, 4>(device const bfloat*, device const bfloat*,             \
      device const bfloat*, device const int*, device const int*, device const int*, device const int*, device float*, \
      constant DattnArgs&, uint3, uint);
DRAFT_ATTN_G(8, "draft_attn8g") DRAFT_ATTN_G(12, "draft_attn12g") DRAFT_ATTN_G(16, "draft_attn16g")
template [[host_name("draft_attn_combine12")]] kernel void draft_attn_combine<12>(device const float*, device bfloat*,
    device const int*, device const int*, constant DattnArgs&, uint2, uint);
template [[host_name("draft_attn_combine16")]] kernel void draft_attn_combine<16>(device const float*, device bfloat*,
    device const int*, device const int*, constant DattnArgs&, uint2, uint);

// Top-16 of each logits row: candidate ids and their logits. One threadgroup of 256 per row: each thread keeps
// a sorted local top-16, then 16 rounds of a threadgroup argmax pop the global top-16.
constant constexpr int TOPK = 16;
kernel void row_top16(device const bfloat* logits [[buffer(0)]],
                      device int* cand [[buffer(1)]],
                      device float* unary [[buffer(2)]],
                      constant int& V [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[8];
  threadgroup int bt[8];
  threadgroup int winner;
  device const bfloat* l = logits + (size_t)row * V;
  float tv[TOPK];
  int ti[TOPK];
  for (int k = 0; k < TOPK; k++) { tv[k] = -INFINITY; ti[k] = 0; }
  for (int i = tid; i < V; i += 256) {
    float v = l[i];
    if (v <= tv[TOPK - 1]) continue;
    int k = TOPK - 1;
    while (k > 0 && tv[k - 1] < v) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
    tv[k] = v;
    ti[k] = i;
  }
  int head = 0;
  for (int out = 0; out < TOPK; out++) {
    float best = head < TOPK ? tv[head] : -INFINITY;
    int who = tid;
    for (ushort o = 16; o > 0; o /= 2) {
      float ov = simd_shuffle_down(best, o);
      int ow = simd_shuffle_down(who, o);
      if (ov > best) { best = ov; who = ow; }
    }
    if (lane == 0) { bv[sg] = best; bt[sg] = who; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float b = bv[0];
      int w = bt[0];
      for (int s = 1; s < 8; s++) if (bv[s] > b) { b = bv[s]; w = bt[s]; }
      winner = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid == winner) {
      cand[row * TOPK + out] = ti[head];
      unary[row * TOPK + out] = tv[head];
      head++;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Sharded top-16 (the draft head's vocabulary is 248k wide; one threadgroup per row left ~8 threadgroups busy):
// stage 1 grid (SH, rows): shard sh keeps the top-16 of its vocabulary slice in cpart/vpart[(row * SH + sh) * 16];
// stage 2 (top16_merge) grid (rows): the top-16 of the SH * 16 survivors, as row_top16 writes them.
template <typename Load>
inline void top16_pop(Load load, int n, uint tid, uint lane, uint sg, device int* cand, device float* unary, int row,
                      threadgroup float* bv, threadgroup int* bt, threadgroup int& winner) {
  float tv[TOPK];
  int ti[TOPK];
  for (int k = 0; k < TOPK; k++) { tv[k] = -INFINITY; ti[k] = 0; }
  for (int i = tid; i < n; i += 256) {
    float v;
    int id;
    load(i, v, id);
    if (v <= tv[TOPK - 1]) continue;
    int k = TOPK - 1;
    while (k > 0 && tv[k - 1] < v) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
    tv[k] = v;
    ti[k] = id;
  }
  int head = 0;
  for (int out = 0; out < TOPK; out++) {
    float best = head < TOPK ? tv[head] : -INFINITY;
    int who = tid;
    for (ushort o = 16; o > 0; o /= 2) {
      float ov = simd_shuffle_down(best, o);
      int ow = simd_shuffle_down(who, o);
      if (ov > best) { best = ov; who = ow; }
    }
    if (lane == 0) { bv[sg] = best; bt[sg] = who; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float b = bv[0];
      int w = bt[0];
      for (int s = 1; s < 8; s++) if (bv[s] > b) { b = bv[s]; w = bt[s]; }
      winner = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid == winner) {
      cand[row * TOPK + out] = ti[head];
      unary[row * TOPK + out] = tv[head];
      head++;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
// V columns per row (also the row stride), split into SH shards; partitions PS per row, this range's first P0;
// candidate ids offset by LO (a sub-range of the vocabulary, e.g. the drafter's special-token tail).
struct Top16Args { int V, SH, PS, P0, LO; };
kernel void row_top16_shard(device const bfloat* logits [[buffer(0)]], device int* cpart [[buffer(1)]],
                            device float* vpart [[buffer(2)]], constant Top16Args& p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[8];
  threadgroup int bt[8];
  threadgroup int winner;
  const int sh = tg.x, row = tg.y, chunk = (p.V + p.SH - 1) / p.SH, lo = sh * chunk, n = max(0, min(p.V, lo + chunk) - lo);
  device const bfloat* l = logits + (size_t)row * p.V + lo;
  top16_pop([&](int i, thread float& v, thread int& id) { v = l[i]; id = p.LO + lo + i; }, n, tid, lane, sg, cpart,
            vpart, row * p.PS + p.P0 + sh, bv, bt, winner);
}
kernel void top16_merge(device const int* cpart [[buffer(0)]], device const float* vpart [[buffer(1)]],
                        device int* cand [[buffer(2)]], device float* unary [[buffer(3)]],
                        constant Top16Args& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[8];
  threadgroup int bt[8];
  threadgroup int winner;
  const size_t o = (size_t)row * p.PS * TOPK;
  top16_pop([&](int i, thread float& v, thread int& id) { v = vpart[o + i]; id = cpart[o + i]; }, p.PS * TOPK, tid, lane,
            sg, cand, unary, row, bv, bt, winner);
}

// DFlash2 candidate selector, greedy: walk the block's draft positions p = 1..R-1; the score of candidate c is
// unary[p][c] + sum_r pred_cb[prev][r] * hproj[p][r] * succ_cb[cand[p][c]][r]; take the argmax, and write it as
// the target's verify token. prev starts at the anchor (the block's first token). One threadgroup of 256
// (= selector rank) per sequence.
kernel void draft_select(device const int* cand [[buffer(0)]],
                         device const float* unary [[buffer(1)]],
                         device const bfloat* hproj [[buffer(2)]],
                         device const bfloat* pred_cb [[buffer(3)]],
                         device const bfloat* succ_cb [[buffer(4)]],
                         device const int* seq_start [[buffer(5)]],
                         device const int* seq_len [[buffer(6)]],
                         device const int* spec [[buffer(7)]],
                         device const int* dtok [[buffer(8)]],
                         device int* vtok [[buffer(9)]],
                         device const int* vstart [[buffer(10)]],
                         uint b [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup int chosen;
  if (!spec[b]) return;
  const int r0 = seq_start[b], R = min(seq_len[b], spec[b]), RANK = 256;  // spec[b]: verify rows to fill (<= block)
  int prev = dtok[r0];
  for (int pp = 1; pp < R; pp++) {
    int row = r0 + pp;
    float ph = float(pred_cb[(size_t)prev * RANK + tid]) * float(hproj[(size_t)row * RANK + tid]);
    float best = -INFINITY;
    int bc = 0;
    for (int c = 0; c < TOPK; c++) {
      int id = cand[row * TOPK + c];
      float s = sp_tg_sum(ph * float(succ_cb[(size_t)id * RANK + tid]), scratch, tid, 256) + unary[row * TOPK + c];
      if (s > best) { best = s; bc = id; }
    }
    if (tid == 0) chosen = bc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prev = chosen;
    if (tid == 0) vtok[vstart[b] + pp] = prev;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Speculative sampling (temperature > 0), 2026-10-31. The drafter's proposal is deterministic (a point mass on d), so
// the exact rule is: accept d with the target's probability p(d) (after temperature, top-k, top-p), otherwise emit a
// sample from p with d removed, renormalised; if every draft is accepted, sample the bonus from the last row. p comes
// from each verify row's top-16 logits (vcand/vun, descending), so top-k is capped at 16; top-p normally cuts well
// inside that. Greedy sequences (t <= 0) take the argmax path exactly as spec_accept. samp[b] = {t, top_p, top_k, seed}.
struct SampParams { float t, top_p; int top_k; uint seed; };
inline float u01(uint seed, uint i) {
  uint x = seed + 0x9E3779B9u * (i + 1);
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return (float)(x >> 8) * (1.0f / 16777216.0f);
}
// Row weights w[0..n) (unnormalised, descending) after temperature, top-k and top-p; returns n.
inline int row_dist(device const float* lv, SampParams sp, thread float* w, thread float& Z) {
  const int k = sp.top_k > 0 ? min(sp.top_k, TOPK) : TOPK;
  float s = 0;
  for (int j = 0; j < k; j++) { w[j] = exp((lv[j] - lv[0]) / sp.t); s += w[j]; }
  const float cut = sp.top_p * s;
  int n = 0;
  Z = 0;
  while (n < k) { Z += w[n++]; if (Z >= cut) break; }
  return n;
}
kernel void spec_accept_s(device const int* tok [[buffer(0)]],
                          device const int* pred [[buffer(1)]],
                          device const int* seq_start [[buffer(2)]],
                          device const int* seq_len [[buffer(3)]],
                          device const int* out_base [[buffer(4)]],
                          device const int* spec [[buffer(5)]],
                          device int* commit [[buffer(6)]],
                          device int* emit [[buffer(7)]],
                          constant int& S [[buffer(8)]],
                          device const SampParams* samp [[buffer(9)]],
                          device const int* vcand [[buffer(10)]],
                          device const float* vun [[buffer(11)]],
                          uint b [[thread_position_in_grid]]) {
  if ((int)b >= S) return;
  int r0 = seq_start[b], L = seq_len[b];
  if (!spec[b]) { commit[b] = L; return; }
  int ob = out_base[b], a = 0;
  const SampParams sp = samp[b];
  if (sp.t <= 0) {
    while (a < L - 1 && tok[r0 + 1 + a] == pred[ob + a]) a++;
    commit[b] = a + 1;
    for (int i = 0; i < a; i++) emit[b * MAX_BLOCK + i] = tok[r0 + 1 + i];
    emit[b * MAX_BLOCK + a] = pred[ob + a];
    return;
  }
  float w[TOPK], Z;
  for (int i = 0; i < L; i++) {
    device const int* ids = vcand + (size_t)(ob + i) * TOPK;
    const int n = row_dist(vun + (size_t)(ob + i) * TOPK, sp, w, Z);
    if (i < L - 1) {
      const int d = tok[r0 + 1 + i];
      int jd = -1;
      for (int j = 0; j < n; j++) if (ids[j] == d) jd = j;
      const float pd = jd >= 0 ? w[jd] / Z : 0.0f;
      if (u01(sp.seed, 2 * i) < pd) { emit[b * MAX_BLOCK + i] = d; continue; }  // accepted
      // Rejected: sample from p without d.
      const float Zr = Z - (jd >= 0 ? w[jd] : 0.0f);
      int pick = jd >= 0 ? jd : 0;
      if (Zr > 0) {
        float r = u01(sp.seed, 2 * i + 1) * Zr;
        pick = -1;
        for (int j = 0; j < n && pick < 0; j++) {
          if (j == jd) continue;
          if ((r -= w[j]) <= 0) pick = j;
        }
        if (pick < 0) for (int j = n - 1; j >= 0 && pick < 0; j--) if (j != jd) pick = j;
      }
      emit[b * MAX_BLOCK + i] = ids[pick];
      commit[b] = i + 1;
      return;
    }
    // Every draft accepted: the bonus token from the last row.
    float r = u01(sp.seed, 2 * i + 1) * Z;
    int pick = n - 1;
    for (int j = 0; j < n; j++) if ((r -= w[j]) <= 0) { pick = j; break; }
    emit[b * MAX_BLOCK + i] = ids[pick];
    commit[b] = L;
  }
}
