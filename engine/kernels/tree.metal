// Token-tree verification (TREE=<nodes>, src/tree.h): a speculative sequence's verify rows are the nodes of a draft
// tree in DFS preorder rather than a chain. Per row t of the step: trow[t] = its node index within its tree (-1: not
// a tree row), tpar[t] = its parent's node index (-1 for the root), tdep[t] = its depth (| 0x100: 2+ children,
// | 0x200: the parent is not the previous row; set_tree), tanc[t] = its ancestor mask
// (bit j: node j is the row itself or one of its ancestors). Row i of a tree occupies KV slot pos0 + i but is
// rotated as position pos0 + depth; the main attention kernels see only the committed context (their position
// buffer says pos0 - 1 for every tree row), attn_tree adds the ancestors, attn_combine_tree merges the two.
// After acceptance the accepted path's KV moves into slots pos0 .. pos0 + a (kv_compact) and the GatedDeltaNet
// state is replayed along the path (the *_tree replays).
#include <metal_stdlib>
using namespace metal;

struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles, own_splits, ns, win, gated; };
struct GdnArgs { int S, Hk, Hv, Dk, Dv, conv_dim, proj_stride; float eps; int verify, sig; };
struct ReplayArgs { int conv_dim, Hk, Hv, Dk, Dv; };
constant constexpr int BLK = 32, MAX_BLOCK = 16;
// KV4 (core.metal): two signed nibbles a byte, even element low; 23 = rotated V (the output is rotated back here).
constant bool kv4_rv_fc [[function_constant(23)]];
constant bool tr_kv4_rv = is_function_constant_defined(kv4_rv_fc) && kv4_rv_fc;
template <bool KV8, bool KV4>
inline float tr_kv(device const void* base, size_t off) {
  if (KV4) {
    const int b = int(((device const uchar*)base)[off >> 1]);
    return float((off & 1) ? (b << 24) >> 28 : (b << 28) >> 28);
  }
  if (KV8) return float(((device const char*)base)[off]);
  return float(((device const bfloat*)base)[off]);
}
inline float tr_hadamard(threadgroup float* x, float v, uint tid, int n) {
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

inline float tr_tg_sum(float v, threadgroup float* scratch, uint tid, uint ntg) {
  v = simd_sum(v);
  uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? scratch[lane] : 0.f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// The tree part of attention: row t, query head h over its ancestors' (and its own) keys. Writes one partial
// (numerator [Dh], max, denominator) per (row, head) in the main partials' format at tpart[tpi[t]]. One simdgroup per
// (head, row), Dh / 32 dims a lane. grid: threadgroups (Hq, ceil(T / 4)), 128 threads (4 rows).
template <bool KV8, bool KV4 = false>
kernel void attn_tree(device const bfloat* q [[buffer(0)]], device const void* Kc [[buffer(1)]],
                      device const void* Vc [[buffer(2)]], device const int* block_table [[buffer(3)]],
                      device const int* pos [[buffer(4)]], device const int* tseq [[buffer(5)]],
                      device float* tpart [[buffer(6)]], constant GqaArgs& p [[buffer(7)]],
                      device const int* trow [[buffer(8)]], device const uint* tanc [[buffer(9)]],
                      device const float* Ks [[buffer(10)]], device const float* Vs [[buffer(11)]],
                      device const int* tpi [[buffer(12)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const int h = tg.x, t = tg.y * 4 + sg;
  if (t >= p.T) return;
  const int i = trow[t];
  if (i < 0) return;
  const int t0 = t - i, kh = h / (p.Hq / p.Hkv), DL = p.Dh / 32;  // dims per lane (8 for Dh 256)
  float qd[16];
  device const bfloat* qr = q + ((size_t)t * p.Hq + h) * p.Dh + lane * DL;
  for (int k = 0; k < DL; k++) qd[k] = float(qr[k]);
  float M = -INFINITY, den = 0, num[16];
  for (int k = 0; k < DL; k++) num[k] = 0;
  for (uint mm = tanc[t]; mm; mm &= mm - 1) {  // the ancestors (and the row itself), in order
    const int j = ctz(mm);
    const int ps = pos[t0 + j];
    const int blk = block_table[tseq[t] * p.max_blocks + ps / BLK];
    const size_t si = (size_t)kh * p.ns + (size_t)blk * BLK + ps % BLK, o = si * p.Dh + lane * DL;
    float dot = 0;
    if (KV8) {
      const float ksc = Ks[si];
      for (int k = 0; k < DL; k++) dot += qd[k] * tr_kv<KV8, KV4>(Kc, o + k);
      dot *= ksc;
    } else {
      for (int k = 0; k < DL; k++) dot += qd[k] * float(((device const bfloat*)Kc)[o + k]);
    }
    const float s = simd_sum(dot) * p.scale;
    const float Mn = max(M, s), f = exp(M - Mn), e = exp(s - Mn);
    const float vsc = KV8 ? Vs[si] : 1.f;
    for (int k = 0; k < DL; k++) {
      const float v = tr_kv<KV8, KV4>(Vc, o + k) * vsc;
      num[k] = num[k] * f + e * v;
    }
    den = den * f + e;
    M = Mn;
  }
  device float* out = tpart + ((size_t)tpi[t] * p.Hq + h) * (p.Dh + 2);
  for (int k = 0; k < DL; k++) out[lane * DL + k] = num[k];
  if (lane == 0) {
    out[p.Dh] = M;
    out[p.Dh + 1] = den;
  }
}
template [[host_name("attn_tree")]] kernel void attn_tree<false>(device const bfloat*, device const void*, device const void*,
    device const int*, device const int*, device const int*, device float*, constant GqaArgs&, device const int*,
    device const uint*, device const float*, device const float*, device const int*, uint2, uint, uint);
template [[host_name("attn_tree_kv4")]] kernel void attn_tree<true, true>(device const bfloat*, device const void*, device const void*,
    device const int*, device const int*, device const int*, device float*, constant GqaArgs&, device const int*,
    device const uint*, device const float*, device const float*, device const int*, uint2, uint, uint);
template [[host_name("attn_tree_kv8")]] kernel void attn_tree<true>(device const bfloat*, device const void*, device const void*,
    device const int*, device const int*, device const int*, device float*, constant GqaArgs&, device const int*,
    device const uint*, device const float*, device const float*, device const int*, uint2, uint, uint);

// attn_combine plus, for tree rows, the attn_tree partial. grid: threadgroups (Hq, T), Dh threads.
kernel void attn_combine_tree(device const float* part [[buffer(0)]], device const bfloat* proj [[buffer(1)]],
                              device bfloat* out [[buffer(2)]], constant GqaArgs& p [[buffer(3)]],
                              device const int* rflag [[buffer(4)]], device const int* shrow [[buffer(5)]],
                              device const float* tpart [[buffer(6)]], device const int* trow [[buffer(7)]],
                              device const int* tpi [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const int h = tg.x, t = tg.y, stride = p.Dh + 2;
  if (rflag[t] == 1) return;
  device const float* pp = part + ((size_t)t * p.Hq + h) * p.splits * stride;
  const int nz = p.own_splits > 0 && !shrow[t] ? p.own_splits : p.splits;
  device const float* tp = trow[t] >= 0 ? tpart + ((size_t)tpi[t] * p.Hq + h) * stride : nullptr;
  float M = -INFINITY;
  for (int z = 0; z < nz; z++) M = max(M, pp[z * stride + p.Dh]);
  if (tp) M = max(M, tp[p.Dh]);
  float num = 0, den = 0;
  for (int z = 0; z < nz; z++) {
    float mz = pp[z * stride + p.Dh];
    if (mz == -INFINITY) continue;
    float f = exp(mz - M);
    num += pp[z * stride + tid] * f;
    den += pp[z * stride + p.Dh + 1] * f;
  }
  if (tp && tp[p.Dh] != -INFINITY) {
    float f = exp(tp[p.Dh] - M);
    num += tp[tid] * f;
    den += tp[p.Dh + 1] * f;
  }
  const float gate = p.gated ? float(proj[(size_t)t * p.proj_stride + h * 2 * p.Dh + p.Dh + tid]) : INFINITY;
  float o;
  if (tr_kv4_rv) {
    threadgroup float hb[512];
    o = float(bfloat(tr_hadamard(hb, num / den, tid, p.Dh)));
  } else {
    o = float(bfloat(num / den));
  }
  out[((size_t)t * p.Hq + h) * p.Dh + tid] = bfloat(o * (1.f / (1.f + exp(-gate))));
}

// gdn_conv for tree sequences (tseqs[k] = the batch sequence): node i's window is its own input and its three
// nearest ancestors' (the committed conv state past the root). The state is left untouched (verify) and the inputs
// go to the tape. Along a run of rows that continue each other the window shifts as in gdn_conv; after a jump
// (tdep bit 9: the parent is not the previous row) it is re-read from the ancestors' inputs.
// grid (conv_dim, number of tree sequences).
kernel void gdn_conv_tree(device const bfloat* proj [[buffer(0)]], device const bfloat* conv_state [[buffer(1)]],
                          device const bfloat* w [[buffer(2)]], device bfloat* out [[buffer(3)]],
                          device const int* slot [[buffer(4)]], constant GdnArgs& p [[buffer(5)]],
                          device const int* seq_start [[buffer(6)]], device const int* seq_len [[buffer(7)]],
                          device bfloat* tape_x [[buffer(8)]], device const int* tseqs [[buffer(9)]],
                          device const int* tpar [[buffer(10)]], device const int* tdep [[buffer(11)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, b = tseqs[gid.y];
  if (c >= p.conv_dim) return;
  device const bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  const float s0 = st[c], s1 = st[p.conv_dim + c], s2 = st[2 * p.conv_dim + c];
  device const bfloat* wc = w + (size_t)c * 4;
  const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
  const int t0 = seq_start[b], n = seq_len[b];
  auto xin = [&](int node) { return float(proj[(size_t)(t0 + node) * p.proj_stride + c]); };
  float v1 = s2, v2 = s1, v3 = s0;  // the window before the current row: parent, grandparent, great-grandparent
  for (int i = 0; i < n; i++) {
    const int t = t0 + i, td = tdep[t], d = td & 0xff;
    const float x = xin(i);
    tape_x[(size_t)t * p.conv_dim + c] = bfloat(x);
    if (td & 0x200) {
      const int a1 = tpar[t], a2 = d >= 2 ? tpar[t0 + a1] : -1, a3 = d >= 3 ? tpar[t0 + a2] : -1;
      v1 = xin(a1);
      v2 = d >= 2 ? xin(a2) : s2;
      v3 = d >= 3 ? xin(a3) : d == 2 ? s2 : s1;
    }
    const float y = float(bfloat(w0 * v3 + w1 * v2 + w2 * v1 + w3 * x));
    out[(size_t)t * p.conv_dim + c] = bfloat(y / (1.f + exp(-y)));
    v3 = v2; v2 = v1; v1 = x;
  }
}

// One delta-rule update of a state row (gdn_step's arithmetic) with row `row`'s k, v, g, beta.
struct GdnRow { float kk[4], v, gg, bt; };
inline float gdn_update(thread float* s, thread const GdnRow& r) {
  float kv = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    s[i] *= r.gg;
    kv += s[i] * r.kk[i];
  }
  kv = simd_sum(kv);
  const float delta = (r.v - kv) * r.bt;
#pragma unroll
  for (int i = 0; i < 4; i++) s[i] += r.kk[i] * delta;
  return 0;
}

// gdn_step for tree sequences: node i continues from its parent's state. Along a run of rows that continue each
// other the state stays in registers as in gdn_step. After a jump (tdep bit 9) the parent's state is rebuilt from the
// state after the root (kept in registers) by replaying the parent's ancestors below the root (tanc): no per-node
// state is stored (a register stack spilled, a threadgroup one cost occupancy). Same arithmetic as gdn_step.
// grid (32, Dv, ntree * Hv), threadgroup (32, 4, 1).
template <typename ST>
kernel void gdn_step_tree(device const bfloat* conv_out [[buffer(0)]], device const float* g [[buffer(1)]],
                          device const float* beta [[buffer(2)]], device const ST* state [[buffer(3)]],
                          device bfloat* y [[buffer(4)]], device const int* slot [[buffer(5)]],
                          constant GdnArgs& p [[buffer(6)]], device const int* seq_start [[buffer(7)]],
                          device const int* seq_len [[buffer(8)]], device const int* tseqs [[buffer(9)]],
                          device const int* tpar [[buffer(10)]], device const int* tdep [[buffer(11)]],
                          device const uint* tanc [[buffer(12)]],
                          uint3 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NPT = 4;
  const int n = gid.z, b = tseqs[n / p.Hv], hv = n % p.Hv, hk = hv / (p.Hv / p.Hk);
  const int dv = gid.y;
  device const ST* st = state + (((size_t)slot[b] * p.Hv + hv) * p.Dv + dv) * p.Dk;
  const int t0 = seq_start[b], nt = seq_len[b];
  auto load = [&](int t) {
    GdnRow r;
    device const bfloat* row = conv_out + (size_t)t * p.conv_dim;
    device const bfloat* k = row + p.Hk * p.Dk + hk * p.Dk;
#pragma unroll
    for (int i = 0; i < NPT; i++) r.kk[i] = k[NPT * lane + i];
    r.v = row[2 * p.Hk * p.Dk + hv * p.Dv + dv];
    r.gg = g[t * p.Hv + hv];
    r.bt = beta[t * p.Hv + hv];
    return r;
  };
  float s[NPT], sr[NPT];  // the current state; the state after the root
#pragma unroll
  for (int i = 0; i < NPT; i++) s[i] = float(st[NPT * lane + i]);
  for (int j = 0; j < nt; j++) {
    const int t = t0 + j, td = tdep[t];
    if (td & 0x200) {
#pragma unroll
      for (int i = 0; i < NPT; i++) s[i] = sr[i];
      for (uint m = tanc[t0 + tpar[t]] & ~1u; m; m &= m - 1) {  // the parent's ancestors below the root, and itself
        const GdnRow r = load(t0 + ctz(m));
        gdn_update(s, r);
      }
    }
    const GdnRow r = load(t);
    gdn_update(s, r);
    device const bfloat* q = conv_out + (size_t)t * p.conv_dim + hk * p.Dk;
    float o = 0;
#pragma unroll
    for (int i = 0; i < NPT; i++) o += s[i] * float(q[NPT * lane + i]);
    o = simd_sum(o);
    if (lane == 0) y[((size_t)t * p.Hv + hv) * p.Dv + dv] = bfloat(o);
    if (j == 0)
#pragma unroll
      for (int i = 0; i < NPT; i++) sr[i] = s[i];
  }
}
template [[host_name("gdn_step_tree")]] kernel void gdn_step_tree<float>(device const bfloat*, device const float*, device const float*, device const float*, device bfloat*, device const int*, constant GdnArgs&, device const int*, device const int*, device const int*, device const int*, device const int*, device const uint*, uint3, uint);
template [[host_name("gdn_step_tree_s16")]] kernel void gdn_step_tree<bfloat>(device const bfloat*, device const float*, device const float*, device const bfloat*, device bfloat*, device const int*, constant GdnArgs&, device const int*, device const int*, device const int*, device const int*, device const int*, device const uint*, uint3, uint);


// Greedy acceptance for tree sequences (after spec_accept, whose results for them it replaces): from the root,
// follow the child whose token is the target's argmax after the current node. commit = path length (root
// included); emit = the path's drafted tokens, then the bonus; tpath[b * MAX_BLOCK + k] = the k-th path node.
kernel void tree_accept(device const int* tok [[buffer(0)]], device const int* pred [[buffer(1)]],
                        device const int* seq_start [[buffer(2)]], device const int* seq_len [[buffer(3)]],
                        device const int* out_base [[buffer(4)]], device int* commit [[buffer(5)]],
                        device int* emit [[buffer(6)]], device int* tpath [[buffer(7)]],
                        device const int* tseqs [[buffer(8)]], device const int* tpar [[buffer(9)]],
                        constant int& NT [[buffer(10)]], uint k [[thread_position_in_grid]]) {
  if ((int)k >= NT) return;
  const int b = tseqs[k], t0 = seq_start[b], n = seq_len[b], ob = out_base[b];
  int cur = 0, a = 0;
  tpath[b * MAX_BLOCK] = 0;
  int pr = pred[ob];
  while (a < MAX_BLOCK - 1) {
    int nx = -1;
    for (int j = cur + 1; j < n; j++)
      if (tpar[t0 + j] == cur && tok[t0 + j] == pr) { nx = j; break; }
    if (nx < 0) break;
    emit[b * MAX_BLOCK + a] = pr;
    tpath[b * MAX_BLOCK + ++a] = nx;
    cur = nx;
    pr = pred[ob + cur];
  }
  emit[b * MAX_BLOCK + a] = pr;
  commit[b] = a + 1;
}

// gdn_conv_replay / gdn_replay along each tree sequence's accepted path (rows t0 + tpath[...]). grid as theirs, with
// the tree sequences in place of all sequences.
kernel void gdn_conv_replay_tree(device const bfloat* tape_x [[buffer(0)]], device bfloat* conv_state [[buffer(1)]],
                                 device const int* slot [[buffer(2)]], device const int* seq_start [[buffer(3)]],
                                 device const int* commit [[buffer(4)]], device const int* tseqs [[buffer(5)]],
                                 constant ReplayArgs& p [[buffer(6)]], device const int* tpath [[buffer(7)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, b = tseqs[gid.y];
  if (c >= p.conv_dim) return;
  device bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  float s0 = st[c], s1 = st[p.conv_dim + c], s2 = st[2 * p.conv_dim + c];
  const int t0 = seq_start[b], n = commit[b];
  for (int j = 0; j < n; j++) {
    s0 = s1; s1 = s2;
    s2 = tape_x[(size_t)(t0 + tpath[b * MAX_BLOCK + j]) * p.conv_dim + c];
  }
  st[c] = bfloat(s0);
  st[p.conv_dim + c] = bfloat(s1);
  st[2 * p.conv_dim + c] = bfloat(s2);
}

template <typename ST>
kernel void gdn_replay_tree(device const bfloat* conv_out [[buffer(0)]], device const float* g [[buffer(1)]],
                            device const float* beta [[buffer(2)]], device ST* state [[buffer(3)]],
                            device const int* slot [[buffer(4)]], device const int* seq_start [[buffer(5)]],
                            device const int* commit [[buffer(6)]], device const int* tseqs [[buffer(7)]],
                            constant ReplayArgs& p [[buffer(8)]], device const int* tpath [[buffer(9)]],
                            uint3 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const int n = gid.z, b = tseqs[n / p.Hv], hv = n % p.Hv, hk = hv / (p.Hv / p.Hk);
  const int dv = gid.y;
  constexpr int NPT = 4;
  device ST* st = state + (((size_t)slot[b] * p.Hv + hv) * p.Dv + dv) * p.Dk;
  float s[NPT];
  for (int i = 0; i < NPT; i++) s[i] = float(st[NPT * lane + i]);
  const int t0 = seq_start[b], nt = commit[b];
  for (int j = 0; j < nt; j++) {
    const int t = t0 + tpath[b * MAX_BLOCK + j];
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
  for (int i = 0; i < NPT; i++) st[NPT * lane + i] = ST(s[i]);
}
template [[host_name("gdn_replay_tree")]] kernel void gdn_replay_tree<float>(device const bfloat*, device const float*, device const float*, device float*, device const int*, device const int*, device const int*, device const int*, constant ReplayArgs&, device const int*, uint3, uint);
template [[host_name("gdn_replay_tree_s16")]] kernel void gdn_replay_tree<bfloat>(device const bfloat*, device const float*, device const float*, device bfloat*, device const int*, device const int*, device const int*, device const int*, constant ReplayArgs&, device const int*, uint3, uint);


// Move the accepted path's K/V (and 8-bit scales) of one attention layer from slots pos0 + tpath[j] to pos0 + j.
// Path nodes ascend with i_j >= j, so walking j upward reads every source before anything overwrites it.
// grid threads (Dh, Hkv, number of tree sequences).
struct CompactArgs { int Dh, ns, max_blocks; };
template <bool KV8, bool KV4 = false>
kernel void kv_compact(device void* Kc [[buffer(0)]], device void* Vc [[buffer(1)]], device float* Ks [[buffer(2)]],
                       device float* Vs [[buffer(3)]], device const int* block_table [[buffer(4)]],
                       device const int* pos [[buffer(5)]], device const int* seq_start [[buffer(6)]],
                       device const int* commit [[buffer(7)]], device const int* tpath [[buffer(8)]],
                       device const int* tseqs [[buffer(9)]], constant CompactArgs& p [[buffer(10)]],
                       uint3 gid [[thread_position_in_grid]]) {
  const int d = gid.x, kh = gid.y, b = tseqs[gid.z];
  const int t0 = seq_start[b], n = commit[b];
  auto si_of = [&](int ps) { return (size_t)kh * p.ns + (size_t)block_table[b * p.max_blocks + ps / BLK] * BLK + ps % BLK; };
  for (int j = 1; j < n; j++) {
    const int src = tpath[b * MAX_BLOCK + j];
    if (src == j) continue;
    const size_t a = si_of(pos[t0 + src]), z = si_of(pos[t0 + j]);
    if (KV4) {  // Dh / 2 bytes a slot: threads d < Dh / 2 move one byte each
      if (d < p.Dh / 2) {
        ((device uchar*)Kc)[z * p.Dh / 2 + d] = ((device const uchar*)Kc)[a * p.Dh / 2 + d];
        ((device uchar*)Vc)[z * p.Dh / 2 + d] = ((device const uchar*)Vc)[a * p.Dh / 2 + d];
      }
      if (d == 0) { Ks[z] = Ks[a]; Vs[z] = Vs[a]; }
    } else if (KV8) {
      ((device char*)Kc)[z * p.Dh + d] = ((device const char*)Kc)[a * p.Dh + d];
      ((device char*)Vc)[z * p.Dh + d] = ((device const char*)Vc)[a * p.Dh + d];
      if (d == 0) { Ks[z] = Ks[a]; Vs[z] = Vs[a]; }
    } else {
      ((device bfloat*)Kc)[z * p.Dh + d] = ((device const bfloat*)Kc)[a * p.Dh + d];
      ((device bfloat*)Vc)[z * p.Dh + d] = ((device const bfloat*)Vc)[a * p.Dh + d];
    }
  }
}
template [[host_name("kv_compact")]] kernel void kv_compact<false>(device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device const int*,
    constant CompactArgs&, uint3);
template [[host_name("kv_compact_kv4")]] kernel void kv_compact<true, true>(device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device const int*,
    constant CompactArgs&, uint3);
template [[host_name("kv_compact_kv8")]] kernel void kv_compact<true>(device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device const int*,
    constant CompactArgs&, uint3);

// The tree builder of src/tree.h (build_tree) on the GPU, so a tree step needs no host round trip between the
// drafter and the verify (TREE_GPU=1). One threadgroup of 256 (= the selector rank) per tree sequence; info[k] =
// {drafter row of the block's first position, target row of the root, pos0, rows}. Writes the verify rows' tokens (after
// the root), rope positions, parents, depths (+ flags) and ancestor masks, as set_tree does.
struct TreeBuildArgs { int budget, max_kids, max_depth, block; float temp; };
constant constexpr int TB_MAX = 32, TB_K = 16;
kernel void tree_build(device const int* cand [[buffer(0)]], device const float* unary [[buffer(1)]],
                       device const bfloat* hp [[buffer(2)]], device const bfloat* pred_cb [[buffer(3)]],
                       device const bfloat* succ_cb [[buffer(4)]], device const int4* info [[buffer(5)]],
                       device int* tok [[buffer(6)]], device int* rpos [[buffer(7)]], device int* tpar [[buffer(8)]],
                       device int* tdep [[buffer(9)]], device uint* tanc [[buffer(10)]],
                       constant TreeBuildArgs& a [[buffer(11)]], uint k [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int ctok[TB_MAX], cpar[TB_MAX], cdep[TB_MAX], nkid[TB_MAX], kid[TB_MAX][TB_K], map[TB_MAX];
  threadgroup float clp[TB_MAX];
  threadgroup int kids_id[TB_MAX][TB_K];
  threadgroup float kids_lp[TB_MAX][TB_K];
  threadgroup float part[8][TB_K];
  // frontier: (lp, node, rank)
  threadgroup float flp[2 * TB_MAX];
  threadgroup int fnode[2 * TB_MAX], frank[2 * TB_MAX];
  threadgroup int nf, nn, expand_u;
  const int4 in = info[k];
  const int r0 = in.x, t0 = in.y, pos0 = in.z, rows = in.w;
  const int budget = min(a.budget, TB_MAX), maxd = min(a.max_depth, a.block - 1);
  // Rank node u's children (block position cdep[u] + 1) into kids_id/kids_lp[u]; all threads.
  auto expand = [&](int u) {
    const int row = r0 + cdep[u] + 1;
    const float ph = float(pred_cb[(size_t)ctok[u] * 256 + tid]) * float(hp[(size_t)row * 256 + tid]);
    for (int c = 0; c < TB_K; c++) {
      const float v = simd_sum(ph * float(succ_cb[(size_t)cand[row * TB_K + c] * 256 + tid]));
      if (lane == 0) part[sg][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float s[TB_K];
      int ord[TB_K];
      for (int c = 0; c < TB_K; c++) {
        float acc = 0;
        for (int g = 0; g < 8; g++) acc += part[g][c];
        s[c] = (acc + unary[row * TB_K + c]) / a.temp;
        ord[c] = c;
      }
      for (int i = 1; i < TB_K; i++) {  // stable insertion sort, descending
        const int o = ord[i];
        int j = i;
        while (j > 0 && s[ord[j - 1]] < s[o]) { ord[j] = ord[j - 1]; j--; }
        ord[j] = o;
      }
      float m = s[ord[0]], z = 0;
      for (int c = 0; c < TB_K; c++) z += exp(s[c] - m);
      const float lz = m + log(z);
      for (int c = 0; c < TB_K; c++) {
        kids_id[u][c] = cand[row * TB_K + ord[c]];
        kids_lp[u][c] = s[ord[c]] - lz;
      }
      flp[nf] = clp[u] + kids_lp[u][0];
      fnode[nf] = u;
      frank[nf] = 0;
      nf++;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  };
  if (tid == 0) {
    ctok[0] = tok[t0];
    cpar[0] = -1;
    cdep[0] = 0;
    clp[0] = 0;
    nkid[0] = 0;
    nf = 0;
    nn = 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (maxd > 0) expand(0);
  while (true) {
    if (tid == 0) {
      expand_u = -1;
      if (nn < budget && nf > 0) {
        int bi = 0;
        for (int i = 1; i < nf; i++)
          if (flp[i] > flp[bi]) bi = i;
        const float lp = flp[bi];
        const int u = fnode[bi], r = frank[bi];
        flp[bi] = flp[nf - 1]; fnode[bi] = fnode[nf - 1]; frank[bi] = frank[nf - 1];
        nf--;
        const int v = nn++;
        ctok[v] = kids_id[u][r];
        cpar[v] = u;
        cdep[v] = cdep[u] + 1;
        clp[v] = lp;
        nkid[v] = 0;
        kid[u][nkid[u]++] = v;
        if (r + 1 < a.max_kids && r + 1 < TB_K) {
          flp[nf] = clp[u] + kids_lp[u][r + 1];
          fnode[nf] = u;
          frank[nf] = r + 1;
          nf++;
        }
        expand_u = cdep[v] < maxd ? v : -2;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u = expand_u;
    if (u == -1) break;
    if (u >= 0) expand(u);
  }
  if (tid != 0) return;
  // DFS preorder (children in creation order), then the rows.
  int stack[TB_MAX], sp = 0, n = 0;
  stack[sp++] = 0;
  int otok[TB_MAX], opar[TB_MAX], odep[TB_MAX];
  uint oanc[TB_MAX];
  while (sp) {
    const int u = stack[--sp];
    const int i = n++;
    map[u] = i;
    otok[i] = ctok[u];
    odep[i] = cdep[u];
    opar[i] = cpar[u] < 0 ? -1 : map[cpar[u]];
    oanc[i] = (opar[i] < 0 ? 0u : oanc[opar[i]]) | (1u << i);
    for (int c = nkid[u] - 1; c >= 0; c--) stack[sp++] = kid[u][c];
  }
  for (int i = 0; i < n; i++) {
    const int t = t0 + i;
    if (i) tok[t] = otok[i];
    for (int c = 0; c < 3; c++) rpos[t * 3 + c] = pos0 + odep[i];
    tpar[t] = opar[i];
    int kids = 0;
    for (int j = i + 1; j < n; j++) kids += opar[j] == i;
    tdep[t] = odep[i] | (kids >= 2 ? 0x100 : 0) | (i > 0 && opar[i] != i - 1 ? 0x200 : 0);
    tanc[t] = oanc[i];
  }
  for (int i = n; i < rows; i++) {  // padding (a tree smaller than its rows): a duplicate of the root's first child,
    const int t = t0 + i;            // which tree_accept never picks over the original
    tok[t] = n > 1 ? otok[1] : otok[0];
    for (int c = 0; c < 3; c++) rpos[t * 3 + c] = pos0 + 1;
    tpar[t] = 0;
    tdep[t] = 1 | 0x200;
    tanc[t] = 1u | (1u << i);
  }
}
