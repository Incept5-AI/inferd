// Qwen3.8-Flash-Next (qwen4_exp) pieces: hyper-connections (hc_count residual streams mixed by a low-rank gate) and
// the per-layer n-gram embedding (PLE). The streams live in hx [T, hc * H]; a block reads one mixed row [T, H] and
// its output is injected back into every stream with a per-stream weight.
#include <metal_stdlib>
using namespace metal;

// hx[t, s * H + h] = x[t, h]. grid (H, T).
struct TileArgs { int H, hc; };
kernel void hc_tile(device const bfloat* x [[buffer(0)]], device bfloat* hx [[buffer(1)]],
                    constant TileArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.H) return;
  const bfloat v = x[(size_t)gid.y * p.H + gid.x];
  for (int s = 0; s < p.hc; s++) hx[(size_t)gid.y * p.hc * p.H + s * p.H + gid.x] = v;
}

// Grouped, zero-centred RMS norm: each H-wide group of a row normalised on its own, times (1 + w).
// in [T, in_stride] (groups start at column 0), out [T, hc * H]. One threadgroup (1024) per (group, row).
struct HcNormArgs { int H, hc, in_stride; float eps; };
kernel void hc_norm(device const bfloat* in [[buffer(0)]], device const bfloat* w [[buffer(1)]],
                    device bfloat* out [[buffer(2)]], constant HcNormArgs& p [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int s = tg.x, t = tg.y;
  device const bfloat* xr = in + (size_t)t * p.in_stride + s * p.H;
  float ss = 0;
  for (int i = tid; i < p.H; i += 1024) { const float v = xr[i]; ss += v * v; }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) part[0] = simd_sum(lane < 32 ? part[lane] : 0.f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = rsqrt(part[0] / p.H + p.eps);
  device bfloat* o = out + (size_t)t * p.hc * p.H + s * p.H;
  for (int i = tid; i < p.H; i += 1024) o[i] = bfloat(float(xr[i]) * r * (1.f + float(w[s * p.H + i])));
}

// From the fused down|inject projection o [T, stride]: mix[t, j] = silu(o / hc) for j < R (bf16), and, when
// inject, injw[t, s] = 2 * sigmoid(o[t, R + s] / hc) (float). grid (R + hc, T).
struct HcActArgs { int R, hc, stride, inject; };
kernel void hc_act(device const bfloat* o [[buffer(0)]], device bfloat* mix [[buffer(1)]],
                   device float* injw [[buffer(2)]], constant HcActArgs& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
  const int j = gid.x, t = gid.y;
  const float v = float(bfloat(float(o[(size_t)t * p.stride + j]) / p.hc));
  if (j < p.R) {
    mix[(size_t)t * p.R + j] = bfloat(v / (1.f + exp(-v)));
  } else if (p.inject && j < p.R + p.hc) {
    injw[t * p.hc + (j - p.R)] = float(bfloat(2.f * float(bfloat(1.f / (1.f + exp(-v))))));
  }
}

// xn[t, h] = mean_s sigmoid(up[t, s * H + h]) * normed[t, s * H + h]. grid (H, T).
struct HcMixArgs { int H, hc; };
kernel void hc_mix(device const bfloat* up [[buffer(0)]], device const bfloat* normed [[buffer(1)]],
                   device bfloat* xn [[buffer(2)]], constant HcMixArgs& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
  const int h = gid.x, t = gid.y;
  if (h >= p.H) return;
  float acc = 0;
  for (int s = 0; s < p.hc; s++) {
    const size_t i = (size_t)t * p.hc * p.H + s * p.H + h;
    const float m = float(bfloat(1.f / (1.f + exp(-float(up[i])))));
    acc += float(bfloat(m * float(normed[i])));
  }
  xn[(size_t)t * p.H + h] = bfloat(acc / p.hc);
}

// hx[t, s * H + h] += branch[t, h] * injw[t, s]. grid (H, T).
kernel void hc_inject(device bfloat* hx [[buffer(0)]], device const bfloat* branch [[buffer(1)]],
                      device const float* injw [[buffer(2)]], constant HcMixArgs& p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int h = gid.x, t = gid.y;
  if (h >= p.H) return;
  const float b = branch[(size_t)t * p.H + h];
  for (int s = 0; s < p.hc; s++) {
    device bfloat& v = hx[(size_t)t * p.hc * p.H + s * p.H + h];
    v = bfloat(float(v) + float(bfloat(b * injw[t * p.hc + s])));
  }
}

// out[i, :] = in[rows[i], :] (hc_ prefix: core has a gather_rows). grid (H, n).
struct GatherRowArgs { int H; };
kernel void hc_gather_rows(device const bfloat* in [[buffer(0)]], device const int* rows [[buffer(1)]],
                        device bfloat* out [[buffer(2)]], constant GatherRowArgs& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.H) return;
  out[(size_t)gid.y * p.H + gid.x] = in[(size_t)rows[gid.y] * p.H + gid.x];
}

// N-gram embedding rows (4-bit, group 32, MLX layout) from a table split into 4 row ranges (one buffer each):
// emb[t, head * W + d] = dequant(row ids[t * heads + head])[d]. grid (W, heads, T).
struct PleGatherArgs { int heads, W, rows_per_part; };
kernel void ple_gather(device const uint* w0 [[buffer(0)]], device const uint* w1 [[buffer(1)]],
                       device const uint* w2 [[buffer(2)]], device const uint* w3 [[buffer(3)]],
                       device const bfloat* s0 [[buffer(4)]], device const bfloat* s1 [[buffer(5)]],
                       device const bfloat* s2 [[buffer(6)]], device const bfloat* s3 [[buffer(7)]],
                       device const bfloat* b0 [[buffer(8)]], device const bfloat* b1 [[buffer(9)]],
                       device const bfloat* b2 [[buffer(10)]], device const bfloat* b3 [[buffer(11)]],
                       device const int* ids [[buffer(12)]], device bfloat* emb [[buffer(13)]],
                       constant PleGatherArgs& p [[buffer(14)]], uint3 gid [[thread_position_in_grid]]) {
  const int d = gid.x, head = gid.y, t = gid.z;
  if (d >= p.W) return;
  const int row = ids[t * p.heads + head], part = row / p.rows_per_part;
  const size_t r = (size_t)(row - part * p.rows_per_part);
  device const uint* W = part == 0 ? w0 : part == 1 ? w1 : part == 2 ? w2 : w3;
  device const bfloat* S = part == 0 ? s0 : part == 1 ? s1 : part == 2 ? s2 : s3;
  device const bfloat* B = part == 0 ? b0 : part == 1 ? b1 : part == 2 ? b2 : b3;
  const uint u = W[r * (p.W / 8) + d / 8];
  const float q = float((u >> (4 * (d % 8))) & 15u);
  const int gi = d / 32, groups = p.W / 32;
  emb[(size_t)t * p.heads * p.W + head * p.W + d] = bfloat(q * float(S[r * groups + gi]) + float(B[r * groups + gi]));
}

// PLE gate: g_s = <keys_s, queries_s> / sqrt(H), g = sign(g) sqrt(max(|g|, 1e-6)), gv[t, s * H + h] =
// sigmoid(g_s) * values[t, h]. keys normed [T, hc * H], queries normed [T, hc * H], values in kv [T, kv_stride] at
// column val_off. One threadgroup (1024) per (stream, row).
struct PleGateArgs { int H, hc, kv_stride, val_off; };
kernel void ple_gate(device const bfloat* keys [[buffer(0)]], device const bfloat* queries [[buffer(1)]],
                     device const bfloat* kv [[buffer(2)]], device bfloat* gv [[buffer(3)]],
                     constant PleGateArgs& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int s = tg.x, t = tg.y;
  const size_t base = (size_t)t * p.hc * p.H + s * p.H;
  float acc = 0;
  for (int i = tid; i < p.H; i += 1024) acc += float(bfloat(float(keys[base + i]) * float(queries[base + i])));
  acc = simd_sum(acc);
  if (lane == 0) part[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) part[0] = simd_sum(lane < 32 ? part[lane] : 0.f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g = float(bfloat(part[0] / sqrt(float(p.H))));
  g = float(bfloat(sign(g) * sqrt(max(fabs(g), 1e-6f))));
  const float sgm = float(bfloat(1.f / (1.f + exp(-g))));
  for (int i = tid; i < p.H; i += 1024)
    gv[base + i] = bfloat(sgm * float(kv[(size_t)t * p.kv_stride + p.val_off + i]));
}

// PLE short conv + residual: hx += gv + silu(conv(normed)), a depthwise causal conv (kernel K, dilation D) over each
// sequence's rows in order, with the last (K - 1) * D inputs of the sequence kept per slot in state
// [nslots, (K - 1) * D, C]. w [C, K]. grid (C, S).
// Verify steps (p.verify): the inputs also go to tape [T, C], and speculative sequences (spec[b]) keep their state
// (ple_conv_replay advances it over the committed rows).
struct PleConvArgs { int C, K, D, verify; };
kernel void ple_conv(device const bfloat* normed [[buffer(0)]], device const bfloat* gv [[buffer(1)]],
                     device const bfloat* w [[buffer(2)]], device bfloat* state [[buffer(3)]],
                     device bfloat* hx [[buffer(4)]], device const int* slot [[buffer(5)]],
                     device const int* seq_start [[buffer(6)]], device const int* seq_len [[buffer(7)]],
                     constant PleConvArgs& p [[buffer(8)]], device bfloat* tape [[buffer(9)]],
                     device const int* spec [[buffer(10)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, b = gid.y;
  if (c >= p.C) return;
  const int L = (p.K - 1) * p.D;  // 9 for K 4, D 3
  device bfloat* st = state + (size_t)slot[b] * L * p.C;
  float hist[16];  // L <= 16
  for (int i = 0; i < L; i++) hist[i] = st[(size_t)i * p.C + c];
  float wk[8];
  for (int k = 0; k < p.K; k++) wk[k] = w[c * p.K + k];
  const int r0 = seq_start[b], n = seq_len[b];
  for (int r = 0; r < n; r++) {
    const size_t i = (size_t)(r0 + r) * p.C + c;
    const float x = normed[i];
    if (p.verify) tape[i] = bfloat(x);
    // tap k reads input (K - 1 - k) * D steps back; the current input is tap K - 1
    float acc = wk[p.K - 1] * x;
    for (int k = 0; k < p.K - 1; k++) acc += wk[k] * hist[L - (p.K - 1 - k) * p.D];
    const float y = float(bfloat(acc));
    const float out = float(bfloat(float(gv[i]) + float(bfloat(y / (1.f + exp(-y))))));
    hx[i] = bfloat(float(hx[i]) + out);
    for (int j = 0; j < L - 1; j++) hist[j] = hist[j + 1];
    hist[L - 1] = x;
  }
  if (p.verify && spec[b]) return;
  for (int i = 0; i < L; i++) st[(size_t)i * p.C + c] = bfloat(hist[i]);
}

// Advance speculative sequences' PLE conv history over their committed rows from the verify tape. grid (C, S).
kernel void ple_conv_replay(device const bfloat* tape [[buffer(0)]], device bfloat* state [[buffer(1)]],
                            device const int* slot [[buffer(2)]], device const int* seq_start [[buffer(3)]],
                            device const int* commit [[buffer(4)]], device const int* spec [[buffer(5)]],
                            constant PleConvArgs& p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, b = gid.y;
  if (c >= p.C || !spec[b]) return;
  const int L = (p.K - 1) * p.D;
  device bfloat* st = state + (size_t)slot[b] * L * p.C;
  bfloat hist[16];
  for (int i = 0; i < L; i++) hist[i] = st[(size_t)i * p.C + c];
  const int r0 = seq_start[b], n = commit[b];
  for (int r = 0; r < n; r++) {
    for (int j = 0; j < L - 1; j++) hist[j] = hist[j + 1];
    hist[L - 1] = tape[(size_t)(r0 + r) * p.C + c];
  }
  for (int i = 0; i < L; i++) st[(size_t)i * p.C + c] = hist[i];
}

// Fused hyper-connection "up" projection + stream mix for small T (decode): xn[t, h] = mean_s sigmoid(<mix[t, :],
// Wup[s * H + h, :]>) * normed[t, s * H + h]. Wup is MLX affine (group 64), 4- or 8-bit, K = hc_low (320): too narrow
// for the 512-wide GEMV, so the unfused path sent one token through a 32-row matmul tile per layer. One simdgroup per
// (h, t), lanes over K; the hc streams' rows are read back to back.
// grid: threadgroups (ceil(H / 8), T), 256 threads.
struct HcUpArgs { int H, hc, K, xr; };
template <int BITS>
kernel void hc_up_mix(device const bfloat* mix [[buffer(0)]], device const uchar* W [[buffer(1)]],
                      device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                      device const bfloat* normed [[buffer(4)]], device bfloat* xn [[buffer(5)]],
                      constant HcUpArgs& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int h = tg.x * 8 + sg, t = tg.y;
  if (h >= p.H) return;
  const int K = p.K, G = K / 64;
  device const bfloat* x = mix + (size_t)t * K;
  float acc = 0;
  for (int s = 0; s < p.hc; s++) {
    const size_t n = (size_t)s * p.H + h;
    float d = 0;
    for (int k = lane; k < K; k += 32) {
      float q;
      if (BITS == 8) q = float(W[n * K + k] ^ uchar(p.xr));
      else q = float(((W[(n * K + k) / 2] >> (4 * (k & 1))) & 15) ^ (uint(p.xr) & 15));
      d += float(x[k]) * (q * float(S[n * G + k / 64]) + float(B[n * G + k / 64]));
    }
    d = simd_sum(d);
    const float m = float(bfloat(1.f / (1.f + exp(-float(bfloat(d))))));
    acc += float(bfloat(m * float(normed[(size_t)t * p.hc * p.H + n])));
  }
  if (lane == 0) xn[(size_t)t * p.H + h] = bfloat(acc / p.hc);
}
template [[host_name("hc_up_mix4")]] kernel void hc_up_mix<4>(device const bfloat*, device const uchar*, device const bfloat*,
    device const bfloat*, device const bfloat*, device bfloat*, constant HcUpArgs&, uint2, uint, uint);
template [[host_name("hc_up_mix8")]] kernel void hc_up_mix<8>(device const bfloat*, device const uchar*, device const bfloat*,
    device const bfloat*, device const bfloat*, device bfloat*, constant HcUpArgs&, uint2, uint, uint);

// PLE n-gram ids on the GPU, for the rows of speculative sequences whose tokens are drafts written on the GPU (the
// host's ids for those rows saw placeholders). Same hash as NGram::ids (src/qwen4.h): ctx[0] = the row's token,
// ctx[k] = the token k positions back (from the step's rows, or hist[b * 2 + i] = the token i + 1 positions before the
// sequence's first row), read as EOS at or before the latest EOS ahead and before position 0; per head h of the
// (h / heads_per + 2)-gram: (ctx[0] * m0 ^ ctx[1] * m1 ^ ...) mod size[h] + off[h] (64-bit, signed remainder).
// tab: int64 [3 multipliers, heads sizes, heads offsets]. grid (heads, rows per sequence, S); only spec[b] sequences.
struct NgArgs { int heads_per, ngram, heads, eos, all; };
kernel void ple_ids_gpu(device const int* tok [[buffer(0)]], device const int* seq_start [[buffer(1)]],
                        device const int* seq_len [[buffer(2)]], device const int* spec [[buffer(3)]],
                        device const int* pos [[buffer(4)]], device const int* hist [[buffer(5)]],
                        device const long* tab [[buffer(6)]], device int* ids [[buffer(7)]],
                        constant NgArgs& p [[buffer(8)]], uint3 gid [[thread_position_in_grid]]) {
  const int h = gid.x, j = gid.y, b = gid.z;
  if (h >= p.heads || j >= seq_len[b] || !(p.all || spec[b])) return;
  const int t = seq_start[b] + j, p0 = pos[seq_start[b]];
  long ctx[3];
  ctx[0] = tok[t];
  bool cut = false;
  for (int k = 1; k < p.ngram; k++) {
    const int pp = p0 + j - k;
    const int tk = pp < 0 ? p.eos : j - k >= 0 ? tok[t - k] : hist[b * 2 + (k - j - 1)];
    cut = cut || pp < 0 || tk == p.eos;
    ctx[k] = cut ? p.eos : tk;
  }
  const int g = h / p.heads_per + 2;
  ulong m = ulong(ctx[0]) * ulong(tab[0]);
  for (int k = 1; k < g; k++) m ^= ulong(ctx[k]) * ulong(tab[k]);
  const long mixed = as_type<long>(m), size = tab[3 + h], off = tab[3 + p.heads + h];
  long r = mixed % size;
  if (r < 0) r += size;
  ids[t * p.heads + h] = int(r + off);
}

// ---- MTP head (src/mtp.h) ----

// x[t, s * H + h] = fe[t, h] + fh[t, s * H + h]: the embedding projection broadcast to every stream plus each
// stream's hidden projection. grid (H, T).
struct MtpFuseArgs { int H, hc; };
kernel void mtp_fuse(device const bfloat* fe [[buffer(0)]], device const bfloat* fh [[buffer(1)]],
                     device bfloat* x [[buffer(2)]], constant MtpFuseArgs& p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
  const int h = gid.x, t = gid.y;
  if (h >= p.H) return;
  const float e = fe[(size_t)t * p.H + h];
  for (int s = 0; s < p.hc; s++) {
    const size_t i = (size_t)t * p.hc * p.H + s * p.H + h;
    x[i] = bfloat(e + float(fh[i]));
  }
}

// A draft pass's argmax: into the next pass's token input (has_next) and the target's verify row vrow[b] + 1 + j.
struct MtpPutArgs { int j, has_next, S; };
kernel void mtp_put(device const int* pred [[buffer(0)]], device int* next_tok [[buffer(1)]],
                    device int* tok [[buffer(2)]], device const int* vrow [[buffer(3)]],
                    constant MtpPutArgs& p [[buffer(4)]], uint b [[thread_position_in_grid]]) {
  if ((int)b >= p.S) return;
  const int v = pred[b];
  if (p.has_next) next_tok[b] = v;
  tok[vrow[b] + 1 + p.j] = v;
}

// th[slot[b]] = hx[seq_start[b] + commit[b] - 1]: each sequence's last committed row (commit = all rows for
// non-speculative sequences). grid (HH, S).
struct MtpSaveArgs { int HH; };
kernel void mtp_save_h(device const bfloat* hx [[buffer(0)]], device const int* commit [[buffer(1)]],
                       device const int* seq_start [[buffer(2)]], device const int* slot [[buffer(3)]],
                       device bfloat* th [[buffer(4)]], constant MtpSaveArgs& p [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x, b = gid.y;
  if (i >= p.HH) return;
  const int row = seq_start[b] + commit[b] - 1;
  th[(size_t)slot[b] * p.HH + i] = hx[(size_t)row * p.HH + i];
}

// Next tokens of a speculative sequence's verify rows: row j < c - 1 is followed by the accepted draft tok[r0 + j + 1],
// row c - 1 by the bonus token emit[c - 1]; later rows (rejected) get the bonus too (their KV is rewritten next cycle).
// emit rows are MAX_BLOCK = 16 wide (kMaxBlock, spec.metal). grid (S).
// A non-speculative sequence whose last row completes a greedy prompt (lastp[b]) gets the target's argmax there.
kernel void mtp_next_spec(device const int* tok [[buffer(0)]], device const int* commit [[buffer(1)]],
                          device const int* emit [[buffer(2)]], device const int* seq_start [[buffer(3)]],
                          device const int* seq_len [[buffer(4)]], device const int* spec [[buffer(5)]],
                          device int* next [[buffer(6)]], device const int* pred [[buffer(7)]],
                          device const int* out_base [[buffer(8)]], device const int* lastp [[buffer(9)]],
                          uint b [[thread_position_in_grid]]) {
  if (!spec[b]) {
    if (lastp[b]) next[seq_start[b] + seq_len[b] - 1] = pred[out_base[b]];
    return;
  }
  const int r0 = seq_start[b], n = seq_len[b], c = commit[b];
  const int bonus = emit[b * 16 + c - 1];
  for (int j = 0; j < n; j++) next[r0 + j] = j < c - 1 ? tok[r0 + j + 1] : bonus;
}

// Fold: the MTP state of each sequence's last committed row, per slot (xs) and gathered (out [S, HH]). grid (HH, S).
kernel void mtp_fold_out(device const bfloat* hs [[buffer(0)]], device const int* commit [[buffer(1)]],
                         device const int* seq_start [[buffer(2)]], device const int* slot [[buffer(3)]],
                         device bfloat* xs [[buffer(4)]], device bfloat* out [[buffer(5)]],
                         constant MtpSaveArgs& p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x, b = gid.y;
  if (i >= p.HH) return;
  const bfloat v = hs[(size_t)(seq_start[b] + commit[b] - 1) * p.HH + i];
  xs[(size_t)slot[b] * p.HH + i] = v;
  out[(size_t)b * p.HH + i] = v;
}

kernel void mtp_scatter_slot(device const int* v [[buffer(0)]], device const int* slot [[buffer(1)]],
                             device int* out [[buffer(2)]], constant int& S [[buffer(3)]],
                             uint b [[thread_position_in_grid]]) {
  if ((int)b < S) out[slot[b]] = v[b];
}

// Start a folded draft: d1 (per slot) into verify row vrow[b] + 1 and pass 1's token; pass 1's hidden (hs row b) =
// the folded MTP state. grid (HH, S).
struct MtpFoldInArgs { int HH, S, has_next; };
kernel void mtp_fold_in(device const int* d1s [[buffer(0)]], device const bfloat* xs [[buffer(1)]],
                        device const int* slot [[buffer(2)]], device const int* vrow [[buffer(3)]],
                        device int* tok [[buffer(4)]], device int* next_tok [[buffer(5)]],
                        device bfloat* hs [[buffer(6)]], constant MtpFoldInArgs& p [[buffer(7)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x, b = gid.y;
  if (i >= p.HH) return;
  hs[(size_t)b * p.HH + i] = xs[(size_t)slot[b] * p.HH + i];
  if (i == 0) {
    const int d = d1s[slot[b]];
    tok[vrow[b] + 1] = d;
    if (p.has_next) next_tok[b] = d;
  }
}

// hc_inject followed by the next block's hc_norm, in one pass: hx[t, s*H + h] += branch[t, h] * injw[t, s], then the
// updated stream is RMS-normalised (zero-centred weight w) into out. One threadgroup (1024) per (stream, row).
kernel void hc_inject_norm(device bfloat* hx [[buffer(0)]], device const bfloat* branch [[buffer(1)]],
                           device const float* injw [[buffer(2)]], device const bfloat* w [[buffer(3)]],
                           device bfloat* out [[buffer(4)]], constant HcNormArgs& p [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  const int s = tg.x, t = tg.y;
  device bfloat* xr = hx + (size_t)t * p.hc * p.H + s * p.H;
  device const bfloat* br = branch + (size_t)t * p.H;
  const float wi = injw[t * p.hc + s];
  float ss = 0;
  for (int i = tid; i < p.H; i += 1024) {
    const float v = float(bfloat(float(xr[i]) + float(bfloat(float(br[i]) * wi))));
    xr[i] = bfloat(v);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) part[0] = simd_sum(lane < 32 ? part[lane] : 0.f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = rsqrt(part[0] / p.H + p.eps);
  device bfloat* o = out + (size_t)t * p.hc * p.H + s * p.H;
  for (int i = tid; i < p.H; i += 1024) o[i] = bfloat(float(xr[i]) * r * (1.f + float(w[s * p.H + i])));
}

// ---- Two-launch hyper-connection for decode-sized steps (HC_V2; Qwen35::hyper_pre) ----
// Launch 1 (hc_nd): threadgroup (g, s, t) RMS-normalises stream s of row t (after applying a pending injection of
// the previous block's output: x + bf16(branch * injw[t, s]), rounded as hc_inject does) into threadgroup memory,
// stores strip g of the normalised stream (each element once), and runs output rows g * 8 + simdgroup of the fused
// [down | inject] projection over the K slice of stream s (K = hc * H, slice s = stream s). Float partials go to
// parts [T, hc, NR]. Launch 2 (hc_um) sums the slices, applies the activations and runs the up projection and the
// stream mix (as hc_act + hc_up_mix), and stores the pending injection into hx (each element by one thread).
// One threadgroup column per row: a version reading each weight once for all rows (rows looped inside) was slower
// at every T >= 2 (fewer threadgroups in flight; the weights are cache-resident), 2026-11-01.
// Weights: QLinear rows (MLX affine, group 64, 4- or 8-bit, xr = flip mask).
struct HcNdArgs { int H, hc, NR, pend, xr; float eps; };
template <int BITS>
inline float hc_dot8(device const uchar* w, int k, threadgroup const bfloat* x, int xr, thread float& sx) {
  // 8 consecutive weights at column k of row w (bytes), dotted with x[k..k+8); sx = sum of those x
  float q = 0, s = 0;
  if (BITS == 8) {
    const uint2 u = *(device const uint2*)(w + k) ^ uint2(uint(xr) * 0x01010101u);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float a = x[k + i], b = x[k + 4 + i];
      q += float((u.x >> (8 * i)) & 0xffu) * a + float((u.y >> (8 * i)) & 0xffu) * b;
      s += a + b;
    }
  } else {
    const uint u = *(device const uint*)(w + k / 2) ^ (uint(xr & 15) * 0x11111111u);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float a = x[k + i];
      q += float((u >> (4 * i)) & 15u) * a;
      s += a;
    }
  }
  sx = s;
  return q;
}
template <int BITS>
kernel void hc_nd(device const bfloat* hx [[buffer(0)]], device const bfloat* branch [[buffer(1)]],
                  device const float* injw [[buffer(2)]], device const bfloat* nw [[buffer(3)]],
                  device const uchar* W [[buffer(4)]], device const bfloat* S [[buffer(5)]],
                  device const bfloat* B [[buffer(6)]], device bfloat* xn [[buffer(7)]],
                  device float* parts [[buffer(8)]], constant HcNdArgs& p [[buffer(9)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xs[4096];
  threadgroup float part[8];
  const int g = tg.x, s = tg.y, t = tg.z, H = p.H, K = p.hc * H;
  device const bfloat* xr = hx + (size_t)t * K + s * H;
  device const bfloat* br = branch + (size_t)t * H;
  const float wi = p.pend ? injw[t * p.hc + s] : 0.f;
  float ss = 0;
  for (int i = tid; i < H; i += 256) {
    float v = xr[i];
    if (p.pend) v = float(bfloat(v + float(bfloat(float(br[i]) * wi))));
    xs[i] = bfloat(v);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0;
  for (int i = 0; i < 8; i++) tot += part[i];
  const float r = rsqrt(tot / H + p.eps);
  device bfloat* o = xn + (size_t)t * K + s * H;
  for (int i = tid; i < H; i += 256) {
    const bfloat y = bfloat(float(xs[i]) * r * (1.f + float(nw[s * H + i])));
    xs[i] = y;
    if (i / 256 == g) o[i] = y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = g * 8 + sg;
  if (n >= p.NR) return;
  const int KG = K / 64;
  device const uchar* w = W + (size_t)n * K * BITS / 8 + (size_t)s * H * BITS / 8;
  device const bfloat* sc = S + (size_t)n * KG + s * H / 64;
  device const bfloat* bi = B + (size_t)n * KG + s * H / 64;
  float acc = 0;
  for (int k = lane * 8; k < H; k += 256) {
    float sx;
    const float q = hc_dot8<BITS>(w, k, xs, p.xr, sx);
    acc += float(sc[k / 64]) * q + float(bi[k / 64]) * sx;
  }
  acc = simd_sum(acc);
  if (lane == 0) parts[((size_t)t * p.hc + s) * p.NR + n] = acc;
}
template [[host_name("hc_nd8")]] kernel void hc_nd<8>(device const bfloat*, device const bfloat*, device const float*,
    device const bfloat*, device const uchar*, device const bfloat*, device const bfloat*, device bfloat*, device float*,
    constant HcNdArgs&, uint3, uint, uint, uint);
template [[host_name("hc_nd4")]] kernel void hc_nd<4>(device const bfloat*, device const bfloat*, device const float*,
    device const bfloat*, device const uchar*, device const bfloat*, device const bfloat*, device bfloat*, device float*,
    constant HcNdArgs&, uint3, uint, uint, uint);

// Launch 2: mix[j] = silu(bf16(bf16(sum_s parts[t, s, j]) / hc)) for j < R in threadgroup memory; threadgroup 0 also
// writes the new injection weights inj_new[t, i] = 2 sigmoid(.) of rows R + i (inject). Then, one simdgroup per h,
// xn_out[t, h] = mean_s bf16(sigmoid(<mix, Wup[s * H + h]>) * normed[t, s * H + h]) (hc_up_mix), and with a pending
// injection hx[t, s * H + h] += bf16(branch[t, h] * inj_old[t, s]) first (branch may alias xn_out: each h is read and
// then written by the same simdgroup). grid: threadgroups (ceil(H / 8), T), 256 threads.
struct HcUmArgs { int H, hc, R, NR, pend, inject, xr; };
template <int BITS>
kernel void hc_um(device const float* parts [[buffer(0)]], device const uchar* W [[buffer(1)]],
                  device const bfloat* S [[buffer(2)]], device const bfloat* B [[buffer(3)]],
                  device const bfloat* normed [[buffer(4)]], device bfloat* out [[buffer(5)]],
                  device float* inj_new [[buffer(6)]], device bfloat* hx [[buffer(7)]],
                  device const bfloat* branch [[buffer(8)]], device const float* inj_old [[buffer(9)]],
                  constant HcUmArgs& p [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat am[1024];
  const int t = tg.y, H = p.H, R = p.R, K = p.hc * H;
  device const float* pp = parts + (size_t)t * p.hc * p.NR;
  for (int j = tid; j < R; j += 256) {
    float o = 0;
    for (int s = 0; s < p.hc; s++) o += pp[s * p.NR + j];
    const float v = float(bfloat(float(bfloat(o)) / p.hc));
    am[j] = bfloat(v / (1.f + exp(-v)));
  }
  if (p.inject && tg.x == 0 && (int)tid < p.hc) {
    float o = 0;
    for (int s = 0; s < p.hc; s++) o += pp[s * p.NR + R + tid];
    const float v = float(bfloat(float(bfloat(o)) / p.hc));
    inj_new[t * p.hc + tid] = float(bfloat(2.f * float(bfloat(1.f / (1.f + exp(-v))))));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int h = tg.x * 8 + sg;
  if (h >= H) return;
  if (p.pend && (int)lane < p.hc) {
    device bfloat& v = hx[(size_t)t * K + lane * H + h];
    v = bfloat(float(v) + float(bfloat(float(branch[(size_t)t * H + h]) * inj_old[t * p.hc + lane])));
  }
  // the four streams' up rows in parallel: 8-lane group s = lane / 8 takes stream s, 8 columns per lane per step
  const int G = R / 64, s = lane / 8, sub = lane % 8;
  const size_t n = (size_t)s * H + h;
  device const uchar* w = W + n * R * BITS / 8;
  float d = 0;
  for (int k = sub * 8; k < R; k += 64) {
    float sx;
    const float q = hc_dot8<BITS>(w, k, am, p.xr, sx);
    d += float(S[n * G + k / 64]) * q + float(B[n * G + k / 64]) * sx;
  }
  d += simd_shuffle_xor(d, 4);
  d += simd_shuffle_xor(d, 2);
  d += simd_shuffle_xor(d, 1);
  const float m = float(bfloat(1.f / (1.f + exp(-float(bfloat(d))))));
  const float pr = float(bfloat(m * float(normed[(size_t)t * K + n])));
  const float p1 = simd_shuffle(pr, 8), p2 = simd_shuffle(pr, 16), p3 = simd_shuffle(pr, 24);
  if (lane == 0) out[(size_t)t * H + h] = bfloat((((pr + p1) + p2) + p3) / p.hc);
}
template [[host_name("hc_um8")]] kernel void hc_um<8>(device const float*, device const uchar*, device const bfloat*,
    device const bfloat*, device const bfloat*, device bfloat*, device float*, device bfloat*, device const bfloat*,
    device const float*, constant HcUmArgs&, uint2, uint, uint, uint);
template [[host_name("hc_um4")]] kernel void hc_um<4>(device const float*, device const uchar*, device const bfloat*,
    device const bfloat*, device const bfloat*, device bfloat*, device float*, device bfloat*, device const bfloat*,
    device const float*, constant HcUmArgs&, uint2, uint, uint, uint);

// ---- Flash-Next verify attention on the tensor units (FN_MPP; Qwen35::encode_layer) ----
// attn_mpp (kernels/attn_mpp.metal, written for 6 query heads per kv head) with the group size GQ a template
// parameter: Flash-Next's 12 heads per kv head x up to 4 rows = the same 48 packed rows. Rows of mode 4 (set_batch).
// The packing below is q_pack / q_pack_i8 with GQ in place of 6. Copied rather than edited so the 27B's kernels
// stay byte-identical.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using FnDT2 = dextents<int32_t, 2>;
struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles, own_splits, ns, win, gated; };
template <int GQ>
kernel void fn_q_pack(device const bfloat* q [[buffer(0)]], device bfloat* qp [[buffer(1)]],
                      device const int* seq_start [[buffer(2)]], device const int* seq_len [[buffer(3)]],
                      constant GqaArgs& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]]) {
  constexpr int R = 48;
  const int kh = tg.x / R, i = tg.x % R, b = tg.y, r = i / GQ, j = i % GQ;
  const int r0 = seq_start[b], n = min(seq_len[b], R / GQ);
  qp[(((size_t)b * p.Hkv + kh) * 256 + d) * R + i] =
      r < n ? q[((size_t)(r0 + r) * p.Hq + kh * GQ + j) * 256 + d] : bfloat(0);
}
template <int GQ>
kernel void fn_q_pack_i8(device const bfloat* q [[buffer(0)]], device int8_t* qp [[buffer(1)]],
                         device const int* seq_start [[buffer(2)]], device const int* seq_len [[buffer(3)]],
                         constant GqaArgs& p [[buffer(4)]], device float* qs [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int R = 48;
  threadgroup float mx[8];
  const int kh = tg.x / R, i = tg.x % R, b = tg.y, r = i / GQ, j = i % GQ;
  const int r0 = seq_start[b], n = min(seq_len[b], R / GQ);
  const float v = r < n ? float(q[((size_t)(r0 + r) * p.Hq + kh * GQ + j) * 256 + d]) : 0.f;
  const float m = simd_max(fabs(v));
  if (lane == 0) mx[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a = 0;
  for (int k = 0; k < 8; k++) a = max(a, mx[k]);
  const float sc = a > 0 ? a / 127.f : 0.f;
  qp[(((size_t)b * p.Hkv + kh) * 256 + d) * R + i] = int8_t(sc > 0 ? clamp(rint(v / sc), -127.f, 127.f) : 0.f);
  if (d == 0) qs[((size_t)b * p.Hkv + kh) * R + i] = sc;
}
template [[host_name("fn_q_pack_g12")]] kernel void fn_q_pack<12>(device const bfloat*, device bfloat*, device const int*,
    device const int*, constant GqaArgs&, uint2, uint);
template [[host_name("fn_q_pack_i8_g12")]] kernel void fn_q_pack_i8<12>(device const bfloat*, device int8_t*,
    device const int*, device const int*, constant GqaArgs&, device float*, uint2, uint, uint, uint);

// MASK (QSA union tiles): kbeg holds one byte per gathered key and sequence b (stride p.qtiles), bit r set when row
// r of the tile attends to that key; otherwise as below.
template <int NSG, bool KV8, int GQ, bool IQ, int R = 48, bool SH = false, bool MASK = false>
kernel void attn_fn(device const void* qp [[buffer(0)]], device const void* Kc [[buffer(1)]],
                     device const void* Vc [[buffer(2)]], device const int* block_table [[buffer(3)]],
                     device const int* pos [[buffer(4)]], device const int* seq_start [[buffer(5)]],
                     device float* part [[buffer(6)]], constant GqaArgs& p [[buffer(7)]],
                     device const int* rflag [[buffer(8)]], device const int* seq_len [[buffer(9)]],
                     device const float* Ks [[buffer(10)]], device const float* Vs [[buffer(11)]],
                     device const int* kbeg [[buffer(12)]], device const float* qsc [[buffer(13)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 256, PG = 32, BK = 64, GK = 128;
  const int kh = tg.x, b = tg.y, z = tg.z;
  int r0 = 0, nrows = 0, Lmax, p0, p1, bseq = b;
  if constexpr (SH) {
    const int2 ti = ((device const int2*)seq_start)[b];
    bseq = ti.x;
    Lmax = ti.y;
    const int ssh = p.splits - p.own_splits, ch = ((Lmax + ssh - 1) / ssh + GK - 1) / GK * GK;
    p0 = z * ch;
    p1 = min(Lmax, p0 + ch);
  } else {
  r0 = seq_start[b];
  nrows = min(seq_len[b], R / GQ);
  if (rflag[r0] != 4) return;
  Lmax = pos[r0 + nrows - 1] + 1;
  // Cascade steps: a sequence in a shared-prefix group attends here to its own positions [kbeg, L) only (attn_shared
  // covers the prefix), split own_splits ways from kbeg; otherwise [0, L) in chunks of p.chunk (a multiple of GK).
  if (p.own_splits > 0) {
    const int kb = kbeg[b], own = p.own_splits;
    const int ch = ((max(Lmax - kb, 1) + own - 1) / own + GK - 1) / GK * GK;
    p0 = kb + z * ch;
    p1 = min(Lmax, p0 + ch);
  } else {
    p0 = z * p.chunk;
    p1 = min(Lmax, p0 + p.chunk);
  }
  }
  const int ld = D, mb0 = bseq * p.max_blocks;  // head planes: a head's consecutive slots are D apart

  // Raw scores of one 64-key half as the matmul stores them, key-major: Ssh[key][row] (a cooperative store keeps the
  // row index contiguous). Row pitch RS and P's pitch BKP are padded so the softmax lanes hit distinct banks.
  constexpr int RS = R == 48 ? 56 : 36, BKP = BK + 8;
  threadgroup metal::conditional_t<IQ, int32_t, float> Ssh[BK][RS];
  threadgroup bfloat Psh[2][R][BKP];  // double-buffered: P V of one half may still be reading while the next writes
  threadgroup float fac[R];
  threadgroup int need_rescale[2];   // per buffer, so a reset for the next half cannot race a late read
  threadgroup float ksh[KV8 ? GK : 1], vsh[KV8 ? GK : 1];  // KV8: the group's key / value scales, loaded once
  threadgroup float qsh[IQ ? R : 1];                         // IQ: the packed rows' query scales
  const int tid = int(sg) * 32 + int(lane);

  // S^T = K Q^T: K is the left operand in its natural row-per-key layout (as a transposed right operand it streamed
  // at ~40% of the bandwidth P V reaches), Q is packed transposed [D][48] by q_pack.
  using QE = metal::conditional_t<IQ, int8_t, bfloat>;
  using ACC = metal::conditional_t<IQ, int32_t, float>;
  auto tQ = tensor<device QE, FnDT2, tensor_inline>((device QE*)qp + ((size_t)b * p.Hkv + kh) * R * D, FnDT2(R, D));
  if constexpr (IQ) {
    if (tid < R) qsh[tid] = qsc[((size_t)b * p.Hkv + kh) * R + tid];
  }
  constexpr auto dqk = matmul2d_descriptor(PG, R, D, false, false, false);
  matmul2d<dqk, execution_simdgroups<NSG>> qk;
  constexpr auto dqk2 = matmul2d_descriptor(BK, R, D, false, false, false);
  matmul2d<dqk2, execution_simdgroups<NSG>> qk2;
  constexpr auto dpv = matmul2d_descriptor(R, D, PG, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroups<NSG>> pv;

  using KVT = metal::conditional_t<KV8, int8_t, bfloat>;
  auto kpages = [&](int blk, int n) {  // n contiguous 32-token pages of kv head kh, dims (D, n * PG), row stride ld
    return tensor<device KVT, FnDT2, tensor_inline>((device KVT*)Kc + ((size_t)kh * p.ns + (size_t)blk * PG) * D,
                                                  FnDT2(D, n * PG), array<int32_t, 2>{1, ld});
  };
  auto vpage = [&](int blk) {
    return tensor<device KVT, FnDT2, tensor_inline>((device KVT*)Vc + ((size_t)kh * p.ns + (size_t)blk * PG) * D,
                                                  FnDT2(D, PG), array<int32_t, 2>{1, ld});
  };
  auto ppage = [&](int buf, int half_) {
    return tensor<threadgroup bfloat, FnDT2, tensor_inline>(&Psh[buf][0][half_ * PG], FnDT2(PG, R), array<int32_t, 2>{1, BKP});
  };
  using QT = decltype(tQ);
  using KT = decltype(kpages(0, 1));
  using PT = decltype(ppage(0, 0));
  using VT = decltype(vpage(0));
  auto O = pv.template get_destination_cooperative_tensor<PT, VT, float>();
  for (uint16_t i = 0; i < O.get_capacity(); ++i) O[i] = 0;
  auto s2 = qk2.template get_destination_cooperative_tensor<KT, QT, ACC>();
  auto s0 = qk.template get_destination_cooperative_tensor<KT, QT, ACC>();
  // Ssh as the destination: element (packed row, key) at key * RS + row.
  auto sdst = [&](int k0, int n) {
    return tensor<threadgroup ACC, FnDT2, tensor_inline>(&Ssh[k0][0], FnDT2(R, n), array<int32_t, 2>{1, RS});
  };

  // Online softmax state of this simdgroup's rows (replicated in every lane), log2 domain. O is rescaled only when
  // a row's reference max grows by more than 2^8 (lazy rescale), so most steps never wait on the previous P V.
  // Softmax lanes: LPR consecutive threads own packed row srow, KPL keys of each 64-key half apiece, so a row's
  // max and sum are log2(LPR) xor shuffles (a whole simdgroup per row, six rows in turn, cost 0.8 ms of every
  // 1.5 ms of a 128k step). The row's running max m1 and sum l1 live replicated in its LPR lanes.
  // Lane c of a row takes keys c, c + LPR, c + 2 LPR, ... of each half.
  constexpr int LPR = R == 48 ? 4 : 256 / R, KPL = BK / LPR;
  static_assert(R * LPR <= 256, "softmax lanes");
  const int srow = tid / LPR, scol = tid % LPR;
  const bool sact = srow < R;
  float m1 = -INFINITY, l1 = 0;
  int Lr1 = 0;
  if (sact) {
    if constexpr (SH) {  // a cascade row sees the prefix; a chunk tile's row (mode 6) is causal: [0, pos + 1)
      const int t = rflag[b * R + srow];
      Lr1 = t >= 0 ? min(Lmax, pos[t] + 1) : 0;
    } else {
      const int r = srow / GQ;
      Lr1 = r < nrows ? pos[r0 + r] + 1 : 0;
    }
  }
  const float scale2 = p.scale * 1.44269504089f;
  int buf = 0;

  for (int g0 = p0; g0 < p1; g0 += GK) {
    int bt[4];
    for (int j = 0; j < 4; j++) bt[j] = block_table[mb0 + min(g0 + j * PG, p1 - 1) / PG];
    if constexpr (KV8) {  // the group's 128 key and value scales, one coalesced load (the last half's reads are done)
      const int j = tid % GK;
      const size_t si = (size_t)kh * p.ns + (size_t)bt[j / PG] * PG + j % PG;
      if (tid < GK) ksh[j] = Ks[si];
      else vsh[j] = Vs[si];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int h = 0; h < 2; h++) {
      const int base = g0 + h * BK;
      if (base >= p1) break;
      // S^T for the half's 64 keys: one 64x48x256 matmul when its two pages are physically contiguous (the
      // allocator hands out ascending runs), else one per page; the result goes straight to Ssh.
      if (base + BK <= p1 && bt[2 * h + 1] == bt[2 * h] + 1) {
        auto k2 = kpages(bt[2 * h], 2);
        auto d2 = sdst(0, BK);
        qk2.run(k2, tQ, s2);
        s2.store(d2);
      } else {
        auto ka = kpages(bt[2 * h], 1), kb = kpages(bt[2 * h + 1], 1);
        auto da = sdst(0, PG), db = sdst(PG, PG);
        qk.run(ka, tQ, s0);
        s0.store(da);
        qk.run(kb, tQ, s0);
        s0.store(db);
      }
      if (sg == 0 && lane == 0) need_rescale[buf] = 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sact) {
        const int lim = min(Lr1, p1) - base;
        float v[KPL];
        float bm = -INFINITY;
        const float rs = scale2 * (IQ ? qsh[srow] : 1.f);
        for (int i = 0; i < KPL; i++) {
          const int k = scol + i * LPR;
          float x = float(Ssh[k][srow]) * rs;
          if constexpr (KV8) x *= ksh[h * BK + k];
          bool vis = k < lim;
          if constexpr (MASK)
            vis = vis && ((((device const uchar*)kbeg)[(size_t)b * p.qtiles + base + k] >> (srow / GQ)) & 1);
          v[i] = vis ? x : -INFINITY;
          bm = max(bm, v[i]);
        }
        for (int o = 1; o < LPR; o <<= 1) bm = max(bm, simd_shuffle_xor(bm, (ushort)o));
        float f = 1.f;
        if (bm > m1 + 8.f) {  // new reference max (always on the first visible block)
          f = m1 == -INFINITY ? 0.f : exp2(m1 - bm);
          if (l1 > 0 && scol == 0) need_rescale[buf] = 1;
          m1 = bm;
        }
        float sum = 0;
        for (int i = 0; i < KPL; i++) {
          const int k = scol + i * LPR;
          const float e = v[i] == -INFINITY ? 0.f : exp2(v[i] - m1);
          sum += e;
          Psh[buf][srow][k] = bfloat(KV8 ? e * vsh[h * BK + k] : e);
        }
        for (int o = 1; o < LPR; o <<= 1) sum += simd_shuffle_xor(sum, (ushort)o);
        l1 = l1 * f + sum;
        if (scol == 0) fac[srow] = f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (need_rescale[buf])
        for (uint16_t i = 0; i < O.get_capacity(); ++i)
          if (O.is_valid_element(i)) O[i] *= fac[O.get_multidimensional_index(i)[1]];
      auto pa = ppage(buf, 0), pb = ppage(buf, 1);
      auto va = vpage(bt[2 * h]), vb = vpage(bt[2 * h + 1]);
      pv.run(pa, va, O);
      pv.run(pb, vb, O);
      buf ^= 1;
    }
  }

  const int stride = D + 2;
  // The partial slot of packed row `row`, or nullptr when the row is padding.
  auto out_row = [&](int row) -> device float* {
    if constexpr (SH) {
      const int t = rflag[b * R + row];
      if (t < 0) return nullptr;
      return part + (((size_t)t * p.Hq + kh * GQ + seq_len[b * R + row]) * p.splits + p.own_splits + z) * stride;
    } else {
      const int r = row / GQ;
      if (r >= nrows) return nullptr;
      return part + (((size_t)(r0 + r) * p.Hq + kh * GQ + row % GQ) * p.splits + z) * stride;
    }
  };
  for (uint16_t i = 0; i < O.get_capacity(); ++i) {
    if (!O.is_valid_element(i)) continue;
    auto ix = O.get_multidimensional_index(i);
    const int d = ix[0], row = ix[1];
    device float* out = out_row(row);
    if (out) out[d] = O[i];
  }
  if (sact && scol == 0) {
    device float* out = out_row(srow);
    if (out) {
      out[D] = m1 == -INFINITY ? -INFINITY : m1 * 0.69314718056f;
      out[D + 1] = l1;
    }
  }
}

#define ATTN_FN(KV8, IQ, NAME)                                                                                  \
  template [[host_name(NAME)]] kernel void attn_fn<8, KV8, 12, IQ>(device const void*, device const void*,         \
      device const void*, device const int*, device const int*, device const int*, device float*, constant GqaArgs&, \
      device const int*, device const int*, device const float*, device const float*, device const int*,          \
      device const float*, uint3, uint, uint);
ATTN_FN(false, false, "attn_fn_g12") ATTN_FN(true, false, "attn_fn_g12_kv8") ATTN_FN(true, true, "attn_fn_g12_kv8_iq")
#define ATTN_FN_M(KV8, IQ, NAME)                                                                                \
  template [[host_name(NAME)]] kernel void attn_fn<8, KV8, 12, IQ, 48, false, true>(device const void*,          \
      device const void*, device const void*, device const int*, device const int*, device const int*, device float*, \
      constant GqaArgs&, device const int*, device const int*, device const float*, device const float*,          \
      device const int*, device const float*, uint3, uint, uint);
ATTN_FN_M(false, false, "attn_fn_g12_m") ATTN_FN_M(true, false, "attn_fn_g12_kv8_m") ATTN_FN_M(true, true, "attn_fn_g12_kv8_iq_m")

// ---------------------------------------------------------------------------------------------------------------------
// QSA: Qwen Sparse Attention (Flash-Next's attention layers; semantics from mlx_vlm 0.7.8 qwen4_exp and oMLX's vendored
// copy, the study log 2026-11-02). Per attention layer an indexer projects the layer input to Hi = 4 query heads and one
// key head of Di = 128 (index_qk_proj). Queries: zero-centred RMS norm (q_layernorm), then the layer's partial rope.
// Keys: the raw (un-normed) key of every token is cached; each complete group of ratio = 4 tokens is pooled: the mean
// of its 4 raw keys (fp32, rounded to bf16), k_layernorm, rope at the group's first position. A query at position p
// scores the nb = (p + 1) / 4 complete groups: s_j = sum_h relu(q_h . k_j) (fp32), keeps the top K = 512 groups, and
// attends to their 2,048 tokens plus the incomplete tail [4 nb, p]. When nb <= K every group is selected: plain causal
// attention (so contexts <= 2,051 tokens are dense). No sinks, no sliding window, no per-head weights.
// Storage, per attention layer, paged like the KV (32-token blocks, the sequence's block table): raw keys [blk][32][Di]
// and pooled keys [blk][8][Di], bf16, in byte regions of the KV pool's extra planes (Qwen35::qsa_loc): raw_blk /
// pool_blk are the bf16 elements between consecutive KV blocks of one region.
struct QsaArgs {
  int T, Hi, Di, rot, rpair, rden;
  float theta, eps;
  int max_blocks, ratio, raw_blk, pool_blk, stride;  // stride: the index projection's row length ((Hi + 1) * Di)
};

inline float qsa_tg_sum(float v, threadgroup float* sc, uint sg, uint lane, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) sc[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0;
  for (uint i = 0; i < nsg; i++) s += sc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return s;
}

// Rope one value of a Di-wide head held in xs (threadgroup) at rope positions (rp0, rp1, rp2): the same partial,
// interleaved-M-RoPE rotation as attn_prep (pairs (i, i + rpair) for i < rot / 2).
inline float qsa_rope(threadgroup const float* xs, uint d, int3 rp, constant QsaArgs& p) {
  float val = xs[d];
  const int rpair = p.rpair;
  if (d < (uint)(2 * rpair) && (int)(d % rpair) < p.rot / 2) {
    const int i = d % rpair;
    const float freq = pow(p.theta, -2.f * i / p.rden);
    const int c = i % 3;
    const float ang = float(c == 0 ? rp.x : c == 1 ? rp.y : rp.z) * freq, cs = cos(ang), sn = sin(ang);
    val = d < (uint)rpair ? xs[d] * cs - xs[d + rpair] * sn : xs[d] * cs + xs[d - rpair] * sn;
  }
  return val;
}

// Index queries (normed, roped) into iq [T, Hi, Di] (float, bf16-rounded); the raw key of every row into the cache;
// rows that complete a group (position % ratio == ratio - 1) also pool it. The group's earlier raw keys come from this
// step's projection rows when they are in the step, else from the cache. grid (Hi + 1, T), Di threads.
kernel void qsa_prep(device const bfloat* ip [[buffer(0)]], device const bfloat* qn [[buffer(1)]],
                     device const bfloat* kn [[buffer(2)]], device float* iq [[buffer(3)]],
                     device bfloat* raw [[buffer(4)]], device bfloat* pooled [[buffer(5)]],
                     device const int* block_table [[buffer(6)]], device const int* pos [[buffer(7)]],
                     device const int* tseq [[buffer(8)]], device const int* rpos [[buffer(9)]],
                     device const int* seq_start [[buffer(10)]], constant QsaArgs& p [[buffer(11)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sc[8];
  threadgroup float xs[256];
  const int h = tg.x, t = tg.y, nsg = (p.Di + 31) / 32;
  device const bfloat* row = ip + (size_t)t * p.stride;
  const int3 rp = int3(rpos[t * 3], rpos[t * 3 + 1], rpos[t * 3 + 2]);
  if (h < p.Hi) {
    const float x = float(row[h * p.Di + d]);
    const float r = rsqrt(qsa_tg_sum(x * x, sc, sg, lane, nsg) / p.Di + p.eps);
    xs[d] = float(bfloat(x * r * (1.f + float(qn[d]))));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    iq[((size_t)t * p.Hi + h) * p.Di + d] = float(bfloat(qsa_rope(xs, d, rp, p)));
    return;
  }
  const int ps = pos[t], sq = tseq[t], off = ps % 32;
  const int blk = block_table[sq * p.max_blocks + ps / 32];
  device bfloat* rk = raw + (size_t)blk * p.raw_blk;
  const bfloat k = row[p.Hi * p.Di + d];
  rk[off * p.Di + d] = k;
  if ((ps + 1) % p.ratio != 0) return;  // uniform per threadgroup
  const int r0 = seq_start[sq];         // this step's first row of the sequence (positions are consecutive)
  float s = 0;
  for (int j = p.ratio - 1; j >= 0; j--) {  // oldest first: ((k0 + k1) + k2) + k3
    const int back = j;
    float v;
    if (back == 0) v = float(k);
    else if (t - back >= r0) v = float(ip[(size_t)(t - back) * p.stride + p.Hi * p.Di + d]);
    else v = float(rk[(off - back) * p.Di + d]);
    s += v;
  }
  const float m = float(bfloat(s / p.ratio));
  const float r = rsqrt(qsa_tg_sum(m * m, sc, sg, lane, nsg) / p.Di + p.eps);
  xs[d] = float(bfloat(m * r * (1.f + float(kn[d]))));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int back = p.ratio - 1;  // rope at the group's first position
  const float v = qsa_rope(xs, d, rp - int3(back), p);
  pooled[(size_t)blk * p.pool_blk + (off / p.ratio) * p.Di + d] = bfloat(v);
}

// Scores of sparse rows rows[row0 + y] against their complete groups: sc[y, j] = sum_h relu(q_h . k_j) for
// j < (pos + 1) / ratio. grid (ceil(max nb / 256), rows in this group), 256 threads (one group per thread).
struct QsaScoreArgs { int Hi, Di, max_blocks, ratio, pool_blk, row0, nsc; };
kernel void qsa_score(device const float* iq [[buffer(0)]], device const bfloat* pooled [[buffer(1)]],
                      device const int* block_table [[buffer(2)]], device const int* pos [[buffer(3)]],
                      device const int* tseq [[buffer(4)]], device const int* rows [[buffer(5)]],
                      device float* sc [[buffer(6)]], constant QsaScoreArgs& p [[buffer(7)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float4 q[4 * 128 / 4];
  const int y = tg.y, t = rows[p.row0 + y];
  const int nb = (pos[t] + 1) / p.ratio, n4 = p.Hi * p.Di / 4;
  for (int i = tid; i < n4; i += 256) q[i] = ((device const float4*)(iq + (size_t)t * p.Hi * p.Di))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int j = tg.x * 256 + tid;
  if (j >= nb) return;
  const int per = 32 / p.ratio;
  const int blk = block_table[tseq[t] * p.max_blocks + j / per];
  device const bfloat4* k = (device const bfloat4*)(pooled + (size_t)blk * p.pool_blk + (j % per) * p.Di);
  const int d4 = p.Di / 4;
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (int i = 0; i < d4; i++) {
    const float4 kv = float4(k[i]);
    s0 += dot(kv, q[i]);
    s1 += dot(kv, q[d4 + i]);
    s2 += dot(kv, q[2 * d4 + i]);
    s3 += dot(kv, q[3 * d4 + i]);
  }
  // relu per head, summed; + 0.f maps -0 to +0 so the bit pattern orders like the value (all scores are >= 0)
  const float s = max(s0, 0.f) + max(s1, 0.f) + max(s2, 0.f) + max(s3, 0.f) + 0.f;
  sc[(size_t)y * p.nsc + j] = s;
}

// Top-K groups of each sparse row from its scores (all >= 0, so their bit patterns order like the values). Each of
// the 1024 threads keeps the top 16 bits of its scores in 16-bit registers (indices tid + i * 1024, at most QSA_VPT
// each). Phase 1: a binary search on those coarse keys (16 block-wide counts, no atomics: positive floats share a
// few exponent bytes, so a radix histogram serialised on 2-3 bins, 0.11 ms per layer at 128k) finds the bucket c of
// the K-th largest. Phase 2: the same search on the low 15 bits among that bucket's scores (read from memory; a
// bucket spans 1/128 of a binade, so usually a handful) gives the exact K-th largest, thr. The groups above thr and
// the lowest-indexed ties at thr are written in ascending order to sel[t, 0..K): each thread compacts a contiguous
// slice of the indices, placed by one block-wide scan. grid (rows in this group), 1024 threads.
struct QsaSelArgs { int K, ratio, row0, nsc; };
// Block-wide sum; red is double-buffered by the caller's parity (one barrier per call: a buffer is rewritten only
// after the next call's barrier, by which time every thread has read it).
inline uint qsa_block_sum(uint v, threadgroup uint (*red)[32], thread uint& par, uint sg, uint lane) {
  v = simd_sum(v);
  if (lane == 0) red[par][sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint s = simd_sum(red[par][lane]);
  par ^= 1;
  return s;
}
// QSA_VPT: register slots per thread (the host picks the smallest of 2 / 8 / 32 / 64 covering the step's groups).
template <int QSA_VPT>
kernel void qsa_select(device const float* scf [[buffer(0)]], device const int* rows [[buffer(1)]],
                       device const int* pos [[buffer(2)]], device int* sel [[buffer(3)]],
                       constant QsaSelArgs& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 1024;
  threadgroup uint red[2][32], wg[32], we[32];
  uint par = 0;
  const int y = tg, t = rows[p.row0 + y];
  const int nb = (pos[t] + 1) / p.ratio, vpt = min((nb + NT - 1) / NT, QSA_VPT);
  device const uint* u = (device const uint*)(scf + (size_t)y * p.nsc);
  ushort v[QSA_VPT];  // coarse keys (score bits >> 15)
  int nv = 0;         // this thread's real entries: indices tid + i * NT < nb for i < nv
#pragma unroll
  for (int i = 0; i < QSA_VPT; i++) {
    const int j = (int)tid + i * NT;
    v[i] = i < vpt && j < nb ? ushort(u[j] >> 15) : ushort(0);
    nv += i < vpt && j < nb;
  }
  // phase 1: c = the largest coarse key with count(key >= c) >= K
  uint c = 0;
  for (int bit = 15; bit >= 0; bit--) {
    const uint cand = c | (1u << bit);
    uint n = 0;
#pragma unroll
    for (int i = 0; i < QSA_VPT; i++) n += i < nv && v[i] >= cand;
    if (qsa_block_sum(n, red, par, sg, lane) >= (uint)p.K) c = cand;
  }
  uint above = 0;
#pragma unroll
  for (int i = 0; i < QSA_VPT; i++) above += i < nv && v[i] > c;
  const uint need = p.K - qsa_block_sum(above, red, par, sg, lane);  // to take from bucket c (>= 1)
  // phase 2: thr = the largest full value in bucket c with count(bucket values >= thr) >= need
  uint thr = c << 15;
  for (int bit = 14; bit >= 0; bit--) {
    const uint cand = thr | (1u << bit);
    uint n = 0;
#pragma unroll
    for (int i = 0; i < QSA_VPT; i++)
      if (i < nv && v[i] == c) n += u[tid + i * NT] >= cand;
    if (qsa_block_sum(n, red, par, sg, lane) >= need) thr = cand;
  }
  uint gt = 0;
#pragma unroll
  for (int i = 0; i < QSA_VPT; i++)
    if (i < nv) gt += v[i] > c || (v[i] == c && u[tid + i * NT] > thr);
  const uint keq = p.K - qsa_block_sum(gt, red, par, sg, lane);  // ties at thr to take, lowest indices first
  // ordered compaction over contiguous slices [j0, j1) (scores re-read: thr compares the full value)
  const int C = (nb + NT - 1) / NT, j0 = min(nb, (int)tid * C), j1 = min(nb, j0 + C);
  uint ng = 0, ne = 0;
  for (int j = j0; j < j1; j++) { const uint x = u[j]; ng += x > thr; ne += x == thr; }
  const uint gb = simd_prefix_exclusive_sum(ng), eb = simd_prefix_exclusive_sum(ne);
  if (lane == 31) { wg[sg] = gb + ng; we[sg] = eb + ne; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint gbefore = gb, ebefore = eb;
  for (uint k = 0; k < sg; k++) { gbefore += wg[k]; ebefore += we[k]; }
  uint o = gbefore + min(ebefore, keq), etake = ebefore >= keq ? 0 : min(ne, keq - ebefore);
  device int* out = sel + (size_t)t * p.K;
  for (int j = j0; j < j1; j++) {
    const uint x = u[j];
    if (x > thr) out[o++] = j;
    else if (x == thr && etake) { out[o++] = j; etake--; }
  }
}
#define QSA_SEL(V)                                                                                              \
  template [[host_name("qsa_select_v" #V)]] kernel void qsa_select<V>(device const float*, device const int*,     \
      device const int*, device int*, constant QsaSelArgs&, uint, uint, uint, uint);
QSA_SEL(2) QSA_SEL(8) QSA_SEL(32) QSA_SEL(64)

// Sparse attention of rows[y] over its selected groups (sel[t]) plus its incomplete tail: attn_gqa_row with the
// positions taken from the selection list (ascending: K groups of `ratio` consecutive tokens, then the tail), split
// p.splits ways over that list. Writes the same partials as attn_gqa_row (attn_combine merges them).
// grid (Hkv, sparse rows, splits), 128 threads.
struct QsaAttnArgs { int K, ratio; };
template <int GQ, int D, bool KV8>
kernel void attn_qsa(device const bfloat* q [[buffer(0)]], device const void* Kc [[buffer(1)]],
                     device const void* Vc [[buffer(2)]], device const int* block_table [[buffer(3)]],
                     device const int* pos [[buffer(4)]], device const int* tseq [[buffer(5)]],
                     device float* part [[buffer(6)]], constant GqaArgs& p [[buffer(7)]],
                     device const int* rows [[buffer(8)]], device const float* Ks [[buffer(9)]],
                     device const float* Vs [[buffer(10)]], device const int* sel [[buffer(11)]],
                     constant QsaAttnArgs& qa [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BLK = 32, DPL = D / 32, GSG = 4;
  threadgroup bfloat qs[GQ][D];
  threadgroup float tm[GSG][GQ], tl[GSG][GQ];
  threadgroup float tacc[GQ][D];
  const int kh = tg.x, t = rows[tg.y], z = tg.z;
  const int L = pos[t] + 1, sq = tseq[t];
  const int nb = L / qa.ratio, nsel = qa.K * qa.ratio, ntok = nsel + (L - nb * qa.ratio);
  const int ch = ((ntok + p.splits - 1) / p.splits + BLK - 1) / BLK * BLK;
  const int p0 = z * ch, p1 = min(ntok, p0 + ch);
  device const int* sl = sel + (size_t)t * qa.K;
  for (int i = tid; i < GQ * D; i += GSG * 32)
    qs[i / D][i % D] = bfloat(float(q[((size_t)t * p.Hq + kh * GQ + i / D) * D + i % D]) * p.scale);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[GQ][DPL], m[GQ], l[GQ];
  for (int j = 0; j < GQ; j++) {
    m[j] = -INFINITY;
    l[j] = 0;
    for (int i = 0; i < DPL; i++) acc[j][i] = 0;
  }
  for (int base = p0 + (int)sg * BLK; base < p1; base += GSG * BLK) {
    const int idx = base + (int)lane;
    const bool valid = idx < p1;
    int slot = 0;
    if (valid) {
      const int tp = idx < nsel ? sl[idx / qa.ratio] * qa.ratio + idx % qa.ratio : nb * qa.ratio + (idx - nsel);
      slot = block_table[sq * p.max_blocks + tp / BLK] * BLK + tp % BLK;
    }
    const size_t si = (size_t)kh * p.ns + slot;
    float sc[GQ];
    for (int j = 0; j < GQ; j++) sc[j] = 0;
    float vsc = 1.f;
    if (valid) {
      for (int d4 = 0; d4 < D / 4; d4++) {
        const float4 k4 = KV8 ? float4(((device const char4*)Kc)[si * (D / 4) + d4])
                              : float4(((device const bfloat4*)Kc)[si * (D / 4) + d4]);
        for (int j = 0; j < GQ; j++)
          sc[j] += k4.x * float(qs[j][4 * d4]) + k4.y * float(qs[j][4 * d4 + 1]) + k4.z * float(qs[j][4 * d4 + 2]) +
                   k4.w * float(qs[j][4 * d4 + 3]);
      }
      if (KV8) {
        const float ksc = Ks[si];
        vsc = Vs[si];
        for (int j = 0; j < GQ; j++) sc[j] *= ksc;
      }
    }
    float e[GQ];
    for (int j = 0; j < GQ; j++) {
      const float s = valid ? sc[j] : -INFINITY;
      const float mn = max(m[j], simd_max(s));
      const float corr = exp(m[j] - mn);
      e[j] = valid ? exp(s - mn) : 0.f;
      l[j] = l[j] * corr + simd_sum(e[j]);
      for (int i = 0; i < DPL; i++) acc[j][i] *= corr;
      m[j] = mn;
    }
    const int n = min(BLK, p1 - base);
    for (int r = 0; r < n; r++) {
      const int vslot = simd_shuffle(slot, (ushort)r);
      const float vs_r = KV8 ? simd_shuffle(vsc, (ushort)r) : 1.f;
      const size_t vo = ((size_t)kh * p.ns + vslot) * D + lane * DPL;
      float4 v0, v1;
      if (KV8) {
        v0 = float4(*(device const char4*)((device const char*)Vc + vo));
        v1 = float4(*(device const char4*)((device const char*)Vc + vo + 4));
      } else {
        v0 = float4(*(device const bfloat4*)((device const bfloat*)Vc + vo));
        v1 = float4(*(device const bfloat4*)((device const bfloat*)Vc + vo + 4));
      }
      for (int j = 0; j < GQ; j++) {
        const float w = simd_shuffle(e[j], (ushort)r) * vs_r;
        acc[j][0] += w * v0.x; acc[j][1] += w * v0.y; acc[j][2] += w * v0.z; acc[j][3] += w * v0.w;
        acc[j][4] += w * v1.x; acc[j][5] += w * v1.y; acc[j][6] += w * v1.z; acc[j][7] += w * v1.w;
      }
    }
  }
  // Merge the 4 simdgroups: rescale to the common max in registers, then sum into tacc one simdgroup at a time.
  for (int j = 0; j < GQ; j++)
    if (lane == 0) { tm[sg][j] = m[j]; tl[sg][j] = l[j]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < GQ; j++) {
    float M = -INFINITY;
    for (int s = 0; s < GSG; s++) M = max(M, tm[s][j]);
    const float f = m[j] == -INFINITY ? 0.f : exp(m[j] - M);
    for (int i = 0; i < DPL; i++) acc[j][i] *= f;
  }
  for (int s0 = 0; s0 < GSG; s0++) {
    if ((int)sg == s0)
      for (int j = 0; j < GQ; j++)
        for (int i = 0; i < DPL; i++) {
          threadgroup float& dd = tacc[j][lane * DPL + i];
          dd = s0 == 0 ? acc[j][i] : dd + acc[j][i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int stride = D + 2;
  for (int j = 0; j < GQ; j++) {
    float M = -INFINITY;
    for (int s = 0; s < GSG; s++) M = max(M, tm[s][j]);
    device float* out = part + (((size_t)t * p.Hq + kh * GQ + j) * p.splits + z) * stride;
    for (int d = tid; d < D; d += GSG * 32) out[d] = tacc[j][d];
    if (tid == 0) {
      float den = 0;
      for (int s = 0; s < GSG; s++) den += tm[s][j] == -INFINITY ? 0.f : tl[s][j] * exp(tm[s][j] - M);
      out[D] = M;
      out[D + 1] = den;
    }
  }
}
#define ATTN_QSA(KV8, NAME)                                                                                       \
  template [[host_name(NAME)]] kernel void attn_qsa<12, 256, KV8>(device const bfloat*, device const void*,       \
      device const void*, device const int*, device const int*, device const int*, device float*, constant GqaArgs&, \
      device const int*, device const float*, device const float*, device const int*, constant QsaAttnArgs&, uint3, \
      uint, uint, uint);
ATTN_QSA(false, "attn_qsa_g12_d256") ATTN_QSA(true, "attn_qsa_g12_d256_kv8")

// QSA decode / verify rows on the tensor units: copy each sparse row's selected K/V (and KV8 scales) into a contiguous
// scratch pool (row i owns pages [i P, (i + 1) P)), so attn_fn runs on it unchanged as one 1-row "sequence" per sparse
// row (seq_start2[i] = t, pos2[t] = ntok - 1, flag2[t] = 4, an identity block table of P pages per row).
// grid threads (D / 4, P * 32, Hkv * nsp).
struct QsaGatherArgs { int K, ratio, max_blocks, ns, ns2, P, D, nsp; };
template <bool KV8>
kernel void qsa_gather(device const void* Kc [[buffer(0)]], device const void* Vc [[buffer(1)]],
                       device const float* Ks [[buffer(2)]], device const float* Vs [[buffer(3)]],
                       device void* Kg [[buffer(4)]], device void* Vg [[buffer(5)]],
                       device float* Kgs [[buffer(6)]], device float* Vgs [[buffer(7)]],
                       device const int* rows [[buffer(8)]], device const int* sel [[buffer(9)]],
                       device const int* block_table [[buffer(10)]], device const int* pos [[buffer(11)]],
                       device const int* tseq [[buffer(12)]], device int* pos2 [[buffer(13)]],
                       device int* flag2 [[buffer(14)]], device int* start2 [[buffer(15)]],
                       device int* len2 [[buffer(16)]], constant QsaGatherArgs& p [[buffer(17)]],
                       uint3 gid [[thread_position_in_grid]]) {
  const int c = gid.x, idx = gid.y, kh = gid.z / p.nsp, i = gid.z % p.nsp;
  const int t = rows[i], L = pos[t] + 1, nb = L / p.ratio, nsel = p.K * p.ratio, ntok = nsel + (L - nb * p.ratio);
  if (c == 0 && idx == 0 && kh == 0) { pos2[t] = ntok - 1; flag2[t] = 4; start2[i] = t; len2[i] = 1; }
  if (idx >= ntok || c >= p.D / 4) return;
  const int tp = idx < nsel ? sel[(size_t)t * p.K + idx / p.ratio] * p.ratio + idx % p.ratio : nb * p.ratio + (idx - nsel);
  const size_t si = (size_t)kh * p.ns + (size_t)block_table[tseq[t] * p.max_blocks + tp / 32] * 32 + tp % 32;
  const size_t di = (size_t)kh * p.ns2 + (size_t)i * p.P * 32 + idx;
  if (KV8) {
    ((device char4*)Kg)[di * (p.D / 4) + c] = ((device const char4*)Kc)[si * (p.D / 4) + c];
    ((device char4*)Vg)[di * (p.D / 4) + c] = ((device const char4*)Vc)[si * (p.D / 4) + c];
    if (c == 0) { Kgs[di] = Ks[si]; Vgs[di] = Vs[si]; }
  } else {
    ((device bfloat4*)Kg)[di * (p.D / 4) + c] = ((device const bfloat4*)Kc)[si * (p.D / 4) + c];
    ((device bfloat4*)Vg)[di * (p.D / 4) + c] = ((device const bfloat4*)Vc)[si * (p.D / 4) + c];
  }
}
template [[host_name("qsa_gather")]] kernel void qsa_gather<false>(device const void*, device const void*,
    device const float*, device const float*, device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device int*,
    device int*, device int*, device int*, constant QsaGatherArgs&, uint3);
template [[host_name("qsa_gather_kv8")]] kernel void qsa_gather<true>(device const void*, device const void*,
    device const float*, device const float*, device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device int*,
    device int*, device int*, device int*, constant QsaGatherArgs&, uint3);

// QSA union tiles: up to 4 consecutive sparse rows of one sequence (a prefill chunk's or a verify block's) share one
// attn_fn tile over the union of their selections. qsa_union: one threadgroup per tile marks each row's 512 groups in a
// threadgroup map (4 row bits per group, nb <= 32768) and writes the union in ascending order as group | mask << 28
// (ulist[u, 0..ucnt[u])). grid (tiles), 1024 threads.
struct QsaUnionArgs { int K, ratio, cap; };  // cap: ulist row stride
kernel void qsa_union(device const int* sel [[buffer(0)]], device const int* ustart [[buffer(1)]],
                      device const int* ulen [[buffer(2)]], device const int* pos [[buffer(3)]],
                      device int* ulist [[buffer(4)]], device int* ucnt [[buffer(5)]],
                      constant QsaUnionArgs& p [[buffer(6)]], uint u [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 1024, WORDS = 32768 / 8;  // 8 groups (4 bits each) per word
  threadgroup atomic_uint map[WORDS];
  threadgroup uint wc[32];
  const int t0 = ustart[u], n = ulen[u];
  const int nb = (pos[t0 + n - 1] + 1) / p.ratio, words = (nb + 7) / 8;
  for (int i = tid; i < words; i += NT) atomic_store_explicit(&map[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < n * p.K; i += NT) {
    const int r = i / p.K, g = sel[(size_t)(t0 + r) * p.K + i % p.K];
    atomic_fetch_or_explicit(&map[g / 8], 1u << ((g % 8) * 4 + r), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int C = (words + NT - 1) / NT, w0 = min(words, (int)tid * C), w1 = min(words, w0 + C);
  uint cnt = 0;
  for (int w = w0; w < w1; w++) {
    const uint x = atomic_load_explicit(&map[w], memory_order_relaxed);
    for (int k = 0; k < 8; k++) cnt += ((x >> (4 * k)) & 15u) != 0;
  }
  const uint pre = simd_prefix_exclusive_sum(cnt);
  if (lane == 31) wc[sg] = pre + cnt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint o = pre, tot = 0;
  for (uint k = 0; k < 32; k++) { if (k < sg) o += wc[k]; tot += wc[k]; }
  device int* out = ulist + (size_t)u * p.cap;
  for (int w = w0; w < w1; w++) {
    const uint x = atomic_load_explicit(&map[w], memory_order_relaxed);
    for (int k = 0; k < 8; k++)
      if (const uint m = (x >> (4 * k)) & 15u) out[o++] = (w * 8 + k) | int(m << 28);
  }
  if (tid == 0) ucnt[u] = tot;
}

// Gather a union tile's keys: the union's groups (4 tokens each, the group's row mask), then the token span
// [4 nb_0, p_last] of the rows' tails (mask: the rows whose tail holds the token). A token may appear twice (a tail of
// one row and a selected group of a later row) with disjoint masks. Writes K/V (+ KV8 scales) to the tile's pages,
// the mask bytes, and the virtual batch (every row of the tile: pos2 = ntok - 1; flag2 = 4 on its first row).
// grid threads (D / 4, cap tokens, Hkv * tiles).
struct QsaUGatherArgs { int ratio, max_blocks, ns, ns2, P, D, nu, cap; };
template <bool KV8>
kernel void qsa_ugather(device const void* Kc [[buffer(0)]], device const void* Vc [[buffer(1)]],
                        device const float* Ks [[buffer(2)]], device const float* Vs [[buffer(3)]],
                        device void* Kg [[buffer(4)]], device void* Vg [[buffer(5)]],
                        device float* Kgs [[buffer(6)]], device float* Vgs [[buffer(7)]],
                        device const int* ustart [[buffer(8)]], device const int* ulen [[buffer(9)]],
                        device const int* ulist [[buffer(10)]], device const int* ucnt [[buffer(11)]],
                        device const int* block_table [[buffer(12)]], device const int* pos [[buffer(13)]],
                        device const int* tseq [[buffer(14)]], device int* pos2 [[buffer(15)]],
                        device int* flag2 [[buffer(16)]], device uchar* gmask [[buffer(17)]],
                        constant QsaUGatherArgs& p [[buffer(18)]], uint3 gid [[thread_position_in_grid]]) {
  const int c = gid.x, idx = gid.y, kh = gid.z / p.nu, u = gid.z % p.nu;
  const int t0 = ustart[u], n = ulen[u], nc = ucnt[u];
  const int nb0 = (pos[t0] + 1) / p.ratio, plast = pos[t0 + n - 1];
  const int ntok = 4 * nc + (plast - nb0 * p.ratio + 1);
  if (c == 0 && idx == 0 && kh == 0) {
    for (int r = 0; r < n; r++) pos2[t0 + r] = ntok - 1;
    flag2[t0] = 4;
  }
  if (idx >= ntok || c >= p.D / 4) return;
  int tp;
  uint m;
  if (idx < 4 * nc) {
    const int e = ulist[(size_t)u * p.cap + idx / 4];
    tp = (e & 0x0fffffff) * p.ratio + idx % 4;
    m = uint(e) >> 28;
  } else {
    tp = nb0 * p.ratio + (idx - 4 * nc);
    m = 0;
    for (int r = 0; r < n; r++) {  // row r's tail: [4 nb_r, p_r]
      const int pr = pos[t0 + r];
      m |= uint(tp >= (pr + 1) / p.ratio * p.ratio && tp <= pr) << r;
    }
  }
  const size_t si = (size_t)kh * p.ns + (size_t)block_table[tseq[t0] * p.max_blocks + tp / 32] * 32 + tp % 32;
  const size_t di = (size_t)kh * p.ns2 + (size_t)u * p.P * 32 + idx;
  if (c == 0 && kh == 0) gmask[(size_t)u * p.P * 32 + idx] = uchar(m);
  if (KV8) {
    ((device char4*)Kg)[di * (p.D / 4) + c] = ((device const char4*)Kc)[si * (p.D / 4) + c];
    ((device char4*)Vg)[di * (p.D / 4) + c] = ((device const char4*)Vc)[si * (p.D / 4) + c];
    if (c == 0) { Kgs[di] = Ks[si]; Vgs[di] = Vs[si]; }
  } else {
    ((device bfloat4*)Kg)[di * (p.D / 4) + c] = ((device const bfloat4*)Kc)[si * (p.D / 4) + c];
    ((device bfloat4*)Vg)[di * (p.D / 4) + c] = ((device const bfloat4*)Vc)[si * (p.D / 4) + c];
  }
}
template [[host_name("qsa_ugather")]] kernel void qsa_ugather<false>(device const void*, device const void*,
    device const float*, device const float*, device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device const int*,
    device const int*, device int*, device int*, device uchar*, constant QsaUGatherArgs&, uint3);
template [[host_name("qsa_ugather_kv8")]] kernel void qsa_ugather<true>(device const void*, device const void*,
    device const float*, device const float*, device void*, device void*, device float*, device float*,
    device const int*, device const int*, device const int*, device const int*, device const int*, device const int*,
    device const int*, device int*, device int*, device uchar*, constant QsaUGatherArgs&, uint3);

// ---------------------------------------------------------------------------------------------------------------------
// Flash-Next prefill kernels (2026-11-02 night): token-parallel GDN conv, the GDN recurrence with several value rows
// per simdgroup, and the MoE routing from bf16 router logits (the router as a GEMM). Each reproduces the arithmetic of
// the kernel it replaces (gdn_conv, gdn_step, moe_route).

struct FnpGdnArgs {
  int S, Hk, Hv, Dk, Dv;
  int conv_dim;
  int proj_stride;
  float eps;
  int verify;
  int sig;
};

// gdn_conv, one thread per (channel, row) instead of a sequential run per channel: row t (position j of its
// sequence) reads its 3 predecessors from the step's projection rows, or from the conv state for j < 3. The state
// update is gdn_conv_pf_state (after this kernel, so no row reads a state already overwritten). grid (conv_dim, T).
kernel void gdn_conv_pf(device const bfloat* proj [[buffer(0)]],
                        device const bfloat* conv_state [[buffer(1)]],
                        device const bfloat* w [[buffer(2)]],
                        device bfloat* out [[buffer(3)]],
                        device const int* slot [[buffer(4)]],
                        constant FnpGdnArgs& p [[buffer(5)]],
                        device const int* seq_start [[buffer(6)]],
                        device const int* tseq [[buffer(7)]],
                        device bfloat* tape_x [[buffer(8)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, t = gid.y;
  if (c >= p.conv_dim) return;
  const int b = tseq[t], j = t - seq_start[b];
  device const bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  float s[3];
  for (int k = 0; k < 3; k++) {  // s[k]: the input at position j - 3 + k
    const int jj = j - 3 + k;
    s[k] = jj >= 0 ? float(proj[(size_t)(t - 3 + k) * p.proj_stride + c]) : float(st[(size_t)(3 + jj) * p.conv_dim + c]);
  }
  device const bfloat* wc = w + (size_t)c * 4;
  const float x = proj[(size_t)t * p.proj_stride + c];
  if (p.verify) tape_x[(size_t)t * p.conv_dim + c] = bfloat(x);
  const float y = float(bfloat(float(wc[0]) * s[0] + float(wc[1]) * s[1] + float(wc[2]) * s[2] + float(wc[3]) * x));
  out[(size_t)t * p.conv_dim + c] = bfloat(y / (1.f + exp(-y)));
}

// The conv state after gdn_conv_pf: the last 3 inputs of each sequence (fewer rows than 3: shifted old state).
// grid (conv_dim, S).
kernel void gdn_conv_pf_state(device const bfloat* proj [[buffer(0)]],
                              device bfloat* conv_state [[buffer(1)]],
                              device const int* slot [[buffer(4)]],
                              constant FnpGdnArgs& p [[buffer(5)]],
                              device const int* seq_start [[buffer(6)]],
                              device const int* seq_len [[buffer(7)]],
                              device const int* spec [[buffer(9)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, b = gid.y;
  if (c >= p.conv_dim) return;
  if (p.verify && spec[b]) return;
  device bfloat* st = conv_state + (size_t)slot[b] * 3 * p.conv_dim;
  const int t0 = seq_start[b], n = seq_len[b];
  bfloat v[3];
  for (int k = 0; k < 3; k++) {  // new state k = input at position n - 3 + k
    const int jj = n - 3 + k;
    v[k] = jj >= 0 ? proj[(size_t)(t0 + jj) * p.proj_stride + c] : st[(size_t)(3 + jj) * p.conv_dim + c];
  }
  for (int k = 0; k < 3; k++) st[(size_t)k * p.conv_dim + c] = v[k];
}

// gdn_step with R value rows per simdgroup (R independent recurrences interleaved: more work in flight per token,
// one wave of simdgroups for a lone sequence) and the next token's inputs loaded before the current token's update.
// Per row the arithmetic is gdn_step's, in the same order. grid: threads (32, Dv / R, S * Hv), threadgroup (32, 4, 1).
template <int R>
kernel void gdn_step_r(device const bfloat* conv_out [[buffer(0)]],
                       device const float* g [[buffer(1)]],
                       device const float* beta [[buffer(2)]],
                       device float* state [[buffer(3)]],
                       device bfloat* y [[buffer(4)]],
                       device const int* slot [[buffer(5)]],
                       constant FnpGdnArgs& p [[buffer(6)]],
                       device const int* seq_start [[buffer(7)]],
                       device const int* seq_len [[buffer(8)]],
                       device const int* spec [[buffer(9)]],
                       uint3 gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const int n = gid.z, b = n / p.Hv, hv = n % p.Hv, hk = hv / (p.Hv / p.Hk);
  const int dv0 = gid.y * R;
  constexpr int NPT = 4;  // Dk / 32 for Dk = 128
  device float* st = state + (((size_t)slot[b] * p.Hv + hv) * p.Dv + dv0) * p.Dk;
  float s[R][NPT];
  for (int r = 0; r < R; r++)
    for (int i = 0; i < NPT; i++) s[r][i] = st[(size_t)r * p.Dk + NPT * lane + i];
  const int t0 = seq_start[b], nt = seq_len[b];
  const int qo = hk * p.Dk + NPT * lane, ko = p.Hk * p.Dk + hk * p.Dk + NPT * lane, vo = 2 * p.Hk * p.Dk + hv * p.Dv + dv0;
  float kk[NPT], qq[NPT], vv[R], gg = 0, bt = 0;
  auto load = [&](int t, thread float* k4, thread float* q4, thread float* v4, thread float& g1, thread float& b1) {
    device const bfloat* row = conv_out + (size_t)t * p.conv_dim;
    for (int i = 0; i < NPT; i++) { k4[i] = row[ko + i]; q4[i] = row[qo + i]; }
    for (int r = 0; r < R; r++) v4[r] = row[vo + r];
    g1 = g[t * p.Hv + hv];
    b1 = beta[t * p.Hv + hv];
  };
  if (nt > 0) load(t0, kk, qq, vv, gg, bt);
  for (int j = 0; j < nt; j++) {
    const int t = t0 + j;
    float nk[NPT], nq[NPT], nv[R], ng = 0, nb = 0;
    if (j + 1 < nt) load(t + 1, nk, nq, nv, ng, nb);
    float kv[R];
    for (int r = 0; r < R; r++) {
      kv[r] = 0;
      for (int i = 0; i < NPT; i++) {
        s[r][i] *= gg;
        kv[r] += s[r][i] * kk[i];
      }
    }
    for (int r = 0; r < R; r++) kv[r] = simd_sum(kv[r]);
    float o[R];
    for (int r = 0; r < R; r++) {
      const float delta = (vv[r] - kv[r]) * bt;
      o[r] = 0;
      for (int i = 0; i < NPT; i++) {
        s[r][i] += kk[i] * delta;
        o[r] += s[r][i] * qq[i];
      }
    }
    for (int r = 0; r < R; r++) o[r] = simd_sum(o[r]);
    if (lane < (uint)R) {
      float ov = o[0];
      for (int r = 1; r < R; r++) ov = lane == (uint)r ? o[r] : ov;
      y[((size_t)t * p.Hv + hv) * p.Dv + dv0 + lane] = bfloat(ov);
    }
    for (int i = 0; i < NPT; i++) { kk[i] = nk[i]; qq[i] = nq[i]; }
    for (int r = 0; r < R; r++) vv[r] = nv[r];
    gg = ng;
    bt = nb;
  }
  if (p.verify && spec[b]) return;
  for (int r = 0; r < R; r++)
    for (int i = 0; i < NPT; i++) st[(size_t)r * p.Dk + NPT * lane + i] = s[r][i];
}
#define GDN_STEP_R(R)                                                                                                  \
  template [[host_name("gdn_step_r" #R)]] kernel void gdn_step_r<R>(device const bfloat*, device const float*,         \
      device const float*, device float*, device bfloat*, device const int*, constant FnpGdnArgs&, device const int*,     \
      device const int*, device const int*, uint3, uint);
GDN_STEP_R(2)
GDN_STEP_R(4)
GDN_STEP_R(8)

// moe_route over bf16 router logits (the router as a bf16 GEMM, rounded to bf16 as the reference's nn.Linear does).
// Softmax over E (fp32), top-k, renormalised. One simdgroup per token, E <= 512.
struct FnpRouteArgs { int E, k, stride, norm; };
kernel void moe_route_bf(device const bfloat* logits [[buffer(0)]],
                         device int* idx [[buffer(1)]],
                         device float* wt [[buffer(2)]],
                         device float* sgate [[buffer(3)]],
                         constant FnpRouteArgs& p [[buffer(4)]],
                         uint t [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  device const bfloat* l = logits + (size_t)t * p.stride;
  constexpr int MAXV = 16;
  float v[MAXV];
  const int per = (p.E + 31) / 32;
  float mx = -INFINITY;
  for (int i = 0; i < per; i++) {
    const int c = i * 32 + lane;
    v[i] = c < p.E ? float(l[c]) : -INFINITY;
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
    if ((who & 31) == (int)lane) v[who / 32] = -1;
    if (lane == 0) { idx[t * p.k + j] = who; wt[t * p.k + j] = m / sum; }
    tot += m / sum;
  }
  if (lane == 0) {
    if (p.norm)
      for (int j = 0; j < p.k; j++) wt[t * p.k + j] /= tot;
    sgate[t] = 1.f / (1.f + exp(-float(l[p.E])));
  }
}
