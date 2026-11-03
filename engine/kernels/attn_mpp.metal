// Verify attention on the tensor units with whole-tile matmul2d ops (NAX_MPP=8). One threadgroup per (kv head,
// sequence, split of positions) handles all of the sequence's query rows for that kv head at once: up to 8 verify
// rows x 6 GQA heads = 48 packed rows. Per 128-key group (four 32-token KV pages):
//   S^T = K Q^T   one 128x48x256 matmul when the four pages are physically contiguous (the scheduler arranges that
//                 where it can), else one 32x48x256 per page; K read in place through strided tensors
//   softmax       online, two 64-key halves in threadgroup memory (one simdgroup per row band, two keys per lane)
//   O += P V      48x256x32 matmuls per page accumulating into a cooperative 48x256 fp32 tile
// attn_paged_nax does the same with 16x16 fragment ops chained through two barriers per 32 keys and runs the math
// at ~16 TFLOPS on a long single sequence; whole-tile ops let the hardware schedule the MMAs. Writes the same
// unnormalised partials [t, Hq, splits, D + 2] (acc, m in natural log units, l) for attn_combine.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using DT2 = dextents<int32_t, 2>;

struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles, own_splits, ns, win, gated; };

// Pack each multi-row sequence's queries per kv head (i = r * 6 + j: token row r, head j of the group), zero past
// the sequence's rows. grid: threadgroups (Hkv * 48, S), 256 threads.
template <int R>
kernel void q_pack(device const bfloat* q [[buffer(0)]], device bfloat* qp [[buffer(1)]],
                   device const int* seq_start [[buffer(2)]], device const int* seq_len [[buffer(3)]],
                   constant GqaArgs& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint d [[thread_index_in_threadgroup]]) {
  const int kh = tg.x / R, i = tg.x % R, b = tg.y, r = i / 6, j = i % 6;
  const int r0 = seq_start[b], n = min(seq_len[b], R / 6);
  // Transposed: qp[(b * Hkv + kh)][d][48], the right operand of S^T = K Q^T (K is the left operand in its
  // natural row-per-key layout; as a transposed right operand it streamed at ~40% of the bandwidth P V reaches).
  qp[(((size_t)b * p.Hkv + kh) * 256 + d) * R + i] =
      r < n ? q[((size_t)(r0 + r) * p.Hq + kh * 6 + j) * 256 + d] : bfloat(0);
}
// IQ (8-bit attention, with KV8): the same packing as int8 with one scale per packed row (absmax / 127 over D),
// qs[(b * Hkv + kh) * R + i]. S^T = K8 Q8^T then runs on the integer path and is rescaled by key and query scale.
template <int R>
kernel void q_pack_i8(device const bfloat* q [[buffer(0)]], device int8_t* qp [[buffer(1)]],
                      device const int* seq_start [[buffer(2)]], device const int* seq_len [[buffer(3)]],
                      constant GqaArgs& p [[buffer(4)]], device float* qs [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float mx[8];
  const int kh = tg.x / R, i = tg.x % R, b = tg.y, r = i / 6, j = i % 6;
  const int r0 = seq_start[b], n = min(seq_len[b], R / 6);
  const float v = r < n ? float(q[((size_t)(r0 + r) * p.Hq + kh * 6 + j) * 256 + d]) : 0.f;
  const float m = simd_max(fabs(v));
  if (lane == 0) mx[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a = 0;
  for (int k = 0; k < 8; k++) a = max(a, mx[k]);
  const float sc = a > 0 ? a / 127.f : 0.f;
  qp[(((size_t)b * p.Hkv + kh) * 256 + d) * R + i] = int8_t(sc > 0 ? clamp(rint(v / sc), -127.f, 127.f) : 0.f);
  if (d == 0) qs[((size_t)b * p.Hkv + kh) * R + i] = sc;
}
template [[host_name("q_pack_i8_48")]] kernel void q_pack_i8<48>(device const bfloat*, device int8_t*, device const int*,
    device const int*, constant GqaArgs&, device float*, uint2, uint, uint, uint);
template [[host_name("q_pack_i8_32")]] kernel void q_pack_i8<32>(device const bfloat*, device int8_t*, device const int*,
    device const int*, constant GqaArgs&, device float*, uint2, uint, uint, uint);
template [[host_name("q_pack48")]] kernel void q_pack<48>(device const bfloat*, device bfloat*, device const int*,
    device const int*, constant GqaArgs&, uint2, uint);
template [[host_name("q_pack32")]] kernel void q_pack<32>(device const bfloat*, device bfloat*, device const int*,
    device const int*, constant GqaArgs&, uint2, uint);

// KV8: int8 K/V with a float scale per (position, kv head) in Ks/Vs: S^T = K8 Q^T then times the key's scale; the
// value scale is folded into P's column (the softmax sum uses the unscaled probabilities).
// SH (attn_shared): one threadgroup per (kv head, tile of R packed rows, split of the keys). Buffer 5 then holds
// int2 (sequence whose block table is read, key limit C) per tile, buffers 8 / 9 the packed rows' token row (-1:
// none) and head in the group; a row sees keys [0, min(C, its position + 1)); partials go to slots own_splits + z.
// Two uses: cascade (rows of several sequences sharing a prefix of C keys) and chunk tiles (mode 6: 8 rows x 6
// heads of one sequence's prefill chunk, C = the tile's last position + 1).
template <int NSG, bool KV8, int R = 48, bool SH = false, bool IQ = false>  // R: packed rows, 48 (8 rows x 6 heads or an SH tile), 32 (<= 5 rows)
kernel void attn_mpp(device const void* qp [[buffer(0)]], device const void* Kc [[buffer(1)]],
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
  nrows = min(seq_len[b], R / 6);
  if (rflag[r0] != (R == 48 ? 4 : 5)) return;
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
  auto tQ = tensor<device QE, DT2, tensor_inline>((device QE*)qp + ((size_t)b * p.Hkv + kh) * R * D, DT2(R, D));
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
    return tensor<device KVT, DT2, tensor_inline>((device KVT*)Kc + ((size_t)kh * p.ns + (size_t)blk * PG) * D,
                                                  DT2(D, n * PG), array<int32_t, 2>{1, ld});
  };
  auto vpage = [&](int blk) {
    return tensor<device KVT, DT2, tensor_inline>((device KVT*)Vc + ((size_t)kh * p.ns + (size_t)blk * PG) * D,
                                                  DT2(D, PG), array<int32_t, 2>{1, ld});
  };
  auto ppage = [&](int buf, int half_) {
    return tensor<threadgroup bfloat, DT2, tensor_inline>(&Psh[buf][0][half_ * PG], DT2(PG, R), array<int32_t, 2>{1, BKP});
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
    return tensor<threadgroup ACC, DT2, tensor_inline>(&Ssh[k0][0], DT2(R, n), array<int32_t, 2>{1, RS});
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
      const int r = srow / 6;
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
          v[i] = k < lim ? x : -INFINITY;
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
      return part + (((size_t)t * p.Hq + kh * 6 + seq_len[b * R + row]) * p.splits + p.own_splits + z) * stride;
    } else {
      const int r = row / 6;
      if (r >= nrows) return nullptr;
      return part + (((size_t)(r0 + r) * p.Hq + kh * 6 + row % 6) * p.splits + z) * stride;
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

#define ATTN_MPP(KV8, R, NAME, SH, IQ)                                                                          \
  template [[host_name(NAME)]] kernel void attn_mpp<8, KV8, R, SH, IQ>(device const void*, device const void*,     \
      device const void*, device const int*, device const int*, device const int*, device float*, constant GqaArgs&, \
      device const int*, device const int*, device const float*, device const float*, device const int*,          \
      device const float*, uint3, uint, uint);
ATTN_MPP(false, 48, "attn_mpp8", false, false) ATTN_MPP(true, 48, "attn_mpp8_kv8", false, false)
ATTN_MPP(false, 32, "attn_mpp8_r32", false, false) ATTN_MPP(true, 32, "attn_mpp8_kv8_r32", false, false)
ATTN_MPP(true, 48, "attn_mpp8_kv8_iq", false, true) ATTN_MPP(true, 32, "attn_mpp8_kv8_r32_iq", false, true)
ATTN_MPP(false, 48, "attn_shared", true, false) ATTN_MPP(true, 48, "attn_shared_kv8", true, false)

// Pack the cascade tiles' queries: qp[(tile * Hkv + kh)][d][48] from token row rows_t[tile * 48 + i] (-1: zero) and
// head rows_j (within the kv head's group). grid: threadgroups (Hkv * 48, tiles), 256 threads. (48-row tiles: a
// 64-row one needs 33 KB of threadgroup memory, over the 32 KB limit.)
kernel void q_pack_sh(device const bfloat* q [[buffer(0)]], device bfloat* qp [[buffer(1)]],
                      device const int* rows_t [[buffer(2)]], device const int* rows_j [[buffer(3)]],
                      constant GqaArgs& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]]) {
  const int kh = tg.x / 48, i = tg.x % 48, tile = tg.y, t = rows_t[tile * 48 + i];
  qp[(((size_t)tile * p.Hkv + kh) * 256 + d) * 48 + i] =
      t >= 0 ? q[((size_t)t * p.Hq + kh * 6 + rows_j[tile * 48 + i]) * 256 + d] : bfloat(0);
}
