// Paged, GQA-packed multi-row attention on the neural accelerators (NAX MMA tiles from the vendored MLX headers).
// A sequence's R rows x the GQ = 6 query heads that share a kv head are packed (index r * GQ + head) into
// 32-row query tiles, so K/V is read per tile rather than per head and the scores are matrix products instead of
// scalar FMAs. R <= 5 (a speculative verify block) is one tile; longer runs (follow-up prefills) take several,
// dispatched next to each other so they share each K/V block through the cache. One 32-position KV tile is
// exactly one paged KV block. Per-row causal masking (row r sees positions <= pos[r]). Writes the same
// unnormalised split partials as attn_gqa ({acc[Dh], m, l}, natural-log units) for attn_combine.
// grid: threadgroups (Hkv * qtiles, S, splits), threads WM * WN * 32. Requires Dh = 256, GQ = 6, chunk % 32 == 0.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

using namespace mlx::steel;

// 8-bit KV (function constant 20, as in core.metal): int8 K/V with a float scale per (position, kv head).
constant bool kv8_fc [[function_constant(20)]];
constant bool kv8 = is_function_constant_defined(kv8_fc) && kv8_fc;

struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles, own_splits, ns, win, gated; };

template <int WM, int WN, int BK = 32, int X = 0, bool VPRE = false>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attn_paged_nax(
    const device bfloat* q [[buffer(0)]],
    const device bfloat* Kc [[buffer(1)]],
    const device bfloat* Vc [[buffer(2)]],
    const device int* block_table [[buffer(3)]],
    const device int* pos [[buffer(4)]],
    const device int* seq_start [[buffer(5)]],
    device float* part [[buffer(6)]],
    constant GqaArgs& p [[buffer(7)]],
    const device int* rflag [[buffer(8)]],
    const device int* seq_len [[buffer(9)]],
    const device float* Ks [[buffer(10)]],
    const device float* Vs [[buffer(11)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  using T = bfloat;
  using AccumType = float;
  constexpr int D = 256, GQ = 6, kU = 16, PG = 32;  // PG: tokens per KV page; BK keys (BK / PG pages) per step
  constexpr int TD = D / kU, TDh = TD / WN, BDh = D / WN, TK = BK / kU;
  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");

  const int kh = tg.x / p.qtiles, qt = tg.x % p.qtiles, b = tg.y, z = tg.z;
  const int r0 = seq_start[b], nrows = seq_len[b];
  if (rflag[r0] != 3) return;
  constexpr int QTR = WM * 16;  // packed query rows per tile
  const int tb = qt * QTR;  // this tile's first packed row
  if (tb >= nrows * GQ) return;
  const int Lmax = pos[r0 + min(nrows - 1, (tb + QTR - 1) / GQ)] + 1;  // the tile's last row sees the most
  const int p0 = z * p.chunk, p1 = min(Lmax, p0 + p.chunk);
  const int ld = D;  // head planes: consecutive slots are D apart

  const short row_group = simd_group_id / WN, d_group = simd_group_id % WN;
  const short tm = kU * row_group;

  using otile_t = NAXTile<AccumType, 1, TDh>;
  using stile_t = NAXTile<AccumType, 1, TK>;
  constexpr short kRowsPT = otile_t::kRowsPerThread;  // 2: rows sm and sm + 8
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  const short2 sc = otile_t::NAXFrag_t::get_coord();
  const short sm = sc.y, sn = sc.x;

  // This thread's two query rows: packed index i = r * GQ + j (token row r of the block, head j of the group).
  int qrow[kRowsPT], Lrow[kRowsPT];
  for (short ii = 0; ii < kRowsPT; ii++) {
    const int i = tb + tm + sm + ii * 8, r = i / GQ;
    qrow[ii] = r < nrows ? ((r0 + r) * p.Hq + kh * GQ + i % GQ) * D : -1;
    Lrow[ii] = r < nrows ? pos[r0 + r] + 1 : 0;
  }

  NAXTile<T, 1, 1> Qtiles[TDh];
  for (short id = 0; id < TDh; id++) {
    thread auto& f = Qtiles[id].frag_at(0, 0);
    for (short ii = 0; ii < kRowsPT; ii++)
      for (short jj = 0; jj < 4; jj++)
        f[ii * 4 + jj] = qrow[ii] >= 0 ? q[qrow[ii] + d_group * BDh + id * kU + sn + jj] : T(0);
  }

  otile_t Otile;
  Otile.clear();
  metal::vec<AccumType, kRowsPT> max_score, sum_score{0};
  for (short i = 0; i < kRowsPT; ++i) max_score[i] = Limits<AccumType>::finite_min;
  const float scale2 = p.scale * 1.44269504089f;

  threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];

  for (int base = p0; base < p1; base += BK) {
    // Pages of this step: subtile ik (16 keys) lives in page (ik * kU) / PG at row (ik * kU) % PG.
    int blks[BK / PG];
    for (int pg = 0; pg < BK / PG; pg++) blks[pg] = block_table[b * p.max_blocks + min(base + pg * PG, p1 - 1) / PG];
    auto kvoff = [&](short ik) { return ((size_t)kh * p.ns + (size_t)blks[(ik * kU) / PG] * PG) * D + d_group * BDh + (size_t)((ik * kU) % PG) * ld; };

    stile_t Stile;
    Stile.clear();
    if (!(X & 8))
    for (short ik = 0; ik < TK; ik += 2) {
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;  // two 16-key subtiles, ik and ik + 1 (same page: PG = 32 = 2 x kU)
        if (kv8)
          Ktile.load((const device char*)Kc + kvoff(ik) + id * kU, ld);
        else
          Ktile.load(Kc + kvoff(ik) + id * kU, ld);
        stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0),
                                metal::false_type{}, Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
      }
    }

    // VPRE: issue this step's V tile loads now, so their latency overlaps the score exchange and softmax.
    NAXTile<T, 1, 2> Vpre[VPRE ? TDh / 2 : 1][VPRE ? TK : 1];
    if (VPRE)
      for (short id = 0; id < TDh; id += 2)
        for (short ik = 0; ik < TK; ik++) {
          if (kv8)
            Vpre[id / 2][ik].load((const device char*)Vc + kvoff(ik) + id * kU, ld);
          else
            Vpre[id / 2][ik].load(Vc + kvoff(ik) + id * kU, ld);
        }

    // Sum the WN head-dim partial scores in a fixed order.
    if (!(X & 1)) {
    threadgroup AccumType* slot = s_xchg[row_group][d_group];
    const short sbase = short(simd_lane_id) * (TK * kEPF);
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      for (short i = 0; i < kEPF; i++) slot[sbase + ik * kEPF + i] = s[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      for (short i = 0; i < kEPF; i++) s[i] = s_xchg[row_group][0][sbase + ik * kEPF + i];
      for (short pg = 1; pg < WN; pg++)
        for (short i = 0; i < kEPF; i++) s[i] += s_xchg[row_group][pg][sbase + ik * kEPF + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale and mask: column c is visible to row ii iff c < Lrow[ii] (covers causality and the block tail).
    constexpr auto neg_inf = Limits<AccumType>::finite_min;
    for (short ik = 0; ik < TK; ik++) {
      thread auto& fg = Stile.frag_at(0, ik);
      for (short ii = 0; ii < kRowsPT; ii++)
        for (short jj = 0; jj < 4; jj++) {
          const int c = base + ik * kU + sn + jj;
          const float ks = kv8 ? Ks[(size_t)kh * p.ns + (size_t)blks[(ik * kU) / PG] * PG + (c - base) % PG] : 1.f;
          fg[ii * 4 + jj] = c < Lrow[ii] && c < p1 ? fg[ii * 4 + jj] * (scale2 * ks) : neg_inf;
        }
    }

    if (X & 2) {  // timing only: no softmax
      NAXTile<T, 1, TK> Ptile0;
      for (short ik = 0; ik < TK; ik++)
        for (short i = 0; i < kEPF; i++) Ptile0.frag_at(0, ik)[i] = T(Stile.frag_at(0, ik)[i]);
      if (!(X & 4))
      for (short id = 0; id < TDh; id += 2)
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          Vtile.load(Vc + kvoff(ik) + id * kU, ld);
          otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Ptile0.frag_at(0, ik),
                                  metal::false_type{}, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      continue;
    }
    metal::vec<AccumType, kRowsPT> new_max, factor;
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    // P in bf16 so P@V runs the bf16 tensor path (a float left operand takes the far slower fp32 path).
    NAXTile<T, 1, TK> Ptile;
    for (short ik = 0; ik < TK; ik++)
      for (short i = 0; i < kEPF; i++) {
        // kv8: fold V's per-position scale into P's column (element i is column sn + i % 4 of the fragment)
        const float vs = kv8 ? Vs[(size_t)kh * p.ns + (size_t)blks[(ik * kU) / PG] * PG + (ik * kU) % PG + sn + i % 4] : 1.f;
        Ptile.frag_at(0, ik)[i] = T(Stile.frag_at(0, ik)[i] * vs);
      }
    if (VPRE)
      for (short id = 0; id < TDh; id += 2)
        for (short ik = 0; ik < TK; ik++)
          otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Ptile.frag_at(0, ik),
                                  metal::false_type{}, Vpre[id / 2][ik].frag_at(0, 0), Vpre[id / 2][ik].frag_at(0, 1),
                                  metal::false_type{});
    if (!(X & 4) && !VPRE)
    for (short id = 0; id < TDh; id += 2) {
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        if (kv8)
          Vtile.load((const device char*)Vc + kvoff(ik) + id * kU, ld);
        else
          Vtile.load(Vc + kvoff(ik) + id * kU, ld);
        otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Ptile.frag_at(0, ik),
                                metal::false_type{}, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
      }
    }
  }

  // Unnormalised partials. A row that saw no visible position in this split reports m = -inf.
  const int stride = D + 2;
  for (short ii = 0; ii < kRowsPT; ii++) {
    const int i = tb + tm + sm + ii * 8, r = i / GQ;
    if (r >= nrows) continue;
    const int t = r0 + r;
    device float* out = part + (((size_t)t * p.Hq + kh * GQ + i % GQ) * p.splits + z) * stride;
    const bool empty = max_score[ii] == Limits<AccumType>::finite_min;
    for (short id = 0; id < TDh; id++)
      for (short jj = 0; jj < 4; jj++)
        out[d_group * BDh + id * kU + sn + jj] = empty ? 0.f : Otile.frag_at(0, id)[ii * 4 + jj];
    if (d_group == 0 && sn == 0) {
      out[D] = empty ? -INFINITY : max_score[ii] * 0.69314718056f;
      out[D + 1] = empty ? 0.f : sum_score[ii];
    }
  }
}

instantiate_kernel("attn_paged_nax_wm2", attn_paged_nax, 2, 4)
instantiate_kernel("attn_paged_nax_wm4", attn_paged_nax, 4, 4)
instantiate_kernel("attn_paged_nax_wm2v", attn_paged_nax, 2, 4, 32, 0, true)
instantiate_kernel("attn_paged_nax_wm2x1", attn_paged_nax, 2, 4, 32, 1)
instantiate_kernel("attn_paged_nax_wm2x2", attn_paged_nax, 2, 4, 32, 2)
instantiate_kernel("attn_paged_nax_wm2x4", attn_paged_nax, 2, 4, 32, 4)
instantiate_kernel("attn_paged_nax_wm2x8", attn_paged_nax, 2, 4, 32, 8)
instantiate_kernel("attn_paged_nax_wm2x15", attn_paged_nax, 2, 4, 32, 15)
  // 64-row tiles for long runs (follow-up prefills)
