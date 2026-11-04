// DFlash2 drafter for Qwen3.8-27B (z-lab/incoai "Qwen3.8-27B-DFlash2"): 5 Qwen3-style layers that attend,
// non-causally within a block, to a sliding window of projected target hidden states, plus a candidate selector.
// Linear weights are quantized to 8-bit at load. Shares the target's embedding and lm_head.
//
// Per cycle and sequence: append the context rows the target just committed (their captured hidden states),
// then run the block [staged token, mask x (L-1)] and write the selected draft tokens into the target's verify
// rows. Everything is encoded into the caller's command buffer.
#pragma once
#include "qwen35.h"

struct DynConvArgs { int H, G, group, half_, kp_stride, has_res; };
struct Top16Args { int V, SH, PS, P0, LO; };
struct DqkArgs { int Hq, Hkv, D, ring, in_stride, with_q; float eps, theta; };
struct DattnArgs { int Hq, Hkv, D, ring, window, splits, chunk; float scale; int seg; };
struct GatherArgs { int cols; };

struct Drafter {
  Gpu& g;
  Qwen35& tgt;
  int H, I, Hq, Hkv, D, nl, block, trained_block, window, ring, mask_id;
  float eps, theta;
  std::vector<int> target_layers;
  QLinear fc, hproj;
  BufRef hidden_norm, norm, pred_cb, succ_cb;
  struct L {
    BufRef in_norm, post_norm, q_norm, k_norm, base_attn, base_mlp;
    QLinear kp_attn, kp_mlp, qkv, o, gate_up, down;
  };
  std::vector<L> layers;
  std::vector<BufRef> kr, vr;  // ring caches per layer: [n_slots, ring, Hkv, D]
  // Scratch (rows = max_tokens for context appends, max_seqs * block for blocks)
  BufRef cx, cxn, kv, x, h, h2, kp, qkv, q, attn, ao, gu, act, xn, hp, cand, unary, part, cpart, vpart, tlog;
  // Draft vocabulary (DRAFT_VOCAB, e.g. 98304; default 0 = all): the drafter scores ids [0, dv) plus the special-token tail
  // [tail_lo, vocab) through views of the target's lm_head. 0.06% of code tokens and 0.18% of English ones lie in
  // between (they can only be rejected drafts; verification is unchanged). DRAFT_VOCAB=0 scores the full vocabulary.
  int dv = 0, tail_lo = 0;
  size_t mem_total = 0, mem_rings = 0;
  // 8-bit context rings (RING8, default on since 2026-11-04; kernels/spec.metal): int8 K/V + a float scale per
  // (position, kv head), about half the bytes per state slot. Affects only the drafts (verification keeps the output
  // exact): acceptance unchanged (laptop spec_test 12.34 / 6.52 vs 6.56 tokens per cycle), M6 drafter 2.8 -> 2.3 GiB
  // (+0.5 GiB of KV), M5 Pro 4.4 -> 3.1. RING8=0: bf16 rings.
  bool ring8 = !getenv("RING8") || atoi(getenv("RING8"));
  size_t slot_ring_bytes() const { return (size_t)ring * Hkv * (ring8 ? D + 4 : D * 2); }  // one slot of one layer's K (or V) ring
  id<MTLComputePipelineState> rk(const char* name) { return g.pso_fc(name, {{30, ring8}}); }  // bytes allocated by the drafter (weights, context rings, scratch) and its rings
  // Inputs for the two phases
  BufRef c_rows[2], c_pos[2], c_tseq[2], c_slot[2], b_tok, b_pos, b_tseq, b_start, b_len, b_spec, b_slot, b_vstart;

  // dflt: the widest block without DRAFT_BLOCK (kMaxBlock; MoE targets 8, see the Scheduler constructor)
  static int env_block(int dflt = kMaxBlock) { return std::clamp(getenv("DRAFT_BLOCK") ? atoi(getenv("DRAFT_BLOCK")) : dflt, 2, kMaxBlock); }
  int max_rows = 0;  // block rows the scratch holds: tgt.spec_rows when set (the scheduler's cap), else max_seqs x kMaxBlock
  Drafter(Gpu& g, Qwen35& tgt, const Checkpoint& ck) : g(g), tgt(tgt) {
    const size_t mem0 = g.dev.currentAllocatedSize;
    const json::Value& c = ck.config;
    H = c["hidden_size"].i();
    I = c["intermediate_size"].i();
    Hq = c["num_attention_heads"].i();
    Hkv = c["num_key_value_heads"].i();
    D = c["head_dim"].i();
    nl = c["num_hidden_layers"].i();
    eps = c["rms_norm_eps"].n;
    theta = c["rope_parameters"]["rope_theta"].n;
    window = c["sliding_window"].i();
    const json::Value& dc = c["dflash_config"];
    mask_id = dc["mask_token_id"].i();
    // The trained block size (DFlash2: 8). Block 8 was useless until 2026-10-30 because draft_attn only computed 5
    // rows (DMAXR = 5): rows 6-8 drafted junk and 4.28 tokens/cycle became 5.68 once fixed. At most kMaxBlock.
    // Since 2026-11-02 the drafter runs blocks past its trained size, up to kMaxBlock (16): the extra rows are more
    // mask tokens at the next positions (as TensorFold's 16-wide lanes do); `block` is the widest block the scheduler
    // may pick (DRAFT_BLOCK overrides; trained_block keeps the config's value).
    trained_block = (int)dc["block_size"].i();
    // MoE targets (35B-A3B): no blocks past the trained 8 unless DRAFT_BLOCK asks (rows cost experts' weights)
    block = env_block(tgt.cfg.experts ? std::min(trained_block, kMaxBlock) : kMaxBlock);
    for (auto& v : dc["target_layer_ids"].a) target_layers.push_back(v.i());
    ring = window + kMaxBlock;
    if (Hq / Hkv != 4 || D != 128 || dc["selector_rank"].i() != 256 || dc["selector_top_k"].i() != 16 ||
        dc["conv_kernel_size"].i() != 2) {
      fprintf(stderr, "drafter: unsupported shape\n");
      exit(1);
    }
    // The drafter's layer linears in 4-bit (DRAFT_BITS=8 for 8-bit): half the bytes per cycle, acceptance unchanged
    // (4.06 vs 4.08 tokens/cycle at block 5, 5.71 vs 5.78 at block 8).
    const int qb = getenv("DRAFT_BITS") ? atoi(getenv("DRAFT_BITS")) : 4;
    fc = load_qlinear_bf16_q8(g, ck, {"fc.weight"});
    hproj = load_qlinear_bf16_q8(g, ck, {"candidate_selector.hidden_projection.weight"});
    tgt.flip({&fc, &hproj});
    tgt.reserve_a8(fc.K);  // fc's K (target layers x hidden) is wider than any target GEMM's W4A8 scratch
    hidden_norm = ck.load(g, "hidden_norm.weight");
    norm = ck.load(g, "norm.weight");
    pred_cb = ck.load(g, "candidate_selector.predecessor_codebook");
    succ_cb = ck.load(g, "candidate_selector.successor_codebook");
    for (int l = 0; l < nl; l++) {
      std::string p = "layers." + std::to_string(l) + ".";
      L ly;
      ly.in_norm = ck.load(g, p + "input_layernorm.weight");
      ly.post_norm = ck.load(g, p + "post_attention_layernorm.weight");
      ly.q_norm = ck.load(g, p + "self_attn.q_norm.weight");
      ly.k_norm = ck.load(g, p + "self_attn.k_norm.weight");
      ly.base_attn = ck.load(g, p + "attention_conv.base_kernel");
      ly.base_mlp = ck.load(g, p + "mlp_conv.base_kernel");
      ly.kp_attn = load_qlinear_bf16_q8(g, ck, {p + "attention_conv.kernel_projection.weight"}, qb);
      ly.kp_mlp = load_qlinear_bf16_q8(g, ck, {p + "mlp_conv.kernel_projection.weight"}, qb);
      ly.qkv = load_qlinear_bf16_q8(g, ck, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                            p + "self_attn.v_proj.weight"}, qb);
      ly.o = load_qlinear_bf16_q8(g, ck, {p + "self_attn.o_proj.weight"}, qb);
      ly.gate_up = load_qlinear_bf16_q8(g, ck, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, qb);
      ly.down = load_qlinear_bf16_q8(g, ck, {p + "mlp.down_proj.weight"}, qb);
      tgt.flip({&ly.kp_attn, &ly.kp_mlp, &ly.qkv, &ly.o, &ly.gate_up, &ly.down});
      layers.push_back(ly);
      kr.push_back(g.alloc((size_t)tgt.n_slots * slot_ring_bytes()));
      vr.push_back(g.alloc((size_t)tgt.n_slots * slot_ring_bytes()));
    }
    // Block rows: the scheduler's cap on verify rows when the caller set one (Scheduler::spec_rows_cap: 128 for 16
    // sequences instead of 16 x 16), else every sequence at the widest block.
    max_rows = tgt.spec_rows > 0 ? tgt.spec_rows : tgt.max_seqs * kMaxBlock;
    int CT = tgt.max_tokens, BT = max_rows, R = std::max(CT, BT);
    cx = g.alloc((size_t)CT * target_layers.size() * H * 2);
    cxn = g.alloc((size_t)CT * H * 2);
    kv = g.alloc((size_t)CT * 2 * Hkv * D * 2);
    x = g.alloc((size_t)BT * H * 2);
    h = g.alloc((size_t)R * H * 2);
    h2 = g.alloc((size_t)BT * H * 2);
    kp = g.alloc((size_t)BT * 4 * (H / 16) * 2);
    qkv = g.alloc((size_t)BT * (Hq + 2 * Hkv) * D * 2);
    q = g.alloc((size_t)BT * Hq * D * 2);
    attn = g.alloc((size_t)BT * Hq * D * 2);
    ao = g.alloc((size_t)BT * H * 2);
    gu = g.alloc((size_t)BT * 2 * I * 2);
    act = g.alloc((size_t)BT * I * 2);
    xn = g.alloc((size_t)BT * H * 2);
    hp = g.alloc((size_t)BT * 256 * 2);
    cand = g.alloc((size_t)BT * 16 * 4);
    cpart = g.alloc((size_t)BT * 256 * 16 * 4);  // sharded top-16: up to 256 shards per row
    vpart = g.alloc((size_t)BT * 256 * 16 * 4);
    unary = g.alloc((size_t)BT * 16 * 4);
    {
      const int V = tgt.cfg.vocab;
      dv = getenv("DRAFT_VOCAB") ? atoi(getenv("DRAFT_VOCAB")) : 0;  // off: on agent text (Studio) acceptance fell 4.46 -> 4.15 (16k) and 5.14 -> 4.62 (64k)
      tail_lo = getenv("DRAFT_TAIL") ? atoi(getenv("DRAFT_TAIL")) : V == 248320 ? 248032 : V;  // Qwen3.5/3.8 specials from 248044 (tile-aligned)
      if (dv <= 0 || dv >= tail_lo || tail_lo > V) dv = 0;
      if (dv) {
        tlog = g.alloc((size_t)BT * (V - tail_lo) * 2);
      }
    }
    {
      const int dchunk = getenv("DRAFT_CHUNK") ? atoi(getenv("DRAFT_CHUNK")) : 64;
      const int max_splits = (window + kMaxBlock - 1 + dchunk - 1) / dchunk;
      part = g.alloc((size_t)BT * Hkv * max_splits * 4 * (D + 2) * 4);  // sequences x row instance x splits x GQ
    }
    for (int k = 0; k < 2; k++) {
      c_rows[k] = g.alloc(CT * 4);
      c_pos[k] = g.alloc(CT * 4);
      c_tseq[k] = g.alloc(CT * 4);
      c_slot[k] = g.alloc(CT * 4);
    }
    b_tok = g.alloc(BT * 4);
    b_pos = g.alloc(BT * 4);
    b_tseq = g.alloc(BT * 4);
    b_start = g.alloc(tgt.max_seqs * 4);
    b_len = g.alloc(tgt.max_seqs * 4);
    b_spec = g.alloc(tgt.max_seqs * 4);
    b_slot = g.alloc(tgt.max_seqs * 4);
    b_vstart = g.alloc(tgt.max_seqs * 4);
    mem_total = g.dev.currentAllocatedSize - mem0;
    mem_rings = (size_t)nl * 2 * tgt.n_slots * slot_ring_bytes();
  }

  // Context append: rows of the target's hidden capture (tgt.hcap) to enter into each sequence's ring, with
  // their positions and slots (up to max_tokens rows). `set` picks one of two input-buffer sets so two appends
  // can share a command buffer.
  struct CtxRow { int hcap_row, pos, slot; };
  void encode_context(Enc& e, const std::vector<CtxRow>& rows, int set = 0) {
    int n = (int)rows.size();
    if (!n) return;
    BufRef &cr = c_rows[set], &cp = c_pos[set], &ct = c_tseq[set], &cs = c_slot[set];
    for (int i = 0; i < n; i++) {
      cr.ptr<int>()[i] = rows[i].hcap_row;
      cp.ptr<int>()[i] = rows[i].pos;
      ct.ptr<int>()[i] = i;  // one "sequence" per row: draft_qk looks the slot up by row
      cs.ptr<int>()[i] = rows[i].slot;
    }
    int cols = (int)target_layers.size() * H;
    GatherArgs ga{cols};
    e.k("gather_rows").buf(0, tgt.hcap).buf(1, cr).buf(2, cx).args(3, ga);
    e.threads(S3(cols, n), S3(256));
    tgt.gemm(e, cx, fc, h, n, 0);
    tgt.rmsnorm(e, h, hidden_norm, cxn, n, H, H, H);
    DqkArgs da{Hq, Hkv, D, ring, 2 * Hkv * D, 0, eps, theta};
    for (int l = 0; l < nl; l++) {
      // k|v rows of the fused qkv weight: run only the k and v parts via an offset view of the weight.
      // (rows_of drops the qmst copy and keeps the lane4 copy only at a tile-aligned offset: a copy taken at the wrong
      // offset computed q rows as k/v and drafted junk.)
      QLinear kvw = rows_of(layers[l].qkv, Hq * D, 2 * Hkv * D);
      tgt.gemm(e, cxn, kvw, kv, n, 0);
      e.k(rk("draft_qk")).buf(0, kv).buf(1, layers[l].q_norm).buf(2, layers[l].k_norm).buf(3, q).buf(4, kr[l]).buf(5, vr[l])
          .buf(6, cp).buf(7, ct).buf(8, cs).args(9, da);
      e.groups(S3(Hkv, n), S3(D));
    }
  }

  // A block per speculative sequence: work[i] = (slot, position of the staged token, staged token, row of
  // the staged token in the target's verify batch). Writes the draft tokens into tgt.tok after that row.
  // tree: the caller builds the verify rows (tree.h). vlen: verify rows the sequence takes (0 = the block): a decode that
  // lands on a snapshot boundary verifies fewer rows than the drafter drafts, and draft_select fills only those.
  struct BlockSeq { int slot, pos, staged, vrow; bool tree = false; int vlen = 0; };
  void encode_block(Enc& e, const std::vector<BlockSeq>& seqs, int L = 0) {
    const int block = L > 0 ? L : this->block;
    int S = (int)seqs.size(), T = S * block;
    if (!S) return;
    if (S * (block <= 5 ? 5 : block <= 8 ? 8 : block <= 12 ? 12 : 16) > max_rows) {  // draft_attn's row instance
      fprintf(stderr, "drafter: %d blocks of %d rows > %d rows of scratch\n", S, block, max_rows);
      abort();
    }
    for (int b = 0; b < S; b++) {
      b_start.ptr<int>()[b] = b * block;
      b_len.ptr<int>()[b] = block;
      // draft_select writes chain drafts into the first b_spec verify rows (0: a tree, built by the caller)
      b_spec.ptr<int>()[b] = seqs[b].tree ? 0 : seqs[b].vlen > 0 ? std::min(seqs[b].vlen, block) : block;
      b_slot.ptr<int>()[b] = seqs[b].slot;
      b_vstart.ptr<int>()[b] = seqs[b].vrow;
      for (int j = 0; j < block; j++) {
        b_tok.ptr<int>()[b * block + j] = j == 0 ? seqs[b].staged : mask_id;
        b_pos.ptr<int>()[b * block + j] = seqs[b].pos + j;
        b_tseq.ptr<int>()[b * block + j] = b;
      }
    }
    EmbedArgs ea{H};
    e.k(tgt.embed.bits == 16 ? "embed16" : tgt.embed.bits == 8 ? "embed8" : "embed4").buf(0, b_tok).buf(1, tgt.embed.wq).buf(2, tgt.embed.s).buf(3, tgt.embed.b).buf(4, x).args(5, ea);
    e.threads(S3(H, T), S3(256));
    const int G = H / 16;
    DqkArgs da{Hq, Hkv, D, ring, (Hq + 2 * Hkv) * D, 1, eps, theta};
    // Keys per drafter attention split: 64 gives 33 one-simdgroup threadgroups per (kv head, sequence, row group)
    // over the 2,048-key window (256: 9; drafter block 4.5 -> 3.8 ms per cycle on the Studio).
    static const int dchunk = getenv("DRAFT_CHUNK") ? atoi(getenv("DRAFT_CHUNK")) : 64;
    int span = window + block - 1, chunk = dchunk, splits = (span + chunk - 1) / chunk;
    // DRAFT_SEG: rows past it see the whole block, rows before it only the first DRAFT_SEG (default: the trained size,
    // so the first 8 drafts of a 16-block equal block 8's; 0 = no segment, all rows see the whole block).
    static const int seg_env = getenv("DRAFT_SEG") ? atoi(getenv("DRAFT_SEG")) : -1;
    const int seg = seg_env < 0 ? trained_block : seg_env > 0 ? seg_env : block;
    DattnArgs aa{Hq, Hkv, D, ring, window, splits, chunk, 1.f / sqrtf((float)D), seg >= block ? 1 << 20 : seg};
    static const int skip = getenv("DRAFT_SKIP") ? atoi(getenv("DRAFT_SKIP")) : 0;  // timing only: 1 attn, 2 MLP, 4 head
    for (int l = 0; l < nl; l++) {
      auto& ly = layers[l];
      // attention sub-block: conv(prepare) -> attention -> conv(finish) + residual
      if (!(skip & 1)) {
      tgt.rmsnorm(e, x, ly.in_norm, h, T, H, H, H);
      tgt.gemm(e, h, ly.kp_attn, kp, T, 0);
      dyn_conv(e, h, ly.base_attn, 0, h2, BufRef(), T, G);
      tgt.gemm(e, h2, ly.qkv, qkv, T, 0);
      e.k(rk("draft_qk")).buf(0, qkv).buf(1, ly.q_norm).buf(2, ly.k_norm).buf(3, q).buf(4, kr[l]).buf(5, vr[l])
          .buf(6, b_pos).buf(7, b_tseq).buf(8, b_slot).args(9, da);
      e.groups(S3(Hq + Hkv, T), S3(D));
      // Rows per instance: 5, 8 (as 2 x 4 across threadgroups unless DRAFT_ATTN_ONE), 12 (3 x 4), 16 (4 x 4). Every
      // row of the block must be covered (block 8 drafted junk at rows 6-8 until 2026-10-30).
      static const bool grouped = !getenv("DRAFT_ATTN_ONE");
      const int dr = block <= 5 ? 5 : block <= 8 ? 8 : block <= 12 ? 12 : 16;
      const int ng = dr == 5 || (dr == 8 && !grouped) ? 1 : dr / 4;
      const char* ak = dr == 5 ? "draft_attn5" : dr == 8 ? (grouped ? "draft_attn8g" : "draft_attn8") :
                       dr == 12 ? "draft_attn12g" : "draft_attn16g";
      e.k(rk(ak)).buf(0, q).buf(1, kr[l]).buf(2, vr[l])
          .buf(3, b_pos).buf(4, b_start).buf(5, b_len).buf(6, b_slot).buf(7, part).args(8, aa);
      e.groups(S3(Hkv, S, splits * ng), S3(32));
      const char* ck = dr == 5 ? "draft_attn_combine5" : dr == 8 ? "draft_attn_combine8" :
                       dr == 12 ? "draft_attn_combine12" : "draft_attn_combine16";
      e.k(ck).buf(0, part).buf(1, attn).buf(2, b_tseq).buf(3, b_start).args(4, aa);
      e.groups(S3(Hq, T), S3(D));
      tgt.gemm(e, attn, ly.o, ao, T, 0);
      dyn_conv(e, ao, ly.base_attn, 1, x, x, T, G);
      }
      // MLP sub-block
      if (skip & 2) continue;
      tgt.rmsnorm(e, x, ly.post_norm, h, T, H, H, H);
      tgt.gemm(e, h, ly.kp_mlp, kp, T, 0);
      dyn_conv(e, h, ly.base_mlp, 0, h2, BufRef(), T, G);
      tgt.gemm(e, h2, ly.gate_up, gu, T, 0);
      ActArgs ac{I};
      e.k("silu_mul").buf(0, gu).buf(1, act).args(2, ac);
      e.threads(S3(I, T), S3(256));
      tgt.gemm(e, act, ly.down, ao, T, 0);
      dyn_conv(e, ao, ly.base_mlp, 1, x, x, T, G);
    }
    if (skip & 4) return;
    tgt.rmsnorm(e, x, norm, xn, T, H, H, H);
    // Draft logits through the target's lm_head (into the target's logits buffer; the verify pass rewrites it).
    static const int sh = getenv("TOP16_SHARDS") ? atoi(getenv("TOP16_SHARDS")) : 64;  // 0: one threadgroup per row
    if (dv && sh > 0) {
      const int tn = tgt.cfg.vocab - tail_lo;  // views taken here: the target flips lm_head after we are built
      tgt.gemm(e, xn, rows_of(tgt.lm_head, 0, dv), tgt.logits, T, 0);
      tgt.gemm(e, xn, rows_of(tgt.lm_head, tail_lo, tn), tlog, T, 0);
      Top16Args ta{dv, sh, sh + 1, 0, 0}, tt{tn, 1, sh + 1, sh, tail_lo};
      e.k("row_top16_shard").buf(0, tgt.logits).buf(1, cpart).buf(2, vpart).args(3, ta);
      e.groups(S3(sh, T), S3(256));
      e.k("row_top16_shard").buf(0, tlog).buf(1, cpart).buf(2, vpart).args(3, tt);
      e.groups(S3(1, T), S3(256));
      e.k("top16_merge").buf(0, cpart).buf(1, vpart).buf(2, cand).buf(3, unary).args(4, ta);
      e.groups(S3(T), S3(256));
    } else if (sh > 0) {
      tgt.gemm(e, xn, tgt.lm_head, tgt.logits, T, 0);
      Top16Args ta{tgt.cfg.vocab, sh, sh, 0, 0};
      e.k("row_top16_shard").buf(0, tgt.logits).buf(1, cpart).buf(2, vpart).args(3, ta);
      e.groups(S3(sh, T), S3(256));
      e.k("top16_merge").buf(0, cpart).buf(1, vpart).buf(2, cand).buf(3, unary).args(4, ta);
      e.groups(S3(T), S3(256));
    } else {
      tgt.gemm(e, xn, tgt.lm_head, tgt.logits, T, 0);
      e.k("row_top16").buf(0, tgt.logits).buf(1, cand).buf(2, unary).args(3, tgt.cfg.vocab);
      e.groups(S3(T), S3(256));
    }
    tgt.gemm(e, xn, hproj, hp, T, 0);
    e.k("draft_select").buf(0, cand).buf(1, unary).buf(2, hp).buf(3, pred_cb).buf(4, succ_cb).buf(5, b_start)
        .buf(6, b_len).buf(7, b_spec).buf(8, b_tok).buf(9, tgt.tok).buf(10, b_vstart);
    e.groups(S3(S), S3(256));
  }

  // Token trees built on the GPU (TREE_GPU=1, kernels/tree.metal tree_build) from this cycle's block, straight into
  // the target's verify rows: seqs[i] = {drafter row of its block, target row of its root, pos0, rows}.
  BufRef tinfo;
  void encode_tree(Enc& e, const std::vector<std::array<int, 4>>& seqs, const TreeParams& tp, int L) {
    if (seqs.empty()) return;
    if (!tinfo.buf) tinfo = g.alloc((size_t)tgt.max_seqs * 16);
    for (size_t i = 0; i < seqs.size(); i++) memcpy(tinfo.ptr<int>() + i * 4, seqs[i].data(), 16);
    struct { int budget, max_kids, max_depth, block; float temp; } a{tp.budget, tp.max_kids, tp.max_depth, L, tp.temp};
    e.k("tree_build").buf(0, cand).buf(1, unary).buf(2, hp).buf(3, pred_cb).buf(4, succ_cb).buf(5, tinfo)
        .buf(6, tgt.tok).buf(7, tgt.rpos).buf(8, tgt.tpar).buf(9, tgt.tdep).buf(10, tgt.tanc).args(11, a);
    e.groups(S3(seqs.size()), S3(256));
  }

  void dyn_conv(Enc& e, const BufRef& in, const BufRef& base, int half_, const BufRef& out, const BufRef& res, int T,
                int G) {
    DynConvArgs a{H, G, 16, half_, 4 * G, res.buf != nil};
    e.k("dyn_conv").buf(0, in).buf(1, kp).buf(2, base.at((size_t)half_ * 2 * H * 2)).buf(3, out)
        .buf(4, res.buf != nil ? res : in).buf(5, b_tseq).buf(6, b_start).args(7, a);
    e.threads(S3(H, T), S3(256));
  }

  void encode_zero_slot(id<MTLBlitCommandEncoder> b, int s) {
    size_t sz = slot_ring_bytes();
    for (int l = 0; l < nl; l++) {
      [b fillBuffer:kr[l].buf range:NSMakeRange(s * sz, sz) value:0];
      [b fillBuffer:vr[l].buf range:NSMakeRange(s * sz, sz) value:0];
    }
  }

  void encode_copy_slot(id<MTLBlitCommandEncoder> b, int src, int dst) {
    size_t sz = slot_ring_bytes();
    for (int l = 0; l < nl; l++) {
      [b copyFromBuffer:kr[l].buf sourceOffset:src * sz toBuffer:kr[l].buf destinationOffset:dst * sz size:sz];
      [b copyFromBuffer:vr[l].buf sourceOffset:src * sz toBuffer:vr[l].buf destinationOffset:dst * sz size:sz];
    }
  }
};
