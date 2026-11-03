// Speculative decoding with Qwen3.8-Flash-Next's own MTP head (mtp.*, loaded into Qwen35 by MTP=1 / DRAFT=mtp).
//
// The head: fused = fc_embedding(norm_e(embed(token))) broadcast to every hyper stream + fc_hidden(norm_h(hidden))
// per stream (norm_h is ONE RMS norm over all hc * H columns), then one gated-attention + MoE decoder layer with
// hyper-connections and its own KV pool, then its own mixer; logits through the target's lm_head. An MTP row at
// position p takes (token at p + 1, hidden at p) and predicts the token at p + 2; its KV sits at position p (causal,
// the target's rope). For committed positions `hidden` is the target's pre-mixer state hx; for draft passes it is
// the previous pass's MTP pre-mixer output.
//
// Per step, all on the GPU in the step's command buffer (no host round trip):
//   1. draft (encode_draft): for each speculated sequence with committed length c and staged token x_c, pass 0 at
//      p = c - 1 with (x_c, the target hidden saved for its slot) -> d1; pass j at p = c - 1 + j with (d_j, pass j-1's
//      output) -> d_{j+1}. One row per sequence per pass; drafts go straight into the target's verify rows.
//   2. the PLE n-gram ids of the verify rows are recomputed from the drafts (ids_fix).
//   3. the target step (verify: accept, GDN / PLE-conv replay).
//   4. context (encode_context), over every row of the step: the next token (prefill: the prompt's; verify rows:
//      the accepted draft or the bonus token, from tok / commit / emit) with the target's hx row -> MTP KV at that
//      position. Rows past the accepted ones write junk KV at positions the next cycle rewrites before reading.
//      Each sequence's last committed row's hx is kept per slot for the next pass 0.
// Positions >= c written by draft passes are provisional: step 4 rewrites them.
#pragma once
#include "qwen35.h"

struct Mtp {
  Gpu& g;
  Qwen35& m;
  int block;  // verify rows per speculated sequence (staged token + block - 1 drafts): MTP_K + 1
  int H, HH;
  BufRef th;              // [n_slots, HH] target pre-mixer hidden at each sequence's last committed position
  BufRef hs;              // MTP hyper state [R, HH] (swapped in as Qwen35::hx while the MTP layer runs)
  BufRef hn, fh;          // norm_h(hidden), fc_hidden(.) [R, HH]
  BufRef ee, en, fe;      // embedding, norm_e(.), fc_embedding(.) [R, H]
  BufRef next;            // [max_tokens] context rows' next tokens
  BufRef hin;             // [max_seqs, HH] pass 0's hidden (gathered from th)
  BufRef dpred, vrow;     // per draft sequence: argmax, the main batch row of its staged token
  BufRef hist, ngtab;     // per main sequence: the 2 tokens before its first row; the n-gram hash table
  BufRef ids_chk;         // MTP_IDS_CHECK: GPU ids for every row, compared with the host's
  // Fold (MTP_FOLD, default on): the context pass's row at a sequence's last committed position c - 1 already takes
  // (x_c, target hidden at c - 1), i.e. it IS the next cycle's pass 0 whenever x_c is known on the GPU (the bonus of a
  // verify step, or the greedy argmax after a prompt). Its logits are taken there: d1 and the MTP state per slot,
  // so the next cycle starts at pass 1 (one MTP layer pass fewer per cycle). The host only uses a fold when the
  // sequence's staged token is the one the GPU assumed (Seq::mtp_fold).
  bool fold = !getenv("MTP_FOLD") || atoi(getenv("MTP_FOLD"));
  BufRef flog;            // the fold's logits [max_seqs, vocab]
  BufRef xs, d1s;         // per slot: the folded row's MTP output [HH] and its argmax
  BufRef lastp;           // per main sequence: 1 = its last row is a completed greedy prompt (next = the target's argmax)
  std::vector<Qwen35::BatchState> passes;
  bool ids_check = getenv("MTP_IDS_CHECK") && atoi(getenv("MTP_IDS_CHECK"));

  Mtp(Gpu& g, Qwen35& m) : g(g), m(m) {
    if (!m.has_mtp) { fprintf(stderr, "MTP: the model has no MTP head loaded\n"); exit(1); }
    const int K = getenv("MTP_K") ? atoi(getenv("MTP_K")) : 3;
    block = std::max(2, std::min(kMaxBlock, K + 1));
    H = m.cfg.hidden;
    HH = m.cfg.hc * H;
    const size_t R = m.max_tokens;
    th = g.alloc((size_t)m.n_slots * HH * 2);
    memset(th.ptr<char>(), 0, (size_t)m.n_slots * HH * 2);
    hs = g.alloc(R * HH * 2);
    hn = g.alloc(R * HH * 2);
    fh = g.alloc(R * HH * 2);
    ee = g.alloc(R * H * 2);
    en = g.alloc(R * H * 2);
    fe = g.alloc(R * H * 2);
    next = g.alloc(R * 4);
    hin = g.alloc((size_t)m.max_seqs * HH * 2);
    dpred = g.alloc(m.max_seqs * 4);
    vrow = g.alloc(m.max_seqs * 4);
    hist = g.alloc(m.max_seqs * 2 * 4);
    xs = g.alloc((size_t)m.n_slots * HH * 2);
    d1s = g.alloc(m.n_slots * 4);
    if (fold) flog = g.alloc((size_t)m.max_seqs * m.cfg.vocab * 2);
    lastp = g.alloc(m.max_seqs * 4);
    if (m.ple_ids_buf.buf) {
      auto t = m.ngram.table();
      ngtab = g.alloc(t.size() * 8);
      memcpy(ngtab.ptr<char>(), t.data(), t.size() * 8);
      if (ids_check) ids_chk = g.alloc(R * m.ngram.heads * 4);
    }
    for (int j = 0; j < kMaxBlock; j++) passes.push_back(m.alloc_batch(m.max_seqs));
  }

  // hs [T, HH] = fc_embedding(norm_e(embed(toks))) per stream + fc_hidden(norm_h(hidden)).
  void fuse(Enc& e, const BufRef& toks, const BufRef& hidden, int T) {
    EmbedArgs ea{H};
    const QLinear& em = m.embed;
    e.k(em.bits == 16 ? "embed16" : em.bits == 8 ? "embed8" : "embed4").buf(0, toks).buf(1, em.wq).buf(2, em.s)
        .buf(3, em.b).buf(4, ee).args(5, ea);
    e.threads(S3(H, T), S3(256));
    struct { int H, hc, in_stride; float eps; } ne{H, 1, H, m.cfg.eps}, nh{HH, 1, HH, m.cfg.eps};
    e.k("hc_norm").buf(0, ee).buf(1, m.mtp_norm_e).buf(2, en).args(3, ne);
    e.groups(S3(1, T), S3(1024));
    e.k("hc_norm").buf(0, hidden).buf(1, m.mtp_norm_h).buf(2, hn).args(3, nh);
    e.groups(S3(1, T), S3(1024));
    m.a8_ready = false;
    m.gemm(e, en, m.mtp_fc_e, fe, T, 0);
    m.gemm(e, hn, m.mtp_fc_h, fh, T * m.cfg.hc, 0);  // [T, hc * H] is [T * hc, H]: every stream separately
    m.flush_split(e);
    struct { int H, hc; } fa{H, m.cfg.hc};
    e.k("mtp_fuse").buf(0, fe).buf(1, fh).buf(2, hs).args(3, fa);
    e.threads(S3(H, T), S3(256));
  }

  // The MTP decoder layer over the current batch's T rows (S sequences), on hs.
  void layer(Enc& e, id<MTLCommandBuffer> __strong& cb, int T, int S) {
    std::swap(m.hx, hs);
    AttnArgs aa = m.attn_args(T);
    GdnArgs ga{S, m.cfg.lhk, m.cfg.lhv, m.cfg.ldk, m.cfg.ldv, m.cfg.conv_dim(), m.cfg.gdn_proj(), m.cfg.eps, 0, 1};
    m.encode_layer(e, cb, m.mtp_ly, -1, T, S, aa, ga);
    m.flush_split(e);
    std::swap(m.hx, hs);
  }

  struct DraftSeq { int slot; const std::vector<int>* blocks; int pos; int staged; int vrow; };
  // Draft npass tokens for each sequence (pos = its committed length c, staged = x_c, vrow = x_c's row in the main
  // batch, whose set_batch has been done). Writes the drafts into m.tok rows vrow + 1 .. vrow + npass.
  // folded: every sequence's d1 and pass-0 state are in d1s / xs (from the last step's context pass).
  void encode_draft(Enc& e, id<MTLCommandBuffer> __strong& cb, const std::vector<DraftSeq>& ds, int npass,
                    bool folded = false) {
    const int S = (int)ds.size();
    if (!S || npass <= 0) return;
    npass = std::min(npass, kMaxBlock);
    Qwen35::BatchState main = m.save_batch();
    std::vector<int> toks(S);
    for (int b = 0; b < S; b++) {
      toks[b] = ds[b].staged;
      vrow.ptr<int>()[b] = ds[b].vrow;
    }
    for (int j = 0; j < npass; j++) {  // plan every pass into its own buffers before encoding any
      passes[j].cascade = false;
      m.load_batch(passes[j]);
      std::vector<Qwen35::SeqWork> w;
      for (int b = 0; b < S; b++) w.push_back({ds[b].slot, ds[b].blocks, ds[b].pos - 1 + j, &toks[b], 1, false});
      m.set_batch(w);
      passes[j] = m.save_batch();
    }
    const QLinear& lm = m.lm_head;
    int j0 = 0;
    if (folded) {  // d1 into the verify row and pass 1's token; pass 1's hidden = the folded row's MTP state
      m.load_batch(passes[0]);
      struct { int HH, S, has_next; } fa{HH, S, npass > 1};
      e.k("mtp_fold_in").buf(0, d1s).buf(1, xs).buf(2, m.slot).buf(3, vrow).buf(4, main.tok)
          .buf(5, npass > 1 ? passes[1].tok : dpred).buf(6, hs).args(7, fa);
      e.threads(S3(HH, S), S3(256));
      j0 = 1;
    }
    for (int j = j0; j < npass; j++) {
      m.load_batch(passes[j]);
      if (j == 0) {  // pass 0's hidden: the target's, saved per slot by the last step
        struct { int H; } gr{HH};
        e.k("hc_gather_rows").buf(0, th).buf(1, m.slot).buf(2, hin).args(3, gr);
        e.threads(S3(HH, S), S3(256));
      }
      fuse(e, m.tok, j == 0 ? hin : hs, S);
      layer(e, cb, S, S);
      std::swap(m.hx, hs);
      m.hyper_pre(e, m.mtp_mixer, S, false, m.blk_out);
      std::swap(m.hx, hs);
      m.a8_ready = false;
      m.gemm(e, m.blk_out, lm, m.logits, S, 0);
      m.flush_split(e);
      e.k("row_argmax").buf(0, m.logits).buf(1, dpred).args(2, m.cfg.vocab);
      e.groups(S3(S), S3(1024));
      struct { int j, has_next, S; } pa{j, j + 1 < npass, S};
      e.k("mtp_put").buf(0, dpred).buf(1, j + 1 < npass ? passes[j + 1].tok : dpred).buf(2, main.tok).buf(3, vrow)
          .args(4, pa);
      e.threads(S3(S), S3(64));
    }
    m.load_batch(main);
  }

  // Recompute the PLE n-gram ids of speculative sequences' rows (main batch) from the drafts now in m.tok.
  // h2[b] = {token at pos0 - 1, token at pos0 - 2} of main sequence b (anything when that position is < 0).
  void ids_fix(Enc& e, const std::vector<std::array<int, 2>>& h2, int maxn, bool any_spec) {
    if (!m.ple_ids_buf.buf || h2.empty()) return;
    const int S = (int)h2.size();
    for (int b = 0; b < S; b++) { hist.ptr<int>()[2 * b] = h2[b][0]; hist.ptr<int>()[2 * b + 1] = h2[b][1]; }
    if (!any_spec) return;
    struct { int heads_per, ngram, heads, eos, all; } na{m.ngram.heads_per, m.ngram.ngram, m.ngram.heads, m.ngram.eos, 0};
    e.k("ple_ids_gpu").buf(0, m.tok).buf(1, m.seq_start).buf(2, m.seq_len).buf(3, m.spec).buf(4, m.pos).buf(5, hist)
        .buf(6, ngtab).buf(7, m.ple_ids_buf).args(8, na);
    e.threads(S3(m.ngram.heads, maxn, S), S3(m.ngram.heads, 1, 1));
  }
  // MTP_IDS_CHECK: the GPU ids of every row (into ids_chk), for comparison with the host's after the step.
  void ids_all(Enc& e, int S, int maxn) {
    if (!ids_chk.buf) return;
    struct { int heads_per, ngram, heads, eos, all; } na{m.ngram.heads_per, m.ngram.ngram, m.ngram.heads, m.ngram.eos, 1};
    e.k("ple_ids_gpu").buf(0, m.tok).buf(1, m.seq_start).buf(2, m.seq_len).buf(3, m.spec).buf(4, m.pos).buf(5, hist)
        .buf(6, ngtab).buf(7, ids_chk).args(8, na);
    e.threads(S3(m.ngram.heads, maxn, S), S3(m.ngram.heads, 1, 1));
  }

  // After the target step over T rows of S sequences: save each sequence's last committed hx row, then run the MTP
  // layer over every row with its next token (host-filled `next`, fixed on the GPU for speculative sequences).
  // last_prompt[b]: sequence b's last row completes a greedy prompt (its next token is the target's argmax).
  void encode_context(Enc& e, id<MTLCommandBuffer> __strong& cb, int T, int S, bool any_spec,
                      const std::vector<int>& last_prompt) {
    struct { int HH; } sa{HH};
    e.k("mtp_save_h").buf(0, m.hx).buf(1, m.commit).buf(2, m.seq_start).buf(3, m.slot).buf(4, th).args(5, sa);
    e.threads(S3(HH, S), S3(256));
    bool any_last = false;
    for (int b = 0; b < S; b++) { lastp.ptr<int>()[b] = last_prompt[b]; any_last |= last_prompt[b] != 0; }
    if (any_spec || (fold && any_last)) {
      e.k("mtp_next_spec").buf(0, m.tok).buf(1, m.commit).buf(2, m.emit).buf(3, m.seq_start).buf(4, m.seq_len)
          .buf(5, m.spec).buf(6, next).buf(7, m.pred).buf(8, m.out_base).buf(9, lastp);
      e.threads(S3(S), S3(64));
    }
    fuse(e, next, m.hx, T);
    layer(e, cb, T, S);
    if (!fold || !(any_spec || any_last)) return;
    // Each sequence's last committed row: its MTP state per slot and (mixer, lm_head, argmax) its d1.
    e.k("mtp_fold_out").buf(0, hs).buf(1, m.commit).buf(2, m.seq_start).buf(3, m.slot).buf(4, xs).buf(5, hin).args(6, sa);
    e.threads(S3(HH, S), S3(256));
    std::swap(m.hx, hin);
    m.hyper_pre(e, m.mtp_mixer, S, false, m.blk_out);
    std::swap(m.hx, hin);
    m.a8_ready = false;
    m.gemm(e, m.blk_out, m.lm_head, flog, S, 0);  // not m.logits: the host samples from it after the step
    m.flush_split(e);
    e.k("row_argmax").buf(0, flog).buf(1, dpred).args(2, m.cfg.vocab);
    e.groups(S3(S), S3(1024));
    e.k("mtp_scatter_slot").buf(0, dpred).buf(1, m.slot).buf(2, d1s).args(3, S);
    e.threads(S3(S), S3(64));
  }
};
