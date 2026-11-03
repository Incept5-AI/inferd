// Qwen3.5 / Qwen3.8 hybrid text model (dense MLP variant), batched decode.
// Layers: Gated DeltaNet (3 of 4) and gated full attention (1 of 4), each followed by a SwiGLU MLP.
#pragma once
#include <sys/sysctl.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

// GPU core count from the IORegistry (AGXAccelerator "gpu-core-count"), or 0 when unavailable.
static int gpu_cores() {
  io_service_t s = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  if (!s) return 0;
  int n = 0;
  if (CFTypeRef v = IORegistryEntryCreateCFProperty(s, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0)) {
    if (CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &n);
    CFRelease(v);
  }
  IOObjectRelease(s);
  return n;
}
#include <array>
#include <string>
#include <thread>
#include <vector>
#include "gpu.h"
#include "weights.h"
#include "ane.h"
#include "tree.h"
#include "moe.h"
#include "qwen4.h"

// Argument structs; must match kernels/core.metal.
struct GemmArgs { int M, N, K, xr = 0; };  // xr: XOR mask for flipped 4-bit nibbles (W4A8)
struct EmbedArgs { int D; };
struct NormArgs { int D, in_stride, out_stride; float eps; };
struct GdnArgs { int S, Hk, Hv, Dk, Dv, conv_dim, proj_stride; float eps; int verify; int sig = 0; };  // sig: sigmoid output gate (Flash-Next)
struct CapArgs { int H, stride, col0; };
struct ReplayArgs { int conv_dim, Hk, Hv, Dk, Dv; };
// Largest speculative block (staged token + drafts); also the emit stride per sequence (spec.metal MAX_BLOCK).
// 16 since 2026-11-02 (DFlash2 at block 16 for one or a few streams).
constexpr int kMaxBlock = 16;
struct GatherKvArgs { int HD, n, max_blocks, seq, Dh, ns; };
// MLX steel GEMMSpiltKParams (sic, steel/gemm/params.h)
struct MlxSplitKParams {
  int M, N, K, lda, ldb, ldc, tiles_n, tiles_m, split_k_partitions, split_k_partition_stride, split_k_partition_size,
      swizzle_log, gemm_k_iterations_aligned;
};
// MLX steel GEMMParams (steel/gemm/params.h)
struct MlxGemmParams {
  int M, N, K, lda, ldb, ldd, tiles_n, tiles_m;
  int64_t batch_stride_a, batch_stride_b, batch_stride_d;
  int swizzle_log, gemm_k_iterations_aligned, batch_ndim;
};
struct GateArgs { int Hq, Dh, row0, proj_stride; };
// MLX steel AttnParams (third_party/mlx/.../steel/attn/params.h)
struct MlxAttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  int window = 0;  // patched steel_attention_nax.h: sliding window
};
struct ActArgs { int I; };
struct SplitArgs { int M, N, K, k_len; };
struct ReduceArgs { int M, N, S, add; };
struct SplitRed { int M, N, S; };
struct QmstArgs { int M, N, K, steps, split, spread; };
struct A8Args { int M, K; };
struct W4A8Args { int M, N, K, gps, split; };
struct FlipArgs { uint32_t n32, mask; };
struct GqaArgs { int T, Hq, Hkv, Dh, max_blocks, splits, chunk, proj_stride; float scale; int qtiles = 1;
                 int own_splits = 0; int ns = 0; int win = 0; int gated = 1; };  // ns: slots per KV head plane (n_blocks * KV_BLOCK)  // own_splits > 0: cascade step, slots [own_splits, splits) hold shared-prefix partials
constexpr int kAttnPartRows = 8192;
constexpr int kVTopShards = 64;  // verify rows' sharded top-16 (speculative sampling)  // T * splits capacity of the attention partials
struct AttnArgs { int B, Hq, Hkv, Dh, rot, proj_stride, max_blocks; float eps, theta, scale; int ns;
                  int qs = 0, koff = 0, voff = 0, rpair = 0, rden = 0, vnorm = 0; };  // set per layer in encode_step
constexpr int KV_BLOCK = 32;
constexpr int kSplitRows = 1024;
constexpr int kSplitMaxN = 34816;  // widest output the split-K scratch is sized for (pmax)

struct Qwen35Config {
  int hidden, inter, layers, vocab, interval = 0;
  int hq, hkv, dh, rot;
  int lhk = 0, lhv = 0, ldk = 0, ldv = 0, conv_k = 0;
  float eps, theta;
  int experts = 0, topk = 0, moe_inter = 0;  // MoE (qwen3_5_moe): routed experts; inter is then the shared expert's

  // Parsed, then validated centrally (Codex review 2026-11-03, R01): a malformed config.json stops the load with a
  // message instead of throwing out of a constructor or reaching a zero divisor or an int overflow later.
  explicit Qwen35Config(const json::Value& root) {
    try {
      parse(root);
    } catch (std::exception& e) {
      fprintf(stderr, "invalid config.json: %s\n", e.what());
      exit(1);
    }
    validate();
  }
  [[noreturn]] static void bad_config(const std::string& what) {
    fprintf(stderr, "invalid or unsupported config.json: %s\n", what.c_str());
    exit(1);
  }
  void validate() const {
    auto in = [](int64_t v, int64_t lo, int64_t hi) { return v >= lo && v <= hi; };
    if (!in(hidden, 64, 1 << 16) || hidden % 64) bad_config("hidden_size " + std::to_string(hidden));
    if (!in(layers, 1, 1024)) bad_config("num_hidden_layers " + std::to_string(layers));
    if (!in(vocab, 1, 1 << 24)) bad_config("vocab_size " + std::to_string(vocab));
    if (!in(hq, 1, 1024) || !in(hkv, 1, 1024) || hq % hkv) bad_config("attention heads " + std::to_string(hq) + "/" + std::to_string(hkv));
    if (!in(dh, 1, 1024) || dh % 2) bad_config("head_dim " + std::to_string(dh));
    if (!in(rot, 0, dh) || rot % 2) bad_config("rotary dims " + std::to_string(rot));
    if (!in(inter, 0, 1 << 20)) bad_config("intermediate_size " + std::to_string(inter));
    if (!(std::isfinite(eps) && eps > 0) || !std::isfinite(theta) || theta < 0) bad_config("rms_norm_eps / rope_theta");
    if (!std::isfinite(post_eps) || !std::isfinite(qk_scale) || !std::isfinite(out_mult) || !std::isfinite(softcap))
      bad_config("non-finite norm/scale constants");
    if (!in(interval, 0, layers)) bad_config("full_attention_interval " + std::to_string(interval));
    if (!kind.empty() && (int)kind.size() != layers) bad_config("layer_types has " + std::to_string(kind.size()) + " entries");
    if ((glimmer || gemma) && (int)kind.size() != layers) bad_config("layer_types is required for this model type");
    if (window < 0) bad_config("sliding_window");
    if (interval) {  // GDN layers: divisors and the state geometry
      if (!in(lhk, 1, 1024) || !in(lhv, 1, 1024) || lhv % lhk || !in(ldk, 1, 1024) || !in(ldv, 1, 1024) || !in(conv_k, 2, 64))
        bad_config("linear attention geometry");
    }
    if (experts && (!in(experts, 1, 1 << 16) || !in(topk, 1, experts) || !in(moe_inter, 1, 1 << 20)))
      bad_config("MoE geometry " + std::to_string(experts) + "/" + std::to_string(topk));
    if (gemma && (!in(gdh, 1, 1024) || !in(ghkv, 1, 1024) || hq % ghkv || !in(grot, 0, gdh) || grot % 2))
      bad_config("global attention geometry");
    if (qwen4) {
      if (!in(hc, 1, 64) || !in(hc_low, 0, 1 << 16)) bad_config("hc_count / hc_lowrank");
      for (int l : ple_ids) if (!in(l, 0, layers - 1)) bad_config("ple_layer_ids");
    }
    if (qsa && (!in(qsa_heads, 1, 64) || !in(qsa_dim, 32, 256) || !in(qsa_ratio, 1, KV_BLOCK) || !in(qsa_topk, 0, 1 << 24)))
      bad_config("QSA indexer geometry");
  }
  void parse(const json::Value& root) {
    const json::Value& c = root.has("text_config") ? root["text_config"] : root;
    hidden = c["hidden_size"].i();
    if (c.has("num_experts") && c["num_experts"].i() > 0) {
      experts = c["num_experts"].i();
      topk = c["num_experts_per_tok"].i();
      moe_inter = c["moe_intermediate_size"].i();
      inter = c["shared_expert_intermediate_size"].i();
    } else {
      inter = c["intermediate_size"].i();
    }
    layers = c["num_hidden_layers"].i();
    vocab = c["vocab_size"].i();
    interval = c["full_attention_interval"].i();
    hq = c["num_attention_heads"].i();
    hkv = c["num_key_value_heads"].i();
    dh = c["head_dim"].i();
    const json::Value& rp = c["rope_parameters"];
    float prf = rp.has("partial_rotary_factor") && rp["partial_rotary_factor"].type == json::Value::Num ? rp["partial_rotary_factor"].n
                : c.has("partial_rotary_factor") ? c["partial_rotary_factor"].n : 1.f;
    theta = rp.has("rope_theta") ? rp["rope_theta"].n : c.has("rope_theta") ? c["rope_theta"].n : 0.f;
    rot = (int)(dh * prf);
    eps = c["rms_norm_eps"].n;
    std::string mt = c.has("model_type") ? c["model_type"].s : "";
    glimmer = mt.rfind("muse_glimmer", 0) == 0;
    gemma = mt.rfind("gemma4", 0) == 0;
    qwen4 = mt.rfind("qwen4_exp", 0) == 0;
    if (qwen4) {
      // MTP=1 (or DRAFT=mtp): load the checkpoint's own MTP head (mtp.*) as a drafter; its KV is one more attention
      // layer's pool in every KV block, so the server's pool sizing (kv_block_bytes) must know before loading.
      const char* dr = getenv("DRAFT");
      const bool want = (getenv("MTP") && atoi(getenv("MTP"))) || (dr && std::string(dr) == "mtp");
      if (want && c.has("mtp_num_hidden_layers") && c["mtp_num_hidden_layers"].i() > 0) mtp = 1;
      hc = c["hc_count"].i();
      hc_low = c["hc_lowrank"].i();
      for (auto& v : c["ple_layer_ids"].a) ple_ids.push_back(v.i() - 1);  // one-indexed in the config
      // QSA sparse-attention indexer (QSA=0: off, every attention layer dense, as before 2026-11-02)
      if (c.has("indexer_head_dim") && (!getenv("QSA") || atoi(getenv("QSA")))) {
        qsa_heads = c["indexer_n_heads"].i();
        qsa_dim = c["indexer_head_dim"].i();
        qsa_ratio = c["indexer_compress_ratio"].i();
        if (qsa_ratio <= 0) bad_config("indexer_compress_ratio");  // a divisor
        qsa_topk = c["indexer_budget"].i() / qsa_ratio;
        qsa = c["indexer_kv_heads"].i() == 1 && qsa_heads == 4 && qsa_dim <= 256 && qsa_dim % 32 == 0 &&
              KV_BLOCK % qsa_ratio == 0 && hq == 12 * hkv && dh == 256;
        if (!qsa) fprintf(stderr, "QSA: unsupported indexer shape, attention stays dense\n");
      }
    }
    if (c.has("layer_types"))
      for (auto& t : c["layer_types"].a) kind.push_back(t.s == "linear_attention" ? 0 : t.s == "sliding_attention" ? 1 : 2);
    if (c.has("sliding_window") && c["sliding_window"].type != json::Value::Null) window = c["sliding_window"].i();
    if (glimmer) {
      post_eps = c["post_norm_eps"].n;
      qk_scale = c["qk_scale_factor"].n;
      out_mult = c["output_multiplier"].n;
      softcap = c["final_logit_softcapping"].n;
      interval = 0;
      return;
    }
    if (gemma) {  // sliding layers: dh/hkv/theta of rope_parameters.sliding_attention; global: the global_* fields
      post_eps = eps;
      softcap = c["final_logit_softcapping"].n;
      gdh = c["global_head_dim"].i();
      ghkv = c["num_global_key_value_heads"].i();
      k_eq_v = c.has("attention_k_eq_v") && c["attention_k_eq_v"].b;
      const json::Value& rs = rp["sliding_attention"];
      const json::Value& rf = rp["full_attention"];
      theta = rs["rope_theta"].n;
      gtheta = rf["rope_theta"].n;
      grot = (int)(gdh * (rf.has("partial_rotary_factor") ? rf["partial_rotary_factor"].n : 1.f));
      rot = dh;
      interval = 0;
      return;
    }
    interval = c["full_attention_interval"].i();
    lhk = c["linear_num_key_heads"].i();
    lhv = c["linear_num_value_heads"].i();
    ldk = c["linear_key_head_dim"].i();
    ldv = c["linear_value_head_dim"].i();
    conv_k = c["linear_conv_kernel_dim"].i();
  }
  // Dense attention families (no GDN): Muse Glimmer. kind per layer: 0 GDN, 1 sliding attention, 2 full attention.
  bool glimmer = false, gemma = false;
  // Qwen3.8-Flash-Next (qwen4_exp): hc residual streams mixed by a rank-hc_low gate around every block (no input
  // norms), n-gram per-layer embeddings on ple_ids, sigmoid-gated GDN output, 512-expert MoE with a bf16 router.
  bool qwen4 = false;
  int hc = 1, hc_low = 0;
  int mtp = 0;  // Flash-Next MTP head loaded (one extra attention layer with its own KV pool)
  std::vector<int> ple_ids;
  // Gemma-4: global (full) layers have their own head dim / kv heads / rope (proportional, partial), K = V there,
  // a scaleless norm on V, GELU MLP, a per-layer output scalar, embeddings x sqrt(hidden), attention scale 1.
  int gdh = 0, ghkv = 0, grot = 0;
  float gtheta = 0;
  bool k_eq_v = false;
  bool sandwich() const { return glimmer || gemma; }
  bool global(int l) const { return !kind.empty() && kind[l] == 2; }
  int layer_dh(int l) const { return gemma && global(l) ? gdh : dh; }
  int layer_hkv(int l) const { return gemma && global(l) ? ghkv : hkv; }
  int max_hd() const { int m = 0; for (int l = 0; l < layers; l++) if (!is_linear(l)) m = std::max(m, layer_dh(l) * layer_hkv(l)); return m; }
  int max_dh() const { return gemma ? std::max(dh, gdh) : dh; }
  // KV pool bytes per 32-token block over every attention layer (int8 values + a float scale per head, or bf16)
  size_t kv_block_bytes(bool kv8) const {
    size_t b = 0;
    for (int l = 0; l < layers; l++)
      if (!is_linear(l)) b += (size_t)KV_BLOCK * layer_hkv(l) * (kv8 ? layer_dh(l) + 4 : layer_dh(l) * 2) * 2;
    b += (size_t)mtp * KV_BLOCK * hkv * (kv8 ? dh + 4 : dh * 2) * 2;  // the MTP layer (same shape as the target's)
    b += (size_t)qsa_planes(kv8) * KV_BLOCK * hkv * (kv8 ? dh + 4 : dh * 2) * 2;  // QSA index keys (extra KV planes)
    return b;
  }
  // QSA (Flash-Next): an indexer per attention layer (and the MTP layer's). qsa_topk groups of qsa_ratio tokens kept.
  bool qsa = false;
  int qsa_heads = 0, qsa_dim = 0, qsa_ratio = 0, qsa_topk = 0;
  int qsa_layers() const { return qsa ? attn_layers() + mtp : 0; }
  // The last position attended densely: a row at p sees (p + 1) / ratio complete groups, all kept while <= topk.
  int qsa_dense_end() const { return (qsa_topk + 1) * qsa_ratio - 1; }
  // Index-key storage per KV block and layer: raw keys [32][qsa_dim] and pooled keys [32 / ratio][qsa_dim], bf16,
  // packed in order (every layer's raw chunk, then every pooled chunk) into regions of one head plane's block bytes
  // U = KV_BLOCK * dh * elem; regions are (kc, head 0..hkv-1), (vc, heads) of extra KV "layers" (kv planes), so the
  // block allocator, prefix cache, preemption and SSD tier carry them with the KV. Returns (region, byte offset).
  std::vector<std::pair<int, size_t>> qsa_chunks(bool kv8) const {
    std::vector<std::pair<int, size_t>> out;
    if (!qsa) return out;
    const size_t U = (size_t)KV_BLOCK * dh * (kv8 ? 1 : 2), raw = (size_t)KV_BLOCK * qsa_dim * 2,
                 pool = (size_t)KV_BLOCK / qsa_ratio * qsa_dim * 2;
    int r = 0;
    size_t off = 0;
    for (int pass = 0; pass < 2; pass++)
      for (int i = 0; i < qsa_layers(); i++) {
        const size_t sz = pass ? pool : raw;
        if (off + sz > U) { r++; off = 0; }
        out.push_back({r, off});
        off += sz;
      }
    return out;
  }
  int qsa_planes(bool kv8) const {
    auto ch = qsa_chunks(kv8);
    return ch.empty() ? 0 : (ch.back().first + 1 + 2 * hkv - 1) / (2 * hkv);
  }
  std::vector<int> kind;
  int window = 0;
  float post_eps = 0, qk_scale = 1, out_mult = 1, softcap = 0;
  int attn_layers() const { int n = 0; for (int l = 0; l < layers; l++) n += !is_linear(l); return n; }
  bool is_linear(int l) const { return interval ? (l + 1) % interval != 0 : false; }
  float attn_scale(int d = 0) const { return gemma ? 1.f : qk_scale / sqrtf((float)(d ? d : dh)); }
  int layer_window(int l) const { return !kind.empty() && kind[l] == 1 ? window : 0; }
  int layer_rot(int l) const {  // Glimmer: NoPE global layers
    return glimmer ? (kind[l] == 1 ? dh : 0) : gemma ? (global(l) ? grot : dh) : rot;
  }
  int conv_dim() const { return 2 * lhk * ldk + lhv * ldv; }
  int gdn_proj() const { return conv_dim() + lhv * ldv + 2 * lhv; }
  int attn_proj() const { return hq * dh * 2 + 2 * hkv * dh; }
};

struct Layer {
  bool linear;
  BufRef in_norm, post_norm;
  QLinear in_proj, out_proj;  // GDN: [qkv|z|b|a] and out_proj. Attention: [q|k|v] and o_proj.
  QLinear gate_up, down;  // MoE: the shared expert
  MoeW router, egu, edown;  // MoE: router (+ shared gate row), routed experts' gate|up and down
  BufRef conv_w, A_log, dt_bias, gdn_norm;  // GDN only
  BufRef q_norm, k_norm;                    // attention only
  BufRef post_attn_norm, post_ffn_norm;     // sandwich norms (Glimmer), applied to the block outputs
  int window = 0, rot = 0;                  // per-layer sliding window and rotary dims
  int hkv = 0, dh = 0, proj = 0;            // attention: kv heads, head dim, in_proj row length
  bool gated = true, k_eq_v = false;        // attention output gate in q_proj (Qwen/Glimmer); V = K projection
  float theta = 0, scalar = 1;              // rope base; Gemma's per-layer output scalar
  struct Hyper { BufRef norm; QLinear dn, up; };  // Flash-Next hyper-connection: norm, [down | inject] (padded), up
  Hyper ahc, mhc;
  bool ple = false;                         // Flash-Next per-layer n-gram embedding
  QLinear ple_kv;                           // [key_proj (hc * H) | value_proj (H)]
  BufRef ple_nk, ple_nq, ple_nc, ple_conv;  // norm_key / norm_query / norm_conv weights, conv [hc * H, K]
  int idx;                                  // index among layers of its kind (state/KV pool slot)
  QLinear iproj;                            // QSA indexer: index_qk_proj [Hi queries | 1 key] x qsa_dim
  BufRef iqn, ikn;                          // QSA q_layernorm / k_layernorm weights (zero-centred, raw)
  bool has_qsa = false;
  int iproj_fused = 0;                      // > 0: index rows are in_proj's rows from here (FN_FUSE_IPROJ)
};

struct Qwen35 {
  Gpu& g;
  std::string skip = getenv("SKIP") ? getenv("SKIP") : "";
  static int env_int(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
  // Knobs re-read by decode_test's BENCH_VARIANTS (timing several configurations on one load).
  void knobs() {
    hc_v2 = env_int("HC_V2", 4);
    moe_v2 = env_int("MOE_V2", 4);
    moe_dedup = env_int("MOE_DEDUP", 1);
    fn_split_row = env_int("FN_SPLIT_ROW", 128);
    fn_mpp_min = env_int("FN_MPP_MIN", 1);
    fn_mpp = cfg.qwen4 && env_int("FN_MPP", 1) && cfg.hq == 12 * cfg.hkv && cfg.dh == 256 && use_gqa && g.nax;
    fn_split_multi = env_int("FN_SPLIT_MULTI", 32);
    qsa_dense = env_int("QSA_DENSE", 0);
    fn_gdn_pf = env_int("FN_GDN_PF", 16);
    fn_gdn_r = env_int("FN_GDN_R", 4);
    fn_router_gemm = env_int("FN_ROUTER_GEMM", 64);
    fn_gqmm_bm = env_int("FN_GQMM_BM", 0);
    fn_qsa_splits = env_int("FN_QSA_SPLITS", 0);
    fn_inj_norm = env_int("FN_INJ_NORM", 1);  // 4 measured no faster (16k prefill) once batches hold 128 tiles
  }
  int fn_gqmm_bm = 0, fn_qsa_splits = 0, fn_inj_norm = 1;
  const char* fn_gqmm_sfx = getenv("FN_GQMM_SFX");  // experiments: "_w2" / "_n4" tiles (kernels/fnp_gather.metal)
  // Flash-Next prefill (2026-11-02 night): GDN conv / recurrence kernels for steps of >= fn_gdn_pf rows (fn_gdn_r
  // value rows per simdgroup); the router as a bf16 GEMM for steps of >= fn_router_gemm rows. 0 turns each off.
  int fn_gdn_pf = 16, fn_gdn_r = 4, fn_router_gemm = 64;
  // GDN_PF_ALL (engine-pf 2026-11-02): the Flash-Next prefill GDN kernels (conv per (channel, row), recurrence with
  // FN_GDN_R value rows per simdgroup; same arithmetic, bit-identical) for every GDN model, not just Flash-Next.
  int gdn_pf_all = getenv("GDN_PF_ALL") ? atoi(getenv("GDN_PF_ALL")) : 1;
  int qsa_dense = 0;  // QSA_DENSE=1: every row attends densely (the indexer cache is still kept); A/B and timing
  bool on(const char* what) const { return skip.find(what) == std::string::npos; }
  Qwen35Config cfg;
  std::vector<Layer> L;
  QLinear embed, lm_head;
  BufRef final_norm;

  // Capacity: a step carries up to max_tokens token rows from up to max_seqs sequences.
  int max_tokens, max_seqs, n_slots, n_blocks, max_blocks;
  // Activations (sized for max_tokens; logits for max_seqs)
  BufRef qpack;  // packed verify queries for attn_mpp
  BufRef qscale;  // IQ: per packed row query scales
  BufRef kbeg, shrow, sh_rows_t, sh_rows_j, sh_tinfo, qpack_sh;  // cascade attention scratch
  BufRef a8u, a8x;  // W4A8 activation scratch: uint8 [T, Kmax] and float2 (scale, sum) [T, Kmax/64]
  int a8_kmax = 0;  // Kmax the scratch is sized for (reserve_a8 grows it for wider GEMMs, e.g. the drafter's fc)
  // Grow the W4A8 activation scratch for a GEMM of K columns over up to max_tokens rows. The DFlash2 drafter's fc
  // has K = 5 x hidden = 25,600 > the target's widest (17,408): until 2026-11-01 its context append overflowed
  // a8u/a8x whenever it ran more than ~0.68 x max_tokens rows (a prompt's prefill chunk), corrupting the context
  // ring with NaN; on the M4 Pro this drafted junk (1.0 tokens/cycle) and the server emitted garbage.
  void reserve_a8(int K) {
    if (!use_w4a8 || K <= a8_kmax) return;
    a8u = g.alloc((size_t)max_tokens * K);
    a8x = g.alloc((size_t)max_tokens * (K / 64) * 8);
    a8_kmax = K;
  }
  BufRef rlog, ridx, rw, rsg, egu_out, eact, edown_out, sh_out, moffs, mperm, minv, mxs;  // MoE scratch
  BufRef x, xn, tmp, splitk, attn_part, proj, conv_out, gbuf, betabuf, ybuf, act, qbuf, logits;
  // Per-token inputs: tok, pos, tseq (index of the token's sequence in this step).
  // Per-sequence inputs: slot, block_table row, seq_start/seq_len (its token rows), out_row (row for logits).
  BufRef tok, pos, tseq, slot, block_table, seq_start, seq_len, out_row;
  // Speculative decoding (enable_spec): per-sequence spec flag and first logits row, per-row argmax, accept
  // results, captured hidden states [T, n_cap * hidden], and per-GDN-layer verify tapes.
  BufRef spec, out_base, pred, commit, emit, hcap;
  BufRef samp, vcand, vun, vcpart, vvpart;  // speculative sampling: per-sequence params, verify rows' top-16
  bool any_sample = false;
  // Multi-row attention: sequences with >= flash_rows new rows use MLX's flash kernel over a gathered copy of
  // their KV (rflag marks their rows so attn_gqa skips them).
  BufRef rflag, kv_k, kv_v;
  // Per-row rope positions [T, 3] and multimodal embedding overrides (ext flags, xin rows).
  BufRef rpos, ext, xin;
  bool any_ext = false;
  struct FlashSeq { int seq, row0, n, kL; };
  std::vector<FlashSeq> flash;
  int flash_rows = getenv("FLASH_ROWS") ? atoi(getenv("FLASH_ROWS")) : 64;
  std::vector<int> cap_layers;
  std::vector<BufRef> tape_x, tape_c, tape_g, tape_b;
  int n_out = 0, max_out = 0;
  size_t mem_tapes = 0;  // bytes of the GDN verify tapes (enable_spec)
  // Token trees (TREE, src/tree.h, kernels/tree.metal; 2026-11-02). A SeqWork with `tree` verifies a draft tree in
  // DFS preorder instead of a chain: per row trow (node index, -1 elsewhere), tpar (parent node), tdep (depth), tanc
  // (ancestor mask); apos = the attention position (pos0 - 1 for tree rows: the main kernels see only the committed
  // context, attn_tree the ancestors); seq_len_g / spec_g hide tree sequences from the chain GDN kernels and replays.
  // Defaults written by set_batch make a tree sequence a chain; set_tree() installs a real tree before the verify runs.
  BufRef apos, trow, tpar, tdep, tanc, tseqs, tpath, tpart, seq_len_g, spec_g;
  int n_tree = 0;
  bool tree_only = false;  // every sequence of the step is a tree: the chain GDN kernels are not dispatched at all
  // TREE_SKIP (timing only, wrong results): 1 attn_tree, 2 gdn_conv_tree, 4 gdn_step_tree, 8 tree replays, 16 kv_compact
  int tree_skip = getenv("TREE_SKIP") ? atoi(getenv("TREE_SKIP")) : 0;
  bool verify = false;
  // Recurrent state per GDN layer, KV pools per attention layer
  std::vector<BufRef> conv_state, ssm_state, kc, vc;
  // 8-bit KV (KV8=1): int8 K/V in kc/vc plus float scales per (position, kv head) in ks/vs.
  // default since 2026-10-31 (KV8=0: bf16). Gemma-4 defaults to bf16: its attention scale of 1 on normed q/k makes
  // int8 key error flip predictions (KL 1.96 vs 0.06 on a repetition prompt, 2026-11-01).
  bool kv8 = getenv("KV8") ? atoi(getenv("KV8")) != 0 : !cfg.gemma;
  std::vector<BufRef> ks, vs;
  size_t kv_elem() const { return kv8 ? 1 : 2; }
  // KV layout: per layer, one plane per kv head, [Hkv][ns][Dh] (scales [Hkv][ns]); slot = block * KV_BLOCK + t.
  int kv_ns() const { return n_blocks * KV_BLOCK; }
  // 8-bit attention (with KV8): int8 queries x int8 keys on the integer path for verify blocks (ATTN_IQ=0 off).
  bool attn_iq = !getenv("ATTN_IQ") || atoi(getenv("ATTN_IQ"));
  id<MTLComputePipelineState> kvk(const char* name) { return g.pso_fc(name, {{20, kv8}}); }

  Qwen35(Gpu& g, const Checkpoint& ck, int max_tokens, int max_seqs, int n_slots, int n_blocks, int max_blocks)
      : g(g), cfg(ck.config), max_tokens(max_tokens), max_seqs(max_seqs), n_slots(n_slots), n_blocks(n_blocks),
        max_blocks(max_blocks) {
    knobs();
    std::string P = ck.has("language_model.model.embed_tokens.weight") ? "language_model." : "";
    embed = load_qlinear_w(g, ck, {P + "model.embed_tokens"});
    if (cfg.qwen4) mixer = load_hyper(ck, P + "model.hyper_connection_mixer.", false);
    lm_head = load_qlinear_w(g, ck, {ck.has(P + "lm_head.weight") ? P + "lm_head" : P + "model.embed_tokens"});  // tied: Gemma
    if (!cfg.qwen4) final_norm = ck.load(g, P + "model.norm.weight", cfg.hidden);
    // Cross-tensor shapes (Codex review 2026-11-03, R01): every token id below vocab must have an embedding row and a
    // logit, and the layers' linears must chain through hidden.
    check_linear(embed, P + "model.embed_tokens", -cfg.vocab, cfg.hidden);
    check_linear(lm_head, "lm_head", -cfg.vocab, cfg.hidden);
    int ng = 0, na = 0;
    for (int l = 0; l < cfg.layers; l++) {
      Layer ly = load_layer(ck, P + "model.layers." + std::to_string(l) + ".", l, true);
      ly.idx = ly.linear ? ng++ : na++;
      L.push_back(ly);
    }
    if (cfg.mtp) load_mtp(ck, P, na);
    for (auto& ly : L) {  // attn_gqa(_row) are instantiated per (GQ, Dh); attn_paged_nax only for GQ=6, Dh=256
      if (ly.linear) continue;
      const int gq = cfg.hq / ly.hkv;
      if (!((gq == 6 || gq == 4 || gq == 8 || gq == 2 || gq == 12) && ly.dh == 256) && !(gq == 16 && ly.dh == 128) &&
          !(gq == 8 && ly.dh == 512)) use_gqa = false;
      if (gq != 6 || ly.dh != 256) use_nax = false;
    }
    if (kv8 && !use_gqa) { fprintf(stderr, "KV8 needs the attn_gqa kernels\n"); exit(1); }
    knobs();  // again: fn_mpp depends on use_gqa
    if (ng == 0) {
    } else if (auto& dt = ck.info(P + "model.layers.0.linear_attn.A_log").dtype; dt != "BF16" && dt != "F32") {
      fprintf(stderr, "expected BF16 or F32 A_log\n");
      exit(1);
    }

    const int B = max_tokens, H = cfg.hidden;
    int pmax = std::max({cfg.gdn_proj(), cfg.attn_proj(), 2 * cfg.inter});
    for (auto& ly : L) pmax = std::max(pmax, ly.proj);
    int amax = std::max({cfg.inter, cfg.lhv * cfg.ldv, cfg.hq * cfg.max_dh()});
    x = g.alloc((size_t)B * H * 2);
    xn = g.alloc((size_t)B * H * 2);
    tmp = g.alloc((size_t)B * H * 2);
    splitk = g.alloc((size_t)kSplitRows * pmax * 4);  // split-K float partials, S * M <= kSplitRows
    if (use_w4a8) {
      const int kmax = std::max(amax, H);
      a8u = g.alloc((size_t)B * kmax);
      a8x = g.alloc((size_t)B * (kmax / 64) * 8);
      a8_kmax = kmax;
      flip_weights();
    }
    if (!g.nax) {
      // No GPU neural accelerators (M4 Pro): MLX's NAX qmm (M >= W4A8_MAX) and flash attention (chunks of more than
      // CHUNK_TILE_MAX rows) give wrong logits at prefill sizes there (a 601-token prompt in one chunk: cos 0.29,
      // top-1 wrong; garbage text in the server). Keep every quantized GEMM on W4A8 and every multi-row chunk on
      // attn_shared (mode 6; attn_gqa when that is unavailable). 4k prefill on the M4 Pro: 110 tok/s, correct.
      // M3/M4 (Apple9): matmul2d's int8 path is emulated there, so from SG_MIN rows on the GEMMs go to MLX's
      // simdgroup qmm (kernels/mlx_qmm_sg.metal, what mlx_lm runs on these GPUs) instead of W4A8. SG_MIN=0: W4A8 only.
      if (!getenv("W4A8_MAX")) w4a8_max = sg_min > 0 ? sg_min : 1 << 30;
      if (!getenv("FLASH_ROWS")) flash_rows = 1 << 30;
      if (!getenv("CHUNK_TILE_MAX")) chunk_tile_max = 1 << 30;
    }
    proj = g.alloc((size_t)B * pmax * 2);
    if (cfg.qwen4) {
      const size_t HH = (size_t)cfg.hc * H;
      hx = g.alloc((size_t)B * HH * 2);
      hnorm = g.alloc((size_t)B * HH * 2);
      hup = g.alloc((size_t)B * HH * 2);
      hlow = g.alloc((size_t)B * 384 * 2);
      hmix = g.alloc((size_t)B * cfg.hc_low * 2);
      injw = g.alloc((size_t)B * cfg.hc * 4);
      injw_alt = g.alloc((size_t)B * cfg.hc * 4);
      hparts = g.alloc((size_t)B * cfg.hc * 384 * 4);
      blk_out = g.alloc((size_t)B * H * 2);
      if (ple_table.W) {
        ple_ids_buf = g.alloc((size_t)B * ngram.heads * 4);
        ple_emb = g.alloc((size_t)B * ngram.heads * ple_table.W * 2);
        ple_kvout = g.alloc((size_t)B * (HH + H) * 2);
        ple_k = g.alloc((size_t)B * HH * 2);
        ple_gv = g.alloc((size_t)B * HH * 2);
        ple_state = g.alloc((size_t)n_slots * kPleHist * HH * 2);
        memset(ple_state.ptr<char>(), 0, (size_t)n_slots * kPleHist * HH * 2);
      }
    }
    if (cfg.sandwich()) {
      blk_out = g.alloc((size_t)B * H * 2);
      ones = const_buf(H, 1.f);
    }
    if (cfg.experts) {
      const size_t R = (size_t)B * cfg.topk;
      rlog = g.alloc((size_t)B * (cfg.experts + 1) * 4);
      ridx = g.alloc(R * 4);
      rw = g.alloc(R * 4);
      rsg = g.alloc((size_t)B * 4);
      egu_out = g.alloc(R * 2 * cfg.moe_inter * 2);
      eact = g.alloc(R * cfg.moe_inter * 2);
      edown_out = g.alloc(R * H * 2);
      sh_out = g.alloc((size_t)B * H * 2);
      moffs = g.alloc((size_t)cfg.experts * 4);
      mperm = g.alloc(R * 4);
      minv = g.alloc(R * 4);
      mxs = g.alloc(R * H * 2);
      if (cfg.qwen4) moe_act2 = g.alloc((size_t)8 * (cfg.topk + 1) * cfg.moe_inter * 2);
      if (cfg.qwen4) moe_plan_buf = g.alloc((size_t)(1 + 8 * cfg.topk * 9) * 4);
    }
    if (const int units = getenv("ANE_UNITS") ? atoi(getenv("ANE_UNITS")) : 0; !cfg.experts && units > 0 && B >= 512 &&
        L[0].gate_up.bits == 4 && cfg.hidden % AneFfn::SEG == 0)
      ane = new AneFfn(g, cfg.hidden, cfg.inter, units, std::min(B, 2048));
    attn_part = g.alloc((size_t)std::max(kAttnPartRows, B) * cfg.hq * (cfg.max_dh() + 2) * 4);
    if (qsa_on) {
      for (auto& ly : L) if (!ly.linear && !ly.has_qsa) { fprintf(stderr, "QSA: an attention layer has no indexer\n"); exit(1); }
      if (has_mtp && !mtp_ly.has_qsa) { fprintf(stderr, "QSA: the MTP layer has no indexer\n"); exit(1); }
      const int Di = cfg.qsa_dim, Hi = cfg.qsa_heads;
      qsa_proj = g.alloc((size_t)B * (Hi + 1) * Di * 2);
      qsa_iq = g.alloc((size_t)B * Hi * Di * 4);
      qsa_sel = g.alloc((size_t)B * cfg.qsa_topk * 4);
      qsa_rows = g.alloc((size_t)B * 4);
      // score scratch: rows x groups (float), processed in row groups that fit (QSA_SC_MB, default 64)
      const size_t maxnb = (size_t)max_blocks * KV_BLOCK / cfg.qsa_ratio;
      qsa_sc_bytes = std::min((size_t)env_knob("QSA_SC_MB", 64, 1, 1 << 16) << 20, (size_t)B * maxnb * 4);  // bounded (R03)
      qsa_sc_bytes = std::max(qsa_sc_bytes, maxnb * 4);
      qsa_sc = g.alloc(qsa_sc_bytes);
      qsa_gmax = env_int("QSA_GATHER", 1) && fn_mpp ? std::min(B, env_int("QSA_GATHER_MAX", 128)) : 0;
      if (qsa_gmax) {
        qsa_P = (cfg.qsa_dense_end() + KV_BLOCK - 1) / KV_BLOCK;
        // union tiles: 4 rows' groups (4 K) x ratio tokens plus the rows' tail span (< 2 ratio)
        qsa_P2 = (4 * cfg.qsa_topk * cfg.qsa_ratio + 2 * cfg.qsa_ratio + KV_BLOCK - 1) / KV_BLOCK;
        // 128 union tiles per batch (was 32: a 2,048-row prefill chunk ran 16 batches of 64 threadgroups one after
        // another; 128 took 16k real-text prefill 4.69 -> 4.37 s, 2026-11-02 night; scratch ~1 GiB)
        qsa_umax = env_int("QSA_UNION", 1) ? std::min(B, env_int("QSA_UNION_MAX", 128)) : 0;
        const size_t ns2 = std::max((size_t)qsa_gmax * qsa_P, (size_t)qsa_umax * qsa_P2) * KV_BLOCK, H = cfg.hkv;
        qsa_ns2 = (int)ns2;
        qsa_ustart = g.alloc((size_t)B * 4);
        qsa_ulen = g.alloc((size_t)B * 4);
        qsa_ulist = g.alloc((size_t)std::max(1, qsa_umax) * 4 * cfg.qsa_topk * 4);
        qsa_ucnt = g.alloc((size_t)std::max(1, qsa_umax) * 4);
        qsa_gmask = g.alloc((size_t)std::max(1, qsa_umax) * qsa_P2 * KV_BLOCK);
        qsa_gk = g.alloc(ns2 * H * cfg.dh * kv_elem());
        qsa_gv = g.alloc(ns2 * H * cfg.dh * kv_elem());
        qsa_gks = g.alloc(kv8 ? ns2 * H * 4 : 16);
        qsa_gvs = g.alloc(kv8 ? ns2 * H * 4 : 16);
        qsa_gbt = g.alloc(ns2 / KV_BLOCK * 4);  // identity page table over the scratch pool
        for (size_t i = 0; i < ns2 / KV_BLOCK; i++) qsa_gbt.ptr<int>()[i] = (int)i;
        qsa_gpos = g.alloc((size_t)B * 4);
        qsa_gflag = g.alloc((size_t)B * 4);
        memset(qsa_gflag.ptr<int>(), 0, (size_t)B * 4);
        qsa_gstart = g.alloc((size_t)qsa_gmax * 4);
        qsa_glen = g.alloc((size_t)qsa_gmax * 4);
        const size_t qt = std::max(qsa_gmax, qsa_umax);
        qsa_gq = g.alloc(qt * H * 48 * cfg.dh * 2);
        qsa_gqs = g.alloc(qt * H * 48 * 4);
        qsa_gkb = g.alloc((size_t)qsa_gmax * 4);
        memset(qsa_gkb.ptr<int>(), 0, (size_t)qsa_gmax * 4);
      }
    }
    qpack = g.alloc((size_t)max_seqs * cfg.hkv * 48 * cfg.dh * 2);  // attn_mpp: 8 rows x 6 heads per kv head
    qscale = g.alloc((size_t)max_seqs * cfg.hkv * 48 * 4);
    {  // cascade (shared-prefix attention): per-sequence key starts, per-row flags, 64-row tiles of (row, head)
      const int max_tiles = (max_tokens * 6 + 47) / 48 + max_seqs;
      kbeg = g.alloc((size_t)max_seqs * 4);
      shrow = g.alloc((size_t)max_tokens * 4);
      sh_rows_t = g.alloc((size_t)max_tiles * 48 * 4);
      sh_rows_j = g.alloc((size_t)max_tiles * 48 * 4);
      sh_tinfo = g.alloc((size_t)max_tiles * 8);
      qpack_sh = g.alloc((size_t)max_tiles * cfg.hkv * 48 * cfg.dh * 2);
      memset(shrow.ptr<int>(), 0, (size_t)max_tokens * 4);
      memset(kbeg.ptr<int>(), 0, (size_t)max_seqs * 4);
    }
    conv_out = g.alloc((size_t)B * cfg.conv_dim() * 2);
    gbuf = g.alloc((size_t)B * cfg.lhv * 4);
    betabuf = g.alloc((size_t)B * cfg.lhv * 4);
    ybuf = g.alloc((size_t)B * cfg.lhv * cfg.ldv * 2);
    act = g.alloc((size_t)B * amax * 2);
    qbuf = g.alloc((size_t)B * cfg.hq * cfg.max_dh() * 2);
    max_out = max_seqs;
    logits = g.alloc((size_t)max_out * cfg.vocab * 2);
    spec = g.alloc(max_seqs * 4);
    samp = g.alloc(max_seqs * 16);
    out_base = g.alloc(max_seqs * 4);
    commit = g.alloc(max_seqs * 4);
    emit = g.alloc(max_seqs * kMaxBlock * 4);
    pred = g.alloc(max_out * 4);
    tok = g.alloc(B * 4);
    pos = g.alloc(B * 4);
    tseq = g.alloc(B * 4);
    rflag = g.alloc(B * 4);
    rpos = g.alloc(B * 3 * 4);
    ext = g.alloc(B * 4);
    xin = g.alloc((size_t)B * H * 2);
    kv_k = g.alloc((size_t)max_blocks * KV_BLOCK * cfg.max_hd() * 2);
    kv_v = g.alloc((size_t)max_blocks * KV_BLOCK * cfg.max_hd() * 2);
    slot = g.alloc(max_seqs * 4);
    seq_start = g.alloc(max_seqs * 4);
    seq_len = g.alloc(max_seqs * 4);
    out_row = g.alloc(std::max(max_seqs * kMaxBlock, 2 * kMaxTree) * 4);  // (token trees: up to kMaxTree rows)
    block_table = g.alloc((size_t)max_seqs * max_blocks * 4);
    for (int i = 0; i < ng; i++) {
      conv_state.push_back(g.alloc((size_t)n_slots * (cfg.conv_k - 1) * cfg.conv_dim() * 2));
      ssm_state.push_back(g.alloc((size_t)n_slots * cfg.lhv * cfg.ldv * cfg.ldk * 4));
    }
    alloc_kv(n_blocks);
  }

  // One decoder layer from tensors under prefix p. l: its index for per-layer config (window, rope, head shape);
  // ple_ok: load the per-layer n-gram embedding if l is a PLE layer. The caller sets idx.
  Layer load_layer(const Checkpoint& ck, const std::string& p, int l, bool ple_ok) {
    Layer ly;
    ly.linear = cfg.is_linear(l);
    ly.window = cfg.layer_window(l);
    ly.rot = cfg.layer_rot(l);
    ly.hkv = cfg.layer_hkv(l);
    ly.dh = cfg.layer_dh(l);
    ly.theta = cfg.gemma && cfg.global(l) ? cfg.gtheta : cfg.theta;
    if (cfg.qwen4) {  // hyper-connections instead of input norms; the PLE layer
      ly.ahc = load_hyper(ck, p + "attn_hyper_connection.", true);
      ly.mhc = load_hyper(ck, p + "mlp_hyper_connection.", true);
      if (ple_ok && std::find(cfg.ple_ids.begin(), cfg.ple_ids.end(), l) != cfg.ple_ids.end()) {
        const std::string q = p + "ple.";
        ly.ple = true;
        ly.ple_kv = load_qlinear_any(g, ck, {q + "key_proj", q + "value_proj"});
        ly.ple_nk = ck.load(g, q + "norm_key.weight");
        ly.ple_nq = ck.load(g, q + "norm_query.weight");
        ly.ple_nc = ck.load(g, q + "norm_conv.weight");
        ly.ple_conv = ck.load(g, q + "conv1d.weight");
        ngram.init(ck.config.has("text_config") ? ck.config["text_config"] : ck.config, ck,
                   q + "ple_embedding.layer_multipliers", (int)(std::find(cfg.ple_ids.begin(), cfg.ple_ids.end(), l) - cfg.ple_ids.begin()));
        fprintf(stderr, "loading the PLE n-gram table...\n");
        ple_table.load(g, ck, q + "ple_embedding.ngram_embedding.shards.");
      }
    } else if (cfg.gemma) {  // plain (not centered) sandwich norms, a GELU MLP below, and a per-layer output scalar
      ly.in_norm = ck.load(g, p + "input_layernorm.weight");
      ly.post_attn_norm = ck.load(g, p + "post_attention_layernorm.weight");
      ly.post_norm = ck.load(g, p + "pre_feedforward_layernorm.weight");
      ly.post_ffn_norm = ck.load(g, p + "post_feedforward_layernorm.weight");
      uint16_t sc = 0;
      ck.need_dtype(p + "layer_scalar", "BF16");
      ck.read_into(ck.info(p + "layer_scalar"), &sc, sizeof sc);
      ly.scalar = bf16_to_f(sc);
    } else if (cfg.glimmer) {  // centered norms: out = norm(x) * (1 + w)
      ly.in_norm = load_norm(ck, p + "input_layernorm.weight", 1.f);
      ly.post_attn_norm = load_norm(ck, p + "post_attention_layernorm.weight", 1.f);
      ly.post_norm = load_norm(ck, p + "pre_feedforward_layernorm.weight", 1.f);
      ly.post_ffn_norm = load_norm(ck, p + "post_feedforward_layernorm.weight", 1.f);
    } else {
      ly.in_norm = ck.load(g, p + "input_layernorm.weight");
      ly.post_norm = ck.load(g, p + "post_attention_layernorm.weight");
    }
    if (cfg.experts) {
      ly.gate_up = load_qlinear_any(g, ck, {p + "mlp.shared_expert.gate_proj", p + "mlp.shared_expert.up_proj"});
      ly.down = load_qlinear_any(g, ck, {p + "mlp.shared_expert.down_proj"});
      if (ck.info(p + "mlp.gate.weight").dtype == "BF16") ly.router = load_router16(ck, p + "mlp.");
      else ly.router = load_moe(g, ck, {p + "mlp.gate", p + "mlp.shared_expert_gate"});
      ly.egu = load_moe(g, ck, {p + "mlp.switch_mlp.gate_proj", p + "mlp.switch_mlp.up_proj"});
      ly.edown = load_moe(g, ck, {p + "mlp.switch_mlp.down_proj"});
    } else {
      ly.gate_up = load_qlinear_w(g, ck, {p + "mlp.gate_proj", p + "mlp.up_proj"});
      ly.down = load_qlinear_w(g, ck, {p + "mlp.down_proj"});
    }
    if (ly.linear) {
      std::string a = p + "linear_attn.";
      ly.in_proj = load_qlinear_any(g, ck, {a + "in_proj_qkv", a + "in_proj_z", a + "in_proj_b", a + "in_proj_a"});
      ly.out_proj = load_qlinear_any(g, ck, {a + "out_proj"});
      ly.conv_w = ck.load(g, a + "conv1d.weight");
      ly.A_log = ck.load_bf16(g, a + "A_log");
      ly.dt_bias = ck.load(g, a + "dt_bias");
      ly.gdn_norm = ck.load(g, a + "norm.weight");
    } else {
      std::string a = p + "self_attn.";
      if (cfg.gemma) {  // [q | k | v] without an output gate; global layers have no v_proj (V = the K projection)
        ly.gated = false;
        ly.k_eq_v = cfg.k_eq_v && cfg.global(l);
        if (ly.k_eq_v) ly.in_proj = load_qlinear_w(g, ck, {a + "q_proj", a + "k_proj"});
        else ly.in_proj = load_qlinear_w(g, ck, {a + "q_proj", a + "k_proj", a + "v_proj"});
        ly.q_norm = ck.load(g, a + "q_norm.weight");
        ly.k_norm = ck.load(g, a + "k_norm.weight");
      } else if (cfg.glimmer) {  // [q | gate | k | v], q and gate interleaved per head as in Qwen's gated q_proj
        ly.in_proj = load_qlinear_w(g, ck, {a + "q_proj", a + "gate_proj", a + "k_proj", a + "v_proj"});
        interleave_heads(ly.in_proj, cfg.hq, cfg.dh);
        ly.q_norm = const_buf(cfg.dh, 1.f);  // scaleless qk-norm; qk_scale_factor is in attn_scale()
        ly.k_norm = const_buf(cfg.dh, 1.f);
      } else if (cfg.qwen4) {  // Qwen3.5 gated attention with zero-centred q/k norms, plus the QSA indexer
        const std::string ix = a + "indexer.";
        // FN_FUSE_IPROJ (default on): the QSA index projection as extra rows of in_proj when that needs no widening
        // (q/k/v and the indexer both plain 4-bit g64, or both widened to 8-bit): one GEMM per attention layer.
        bool fuse = false;
        if (cfg.qsa && ck.has(ix + "index_qk_proj.weight") && env_int("FN_FUSE_IPROJ", 1)) {
          auto plain = [&](const std::vector<std::string>& v) {
            for (auto& p : v) { auto q = qspec(ck, p); if (q.bits != 4 || q.gs != 64) return false; }
            return true;
          };
          const bool p1 = plain({a + "q_proj", a + "k_proj", a + "v_proj"}), p2 = plain({ix + "index_qk_proj"});
          fuse = p1 == p2 && (p1 || !(getenv("Q4_NATIVE56") && atoi(getenv("Q4_NATIVE56"))));
        }
        if (fuse) ly.in_proj = load_qlinear_any(g, ck, {a + "q_proj", a + "k_proj", a + "v_proj", ix + "index_qk_proj"});
        else ly.in_proj = load_qlinear_any(g, ck, {a + "q_proj", a + "k_proj", a + "v_proj"});
        ly.q_norm = load_norm(ck, a + "q_norm.weight", 1.f);
        ly.k_norm = load_norm(ck, a + "k_norm.weight", 1.f);
        if (cfg.qsa) {
          if (!ck.has(ix + "index_qk_proj.weight")) { fprintf(stderr, "QSA: %sindex_qk_proj missing\n", ix.c_str()); exit(1); }
          if (fuse) {  // a row view of in_proj (only its shape is used: the GEMM is in_proj's)
            ly.iproj_fused = ly.in_proj.N - (cfg.qsa_heads + 1) * cfg.qsa_dim;
            ly.iproj = rows_of(ly.in_proj, ly.iproj_fused, (cfg.qsa_heads + 1) * cfg.qsa_dim);
          } else {
            ly.iproj = load_qlinear_any(g, ck, {ix + "index_qk_proj"});
          }
          ly.iqn = ck.load(g, ix + "q_layernorm.weight");
          ly.ikn = ck.load(g, ix + "k_layernorm.weight");
          ly.has_qsa = true;
        }
      } else {
        ly.in_proj = load_qlinear_w(g, ck, {a + "q_proj", a + "k_proj", a + "v_proj"});
        ly.q_norm = ck.load(g, a + "q_norm.weight");
        ly.k_norm = ck.load(g, a + "k_norm.weight");
      }
      ly.out_proj = load_qlinear_any(g, ck, {a + "o_proj"});
      ly.proj = ly.in_proj.N;
    }
    check_layer(ck, p, ly);
    return ly;
  }
  // A linear's [N, K] against the model's geometry (n < 0: at least -n rows, the embedding / lm_head may be padded;
  // 0: not checked). The kernels index by the config's dimensions, so a mismatch is a GPU out-of-bounds read.
  void check_linear(const QLinear& q, const std::string& name, int n, int k) const {
    if ((n > 0 && q.N != n) || (n < 0 && q.N < -n) || (k && q.K != k)) {
      fprintf(stderr, "invalid checkpoint: %s is [%d, %d], the config needs [%s%d, %d]\n", name.c_str(), q.N, q.K,
              n < 0 ? ">= " : "", n < 0 ? -n : n, k);
      exit(1);
    }
  }
  void check_layer(const Checkpoint& ck, const std::string& p, const Layer& ly) const {
    const int H = cfg.hidden;
    check_linear(ly.out_proj, p + "out_proj/o_proj", H, 0);
    if (cfg.experts) check_linear(ly.gate_up, p + "mlp.shared_expert.gate_up", 0, H);
    else check_linear(ly.gate_up, p + "mlp.gate_up", 0, H), check_linear(ly.down, p + "mlp.down_proj", H, ly.gate_up.N / 2);
    if (cfg.experts) check_linear(ly.down, p + "mlp.shared_expert.down_proj", H, ly.gate_up.N / 2);
    if (cfg.qwen4 || cfg.glimmer || cfg.gemma) {
      check_linear(ly.in_proj, p + "in_proj", 0, H);
      return;
    }
    // The Qwen3.5/3.6/3.8 hybrid: exact projections, state and norm sizes (n = elements of a vector tensor).
    auto numel = [&](const std::string& t, int64_t n) {
      auto& ti = ck.info(t);
      const size_t es = Checkpoint::dtype_size(ti.dtype);
      if (!es || ti.nbytes != (size_t)n * es) {
        fprintf(stderr, "invalid checkpoint: %s holds %zu bytes of %s, the config needs %lld values\n", t.c_str(), ti.nbytes,
                ti.dtype.c_str(), (long long)n);
        exit(1);
      }
    };
    numel(p + "input_layernorm.weight", H);
    numel(p + "post_attention_layernorm.weight", H);
    if (!cfg.experts) check_linear(ly.gate_up, p + "mlp.gate_up", 2 * cfg.inter, H);
    if (ly.linear) {
      const std::string a = p + "linear_attn.";
      check_linear(ly.in_proj, a + "in_proj", cfg.gdn_proj(), H);
      check_linear(ly.out_proj, a + "out_proj", H, cfg.lhv * cfg.ldv);
      numel(a + "conv1d.weight", (int64_t)cfg.conv_dim() * cfg.conv_k);
      numel(a + "A_log", cfg.lhv);
      numel(a + "dt_bias", cfg.lhv);
      numel(a + "norm.weight", cfg.ldv);
    } else {
      const std::string a = p + "self_attn.";
      check_linear(ly.in_proj, a + "q/k/v_proj", cfg.attn_proj(), H);
      check_linear(ly.out_proj, a + "o_proj", H, cfg.hq * cfg.dh);
      numel(a + "q_norm.weight", cfg.dh);
      numel(a + "k_norm.weight", cfg.dh);
    }
  }

  // Flash-Next MTP head (mtp.*, MTP=1): pre-fusion norms and projections, one gated-attention + MoE decoder layer with
  // its own KV pool (index na, one past the target's attention layers), and its own final hyper-connection mixer.
  // Shares embed_tokens and lm_head with the target. Driven by src/mtp.h.
  bool has_mtp = false;
  Layer mtp_ly;
  Layer::Hyper mtp_mixer;
  QLinear mtp_fc_e, mtp_fc_h;
  BufRef mtp_norm_e, mtp_norm_h;  // raw (zero-centred) weights: hc_norm applies 1 + w
  void load_mtp(const Checkpoint& ck, const std::string& P, int na) {
    std::string mp;
    for (const std::string& c : {std::string("mtp."), P + "mtp.", P + "model.mtp.", std::string("model.mtp.")})
      if (ck.has(c + "fc_embedding.weight")) { mp = c; break; }
    if (mp.empty()) { fprintf(stderr, "MTP: no mtp.* tensors in the checkpoint\n"); exit(1); }
    int la = 0;
    while (la < cfg.layers && cfg.is_linear(la)) la++;  // an attention layer's index for the per-layer config
    mtp_ly = load_layer(ck, mp + "layers.0.", la, false);
    mtp_ly.idx = na;
    mtp_mixer = load_hyper(ck, mp + "hyper_connection_mixer.", false);
    mtp_fc_e = load_qlinear_any(g, ck, {mp + "fc_embedding"});
    mtp_fc_h = load_qlinear_any(g, ck, {mp + "fc_hidden"});
    mtp_norm_e = ck.load(g, mp + "pre_fc_norm_embedding.weight");
    mtp_norm_h = ck.load(g, mp + "pre_fc_norm_hidden.weight");
    has_mtp = true;
    fprintf(stderr, "MTP head loaded from %s* (KV pool %d)\n", mp.c_str(), na);
  }

  // The paged KV pool (re-sized by the server when it sizes the pool from what is left of the GPU budget).
  void alloc_kv(int nb) {
    n_blocks = nb;
    kc.clear(); vc.clear(); ks.clear(); vs.clear();
    for (auto& ly : L) {
      if (ly.linear) continue;
      kc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * ly.hkv * ly.dh * kv_elem()));
      vc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * ly.hkv * ly.dh * kv_elem()));
      ks.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * ly.hkv * 4 : 16));
      vs.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * ly.hkv * 4 : 16));
    }
    if (has_mtp) {  // the MTP layer's pool: same block ids and geometry, index mtp_ly.idx
      const Layer& ly = mtp_ly;
      kc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * ly.hkv * ly.dh * kv_elem()));
      vc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * ly.hkv * ly.dh * kv_elem()));
      ks.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * ly.hkv * 4 : 16));
      vs.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * ly.hkv * 4 : 16));
    }
    // QSA index keys: extra KV planes of the target's geometry (cfg.qsa_chunks), after every real layer's pool
    qsa_loc.clear();
    if (qsa_on) {
      const int base = (int)kc.size(), planes = cfg.qsa_planes(kv8), H = cfg.hkv;
      for (int i = 0; i < planes; i++) {
        kc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * H * cfg.dh * kv_elem()));
        vc.push_back(g.alloc((size_t)n_blocks * KV_BLOCK * H * cfg.dh * kv_elem()));
        ks.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * H * 4 : 16));
        vs.push_back(g.alloc(kv8 ? (size_t)n_blocks * KV_BLOCK * H * 4 : 16));
      }
      const size_t SB = (size_t)cfg.dh * kv_elem();  // bytes per slot of a head plane
      auto ch = cfg.qsa_chunks(kv8);
      const int n = cfg.qsa_layers();
      auto ref = [&](std::pair<int, size_t> c) {
        const int r = c.first, pl = r / (2 * H), which = (r / H) % 2, h = r % H;
        return (which ? vc : kc)[base + pl].at((size_t)h * kv_ns() * SB + c.second);
      };
      for (int i = 0; i < n; i++) qsa_loc.push_back({ref(ch[i]), ref(ch[n + i])});
      qsa_blk = (int)(KV_BLOCK * SB / 2);
    }
  }
  // QSA: per attention layer (index Layer::idx; the MTP layer's is the last) its raw and pooled index-key regions,
  // bound at block 0; consecutive KV blocks are qsa_blk bf16 elements apart in both.
  bool qsa_on = cfg.qsa;
  struct QsaLoc { BufRef raw, pool; };
  std::vector<QsaLoc> qsa_loc;
  int qsa_blk = 0;
  BufRef qsa_proj, qsa_iq, qsa_sel, qsa_rows, qsa_sc;  // index projection [T, 5 Di]; queries; selections; sparse rows; scores
  int qsa_nsp = 0, qsa_maxnb = 0;  // this step's sparse rows and their most complete groups
  int qsa_maxn = 0;  // the most rows any sequence with sparse rows has in this step (> 8: a prefill chunk)
  size_t qsa_sc_bytes = 0;
  // Gathered path (QSA_GATHER, default on with fn_mpp): steps of at most qsa_gmax sparse rows copy each row's selected
  // K/V into a contiguous pool (qsa_gk / qsa_gv, P pages per row) and run attn_fn on it; more rows (prefill) use
  // attn_qsa. qsa_g*: per-row virtual batch (pos, flag indexed by the real row; start / len per sparse row).
  int qsa_gmax = 0, qsa_P = 0, qsa_ns2 = 0;
  BufRef qsa_gk, qsa_gv, qsa_gks, qsa_gvs, qsa_gbt, qsa_gpos, qsa_gflag, qsa_gstart, qsa_glen, qsa_gq, qsa_gqs, qsa_gkb;
  // Union tiles (QSA_UNION, default on with the gathered path; contexts up to 128k): up to 4 consecutive sparse rows of
  // a sequence share one attn_fn tile over the union of their selections, with a per-key row mask (attn_fn MASK).
  // qsa_ustart / qsa_ulen: the step's tiles (host); qsa_ulist / qsa_ucnt: unions; qsa_gmask: mask bytes per key.
  int qsa_umax = 0, qsa_P2 = 0, qsa_nu = 0;
  BufRef qsa_ustart, qsa_ulen, qsa_ulist, qsa_ucnt, qsa_gmask;

  // GPU copy of one slot's recurrent state (conv + GDN) to another: prefix-cache snapshot and restore.
  size_t ple_slot_bytes() const { return ple_state.buf ? (size_t)kPleHist * cfg.hc * cfg.hidden * 2 : 0; }
  void encode_copy_slot(id<MTLBlitCommandEncoder> b, int src, int dst) {
    if (const size_t ps = ple_slot_bytes())
      [b copyFromBuffer:ple_state.buf sourceOffset:src * ps toBuffer:ple_state.buf destinationOffset:dst * ps size:ps];
    size_t cs = (size_t)(cfg.conv_k - 1) * cfg.conv_dim() * 2, ss = (size_t)cfg.lhv * cfg.ldv * cfg.ldk * 4;
    for (size_t i = 0; i < conv_state.size(); i++) {
      [b copyFromBuffer:conv_state[i].buf sourceOffset:src * cs toBuffer:conv_state[i].buf destinationOffset:dst * cs size:cs];
      [b copyFromBuffer:ssm_state[i].buf sourceOffset:src * ss toBuffer:ssm_state[i].buf destinationOffset:dst * ss size:ss];
    }
  }
  void encode_zero_slot(id<MTLBlitCommandEncoder> b, int s) {
    if (const size_t ps = ple_slot_bytes()) [b fillBuffer:ple_state.buf range:NSMakeRange(s * ps, ps) value:0];
    size_t cs = (size_t)(cfg.conv_k - 1) * cfg.conv_dim() * 2, ss = (size_t)cfg.lhv * cfg.ldv * cfg.ldk * 4;
    for (size_t i = 0; i < conv_state.size(); i++) {
      [b fillBuffer:conv_state[i].buf range:NSMakeRange(s * cs, cs) value:0];
      [b fillBuffer:ssm_state[i].buf range:NSMakeRange(s * ss, ss) value:0];
    }
  }
  size_t slot_bytes() const {
    return ple_slot_bytes() + conv_state.size() * ((size_t)(cfg.conv_k - 1) * cfg.conv_dim() * 2 + (size_t)cfg.lhv * cfg.ldv * cfg.ldk * 4);
  }

  // Allocate what speculative verify needs: logits/argmax for every row of up to max_seqs blocks, the hidden
  // capture for the drafter, and a tape per GDN layer.
  // rows > 0 (the server: Scheduler::spec_rows_cap, the most verify rows its block policy ever plans in a step)
  // sizes the logits and the GDN tapes for those rows instead of max_seqs x block and every row of a step (max_tokens).
  // The tapes were most of the drafter's cost: 48 GDN layers x 41 KB a row x 2,304 rows = 4.3 GiB on a 16-sequence
  // mini (2026-11-02), where at most 128 rows ever need replaying. Steps with more rows than the tapes hold (a prefill
  // chunk next to verify blocks) run the GDN kernels on the ordinary scratch and copy the verify rows, which the
  // scheduler puts first, into the tapes (tape_save).
  int spec_rows = 0, tape_rows = 0, spec_end = 0;  // spec_end: end of the step's last speculative row (set_batch)
  bool tape_force = getenv("TAPE_COPY") && atoi(getenv("TAPE_COPY"));  // test: every verify step on the copy path
  BufRef tape_xs;  // copy mode: the conv inputs of every row (tape_x's stand-in)
  void enable_spec(const std::vector<int>& layers, int block, int rows = 0) {
    if (!tape_x.empty() || hcap.buf) return;  // already allocated (the server does it before sizing the KV pool)
    cap_layers = layers;
    max_out = std::max(max_seqs * block, cfg.qwen4 ? 0 : 2 * kMaxTree);  // token trees: up to kMaxTree rows a sequence
    // n_out = verify rows + one row per other sequence in the step
    if (rows > 0) max_out = std::max(std::min(max_out, rows + max_seqs), cfg.qwen4 ? 0 : 2 * kMaxTree);
    spec_rows = rows > 0 ? rows : max_seqs * kMaxBlock;
    tape_rows = rows > 0 ? std::min(max_tokens, max_out) : max_tokens;
    logits = g.alloc((size_t)max_out * cfg.vocab * 2);
    pred = g.alloc(max_out * 4);
    vcand = g.alloc((size_t)max_out * 16 * 4);
    vun = g.alloc((size_t)max_out * 16 * 4);
    vcpart = g.alloc((size_t)max_out * kVTopShards * 16 * 4);
    vvpart = g.alloc((size_t)max_out * kVTopShards * 16 * 4);
    hcap = g.alloc((size_t)max_tokens * layers.size() * cfg.hidden * 2);
    if (!cfg.qwen4) {  // token trees (TREE)
      for (BufRef* r : {&apos, &trow, &tpar, &tdep, &tanc}) *r = g.alloc((size_t)max_tokens * 4);
      for (BufRef* r : {&tseqs, &seq_len_g, &spec_g}) *r = g.alloc((size_t)max_seqs * 4);
      tpath = g.alloc((size_t)max_seqs * kMaxBlock * 4);
      tpart = g.alloc((size_t)max_out * cfg.hq * (cfg.max_dh() + 2) * 4);
    }
    if (ple_state.buf) ple_tape = g.alloc((size_t)max_tokens * cfg.hc * cfg.hidden * 2);
    mem_tapes = conv_state.size() * (size_t)tape_rows * (2 * cfg.conv_dim() * 2 + 2 * cfg.lhv * 4);
    if (tape_rows < max_tokens && !conv_state.empty()) {
      tape_xs = g.alloc((size_t)max_tokens * cfg.conv_dim() * 2);
      mem_tapes += (size_t)max_tokens * cfg.conv_dim() * 2;
    }
    for (size_t i = 0; i < conv_state.size(); i++) {
      tape_x.push_back(g.alloc((size_t)tape_rows * cfg.conv_dim() * 2));
      tape_c.push_back(g.alloc((size_t)tape_rows * cfg.conv_dim() * 2));
      tape_g.push_back(g.alloc((size_t)tape_rows * cfg.lhv * 4));
      tape_b.push_back(g.alloc((size_t)tape_rows * cfg.lhv * 4));
    }
  }

  void reset_slot(int s) {
    if (const size_t ps = ple_slot_bytes()) memset(ple_state.ptr<char>() + s * ps, 0, ps);
    for (size_t i = 0; i < conv_state.size(); i++) {
      size_t cs = (size_t)(cfg.conv_k - 1) * cfg.conv_dim() * 2, ss = (size_t)cfg.lhv * cfg.ldv * cfg.ldk * 4;
      memset(conv_state[i].ptr<char>() + s * cs, 0, cs);
      memset(ssm_state[i].ptr<char>() + s * ss, 0, ss);
    }
  }

  // M <= qmv_max: our matvec (weights reused across R rows). Larger M: MLX's NAX matmul (third_party/mlx),
  // which has no residual epilogue, so mode 1 goes through `tmp` plus an add.
  int qmv_max = getenv("QMV_MAX") ? atoi(getenv("QMV_MAX")) : 2;
  int qmv_rows = getenv("QMV_ROWS") ? atoi(getenv("QMV_ROWS")) : 2;
  int qmvt_max = getenv("QMVT_MAX") ? atoi(getenv("QMVT_MAX")) : 0;  // qmvt4 is slower than MLX's qmm so far
  // W4A8 (kernels/w4a8.metal): 4-bit weights x int8 activations on the integer tensor path for every GEMM with
  // M > qmv_max. W4A8=0 turns it off (bf16 activations through MLX's qmm, weights left unflipped).
  bool use_w4a8 = !getenv("W4A8") || atoi(getenv("W4A8")) != 0;
  int w4a8_max = getenv("W4A8_MAX") ? atoi(getenv("W4A8_MAX")) : 256;  // M >= this: MLX qmm (flip-aware; faster from ~256 rows)
  int sg_bm = getenv("SG_BM") ? atoi(getenv("SG_BM")) : 32, sg_bk = getenv("SG_BK") ? atoi(getenv("SG_BK")) : 32,
      sg_bn = getenv("SG_BN") ? atoi(getenv("SG_BN")) : 32, sg_w = getenv("SG_W") ? atoi(getenv("SG_W")) : 0;  // mlx_qmm_sg tile shape (variants in mlx_qmm_sg.metal)
  int sg_min = getenv("SG_MIN") ? atoi(getenv("SG_MIN")) : 11;  // no neural accelerators: M >= this goes to mlx_qmm_sg
  // (bf16 activations, tile by sg_tile): faster than W4A8 from 12 rows on M3 Ultra and ~13x lower KL on the logits
  void flip_weights() {
    std::vector<QLinear*> ws = {&lm_head};
    for (auto& l : L) for (QLinear* q : {&l.in_proj, &l.out_proj, &l.gate_up, &l.down}) ws.push_back(q);
    if (has_mtp)
      for (QLinear* q : {&mtp_ly.in_proj, &mtp_ly.out_proj, &mtp_ly.gate_up, &mtp_ly.down, &mtp_fc_e, &mtp_fc_h}) ws.push_back(q);
    for (auto& l : L) if (l.has_qsa && !l.iproj_fused) ws.push_back(&l.iproj);  // fused: flipped with in_proj
    if (has_mtp && mtp_ly.has_qsa && !mtp_ly.iproj_fused) ws.push_back(&mtp_ly.iproj);
    flip(ws);
    // not the lm_head. Flash-Next's hyper-connection / PLE projections stay on MLX's qmm (PF8_HC=1 adds them): no speed
    // gain at 16k (4,638 vs 4,681 tok/s) and the 32k tail moved to KL 0.0248 (base 0.0205).
    pf8_list.assign(ws.begin() + 1, ws.end());
    if (getenv("PF8_HC") && atoi(getenv("PF8_HC")))
      for (auto& l : L) for (QLinear* q : {&l.ahc.dn, &l.ahc.up, &l.mhc.dn, &l.mhc.up, &l.ple_kv}) if (q->N > 0) pf8_list.push_back(q);
    if (pf8_on < 0 && !pf8_defer)  // tools; Flash-Next opt-in (PF8=1), as in the server
      pf8_on = !cfg.qwen4 && g.nax && g.dev.recommendedMaxWorkingSetSize >= (uint64_t)40 << 30;
    if (pf8_on > 0) pf8_prepare(pf8_list);
  }
  // PF8 (kernels/pf8.metal, engine-pf 2026-11-02): for steps of >= PF8_MIN rows (default 512) the linears run as
  // one int8 x int8 matmul2d over all of K on Hadamard-rotated, per-row-scaled copies (activations rotated and
  // quantized per row per step). The copies are built at load (int8: 2x the 4-bit weights' bytes: 27B +22.7 GiB,
  // Flash-Next ~+3 GiB). Default (server, tools/server.mm): auto for the 27B-class models (on when the KV pool left after
  // the copies still holds max_running x max_ctx; tools: on with >= 40 GiB working set), OFF for Flash-Next (PF8=1
  // enables it: +9% prefill, but its 32k tail moved to KL 0.0229 vs 0.0205). PF8=0 / PF8=1 override. Studio 16k: 27B
  // 1,841 -> 2,555 tok/s, Flash-Next 4,279 -> 4,677; M5 Pro 8k: 27B 424-463 -> 823-837 (2 Nov 2026).
  int pf8_on = getenv("PF8") ? atoi(getenv("PF8")) : -1;  // -1: auto (the server decides by KV room; tools by machine)
  static inline bool pf8_defer = false;  // set by the server before construction: leave an auto PF8 undecided
  std::vector<QLinear*> pf8_list;
  bool pf8_eligible(const QLinear* q) const { return (q->bits == 4 || q->bits == 8) && q->K % 512 == 0 && !q->ldk && q->N > 0; }
  size_t pf8_estimate() const {  // bytes pf8_prepare would allocate
    size_t b = 0;
    int mk = 0;
    for (const QLinear* q : pf8_list)
      if (pf8_eligible(q) && q->r8.buf == nil) { b += (size_t)q->N * q->K + (size_t)q->N * 4; mk = std::max(mk, q->K); }
    return b + (size_t)max_tokens * mk + (size_t)max_tokens * 4;
  }
  int pf8_min = getenv("PF8_MIN") ? atoi(getenv("PF8_MIN")) : 512;
  int pf8_tile = getenv("PF8_TILE") ? atoi(getenv("PF8_TILE")) : 0;  // 0: 128x128 (8 simdgroups), 1: 64x128, 2: 128x64
  BufRef pf8_sgn, pf8_a, pf8_sx;
  int pf8_maxk = 0;
  size_t pf8_bytes = 0;
  void pf8_prepare(const std::vector<QLinear*>& ws) {
    if (pf8_sgn.buf == nil) {
      pf8_sgn = g.alloc(512 * 4);
      uint32_t h = 0x9E3779B9u;  // fixed signs (xorshift)
      for (int i = 0; i < 512; i++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        pf8_sgn.ptr<float>()[i] = (h >> 7) & 1 ? 1.f : -1.f;
      }
    }
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    Enc e(g, cb);
    for (QLinear* q : ws) {
      if (!pf8_eligible(q) || q->r8.buf != nil) continue;
      id<MTLBuffer> b = g.alloc((size_t)q->N * q->K + (size_t)q->N * 4);
      q->r8 = BufRef(b, 0);
      q->r8s = BufRef(b, (size_t)q->N * q->K);
      pf8_bytes += (size_t)q->N * q->K + (size_t)q->N * 4;
      struct { int K, bits, xr; } a{q->K, q->bits, q->flipped ? (q->bits == 4 ? 8 : 0x80) : 0};
      e.k("pf8_wrot").buf(0, q->wq).buf(1, q->s).buf(2, q->b).buf(3, pf8_sgn).buf(4, q->r8).buf(5, q->r8s).args(6, a);
      e.groups(S3(q->N), S3(32 * std::min(16, q->K / 512)));
      pf8_maxk = std::max(pf8_maxk, q->K);
    }
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
    if (pf8_maxk) {
      pf8_a = g.alloc((size_t)max_tokens * pf8_maxk);
      pf8_sx = g.alloc((size_t)max_tokens * 4);
    }
  }
  bool pf8_route(const QLinear& W, int M) const {
    return pf8_on > 0 && W.r8.buf != nil && M >= pf8_min && M <= max_tokens && !a8_ready && W.K <= pf8_maxk && W.K % 512 == 0 && g.nax;
  }
  void pf8(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode) {
    const int nb = W.K / 512, nsg = std::min(32, (nb + 1) / 2);
    struct { int K, lda; } aa{W.K, W.K};
    e.k("pf8_act").buf(0, A).buf(1, pf8_sgn).buf(2, pf8_a).buf(3, pf8_sx).args(4, aa);
    e.groups(S3(M), S3(32 * nsg));
    struct { int M, N, K, ldc; } ga{M, W.N, W.K, W.N};
    const int TM = pf8_tile == 1 ? 64 : 128, TN = pf8_tile == 2 ? 64 : 128, NS = pf8_tile == 0 ? 8 : 4;
    char nm[32];
    snprintf(nm, sizeof nm, "pf8_%dx%ds%d_m%d", TM, TN, NS, mode);
    e.k(nm).buf(0, pf8_a).buf(1, W.r8).buf(2, pf8_sx).buf(3, W.r8s).buf(4, C).args(5, ga);
    e.groups(S3((M + TM - 1) / TM, (W.N + TN - 1) / TN), S3(32 * NS));
  }
  // Flip quantized weights in place for the integer tensor path (each underlying matrix exactly once).
  // qmst (kernels/qmst.metal): small-batch 4-bit GEMM on repacked weights (QMST=1). The repack is a copy for now.
  int qmst_on = getenv("QMST") ? atoi(getenv("QMST")) : 0;  // off: in the server it cut 8k decode 189 -> 111 tok/s
  int qmst_max = getenv("QMST_MAX") ? atoi(getenv("QMST_MAX")) : 16;
  void repack(const std::vector<QLinear*>& ws) {
    for (QLinear* q : ws) {
      if (q->bits != 4 || q->K % 64 || q->rp.buf != nil) continue;
      const int N = q->N, K = q->K, KJ = K / 32, KG = K / 64, tiles = (N + 31) / 32;
      q->rp = g.alloc((size_t)tiles * KJ * 32 * 16);
      q->rsb = g.alloc((size_t)tiles * KG * 32 * 4);
      const uint32_t* W = q->wq.ptr<uint32_t>();
      const uint16_t *S = q->s.ptr<uint16_t>(), *B = q->b.ptr<uint16_t>();
      uint32_t* P = q->rp.ptr<uint32_t>();
      uint16_t* PS = q->rsb.ptr<uint16_t>();
      const uint32_t xr = q->flipped ? 0x88888888u : 0;
      dispatch_apply(tiles, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        for (int c = 0; c < 32; c++) {
          const int n = std::min((int)t * 32 + c, N - 1);
          for (int j = 0; j < KJ; j++)
            for (int w = 0; w < 4; w++) P[(((size_t)t * KJ + j) * 32 + c) * 4 + w] = W[(size_t)n * (K / 8) + j * 4 + w] ^ xr;
          for (int gg = 0; gg < KG; gg++) {
            PS[(((size_t)t * KG + gg) * 32 + c) * 2] = S[(size_t)n * KG + gg];
            PS[(((size_t)t * KG + gg) * 32 + c) * 2 + 1] = B[(size_t)n * KG + gg];
          }
        }
      });
    }
  }
  // lane4 (kernels/lane.metal, LANE=1): TensorFold-style small-batch GEMM on a repacked copy (experiment).
  // Off by default: a win in decode_test on the M5 Max (40 cores) but slower on the M5 Ultra, and drafter acceptance
  // fell on both (2.82 -> 2.33 tokens per cycle on the Studio). The copy is the 4-bit weights again (~15 GB).
  int lane_on = getenv("LANE") ? atoi(getenv("LANE")) : 0;
  int lane_max = getenv("LANE_MAX") ? atoi(getenv("LANE_MAX")) : 16;
  int lane_min = getenv("LANE_MIN") ? atoi(getenv("LANE_MIN")) : 2;  // 1-2 rows stay on qmv4 (faster in the full step)
  int lane_sk = getenv("LANE_SK") ? atoi(getenv("LANE_SK")) : 0;  // 0: by the tile count
  BufRef lane_xs;
  void repack_lane(const std::vector<QLinear*>& ws) {
    for (QLinear* q : ws) {
      if (q->bits != 4 || q->K % 64 || q->lp.buf != nil) continue;
      const int N = q->N, K = q->K, KG = K / 64, tiles = (N + 31) / 32;
      q->lp = g.alloc((size_t)tiles * KG * 1024);
      q->lsb = g.alloc((size_t)KG * N * 4);
      q->sbn = N;
      const uint32_t* W = q->wq.ptr<uint32_t>();
      const uint16_t *S = q->s.ptr<uint16_t>(), *B = q->b.ptr<uint16_t>();
      uint32_t* P = q->lp.ptr<uint32_t>();
      uint16_t* SB = q->lsb.ptr<uint16_t>();
      const uint32_t xr = q->flipped ? 0 : 0x88888888u;  // the copy is always flipped (signed int4 q - 8)
      dispatch_apply(tiles, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        for (int c = 0; c < 32; c++) {
          const int n = std::min((int)t * 32 + c, N - 1);
          for (int gg = 0; gg < KG; gg++) {
            for (int w = 0; w < 8; w++) P[(((size_t)t * KG + gg) * 32 + c) * 8 + w] = W[(size_t)n * (K / 8) + gg * 8 + w] ^ xr;
            if ((int)t * 32 + c < N) {
              SB[((size_t)gg * N + n) * 2] = S[(size_t)n * KG + gg];
              SB[((size_t)gg * N + n) * 2 + 1] = B[(size_t)n * KG + gg];
            }
          }
        }
      });
    }
  }
  bool lane_route(const QLinear& W, int M) const { return lane_on && W.lp.buf != nil && M > lane_min && M <= lane_max; }
  void lane(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode) {
    const int KG = W.K / 64, tiles = (W.N + 31) / 32;
    int sk = lane_sk;
    if (!sk) sk = tiles >= 1000 && tiles < 5000 ? 8 : 4;  // M5 Max, M=8: 4 simdgroups per tile ~508 GB/s, 8 on gate_up
    struct { int M, K; } xa{M, W.K};
    e.k("lane_xsum").buf(0, A).buf(1, lane_xs).args(2, xa);
    e.threads(S3(KG * 32, M), S3(256));
    struct { int M, N, K, SBN; } la{M, W.N, W.K, W.sbn};
    const int rt = 1;
    char name[32];
    snprintf(name, sizeof name, "lane4_s%d_m%d_r%d", sk, mode, rt);
    e.k(name).buf(0, A).buf(1, W.lp).buf(2, W.lsb).buf(3, lane_xs).buf(4, C).args(5, la);
    e.groups(S3((M + 16 * rt - 1) / (16 * rt), tiles), S3(32 * sk));
  }
  bool qmst_route(const QLinear& W, int M) const { return qmst_on && W.rp.buf != nil && M > qmv_max && M <= qmst_max; }
  void qmst(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode) {
    static int nsg = getenv("QMST_NSG") ? atoi(getenv("QMST_NSG")) : 2;
    static int per_core = getenv("QMST_TGS") ? atoi(getenv("QMST_TGS")) : 6;   // split K until this many TGs per core
    static int spread = getenv("QMST_SPREAD") ? atoi(getenv("QMST_SPREAD")) : 1;
    const int KJ = W.K / 32, tgs = (W.N + 32 * nsg - 1) / (32 * nsg);
    int S = 1;
    if (mode != 2)
      while (S < 8 && tgs * S < per_core * g.cores && KJ / (S * 2) >= 16 && S * 2 * M <= kSplitRows && W.N <= kSplitMaxN) S *= 2;
    const int steps = (KJ + S - 1) / S;
    QmstArgs a{M, W.N, W.K, steps, S, spread};
    char name[32];
    snprintf(name, sizeof name, "qmst_r%d_s%d_m%d", M <= 8 ? 8 : 16, nsg, S > 1 ? 3 : mode);
    e.k(name).buf(0, A).buf(1, W.rp).buf(2, W.rsb).buf(4, S > 1 ? splitk : C).args(5, a);
    e.groups(S3(tgs, 1, S), S3(32 * nsg));
    if (S > 1) {
      ReduceArgs ra{M, W.N, S, mode == 1};
      e.k("splitk_reduce").buf(0, splitk).buf(1, C).args(2, ra);
      e.threads(S3(W.N, M), S3(256));
    }
  }

  void flip(const std::vector<QLinear*>& ws) {
    if (qmst_on) repack(ws);
    if (lane_on) { repack_lane(ws); if (lane_xs.buf == nil) lane_xs = g.alloc((size_t)256 * 512 * 4); }
    if (!use_w4a8) return;
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    Enc e(g, cb);
    for (QLinear* q : ws) {
      if ((q->bits != 4 && q->bits != 8) || q->K % 512 || q->flipped) continue;
      FlipArgs a{(uint32_t)((size_t)q->N * q->K * q->bits / 32), q->bits == 4 ? 0x88888888u : 0x80808080u};
      e.k("w_flip").buf(0, q->wq).args(1, a);
      e.threads(S3(a.n32), S3(256));
      q->flipped = true;
    }
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
  }
  void w4a8(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode, bool defer = false) {
    const int kg = W.K / 64;
    if (!a8_ready) {
      A8Args qa{M, W.K};
      e.k("a8_quant").buf(0, A).buf(1, a8u).buf(2, a8x).args(3, qa);
      e.threads(S3(kg * 32, M), S3(256));
    }
    a8_ready = false;
    // Row tile 32, except 16 for M <= 16 (one 5-row verify block: 22.5 -> 20.4 ms per step, more threadgroups in
    // flight on a bandwidth-bound shape). W4A8_TM / W4A8_TN to retest.
    static int TMe = getenv("W4A8_TM") ? atoi(getenv("W4A8_TM")) : 0;
    static int TNe = getenv("W4A8_TN") ? atoi(getenv("W4A8_TN")) : 128;
    const int TM = TMe ? TMe : M <= 16 ? 16 : 32;
    const int TN = TM == 16 ? TNe : 128, tn = (W.N + TN - 1) / TN, tm = (M + TM - 1) / TM, tiles = tn * tm;
    int S = 1;
    // Split K below 480 tiles on the 80-core Ultra; small GPUs have enough tiles sooner (M5 Pro best at 60-120,
    // M6 best without splitting: 3 x cores = 36 skips N=5120's 40 tiles).
    static int split_below = getenv("W4A8_SPLIT_TILES") ? atoi(getenv("W4A8_SPLIT_TILES")) : g.cores >= 60 ? 480 : 3 * g.cores;
    static int split_cape = getenv("W4A8_SPLIT_CAP") ? atoi(getenv("W4A8_SPLIT_CAP")) : 0;
    // Fewer, larger splits pay off (Ultra, 8-row verify: cap 2560 -> 480 took the step 25.4 -> 24.2 ms; 40 rows best
    // at 960). Scaled by core count: 6x / 12x cores.
    const int split_cap = split_cape ? split_cape : (TM == 16 ? 6 : 12) * g.cores;
    if (mode != 2 && tiles < split_below)  // narrow outputs (N=5120: 40 tiles per 32 rows) split K over groups
      for (int c : {8, 4, 2})
        if (kg % c == 0 && tiles * c <= split_cap && c * M <= kSplitRows && W.N <= kSplitMaxN) { S = c; break; }
    W4A8Args a{M, W.N, W.K, kg / S, S};
    char name[32];
    static int NSGe = getenv("W4A8_NSG") ? atoi(getenv("W4A8_NSG")) : 4;  // simdgroups per tile (experiment)
    static int NSGw = getenv("W4A8_NSGW") ? atoi(getenv("W4A8_NSGW")) : 4;  // simdgroups for 32/64-row tiles (experiment)
    const int NSG = W.bits != 4 ? 4 : TM == 16 ? NSGe : TM == 64 ? 8 : NSGw;
    if (NSG != 4) snprintf(name, sizeof name, "w4a8_%dx%ds%d_m%d", TM, TN, NSG, S > 1 ? 3 : mode);
    else snprintf(name, sizeof name, "w%da8_%dx%d_m%d", W.bits, TM, TN, S > 1 ? 3 : mode);
    e.k(name).buf(0, a8u).buf(1, W.wq).buf(2, W.s).buf(3, W.b).buf(5, a8x).buf(6, S > 1 ? splitk : C).args(7, a);
    e.groups(S3(tm, tn, S), S3(32 * NSG));
    if (S > 1 && defer) {
      pending = PendingSplit{C, M, W.N, S};
      return;
    }
    if (S > 1) {
      ReduceArgs ra{M, W.N, S, mode == 1};
      e.k("splitk_reduce").buf(0, splitk).buf(1, C).args(2, ra);
      e.threads(S3(W.N, M), S3(256));
    }
  }

  // Tile for mlx_qmm_sg by M (M3 Ultra, ms per 27B step over all linears; W4A8 in brackets): the kernel is MMA-bound
  // (~20 ms + 2.2 ms per tile row), so the tile that pads M least wins, but a partial tile pays for MLX's bounds-checked
  // loads, so the 1x4 copy (mlx_qmm_sgw, BM in steps of 8) is used only where M fills it exactly. 12/16 rows 2x2
  // 16x64x16 61.5/54.0 [63.9/59.4]; 20/32 rows 2x2 32x64x16 ~89.5 [102.5/97.3]; 24 rows 1x4 24x32 81.0 [102.8];
  // 40/48 rows 1x4 109.2/124.5, other 33..47 2x2 16x64x32 ~134 [~163 as 32x32x32]; 49..64 2x2 32x64x32 158 [161];
  // above 64, MLX's default 32x32x32.
  static void sg_tile(int M, bool& sgw, int& bm, int& bk, int& bn) {
    sgw = false, bk = 64;
    if (M == 24 || M == 40 || M == 48) sgw = true, bm = M, bn = 32;
    else if (M <= 16) bm = 16, bn = 16;
    else if (M <= 32) bm = 32, bn = 16;
    else if (M <= 48) bm = 16, bn = 32;
    else if (M <= 64) bm = 32, bn = 32;
    else bm = 32, bk = 32, bn = 32;
  }

  // MLX's NAX quantized matmul (patched: buffer 11 = the weight row length of a K-slice view); mode 1 adds into C.
  void mlx_qmm(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode, int bm) {
    char name[48];
    if (!g.nax) {  // no neural accelerators: MLX's simdgroup qmm (tile per sg_tile, 128 threads)
      int bm = sg_bm, bk = sg_bk, bn = sg_bn;
      bool sgw = sg_w;
      if (!getenv("SG_BM")) sg_tile(M, sgw, bm, bk, bn);
      if (W.ldk && !sgw) sgw = true, bm = bm == 16 ? 16 : 32, bn = 32;  // K-slice views: only the 1x4 copy takes ldw
      if (sgw)
        snprintf(name, sizeof name, "mlx_qmm_sgw_b%d_al%s_%dx%d", W.bits, W.N % bn == 0 ? "true" : "false", bm, bn);
      else if (bm == 32 && bk == 32 && bn == 32)
        snprintf(name, sizeof name, "mlx_qmm_sg_b%d_al%s", W.bits, W.N % 32 == 0 ? "true" : "false");
      else
        snprintf(name, sizeof name, "mlx_qmm_sg_b%d_al%s_%dx%dx%d", W.bits, W.N % bn == 0 ? "true" : "false", bm, bk, bn);
      const BufRef& out = mode == 1 ? tmp : C;
      e.k(g.pso_fc(name, {{40, W.flipped}})).buf(0, W.wq).buf(1, W.s).buf(2, W.b).buf(3, A).buf(4, out).args(5, W.K).args(6, W.N).args(7, M);
      if (sgw) { int ldw = W.ldk ? W.ldk : W.K; e.args(8, ldw); }
      e.groups(S3((W.N + bn - 1) / bn, (M + bm - 1) / bm), S3(128));
      if (mode == 1) {
        ActArgs aa{W.N};
        e.k("add_inplace").buf(0, C).buf(1, tmp).args(2, aa);
        e.threads(S3(W.N, M), S3(256));
      }
      return;
    }
    snprintf(name, sizeof name, "mlx_qmm_b%d_bm%d_al%s", W.bits, bm, W.N % 64 == 0 ? "true" : "false");
    const BufRef& out = mode == 1 ? tmp : C;
    int zero = 0;
    e.k(g.pso_fc(name, {{40, W.flipped}})).buf(0, W.wq).buf(1, W.s).buf(2, W.b).buf(3, A).buf(4, out).args(5, W.K).args(6, W.N).args(7, M);
    e.args(8, zero).args(11, W.ldk);  // patched: activation / weight row lengths (0 = K)
    e.groups(S3((W.N + 63) / 64, (M + bm - 1) / bm), S3(128));
    if (mode == 1) {
      ActArgs aa{W.N};
      e.k("add_inplace").buf(0, C).buf(1, tmp).args(2, aa);
      e.threads(S3(W.N, M), S3(256));
    }
  }

  // defer: a split-K residual (mode 1) result may leave its reduce to the next rmsnorm_for of C.
  void gemm(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode, bool defer = false) {
    if (!on("gemm")) return;
    flush_split(e);
    // Flipped weights: qmv4/qmv8 (xr), WxA8, and MLX's qmm / our split-K (function constant 40) all flip back.
    if (W.ldk && (M <= qmv_max || W.bits == 16 || a8_ready)) { fprintf(stderr, "K-slice view off the qmm path\n"); exit(1); }
    if (W.ldk) { mlx_qmm(e, A, W, C, M, mode, M <= 32 ? 32 : 64); return; }  // K-slice views: MLX's qmm only
    if (pf8_route(W, M)) { pf8(e, A, W, C, M, mode); return; }
    if (wide_route(W, M) && !a8_ready) { qmv_wide(e, A, W, C, M, mode); return; }
    if (lane_route(W, M) && !a8_ready) { lane(e, A, W, C, M, mode); return; }
    if (qmst_route(W, M) && !a8_ready) { qmst(e, A, W, C, M, mode); return; }
    if (w4a8_route(W, M)) { w4a8(e, A, W, C, M, mode, defer && defer_split && mode == 1); return; }
    if (a8_ready) { fprintf(stderr, "a8_ready with a non-W4A8 GEMM\n"); exit(1); }
    GemmArgs a{M, W.N, W.K, !W.flipped ? 0 : W.bits == 4 ? 0x8888 : (int)0x80808080u};
    char name[48];
    if (W.bits == 16) {
      if (M <= 2 && W.K % 512 == 0) {
        snprintf(name, sizeof name, "gemv16_r%d_m%d", M <= 1 ? 1 : 2, mode);
        e.k(name).buf(0, A).buf(1, W.wq).buf(4, C).args(5, a);
        e.groups(S3((M + 1) / 2, (W.N + 7) / 8), S3(64));
        return;
      }
      if (M <= 64) {
        // Split K: a 64x64 output tile grid this small (N=5120: 80 tiles) cannot fill the GPU.
        const int tiles_n = (W.N + 63) / 64, tiles_m = (M + 63) / 64, tiles = tiles_n * tiles_m;
        int S = 1;
        for (int c : {16, 8, 4, 2})
          if (tiles * c <= 2560 && c * M <= kSplitRows && W.K / c >= 256) { S = c; break; }
        int part = (W.K / S + 255) / 256 * 256;
        S = (W.K + part - 1) / part;
        MlxSplitKParams sp{M, W.N, W.K, W.K, W.K, W.N, tiles_n, tiles_m, S, M * W.N, part, 0, part / 256};
        auto pso = g.pso_fc("mlx_gemm_splitk_bf16_nt_bm64_bn64_bk256", {{200, M % 64 == 0}, {201, W.N % 64 == 0}});
        e.k(pso).buf(0, A).buf(1, W.wq).buf(2, splitk).args(3, sp);
        e.groups(S3(tiles * S, 1, 1), S3(128));
        ReduceArgs ra{M, W.N, S, mode == 1};
        e.k("splitk_reduce").buf(0, splitk).buf(1, C).args(2, ra);
        e.threads(S3(W.N, M), S3(256));
        return;
      }
      // MLX NAX bf16 GEMM (transposed weights); mode 1 adds the result into C afterwards.
      const bool big = M >= 512;
      const int BM = big ? 128 : 64, BN = 128, BK = big ? 512 : 256;
      MlxGemmParams gp{M, W.N, W.K, W.K, W.K, W.N, (W.N + BN - 1) / BN, (M + BM - 1) / BM, 0, 0, 0, 0, W.K / BK, 0};
      auto pso = g.pso_fc(big ? "mlx_gemm_bf16_nt_bm128_bn128_bk512" : "mlx_gemm_bf16_nt_bm64_bn128_bk256",
                          {{10, false}, {100, false}, {110, false}, {200, M % BM == 0}, {201, W.N % BN == 0},
                           {202, W.K % BK == 0}});
      const BufRef& out = mode == 1 ? tmp : C;
      e.k(pso).buf(0, A).buf(1, W.wq).buf(3, out).args(4, gp);
      e.groups(S3(gp.tiles_n, gp.tiles_m, 1), S3(big ? 512 : 256));
      if (mode == 1) {
        ActArgs aa{W.N};
        e.k("add_inplace").buf(0, C).buf(1, tmp).args(2, aa);
        e.threads(S3(W.N, M), S3(256));
      }
      return;
    }
    // Narrow output, long K (Flash-Next's hyper-connection down projections: 384 x 10,240): the GEMV gives N / 8
    // threadgroups, too few to stream the weights at bandwidth; the split-K path below spreads K instead.
    static const int narrow_n = getenv("QMV_NARROW_N") ? atoi(getenv("QMV_NARROW_N")) : 512;  // Flash-Next B=1 18.83 -> 17.02 ms
    const bool narrow = W.N <= narrow_n && W.K >= 4096 && (W.bits == 4 || W.bits == 8);
    if (M <= qmv_max && W.K % 64 == 0 && !narrow) {  // any K % 64 (the GEMVs skip a partial tail block)
      int R = M <= 1 ? 1 : qmv_rows;
      snprintf(name, sizeof name, "qmv%d_r%d_m%d", W.bits, R, mode);
      e.k(name).buf(0, A).buf(1, W.wq).buf(2, W.s).buf(3, W.b).buf(4, C).args(5, a);
      e.groups(S3((M + R - 1) / R, (W.N + 7) / 8), S3(64));
      return;
    }
    if (W.bits == 4 && M <= qmvt_max && W.K % 512 == 0) {
      int R = M <= 8 ? 8 : 16;
      snprintf(name, sizeof name, "qmvt4_r%d_m%d", R, mode);
      e.k(name).buf(0, A).buf(1, W.wq).buf(2, W.s).buf(3, W.b).buf(4, C).args(5, a);
      e.groups(S3((M + R - 1) / R, (W.N + 7) / 8), S3(64));
      return;
    }
    int bm = M <= 32 ? 32 : 64;
    int tiles = (W.N + 63) / 64 * ((M + bm - 1) / bm);
    if (tiles < 640 && g.nax) {  // qmm_splitk is built on the NAX tiles
      // Narrow output: split K so the GPU has enough threadgroups, then reduce (and add the residual).
      int S = 1;
      for (int c : {16, 8, 4, 2})
        if ((W.K / 64) % c == 0 && tiles * c <= 2560 && c * M <= kSplitRows) { S = c; break; }
      if (S > 1) {
        SplitArgs sa{M, W.N, W.K, W.K / S};
        snprintf(name, sizeof name, "qmm_splitk_b%d_bm%d", W.bits, bm);
        e.k(g.pso_fc(name, {{40, W.flipped}})).buf(0, W.wq).buf(1, W.s).buf(2, W.b).buf(3, A).buf(4, splitk).args(5, sa);
        e.groups(S3((W.N + 63) / 64, (M + bm - 1) / bm, S), S3(128));
        ReduceArgs ra{M, W.N, S, mode == 1};
        e.k("splitk_reduce").buf(0, splitk).buf(1, C).args(2, ra);
        e.threads(S3(W.N, M), S3(256));
        return;
      }
    }
    mlx_qmm(e, A, W, C, M, mode, bm);
  }

  // The next GEMM's int8 activations are already in a8u/a8x (written by a fused producer: rmsnorm_a8, silu_mul_a8).
  bool a8_ready = false;
  // A split-K residual GEMM whose reduce was deferred into the next rmsnorm (splitk_res_norm): its target buffer,
  // rows and splits. flush_split() runs the plain reduce when something else needs the buffer first.
  struct PendingSplit { BufRef C; int M = 0, N = 0, S = 0; } pending;
  // Off by default: splitk_res_norm runs one threadgroup per row, and at 8 rows that is slower than the parallel
  // splitk_reduce it replaces (Studio, 8-row verify step 23.9 -> 24.7 ms) although it saves ~290 dispatches.
  bool defer_split = getenv("DEFER_SPLIT") && atoi(getenv("DEFER_SPLIT")) != 0;
  void flush_split(Enc& e) {
    if (!pending.S) return;
    ReduceArgs ra{pending.M, pending.N, pending.S, 1};
    e.k("splitk_reduce").buf(0, splitk).buf(1, pending.C).args(2, ra);
    e.threads(S3(pending.N, pending.M), S3(256));
    pending.S = 0;
  }
  int fuse_a8 = getenv("FUSE_A8") ? atoi(getenv("FUSE_A8")) : 3;  // 0 off, 1 silu_mul_a8, 3 also rmsnorm_a8
  bool w4a8_route(const QLinear& W, int M) const { return W.flipped && M > qmv_max && M < w4a8_max && !qmst_route(W, M) && !lane_route(W, M) && !wide_route(W, M); }
  // No neural accelerators (M3/M4): MLX's qmv_wide for small blocks (verify rows), whose int8 matmul2d is emulated.
  // WIDE_MIN..WIDE_MAX rows (WIDE_MAX=0: off), WIDE_NV vectors per threadgroup (MLX caps it at 5), WIDE_KL lanes per row.
  int wide_min = getenv("WIDE_MIN") ? atoi(getenv("WIDE_MIN")) : 1;  // M3 Ultra decode: qmv4 30.0 ms, qmv_wide 27.2
  int wide_max = getenv("WIDE_MAX") ? atoi(getenv("WIDE_MAX")) : 10;  // M3 Ultra: 8 rows 57 vs 69 ms (W4A8), 12 rows 78 vs 69
  int wide_nv = getenv("WIDE_NV") ? atoi(getenv("WIDE_NV")) : 5;
  int wide_kl = getenv("WIDE_KL") ? atoi(getenv("WIDE_KL")) : 0;  // 0: 16 lanes up to 3 rows, else 8
  bool wide_route(const QLinear& W, int M) const {
    return !g.nax && M >= wide_min && M <= wide_max && !W.ldk && (W.bits == 4 || W.bits == 8) && W.K % 64 == 0;
  }
  void qmv_wide(Enc& e, const BufRef& A, const QLinear& W, const BufRef& C, int M, int mode) {
    const int tiles = (M + wide_nv - 1) / wide_nv, nv = (M + tiles - 1) / tiles;  // fewest tiles, smallest NV
    char name[48];
    const int kl = wide_kl ? wide_kl : M <= 3 ? 16 : 8;
    snprintf(name, sizeof name, "mlx_qmv_wide_b%d_nv%d_kl%d", W.bits, nv, kl);
    const BufRef& out = mode == 1 ? tmp : C;
    const int rows_tg = 2 * 32 / kl;
    e.k(g.pso_fc(name, {{40, W.flipped}})).buf(0, W.wq).buf(1, W.s).buf(2, W.b).buf(3, A).buf(4, out).args(5, W.K).args(6, W.N).args(7, M);
    e.groups(S3(tiles, (W.N + rows_tg - 1) / rows_tg), S3(32, 2));
    if (mode == 1) {
      ActArgs aa{W.N};
      e.k("add_inplace").buf(0, C).buf(1, tmp).args(2, aa);
      e.threads(S3(W.N, M), S3(256));
    }
  }

  // rmsnorm, fused with the W4A8 activation quantization when `next` (the GEMM that consumes `out`) takes that path.
  BufRef ones, blk_out;  // Glimmer: weightless embedding norm; a block's output before its sandwich norm
  // Flash-Next: hyper streams [T, hc * H] and their scratch; the PLE n-gram ids, table and conv state per slot
  BufRef hx, hnorm, hup, hlow, hmix, injw;
  Layer::Hyper mixer;
  NGram ngram;
  PleTable ple_table;
  static constexpr int kPleHist = 9;  // (conv kernel 4 - 1) * dilation 3
  BufRef ple_ids_buf, ple_emb, ple_kvout, ple_k, ple_gv, ple_state;
  BufRef ple_tape;  // verify steps: the PLE conv inputs [max_tokens, hc * H] (enable_spec)
  Layer::Hyper load_hyper(const Checkpoint& ck, const std::string& p, bool inject) {
    Layer::Hyper h;
    h.norm = ck.load(g, p + "hc_norm.weight");
    // 8-bit even with Q4_NATIVE56 (the two-launch kernels read 4/8-bit; these weights are ~3% of a step's bytes)
    if (inject) h.dn = load_qlinear_any(g, ck, {p + "input_mix_weight_down", p + "block_inject_weight"}, 384, true);
    else h.dn = load_qlinear_any(g, ck, {p + "input_mix_weight_down"}, 384, true);
    h.up = load_qlinear_any(g, ck, {p + "input_mix_weight_up"}, 0, true);
    return h;
  }
  // bf16 router [E, H] plus the shared expert's gate as row E (dequantized to bf16)
  MoeW load_router16(const Checkpoint& ck, const std::string& p) {
    std::vector<uint16_t> rows;
    append_bf16_rows(ck, p + "gate", rows, cfg.hidden);
    append_bf16_rows(ck, p + "shared_expert_gate", rows, cfg.hidden);
    MoeW m;
    m.E = 1, m.K = cfg.hidden, m.bits = 16, m.N = (int)(rows.size() / cfg.hidden);
    m.wq = g.alloc(rows.size() * 2);
    memcpy(m.wq.ptr<char>(), rows.data(), rows.size() * 2);
    m.s = m.wq;
    m.b = m.wq;
    return m;
  }
  // The hyper-connection in front of a block: xn [T, H] = the streams' mix, injw [T, hc] = the injection weights.
  void hyper_pre(Enc& e, const Layer::Hyper& h, int T, bool inject, const BufRef& out) {
    flush_split(e);
    if (!on("hyper")) { pend_for = nullptr; return; }  // SKIP=hyper: timing only
    const int H = cfg.hidden;
    if (hc2(h, T)) {  // two launches (kernels/qwen4.metal hc_nd, hc_um), with any pending injection folded in
      const bool pend = pend_for == &h;
      if (!pend) flush_inject(e);
      const int NR = cfg.hc_low + (inject ? cfg.hc : 0);
      const int xd = h.dn.flipped ? (h.dn.bits == 4 ? 8 : 0x80) : 0, xu = h.up.flipped ? (h.up.bits == 4 ? 8 : 0x80) : 0;
      struct { int H, hc, NR, pend, xr; float eps; } na{H, cfg.hc, NR, pend, xd, cfg.eps};
      if (on("hcnd"))  // SKIP=hcnd / hcum / mrouter / mgu / mdn: timing only
      e.k(h.dn.bits == 4 ? "hc_nd4" : "hc_nd8").buf(0, hx).buf(1, pend ? pend_branch : hx).buf(2, injw).buf(3, h.norm)
          .buf(4, h.dn.wq).buf(5, h.dn.s).buf(6, h.dn.b).buf(7, hnorm).buf(8, hparts).args(9, na);
      if (on("hcnd")) e.groups(S3((NR + 7) / 8, cfg.hc, T), S3(256));
      struct { int H, hc, R, NR, pend, inject, xr; } ua{H, cfg.hc, cfg.hc_low, NR, pend, inject, xu};
      if (on("hcum")) e.k(h.up.bits == 4 ? "hc_um4" : "hc_um8").buf(0, hparts).buf(1, h.up.wq).buf(2, h.up.s).buf(3, h.up.b)
          .buf(4, hnorm).buf(5, out).buf(6, injw_alt).buf(7, hx).buf(8, pend ? pend_branch : hx).buf(9, injw).args(10, ua);
      if (on("hcum")) e.groups(S3((H + 7) / 8, T), S3(256));
      pend_for = nullptr;
      a8_ready = false;
      if (inject) std::swap(injw, injw_alt);
      return;
    }
    struct { int H, hc, in_stride; float eps; } na{H, cfg.hc, cfg.hc * H, cfg.eps};
    if (pend_for == &h) {  // FN_INJ_NORM: the previous block's deferred injection and this norm in one pass
      pend_for = nullptr;
      e.k("hc_inject_norm").buf(0, hx).buf(1, pend_branch).buf(2, injw).buf(3, h.norm).buf(4, hnorm).args(5, na);
    } else {
      flush_inject(e);
      e.k("hc_norm").buf(0, hx).buf(1, h.norm).buf(2, hnorm).args(3, na);
    }
    e.groups(S3(cfg.hc, T), S3(1024));
    a8_ready = false;
    gemm(e, hnorm, h.dn, hlow, T, 0);
    flush_split(e);
    struct { int R, hc, stride, inject; } aa{cfg.hc_low, cfg.hc, h.dn.N, inject};
    e.k("hc_act").buf(0, hlow).buf(1, hmix).buf(2, injw).args(3, aa);
    e.threads(S3(cfg.hc_low + cfg.hc, T), S3(64));
    static const int fuse_up = getenv("HC_FUSE_UP") ? atoi(getenv("HC_FUSE_UP")) : 1;  // rows up to which to fuse (B16 is slower fused)
    if (T <= fuse_up && (h.up.bits == 4 || h.up.bits == 8)) {  // up GEMV + stream mix in one kernel (decode)
      struct { int H, hc, K, xr; } ua{H, cfg.hc, h.up.K, h.up.flipped ? (h.up.bits == 4 ? 0x88 : 0x80) : 0};
      e.k(h.up.bits == 4 ? "hc_up_mix4" : "hc_up_mix8").buf(0, hmix).buf(1, h.up.wq).buf(2, h.up.s).buf(3, h.up.b)
          .buf(4, hnorm).buf(5, out).args(6, ua);
      e.groups(S3((H + 7) / 8, T), S3(256));
      return;
    }
    gemm(e, hmix, h.up, hup, T, 0);
    flush_split(e);
    struct { int H, hc; } ma{H, cfg.hc};
    e.k("hc_mix").buf(0, hup).buf(1, hnorm).buf(2, out).args(3, ma);
    e.threads(S3(H, T), S3(256));
  }
  // Flash-Next two-launch hyper-connections (HC_V2 = the most rows a step may have, 0 off): the norm, [down | inject]
  // and up projections in hc_nd + hc_um instead of hc_norm, split-K GEMM + reduce, hc_act and hc_up_mix, and a
  // block's injection deferred into the next hyper-connection's launches (pend_for: that hyper-connection,
  // pend_branch: the block output) instead of hc_inject. Studio, Flash-Next 2k, B=1 decode step 16.8 -> 14.4 ms
  // (1300 -> 914 dispatches); default up to 4 rows (from 6 rows the per-row threadgroups lose to the old path).
  int hc_v2 = 0;
  int fn_split_row = 128, fn_split_multi = 32;  // Flash-Next attention split lengths (encode_layer)
  // Flash-Next verify blocks of fn_mpp_min..4 rows on the tensor units (attn_fn; FN_MPP=0 off). Needs use_gqa (set
  // after the layers load: knobs() runs again there) and the GPU's neural accelerators.
  bool fn_mpp = false;
  int fn_mpp_min = 1;
  const Layer::Hyper* pend_for = nullptr;
  BufRef pend_branch, injw_alt, hparts;
  bool hc2(const Layer::Hyper& h, int T) const {
    return cfg.qwen4 && T <= std::min(hc_v2, 8) && cfg.hc == 4 && cfg.hidden % 256 == 0 && cfg.hidden <= 4096 &&
           cfg.hidden / 256 <= (cfg.hc_low + 7) / 8 && cfg.hc_low % 64 == 0 && cfg.hc_low <= 1024 &&
           (h.dn.bits == 4 || h.dn.bits == 8) && (h.up.bits == 4 || h.up.bits == 8) && h.dn.K == cfg.hc * cfg.hidden &&
           h.dn.N >= cfg.hc_low + cfg.hc && h.up.K == cfg.hc_low;
  }
  // The deferred injection as its own dispatch (something other than its hyper-connection reads hx first).
  void flush_inject(Enc& e) {
    if (!pend_for) return;
    pend_for = nullptr;
    struct { int H, hc; } ma{cfg.hidden, cfg.hc};
    e.k("hc_inject").buf(0, hx).buf(1, pend_branch).buf(2, injw).args(3, ma);
    e.threads(S3(cfg.hidden, (int)pend_rows), S3(256));
  }
  int pend_rows = 0;
  // next: the hyper-connection that reads the streams right after this injection (nullptr: inject now).
  void hyper_inject(Enc& e, const BufRef& branch, int T, const Layer::Hyper* next = nullptr) {
    flush_split(e);
    if (!on("hyper")) return;
    flush_inject(e);
    // Larger steps (prefill): defer into the next hyper-connection's hc_inject_norm (FN_INJ_NORM, default on;
    // bit-identical, one read of the streams fewer per block)
    if (next && (hc2(*next, T) || (fn_inj_norm && !hc2(*next, T)))) {
      pend_for = next;
      pend_branch = branch;
      pend_rows = T;
      return;
    }
    struct { int H, hc; } ma{cfg.hidden, cfg.hc};
    e.k("hc_inject").buf(0, hx).buf(1, branch).buf(2, injw).args(3, ma);
    e.threads(S3(cfg.hidden, T), S3(256));
  }
  void encode_ple(Enc& e, const Layer& ly, int T, int S) {
    flush_inject(e);
    if (!on("ple")) return;
    const int H = cfg.hidden, HH = cfg.hc * H;
    struct { int heads, W, rows_per_part; } ga{ngram.heads, ple_table.W, ple_table.rows_per_part};
    auto& t = ple_table;
    e.k("ple_gather").buf(0, t.w[0]).buf(1, t.w[1]).buf(2, t.w[2]).buf(3, t.w[3]).buf(4, t.s[0]).buf(5, t.s[1])
        .buf(6, t.s[2]).buf(7, t.s[3]).buf(8, t.b[0]).buf(9, t.b[1]).buf(10, t.b[2]).buf(11, t.b[3])
        .buf(12, ple_ids_buf).buf(13, ple_emb).args(14, ga);
    e.threads(S3(t.W, ngram.heads, T), S3(32, 8, 1));
    a8_ready = false;
    gemm(e, ple_emb, ly.ple_kv, ple_kvout, T, 0);
    flush_split(e);
    struct { int H, hc, in_stride; float eps; } nk{H, cfg.hc, HH + H, cfg.eps}, nq{H, cfg.hc, HH, cfg.eps};
    e.k("hc_norm").buf(0, ple_kvout).buf(1, ly.ple_nk).buf(2, ple_k).args(3, nk);
    e.groups(S3(cfg.hc, T), S3(1024));
    e.k("hc_norm").buf(0, hx).buf(1, ly.ple_nq).buf(2, hnorm).args(3, nq);
    e.groups(S3(cfg.hc, T), S3(1024));
    struct { int H, hc, kv_stride, val_off; } pg{H, cfg.hc, HH + H, HH};
    e.k("ple_gate").buf(0, ple_k).buf(1, hnorm).buf(2, ple_kvout).buf(3, ple_gv).args(4, pg);
    e.groups(S3(cfg.hc, T), S3(1024));
    e.k("hc_norm").buf(0, ple_gv).buf(1, ly.ple_nc).buf(2, hnorm).args(3, nq);
    e.groups(S3(cfg.hc, T), S3(1024));
    // Verify steps keep speculative sequences' conv history and tape the conv inputs; ple_conv_replay advances the
    // history over the committed rows once acceptance is known (like gdn_conv / gdn_conv_replay).
    struct { int C, K, D, verify; } pc{HH, 4, 3, verify && ple_tape.buf ? 1 : 0};
    e.k("ple_conv").buf(0, hnorm).buf(1, ple_gv).buf(2, ly.ple_conv).buf(3, ple_state).buf(4, hx).buf(5, slot)
        .buf(6, seq_start).buf(7, seq_len).args(8, pc).buf(9, pc.verify ? ple_tape : hnorm).buf(10, spec);
    e.threads(S3(HH, S), S3(256));
  }
  BufRef const_buf(int n, float v) {
    BufRef b = g.alloc((size_t)n * 2);
    for (int i = 0; i < n; i++) b.ptr<uint16_t>()[i] = f_to_bf16(v);
    return b;
  }
  BufRef load_norm(const Checkpoint& ck, const std::string& name, float add) {
    BufRef b = ck.load(g, name);
    auto& ti = ck.info(name);
    uint16_t* w = b.ptr<uint16_t>();
    for (size_t i = 0; i < ti.nbytes / 2; i++) w[i] = f_to_bf16(bf16_to_f(w[i]) + add);
    return b;
  }
  // Rows [A (H*D) | B (H*D) | rest] -> [A_0 B_0 A_1 B_1 ... | rest], in place.
  void interleave_heads(QLinear& q, int H, int D) {
    const size_t wr = (size_t)q.K * q.bits / 8, sr = (size_t)q.K / 64 * 2;
    for (auto [base, rb] : {std::pair<char*, size_t>{q.wq.ptr<char>(), wr}, {q.s.ptr<char>(), sr}, {q.b.ptr<char>(), sr}}) {
      std::vector<char> tmp((size_t)2 * H * D * rb);
      memcpy(tmp.data(), base, tmp.size());
      for (int h = 0; h < H; h++) {
        memcpy(base + (size_t)(2 * h) * D * rb, tmp.data() + (size_t)h * D * rb, D * rb);
        memcpy(base + (size_t)(2 * h + 1) * D * rb, tmp.data() + (size_t)(H + h) * D * rb, D * rb);
      }
    }
  }
  // rmsnorm(in) * w with its own eps into out, then x += out (sandwich norm on a block's output).
  void norm_add(Enc& e, const BufRef& in, const BufRef& w, int rows, float eps) {
    flush_split(e);
    const int D = cfg.hidden;
    NormArgs a{D, D, D, eps};
    e.k("rmsnorm").buf(0, in).buf(1, w).buf(2, in).args(3, a);
    e.groups(S3(rows), S3(std::min(1024, (D + 31) / 32 * 32)));
    ActArgs aa{D};
    e.k("add_inplace").buf(0, x).buf(1, in).args(2, aa);
    e.threads(S3(D, rows), S3(256));
  }

  // MoE FFN with residual: x += sum_j w_j * expert_j(xn) + sigmoid(g) * shared(xn), xn = rmsnorm(x).
  void moe_mv(Enc& e, const BufRef& in, const MoeW& W, const BufRef& out, int R, int div, bool use_idx, bool outf) {
    struct { int N, K, div, use_idx; } a{W.N, W.K, div, use_idx};
    char nm[16];
    snprintf(nm, sizeof nm, "moe_qmv%d%s", W.bits, outf ? "f" : "");
    e.k(nm).buf(0, in).buf(1, W.wq).buf(2, W.s).buf(3, W.b).buf(4, ridx).buf(5, out).args(6, a);
    e.groups(S3((W.N + 7) / 8, R), S3(256));
  }
  const Layer::Hyper* moe_next = nullptr;  // set per layer in encode_layer
  // Flash-Next MoE in four launches (MOE_V2 = the most rows a step may have, at most 8; 0 off): moe_router16 (all
  // rows at once), moe_plan (routing + the step's unique experts; MOE_DEDUP=0: one slot per pick), moe_gu2 (gate/up
  // SwiGLU, each unique expert's weights read once for every row that picked it, plus the shared expert), moe_dn2
  // (routed + shared down and the weighted sum) in kernels/moe.metal, instead of ~9 dispatches. Real text: a 4-row
  // verify block picks ~27 distinct experts of 40 (0.68), 8 rows 0.53 (mlx_vlm router, 700 tokens).
  int moe_v2 = 0;
  BufRef moe_act2;  // [8, topk + 1, moe_inter]
  int moe_dedup = 1;
  BufRef moe_plan_buf;  // moe_plan's output: [unique count, unique experts [8 k], per unique expert its 8 rows' picks]
  bool moe2(const Layer& ly, int T) const {
    return cfg.qwen4 && T <= std::min(moe_v2, 8) && moe_act2.buf && ly.router.bits == 16 && cfg.topk <= 16 &&
           cfg.experts <= 512 && ly.egu.bits == 4 && ly.edown.bits == 4 && (ly.gate_up.bits == 4 || ly.gate_up.bits == 8) &&
           (ly.down.bits == 4 || ly.down.bits == 8) && cfg.inter == cfg.moe_inter && cfg.moe_inter % 64 == 0 &&
           cfg.hidden % 64 == 0 && cfg.moe_inter % 128 == 0 && ly.router.N <= cfg.experts + 1;
  }
  int moe_sort_min = getenv("MOE_SORT_MIN") ? atoi(getenv("MOE_SORT_MIN")) : 128;  // routed rows from which to sort (fewer: per-row GEMV, faster to ~8 tokens)
  void gqmm(Enc& e, const BufRef& A, const MoeW& W, const BufRef& C, int R) {
    int bm = R / cfg.experts >= 16 ? 64 : 16;  // rows per expert on average
    if (cfg.qwen4 && fn_gqmm_bm) bm = fn_gqmm_bm;  // FN_GQMM_BM (16 / 64): Flash-Next prefill experiments
    char nm[40];
    snprintf(nm, sizeof nm, "mlx_gqmm_b%d_bm%d%s", W.bits, bm, cfg.qwen4 && fn_gqmm_sfx ? fn_gqmm_sfx : "");
    e.k(g.pso_fc(nm, {{40, false}, {200, false}, {201, W.N % 64 == 0}, {202, W.K % 64 == 0}}))
        .buf(0, A).buf(1, W.wq).buf(2, W.s).buf(3, W.b).buf(4, moffs).buf(5, C).args(6, R).args(7, W.N).args(8, W.K)
        .args(9, W.E);
    const int tiles = (R + bm - 1) / bm + std::min(W.E, R);
    int tgs = bm == 64 ? 128 : 64;
    if (cfg.qwen4 && fn_gqmm_bm) {  // kernels/fnp_gather.metal shapes: WM x WN simdgroups
      const std::string sx = fn_gqmm_sfx ? fn_gqmm_sfx : "";
      tgs = bm == 128 || sx == "_n4" ? 256 : sx == "_w2" ? 128 : bm == 32 ? 64 : tgs;
    }
    e.groups(S3((W.N + 63) / 64, tiles), S3(tgs));
  }
  void encode_moe(Enc& e, const Layer& ly, int T) {
    const int H = cfg.hidden, k = cfg.topk, R = T * k;
    if (!on("moe")) { if (cfg.qwen4 && on("hyper")) hyper_pre(e, ly.mhc, T, true, xn); return; }  // SKIP=moe: timing only
    // Flash-Next: the hyper-connection makes the block input and the output goes to blk_out for injection
    const BufRef& dst = cfg.qwen4 ? blk_out : x;
    if (cfg.qwen4) hyper_pre(e, ly.mhc, T, true, xn);
    else rmsnorm_for(e, x, ly.post_norm, xn, T, H, ly.gate_up);
    if (moe2(ly, T)) {  // four launches: router GEMV; routing plan; gate/up SwiGLU (routed and shared); down + combine
      struct { int N, K, T; } r1{ly.router.N, H, T};
      if (on("mrouter")) {
        e.k("moe_router16").buf(0, xn).buf(1, ly.router.wq).buf(2, rlog).args(3, r1);
        e.groups(S3((ly.router.N + 3) / 4), S3(256));
      }
      const int xg = ly.gate_up.flipped ? (ly.gate_up.bits == 4 ? 8 : 0x80) : 0;
      const int xd = ly.down.flipped ? (ly.down.bits == 4 ? 8 : 0x80) : 0;
      struct { int E, k, stride, T, dedup; } pa{cfg.experts, k, ly.router.N, T, moe_dedup};
      e.k("moe_plan").buf(0, rlog).buf(1, ridx).buf(2, rw).buf(3, rsg).buf(4, moe_plan_buf).args(5, pa);
      e.groups(S3(1), S3(256));
      struct { int k, I, K, xr, T; } ga{k, cfg.moe_inter, H, xg, T};
      char nm[32];
      snprintf(nm, sizeof nm, "moe_gu2_%d_%d_t%d", ly.egu.bits, ly.gate_up.bits,
               T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8);
      e.k(nm).buf(0, xn).buf(1, moe_plan_buf).buf(2, ly.egu.wq).buf(3, ly.egu.s).buf(4, ly.egu.b).buf(5, ly.gate_up.wq)
          .buf(6, ly.gate_up.s).buf(7, ly.gate_up.b).buf(8, moe_act2).args(9, ga);
      if (on("mgu")) e.groups(S3(cfg.moe_inter / 8, T * k + T), S3(256));
      struct { int k, I, H, xr, assign; } da{k, cfg.moe_inter, H, xd, cfg.qwen4};
      snprintf(nm, sizeof nm, "moe_dn2_%d_%d", ly.edown.bits, ly.down.bits);
      e.k(nm).buf(0, moe_act2).buf(1, ridx).buf(2, rw).buf(3, rsg).buf(4, ly.edown.wq).buf(5, ly.edown.s)
          .buf(6, ly.edown.b).buf(7, ly.down.wq).buf(8, ly.down.s).buf(9, ly.down.b).buf(10, dst).args(11, da);
      if (on("mdn")) e.groups(S3((H + 7) / 8, T), S3(256));
      if (cfg.qwen4) hyper_inject(e, blk_out, T, moe_next);
      return;
    }
    // shared expert -> sh_out
    gemm(e, xn, ly.gate_up, proj, T, 0);
    ActArgs aca{cfg.inter};
    e.k("silu_mul").buf(0, proj).buf(1, act).args(2, aca);
    e.threads(S3(cfg.inter, T), S3(256));
    a8_ready = false;
    gemm(e, act, ly.down, sh_out, T, 0);
    flush_split(e);
    // router
    struct { int E, k, stride, norm; } ra{cfg.experts, k, ly.router.N, 1};
    if (cfg.qwen4 && fn_router_gemm > 0 && T >= fn_router_gemm && ly.router.bits == 16) {
      // Flash-Next prefill: the bf16 router as a GEMM with bf16 logits (as the reference's nn.Linear), 83 -> 3 ms
      // per 2,048-row chunk (moe_qmv16f is a per-row GEMV)
      QLinear rq;
      rq.wq = rq.s = rq.b = ly.router.wq;
      rq.N = ly.router.N, rq.K = ly.router.K, rq.bits = 16;
      a8_ready = false;
      gemm(e, xn, rq, rlog, T, 0);
      flush_split(e);
      e.k("moe_route_bf").buf(0, rlog).buf(1, ridx).buf(2, rw).buf(3, rsg).args(4, ra);
      e.groups(S3(T), S3(32));
    } else {
    moe_mv(e, xn, ly.router, rlog, T, 1, false, true);
    e.k("moe_route").buf(0, rlog).buf(1, ridx).buf(2, rw).buf(3, rsg).args(4, ra);
    e.groups(S3(T), S3(32));
    }
    // routed experts
    struct { int I; } ma{cfg.moe_inter};
    struct { int H, k, assign; } ca{H, k, cfg.qwen4};
    if (R < moe_sort_min) {  // a few rows: one GEMV per (token, expert) row
      moe_mv(e, xn, ly.egu, egu_out, R, k, true, false);
      e.k("moe_silu_mul").buf(0, egu_out).buf(1, eact).args(2, ma);
      e.threads(S3(cfg.moe_inter, R), S3(256));
      moe_mv(e, eact, ly.edown, edown_out, R, 1, true, false);
      e.k("moe_combine").buf(0, dst).buf(1, edown_out).buf(2, rw).buf(3, sh_out).buf(4, rsg).args(5, ca);
      e.threads(S3(H, T), S3(256));
      if (cfg.qwen4) hyper_inject(e, blk_out, T, moe_next);
      return;
    }
    // rows sorted by expert, then MLX's gathered matmul reads each expert's weights once per row tile
    struct { int R, E; } sa{R, cfg.experts};
    e.k("moe_sort").buf(0, ridx).buf(1, moffs).buf(2, mperm).buf(3, minv).args(4, sa);
    e.groups(S3(1), S3(1024));
    struct { int K, div; } ga{H, k};
    e.k("moe_gather").buf(0, xn).buf(1, mperm).buf(2, mxs).args(3, ga);
    e.threads(S3(H, R), S3(256));
    gqmm(e, mxs, ly.egu, egu_out, R);
    e.k("moe_silu_mul").buf(0, egu_out).buf(1, eact).args(2, ma);
    e.threads(S3(cfg.moe_inter, R), S3(256));
    gqmm(e, eact, ly.edown, edown_out, R);
    e.k("moe_combine_sorted").buf(0, dst).buf(1, edown_out).buf(2, rw).buf(3, sh_out).buf(4, rsg).args(5, ca).buf(6, minv);
    e.threads(S3(H, T), S3(256));
    if (cfg.qwen4) hyper_inject(e, blk_out, T, moe_next);
  }

  void rmsnorm_for(Enc& e, const BufRef& in, const BufRef& w, const BufRef& out, int rows, int D, const QLinear& next) {
    if (pending.S && pending.C.buf == in.buf && pending.C.off == in.off && pending.M == rows && pending.N == D && on("norm")) {
      const bool a8 = fuse_a8 >= 3 && w4a8_route(next, rows) && D % 64 == 0;
      NormArgs a{D, D, D, cfg.eps};
      SplitRed sr{rows, D, pending.S};
      e.k(a8 ? "splitk_res_norm_a8" : "splitk_res_norm").buf(0, in).buf(1, w).buf(2, out).args(3, a).buf(4, a8u)
          .buf(5, a8x).buf(6, splitk).args(7, sr);
      e.groups(S3(rows), S3(std::min(1024, (D + 31) / 32 * 32)));
      pending.S = 0;
      a8_ready = a8;
      return;
    }
    flush_split(e);
    // FUSE_A8 < 3 turns this off. On the Studio it saves 0.5 ms per 8-row verify step (24.4 -> 23.9); an earlier
    // laptop measurement that called it slower was within that machine's run-to-run noise.
    if (fuse_a8 < 3 || !w4a8_route(next, rows) || D % 64 || !on("norm")) { rmsnorm(e, in, w, out, rows, D, D, D); return; }
    NormArgs a{D, D, D, cfg.eps};
    e.k("rmsnorm_a8").buf(0, in).buf(1, w).buf(2, out).args(3, a).buf(4, a8u).buf(5, a8x);
    e.groups(S3(rows), S3(std::min(1024, (D + 31) / 32 * 32)));
    a8_ready = true;
  }

  void rmsnorm(Enc& e, const BufRef& in, const BufRef& w, const BufRef& out, int rows, int D, int in_stride,
               int out_stride) {
    flush_split(e);
    if (!on("norm")) return;
    NormArgs a{D, in_stride, out_stride, cfg.eps};
    e.k("rmsnorm").buf(0, in).buf(1, w).buf(2, out).args(3, a);
    e.groups(S3(rows), S3(std::min(1024, (D + 31) / 32 * 32)));
  }

  // Work for one sequence in a step: its state slot, KV block list, the position of its first new token,
  // and the new tokens (one for decode, a chunk for prefill).
  struct SeqWork {
    int slot;
    const std::vector<int>* blocks;
    int pos0;
    const int* tokens;
    int n;
    bool spec = false;  // verify block: logits for every row, state kept until acceptance is known
    bool tree = false;  // spec rows are a token tree (set_tree), not a chain
    // Sampling for a speculative block (temperature > 0: speculative sampling in spec_accept_s; 0: greedy).
    float temp = 0, top_p = 1;
    int top_k = 0;
    uint32_t seed = 0;
    // Multimodal prompts: M-RoPE positions (t, h, w) for positions < mrope_len, then position + rope_delta;
    // rows whose ext_idx[position] >= 0 take embedding row ext_idx from emb (bf16 [*, hidden]) instead of a token.
    const int* mrope = nullptr;
    int mrope_len = 0, rope_delta = 0;
    const uint16_t* emb = nullptr;
    const int* ext_idx = nullptr;
    int ext_len = 0;
  };

  // Fill the step's input buffers. Returns the number of token rows T.
  int max_ctx = 1;
  bool any_multi = false, any_nax = false;
  int attn_units = 1;  // attention work items: single-row rows plus multi-row sequences (one threadgroup column each)
  bool use_gqa = !getenv("ATTN_V1");
  char gqa_sfx[16] = "_g6_d256";
  bool use_nax = !getenv("NO_NAX_ATTN");  // attn_paged_nax for multi-row runs of <= nax_rows rows
  int nax_rows = getenv("NAX_ROWS") ? atoi(getenv("NAX_ROWS")) : 8;  // longer runs: attn_shared (mode 6) or flash
  int nax_tiles = 1;  // most query tiles of any NAX sequence this step
  bool any_mpp = false;  // a verify block goes to attn_mpp this step
  bool any_mpp32 = false;  // a 2..5-row block goes to the 32-row attn_mpp
  // attn_mpp (kernels/attn_mpp.metal) for verify blocks of mpp_min..8 rows; 2..5-row blocks (several agents) stay on
  // attn_paged_nax, whose 32-row tiles fit them (attn_mpp's 48-row tile was 3-12% slower there). NAX_MPP=0: off.
  bool use_mpp = (!getenv("NAX_MPP") || atoi(getenv("NAX_MPP")) != 0) && cfg.dh == 256 && cfg.hq == 6 * cfg.hkv;
  // 2..5-row blocks (several agents, block 4/5) on the 32-row attn_mpp: 128k x 16 agents 227 -> 196 ms per step,
  // 16k x 16 85 -> 80 (5-row blocks; 4-row: 212 -> 186 at 128k). NAX_MPP32=0: attn_paged_nax.
  bool use_mpp32 = use_mpp &&  // after use_mpp (members initialize in declaration order)
      (!getenv("NAX_MPP32") || atoi(getenv("NAX_MPP32")) != 0);
  int mpp_min = getenv("NAX_MPP_MIN") ? atoi(getenv("NAX_MPP_MIN")) : 6;
  int nax_tile_rows = getenv("NAX_TILE") ? atoi(getenv("NAX_TILE")) : 32;  // 32 (wm2) or 64 (wm4) packed rows
  int set_batch(const std::vector<SeqWork>& w) {
    int T = 0;
    max_ctx = 1;
    any_multi = false;
    any_nax = false;
    nax_tiles = 1;
    any_mpp = false;
    any_mpp32 = false;
    any_chunk = false;
    any_ext = false;
    attn_units = 0;
    n_out = 0;
    any_sample = false;
    flash.clear();
    qsa_nsp = 0;
    qsa_maxnb = 0;
    qsa_maxn = 0;
    for (size_t b = 0; b < w.size(); b++) {
      slot.ptr<int>()[b] = w[b].slot;
      seq_start.ptr<int>()[b] = T;
      seq_len.ptr<int>()[b] = w[b].n;
      if ((int)w[b].blocks->size() > max_blocks) {  // a row of the table holds max_blocks ids (Codex review R02)
        fprintf(stderr, "set_batch: %zu blocks > table width %d\n", w[b].blocks->size(), max_blocks);
        abort();
      }
      for (size_t j = 0; j < w[b].blocks->size(); j++) block_table.ptr<int>()[b * max_blocks + j] = (*w[b].blocks)[j];
      // attention mode per row: 0 single-row kernel, 1 flash (big prefill chunks), 2 multi-row kernel,
      // 3 the NAX paged kernel (runs of up to nax_rows rows, in 32-row tiles of rows x GQ heads),
      // 4 attn_mpp (verify blocks of mpp_min..8 rows: all 48 packed rows in one whole-tile matmul pipeline)
      const int gq = cfg.hq / cfg.hkv, tiles = (w[b].n * gq + nax_tile_rows - 1) / nax_tile_rows;
      const bool nax = use_nax && w[b].n > 1 && w[b].n <= nax_rows;
      const bool mppk = nax && use_mpp && w[b].n >= mpp_min && w[b].n <= 8;
      const bool mpp32 = nax && use_mpp32 && w[b].n >= 2 && w[b].n <= 5 && !mppk;  // mode 5: the 32-row attn_mpp
      // mode 6: a chunk of nax_rows+1 .. chunk_tile_max rows (CHUNK_TILE_MAX, 256) goes to attn_shared
      // in tiles of 8 rows x 6 heads split over the keys (an agent's follow-up turn on a long context: flash gathers
      // the whole KV and runs ceil(n / 64) x Hq threadgroups with no key split).
      const bool ct = use_gqa && use_mpp && !use_cascade && !nax && w[b].n > 1 && w[b].n <= chunk_tile_max &&
                      w[b].pos0 >= chunk_tile_ctx;
      bool fl = use_gqa && !nax && !ct && w[b].n >= flash_rows;
      // Flash-Next (GQ 12): blocks of fn_mpp_min..4 rows on attn_fn (kernels/qwen4.metal), mode 4
      const bool fnm = fn_mpp && !fl && !ct && w[b].n >= fn_mpp_min && w[b].n <= 48 / gq;
      int mode = fl ? 1 : ct ? 6 : fnm ? 4 : w[b].n > 1 ? (mppk ? 4 : mpp32 ? 5 : nax ? 3 : 2) : 0;
      // QSA: rows past the dense end attend to their selected groups (mode 7, attn_qsa); the sequence's dense rows
      // (a prefix of it) go to flash when there are flash_rows of them, else to the single-row kernel.
      const int nd = qsa_on && !qsa_dense ? std::clamp(cfg.qsa_dense_end() - w[b].pos0, 0, w[b].n) : w[b].n;
      if (nd < w[b].n) {
        fl = use_gqa && nd >= flash_rows;
        mode = fl ? 1 : 0;
      }
      any_multi |= mode == 2;
      any_nax |= mode == 3;
      any_mpp |= mode == 4;
      any_mpp32 |= mode == 5;
      if (mode == 3) nax_tiles = std::max(nax_tiles, tiles);
      any_chunk |= mode == 6;
      attn_units += nd < w[b].n ? (mode == 0 ? w[b].n : w[b].n - nd)
                    : mode == 0 ? w[b].n : mode == 1 ? 0 : mode == 3 ? tiles : mode == 6 ? (w[b].n + 7) / 8 : 1;
      if (fl) flash.push_back({(int)b, T, nd, w[b].pos0 + nd});
      if (ple_ids_buf.buf) ngram.ids(w[b].tokens, w[b].pos0, w[b].n, ple_ids_buf.ptr<int>() + (size_t)T * ngram.heads);
      for (int j = 0; j < w[b].n; j++, T++) {
        const int p = w[b].pos0 + j;
        rflag.ptr<int>()[T] = j < nd ? mode : 7;
        if (j >= nd) {
          qsa_maxn = std::max(qsa_maxn, w[b].n);
          qsa_rows.ptr<int>()[qsa_nsp++] = T;
          qsa_maxnb = std::max(qsa_maxnb, (p + 1) / cfg.qsa_ratio);
        }
        tok.ptr<int>()[T] = w[b].tokens[j];
        pos.ptr<int>()[T] = p;
        for (int c = 0; c < 3; c++)
          rpos.ptr<int>()[T * 3 + c] = p < w[b].mrope_len ? w[b].mrope[p * 3 + c] : p + w[b].rope_delta;
        int xi = p < w[b].ext_len ? w[b].ext_idx[p] : -1;
        ext.ptr<int>()[T] = xi >= 0;
        if (xi >= 0) {
          tok.ptr<int>()[T] = 0;  // the token id is a content key >= vocab; the embedding comes from xin
          memcpy(xin.ptr<uint16_t>() + (size_t)T * cfg.hidden, w[b].emb + (size_t)xi * cfg.hidden, cfg.hidden * 2);
          any_ext = true;
        }
        tseq.ptr<int>()[T] = (int)b;
      }
      spec.ptr<int>()[b] = w[b].spec;
      {
        struct { float t, p; int k; uint32_t seed; } sp{w[b].spec ? w[b].temp : 0.f, w[b].top_p, w[b].top_k, w[b].seed};
        memcpy(samp.ptr<char>() + b * 16, &sp, 16);
        any_sample |= w[b].spec && w[b].temp > 0;
      }
      out_base.ptr<int>()[b] = n_out;
      if (w[b].spec) {
        for (int j = 0; j < w[b].n; j++) out_row.ptr<int>()[n_out++] = T - w[b].n + j;
      } else {
        out_row.ptr<int>()[n_out++] = T - 1;
      }
      // sparse rows attend to at most topk * ratio + ratio - 1 positions (attention splits are planned on that)
      max_ctx = std::max(max_ctx, nd < w[b].n ? std::max(nd ? w[b].pos0 + nd : 0, cfg.qsa_dense_end()) : w[b].pos0 + w[b].n);
    }
    spec_end = 0;
    for (size_t b = 0; b < w.size(); b++)
      if (w[b].spec) spec_end = seq_start.ptr<int>()[b] + w[b].n;
    if (verify && T > tape_rows && spec_end > tape_rows) {  // the scheduler puts verify blocks first; tools stay small
      fprintf(stderr, "set_batch: verify rows end at %d > %d tape rows (%d rows in the step)\n", spec_end, tape_rows, T);
      abort();
    }
    tree_batch(w, T);
    qsa_nu = 0;
    if (qsa_nsp && qsa_umax) {  // union tiles: runs of <= 4 consecutive sparse rows of one sequence
      const int* rw = qsa_rows.ptr<int>();
      for (int i = 0; i < qsa_nsp;) {
        int n = 1;
        while (n < 4 && i + n < qsa_nsp && rw[i + n] == rw[i] + n && tseq.ptr<int>()[rw[i + n]] == tseq.ptr<int>()[rw[i]]) n++;
        qsa_ustart.ptr<int>()[qsa_nu] = rw[i];
        qsa_ulen.ptr<int>()[qsa_nu++] = n;
        i += n;
      }
    }
    if (qsa_nsp) {  // cascade attention assumes every row attends to its whole prefix
      if (cascade) memset(shrow.ptr<int>(), 0, (size_t)max_tokens * 4);
      cascade = false;
      sh_tiles = 0;
      if (any_chunk) plan_chunks(w);
      return T;
    }
    plan_cascade(w, T);
    if (any_chunk) plan_chunks(w);
    return T;
  }

  // Everything set_batch writes (per-row and per-sequence input buffers, and the step-wide flags it derives), so a
  // second batch (the MTP head's draft passes) can be planned into its own buffers and encoded in the same command
  // buffer as the main step. load_batch(other) then set_batch(...) fills `other`'s buffers.
  struct BatchState {
    BufRef tok, pos, tseq, slot, block_table, seq_start, seq_len, out_row, spec, samp, out_base, rflag, rpos, ext,
        ple_ids, qsa_rows, qsa_ustart, qsa_ulen;
    int qsa_nu = 0;
    int max_ctx = 1, attn_units = 1, nax_tiles = 1, n_out = 0, sh_tiles = 0, qsa_nsp = 0, qsa_maxnb = 0, qsa_maxn = 0;
    bool any_multi = false, any_nax = false, any_mpp = false, any_mpp32 = false, any_chunk = false, any_ext = false,
         any_sample = false, cascade = false;
    std::vector<FlashSeq> flash;
  };
  BatchState save_batch() const {
    BatchState b{tok, pos, tseq, slot, block_table, seq_start, seq_len, out_row, spec, samp, out_base, rflag, rpos, ext,
                 ple_ids_buf, qsa_rows, qsa_ustart, qsa_ulen};
    b.qsa_nu = qsa_nu;
    b.qsa_nsp = qsa_nsp; b.qsa_maxnb = qsa_maxnb; b.qsa_maxn = qsa_maxn;
    b.max_ctx = max_ctx; b.attn_units = attn_units; b.nax_tiles = nax_tiles; b.n_out = n_out; b.sh_tiles = sh_tiles;
    b.any_multi = any_multi; b.any_nax = any_nax; b.any_mpp = any_mpp; b.any_mpp32 = any_mpp32;
    b.any_chunk = any_chunk; b.any_ext = any_ext; b.any_sample = any_sample; b.cascade = cascade;
    b.flash = flash;
    return b;
  }
  void load_batch(const BatchState& b) {
    tok = b.tok; pos = b.pos; tseq = b.tseq; slot = b.slot; block_table = b.block_table; seq_start = b.seq_start;
    seq_len = b.seq_len; out_row = b.out_row; spec = b.spec; samp = b.samp; out_base = b.out_base; rflag = b.rflag;
    rpos = b.rpos; ext = b.ext; ple_ids_buf = b.ple_ids; qsa_rows = b.qsa_rows;
    qsa_ustart = b.qsa_ustart; qsa_ulen = b.qsa_ulen; qsa_nu = b.qsa_nu;
    qsa_nsp = b.qsa_nsp; qsa_maxnb = b.qsa_maxnb; qsa_maxn = b.qsa_maxn;
    max_ctx = b.max_ctx; attn_units = b.attn_units; nax_tiles = b.nax_tiles; n_out = b.n_out; sh_tiles = b.sh_tiles;
    any_multi = b.any_multi; any_nax = b.any_nax; any_mpp = b.any_mpp; any_mpp32 = b.any_mpp32;
    any_chunk = b.any_chunk; any_ext = b.any_ext; any_sample = b.any_sample; cascade = b.cascade;
    flash = b.flash;
  }
  // Buffers for a batch of up to `seqs` single-row sequences (no PLE ids, no cascade).
  BatchState alloc_batch(int seqs) {
    BatchState b;
    b.tok = g.alloc(seqs * 4); b.pos = g.alloc(seqs * 4); b.tseq = g.alloc(seqs * 4); b.slot = g.alloc(seqs * 4);
    b.block_table = g.alloc((size_t)seqs * max_blocks * 4); b.seq_start = g.alloc(seqs * 4);
    b.seq_len = g.alloc(seqs * 4); b.out_row = g.alloc(seqs * kMaxBlock * 4); b.spec = g.alloc(seqs * 4);
    b.samp = g.alloc(seqs * 16); b.out_base = g.alloc(seqs * 4); b.rflag = g.alloc(seqs * 4);
    b.rpos = g.alloc(seqs * 3 * 4); b.ext = g.alloc(seqs * 4);
    if (qsa_rows.buf) b.qsa_rows = g.alloc(seqs * 4);
    if (qsa_ustart.buf) { b.qsa_ustart = g.alloc(seqs * 4); b.qsa_ulen = g.alloc(seqs * 4); }
    return b;
  }

  // Cascade attention: attn_mpp sequences that begin with the same physical KV blocks (the prefix cache shares
  // them) attend to that shared prefix together, once, in attn_shared (48-row tiles of their packed rows), and each
  // to its own suffix in attn_mpp. Only when every multi-row sequence of the step is on attn_mpp and no single-row
  // kernel runs. Off by default (CASCADE=1 on; CASCADE_MIN: shared blocks needed, 64 = 2k tokens): measured slower
  // on the M5 Ultra, because sharing the physical KV blocks already lets the GPU caches serve the repeated reads
  // (64k x 16 agents, 5-row blocks: 147 ms per step with private prefixes, 110 shared without cascade, 124 with it);
  // what remains is attention compute (~47 TFLOPS), which cascade does not reduce.
  bool cascade = false;
  int sh_tiles = 0;
  bool use_cascade = getenv("CASCADE") && atoi(getenv("CASCADE")) != 0;
  int cascade_min = getenv("CASCADE_MIN") ? atoi(getenv("CASCADE_MIN")) : 64;
  uint64_t casc_steps = 0, casc_seen = 0, casc_block[4] = {0, 0, 0, 0};  // debug counters (CASCADE_STATS)
  void plan_cascade(const std::vector<SeqWork>& w, int T) {
    if (cascade) memset(shrow.ptr<int>(), 0, (size_t)max_tokens * 4);
    cascade = false;
    sh_tiles = 0;
    static const bool stats = getenv("CASCADE_STATS") != nullptr;
    if (stats && ++casc_seen % 200 == 0)
      fprintf(stderr, "cascade: %llu of %llu steps; blocked by nax %llu, multi %llu, mode 0/2/3 %llu, no group %llu\n",
              (unsigned long long)casc_steps, (unsigned long long)casc_seen, (unsigned long long)casc_block[0],
              (unsigned long long)casc_block[1], (unsigned long long)casc_block[2], (unsigned long long)casc_block[3]);
    if (!use_cascade || !(any_mpp || any_mpp32)) return;
    if (any_nax) { casc_block[0]++; return; }
    if (any_multi) { casc_block[1]++; return; }
    std::vector<int> start(w.size());
    for (size_t b = 0, t = 0; b < w.size(); b++) { start[b] = (int)t; t += w[b].n; }
    std::map<int, std::vector<int>> by_first;
    for (size_t b = 0; b < w.size(); b++) {
      const int mode = rflag.ptr<int>()[start[b]];
      if (mode == 0 || mode == 2 || mode == 3) { casc_block[2]++; return; }  // another kernel shares the slots
      if ((mode == 4 || mode == 5) && !w[b].blocks->empty()) by_first[(*w[b].blocks)[0]].push_back((int)b);
    }
    int* kb = kbeg.ptr<int>();
    for (size_t b = 0; b < w.size(); b++) kb[b] = 0;
    int* rt = sh_rows_t.ptr<int>();
    int* rj = sh_rows_j.ptr<int>();
    int* ti = sh_tinfo.ptr<int>();
    for (auto& [first, mem] : by_first) {
      if (mem.size() < 2) continue;
      // Common block prefix, capped so that every member's new rows lie after it.
      const auto& lead = *w[mem[0]].blocks;
      int c = (int)lead.size();
      for (int b : mem) {
        const auto& bl = *w[b].blocks;
        int k = 0;
        while (k < c && k < (int)bl.size() && bl[k] == lead[k]) k++;
        c = std::min(k, w[b].pos0 / KV_BLOCK);
      }
      if (c < cascade_min) continue;
      std::vector<std::pair<int, int>> rows;  // (token row, head in group)
      for (int b : mem) {
        kb[b] = c * KV_BLOCK;
        for (int r = 0; r < w[b].n; r++) {
          shrow.ptr<int>()[start[b] + r] = 1;
          for (int j = 0; j < 6; j++) rows.push_back({start[b] + r, j});
        }
      }
      for (size_t i = 0; i < rows.size(); i += 48, sh_tiles++) {
        ti[sh_tiles * 2] = mem[0];
        ti[sh_tiles * 2 + 1] = c * KV_BLOCK;
        for (int k = 0; k < 48; k++) {
          const bool ok = i + k < rows.size();
          rt[sh_tiles * 48 + k] = ok ? rows[i + k].first : -1;
          rj[sh_tiles * 48 + k] = ok ? rows[i + k].second : 0;
        }
      }
      cascade = true;
    }
    if (cascade) casc_steps++; else casc_block[3]++;
  }

  // Chunk tiles (mode 6): each such sequence's rows x 6 heads, 48 to a tile, every tile reading its sequence's
  // block table up to its last row's position (rows mask their own causal end in the kernel).
  bool any_chunk = false;
  // decode_test followup (2026-10-30, Studio): 16 agents x 32-token turns at 128k 2866 -> 844 ms per step, x 136 at
  // 128k 4986 -> 3418, x 16 at 16k 300 -> 198; a lone 32-token turn at 128k 177 -> 79. A tie at 2k, 1-2% slower at
  // 512 rows (flash already fills the GPU there), hence the cap.
  int chunk_tile_max = getenv("CHUNK_TILE_MAX") ? atoi(getenv("CHUNK_TILE_MAX")) : 256;
  int chunk_tile_ctx = getenv("CHUNK_TILE_CTX") ? atoi(getenv("CHUNK_TILE_CTX")) : 0;
  void plan_chunks(const std::vector<SeqWork>& w) {
    int* rt = sh_rows_t.ptr<int>();
    int* rj = sh_rows_j.ptr<int>();
    int* ti = sh_tinfo.ptr<int>();
    sh_tiles = 0;
    for (size_t b = 0, t0 = 0; b < w.size(); t0 += w[b].n, b++) {
      if (rflag.ptr<int>()[t0] != 6) continue;
      const int rows = w[b].n * 6;
      for (int i = 0; i < rows; i += 48, sh_tiles++) {
        const int last = std::min(rows, i + 48) - 1;
        ti[sh_tiles * 2] = (int)b;
        ti[sh_tiles * 2 + 1] = w[b].tree ? w[b].pos0 : w[b].pos0 + last / 6 + 1;  // a tree's rows: the context only
        for (int k = 0; k < 48; k++) {
          const bool ok = i + k < rows;
          rt[sh_tiles * 48 + k] = ok ? (int)t0 + (i + k) / 6 : -1;
          rj[sh_tiles * 48 + k] = (i + k) % 6;
        }
      }
    }
  }

  // Flash attention for one sequence's rows [row0, row0 + n), whose queries are the last n of kL positions.
  void encode_flash(Enc& e, const Layer& ly, const FlashSeq& f) {
    const bool d128 = ly.dh == 128, d512 = ly.dh == 512;
    const int HD = ly.hkv * ly.dh;
    GatherKvArgs ga{HD, f.kL, max_blocks, f.seq, ly.dh, kv_ns()};
    e.k(kvk("kv_gather")).buf(0, kc[ly.idx]).buf(1, vc[ly.idx]).buf(2, block_table).buf(3, kv_k).buf(4, kv_v)
        .args(5, ga).buf(6, ks[ly.idx]).buf(7, vs[ly.idx]);
    e.threads(S3(HD, f.kL), S3(256));
    const int bq = 64, bk = 32;
    MlxAttnParams ap{1, cfg.hq, ly.dh, f.n, f.kL, cfg.hq / ly.hkv, cfg.attn_scale(ly.dh),
                     (f.n + bq - 1) / bq, (f.kL + bk - 1) / bk, f.n / bq, f.kL / bk, f.n % bq, f.kL % bk, f.kL - f.n,
                     {0, ly.dh, (int64_t)cfg.hq * ly.dh}, {0, ly.dh, HD}, {0, ly.dh, HD},
                     {0, ly.dh, (int64_t)cfg.hq * ly.dh}, ly.window};
    auto pso = g.pso_fc(d128 ? "mlx_attn_bf16_d128" : d512 ? "mlx_attn_bf16_d512" : "mlx_attn_bf16_d256", {{200, f.n % bq == 0}, {201, f.kL % bk == 0}, {300, false}, {301, true}, {302, false}});
    size_t qoff = (size_t)f.row0 * cfg.hq * ly.dh * 2;
    e.k(pso).buf(0, qbuf.at(qoff)).buf(1, kv_k).buf(2, kv_v).buf(3, act.at(qoff)).args(4, ap);
    e.groups(S3(ap.NQ, cfg.hq, 1), d128 ? S3(32, 4, 1) : d512 ? S3(32, 4, 4) : S3(32, 4, 2));
    if (!ly.gated) return;
    GateArgs gt{cfg.hq, ly.dh, f.row0, ly.proj};
    e.k("attn_gate").buf(0, act).buf(1, proj).args(2, gt);
    e.threads(S3(cfg.hq * ly.dh, f.n), S3(256));
  }

  // One forward pass over T token rows from S sequences (inputs from set_batch). K/V and recurrent state are
  // updated in place; logits for each sequence's last token land in `logits` as bfloat [S, vocab].
  // Command buffers committed early in this step (the GPU starts on them while the CPU encodes the rest; ~1,400
  // dispatches take ~1 ms to encode). The caller waits on its final buffer and checks these for errors.
  std::vector<id<MTLCommandBuffer>> early;
  // Layers per command buffer (0 = one buffer). The server sets 16 (CB_SPLIT overrides); the test tools keep one
  // buffer so their GPU timing covers the whole step.
  int cb_split = getenv("CB_SPLIT") ? atoi(getenv("CB_SPLIT")) : 0;
  // Flash-Next: steps with prompt rows run as one command buffer (set per step by the scheduler; FN_SPLIT_PF=1 keeps
  // splitting). Split, the step that ends a 16k prompt (18 rows) took 350-780 ms on the host instead of 45 ms
  // (2026-11-02 night; cause not found: the GPU idled between the early command buffers while the host encoded).
  bool no_split_step = false;
  AneFfn* ane = nullptr;  // the Neural Engine FFN split (ANE_UNITS > 0): prefill steps of >= 512 rows
  void commit_early(id<MTLCommandBuffer> __strong& cb) {
    [cb commit];
    early.push_back(cb);
    cb = [g.queue commandBuffer];
  }

  // The layer's ANE weights into weight set `set`: rotated int8 rows of gate [Cg, I) and up [I + Cg, 2I) (W8A8) or
  // plain int8 rows (W8A16), and down's rows over input columns [Cg, I), each with per-row scales.
  void ane_requant(Enc& e, Layer& ly, int set) {
    const int H = cfg.hidden, I = cfg.inter, Cg = ane->Cg, C = ane->C, seg = AneFfn::SEG;
    AneFfn::WSet& w = ane->ws[set];
    struct AW { int K, n0, c0, cn, flip, scst, ost, q0, qn; };
    const int fl = ly.gate_up.flipped ? 8 : 0;
    for (int up = 0; up < 2; up++) {
      AneSurf& sc = up ? w.su : w.sg;
      auto& o = up ? w.wu : w.wg;
      if (ane->a8) {
        AW a{H, up * I + Cg, 0, H, fl, sc.stride / 2, o[0].stride, 0, 0};
        e.k("ane_req_rot").buf(0, ly.gate_up.wq).buf(1, ly.gate_up.s).buf(2, ly.gate_up.b).buf(3, sc.buf).args(4, a)
            .buf(5, o[0].buf).buf(6, o[1].buf);
        e.groups(S3(C), S3(256));
        continue;
      }
      AW a{H, up * I + Cg, 0, H, fl, sc.stride / 2, 0, 0, 0};
      e.k("ane_rowscale").buf(0, ly.gate_up.wq).buf(1, ly.gate_up.s).buf(2, ly.gate_up.b).buf(3, sc.buf).args(4, a);
      e.groups(S3(C), S3(256));
      for (size_t k = 0; k < o.size(); k++) {
        AW q{H, up * I + Cg, 0, H, fl, sc.stride / 2, o[k].stride, (int)k * seg, seg};
        e.k("ane_quant").buf(0, ly.gate_up.wq).buf(1, ly.gate_up.s).buf(2, ly.gate_up.b).buf(3, sc.buf).buf(5, o[k].buf).args(4, q);
        e.threads(S3(seg / 8, C), S3(256));
      }
    }
    const int fd = ly.down.flipped ? 8 : 0;
    if (ane->a8d) {  // rotated per 512-block like the program's intermediate
      AW a{I, 0, Cg, C, fd, w.sd.stride / 2, w.wd[0].stride, 0, 0};
      e.k("ane_req_rot_cols").buf(0, ly.down.wq).buf(1, ly.down.s).buf(2, ly.down.b).buf(3, w.sd.buf).args(4, a);
      for (int i = 0; i < 7; i++) e.buf(5 + i, w.wd[std::min<size_t>(i, w.wd.size() - 1)].buf);
      e.groups(S3(H), S3(256));
      return;
    }
    AW a{I, 0, Cg, C, fd, w.sd.stride / 2, 0, 0, 0};
    e.k("ane_rowscale").buf(0, ly.down.wq).buf(1, ly.down.s).buf(2, ly.down.b).buf(3, w.sd.buf).args(4, a);
    e.groups(S3(H), S3(256));
    for (size_t i = 0, q0 = 0; i < w.wd.size(); q0 += ane->dw[i], i++) {
      AW q{I, 0, Cg, C, fd, w.sd.stride / 2, w.wd[i].stride, (int)q0, ane->dw[i]};
      e.k("ane_quant").buf(0, ly.down.wq).buf(1, ly.down.s).buf(2, ly.down.b).buf(3, w.sd.buf).buf(5, w.wd[i].buf).args(4, q);
      e.threads(S3(ane->dw[i] / 8, H), S3(256));
    }
  }

  // One layer's FFN split between the GPU (channels [0, Cg)) and the Neural Engine (the rest), for T rows padded to
  // the ANE function's R. The GPU packs the normed rows (and, for the step's first layer, re-quantizes its ANE
  // weights), commits and waits; then the ANE runs on a thread while the GPU computes its channels and re-quantizes
  // the next layer's ANE weights into the other weight set; the ANE's partial down projection then joins the
  // residual. Synchronised from the host per layer (a chunk of >= 512 rows takes seconds on the small GPUs).
  void ffn_split(Enc& e, id<MTLCommandBuffer> __strong& cb, int li, int T, int R) {
    Layer& ly = L[li];
    const int H = cfg.hidden, I = cfg.inter, Cg = ane->Cg, seg = AneFfn::SEG, set = li & 1;
    flush_split(e);
    rmsnorm(e, x, ly.post_norm, xn, T, H, H, H);
    for (size_t k = 0; k < ane->xs.size(); k++) {
      struct { int T, H, k0, st; } pa{T, H, (int)k * seg, ane->xs[k].stride / (ane->a8 ? 1 : 2)};
      if (ane->a8) {
        e.k("ane_pack_x8").buf(0, xn).buf(1, ane->xs[k].buf).buf(2, ane->tx[k].buf).args(3, pa);
        e.groups(S3(R), S3(256));
      } else {
        e.k("ane_pack_x").buf(0, xn).buf(1, ane->xs[k].buf).args(2, pa);
        e.threads(S3(R, seg), S3(256));
      }
    }
    if (ane_ready != li) ane_requant(e, ly, set);
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
    const auto t1 = clk::now();
    std::thread th([this, R, set] { ane->eval(R, set); });
    cb = [g.queue commandBuffer];
    e.restart(cb);
    const BufRef gb = proj, ub = proj.at((size_t)T * Cg * 2);
    gemm(e, xn, rows_of(ly.gate_up, 0, Cg), gb, T, 0);
    gemm(e, xn, rows_of(ly.gate_up, I, Cg), ub, T, 0);
    struct { int C; } sa{Cg};
    e.k("silu_mul2").buf(0, gb).buf(1, ub).buf(2, act).args(3, sa);
    e.threads(S3(Cg, T), S3(256));
    gemm(e, act, kslice_of(ly.down, 0, Cg), x, T, 1);
    if (li + 1 < (int)L.size()) { ane_requant(e, L[li + 1], set ^ 1); ane_ready = li + 1; }
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
    const auto t2 = clk::now();
    th.join();
    const auto t3 = clk::now();
    if (getenv("ANE_DEBUG") && li < 3) {  // the ANE's output: NaNs, magnitude, the per-token scale row
      const _Float16* yp = ane->y.buf.ptr<_Float16>();
      const int st = ane->y.stride / 2, rows = H + (ane->a8d ? 1 : 0);
      int nan = 0; float mx = 0;
      for (int h = 0; h < H; h++) for (int t = 0; t < T; t++) { float v = yp[(size_t)h * st + t]; if (v != v) nan++; else mx = std::max(mx, fabsf(v)); }
      fprintf(stderr, "ane debug layer %d: %d NaN of %d, max |y| %.4g", li, nan, H * T, mx);
      if (rows > H) fprintf(stderr, ", scale row t0..2: %.4g %.4g %.4g", (float)yp[(size_t)H * st], (float)yp[(size_t)H * st + 1], (float)yp[(size_t)H * st + 2]);
      const _Float16* sd = ane->ws[set].sd.buf.ptr<_Float16>();
      fprintf(stderr, ", sd0 %.4g, wd0[0..3] %d %d %d\n", (float)sd[0], ane->ws[set].wd[0].buf.ptr<int8_t>()[0],
              ane->ws[set].wd[0].buf.ptr<int8_t>()[1], ane->ws[set].wd[0].buf.ptr<int8_t>()[2]);
    }
    if (ane_prof) {  // per layer: GPU up to the hand-off (mixer, norm, pack), GPU share, wait for the ANE after it
      auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      prof_pre += ms(t0, t1); prof_gpu += ms(t1, t2); prof_wait += ms(t2, t3); prof_layers++;
      if (li + 1 == (int)L.size()) {
        fprintf(stderr, "ane: T=%d R=%d per layer: before %.2f ms, GPU share %.2f, then ANE wait %.2f; ANE eval %.2f ms\n", T, R,
                prof_pre / prof_layers, prof_gpu / prof_layers, prof_wait / prof_layers, ane->ms_total / std::max(1L, ane->evals));
        prof_pre = prof_gpu = prof_wait = 0; prof_layers = 0; ane->ms_total = 0; ane->evals = 0;
      }
    }
    cb = [g.queue commandBuffer];
    e.restart(cb);
    struct { int H, st; } ja{H, ane->y.stride / 2};
    e.k(ane->a8d ? "ane_join_s" : "ane_join").buf(0, x).buf(1, ane->y.buf).args(2, ja);
    e.threads(S3(H, T), S3(256));
    for (size_t c = 0; c < cap_layers.size(); c++)
      if (cap_layers[c] == li) {
        CapArgs ca{H, (int)cap_layers.size() * H, (int)c * H};
        e.k("capture_cols").buf(0, x).buf(1, hcap).args(2, ca);
        e.threads(S3(H, T), S3(256));
      }
  }
  bool ane_prof = getenv("ANE_PROF") && atoi(getenv("ANE_PROF"));
  double prof_pre = 0, prof_gpu = 0, prof_wait = 0;
  long prof_layers = 0;
  int ane_ready = -1;  // the layer whose ANE weights the other set already holds (re-quantized during the previous layer)

  // QSA indexer for one attention layer over the step's T rows: index queries, raw keys and newly completed pooled
  // groups (qsa_prep, every step, so the cache is complete when a sequence passes the dense end); then, for the step's
  // sparse rows, scores (qsa_score) and the top-k selection (qsa_select), in row groups that fit the score scratch.
  void encode_qsa(Enc& e, const Layer& ly, int T) {
    if (!on("qsa")) return;  // SKIP=qsa: timing only (attention reads stale selections)
    const QsaLoc& lc = qsa_loc[ly.idx];
    const int Hi = cfg.qsa_heads, Di = cfg.qsa_dim;
    struct { int T, Hi, Di, rot, rpair, rden; float theta, eps; int max_blocks, ratio, raw_blk, pool_blk, stride; } qa{
        T, Hi, Di, ly.rot, ly.rot / 2, ly.rot, ly.theta, cfg.eps, max_blocks, cfg.qsa_ratio, qsa_blk, qsa_blk,
        (Hi + 1) * Di};
    if (ly.iproj_fused) qa.stride = ly.proj;  // index rows inside in_proj's output
    e.k("qsa_prep").buf(0, ly.iproj_fused ? proj.at((size_t)ly.iproj_fused * 2) : qsa_proj).buf(1, ly.iqn).buf(2, ly.ikn).buf(3, qsa_iq).buf(4, lc.raw).buf(5, lc.pool)
        .buf(6, block_table).buf(7, pos).buf(8, tseq).buf(9, rpos).buf(10, seq_start).args(11, qa);
    e.groups(S3(Hi + 1, T), S3(Di));
    if (!qsa_nsp) return;
    const int nsc = qsa_maxnb, G = std::max(1, (int)std::min<size_t>(qsa_nsp, qsa_sc_bytes / ((size_t)nsc * 4)));
    for (int r0 = 0; r0 < qsa_nsp; r0 += G) {
      const int n = std::min(G, qsa_nsp - r0);
      struct { int Hi, Di, max_blocks, ratio, pool_blk, row0, nsc; } sa{Hi, Di, max_blocks, cfg.qsa_ratio, qsa_blk, r0, nsc};
      e.k("qsa_score").buf(0, qsa_iq).buf(1, lc.pool).buf(2, block_table).buf(3, pos).buf(4, tseq).buf(5, qsa_rows)
          .buf(6, qsa_sc).args(7, sa);
      e.groups(S3((nsc + 255) / 256, n), S3(256));
      struct { int K, ratio, row0, nsc; } sl{cfg.qsa_topk, cfg.qsa_ratio, r0, nsc};
      const int vpt = (nsc + 1023) / 1024;
      e.k(vpt <= 2 ? "qsa_select_v2" : vpt <= 8 ? "qsa_select_v8" : vpt <= 32 ? "qsa_select_v32" : "qsa_select_v64").buf(0, qsa_sc).buf(1, qsa_rows).buf(2, pos).buf(3, qsa_sel).args(4, sl);
      e.groups(S3(n), S3(1024));
    }
  }

  AttnArgs attn_args(int T) const {
    return AttnArgs{T, cfg.hq, cfg.hkv, cfg.dh, cfg.rot, cfg.attn_proj(), max_blocks, cfg.eps, cfg.theta,
                    cfg.attn_scale(), kv_ns()};
  }
  // One decoder layer over the step's T rows from S sequences (batch inputs from set_batch). li: the layer's index
  // in L, or -1 for a layer outside it (the MTP head's), which has no ANE split, capture or PLE.
  void encode_layer(Enc& e, id<MTLCommandBuffer> __strong& cb, Layer& ly, int li, int T, int S, AttnArgs& aa,
                    const GdnArgs& ga) {
    const int H = cfg.hidden;
    if (cfg.qwen4) {
      if (ly.ple) encode_ple(e, ly, T, S);
      hyper_pre(e, ly.ahc, T, true, xn);
    } else {
      rmsnorm_for(e, x, ly.in_norm, xn, T, H, ly.in_proj);
    }
    gemm(e, xn, ly.in_proj, proj, T, 0);
    if (ly.has_qsa && qsa_on && on("attn") && !ly.iproj_fused) {
      // the in_proj GEMM left xn's int8 rows in a8u when both take the W4A8 path (same K): reuse them
      if (w4a8_route(ly.in_proj, T) && w4a8_route(ly.iproj, T) && ly.in_proj.K == ly.iproj.K && on("gemm")) a8_ready = true;
      gemm(e, xn, ly.iproj, qsa_proj, T, 0);
    }
    if (!ly.linear) {  // per-layer attention shape (Gemma-4 global layers differ from its sliding ones)
      aa.rot = ly.rot; aa.Hkv = ly.hkv; aa.Dh = ly.dh; aa.proj_stride = ly.proj; aa.theta = ly.theta;
      aa.scale = cfg.attn_scale(ly.dh);
      aa.qs = ly.gated ? 2 * ly.dh : ly.dh;
      aa.koff = cfg.hq * aa.qs;
      aa.voff = ly.k_eq_v ? aa.koff : aa.koff + ly.hkv * ly.dh;
      const bool prop = cfg.gemma && li >= 0 && cfg.global(li);  // proportional rope: pairs (i, i + Dh/2), exponent over Dh
      aa.rpair = prop ? ly.dh / 2 : ly.rot / 2;
      aa.rden = prop ? ly.dh : ly.rot;
      aa.vnorm = cfg.gemma;
    }
    if (ly.linear && on("gdn")) {
      // In a verify step each GDN layer keeps its own copies (the tape) so acceptance can replay them.
      // Steps with more rows than the tapes hold use the scratch and copy the verify rows (rows < spec_end) after.
      const bool direct = verify && T <= tape_rows && !(tape_force && tape_xs.buf), tcopy = verify && !direct;
      const BufRef& co = direct ? tape_c[ly.idx] : conv_out;
      const BufRef& gb = direct ? tape_g[ly.idx] : gbuf;
      const BufRef& bb = direct ? tape_b[ly.idx] : betabuf;
      const BufRef& tx = direct ? tape_x[ly.idx] : tcopy ? tape_xs : conv_out;
      // Flash-Next prefill (FN_GDN_PF, rows per step from which; 0 off): the conv one thread per (channel, row) and
      // the recurrence with FN_GDN_R value rows per simdgroup (kernels/qwen4.metal; bit-identical outputs).
      const bool fpf = (cfg.qwen4 || gdn_pf_all) && fn_gdn_pf > 0 && T >= fn_gdn_pf && cfg.ldk == 128 && cfg.ldv % (4 * fn_gdn_r) == 0;
      // Token trees (TREE; never with Flash-Next): tree sequences have their own kernels below, the chain kernels see
      // them as empty (seq_len_g) and are not dispatched at all when every sequence is a tree (an empty dispatch still
      // costs a few us: ~400 of them were 1 ms of a tree cycle).
      const BufRef& sl = n_tree ? seq_len_g : seq_len;
      if (fpf) {
        e.k("gdn_conv_pf").buf(0, proj).buf(1, conv_state[ly.idx]).buf(2, ly.conv_w).buf(3, co).buf(4, slot)
            .args(5, ga).buf(6, seq_start).buf(7, tseq).buf(8, tx);
        e.threads(S3(cfg.conv_dim(), T), S3(256));
        e.k("gdn_conv_pf_state").buf(0, proj).buf(1, conv_state[ly.idx]).buf(4, slot).args(5, ga).buf(6, seq_start)
            .buf(7, seq_len).buf(9, spec);
        e.threads(S3(cfg.conv_dim(), S), S3(256));
      } else if (!tree_only) {
        e.k("gdn_conv").buf(0, proj).buf(1, conv_state[ly.idx]).buf(2, ly.conv_w).buf(3, co).buf(4, slot)
            .args(5, ga).buf(6, seq_start).buf(7, sl).buf(8, tx).buf(9, spec);
        e.threads(S3(cfg.conv_dim(), S), S3(256));
      }
      if (n_tree && !(tree_skip & 2)) {
        e.k("gdn_conv_tree").buf(0, proj).buf(1, conv_state[ly.idx]).buf(2, ly.conv_w).buf(3, co).buf(4, slot)
            .args(5, ga).buf(6, seq_start).buf(7, seq_len).buf(8, tx).buf(9, tseqs).buf(10, tpar).buf(11, tdep);
        e.threads(S3(cfg.conv_dim(), n_tree), S3(256));
      }
      e.k("gdn_prep").buf(0, co).buf(1, proj).buf(2, ly.A_log).buf(3, ly.dt_bias).buf(4, gb).buf(5, bb).args(6, ga);
      e.groups(S3(2 * cfg.lhk + 1, T), S3(cfg.ldk));
      if (fpf && fn_gdn_r > 1) {
        char sn[24];
        snprintf(sn, sizeof sn, "gdn_step_r%d", fn_gdn_r);
        e.k(sn).buf(0, co).buf(1, gb).buf(2, bb).buf(3, ssm_state[ly.idx]).buf(4, ybuf).buf(5, slot)
            .args(6, ga).buf(7, seq_start).buf(8, seq_len).buf(9, spec);
        e.threads(S3(32, cfg.ldv / fn_gdn_r, S * cfg.lhv), S3(32, 4, 1));
      } else if (!tree_only) {
        e.k("gdn_step").buf(0, co).buf(1, gb).buf(2, bb).buf(3, ssm_state[ly.idx]).buf(4, ybuf).buf(5, slot)
            .args(6, ga).buf(7, seq_start).buf(8, sl).buf(9, spec);
        e.threads(S3(32, cfg.ldv, S * cfg.lhv), S3(32, 4, 1));
      }
      if (n_tree && !(tree_skip & 4)) {
        e.k("gdn_step_tree").buf(0, co).buf(1, gb).buf(2, bb).buf(3, ssm_state[ly.idx]).buf(4, ybuf).buf(5, slot)
            .args(6, ga).buf(7, seq_start).buf(8, seq_len).buf(9, tseqs).buf(10, tpar).buf(11, tdep).buf(12, tanc);
        e.threads(S3(32, cfg.ldv, n_tree * cfg.lhv), S3(32, 4, 1));
      }
      if (tcopy && spec_end > 0) {
        struct { int n, conv_dim, hv; } ta{spec_end, cfg.conv_dim(), cfg.lhv};
        e.k("tape_save").buf(0, tx).buf(1, co).buf(2, gb).buf(3, bb).buf(4, tape_x[ly.idx]).buf(5, tape_c[ly.idx])
            .buf(6, tape_g[ly.idx]).buf(7, tape_b[ly.idx]).args(8, ta);
        e.threads(S3(cfg.conv_dim(), spec_end), S3(256));
      }
      e.k("gdn_out_norm").buf(0, ybuf).buf(1, proj).buf(2, ly.gdn_norm).buf(3, act).args(4, ga);
      e.groups(S3(cfg.lhv, T), S3(cfg.ldv));
    } else if (!ly.linear && on("attn")) {
      char sfx[16];
      snprintf(sfx, sizeof sfx, "_g%d_d%d", cfg.hq / ly.hkv, ly.dh);
      e.k(kvk("attn_prep")).buf(0, proj).buf(1, ly.q_norm).buf(2, ly.k_norm).buf(3, qbuf).buf(4, kc[ly.idx])
          .buf(5, vc[ly.idx]).buf(6, block_table).buf(7, pos).args(8, aa).buf(9, tseq).buf(10, ks[ly.idx])
          .buf(11, vs[ly.idx]).buf(12, rpos);
      e.groups(S3(cfg.hq + ly.hkv, T), S3(ly.dh));
      if (ly.has_qsa && qsa_on) encode_qsa(e, ly, T);
      if (use_gqa) {
        const BufRef& pos = n_tree ? apos : this->pos;  // tree rows: the committed context only (attn_tree adds the rest)
        // One threadgroup per (kv head, sequence, split of positions); enough splits to fill the GPU.
        int units = std::max(1, attn_units);
        static int split_len = getenv("ATTN_SPLIT_LEN") ? atoi(getenv("ATTN_SPLIT_LEN")) : 256;  // min positions per split
        // ~1024 threadgroups per layer: 2048 made a lone 64k verify block write and merge 256 splits of partials
        // (36.9 -> 35.1 ms per step at 64k, 30.6 -> 30.1 at 32k, 8k unchanged).
        static int split_tgs = getenv("ATTN_SPLIT_TGS") ? atoi(getenv("ATTN_SPLIT_TGS")) : 1024;
        static int split_tgs_mpp = getenv("ATTN_SPLIT_TGS_MPP") ? atoi(getenv("ATTN_SPLIT_TGS_MPP")) : 128;
        // attn_mpp alone (one long verify block) does best with few, long splits (4 kv heads x 32 splits).
        static int split_tgs_mpp32 = getenv("ATTN_SPLIT_TGS_MPP32") ? atoi(getenv("ATTN_SPLIT_TGS_MPP32")) : 128;
        // One sequence alone: two threadgroups per GPU core (160 on the 80-core M5 Ultra: 256k x 1 R8 38.5 -> 36.4 ms,
        // 128k 30.8 -> 29.9, 32k unchanged). With 2-4 sequences 160 was 20-25% slower than 128 at 128k, so only here.
        static int split_tgs_one = getenv("ATTN_SPLIT_TGS_ONE") ? atoi(getenv("ATTN_SPLIT_TGS_ONE")) : std::max(128, 2 * gpu_cores());
        int tgs = (any_mpp || any_mpp32) && !any_nax && !any_multi && !any_chunk ? (any_mpp32 ? split_tgs_mpp32 : split_tgs_mpp) : split_tgs;
        if ((any_mpp || any_mpp32) && !any_nax && !any_multi && !any_chunk && attn_units == 1) tgs = split_tgs_one;
        // Flash-Next (GQ 12): shorter splits (FN_SPLIT_ROW / FN_SPLIT_MULTI positions), 2k context B=1: decode step
        // 12.7 -> 12.1 ms, 4-row verify 24.2 -> 19.4 (2026-11-01 night). ATTN_SPLIT_LEN still overrides.
        // (multi-row blocks: at most 64 splits; 8k x 4 rows: 32-position splits 26.3 ms, 256 24.3).
        const int slen = cfg.qwen4 && !getenv("ATTN_SPLIT_LEN")
                             ? (any_multi ? std::max(fn_split_multi, (max_ctx + 63) / 64) : fn_split_row) : split_len;
        int splits = std::max(1, std::min({(tgs + ly.hkv * units - 1) / (ly.hkv * units),
                                           (max_ctx + slen - 1) / slen, std::max(1, kAttnPartRows / T)}));
        // Flash-Next prefill chunks with QSA union tiles: every sparse row counts as a unit above, which leaves one
        // split, i.e. 2 threadgroups per tile and batches of qsa_umax tiles one after another; FN_QSA_SPLITS more.
        if (cfg.qwen4 && qsa_nu && qsa_maxn > kMaxBlock && fn_qsa_splits > splits)
          splits = std::max(splits, std::min(fn_qsa_splits, kAttnPartRows / T));
        const int align = any_mpp || any_mpp32 ? 128 : KV_BLOCK;  // attn_mpp works in 128-key groups
        int chunk = ((max_ctx + splits - 1) / splits + align - 1) / align * align;
        // Cascade: own splits as computed, plus sh_splits slots for the shared prefix (attn_shared).
        static int sh_tgs = getenv("CASCADE_TGS") ? atoi(getenv("CASCADE_TGS")) : 256;
        int sh_splits = 0;
        if (cascade) {
          sh_splits = std::max(1, std::min(64, (sh_tgs + ly.hkv * sh_tiles - 1) / (ly.hkv * sh_tiles)));
          if ((splits + sh_splits) * T > kAttnPartRows) sh_splits = 0;
        }
        const bool casc = cascade && sh_splits > 0;
        GqaArgs qa{T, cfg.hq, ly.hkv, ly.dh, max_blocks, splits + sh_splits, chunk, ly.proj, cfg.attn_scale(ly.dh),
                   nax_tiles, casc ? splits : 0, kv_ns(), ly.window, ly.gated};
        e.k(kvk((std::string("attn_gqa_row") + sfx).c_str())).buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx]).buf(3, block_table)
            .buf(4, pos).buf(5, tseq).buf(6, attn_part).args(7, qa).buf(8, rflag).buf(9, ks[ly.idx])
            .buf(10, vs[ly.idx]);
        e.groups(S3(ly.hkv, T, splits), S3(128));
        if (any_multi) {
          e.k(kvk((std::string("attn_gqa") + sfx).c_str())).buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx]).buf(3, block_table)
              .buf(4, pos).buf(5, seq_start).buf(6, attn_part).args(7, qa).buf(8, rflag).buf(9, seq_len)
              .buf(10, ks[ly.idx]).buf(11, vs[ly.idx]);
          const int gd = cfg.hq / ly.hkv * ly.dh;  // GROWS in attn_gqa
          e.groups(S3(ly.hkv, S, splits), S3(32, gd <= 1536 ? 8 : gd <= 3072 ? 4 : 2, 1));
        }
        if (cfg.qwen4 && any_mpp) {  // Flash-Next: attn_fn (12 heads per kv head x <= 4 rows = 48 packed rows)
          const bool iq = kv8 && attn_iq;
          e.k(iq ? "fn_q_pack_i8_g12" : "fn_q_pack_g12").buf(0, qbuf).buf(1, qpack).buf(2, seq_start).buf(3, seq_len)
              .args(4, qa).buf(5, qscale);
          e.groups(S3(ly.hkv * 48, S), S3(256));
          e.k(iq ? "attn_fn_g12_kv8_iq" : kv8 ? "attn_fn_g12_kv8" : "attn_fn_g12").buf(0, qpack).buf(1, kc[ly.idx])
              .buf(2, vc[ly.idx]).buf(3, block_table).buf(4, pos).buf(5, seq_start).buf(6, attn_part).args(7, qa)
              .buf(8, rflag).buf(9, seq_len).buf(10, ks[ly.idx]).buf(11, vs[ly.idx]).buf(12, kbeg).buf(13, qscale);
          e.groups(S3(ly.hkv, S, splits), S3(256));
        }
        for (int r : {48, 32}) {
          if (cfg.qwen4 || (r == 48 ? !any_mpp : !any_mpp32)) continue;
          const bool iq = kv8 && attn_iq;
          e.k(iq ? (r == 48 ? "q_pack_i8_48" : "q_pack_i8_32") : (r == 48 ? "q_pack48" : "q_pack32")).buf(0, qbuf)
              .buf(1, qpack).buf(2, seq_start).buf(3, seq_len).args(4, qa).buf(5, qscale);
          e.groups(S3(ly.hkv * r, S), S3(256));
          std::string nm = r == 48 ? (kv8 ? "attn_mpp8_kv8" : "attn_mpp8") : (kv8 ? "attn_mpp8_kv8_r32" : "attn_mpp8_r32");
          e.k((iq ? nm + "_iq" : nm).c_str()).buf(0, qpack).buf(1, kc[ly.idx]).buf(2, vc[ly.idx])
              .buf(3, block_table).buf(4, pos).buf(5, seq_start).buf(6, attn_part).args(7, qa).buf(8, rflag)
              .buf(9, seq_len).buf(10, ks[ly.idx]).buf(11, vs[ly.idx]).buf(12, kbeg).buf(13, qscale);
          e.groups(S3(ly.hkv, S, splits), S3(256));
        }
        if (any_chunk) {
          e.k("q_pack_sh").buf(0, qbuf).buf(1, qpack_sh).buf(2, sh_rows_t).buf(3, sh_rows_j).args(4, qa);
          e.groups(S3(ly.hkv * 48, sh_tiles), S3(256));
          e.k(kv8 ? "attn_shared_kv8" : "attn_shared").buf(0, qpack_sh).buf(1, kc[ly.idx]).buf(2, vc[ly.idx])
              .buf(3, block_table).buf(4, pos).buf(5, sh_tinfo).buf(6, attn_part).args(7, qa).buf(8, sh_rows_t)
              .buf(9, sh_rows_j).buf(10, ks[ly.idx]).buf(11, vs[ly.idx]).buf(12, kbeg).buf(13, qscale);
          e.groups(S3(ly.hkv, sh_tiles, splits), S3(256));
        }
        if (casc) {
          e.k("q_pack_sh").buf(0, qbuf).buf(1, qpack_sh).buf(2, sh_rows_t).buf(3, sh_rows_j).args(4, qa);
          e.groups(S3(ly.hkv * 48, sh_tiles), S3(256));
          e.k(kv8 ? "attn_shared_kv8" : "attn_shared").buf(0, qpack_sh).buf(1, kc[ly.idx]).buf(2, vc[ly.idx])
              .buf(3, block_table).buf(4, pos).buf(5, sh_tinfo).buf(6, attn_part).args(7, qa).buf(8, sh_rows_t)
              .buf(9, sh_rows_j).buf(10, ks[ly.idx]).buf(11, vs[ly.idx]).buf(12, kbeg).buf(13, qscale);
          e.groups(S3(ly.hkv, sh_tiles, sh_splits), S3(256));
        }
        if (any_nax) {
          static const char* nx = getenv("NAX_X");  // timing experiments (wrong results): 1, 2, 4, 8, 15
          char nk[40];
          static const bool vpre = getenv("NAX_VPRE") != nullptr;
          snprintf(nk, sizeof nk, "attn_paged_nax_wm%d%s%s%s", nax_tile_rows / 16, nx ? "x" : "", nx ? nx : "", vpre ? "v" : "");
          e.k(kvk(nk)).buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx]).buf(3, block_table)
              .buf(4, pos).buf(5, seq_start).buf(6, attn_part).args(7, qa).buf(8, rflag).buf(9, seq_len)
              .buf(10, ks[ly.idx]).buf(11, vs[ly.idx]);
          e.groups(S3(ly.hkv * nax_tiles, S, splits), S3(nax_tile_rows / 16 * 4 * 32));
        }
        if (qsa_nsp && ly.has_qsa && qsa_on && qsa_nu && qsa_maxnb <= 32768 && on("qsa")) {  // union tiles on attn_fn
          const int cap = 4 * cfg.qsa_topk, tk = qsa_P2 * KV_BLOCK;
          GqaArgs q2 = qa;
          q2.max_blocks = qsa_P2;
          q2.ns = qsa_ns2;
          q2.own_splits = 0;
          q2.qtiles = tk;  // the mask's stride (attn_fn MASK)
          q2.chunk = ((tk + splits - 1) / splits + 127) / 128 * 128;
          const bool iq = kv8 && attn_iq && qsa_maxn <= kMaxBlock;
          for (int u0 = 0; u0 < qsa_nu; u0 += qsa_umax) {
            const int n = std::min(qsa_umax, qsa_nu - u0);
            const BufRef us = qsa_ustart.at((size_t)u0 * 4), ul = qsa_ulen.at((size_t)u0 * 4);
            struct { int K, ratio, cap; } ua{cfg.qsa_topk, cfg.qsa_ratio, cap};
            e.k("qsa_union").buf(0, qsa_sel).buf(1, us).buf(2, ul).buf(3, pos).buf(4, qsa_ulist).buf(5, qsa_ucnt).args(6, ua);
            e.groups(S3(n), S3(1024));
            struct { int ratio, max_blocks, ns, ns2, P, D, nu, cap; } ug{cfg.qsa_ratio, max_blocks, kv_ns(), qsa_ns2, qsa_P2,
                                                                        ly.dh, n, cap};
            e.k(kv8 ? "qsa_ugather_kv8" : "qsa_ugather").buf(0, kc[ly.idx]).buf(1, vc[ly.idx]).buf(2, ks[ly.idx])
                .buf(3, vs[ly.idx]).buf(4, qsa_gk).buf(5, qsa_gv).buf(6, qsa_gks).buf(7, qsa_gvs).buf(8, us).buf(9, ul)
                .buf(10, qsa_ulist).buf(11, qsa_ucnt).buf(12, block_table).buf(13, pos).buf(14, tseq).buf(15, qsa_gpos)
                .buf(16, qsa_gflag).buf(17, qsa_gmask).args(18, ug);
            e.threads(S3(ly.dh / 4, tk, ly.hkv * n), S3(64, 4, 1));
            e.k(iq ? "fn_q_pack_i8_g12" : "fn_q_pack_g12").buf(0, qbuf).buf(1, qsa_gq).buf(2, us).buf(3, ul).args(4, q2)
                .buf(5, qsa_gqs);
            e.groups(S3(ly.hkv * 48, n), S3(256));
            e.k(iq ? "attn_fn_g12_kv8_iq_m" : kv8 ? "attn_fn_g12_kv8_m" : "attn_fn_g12_m").buf(0, qsa_gq).buf(1, qsa_gk)
                .buf(2, qsa_gv).buf(3, qsa_gbt).buf(4, qsa_gpos).buf(5, us).buf(6, attn_part).args(7, q2)
                .buf(8, qsa_gflag).buf(9, ul).buf(10, qsa_gks).buf(11, qsa_gvs).buf(12, qsa_gmask).buf(13, qsa_gqs);
            e.groups(S3(ly.hkv, n, splits), S3(256));
          }
        } else if (qsa_nsp && ly.has_qsa && qsa_on && qsa_gmax && on("qsa")) {  // gathered: attn_fn, qsa_gmax rows at a time
          const int ns2 = qsa_ns2;
          GqaArgs q2 = qa;
          q2.max_blocks = qsa_P;
          q2.ns = ns2;
          q2.own_splits = 0;
          q2.chunk = ((cfg.qsa_dense_end() + splits - 1) / splits + 127) / 128 * 128;
          // int8 queries for decode / verify rows only, as on the dense path (prefill chunks: bf16 flash there)
          const bool iq = kv8 && attn_iq && qsa_maxn <= 8;
          for (int g0 = 0; g0 < qsa_nsp; g0 += qsa_gmax) {
            const int n = std::min(qsa_gmax, qsa_nsp - g0);
            const BufRef rws = qsa_rows.at((size_t)g0 * 4);
            struct { int K, ratio, max_blocks, ns, ns2, P, D, nsp; } ga2{cfg.qsa_topk, cfg.qsa_ratio, max_blocks, kv_ns(),
                                                                        ns2, qsa_P, ly.dh, n};
            e.k(kv8 ? "qsa_gather_kv8" : "qsa_gather").buf(0, kc[ly.idx]).buf(1, vc[ly.idx]).buf(2, ks[ly.idx])
                .buf(3, vs[ly.idx]).buf(4, qsa_gk).buf(5, qsa_gv).buf(6, qsa_gks).buf(7, qsa_gvs).buf(8, rws)
                .buf(9, qsa_sel).buf(10, block_table).buf(11, pos).buf(12, tseq).buf(13, qsa_gpos).buf(14, qsa_gflag)
                .buf(15, qsa_gstart).buf(16, qsa_glen).args(17, ga2);
            e.threads(S3(ly.dh / 4, qsa_P * KV_BLOCK, ly.hkv * n), S3(64, 4, 1));
            e.k(iq ? "fn_q_pack_i8_g12" : "fn_q_pack_g12").buf(0, qbuf).buf(1, qsa_gq).buf(2, qsa_gstart)
                .buf(3, qsa_glen).args(4, q2).buf(5, qsa_gqs);
            e.groups(S3(ly.hkv * 48, n), S3(256));
            e.k(iq ? "attn_fn_g12_kv8_iq" : kv8 ? "attn_fn_g12_kv8" : "attn_fn_g12").buf(0, qsa_gq).buf(1, qsa_gk)
                .buf(2, qsa_gv).buf(3, qsa_gbt).buf(4, qsa_gpos).buf(5, qsa_gstart).buf(6, attn_part).args(7, q2)
                .buf(8, qsa_gflag).buf(9, qsa_glen).buf(10, qsa_gks).buf(11, qsa_gvs).buf(12, qsa_gkb).buf(13, qsa_gqs);
            e.groups(S3(ly.hkv, n, splits), S3(256));
          }
        } else if (qsa_nsp && ly.has_qsa && qsa_on) {  // QSA rows (mode 7): their selected groups and tail
          struct { int K, ratio; } aq{cfg.qsa_topk, cfg.qsa_ratio};
          e.k(kv8 ? "attn_qsa_g12_d256_kv8" : "attn_qsa_g12_d256").buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx])
              .buf(3, block_table).buf(4, pos).buf(5, tseq).buf(6, attn_part).args(7, qa).buf(8, qsa_rows)
              .buf(9, ks[ly.idx]).buf(10, vs[ly.idx]).buf(11, qsa_sel).args(12, aq);
          e.groups(S3(ly.hkv, qsa_nsp, splits), S3(128));
        }
        if (n_tree && !(tree_skip & 1)) {
          e.k(kv8 ? "attn_tree_kv8" : "attn_tree").buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx]).buf(3, block_table)
              .buf(4, this->pos).buf(5, tseq).buf(6, tpart).args(7, qa).buf(8, trow).buf(9, tanc).buf(10, ks[ly.idx])
              .buf(11, vs[ly.idx]).buf(12, tpi);
          e.groups(S3(cfg.hq, (T + 3) / 4), S3(128));
          e.k("attn_combine_tree").buf(0, attn_part).buf(1, proj).buf(2, act).args(3, qa).buf(4, rflag).buf(5, shrow)
              .buf(6, tpart).buf(7, trow).buf(8, tpi);
        } else {
          e.k("attn_combine").buf(0, attn_part).buf(1, proj).buf(2, act).args(3, qa).buf(4, rflag).buf(5, shrow);
        }
        e.groups(S3(cfg.hq, T), S3(ly.dh));
        for (auto& f : flash) encode_flash(e, ly, f);
      } else {
        e.k("attn_decode").buf(0, qbuf).buf(1, kc[ly.idx]).buf(2, vc[ly.idx]).buf(3, block_table).buf(4, pos)
            .buf(5, proj).buf(6, act).args(7, aa).buf(8, tseq);
        e.groups(S3(cfg.hq, T), S3(256));
      }
    }
    if (cfg.qwen4) {
      gemm(e, act, ly.out_proj, blk_out, T, 0);
      hyper_inject(e, blk_out, T, &ly.mhc);
      // the hyper-connection after this layer's MoE: the next layer's (unless it starts with the PLE, which reads
      // the streams first), the final mixer after the last layer, none for a layer outside L (the MTP head's)
      moe_next = li < 0 ? nullptr : li + 1 < (int)L.size() ? (L[li + 1].ple ? nullptr : &L[li + 1].ahc) : &mixer;
    } else if (cfg.sandwich()) {
      gemm(e, act, ly.out_proj, blk_out, T, 0);
      norm_add(e, blk_out, ly.post_attn_norm, T, cfg.post_eps);
    } else {
      gemm(e, act, ly.out_proj, x, T, 1, true);
    }
    if (const int aR = ane && li >= 0 && on("ane") ? ane->rows_for(T) : 0) {  // FFN split with the Neural Engine
      ffn_split(e, cb, li, T, aR);
      return;
    }
    if (cfg.experts) {
      encode_moe(e, ly, T);
      return;
    }
    rmsnorm_for(e, x, ly.post_norm, xn, T, H, ly.gate_up);
    gemm(e, xn, ly.gate_up, proj, T, 0);
    ActArgs aca{cfg.inter};
    if (cfg.gemma) {
      e.k("gelu_mul").buf(0, proj).buf(1, act).args(2, aca);
      e.threads(S3(cfg.inter, T), S3(256));
    } else if (on("act") && fuse_a8 && w4a8_route(ly.down, T) && cfg.inter % 64 == 0) {
      e.k("silu_mul_a8").buf(0, proj).buf(1, a8u).buf(2, a8x).args(3, aca);
      e.threads(S3(cfg.inter / 64 * 32, T), S3(256));
      a8_ready = true;
    } else if (on("act")) {
      e.k("silu_mul").buf(0, proj).buf(1, act).args(2, aca);
      e.threads(S3(cfg.inter, T), S3(256));
    }
    const bool captured = std::find(cap_layers.begin(), cap_layers.end(), li) != cap_layers.end();
    if (cfg.sandwich()) {
      gemm(e, act, ly.down, blk_out, T, 0);
      norm_add(e, blk_out, ly.post_ffn_norm, T, cfg.post_eps);
      if (ly.scalar != 1.f) {  // Gemma: the whole residual stream times the layer's scalar
        struct { int n; float s; } sa{H, ly.scalar};
        e.k("scale_rows").buf(0, x).args(1, sa);
        e.threads(S3(H, T), S3(256));
      }
    } else {
      gemm(e, act, ly.down, x, T, 1, !captured && li >= 0 && li + 1 < (int)L.size());
    }
    for (size_t c = 0; c < cap_layers.size(); c++)
      if (cap_layers[c] == li) {
        CapArgs ca{H, (int)cap_layers.size() * H, (int)c * H};
        e.k("capture_cols").buf(0, x).buf(1, hcap).args(2, ca);
        e.threads(S3(H, T), S3(256));
      }
  }

  void encode_step(id<MTLCommandBuffer> __strong& cb, int T, int S) {
    Enc e(g, cb);
    const int H = cfg.hidden;
    EmbedArgs ea{H};
    e.k(embed.bits == 16 ? "embed16" : embed.bits == 8 ? "embed8" : "embed4").buf(0, tok).buf(1, embed.wq).buf(2, embed.s).buf(3, embed.b).buf(4, x).args(5, ea);
    e.threads(S3(H, T), S3(256));
    if (any_ext) {
      e.k("embed_ext").buf(0, ext).buf(1, xin).buf(4, x).args(5, ea);
      e.threads(S3(H, T), S3(256));
    }
    if (cfg.glimmer) rmsnorm(e, x, ones, x, T, H, H, H);  // scaleless norm on the embeddings
    if (cfg.gemma) {  // embeddings x sqrt(hidden), the scale rounded to bf16 as in MLX
      struct { int n; float s; } sa{H, bf16_to_f(f_to_bf16(sqrtf((float)H)))};
      e.k("scale_rows").buf(0, x).args(1, sa);
      e.threads(S3(H, T), S3(256));
    }
    GdnArgs ga{S, cfg.lhk, cfg.lhv, cfg.ldk, cfg.ldv, cfg.conv_dim(), cfg.gdn_proj(), cfg.eps, verify ? 1 : 0, cfg.qwen4};
    if (cfg.qwen4) {  // the residual becomes hc identical streams
      struct { int H, hc; } ta{H, cfg.hc};
      e.k("hc_tile").buf(0, x).buf(1, hx).args(2, ta);
      e.threads(S3(H, T), S3(256));
    }
    int li = 0;
    AttnArgs aa = attn_args(T);
    for (auto& ly : L) {
      if (cb_split && li > 0 && li % cb_split == 0 && !pending.S && !a8_ready && !(cfg.qwen4 && no_split_step)) {
        e.end();
        commit_early(cb);
        e.restart(cb);
      }
      encode_layer(e, cb, ly, li, T, S, aa, ga);
      li++;
    }
    flush_split(e);
    if (cfg.qwen4) {  // the final mixer folds the streams (its norm is the final norm), then the output rows
      hyper_pre(e, mixer, T, false, blk_out);
      struct { int H; } gr{H};
      e.k("hc_gather_rows").buf(0, blk_out).buf(1, out_row).buf(2, xn).args(3, gr);
      e.threads(S3(H, n_out), S3(256));
    } else {
      NormArgs na{H, H, H, cfg.eps};
      e.k("rmsnorm_gather").buf(0, x).buf(1, final_norm).buf(2, xn).args(3, na).buf(4, out_row);
      e.groups(S3(n_out), S3(1024));
    }
    gemm(e, xn, lm_head, logits, n_out, 0);
    if (cfg.softcap > 0) {  // Glimmer: logits * output_multiplier, then cap * tanh(. / cap)
      struct { int n; float mult, cap; } sa{cfg.vocab, cfg.out_mult, cfg.softcap};
      e.k("logit_softcap").buf(0, logits).args(1, sa);
      e.threads(S3(cfg.vocab, n_out), S3(256));
    }
    if (verify) {
      // Argmax every output row, accept greedily, then replay each GDN layer over the committed rows.
      e.k("row_argmax").buf(0, logits).buf(1, pred).args(2, cfg.vocab);
      e.groups(S3(n_out), S3(1024));
      if (any_sample) {  // speculative sampling: each verify row's top-16, then accept/resample per sequence
        struct { int V, SH, PS, P0, LO; } ta{cfg.vocab, kVTopShards, kVTopShards, 0, 0};
        e.k("row_top16_shard").buf(0, logits).buf(1, vcpart).buf(2, vvpart).args(3, ta);
        e.groups(S3(kVTopShards, n_out), S3(256));
        e.k("top16_merge").buf(0, vcpart).buf(1, vvpart).buf(2, vcand).buf(3, vun).args(4, ta);
        e.groups(S3(n_out), S3(256));
        e.k("spec_accept_s").buf(0, tok).buf(1, pred).buf(2, seq_start).buf(3, seq_len).buf(4, out_base).buf(5, spec)
            .buf(6, commit).buf(7, emit).args(8, S).buf(9, samp).buf(10, vcand).buf(11, vun);
      } else {
        e.k("spec_accept").buf(0, tok).buf(1, pred).buf(2, seq_start).buf(3, seq_len).buf(4, out_base).buf(5, spec)
            .buf(6, commit).buf(7, emit).args(8, S);
      }
      e.threads(S3(S), S3(64));
      if (n_tree) {
        e.k("tree_accept").buf(0, tok).buf(1, pred).buf(2, seq_start).buf(3, seq_len).buf(4, out_base).buf(5, commit)
            .buf(6, emit).buf(7, tpath).buf(8, tseqs).buf(9, tpar).args(10, n_tree);
        e.threads(S3(n_tree), S3(32));
      }
      const BufRef& spec = n_tree ? spec_g : this->spec;  // tree sequences replay along their path (below)
      ReplayArgs ra{cfg.conv_dim(), cfg.lhk, cfg.lhv, cfg.ldk, cfg.ldv};
      if (ple_tape.buf) {
        struct { int C, K, D, verify; } pc{cfg.hc * H, 4, 3, 1};
        e.k("ple_conv_replay").buf(0, ple_tape).buf(1, ple_state).buf(2, slot).buf(3, seq_start).buf(4, commit)
            .buf(5, spec).args(6, pc);
        e.threads(S3(pc.C, S), S3(256));
      }
      for (auto& ly : L) {
        if (!ly.linear || tree_only) continue;
        e.k("gdn_conv_replay").buf(0, tape_x[ly.idx]).buf(1, conv_state[ly.idx]).buf(2, slot).buf(3, seq_start)
            .buf(4, commit).buf(5, spec).args(6, ra);
        e.threads(S3(cfg.conv_dim(), S), S3(256));
        e.k("gdn_replay").buf(0, tape_c[ly.idx]).buf(1, tape_g[ly.idx]).buf(2, tape_b[ly.idx]).buf(3, ssm_state[ly.idx])
            .buf(4, slot).buf(5, seq_start).buf(6, commit).buf(7, spec).args(8, ra);
        e.threads(S3(32, cfg.ldv, S * cfg.lhv), S3(32, 4, 1));
      }
      if (n_tree) encode_tree_commit(e);
    }
    e.end();
  }

  // ---- token trees (TREE) ----
  // Per-row tree inputs for this step's tree sequences (w[b].tree): chain defaults until set_tree() installs a tree.
  // tpi[t]: the row's slot in tpart (its logits row, out_base + node index).
  BufRef tpi;
  void tree_batch(const std::vector<SeqWork>& w, int T) {
    n_tree = 0;
    for (auto& x : w) n_tree += x.tree;
    tree_only = n_tree && n_tree == (int)w.size();
    if (!n_tree) return;
    if (!tpi.buf) tpi = g.alloc((size_t)max_tokens * 4);
    memcpy(apos.ptr<int>(), pos.ptr<int>(), (size_t)T * 4);
    std::fill(trow.ptr<int>(), trow.ptr<int>() + T, -1);
    int k = 0;
    for (size_t b = 0; b < w.size(); b++) {
      seq_len_g.ptr<int>()[b] = w[b].tree ? 0 : w[b].n;
      spec_g.ptr<int>()[b] = w[b].tree ? 0 : (int)w[b].spec;
      if (!w[b].tree) continue;
      tseqs.ptr<int>()[k++] = (int)b;
      const int t0 = seq_start.ptr<int>()[b], ob = out_base.ptr<int>()[b];
      for (int i = 0; i < w[b].n; i++) {
        apos.ptr<int>()[t0 + i] = w[b].pos0 - 1;
        trow.ptr<int>()[t0 + i] = i;
        tpar.ptr<int>()[t0 + i] = i - 1;
        tdep.ptr<int>()[t0 + i] = i;
        tanc.ptr<uint32_t>()[t0 + i] = (2u << i) - 1;
        tpi.ptr<int>()[t0 + i] = ob + i;
      }
    }
  }
  // Install a tree for batch sequence b (its rows must be a tree SeqWork of t.n rows): tokens after the root, rope
  // positions pos0 + depth, parents, depths, ancestor masks. Host writes: call before the verify command buffer runs.
  void set_tree(int b, int pos0, const Tree& t) {
    const int t0 = seq_start.ptr<int>()[b], rows = seq_len.ptr<int>()[b];
    for (int i = t.n; i < rows; i++) {  // padding (a tree smaller than its rows): a duplicate of the root's first child,
      tok.ptr<int>()[t0 + i] = t.n > 1 ? t.tok[1] : t.tok[0];  // which tree_accept never picks over the original
      for (int c = 0; c < 3; c++) rpos.ptr<int>()[(t0 + i) * 3 + c] = pos0 + 1;
      tpar.ptr<int>()[t0 + i] = 0;
      tdep.ptr<int>()[t0 + i] = 1 | 0x200;
      tanc.ptr<uint32_t>()[t0 + i] = 1u | (1u << i);
    }
    for (int i = 0; i < t.n; i++) {
      if (i) tok.ptr<int>()[t0 + i] = t.tok[i];
      for (int c = 0; c < 3; c++) rpos.ptr<int>()[(t0 + i) * 3 + c] = pos0 + t.depth[i];
      tpar.ptr<int>()[t0 + i] = t.par[i];
      // depth | 0x100 when the node has 2+ children | 0x200 when its parent is not the previous row (the GDN kernels
      // rebuild the parent's conv window and state there)
      int kids = 0;
      for (int j = i + 1; j < t.n; j++) kids += t.par[j] == i;
      tdep.ptr<int>()[t0 + i] = t.depth[i] | (kids >= 2 ? 0x100 : 0) | (i > 0 && t.par[i] != i - 1 ? 0x200 : 0);
      tanc.ptr<uint32_t>()[t0 + i] = t.anc[i];
    }
  }
  // After tree_accept: replay the GDN layers along each accepted path, and move the path's KV into place.
  void encode_tree_commit(Enc& e) {
    ReplayArgs ra{cfg.conv_dim(), cfg.lhk, cfg.lhv, cfg.ldk, cfg.ldv};
    struct { int Dh, ns, max_blocks; } ca{0, kv_ns(), max_blocks};
    for (auto& ly : L) {
      if (ly.linear && (tree_skip & 8)) continue;
      if (!ly.linear && (tree_skip & 16)) continue;
      if (ly.linear) {
        e.k("gdn_conv_replay_tree").buf(0, tape_x[ly.idx]).buf(1, conv_state[ly.idx]).buf(2, slot).buf(3, seq_start)
            .buf(4, commit).buf(5, tseqs).args(6, ra).buf(7, tpath);
        e.threads(S3(cfg.conv_dim(), n_tree), S3(256));
        e.k("gdn_replay_tree").buf(0, tape_c[ly.idx]).buf(1, tape_g[ly.idx]).buf(2, tape_b[ly.idx])
            .buf(3, ssm_state[ly.idx]).buf(4, slot).buf(5, seq_start).buf(6, commit).buf(7, tseqs).args(8, ra)
            .buf(9, tpath);
        e.threads(S3(32, cfg.ldv, n_tree * cfg.lhv), S3(32, 4, 1));
      } else {
        ca.Dh = ly.dh;
        e.k(kv8 ? "kv_compact_kv8" : "kv_compact").buf(0, kc[ly.idx]).buf(1, vc[ly.idx]).buf(2, ks[ly.idx])
            .buf(3, vs[ly.idx]).buf(4, block_table).buf(5, pos).buf(6, seq_start).buf(7, commit).buf(8, tpath)
            .buf(9, tseqs).args(10, ca);
        e.threads(S3(ly.dh, ly.hkv, n_tree), S3(std::min(ly.dh, 256)));
      }
    }
  }
};
