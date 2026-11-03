// Qwen3.8-Flash-Next (qwen4_exp) host pieces: mixed-precision linear loading, the n-gram ids of the per-layer
// embedding (PLE), and its table. Kernels: kernels/qwen4.metal.
#pragma once
#include <cmath>
#include "weights.h"

// MLX quantization of one tensor prefix: the per-tensor override in config.json's "quantization", else the default.
struct QSpec { int bits = 4, gs = 64; };
// Widths 2-8 bits and groups that are a positive multiple of 32 (Codex review 2026-11-03, R01: bits 0 or group 0
// divided by zero, and the loaders sized buffers from them).
inline QSpec qspec(const Checkpoint& ck, const std::string& prefix) {
  const json::Value& q = ck.config["quantization"];
  QSpec s;
  try {
    if (q.has("bits")) s.bits = q["bits"].i();
    if (q.has("group_size")) s.gs = q["group_size"].i();
    if (q.has(prefix) && q[prefix].type == json::Value::Obj) {
      s.bits = q[prefix]["bits"].i();
      s.gs = q[prefix]["group_size"].i();
    }
  } catch (std::exception& e) {
    ck.reject(prefix, std::string("quantization config: ") + e.what());
  }
  if (s.bits < 2 || s.bits > 8 || s.gs < 32 || s.gs > (1 << 16) || s.gs % 32)
    ck.reject(prefix, "unsupported quantization: " + std::to_string(s.bits) + " bits, group " + std::to_string(s.gs));
  return s;
}

// A quantized part [n, K] at spec sp: U32 weight [n, K * bits / 32], BF16 scales and biases [n, K / gs]. Returns n and
// sets K, a multiple of kmul (R01: the mixed-width loaders copied scales sized by the tensor into vectors sized by the
// config).
inline int check_qpart(const Checkpoint& ck, const std::string& p, const QSpec& sp, int* K, int kmul = 64) {
  ck.need_dtype(p + ".weight", "U32");
  ck.need_dtype(p + ".scales", "BF16");
  ck.need_dtype(p + ".biases", "BF16");
  const int n = ck.dim(p + ".weight", 2, 0), wk = ck.dim(p + ".weight", 2, 1, 1, (1 << 30) / 32);
  if ((int64_t)wk * 32 % sp.bits) ck.reject(p + ".weight", "row is not a whole number of " + std::to_string(sp.bits) + "-bit values");
  const int k = (int)((int64_t)wk * 32 / sp.bits);
  if (k % kmul || k % sp.gs)
    ck.reject(p + ".weight", "K " + std::to_string(k) + " is not a multiple of " + std::to_string(kmul) + " and the group size");
  if (ck.dim(p + ".scales", 2, 0) != n || ck.dim(p + ".scales", 2, 1) != k / sp.gs || ck.info(p + ".biases").shape != ck.info(p + ".scales").shape)
    ck.reject(p + ".scales", "scales/biases are not [" + std::to_string(n) + ", " + std::to_string(k / sp.gs) + "]");
  *K = k;
  return n;
}

// Linears [N_i, K] stacked along N from tensors of any MLX affine width (2-8 bits, packed as a little-endian bit
// stream) and group size (a multiple of 32). Already 4-bit / group 64 everywhere: the usual loader. Otherwise every
// part is widened to 8-bit / group 64 with the same scale and bias per 64-group, which is exact (the quantized values
// fit in 8 bits and a group of 128 or 32 maps onto 64-groups with shared or split scales; group 32 must not straddle,
// so it is only accepted for 4-bit -> kept as is by the caller). pad_n: zero rows appended up to this N.
inline QLinear load_qlinear_any(Gpu& g, const Checkpoint& ck, const std::vector<std::string>& prefixes, int pad_n = 0,
                                bool force8 = false) {
  std::vector<QSpec> sp;
  bool plain = !force8 && pad_n == 0;
  for (auto& p : prefixes) {
    sp.push_back(qspec(ck, p));
    if (sp.back().bits != 4 || sp.back().gs != 64) plain = false;
  }
  if (plain) return load_qlinear(g, ck, prefixes);
  // Target width: 8 bits by default (W8A8 serves batched rows). Q4_NATIVE56=1 keeps the widest part's native 5 or 6
  // bits (qmvb GEMVs, MLX qmm); narrower parts are re-encoded at that width (exact). Studio, Flash-Next 2k, decode
  // step B=1 18.92 -> 18.34 ms but B=16 36.44 -> 37.05 ms (2026-11-01), so native is opt-in.
  static const bool widen8 = !(getenv("Q4_NATIVE56") && atoi(getenv("Q4_NATIVE56")));
  int tb = 4;
  for (auto& x : sp) tb = std::max(tb, x.bits);
  if (force8 || widen8 || tb > 6) tb = 8;
  QLinear q;
  q.bits = tb;
  std::vector<int> ns;
  for (size_t i = 0; i < prefixes.size(); i++) {
    int K = 0;
    const int n = check_qpart(ck, prefixes[i], sp[i], &K);
    if (q.K && q.K != K) { fprintf(stderr, "%s: fused K mismatch\n", prefixes[i].c_str()); exit(1); }
    if (sp[i].gs % 64 && sp[i].gs != 32) { fprintf(stderr, "%s: group %d\n", prefixes[i].c_str(), sp[i].gs); exit(1); }
    if (q.N > (1 << 30) - n) ck.reject(prefixes[i] + ".weight", "fused rows overflow");
    q.K = K;
    ns.push_back(n);
    q.N += n;
  }
  if (pad_n < 0 || pad_n > (1 << 30)) { fprintf(stderr, "%s: bad padding %d\n", prefixes[0].c_str(), pad_n); exit(1); }
  const int N = std::max(q.N, pad_n), K = q.K, KG = K / 64;
  q.N = N;
  const size_t drow = (size_t)K * tb / 8;  // packed bytes per destination row
  id<MTLBuffer> buf = g.alloc((size_t)N * drow + (size_t)N * KG * 4);
  memset(buf.contents, 0, buf.length);
  q.wq = BufRef(buf, 0);
  q.s = BufRef(buf, (size_t)N * drow);
  q.b = BufRef(buf, (size_t)N * drow + (size_t)N * KG * 2);
  uint8_t* W = q.wq.ptr<uint8_t>();
  uint16_t *S = q.s.ptr<uint16_t>(), *B = q.b.ptr<uint16_t>();
  int row = 0;
  for (size_t i = 0; i < prefixes.size(); i++) {
    const int bits = sp[i].bits, gs = sp[i].gs, n = ns[i], G = K / gs;
    auto& wi = ck.info(prefixes[i] + ".weight");
    std::vector<uint8_t> wr(wi.nbytes);
    std::vector<uint16_t> sr((size_t)n * G), br((size_t)n * G);
    ck.read_into(wi, wr.data(), wr.size());
    ck.read_into(ck.info(prefixes[i] + ".scales"), sr.data(), sr.size() * 2);
    ck.read_into(ck.info(prefixes[i] + ".biases"), br.data(), br.size() * 2);
    const size_t rb = wi.nbytes / n;  // bytes per packed row (n >= 1: check_qpart)
    if (gs == 32) {  // two 32-groups in one 64-group: only exact when their scale and bias agree
      for (int r = 0; r < n; r++)
        for (int j = 0; j < G; j += 2)
          if (sr[(size_t)r * G + j] != sr[(size_t)r * G + j + 1] || br[(size_t)r * G + j] != br[(size_t)r * G + j + 1]) {
            fprintf(stderr, "%s: group 32 not representable at group 64\n", prefixes[i].c_str());
            exit(1);
          }
    }
    const uint8_t* wp = wr.data();
    dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
      const uint8_t* src = wp + r * rb;
      uint8_t* dst = W + (size_t)(row + r) * drow;
      for (int k = 0; k < K; k++) {
        const size_t bit = (size_t)k * bits;
        uint32_t v = 0;
        for (int b = 0; b < bits; b++) v |= ((src[(bit + b) >> 3] >> ((bit + b) & 7)) & 1u) << b;
        if (tb == 8) {
          dst[k] = (uint8_t)v;
        } else {  // little-endian bit stream at tb bits (the destination row was zeroed)
          const size_t ob = (size_t)k * tb;
          for (int b = 0; b < tb; b++)
            if ((v >> b) & 1u) dst[(ob + b) >> 3] |= (uint8_t)(1u << ((ob + b) & 7));
        }
      }
    });
    for (int r = 0; r < n; r++)
      for (int j = 0; j < KG; j++) {
        const int src_g = j * 64 / gs;
        S[(size_t)(row + r) * KG + j] = sr[(size_t)r * G + src_g];
        B[(size_t)(row + r) * KG + j] = br[(size_t)r * G + src_g];
      }
    row += n;
  }
  return q;
}

// Dense-model linears (2026-11-02): bf16, or every part one width of 4 or 8 bits at group 64, go to the usual loader
// unchanged; anything else (lmstudio 5/6-bit, the Jundot oQ*e mixes with 5-bit down/k projections) is widened
// exactly by load_qlinear_any instead of stopping at "unsupported 5-bit weights".
inline QLinear load_qlinear_w(Gpu& g, const Checkpoint& ck, const std::vector<std::string>& prefixes) {
  if (ck.info(prefixes[0] + ".weight").dtype == "BF16") return load_qlinear(g, ck, prefixes);
  int b0 = 0;
  for (auto& p : prefixes) {
    const QSpec q = qspec(ck, p);
    if (q.gs != 64 || (q.bits != 4 && q.bits != 8) || (b0 && q.bits != b0)) return load_qlinear_any(g, ck, prefixes);
    b0 = q.bits;
  }
  return load_qlinear(g, ck, prefixes);
}

// bf16 copy of a dequantized (or plain bf16) [N, K] tensor, rows appended to out.
inline void append_bf16_rows(const Checkpoint& ck, const std::string& prefix, std::vector<uint16_t>& out, int K) {
  auto& w = ck.info(prefix + ".weight");
  if (w.dtype == "BF16") {  // [n, K] exactly: the router reads K-wide rows (R01)
    const int n = ck.dim(prefix + ".weight", 2, 0);
    if (ck.dim(prefix + ".weight", 2, 1) != K) ck.reject(prefix + ".weight", "expected rows of " + std::to_string(K));
    const size_t at = out.size();
    out.resize(at + (size_t)n * K);
    ck.read_into(w, out.data() + at, (size_t)n * K * 2);
    return;
  }
  const QSpec s = qspec(ck, prefix);
  int k = 0;
  const int n = check_qpart(ck, prefix, s, &k), G = K / s.gs;
  if (k != K) ck.reject(prefix + ".weight", "expected rows of " + std::to_string(K) + ", got " + std::to_string(k));
  std::vector<uint8_t> wr(w.nbytes);
  std::vector<uint16_t> sr((size_t)n * G), br((size_t)n * G);
  ck.read_into(w, wr.data(), wr.size());
  ck.read_into(ck.info(prefix + ".scales"), sr.data(), sr.size() * 2);
  ck.read_into(ck.info(prefix + ".biases"), br.data(), br.size() * 2);
  const size_t rb = w.nbytes / n;
  for (int r = 0; r < n; r++)
    for (int k = 0; k < K; k++) {
      const size_t bit = (size_t)k * s.bits;
      uint32_t v = 0;
      for (int b = 0; b < s.bits; b++) v |= ((wr[r * rb + ((bit + b) >> 3)] >> ((bit + b) & 7)) & 1u) << b;
      const float x = v * bf16_to_f(sr[(size_t)r * G + k / s.gs]) + bf16_to_f(br[(size_t)r * G + k / s.gs]);
      out.push_back(f_to_bf16(x));
    }
}

// The n-gram ids of one PLE layer: 2-grams and 3-grams of the current token and its predecessors (positions before
// the sequence start, and any position at or before the latest EOS ahead of the token, read as EOS), hashed with
// the layer's multipliers into heads_per_ngram heads each, each head a prime-sized slice of one table.
struct NGram {
  int ngram = 3, heads_per = 8, heads = 16, eos = 0;
  int64_t mult[3] = {0, 0, 0};
  std::vector<int64_t> size, offset;
  static bool prime(int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2)
      if (v % d == 0) return false;
    return true;
  }
  void init(const json::Value& c, const Checkpoint& ck, const std::string& mult_name, int ple_index) {
    int64_t base = 0;
    try {
      ngram = c.has("ngram_size") ? c["ngram_size"].i() : 3;
      heads_per = c.has("heads_per_ngram") ? c["heads_per_ngram"].i() : 8;
      const json::Value& e = c["eos_token_id"];
      eos = e.type == json::Value::Arr ? e[0].i() : e.i();
      base = c.has("ngram_vocab_size_base") ? c["ngram_vocab_size_base"].i64() : 20000000;
    } catch (std::exception& ex) {
      ck.reject(mult_name, std::string("n-gram config: ") + ex.what());
    }
    // ids() keeps ngram context tokens in a fixed array of 3 and hashes with mult[0..ngram); the table's sizes are
    // primes above base (R01: the multipliers were read into mult[3] at whatever size the tensor had).
    if (ngram < 2 || ngram > 3 || heads_per < 1 || heads_per > 64 || base < 2 || base > (1ll << 40) || ple_index < 0 ||
        ple_index > 1024)
      ck.reject(mult_name, "unsupported n-gram config (ngram_size 2-3, heads_per_ngram 1-64)");
    heads = (ngram - 1) * heads_per;
    {
      auto& ti = ck.info(mult_name);
      if (ti.dtype != "I64" || ti.nbytes != (size_t)ngram * 8)
        ck.reject(mult_name, "expected " + std::to_string(ngram) + " I64 multipliers, got " + ti.dtype + " of " +
                                 std::to_string(ti.nbytes) + " bytes");
    }
    int64_t p = base - 1, total = 0;
    for (int h = 0; h < (ple_index + 1) * heads; h++) {  // the (global head + 1)-th prime after base - 1
      do p++; while (!prime(p));
      if (h >= ple_index * heads) {
        size.push_back(p);
        offset.push_back(total);
        total += p;
      }
    }
    ck.read_into(ck.info(mult_name), mult, sizeof mult);
  }
  // The GPU kernel's table (ple_ids_gpu): int64 [mult 0..2, size[heads], offset[heads]].
  std::vector<int64_t> table() const {
    std::vector<int64_t> t(mult, mult + 3);
    t.insert(t.end(), size.begin(), size.end());
    t.insert(t.end(), offset.begin(), offset.end());
    return t;
  }
  // ids [n * heads] for tokens tok[0..n) at positions pos0.., tok[-1], tok[-2] readable when pos0 >= 1, 2.
  void ids(const int* tok, int pos0, int n, int* out) const {
    for (int i = 0; i < n; i++) {
      int64_t ctx[3];
      ctx[0] = tok[i];
      bool cut = false;  // an EOS between the token and the context position
      for (int k = 1; k < ngram; k++) {
        const int p = pos0 + i - k;
        const int t = p >= 0 ? tok[i - k] : eos;
        cut = cut || p < 0 || t == eos;
        ctx[k] = cut ? eos : t;
      }
      for (int g = 2, h = 0; g <= ngram; g++) {
        uint64_t m = (uint64_t)ctx[0] * (uint64_t)mult[0];
        for (int k = 1; k < g; k++) m ^= (uint64_t)ctx[k] * (uint64_t)mult[k];
        const int64_t mixed = (int64_t)m;
        for (int j = 0; j < heads_per; j++, h++) {
          int64_t r = mixed % size[h];
          if (r < 0) r += size[h];
          out[i * heads + h] = (int)(r + offset[h]);
        }
      }
    }
  }
};

// The PLE n-gram table: shards (MLX 4-bit, group 32, row width W) concatenated into 4 row ranges.
struct PleTable {
  BufRef w[4], s[4], b[4];
  int rows_per_part = 0, W = 0;
  void load(Gpu& g, const Checkpoint& ck, const std::string& prefix) {  // prefix ...ngram_embedding.shards.
    std::vector<std::string> names;
    for (int i = 0; ck.has(prefix + std::to_string(i) + ".weight"); i++) names.push_back(prefix + std::to_string(i));
    const int nsh = (int)names.size();
    if (nsh == 0 || nsh % 4) { fprintf(stderr, "PLE: %d shards\n", nsh); exit(1); }
    int rows = 0;
    for (int i = 0; i < nsh; i++) {  // every shard 4-bit g32 [rows, W] (R01: only shard 0's spec, row count checked)
      const QSpec sp = qspec(ck, names[i]);
      if (sp.bits != 4 || sp.gs != 32) { fprintf(stderr, "PLE: expected 4-bit g32\n"); exit(1); }
      int k = 0;
      const int r = check_qpart(ck, names[i], sp, &k, 32);
      if (i == 0) rows = r, W = k;
      if (r != rows || k != W) { fprintf(stderr, "PLE: uneven shards\n"); exit(1); }
    }
    if ((int64_t)rows * (nsh / 4) > (1 << 30)) { fprintf(stderr, "PLE: table too large\n"); exit(1); }
    rows_per_part = rows * (nsh / 4);
    const size_t wrow = (size_t)W / 2, srow = (size_t)W / 32 * 2;
    for (int part = 0; part < 4; part++) {
      w[part] = g.alloc((size_t)rows_per_part * wrow);
      s[part] = g.alloc((size_t)rows_per_part * srow);
      b[part] = g.alloc((size_t)rows_per_part * srow);
      for (int j = 0; j < nsh / 4; j++) {
        const std::string& n = names[part * (nsh / 4) + j];
        ck.read_into(ck.info(n + ".weight"), w[part].ptr<char>() + (size_t)j * rows * wrow, (size_t)rows * wrow);
        ck.read_into(ck.info(n + ".scales"), s[part].ptr<char>() + (size_t)j * rows * srow, (size_t)rows * srow);
        ck.read_into(ck.info(n + ".biases"), b[part].ptr<char>() + (size_t)j * rows * srow, (size_t)rows * srow);
      }
    }
  }
};
