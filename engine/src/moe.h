// Mixture-of-experts weights (Qwen3.5/3.6 MoE, MLX layout): the routed experts' gate/up fused per expert
// ([E][2I][K]) and down ([E][H][I]), plus the router with the shared expert's gate as one extra row.
// Kernels: kernels/moe.metal. These stay unflipped (W4A8 never sees them).
#pragma once
#include "weights.h"
#include "qwen4.h"  // qspec: the MLX group size from config.json

struct MoeW {  // one stacked affine matrix [E, N, K]
  BufRef wq, s, b;
  int E = 0, N = 0, K = 0, bits = 0;
};

// Stack tensors that share E and K along N per expert: out[e] = [t0[e]; t1[e]; ...]. 2-D tensors count as E = 1.
inline MoeW load_moe(Gpu& g, const Checkpoint& ck, const std::vector<std::string>& prefixes) {
  MoeW m;
  std::vector<int> ns;
  int gs0 = 0;  // every part's group size (one stack, one expansion)
  for (auto& p : prefixes) {
    auto& w = ck.info(p + ".weight");
    auto& s = ck.info(p + ".scales");
    // Codex review 2026-11-03, R01: ranks, dtypes, the width and scales/biases against the weight are checked before
    // anything is sized from them (the shapes were indexed blindly and rows copied at the inferred sizes).
    const bool three = w.shape.size() == 3;
    const int r = three ? 3 : 2;
    ck.need_dtype(p + ".weight", "U32");
    ck.need_dtype(p + ".scales", "BF16");
    ck.need_dtype(p + ".biases", "BF16");
    const int E = three ? ck.dim(p + ".weight", 3, 0, 1, 1 << 16) : 1, N = ck.dim(p + ".weight", r, r - 2);
    // Group size from config.json (2026-11-04): the official Qwen MLX 4-bit (and 8-bit) 35B-A3B quantizes the experts
    // at group 128, whose shapes are identical to 8-bit / group 64 at half the K; inferred from the shapes alone, they
    // loaded as K/2 8-bit and the model computed garbage (cos -0.3 vs mlx_lm, top-1 wrong on every sequence). A group
    // of 128 becomes two 64-groups with the same scale and bias (exact; the 4-bit codes are kept).
    const int gs = qspec(ck, p).gs;
    if (gs % 64) ck.reject(p + ".scales", "MoE group size " + std::to_string(gs) + " is not a multiple of 64");
    const int wk = ck.dim(p + ".weight", r, r - 1, 1, (1 << 30) / 32), G = ck.dim(p + ".scales", r, r - 1, 1, (1 << 30) / gs);
    const int K = G * gs, bits = (int)((int64_t)wk * 32 / K);
    if ((bits != 4 && bits != 8) || (int64_t)wk * 32 != (int64_t)K * bits) ck.reject(p + ".weight", "unsupported width");
    if (ck.dim(p + ".scales", r, r - 2) != N || (three && ck.dim(p + ".scales", 3, 0) != E) ||
        ck.info(p + ".biases").shape != s.shape)
      ck.reject(p + ".scales", "scales/biases do not match the weight");
    if ((m.E && m.E != E) || (m.K && (m.K != K || m.bits != bits))) {
      fprintf(stderr, "%s: MoE stack mismatch\n", p.c_str());
      exit(1);
    }
    if ((int64_t)m.N + N > (1 << 30) || (int64_t)E * (m.N + N) > (1ll << 34)) ck.reject(p + ".weight", "stack too large");
    if (gs0 && gs0 != gs) ck.reject(p + ".scales", "MoE stack mixes group sizes");
    gs0 = gs;
    m.E = E, m.K = K, m.bits = bits;
    m.N += N;
    ns.push_back(N);
  }
  const size_t wrow = (size_t)m.K * m.bits / 8, srow = (size_t)m.K / 64 * 2;  // bytes per row
  const size_t wq_bytes = (size_t)m.E * m.N * wrow, s_bytes = (size_t)m.E * m.N * srow;
  id<MTLBuffer> buf = g.alloc(wq_bytes + 2 * s_bytes);
  m.wq = BufRef(buf, 0);
  m.s = BufRef(buf, wq_bytes);
  m.b = BufRef(buf, wq_bytes + s_bytes);
  int n0 = 0;
  for (size_t i = 0; i < prefixes.size(); i++) {
    const int N = ns[i];
    struct Part { const char* sfx; char* dst; size_t rb; } parts[3] = {
        {".weight", m.wq.ptr<char>(), wrow}, {".scales", m.s.ptr<char>(), srow}, {".biases", m.b.ptr<char>(), srow}};
    const int f = gs0 / 64;  // 64-groups per source group (scales and biases are repeated f times)
    for (auto& pt : parts) {
      auto& ti = ck.info(prefixes[i] + pt.sfx);
      const bool sb = pt.dst != m.wq.ptr<char>();
      const size_t sr = sb ? pt.rb / f : pt.rb;  // source bytes per row
      std::vector<char> tmp((size_t)m.E * N * sr);  // exactly E x N rows of sr bytes (shapes checked above)
      ck.read_into(ti, tmp.data(), tmp.size());
      if (f == 1 || !sb) {
        for (int e = 0; e < m.E; e++)
          memcpy(pt.dst + ((size_t)e * m.N + n0) * pt.rb, tmp.data() + (size_t)e * N * pt.rb, (size_t)N * pt.rb);
        continue;
      }
      const int Gs = (int)(sr / 2);
      for (int e = 0; e < m.E; e++)
        for (int n = 0; n < N; n++) {
          const uint16_t* src = (const uint16_t*)(tmp.data() + ((size_t)e * N + n) * sr);
          uint16_t* dst = (uint16_t*)(pt.dst + ((size_t)e * m.N + n0 + n) * pt.rb);
          for (int j = 0; j < Gs; j++)
            for (int k = 0; k < f; k++) dst[j * f + k] = src[j];
        }
    }
    n0 += N;
  }
  return m;
}
