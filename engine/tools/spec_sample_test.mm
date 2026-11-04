// spec_sample_test: the GPU speculative sampler against the one sampling contract (src/sampling.h; Codex review
// 2026-11-04, R03). No model: random bf16 logits rows of the 27B's vocabulary.
//   1. row_top{16,64}_shard_v + top{16,64}_merge_v give exactly the contract's top K of each row (logit descending,
//      ties by token id: bf16 logits tie often, and a different tie order changed the support at the K-th place), and
//      verify_zpart's softmax normaliser matches a double-precision sum.
//   2. spec_accept_s's first emitted token, over many seeds with one draft, follows the contract's distribution p for
//      temperature / top_k / top_p settings (top_k 16, 20, 64; top_k off with top_p cuts), whether the draft is a
//      likely token or not (chi-square); a row whose nucleus does not fit in K is handed to the host (host[b] = 0).
//   build/spec_sample_test        (from engine/, after ./build.sh)
#include <random>
#include "../src/gpu.h"
#include "../src/sampling.h"

static uint16_t tobf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16); }
struct Top16Args { int V, SH, PS, P0, LO; };
struct SampParams { float t, top_p; int top_k; uint32_t seed; };
constexpr int V = 248320, SH = 64, MAXB = 16;
static int fails = 0, runs = 0;
static void check(bool ok, const char* what) { runs++; if (!ok) { fails++; printf("FAIL %s\n", what); } }

// The contract's distribution over a row (double): top_k, then top_p after top_k, at temperature t.
static std::vector<double> contract(const std::vector<float>& l, float t, float top_p, int top_k) {
  std::vector<std::pair<float, int>> c;
  for (int i = 0; i < V; i++) c.push_back({l[i], i});
  std::sort(c.begin(), c.end(), sampling::desc);
  if (top_k > 0) c.resize(top_k);
  std::vector<double> w(c.size());
  double Z = 0;
  for (size_t i = 0; i < c.size(); i++) Z += w[i] = std::exp((double)(c[i].first - c[0].first) / t);
  double acc = 0;
  size_t n = 0;
  while (n < c.size()) { acc += w[n++]; if (acc >= top_p * Z) break; }
  std::vector<double> p(V, 0);
  for (size_t i = 0; i < n; i++) p[c[i].second] = w[i] / acc;
  return p;
}

int main() {
  @autoreleasepool {
    Gpu g("build/core.metallib");
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    // Rows: 0 broad (many plausible tokens), 1 peaked, 2 flat head of 30 equal logits, 3 very broad (nucleus > 64).
    const int R = 4;
    std::vector<std::vector<float>> rows(R, std::vector<float>(V));
    for (int r = 0; r < R; r++)
      for (int i = 0; i < V; i++) {
        float v = nd(rng) * 2.0f;
        if (r == 0 && i % 997 == 0) v += 9.0f - 0.01f * (i / 997);
        if (r == 1 && i < 40) v = 18.0f - 0.6f * i;
        if (r == 2) v = i < 30 ? 12.0f : nd(rng) * 1.5f;
        if (r == 3) v = nd(rng) * 0.3f;
        rows[r][i] = sampling::bf16(tobf(v));
      }
    BufRef logits = g.alloc((size_t)R * V * 2);
    for (int r = 0; r < R; r++)
      for (int i = 0; i < V; i++) logits.ptr<uint16_t>()[(size_t)r * V + i] = tobf(rows[r][i]);

    for (int K : {16, 64}) {  // 1. exact top K and the normaliser
      BufRef cp = g.alloc((size_t)R * SH * K * 4), vp = g.alloc((size_t)R * SH * K * 4), cand = g.alloc(R * K * 4),
             un = g.alloc(R * K * 4), zp = g.alloc((size_t)R * SH * 8), samp = g.alloc(R * 16), ob = g.alloc(R * 4),
             spec = g.alloc(R * 4);
      for (int r = 0; r < R; r++) {
        SampParams sp{0.8f, 0.9f, 0, 1};
        memcpy(samp.ptr<char>() + r * 16, &sp, 16);
        ob.ptr<int>()[r] = r;
        spec.ptr<int>()[r] = 1;
      }
      Top16Args ta{V, SH, SH, 0, 0};
      int S = R;
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      Enc e(g, cb);
      char a[32], b[32];
      snprintf(a, sizeof a, "row_top%d_shard_v", K);
      snprintf(b, sizeof b, "top%d_merge_v", K);
      e.k(a).buf(0, logits).buf(1, cp).buf(2, vp).args(3, ta);
      e.groups(S3(SH, R), S3(256));
      e.k(b).buf(0, cp).buf(1, vp).buf(2, cand).buf(3, un).args(4, ta);
      e.groups(S3(R), S3(256));
      e.k("verify_zpart").buf(0, logits).buf(1, zp).args(2, ta).buf(3, samp).buf(4, ob).args(5, S).buf(6, spec);
      e.groups(S3(SH, R), S3(256));
      e.end();
      [cb commit];
      [cb waitUntilCompleted];
      for (int r = 0; r < R; r++) {
        std::vector<std::pair<float, int>> o;
        for (int i = 0; i < V; i++) o.push_back({rows[r][i], i});
        std::sort(o.begin(), o.end(), sampling::desc);  // the contract's order: logit desc, then id asc
        std::vector<float> s(V);
        for (int i = 0; i < V; i++) s[i] = o[i].first;
        bool vals = true, ids = true;
        for (int j = 0; j < K; j++) {
          vals &= un.ptr<float>()[r * K + j] == o[j].first;
          ids &= cand.ptr<int>()[r * K + j] == o[j].second;
        }
        double z = 0;
        for (int i = 0; i < V; i++) z += std::exp((double)(rows[r][i] - s[0]) / 0.8);
        double zg = 0;
        for (int h = 0; h < SH; h++) {
          const float* q = zp.ptr<float>() + ((size_t)r * SH + h) * 2;
          if (q[1] > 0) zg += q[1] * std::exp((double)(q[0] - s[0]) / 0.8);
        }
        char what[96];
        snprintf(what, sizeof what, "top-%d values, row %d", K, r);
        check(vals, what);
        snprintf(what, sizeof what, "top-%d ids in the contract's order (ties by id), row %d", K, r);
        check(ids, what);
        snprintf(what, sizeof what, "normaliser row %d: gpu %.6g cpu %.6g", r, zg, z);
        check(std::fabs(zg - z) <= 1e-4 * z, what);
        if (r == 0) printf("top-%d ok on %d rows; normaliser rel err %.2g\n", K, R, std::fabs(zg - z) / z);
      }
    }

    // 2. the accept/resample distribution: N sequences share one verify block (rows {r, r}), draft d.
    const int N = 40000, K = 64;
    struct Case { const char* name; int row; float t, top_p; int top_k; int draft_rank; bool host; };
    const Case cases[] = {{"broad, t1 top_k 20", 0, 1.0f, 1.0f, 20, 0, false},
                          {"broad, t0.7 top_k 20 top_p 0.95, unlikely draft", 0, 0.7f, 0.95f, 20, 30, false},
                          {"broad, t1 top_k 64", 0, 1.0f, 1.0f, 64, 5, false},
                          {"broad, t1 top_k 16", 0, 1.0f, 1.0f, 16, 2, false},
                          {"peaked, t1 top_k off top_p 0.95", 1, 1.0f, 0.95f, 0, 1, false},
                          {"peaked, t0.6 top_k off top_p 0.8, draft off-nucleus", 1, 0.6f, 0.8f, 0, 50, false},
                          {"flat 30, t1 top_k 20 (ties)", 2, 1.0f, 1.0f, 20, 3, false},
                          {"flat 30, t1 top_k off top_p 0.5", 2, 1.0f, 0.5f, 0, 0, false},
                          {"very broad, top_k off top_p 0.9: host", 3, 1.0f, 0.9f, 0, 0, true}};
    for (const Case& c : cases) {
      BufRef cp = g.alloc((size_t)2 * SH * K * 4), vp = g.alloc((size_t)2 * SH * K * 4), cand = g.alloc(2 * K * 4),
             un = g.alloc(2 * K * 4), zp = g.alloc((size_t)2 * SH * 8), samp = g.alloc(N * 16), ob = g.alloc(N * 4),
             spec = g.alloc(N * 4), tok = g.alloc(8), pred = g.alloc(8), sst = g.alloc(N * 4), sl = g.alloc(N * 4),
             commit = g.alloc(N * 4), emit = g.alloc((size_t)N * MAXB * 4), host = g.alloc(N * 4);
      BufRef lg = g.alloc((size_t)2 * V * 2);
      for (int k = 0; k < 2; k++) memcpy(lg.ptr<uint16_t>() + (size_t)k * V, logits.ptr<uint16_t>() + (size_t)c.row * V, V * 2);
      std::vector<std::pair<float, int>> srt;
      for (int i = 0; i < V; i++) srt.push_back({rows[c.row][i], i});
      std::sort(srt.begin(), srt.end(), sampling::desc);
      const int d = srt[c.draft_rank].second;
      tok.ptr<int>()[0] = 1, tok.ptr<int>()[1] = d;
      for (int b = 0; b < N; b++) {
        SampParams sp{c.t, c.top_p, c.top_k, (uint32_t)(b * 2654435761u + 17)};
        memcpy(samp.ptr<char>() + b * 16, &sp, 16);
        ob.ptr<int>()[b] = 0, spec.ptr<int>()[b] = 1, sst.ptr<int>()[b] = 0, sl.ptr<int>()[b] = 2;
      }
      Top16Args ta{V, SH, SH, 0, 0};
      int S = N, S1 = 1;
      struct { int K, SH; } kz{K, SH};
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      Enc e(g, cb);
      e.k("row_top64_shard_v").buf(0, lg).buf(1, cp).buf(2, vp).args(3, ta);
      e.groups(S3(SH, 2), S3(256));
      e.k("top64_merge_v").buf(0, cp).buf(1, vp).buf(2, cand).buf(3, un).args(4, ta);
      e.groups(S3(2), S3(256));
      e.k("verify_zpart").buf(0, lg).buf(1, zp).args(2, ta).buf(3, samp).buf(4, ob).args(5, S1).buf(6, spec);
      e.groups(S3(SH, 2), S3(256));
      e.k("spec_accept_s").buf(0, tok).buf(1, pred).buf(2, sst).buf(3, sl).buf(4, ob).buf(5, spec).buf(6, commit)
          .buf(7, emit).args(8, S).buf(9, samp).buf(10, cand).buf(11, un).args(12, kz).buf(13, zp).buf(14, host);
      e.threads(S3(N), S3(64));
      e.end();
      [cb commit];
      [cb waitUntilCompleted];
      int nhost = 0;
      for (int b = 0; b < N; b++) nhost += host.ptr<int>()[b] == 0;
      char what[160];
      if (c.host) {
        snprintf(what, sizeof what, "%s: %d of %d rows handed to the host", c.name, nhost, N);
        check(nhost == N, what);
        printf("%-52s host rows %d/%d ok\n", c.name, nhost, N);
        continue;
      }
      snprintf(what, sizeof what, "%s: %d rows handed to the host", c.name, nhost);
      check(nhost == 0, what);
      auto p = contract(rows[c.row], c.t, c.top_p, c.top_k);
      std::vector<int> cnt(V, 0);
      int out = 0;
      for (int b = 0; b < N; b++) { int t = emit.ptr<int>()[b * MAXB]; if (t >= 0 && t < V) cnt[t]++; else out++; }
      double chi = 0, pool_e = 0, pool_o = 0;
      int df = -1;
      bool leak = out > 0;
      for (int t = 0; t < V; t++) {
        if (p[t] == 0) { leak |= cnt[t] > 0; continue; }
        if (N * p[t] < 5) { pool_e += N * p[t]; pool_o += cnt[t]; continue; }
        chi += (cnt[t] - N * p[t]) * (cnt[t] - N * p[t]) / (N * p[t]);
        df++;
      }
      if (pool_e >= 5) { chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e; df++; }
      const bool ok = !leak && chi <= df + 6 * std::sqrt(2.0 * std::max(df, 1)) + 10;
      snprintf(what, sizeof what, "%s: chi2 %.1f df %d%s", c.name, chi, df, leak ? ", mass outside the support" : "");
      check(ok, what);
      printf("%-52s chi2 %7.1f df %3d %s\n", c.name, chi, df, ok ? "ok" : "FAIL");
    }
    printf("spec_sample_test: %d/%d passed\n", runs - fails, runs);
    return fails != 0;
  }
}
