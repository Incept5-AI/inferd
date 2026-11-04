// The one sampling contract for every path that draws a token (Codex review 2026-11-04, R03). Host only: the plain
// sampler (Scheduler::sample), the grammar's (gram::Matcher::sample) and the gate for the GPU's speculative sampler.
//
//   softmax(logits / temperature) over the candidate tokens (all of them, or the grammar's allowed ones), restricted to
//   the top_k of them by logit (top_k <= 0: no limit, the whole vocabulary), then to the smallest prefix in descending
//   order whose mass reaches top_p of what remains (the token that crosses it is included), renormalised; then a draw.
//
// Tokens whose weight is below 2^-64 of the largest one are left out before anything is summed: each has probability
// < 2^-64 (all of them together < 2^-46 for a 248k vocabulary), below anything a sample could show, and leaving them
// out keeps the work linear in the vocabulary with no sort of the tail. Ties sort by token id.
//
// The speculative path (kernels/spec.metal, spec_accept_s) draws from each verify row's top K logits (K = 64 by default,
// SPEC_TOPK=16 for the old set), so it implements this contract exactly when 1 <= top_k <= K; with top_k off and
// top_p < 1 it places the nucleus against the full softmax normaliser and is exact whenever the nucleus lies within
// the K candidates, handing any other row to this sampler (Scheduler: host-sampled rows, counted). spec_exact() is the
// admission rule; every other sampled request (top_k > K, or top_k off with top_p = 1) decodes one token per step here,
// unless it opted into truncation to the K candidates (spec_truncate). Before 2026-11-04 the host path capped "top_k
// off" at 1,024 candidates and the speculative path at 16, silently.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace sampling {

constexpr int kSpecTopK = 64;           // candidates per verify row on the GPU (spec.metal VKMAX; Qwen35::vtop_k)
constexpr float kSpan = 44.3614196f;   // 64 ln 2: weights below 2^-64 of the maximum are dropped

inline float bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float x;
  memcpy(&x, &u, 4);
  return x;
}

// Can the speculative sampler draw this request's distribution exactly with K candidates per row? Greedy always
// (argmax); top_k off with top_p < 1 row by row (a row whose nucleus does not fit goes to the host sampler).
inline bool spec_exact(float temperature, int top_k, float top_p = 1, int K = kSpecTopK) {
  return temperature <= 0 || (top_k >= 1 && top_k <= K) || (top_k <= 0 && top_p < 1);
}

// Descending logit, then ascending id.
inline bool desc(const std::pair<float, int>& a, const std::pair<float, int>& b) {
  return a.first > b.first || (a.first == b.first && a.second < b.second);
}

// Draw from candidates c = (logit, id) (any order; the caller has already dropped tokens below max - kSpan * temp and
// anything the grammar forbids). Applies top_k, temperature, top_p. c is reordered. Returns the token id, or -1 if c
// is empty.
inline int draw(std::vector<std::pair<float, int>>& c, float temp, float top_p, int top_k, std::mt19937_64& rng) {
  if (c.empty()) return -1;
  size_t n = c.size();
  if (top_k > 0 && (size_t)top_k < n) {
    std::nth_element(c.begin(), c.begin() + (top_k - 1), c.end(), desc);
    n = (size_t)top_k;
    c.resize(n);
  }
  float mx = -INFINITY;
  for (auto& x : c) mx = std::max(mx, x.first);
  std::vector<double> w(n);
  double Z = 0;
  for (size_t i = 0; i < n; i++) Z += w[i] = std::exp((double)(c[i].first - mx) / temp);
  if (top_p < 1) {  // the nucleus: sort only as far as it reaches (usually within the first 1,024)
    const double cut = top_p * Z;
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    auto by = [&](int a, int b) { return desc(c[a], c[b]); };
    size_t sorted = 0, m = std::min<size_t>(n, 1024), k = 0;
    double acc = 0;
    bool reached = false;
    while (!reached) {
      std::partial_sort(ord.begin() + sorted, ord.begin() + m, ord.end(), by);
      for (; k < m; k++) {
        acc += w[ord[k]];
        if (acc >= cut) { k++; reached = true; break; }
      }
      if (m == n) break;
      sorted = m;
      m = std::min(n, m * 8);
    }
    std::uniform_real_distribution<double> u(0, acc);
    double r = u(rng);
    for (size_t i = 0; i < k; i++)
      if ((r -= w[ord[i]]) <= 0) return c[ord[i]].second;
    return c[ord[k - 1]].second;
  }
  std::uniform_real_distribution<double> u(0, Z);
  double r = u(rng);
  for (size_t i = 0; i < n; i++)
    if ((r -= w[i]) <= 0) return c[i].second;
  return c[n - 1].second;
}

// The unconstrained sampler over bf16 logits [V]: greedy (temperature <= 0) is the argmax (first on ties).
inline int sample(const uint16_t* lg, int V, float temp, float top_p, int top_k, std::mt19937_64& rng) {
  int best = 0;
  float bv = -INFINITY;
  for (int i = 0; i < V; i++) {
    const float v = bf16(lg[i]);
    if (v > bv) { bv = v; best = i; }
  }
  if (temp <= 0 || !std::isfinite(bv)) return best;
  const float lo = bv - kSpan * temp;
  static thread_local std::vector<std::pair<float, int>> c;
  c.clear();
  for (int i = 0; i < V; i++) {
    const float v = bf16(lg[i]);
    if (v >= lo) c.push_back({v, i});
  }
  const int t = draw(c, temp, top_p, top_k, rng);
  return t < 0 ? best : t;
}

}  // namespace sampling
