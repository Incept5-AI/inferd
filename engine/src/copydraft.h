// Context-copy drafts (prompt lookup / n-gram drafting) for the Flash-Next MTP path (COPY_DRAFT=1).
//
// When the last `ng` tokens of a sequence (ending at its staged token) occurred earlier in the sequence (prompt or
// output), the tokens that followed that earlier occurrence are proposed as the continuation. The scheduler verifies
// such a chain instead of the MTP head's K drafts when it is longer and its first token agrees with the MTP head's own
// first draft (the fold's d1, already on the host after the previous step), so a copy never replaces a draft the MTP
// head disagrees with at depth 1.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

struct CopyIndex {
  std::unordered_map<uint64_t, int> last;  // n-gram key -> the most recent position its last token sits at
  int upto = 0;                            // n-grams ending at positions < upto are in the map
  int ng = 3;

  static uint64_t key(const int* t, int n) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; i++) h = (h ^ (uint32_t)t[i]) * 0x100000001B3ull + (h >> 29);
    return h;
  }
  // Propose up to K tokens following toks[0..e] (e: the staged token's index). Returns the proposal length (0: no
  // match); match: how many tokens before the continuation agree with the end of the sequence (>= ng, capped at 64).
  int propose(const std::vector<int>& toks, int e, int K, std::vector<int>& out, int& match) {
    out.clear();
    match = 0;
    if (e + 1 < ng + 1 || K <= 0) return 0;
    for (; upto < e; upto++)  // index every n-gram that ends before e
      if (upto >= ng - 1) last[key(&toks[upto - ng + 1], ng)] = upto;
    auto it = last.find(key(&toks[e - ng + 1], ng));
    if (it == last.end()) return 0;
    const int p = it->second;
    for (int i = 0; i < ng; i++)
      if (toks[p - i] != toks[e - i]) return 0;  // hash collision
    match = ng;
    while (match < 64 && p - match >= 0 && toks[p - match] == toks[e - match]) match++;
    for (int i = 1; i <= K && p + i <= e; i++) out.push_back(toks[p + i]);
    return (int)out.size();
  }
};

// Whether (and how long) to verify a copy chain instead of the MTP head's drafts, by expected tokens per ms.
// Verify-step cost by rows (COPY_COST "rows:ms,...", Studio Flash-Next B=1 at 2k, 2026-11-02: 4 rows 16.8 ms, 8 rows
// 26.2, 16 rows 36.0) and per-token acceptance of copy drafts by match-length bucket (learned from every copy cycle,
// starting from a prior), against the MTP head's measured tokens per cycle at its own block.
struct CopyPolicy {
  std::vector<std::pair<int, double>> cost{{1, 11.4}, {4, 16.75}, {6, 23.9}, {8, 26.2}, {12, 31.3}, {16, 36.0}};
  static constexpr int NB = 5;
  double succ[NB], tried[NB];
  double mtp_tokens = 3.5;  // EMA of MTP cycles' committed tokens
  CopyPolicy() {
    const double prior[NB] = {0.55, 0.7, 0.8, 0.88, 0.93};  // match 3-4, 5-8, 9-16, 17-32, 33+
    for (int b = 0; b < NB; b++) { tried[b] = 20; succ[b] = 20 * prior[b]; }
    if (const char* c = getenv("COPY_COST")) {
      cost.clear();
      for (const char* p = c; *p;) {
        int r = atoi(p);
        const char* q = strchr(p, ':');
        if (!q) break;
        cost.push_back({r, atof(q + 1)});
        p = strchr(q, ',');
        if (!p) break;
        p++;
      }
    }
  }
  static int bucket(int match) { return match <= 4 ? 0 : match <= 8 ? 1 : match <= 16 ? 2 : match <= 32 ? 3 : 4; }
  double t(int rows) const {  // linear interpolation in the cost table
    if (rows <= cost.front().first) return cost.front().second;
    for (size_t i = 1; i < cost.size(); i++)
      if (rows <= cost[i].first) {
        const auto [r0, t0] = cost[i - 1];
        const auto [r1, t1] = cost[i];
        return t0 + (t1 - t0) * (rows - r0) / (double)(r1 - r0);
      }
    const auto [r0, t0] = cost.back();
    return t0 * rows / r0;
  }
  // Copy drafts to verify (0: use the MTP head): the n <= avail maximising expected committed tokens per ms, if
  // that beats MTP's block of L rows. The first draft equals the MTP head's d1 (checked by the caller).
  int choose(int match, int avail, int L) const {
    const int b = bucket(match);
    const double a = std::min(0.995, succ[b] / tried[b]);
    const double base = mtp_tokens / t(L);
    int best = 0;
    double bv = base;
    double e = 1, ap = 1;  // expected committed tokens with n drafts: 1 + a + a^2 + ... + a^n
    for (int n = 1; n <= avail; n++) {
      ap *= a;
      e += ap;
      if (n < L - 1) continue;  // shorter than the MTP block: never better
      const double v = e / t(n + 1);
      if (v > bv) { bv = v; best = n; }
    }
    return best;
  }
  // After a copy cycle with n drafts that committed c tokens (c - 1 drafts accepted, then the bonus).
  void copy_done(int match, int n, int c) {
    const int b = bucket(match);
    succ[b] += c - 1;
    tried[b] += std::min(c, n);
  }
  void mtp_done(int c) { mtp_tokens = 0.98 * mtp_tokens + 0.02 * c; }
};
