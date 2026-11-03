// Token trees for speculative verification (2026-11-02, TREE=<nodes>).
//
// DFlash2 drafts a block in parallel: for each position p of the block it leaves the 16 best candidate ids
// (cand, unary = their logits) and the selector's hidden projection hp[p] (rank 256). The selector scores a candidate c
// at p given the token chosen at p - 1 (prev) as unary[p][c] + sum_r pred_cb[prev][r] * hp[p][r] * succ_cb[c][r];
// the chain drafter takes the argmax at every position. A tree keeps alternatives: a node at depth d with token u has
// as children the candidates of position d + 1 ranked by that score with prev = u, and the tree grows best-first by
// estimated path probability (the product of the softmax of those 16 scores along the path), up to `budget` nodes.
// With one child per node (TREE_KIDS=1) it is exactly the chain draft_select produces.
//
// Layout handed to the target: nodes in DFS preorder (children by rank), node 0 = the staged token; node i has a
// parent row par[i] < i, a depth (its position is pos0 + depth) and an ancestor mask (bit j: node j is i or one of
// its ancestors). Preorder keeps every accepted path increasing (i_k >= k), so compacting it into rows 0..a is safe in
// place, and the GDN recurrence can keep one state per depth.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <vector>

constexpr int kMaxTree = 32;  // most nodes per tree (the ancestor mask is a uint32)

struct Tree {
  int n = 0;
  int tok[kMaxTree], par[kMaxTree], depth[kMaxTree];
  uint32_t anc[kMaxTree];
  float lp[kMaxTree];  // estimated log-probability of the path to the node
};

struct TreeParams {
  int budget = 16;
  int max_kids = 4;      // TREE_KIDS: alternatives per node
  int max_depth = 15;    // TREE_DEPTH (also capped by the drafted block)
  float temp = 1.5f;     // TREE_TEMP: softmax temperature on the selector scores (1.5: +2.4% committed vs 1)
  float decay = 0.f;     // TREE_DECAY: log-probability added per depth (< 0 favours width)
  float slope = 0.f;     // TREE_SLOPE: temperature at block position p is temp + slope * (p - 1)
  static TreeParams from_env(int budget) {
    TreeParams p;
    p.budget = budget;
    if (getenv("TREE_KIDS")) p.max_kids = atoi(getenv("TREE_KIDS"));
    if (getenv("TREE_DEPTH")) p.max_depth = atoi(getenv("TREE_DEPTH"));
    if (getenv("TREE_TEMP")) p.temp = atof(getenv("TREE_TEMP"));
    if (getenv("TREE_DECAY")) p.decay = atof(getenv("TREE_DECAY"));
    if (getenv("TREE_SLOPE")) p.slope = atof(getenv("TREE_SLOPE"));
    return p;
  }
};

inline float bf2f(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// The drafter's per-position output for one sequence's block (rows row0 .. row0 + block - 1 of its buffers).
struct DraftOut {
  const int* cand;        // [rows, 16]
  const float* unary;     // [rows, 16]
  const uint16_t* hp;     // [rows, 256] bf16
  const uint16_t* pred_cb, *succ_cb;  // [vocab, 256] bf16
  int row0, block;
};

// Children of a node with token `prev` at block position p (1 .. block - 1): candidate ids by descending score,
// with log-softmax probabilities.
inline void tree_children(const DraftOut& d, int prev, int p, float temp, int* ids, float* lps) {
  constexpr int K = 16, R = 256;
  const int row = d.row0 + p;
  float ph[R];
  const uint16_t* pc = d.pred_cb + (size_t)prev * R;
  const uint16_t* h = d.hp + (size_t)row * R;
  for (int r = 0; r < R; r++) ph[r] = bf2f(pc[r]) * bf2f(h[r]);
  float s[K];
  int ord[K];
  for (int c = 0; c < K; c++) {
    const uint16_t* sc = d.succ_cb + (size_t)d.cand[row * K + c] * R;
    float acc = 0;
    for (int r = 0; r < R; r++) acc += ph[r] * bf2f(sc[r]);
    s[c] = (acc + d.unary[row * K + c]) / temp;
    ord[c] = c;
  }
  std::stable_sort(ord, ord + K, [&](int a, int b) { return s[a] > s[b]; });
  float m = s[ord[0]], z = 0;
  for (int c = 0; c < K; c++) z += std::exp(s[c] - m);
  const float lz = m + std::log(z);
  for (int k = 0; k < K; k++) {
    ids[k] = d.cand[row * K + ord[k]];
    lps[k] = s[ord[k]] - lz;
  }
}

// Best-first tree of up to tp.budget nodes rooted at the staged token.
inline void build_tree(const DraftOut& d, int staged, const TreeParams& tp, Tree& t) {
  struct Cand { float lp; int node, rank; bool operator<(const Cand& o) const { return lp < o.lp; } };
  struct Kids { int ids[16]; float lps[16]; };
  const int budget = std::min(tp.budget, kMaxTree), maxd = std::min(tp.max_depth, d.block - 1);
  // creation order first, then a DFS relabel
  int ctok[kMaxTree], cpar[kMaxTree], cdep[kMaxTree], nkid[kMaxTree];
  float clp[kMaxTree];
  int kid[kMaxTree][16];
  static thread_local std::vector<Kids> kids(kMaxTree);
  int n = 1;
  ctok[0] = staged; cpar[0] = -1; cdep[0] = 0; clp[0] = 0; nkid[0] = 0;
  std::priority_queue<Cand> pq;
  auto expand = [&](int u) {  // rank u's children and offer the best one
    if (cdep[u] >= maxd) return;
    tree_children(d, ctok[u], cdep[u] + 1, std::max(0.1f, tp.temp + tp.slope * cdep[u]), kids[u].ids, kids[u].lps);
    pq.push({clp[u] + kids[u].lps[0] + tp.decay, u, 0});
  };
  expand(0);
  while (n < budget && !pq.empty()) {
    Cand c = pq.top();
    pq.pop();
    const int v = n++;
    ctok[v] = kids[c.node].ids[c.rank];
    cpar[v] = c.node;
    cdep[v] = cdep[c.node] + 1;
    clp[v] = c.lp;
    nkid[v] = 0;
    kid[c.node][nkid[c.node]++] = v;
    if (c.rank + 1 < tp.max_kids && c.rank + 1 < 16)
      pq.push({clp[c.node] + kids[c.node].lps[c.rank + 1] + tp.decay, c.node, c.rank + 1});
    expand(v);
  }
  // DFS preorder (children in rank order = creation order among siblings)
  int stack[kMaxTree], sp = 0, map[kMaxTree];
  stack[sp++] = 0;
  t.n = 0;
  while (sp) {
    const int u = stack[--sp];
    const int i = t.n++;
    map[u] = i;
    t.tok[i] = ctok[u];
    t.depth[i] = cdep[u];
    t.lp[i] = clp[u];
    t.par[i] = cpar[u] < 0 ? -1 : map[cpar[u]];
    t.anc[i] = (t.par[i] < 0 ? 0u : t.anc[t.par[i]]) | (1u << i);
    for (int k = nkid[u] - 1; k >= 0; k--) stack[sp++] = kid[u][k];
  }
}

// Longest path whose tokens the target confirms: truth[k] is the target's token after the path's k-th node. Returns
// the accepted drafts (path nodes after the root); path[0..a] are the node indices.
inline int tree_walk_truth(const Tree& t, const int* truth, int ntruth, int* path) {
  int cur = 0, a = 0;
  path[0] = 0;
  while (a < ntruth) {
    int nx = -1;
    for (int i = cur + 1; i < t.n; i++)
      if (t.par[i] == cur && t.tok[i] == truth[a]) { nx = i; break; }
    if (nx < 0) break;
    cur = nx;
    path[++a] = cur;
  }
  return a;
}
