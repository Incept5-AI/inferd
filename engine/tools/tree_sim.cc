// tree_sim <codebooks> <a.cycles> [b.cycles ...]: offline token-tree study on the CPU (no GPU, no model).
//   The .cycles files come from spec_test with TREE_SIM=... TREE_SIM_DUMP=<prefix> (the drafter's per-position
//   candidates of every cycle of a chain run, plus the run's greedy output). For each tree shape it builds the
//   tree (src/tree.h) and walks it against the output, so it reports the tokens a tree verify would commit per cycle.
//   Env: SIM_BUDGETS (12,16), SIM_KIDS (4), SIM_TEMPS (1,1.5,2), SIM_DEPTHS (15).
// Build: clang++ -std=c++20 -O2 tools/tree_sim.cc -o build/tree_sim
#include <cstdio>
#include <string>
#include <vector>
#include "../src/tree.h"

struct Cyc { int b, pos0, staged, commit; std::vector<int> cand; std::vector<float> unary; std::vector<uint16_t> hp; };
struct Run { std::string name; int block; std::vector<Cyc> cyc; std::vector<std::vector<int>> toks; };

static std::vector<float> list(const char* env, const char* dflt) {
  std::vector<float> v;
  for (const char* s = getenv(env) ? getenv(env) : dflt; *s;) {
    v.push_back(atof(s));
    while (*s && *s != ',') s++;
    if (*s) s++;
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: tree_sim <codebooks> <cycles...>\n"); return 1; }
  FILE* f = fopen(argv[1], "rb");
  int64_t sz;
  fread(&sz, 8, 1, f);
  std::vector<uint16_t> pred(sz / 2), succ(sz / 2);
  fread(pred.data(), 1, sz, f);
  fread(succ.data(), 1, sz, f);
  fclose(f);
  std::vector<Run> runs;
  for (int a = 2; a < argc; a++) {
    Run r;
    r.name = argv[a];
    FILE* g = fopen(argv[a], "rb");
    int hdr[3];
    fread(hdr, 4, 3, g);
    r.block = hdr[1];
    for (int i = 0; i < hdr[0]; i++) {
      Cyc c;
      int h[4];
      fread(h, 4, 4, g);
      c.b = h[0]; c.pos0 = h[1]; c.staged = h[2]; c.commit = h[3];
      c.cand.resize(r.block * 16); c.unary.resize(r.block * 16); c.hp.resize(r.block * 256);
      fread(c.cand.data(), 4, c.cand.size(), g);
      fread(c.unary.data(), 4, c.unary.size(), g);
      fread(c.hp.data(), 2, c.hp.size(), g);
      r.cyc.push_back(std::move(c));
    }
    for (int s = 0; s < hdr[2]; s++) {
      int n;
      fread(&n, 4, 1, g);
      std::vector<int> t(n);
      fread(t.data(), 4, n, g);
      r.toks.push_back(t);
    }
    fclose(g);
    double ch = 0;
    for (auto& c : r.cyc) ch += c.commit;
    printf("%s: %zu cycles, chain commit %.3f\n", argv[a], r.cyc.size(), ch / r.cyc.size());
    runs.push_back(std::move(r));
  }
  for (float bud : list("SIM_BUDGETS", "12,16"))
    for (float kk : list("SIM_KIDS", "4"))
      for (float dep : list("SIM_DEPTHS", "15"))
        for (float temp : list("SIM_TEMPS", "1,1.5,2"))
        for (float dec : list("SIM_DECAYS", "0"))
        for (float sl : list("SIM_SLOPES", "0")) {
          TreeParams tp;
          tp.decay = dec;
          tp.slope = sl;
          tp.budget = (int)bud; tp.max_kids = (int)kk; tp.max_depth = (int)dep; tp.temp = temp;
          printf("budget %2d kids %d depth %2d temp %.2f decay %.2f slope %.2f:", tp.budget, tp.max_kids, tp.max_depth, tp.temp,
                 tp.decay, tp.slope);
          double gm = 1;
          for (auto& r : runs) {
            double sum = 0;
            for (auto& c : r.cyc) {
              DraftOut o{c.cand.data(), c.unary.data(), c.hp.data(), pred.data(), succ.data(), 0, r.block};
              Tree t;
              build_tree(o, c.staged, tp, t);
              const auto& tk = r.toks[c.b];
              int nt = std::max(0, std::min((int)tk.size() - c.pos0 - 1, kMaxTree));
              int path[kMaxTree + 1];
              sum += tree_walk_truth(t, tk.data() + c.pos0 + 1, nt, path) + 1;
            }
            const double m = sum / r.cyc.size();
            gm *= m;
            printf(" %.3f", m);
          }
          printf("  | geomean %.3f\n", std::pow(gm, 1.0 / runs.size()));
        }
}
