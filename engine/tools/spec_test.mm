// spec_test <target_dir> <draft_dir> <tokens.txt> <n_gen> [B]
//   Greedy-decodes n_gen tokens after each prompt in tokens.txt ("B T" + rows), first plainly and then with
//   DFlash2 speculation, for B sequences batched together. Checks the two outputs are identical (speculation
//   must be lossless) and reports tokens/s and tokens committed per cycle.
#include <chrono>
#include <fstream>
#include <sstream>
#include "../src/dflash.h"
#include "../src/tree.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double g_wait_s = 0, g_sched_s = 0;  // host time in commit+wait; submit -> GPU start
static void commit_cb(id<MTLCommandBuffer> cb) {
  double t0 = now();
  [cb commit];
  [cb waitUntilCompleted];
  g_wait_s += now() - t0;
  g_sched_s += cb.GPUStartTime - cb.kernelStartTime;
  if (cb.error) { fprintf(stderr, "cb: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
}

struct SimCycle { int b, pos0, staged, commit; std::vector<int> cand; std::vector<float> unary; std::vector<uint16_t> hp; };
static const bool tree_sim_on = getenv("TREE_SIM") != nullptr;

// Offline tree study: for every recorded cycle, build trees from the drafter's output and walk them against the
// greedy continuation the run produced (the target's own tokens), so acceptance per tree shape needs no verify pass.
static void tree_sim(const std::vector<SimCycle>& sim, const std::vector<std::vector<int>>& toks, Drafter& d) {
  std::vector<int> budgets;
  for (const char* s = getenv("TREE_SIM"); *s;) { budgets.push_back(atoi(s)); while (*s && *s != ',') s++; if (*s) s++; }
  std::vector<int> kids{1, 2, 3, 4, 8};
  if (getenv("TREE_SIM_KIDS")) { kids.clear(); for (const char* s = getenv("TREE_SIM_KIDS"); *s;) { kids.push_back(atoi(s)); while (*s && *s != ',') s++; if (*s) s++; } }
  std::vector<float> temps{1.f};
  if (getenv("TREE_SIM_TEMPS")) { temps.clear(); for (const char* s = getenv("TREE_SIM_TEMPS"); *s;) { temps.push_back(atof(s)); while (*s && *s != ',') s++; if (*s) s++; } }
  if (const char* dump = getenv("TREE_SIM_DUMP")) {  // for tools/tree_sim.cc (offline, CPU): cycles + outputs + codebooks
    FILE* f = fopen((std::string(dump) + ".cycles").c_str(), "wb");
    int hdr[3] = {(int)sim.size(), d.block, (int)toks.size()};
    fwrite(hdr, 4, 3, f);
    for (auto& c : sim) {
      int h[4] = {c.b, c.pos0, c.staged, c.commit};
      fwrite(h, 4, 4, f);
      fwrite(c.cand.data(), 4, c.cand.size(), f);
      fwrite(c.unary.data(), 4, c.unary.size(), f);
      fwrite(c.hp.data(), 2, c.hp.size(), f);
    }
    for (auto& t : toks) { int n = (int)t.size(); fwrite(&n, 4, 1, f); fwrite(t.data(), 4, n, f); }
    fclose(f);
    std::string cbp = std::string(dump) + ".codebooks";
    if (FILE* e = fopen(cbp.c_str(), "rb")) fclose(e);
    else {
      FILE* g = fopen(cbp.c_str(), "wb");
      const size_t n = d.pred_cb.buf.length - d.pred_cb.off;
      int64_t sz = (int64_t)std::min(n, (size_t)d.succ_cb.buf.length - d.succ_cb.off);
      fwrite(&sz, 8, 1, g);
      fwrite(d.pred_cb.ptr<char>(), 1, sz, g);
      fwrite(d.succ_cb.ptr<char>(), 1, sz, g);
      fclose(g);
    }
  }
  double actual = 0;
  for (auto& c : sim) actual += c.commit;
  printf("tree sim: %zu cycles, actual chain commit %.3f per cycle\n", sim.size(), actual / sim.size());
  for (float temp : temps)
  for (int kk : kids)
    for (int bud : budgets) {
      TreeParams tp;
      tp.budget = bud; tp.max_kids = kk; tp.temp = temp;
      double sum = 0, nodes = 0, maxdep = 0;
      std::vector<int> hist(kMaxTree + 1, 0);
      for (auto& c : sim) {
        DraftOut o{c.cand.data(), c.unary.data(), c.hp.data(), d.pred_cb.ptr<uint16_t>(), d.succ_cb.ptr<uint16_t>(), 0, d.block};
        Tree t;
        build_tree(o, c.staged, tp, t);
        const auto& tk = toks[c.b];
        int nt = std::max(0, std::min((int)tk.size() - c.pos0 - 1, kMaxTree));
        int path[kMaxTree + 1];
        int a = tree_walk_truth(t, tk.data() + c.pos0 + 1, nt, path);
        sum += a + 1;
        hist[a + 1]++;
        nodes += t.n;
        int md = 0;
        for (int i = 0; i < t.n; i++) md = std::max(md, t.depth[i]);
        maxdep += md;
      }
      printf("  temp %.2f kids %d budget %2d: commit %.3f per cycle (nodes %.1f, depth %.1f) hist", temp, kk, bud,
             sum / sim.size(), nodes / sim.size(), maxdep / sim.size());
      for (int i = 1; i <= kMaxTree; i++) if (hist[i]) printf(" %d:%d", i, hist[i]);
      printf("\n");
    }
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 5) { fprintf(stderr, "usage: spec_test <target> <draft> <tokens.txt> <n_gen> [B]\n"); return 1; }
  @autoreleasepool {
    Gpu g("build/core.metallib");
    Checkpoint tck(argv[1]), dck(argv[2]);
    std::ifstream f(argv[3]);
    int B0, T;
    f >> B0 >> T;
    std::vector<std::vector<int>> prompts(B0, std::vector<int>(T));
    for (auto& r : prompts) for (auto& v : r) f >> v;
    int n_gen = atoi(argv[4]), B = argc > 5 ? atoi(argv[5]) : B0;
    int maxb = (T + n_gen + 16) / KV_BLOCK + 2;
    Qwen35 m(g, tck, std::max(T, B * kMaxBlock), B, B, B * maxb, maxb);  // prefill one prompt per step, like the server
    double t0 = now();
    Drafter d(g, m, dck);
    m.enable_spec(d.target_layers, d.block);
    printf("drafter loaded in %.1f s\n", now() - t0);
    std::vector<std::vector<int>> blocks(B);
    for (int b = 0; b < B; b++)
      for (int j = 0; j < maxb; j++) blocks[b].push_back(b * maxb + j);

    std::vector<std::vector<int>> outs[2];
    // SPEC_REF=<file>: the plain run's output is read from the file when it exists (else run and saved there).
    const char* ref = getenv("SPEC_REF");
    bool have_ref = false;
    if (ref) {
      std::ifstream rf(ref);
      std::vector<std::vector<int>> r;
      std::string line;
      while (std::getline(rf, line)) {
        std::istringstream is(line);
        std::vector<int> v;
        int x;
        while (is >> x) v.push_back(x);
        if (!v.empty()) r.push_back(v);
      }
      if ((int)r.size() == B && (int)r[0].size() >= T + n_gen) { outs[0] = r; have_ref = true; printf("plain: from %s\n", ref); }
    }
    for (int mode = have_ref ? 1 : 0; mode < 2; mode++) {
      bool spec_on = mode == 1;
      m.verify = true;  // also captures hidden and computes argmax for the plain run
      std::vector<std::vector<int>> toks(B);
      std::vector<int> computed(B, 0), staged(B);
      for (int b = 0; b < B; b++) { m.reset_slot(b); toks[b] = prompts[b % B0]; }
      // Prefill one prompt per step and enter its rows into the drafter context, as the server does (a single step
      // over all prompts sized the per-layer tapes for T * B tokens: 177 GiB at 16 x 6k).
      std::vector<Drafter::CtxRow> ctx;
      for (int b = 0; b < B; b++) {
        std::vector<Qwen35::SeqWork> w{{b, &blocks[b], 0, toks[b].data(), T, false}};
        int TT = m.set_batch(w);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        m.encode_step(cb, TT, 1);
        commit_cb(cb);
        staged[b] = m.pred.ptr<int>()[0];
        toks[b].push_back(staged[b]);
        computed[b] = T;
        for (int j = 0; j < T; j++) ctx.push_back({j, j, b});
        if (spec_on) {
          cb = [g.queue commandBuffer];
          Enc e(g, cb);
          d.encode_context(e, ctx, 0);
          e.end();
          commit_cb(cb);
        }
        ctx.clear();
      }
      id<MTLCommandBuffer> cb;
      std::vector<int> vrow(B);
      // TREE=<nodes>: verify a token tree per sequence (src/tree.h) instead of a chain of d.block rows.
      static const int tree_n = getenv("TREE") ? std::min(atoi(getenv("TREE")), kMaxTree) : 0;
      static const TreeParams tparams = TreeParams::from_env(tree_n);
      const bool tree_on = spec_on && tree_n > 1;
      double tree_ms = 0;
      std::vector<SimCycle> sim;
      int cycles = 0;
      std::vector<int> chist(kMaxBlock + 1, 0);  // cycles committing c tokens (c = accepted drafts + 1)
      uint64_t disp0 = Enc::dispatches;
      if (Enc::hist) Enc::hist->clear();
      if (Enc::prof) Enc::prof->clear();
      double gpu_ms = 0, draft_ms = 0, ctx_ms = 0;
      double ts = now();
      g_wait_s = g_sched_s = 0;
      auto shortest = [&] { size_t n = SIZE_MAX; for (auto& t : toks) n = std::min(n, t.size()); return (int)n; };
      while (shortest() < T + n_gen + 1) {
        cb = [g.queue commandBuffer];
        std::vector<Qwen35::SeqWork> vw;
        std::vector<std::vector<int>> vt(B);
        std::vector<Drafter::BlockSeq> bs;
        int row = 0;
        for (int b = 0; b < B; b++) {
          int L = tree_on ? tree_n : spec_on ? d.block : 1;
          vt[b].assign(L, 0);
          vt[b][0] = staged[b];
          vw.push_back({b, &blocks[b], computed[b], vt[b].data(), L, spec_on});
          vw.back().tree = tree_on;
          bs.push_back({b, computed[b], staged[b], row, tree_on});
          vrow[b] = row;
          row += L;
        }
        int TT2 = m.set_batch(vw);
        static const bool tree_gpu = getenv("TREE_GPU") && atoi(getenv("TREE_GPU"));
        if (spec_on && !getenv("SKIP_DRAFT")) {
          Enc e(g, cb);
          d.encode_context(e, ctx, 0);
          if (getenv("SPEC_PROF")) {  // context append on its own
            e.end();
            commit_cb(cb);
            ctx_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            cb = [g.queue commandBuffer];
            e.e = [cb computeCommandEncoder];
          }
          d.encode_block(e, bs);
          if (tree_on && tree_gpu) {
            std::vector<std::array<int, 4>> ts;
            for (int b = 0; b < B; b++) ts.push_back({b * d.block, vrow[b], computed[b], tree_n});
            d.encode_tree(e, ts, tparams, d.block);
          }
          e.end();
          if (getenv("SPEC_PROF")) {  // time the drafter on its own (costs one extra sync per cycle)
            commit_cb(cb);
            draft_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            cb = [g.queue commandBuffer];
          }
        }
        id<MTLCommandBuffer> dcb = nil;
        static const bool split_draft = getenv("SPLIT_DRAFT") != nullptr;  // timing: a chain run with the tree's split
        if ((tree_on && !tree_gpu) || (spec_on && split_draft)) {  // the drafter runs while the verify is encoded; the trees are built in between
          dcb = cb;
          [dcb commit];
          cb = [g.queue commandBuffer];
        }
        m.encode_step(cb, TT2, B);
        if (dcb) {
          [dcb waitUntilCompleted];
          gpu_ms += (dcb.GPUEndTime - dcb.GPUStartTime) * 1e3;
          double tt = now();
          for (int b = 0; b < B && tree_on; b++) {
            DraftOut o{d.cand.ptr<int>(), d.unary.ptr<float>(), d.hp.ptr<uint16_t>(), d.pred_cb.ptr<uint16_t>(),
                       d.succ_cb.ptr<uint16_t>(), b * d.block, d.block};
            Tree t;
            build_tree(o, staged[b], tparams, t);
            m.set_tree(b, computed[b], t);
          }
          tree_ms += (now() - tt) * 1e3;
        }
        commit_cb(cb);
        gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        cycles++;
        ctx.clear();
        if (spec_on && tree_sim_on) {  // TREE_SIM: keep the drafter's per-position output for the offline tree study
          const int K = 16, R = 256, bl = d.block;
          for (int b = 0; b < B; b++) {
            SimCycle sc;
            sc.b = b; sc.pos0 = computed[b]; sc.staged = staged[b];
            sc.cand.assign(d.cand.ptr<int>() + b * bl * K, d.cand.ptr<int>() + (b + 1) * bl * K);
            sc.unary.assign(d.unary.ptr<float>() + b * bl * K, d.unary.ptr<float>() + (b + 1) * bl * K);
            sc.hp.assign(d.hp.ptr<uint16_t>() + (size_t)b * bl * R, d.hp.ptr<uint16_t>() + (size_t)(b + 1) * bl * R);
            sc.commit = m.commit.ptr<int>()[b];
            sim.push_back(std::move(sc));
          }
        }
        for (int b = 0; b < B; b++) {
          int c = m.commit.ptr<int>()[b];
          if (spec_on) {
            chist[std::min(c, kMaxBlock)]++;
            for (int i = 0; i < c; i++) toks[b].push_back(m.emit.ptr<int>()[b * kMaxBlock + i]);
          } else {
            toks[b].push_back(m.pred.ptr<int>()[b]);
          }
          for (int i = 0; i < c; i++)
            ctx.push_back({vrow[b] + (tree_on ? m.tpath.ptr<int>()[b * kMaxBlock + i] : i), computed[b] + i, b});
          computed[b] += c;
          staged[b] = toks[b].back();
        }
      }
      double el = now() - ts;
      if (spec_on && !sim.empty()) tree_sim(sim, toks, d);
      printf("%s: host %.1f ms per cycle, %.1f ms of it in commit+wait, %.1f ms submit->GPU start\n", spec_on ? "spec " : "plain",
             el * 1e3 / cycles, g_wait_s * 1e3 / cycles, g_sched_s * 1e3 / cycles);
      int gen = 0;
      for (auto& t : toks) gen += (int)t.size() - T;
      printf("%s: %.0f kernel dispatches per cycle\n", spec_on ? "spec " : "plain", (double)(Enc::dispatches - disp0) / cycles);
      if (Enc::prof && spec_on) {
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& [k, t] : *Enc::prof) { v.push_back({t, k}); tot += t; }
        std::sort(v.rbegin(), v.rend());
        printf("  kernel GPU time per cycle (each dispatch alone): %.1f ms total\n", tot / cycles);
        for (size_t i = 0; i < v.size() && i < 30; i++) printf("  %7.3f ms  %s\n", v[i].first / cycles, v[i].second.c_str());
      }
      if (Enc::hist && spec_on) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (auto& [k, n] : *Enc::hist) v.push_back({n, k});
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 25; i++) printf("  %6.1f per cycle  %s\n", (double)v[i].first / cycles, v[i].second.c_str());
      }
      printf("%s: %d tokens in %.2f s = %.1f tok/s total over %d seqs, %.2f tokens per cycle, %.1f ms GPU per cycle\n",
             spec_on ? "spec " : "plain", gen, el, gen / el, B, (double)gen / B / cycles, gpu_ms / cycles);
      if (spec_on) {
        printf("  commit histogram (tokens: cycles):");
        for (int c = 1; c <= kMaxBlock; c++) if (chist[c]) printf(" %d:%d", c, chist[c]);
        printf("\n");
      }
      if (tree_on) printf("  tree: %d nodes, kids %d, depth %d, temp %.2f; host build %.3f ms per cycle\n", tree_n,
                          tparams.max_kids, tparams.max_depth, tparams.temp, tree_ms / cycles);
      if (draft_ms > 0)
        printf("  drafter context %.1f ms + block %.1f ms + target step %.1f ms per cycle\n", ctx_ms / cycles,
               draft_ms / cycles, (gpu_ms - draft_ms - ctx_ms) / cycles);
      outs[mode] = toks;
      if (mode == 0 && ref) {
        std::ofstream o(ref);
        for (auto& t : toks) { for (int v : t) o << v << " "; o << "\n"; }
      }
    }
    int same = 0;
    for (int b = 0; b < B; b++) {
      int n = T + n_gen, k = T;
      while (k < n && k < (int)outs[0][b].size() && k < (int)outs[1][b].size() && outs[0][b][k] == outs[1][b][k]) k++;
      same += k >= n;
      if (k < n) printf("seq %d diverges after %d generated tokens (plain %d vs spec %d)\n", b, k - T, outs[0][b][k], outs[1][b][k]);
    }
    {
      std::ofstream o("/tmp/spec_out.txt");
      for (int mode = 0; mode < 2; mode++)
        for (auto& t : outs[mode]) { for (int v : t) o << v << " "; o << "\n"; }
    }
    printf("lossless check: %d/%d sequences identical over %d tokens\n", same, B, n_gen);
  }
}
