// Continuous-batching scheduler with a prefix cache for hybrid models.
//
// Each step mixes one decode token for every generating sequence with prefill chunks for new ones, under a
// token budget (vLLM-style chunked prefill). KV lives in 32-token blocks from a refcounted pool. GDN state
// lives in slots; a slot is either bound to a running sequence or holds a prefix-cache snapshot.
//
// Prefix cache: full KV blocks are entered into a trie keyed by their tokens, and immutable once there. A cache
// hit also needs the recurrent state at exactly that position, which cannot be rebuilt from KV, so each prompt's
// prefill is split at a snapshot point a little before its end (chat templates re-render the previous turn's
// tail differently, so follow-up turns diverge a few tokens early) and the slot is copied to a snapshot there.
// A new request resumes from the deepest trie node that carries a snapshot.
#pragma once
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <random>
#include <unordered_map>
#include "spill.h"
#include "mtp.h"
#include "copydraft.h"
#include "grammar.h"

struct Request {
  uint64_t id;
  std::vector<int> prompt;
  int max_tokens = 256;
  float temperature = 0, top_p = 1;
  int top_k = 0;
  std::vector<int> stop_ids;
  uint64_t seed = 0;
  bool has_seed = false;
  // Per-agent policy: max_context caps prompt + generated tokens (0: the server's max_ctx only; a longer prompt is
  // refused, generation stops at the cap with "length"); priority decides who is preempted first when the KV pool
  // runs dry (lowest first, then newest). Both act only at submit / preemption time, never per step.
  int max_context = 0;
  int priority = 0;
  // Prompt positions worth a prefix-cache snapshot (the front end sends where the system prompt ends), so agents that
  // share that much but diverge after it resume there instead of recomputing it.
  std::vector<int> snap_hints;
  // The next turn's prompt will extend this one's prompt + output token for token (thinking off: the template
  // re-renders the answer exactly), so snapshots taken while decoding let that turn resume near the answer's end.
  bool gen_snap = false;
  // Multimodal: M-RoPE positions [prompt, 3], the delta for later positions, and bf16 image features
  // [n, hidden] for the prompt rows whose token id is >= vocab (in prompt order).
  std::vector<int> mrope;
  int rope_delta = 0;
  std::vector<uint16_t> embeds;
  // Structured output: a JSON Schema matcher (src/grammar.h). Host-sampled tokens are drawn under it. Greedy requests
  // stay speculative: a block's committed tokens are checked against it, and the first rejected one is replaced by that
  // row's masked argmax (exact); if drafts past it were already folded into the state, the sequence is re-queued to
  // re-prefill its prompt + output (Scheduler::step, rollback). Sampled requests are speculated only while the grammar
  // is inactive (reasoning): the block is cut right after </think>, and from there they decode one token per step under
  // Matcher::sample (the GPU's speculative sampler keeps 16 candidates; masking only afterwards is not exact).
  std::shared_ptr<gram::Matcher> grammar;
  // Agent contexts (docs/CONTEXTS.md). prefill_only: compute the prompt (cut to whole KV blocks), snapshot at its end and
  // finish without sampling; with ctx_name the chain is registered under that name, pinned (ctx_pin: exempt from
  // eviction) and/or written to the SSD tier (ctx_ssd). no_cache: take no snapshots for this request (one-off agents:
  // nothing of theirs stays in the prefix cache). session: the agent this turn belongs to; its latest snapshot is
  // evicted after other cached prefixes (and goes to the SSD tier first when there is one).
  bool prefill_only = false, no_cache = false, ctx_pin = false, ctx_ssd = false;
  std::string ctx_name, session;
};

// Engine → caller events. tokens: newly generated; done: finish reason "stop"/"length"/"error".
struct Event {
  uint64_t id;
  std::vector<int> tokens;
  bool done = false;
  std::string reason;
  int prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
  std::string extra;  // raw JSON members appended to the done event (agent contexts: ,"ctx":{...})
};

struct PrefixCache {
  struct Node {
    int parent;          // -1 for children of the root
    std::vector<int> tokens;  // the block's KV_BLOCK tokens
    int block;           // KV block id (the trie holds one reference)
    int snapshot = -1;   // state slot holding the state after this block, or -1
    int children = 0;
    uint64_t last_used = 0;
    int pin = 0;       // pinned contexts whose chain runs through this node: never evicted
    int pin_snap = 0;  // pinned contexts ending here: the snapshot is never dropped
    int sess = 0;      // the latest snapshot of an agent session: evicted after unmarked prefixes
  };
  std::unordered_map<int, Node> nodes;
  std::unordered_map<uint64_t, std::vector<int>> index;  // hash(parent, tokens) -> node ids
  int next_id = 0;

  static uint64_t key(int parent, const int* t) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t)(parent + 1);
    for (int i = 0; i < KV_BLOCK; i++) h = (h ^ (uint32_t)t[i]) * 1099511628211ull;
    return h;
  }
  int find(int parent, const int* t) const {
    auto it = index.find(key(parent, t));
    if (it == index.end()) return -1;
    for (int id : it->second) {
      const Node& n = nodes.at(id);
      if (n.parent == parent && std::equal(n.tokens.begin(), n.tokens.end(), t)) return id;
    }
    return -1;
  }
  int insert(int parent, const int* t, int block) {
    int id = next_id++;
    nodes[id] = Node{parent, std::vector<int>(t, t + KV_BLOCK), block};
    index[key(parent, t)].push_back(id);
    if (parent >= 0) nodes[parent].children++;
    return id;
  }
  void erase(int id) {
    Node& n = nodes[id];
    const auto k = key(n.parent, n.tokens.data());
    auto& v = index[k];
    v.erase(std::find(v.begin(), v.end(), id));
    if (v.empty()) index.erase(k);  // else the index grows with every evicted prefix (Codex review F06)
    if (n.parent >= 0) nodes[n.parent].children--;
    nodes.erase(id);
  }
};

struct Scheduler {
  Qwen35& m;
  Gpu& g;
  int max_running, step_tokens, snap_margin;
  uint64_t clock = 0;

  // KV block pool (refcounted) and state-slot pool.
  std::vector<int> block_ref;
  // Free KV blocks: a stack with lazy removal (is_free is the truth), so a sequence can also take a specific
  // block: the one right after its previous block, keeping groups of 4 logical pages physically contiguous where
  // possible (attn_mpp then reads a 128-key K tile in one matmul).
  std::vector<int> free_blocks;
  std::vector<char> is_free;
  size_t n_free = 0;
  // Aligned groups of 4 blocks that are entirely free (lazy stack, grp_free is the truth). A sequence starting a new
  // group of 4 logical pages takes a whole free group, so the pool does not fragment over a long uptime: scattered
  // pages cost 10% at 256k x 1 and 14% at 128k x 4 per verify step (decode_test BENCH_SHUFFLE, 2026-10-31), the two
  // cells the milestone run lost to Splash after hours of earlier runs on the same server.
  std::vector<uint8_t> grp_free;
  std::vector<int> free_groups;
  void push_free(int b) {
    is_free[b] = 1; n_free++; free_blocks.push_back(b);
    if (++grp_free[b / 4] == 4) free_groups.push_back(b / 4);
    // The lazy stacks keep stale entries for blocks taken by group or by hint; rebuild them from the flags before
    // they outgrow the pool (Codex review R03: 4 blocks left 40,004 stack entries after 10,000 cycles).
    if (free_blocks.size() > 2 * (size_t)m.n_blocks + 64 || free_groups.size() > (size_t)m.n_blocks / 2 + 64) {
      free_blocks.clear();
      free_groups.clear();
      for (int x = 0; x < m.n_blocks; x++) if (is_free[x]) free_blocks.push_back(x);
      for (int gi = 0; gi < (m.n_blocks + 3) / 4; gi++) if (grp_free[gi] == 4) free_groups.push_back(gi);
    }
  }
  std::vector<int> free_slots;
  PrefixCache cache;
  int snapshot_count = 0;
  // Models with no per-sequence state besides KV (pure attention: Muse Glimmer, Gemma-4; no drafter or MTP head,
  // whose context rings and state live in slots too) can resume at any cached block, so their prefixes need no
  // snapshot slot. Until 2026-11-02 they were capped by slots like the hybrids: 16 agents x 32k on a mini kept one
  // cached prefix per slot and re-prefilled the rest one after another (Glimmer, M5 Pro: worst TTFT 503 s). Now
  // every trie node is a resume point; snapshot points are only marked (kNoState) so the SSD tier writes those chains
  // when they are evicted. PREFIX_ANY=0 restores the slot-capped behaviour.
  bool stateless = false;
  static constexpr int kNoState = -2;
  // Trie nodes entered (and snapshots published) while encoding the current step: their KV and state are written by
  // that step's command buffer, so eviction and the SSD tier must not touch them until it has completed.
  std::unordered_set<int> pending_nodes;
  std::vector<int> pending_snaps;
  // Exact context limit in tokens (the server's max_ctx); 0: the block table's capacity.
  int ctx_limit = 0;
  // Flash-Next without the QSA indexer (QSA=0) is exact only while every query sees <= 2,048 keys.
  int qwen4_exact = 2048;
  bool qwen4_long = getenv("FN_ALLOW_LONG") && atoi(getenv("FN_ALLOW_LONG"));

  struct Seq {
    Request req;
    std::vector<int> tokens;  // prompt + generated
    int computed = 0;         // tokens whose KV/state are in the slot
    int slot = -1;
    std::vector<int> blocks;
    int cached = 0;
    std::vector<int> snap_pts;  // positions to snapshot at (block-aligned, ascending): the prompt's end, plus any
                                // shared-prefix point a fan-out follower asked for
    int next_snap() const { return snap_pts.empty() ? -1 : snap_pts.front(); }
    void add_snap(int pos) {
      auto it = std::lower_bound(snap_pts.begin(), snap_pts.end(), pos);
      if (it == snap_pts.end() || *it != pos) snap_pts.insert(it, pos);
    }
    int generated = 0;
    int bypassed = 0;  // admissions that went past this waiting request (ADMIT_LOOKAHEAD aging)
    bool spec_ok = false;                      // greedy and a drafter is loaded: decode by draft + verify
    std::vector<Drafter::CtxRow> pend;         // last step's committed rows, to enter into the drafter context
    std::mt19937_64 rng;
    std::vector<int> ext_idx;  // per prompt position: image feature row, or -1 (empty for text-only prompts)
    bool mtp_fold = false;     // MTP: the last step left this sequence's d1 / pass-0 state on the GPU (Mtp::fold)
    CopyIndex copy;            // COPY_DRAFT: n-gram index of tokens (src/copydraft.h)
    std::vector<int> cdraft;   // COPY_DRAFT: this step's copy chain (verified instead of the MTP drafts), or empty
    int cmatch = 0;            // COPY_DRAFT: that chain's match length
    bool resumed = false;      // preempted mid-generation: re-prefilling prompt + output before decoding again
    bool snap_due = false;     // SNAP_GEN: the last step ended exactly on a decode snapshot boundary
    int last_snap = 0;         // SNAP_GEN: position of this sequence's last decode snapshot
    bool prefilling() const { return computed < (int)tokens.size(); }
    // Generating: only the staged (last sampled, not yet processed) token is outstanding.
    bool decoding() const { return generated > 0 && !resumed; }
    // Blocks this sequence will hold by the end of its turn, with up to `reserve` generated tokens.
    int target_blocks(int reserve) const {
      // context so far plus the remaining output allowance (capped by the reserve)
      int base = (int)tokens.size() + std::min(std::max(0, req.max_tokens - generated), reserve) + kMaxBlock + 1;
      return std::max((int)blocks.size(), (base + KV_BLOCK - 1) / KV_BLOCK);
    }
  };
  std::deque<Seq> waiting;
  std::vector<Seq> running;
  std::function<void(const Event&)> emit;

  Drafter* d = nullptr;
  Mtp* mtp = nullptr;  // Flash-Next's own MTP head as the drafter (src/mtp.h); exclusive with d
  std::vector<int> last_prompt;  // per planned sequence: its last row completes a greedy prompt (MTP fold)
  bool debug = getenv("INFERD_DEBUG") != nullptr;
  std::vector<std::pair<int, int>> spec_blocks = parse_blocks(getenv("SPEC_BLOCKS") ? getenv("SPEC_BLOCKS") : "");
  static std::vector<std::pair<int, int>> parse_blocks(const std::string& v) {
    // "8:4,16:3,32:2": more than 8 decoding -> block 4, more than 16 -> 3, ...
    std::vector<std::pair<int, int>> r;
    size_t i = 0;
    while (i < v.size()) {
      size_t c = v.find(':', i), e = v.find(',', i);
      if (c == std::string::npos) break;
      r.push_back({atoi(v.substr(i, c - i).c_str()), atoi(v.substr(c + 1, e - c - 1).c_str())});
      if (e == std::string::npos) break;
      i = e + 1;
    }
    return r;
  }
  uint64_t stat_cycles = 0, stat_committed = 0, stat_preempted = 0;
  uint64_t stat_grammar_fix = 0, stat_grammar_rollback = 0;  // structured output: replaced block tokens, re-prefills
  // COPY_DRAFT (MTP path, greedy sequences; 0 off): context-copy chains of up to COPY_K drafts (n-gram COPY_NG) replace
  // the MTP head's drafts when longer than them and (COPY_CHECK) their first token is the MTP head's own d1.
  // Default (since 3 Nov): any number of decoders (engine-pf gates, Studio, Flash-Next: wall.py decode at N=4 309 -> 341,
  // N=8 444 -> 473 tok/s; conc 16k x8/16/32 483.6/796.6/1033.6 vs 499.1/739.9/1006.0; exactness vs plain greedy
  // decoding as MTP alone, see the study log). COPY_DRAFT=1: any load (as the default); COPY_SOLO=1: only while a single sequence
  // decodes (the 2 Nov default); 0: off.
  int copy_on = getenv("COPY_DRAFT") ? atoi(getenv("COPY_DRAFT")) : 1;
  bool copy_solo = getenv("COPY_SOLO") && atoi(getenv("COPY_SOLO"));
  int copy_k = (int)env_knob("COPY_K", 11, 0, kMaxBlock - 1);
  int copy_check = getenv("COPY_CHECK") ? atoi(getenv("COPY_CHECK")) : 1;
  int copy_trust = getenv("COPY_TRUST") ? atoi(getenv("COPY_TRUST")) : 1 << 30;  // match length to use without d1
  uint64_t stat_copy_cycles = 0, stat_copy_committed = 0;
  CopyPolicy copy_pol;  // COPY_DRAFT=1: adaptive chain length (tokens per ms); COPY_DRAFT=2: always the longest chain
  int prefill_step = 0;  // see step(); set by the server (PREFILL_STEP)
  // From the agent-contexts prototype (engine-ctx-prefill-policy.patch, 2 Nov): PREFILL_SHORT orders the prefill
  // budget (prompts with at most this many tokens left first), PREFILL_PRIORITY orders running work by priority.
  int prefill_short = getenv("PREFILL_SHORT") ? atoi(getenv("PREFILL_SHORT")) : 2048;  // see step()
  bool prefill_priority = !getenv("PREFILL_PRIORITY") || atoi(getenv("PREFILL_PRIORITY"));  // see step()
  // PREFILL_PRIO (2026-11-02, engine-pf): while prompts are still prefilling and at most this many sequences decode,
  // a step takes the full prefill budget (max(step_tokens, PREFILL_STEP) prompt rows) on top of the decoders' rows,
  // instead of the decoders' rows coming out of step_tokens: users released together reach their first token sooner
  // and the prompts run at the large-step rate. 0: off (the decoders' rows come out of step_tokens). Default 16 since
  // 3 Nov (Studio wall.py, 14k prompts + 6,400 tokens, 2 trials: Flash-Next N=8 last first token 32.5 -> 29.5 s, total
  // 149.2 -> 144.9 s, N=4 99.4 -> 98.2; 27B with PF8 neutral within the spread (N=4 70.0 -> 72.9, N=8 123.0 / 175.1
  // -> 123.8 / 122.6 per trial); conc 16k x 8/16/32 level for both).
  int prefill_prio = getenv("PREFILL_PRIO") ? atoi(getenv("PREFILL_PRIO")) : 16;
  // PREFILL_MIN: prompt rows every step keeps for prefill even when the decoders' rows use up step_tokens (a floor:
  // without it a step with many wide verify blocks can starve new prompts). 0: off.
  int prefill_min = getenv("PREFILL_MIN") ? atoi(getenv("PREFILL_MIN")) : 0;
  // ADMIT_LOOKAHEAD (Codex S05 / R11): requests behind a head that cannot be admitted (KV budget, slot, fan-out leader)
  // may be admitted if they fit; one bypassed ADMIT_AGE times stops further bypassing (the starvation bound). Default 8
  // since 3 Nov: level on conc 16k x 8/16/32 for both models and on mixed arrivals with small KV pools; 0 off.
  int admit_lookahead = getenv("ADMIT_LOOKAHEAD") ? atoi(getenv("ADMIT_LOOKAHEAD")) : 8;
  int admit_age = getenv("ADMIT_AGE") ? atoi(getenv("ADMIT_AGE")) : 4;
  SpillTier* spill = nullptr;  // SSD tier (SPILL_DIR), or none
  // Most decoders that still get the full (8-token) block. 16k follow-ups, block 8 vs 5: N=4 339 vs 314 tok/s,
  // N=8 405 vs 445, N=16 450 vs 515 (real agent text accepts less than the spec_test prompts suggest).
  int spec_wide = getenv("SPEC_WIDE") ? atoi(getenv("SPEC_WIDE")) : 4;
  // Blocks past 8 (12, 16: 2026-11-02) only while decoders x block stays within spec_wide_rows verify rows: one stream
  // verifies 16 rows for about the price of 8 (the linears are weight-bandwidth-bound and W4A8 runs 16-row tiles), so
  // block 16 pays alone and for a few streams; fleets keep 4-8. The fixed rule (SPEC_AUTO=0) uses the widest such block.
  int spec_wide_rows = (int)env_knob("SPEC_WIDE_ROWS", 32, 1, 1 << 16);
  int max_block_for(int n_dec) const {
    int L = d ? d->block : 1;
    while (L > 8 && n_dec * L > spec_wide_rows) L = L > 12 ? 12 : 8;
    return L;
  }
  // The most verify rows the policy above (and token trees) can plan in one step for up to max_seqs decoders with a
  // drafter of widest block `block`, counted at the drafter attention's row instance (5/8/12/16). The server sizes
  // the verify scratch, the GDN tapes and the drafter's block scratch with it (Qwen35::enable_spec): 16 sequences
  // need 128 rows (16 x 8: the tuner may pick 8 for a fleet), not 16 x 16. SPEC_BLOCKS overrides the policy, so
  // then every sequence at the widest block. step() clamps chain blocks to it (m.spec_rows) as a guard.
  static int spec_rows_cap(int max_seqs, int block) {
    if (getenv("SPEC_BLOCKS") && *getenv("SPEC_BLOCKS")) return max_seqs * kMaxBlock;
    // Launch knobs that size the verify scratch are range-checked (Codex review 2026-11-03, R03).
    const int wide = (int)env_knob("SPEC_WIDE_ROWS", 32, 1, 1 << 16);
    auto inst = [](int L) { return L <= 5 ? 5 : L <= 8 ? 8 : L <= 12 ? 12 : 16; };
    int cap = 0;
    for (int n = 1; n <= max_seqs; n++) {
      int L = block;
      while (L > 8 && n * L > wide) L = L > 12 ? 12 : 8;
      cap = std::max(cap, n * inst(L));
    }
    const int tn = (int)env_knob("TREE", 16, 0, kMaxTree);
    const int ts = std::min(max_seqs, (int)env_knob("TREE_SEQS", 1, 0, 1 << 16));
    if (tn > 1) cap = std::max(cap, ts * std::max(tn, inst(block)));
    return std::min(cap, max_seqs * kMaxBlock);
  }

  // Self-tuning speculation width (SPEC_AUTO=0: the fixed rule above). The best block depends on load and context
  // (2026-10-31, Studio, 8-bit KV: 8 agents x 16k favour block 4 or 8 equally, x 128k block 8 by ~10%; 16 x 16k
  // block 4 by ~10%) and on how well the drafter reads the current text, so neither is fixed: acceptance is an EMA
  // of "the first k drafts were all accepted" per position (from every greedy cycle), giving E(L) committed tokens
  // per cycle; the cost is the measured wall time of decode-only steps per regime (decoders x mean context) and
  // block. Each regime tries every candidate a few times, then takes the best n * E(L) / T(L), re-testing one
  // other candidate every 16th step.
  struct BlockTuner {
    static constexpr int kC = 5;
    int far_every = 1;  // set to 4 without neural accelerators (see choose)
    const int cand[kC] = {4, 5, 8, 12, 16};
    double hits[kMaxBlock] = {}, trials[kMaxBlock] = {};
    struct Cell { double t[kC] = {}; int n[kC] = {}; uint64_t steps = 0; };
    std::unordered_map<int, Cell> cells;
    static int key(int n_dec, double ctx) {
      const int nb = n_dec <= 2 ? n_dec : n_dec <= 4 ? 4 : n_dec <= 6 ? 6 : n_dec <= 8 ? 8 : n_dec <= 12 ? 12 :
                     n_dec <= 16 ? 16 : n_dec <= 24 ? 24 : n_dec <= 32 ? 32 : 64;
      const int cb = ctx < 12e3 ? 0 : ctx < 24e3 ? 1 : ctx < 48e3 ? 2 : ctx < 96e3 ? 3 : ctx < 192e3 ? 4 : 5;
      return nb * 8 + cb;
    }
    double expect(int L) const {  // committed tokens per cycle at block L (the bonus token included)
      double e = 1;
      for (int k = 1; k < L; k++) e += trials[k] > 4 ? hits[k] / trials[k] : std::pow(0.75, k);
      return e;
    }
    // Index into cand. A regime starts on the fixed rule's block (dflt) and explores on one step in 8 until every
    // candidate has 3 timings, then re-tests one other candidate every 16th step: agents joining and leaving keep
    // moving the fleet between regimes, and forced exploration in each one cost ~10% of a mini's fleet throughput.
    int choose(int k, int n_dec, int maxL, int dflt) {
      Cell& c = cells[k];
      c.steps++;
      int best = -1, least = -1;
      double bs = 0;
      for (int i = 0; i < kC; i++) {
        if (cand[i] > maxL) continue;
        if (least < 0 || c.n[i] < c.n[least]) least = i;
        if (c.n[i] == 0) continue;
        const double sc = n_dec * expect(cand[i]) / c.t[i];
        if (best < 0 || sc > bs) best = i, bs = sc;
      }
      if (least >= 0 && c.n[least] < 3) {
        if (c.steps % 8 == 0) return least;
        if (best < 0)  // nothing timed yet: the fixed rule's block
          for (int i = 0; i < kC; i++)
            if (cand[i] == dflt) return i;
        return best >= 0 ? best : least;
      }
      if (best >= 0 && c.steps % 16 == 0) {  // re-test another candidate
        const int alt = (best + 1 + (int)(c.steps / 16) % (kC - 1)) % kC;
        // far_every (no neural accelerators: 4): a candidate scoring under 75% of the best is re-tested only on every
        // 4th turn. M3 Ultra, 16 x 16k: blocks 4/5/8 time 277/375/510 ms per step, so re-testing 5 and 8 on one step
        // in 16 cost ~4% of throughput (auto 159.6 vs block 4 forced 170.4 tok/s).
        const bool far = c.n[alt] > 0 && n_dec * expect(cand[alt]) / c.t[alt] < 0.75 * bs;
        if (cand[alt] <= maxL && (!far || (c.steps / 16) % far_every == 0)) return alt;
      }
      return best < 0 ? 0 : best;
    }
    void accepted(int L, int c) {  // one greedy cycle of block L committed c tokens
      for (int k = 1; k < L; k++) {
        trials[k] = trials[k] * 0.999 + 1;
        hits[k] = hits[k] * 0.999 + (c - 1 >= k);
      }
    }
    void timed(int k, int i, double ms) {
      Cell& c = cells[k];
      c.t[i] = c.n[i] ? 0.8 * c.t[i] + 0.2 * ms : ms;
      c.n[i]++;
    }
  } tuner;
  bool spec_auto = !getenv("SPEC_AUTO") || atoi(getenv("SPEC_AUTO"));
  // Token trees (TREE=<nodes>, up to kMaxBlock; src/tree.h): with at most tree_seqs greedy decoders, each verifies
  // a draft tree built on the host from the drafter's per-position candidates instead of a chain. Fleets keep chains.
  // Default 16 nodes (TREE=0: chains only). Single stream, Studio, fixed prompts: 213 -> 231 tok/s (2026-11-02).
  int tree_n = (int)env_knob("TREE", 16, 0, kMaxTree);
  // Always used while tree_ok (trees as SPEC_AUTO candidates measured worse: 218 vs 231 tok/s single stream, the
  // exploration of chain blocks cost more than it found).
  int tree_seqs = (int)env_knob("TREE_SEQS", 1, 0, 1 << 16);
  TreeParams tree_params = TreeParams::from_env(tree_n);
  bool tree_gpu = getenv("TREE_GPU") && atoi(getenv("TREE_GPU"));  // build the trees on the GPU (no host round trip)
  uint64_t tree_cycles = 0, tree_committed = 0, tree_hist[kMaxBlock + 1] = {};

  // Write the chain ending at node `id` (which carries a snapshot) to the SSD tier, if it is not there already. Returns
  // whether the chain is on the tier afterwards (Codex review 2026-11-03, R05): whatever acknowledges persistence (a
  // context saved with "ssd", an offload) checks it, and a session learns its entry (Session::ssd) only once it exists.
  bool spill_chain(int id, int prio = 0) {
    if (!spill) return false;
    {
      const auto& t = cache.nodes.at(id);
      if (t.snapshot == -1) return false;  // no state to write (a session mark whose snapshot was dropped)
      if (t.pin_snap || t.sess) prio = std::max(prio, 1);  // named contexts and sessions outlive anonymous prefixes
    }
    std::vector<int> chain;
    for (int x = id; x >= 0; x = cache.nodes.at(x).parent) chain.push_back(x);
    std::reverse(chain.begin(), chain.end());
    std::vector<int> tokens, blocks;
    uint64_t h = SpillTier::kSeed;
    for (int x : chain) {
      auto& n = cache.nodes.at(x);
      tokens.insert(tokens.end(), n.tokens.begin(), n.tokens.end());
      blocks.push_back(n.block);
      h = SpillTier::extend(h, n.tokens.data());
    }
    size_t before = spill->n_spilled;
    const bool ok = spill->spill(h, tokens, blocks, std::max(-1, cache.nodes.at(id).snapshot), prio);
    if (!ok)
      fprintf(stderr, "spill: FAILED to write a %zu-block chain (%.1f GiB used of %.1f)\n", blocks.size(),
              spill->used / 1073741824.0, spill->quota / 1073741824.0);
    if (spill->n_spilled != before)
      fprintf(stderr, "spill: wrote %zu blocks, %.2f s total so far, %.1f GiB on disk\n", blocks.size(), spill->spill_s,
              spill->used / 1073741824.0);
    if (ok && cache.nodes.at(id).sess)  // every session whose latest snapshot this is (R06: not only the last to mark it)
      for (auto& [name, se] : sessions)
        if (se.node == id) se.ssd = h;
    return ok;
  }

  // Write every cached snapshot chain the SSD tier lacks (shutdown with SPILL_PERSIST). Returns how many it wrote.
  size_t spill_all() {
    if (!spill) return 0;
    size_t before = spill->n_spilled;
    std::vector<int> ids;
    for (auto& [id, n] : cache.nodes)
      if (n.snapshot != -1) ids.push_back(id);
    for (int id : ids) spill_chain(id);
    return spill->n_spilled - before;
  }

  // Bring back the deepest spilled prefix of s.tokens that is deeper than what the trie holds with a snapshot
  // (`ram_depth`), re-entering its blocks in the trie with a snapshot at its end. `path` = trie nodes matched so far.
  void restore_spilled(Seq& s, const std::vector<int>& path, int ram_depth) {
    int P = (int)s.tokens.size(), maxd = (s.req.prefill_only ? P : P - 1) / KV_BLOCK;
    uint64_t h = 0;
    const SpillTier::Entry* e = spill->lookup(s.tokens, ram_depth, maxd, &h);
    static const bool sdebug = getenv("SPILL_DEBUG") != nullptr;
    if (!e && sdebug && P - ram_depth * KV_BLOCK > 2048) {
      // A big miss: how deep do the trie's KV and the tier's entries go along this prompt (any depth)?
      int deepest = 0;
      uint64_t hh = SpillTier::kSeed;
      for (int dpt = 1; dpt <= maxd; dpt++) {
        hh = SpillTier::extend(hh, s.tokens.data() + (size_t)(dpt - 1) * KV_BLOCK);
        if (spill->entries.count(hh)) deepest = dpt;
      }
      fprintf(stderr, "spill-miss id%llu: %d tokens, trie KV %zu blocks, trie snapshot %d, tier deepest %d, %zu entries, %d snapshots\n",
              (unsigned long long)s.req.id, P, path.size(), ram_depth, deepest, spill->entries.size(), snapshot_count);
    }
    if (!e) return;
    const int D = e->depth;
    spill->pinned = h;  // allocation below may evict and spill other chains: keep this entry and its files
    int keep = std::min((int)path.size(), D);  // blocks already in memory (shared prefix)
    // Pin the deepest kept node so eviction cannot take it while we allocate.
    if (keep > 0) cache.nodes[path[keep - 1]].children++;
    std::vector<int> fresh;
    bool ok = true;
    for (int i = keep; i < D && ok; i++) {
      int b = alloc_block(i % 4 ? (fresh.empty() ? (keep > 0 ? cache.nodes[path[keep - 1]].block + 1 : -1) : fresh.back() + 1) : -1);
      if (b < 0) ok = false; else fresh.push_back(b);
    }
    int slot = -1;
    if (ok && !stateless && (free_slots.empty() && !evict(0, 1))) ok = false;
    if (ok && !stateless) { slot = free_slots.back(); free_slots.pop_back(); }
    if (keep > 0) cache.nodes[path[keep - 1]].children--;
    std::vector<int> blocks(keep, -1);
    blocks.insert(blocks.end(), fresh.begin(), fresh.end());
    const bool got_blocks = ok && (slot >= 0 || stateless);
    if (ok) {
      auto it = spill->entries.find(h);  // the lookup pointer may be stale after the evictions above
      ok = it != spill->entries.end() && spill->load(it->second, blocks, slot);
    }
    spill->pinned = 0;
    if (!ok) {
      fprintf(stderr, "spill: restore FAILED for id%llu (%s; %d of %d blocks fresh, %zu free blocks, %zu free slots)\n",
              (unsigned long long)s.req.id, got_blocks ? "read" : slot < 0 && fresh.size() == (size_t)(D - keep) ? "no slot" : "no blocks",
              D - keep, D, n_free, free_slots.size());
      for (int b : fresh) unref_block(b);
      if (slot >= 0) free_slots.push_back(slot);
      return;
    }
    int parent = keep > 0 ? path[keep - 1] : -1;
    for (int i = keep; i < D; i++) {
      int id = cache.insert(parent, s.tokens.data() + (size_t)i * KV_BLOCK, fresh[i - keep]);
      cache.nodes[id].last_used = ++clock;
      parent = id;
    }
    auto& node = cache.nodes[parent];
    if (stateless) {
      if (node.snapshot == -1) node.snapshot = kNoState;
    } else {
      if (node.snapshot >= 0) { free_slots.push_back(node.snapshot); snapshot_count--; }
      node.snapshot = slot;
      snapshot_count++;
    }
    fprintf(stderr, "spill: restored %d blocks (%d already in memory) for id%llu, %.2f s total so far\n", D, keep,
            (unsigned long long)s.req.id, spill->restore_s);
  }

  Scheduler(Qwen35& m, Gpu& g, int max_running, int step_tokens, int snap_margin = 16, Drafter* d = nullptr,
            Mtp* mtp = nullptr)
      : m(m), g(g), max_running(max_running), step_tokens(step_tokens), snap_margin(snap_margin), d(d), mtp(mtp) {
    if (d) {
      m.enable_spec(d->target_layers, d->block);
      m.verify = true;
    } else if (mtp) {
      m.enable_spec({}, mtp->block);
      m.verify = true;
    }
    stateless = m.conv_state.empty() && !m.ple_state.buf && !d && !mtp && (!getenv("PREFIX_ANY") || atoi(getenv("PREFIX_ANY")));
    block_ref.assign(m.n_blocks, 0);
    is_free.assign(m.n_blocks, 0);
    grp_free.assign(m.n_blocks / 4 + 1, 0);
    for (int b = m.n_blocks - 1; b >= 0; b--) push_free(b);
    for (int s = m.n_slots - 1; s >= 0; s--) free_slots.push_back(s);
  }

  // Image rows: prompt token ids in [vocab, vocab + kImageKeys) are content keys (frontend/vision.py image_keys) whose
  // embeddings are the next rows of Request::embeds.
  static constexpr int64_t kImageKeys = 1 << 24;
  // The native request contract (Codex review 2026-11-03, R03), checked before anything is scheduled: the Unix socket
  // is reachable without the HTTP front end. Text token ids index the embedding table (embed4 casts the id to size_t),
  // so they must lie in [0, vocab); image key ids need exactly one embeds row each, and M-RoPE positions cover the
  // prompt exactly. Returns "" or the reason ("bad request: ..."; the front end answers those with HTTP 400).
  std::string invalid(const Request& r) const {
    if (r.prompt.empty()) return "bad request: empty prompt";
    const int V = m.cfg.vocab, H = m.cfg.hidden;
    size_t images = 0;
    for (int t : r.prompt) {
      if (t >= 0 && t < V) continue;
      if (t < V || (int64_t)t >= (int64_t)V + kImageKeys) return "bad request: token id " + std::to_string(t) + " out of range";
      images++;
    }
    if (images && r.prefill_only) return "error: contexts with images are not supported";
    if (r.embeds.size() != images * (size_t)H)
      return "bad request: embeds hold " + std::to_string(r.embeds.size() / H) + (r.embeds.size() % H ? "+" : "") +
             " rows of hidden size for " + std::to_string(images) + " image tokens";
    if (!r.mrope.empty() && r.mrope.size() != 3 * r.prompt.size()) return "bad request: mrope needs 3 positions per prompt token";
    if (r.max_tokens < 0 || r.max_context < 0 || r.top_k < 0) return "bad request: negative max_tokens / max_context / top_k";
    if (!std::isfinite(r.temperature) || r.temperature < 0 || !std::isfinite(r.top_p) || r.top_p < 0 || r.top_p > 1)
      return "bad request: temperature must be >= 0 and top_p in [0, 1]";
    if (r.ctx_ssd && !spill) return "error: persist ssd needs the SSD tier (SPILL_DIR)";  // R05: never acknowledged
    return "";
  }

  void submit(Request r) {
    if (std::string why = invalid(r); !why.empty()) {
      emit(Event{r.id, {}, true, why.rfind("error: ", 0) == 0 ? why : "error: " + why, (int)r.prompt.size(), 0, 0});
      return;
    }
    Seq s;
    s.req = std::move(r);
    s.tokens = s.req.prompt;
    if (!s.req.embeds.empty()) {  // one row per image key id, in order (invalid() checked the counts)
      int k = 0;
      for (int t : s.tokens) s.ext_idx.push_back(t >= m.cfg.vocab ? k++ : -1);
    }
    s.rng.seed(s.req.has_seed ? s.req.seed : s.req.id * 0x9E3779B97F4A7C15ull);  // seed 0 is a seed (Codex F05)
    if (s.req.prefill_only) {  // a saved context: whole KV blocks only (the tail is recomputed by each launch)
      s.tokens.resize(s.tokens.size() / KV_BLOCK * KV_BLOCK);
      s.req.prompt = s.tokens;
      s.req.max_tokens = 0;
      s.req.max_context = 0;
      if (s.tokens.empty() || !s.req.embeds.empty()) {
        emit(Event{s.req.id, {}, true, s.tokens.empty() ? "error: a context needs at least one KV block (32 tokens)"
                                                        : "error: contexts with images are not supported", 0, 0, 0});
        return;
      }
    }
    if (s.req.prefill_only || s.req.no_cache) s.req.gen_snap = false;
    if (s.req.max_context > 0) {
      const int room = s.req.max_context - (int)s.tokens.size();
      if (room <= 0) {
        emit(Event{s.req.id, {}, true, "error: prompt exceeds max_context", (int)s.tokens.size(), 0, 0});
        return;
      }
      s.req.max_tokens = std::min(s.req.max_tokens, room);
    }
    // Sampled requests are speculated too (exact speculative sampling on the GPU, spec_accept_s) unless SPEC_SAMPLE=0.
    static const bool spec_sample = !getenv("SPEC_SAMPLE") || atoi(getenv("SPEC_SAMPLE"));
    s.spec_ok = (d || mtp) && (s.req.temperature <= 0 || spec_sample) &&
                !(s.req.grammar && s.req.grammar->active && s.req.temperature > 0);  // see Request::grammar
    if (debug) fprintf(stderr, "[%.3f s] ", std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
    if (debug)
      fprintf(stderr, "submit id%llu: %zu prompt tokens, temperature %.2f, spec %d\n", (unsigned long long)s.req.id,
              s.tokens.size(), s.req.temperature, (int)s.spec_ok);
    waiting.push_back(std::move(s));
  }
  bool idle() const { return waiting.empty() && running.empty(); }

  void cancel(uint64_t id) {
    for (auto it = waiting.begin(); it != waiting.end(); ++it)
      if (it->req.id == id) {
        emit(Event{id, {}, true, "cancelled", (int)it->req.prompt.size(), 0, 0});
        waiting.erase(it);
        return;
      }
    for (size_t i = 0; i < running.size(); i++)
      if (running[i].req.id == id) { finish(i, "cancelled"); return; }
  }

  void unref_block(int b) {
    if (--block_ref[b] == 0) push_free(b);
  }
  // Remove node `id` and everything below it from the trie (their blocks back to the pool, snapshot slots freed).
  void drop_subtree(int id) {
    if (!cache.nodes.count(id)) return;
    std::vector<int> kids;
    for (auto& [k, n] : cache.nodes)
      if (n.parent == id) kids.push_back(k);
    for (int k : kids) drop_subtree(k);
    PrefixCache::Node& n = cache.nodes[id];
    if (n.snapshot >= 0) { free_slots.push_back(n.snapshot); snapshot_count--; }
    unref_block(n.block);
    cache.erase(id);
  }

  bool redundant_evict = !getenv("SNAP_REDUNDANT") || atoi(getenv("SNAP_REDUNDANT"));  // see redundant_victim
  // LRU order with agent sessions' latest snapshots after everything else (pinned contexts are never candidates).
  static uint64_t evict_key(const PrefixCache::Node& n) { return n.last_used + (n.sess ? 1ull << 62 : 0); }
  // Evict least-recently-used trie leaves until `blocks` blocks and `slots` slots are free.
  // `keep`: a node whose snapshot the caller is about to use (never the victim, nor its snapshot dropped).
  bool evict(size_t blocks, size_t slots, int keep = -1) {
    while (n_free < blocks || free_slots.size() < slots) {
      int victim = -1;
      uint64_t best = UINT64_MAX;
      bool need_slot = free_slots.size() < slots;
      if (need_slot && redundant_evict) {
        if (int r = redundant_victim(keep); r >= 0) {
          spill_chain(r);
          free_slots.push_back(cache.nodes[r].snapshot);
          cache.nodes[r].snapshot = -1;
          snapshot_count--;
          continue;
        }
      }
      for (auto& [id, n] : cache.nodes) {
        if (n.children || n.pin || id == keep || pending_nodes.count(id)) continue;
        if (need_slot && n.snapshot < 0 && n_free >= blocks) continue;
        if (evict_key(n) < best) { best = evict_key(n); victim = id; }
      }
      if (victim < 0) {
        // No leaf to free; drop the oldest snapshot anywhere when a slot is what's missing.
        if (!need_slot) return false;
        for (auto& [id, n] : cache.nodes)
          if (n.snapshot >= 0 && !n.pin_snap && id != keep && !pending_nodes.count(id) && evict_key(n) < best) { best = evict_key(n); victim = id; }
        if (victim < 0) return false;
        // Inner nodes too: an agent's turn start becomes one as soon as its next turn extends it, and a later prompt
        // that resumes there (a retry, a fork, a fleet restarting from its primed contexts) would re-prefill it all.
        // Content-addressed, so this writes the state plus whatever blocks the tier lacks. (Leaves only until
        // 2026-10-30: M5 Pro 16k 16:16 after other cells re-prefilled 192k tokens, 539 s vs 211 from fresh.)
        spill_chain(victim);
        free_slots.push_back(cache.nodes[victim].snapshot);
        cache.nodes[victim].snapshot = -1;
        snapshot_count--;
        if (cache.nodes[victim].sess) {  // now on SSD only (Session::ssd), for every session that marked it (R06)
          cache.nodes[victim].sess = 0;
          for (auto& [name, se] : sessions)
            if (se.node == victim) se.node = -1;
        }
        continue;
      }
      PrefixCache::Node& n = cache.nodes[victim];
      if (n.snapshot != -1) spill_chain(victim);
      if (n.snapshot >= 0) { free_slots.push_back(n.snapshot); snapshot_count--; }
      unref_block(n.block);
      cache.erase(victim);
    }
    return true;
  }

  // A free block, preferring `prefer` when it is free.
  // With the pool full, evict a batch of blocks at once (EVICT_BATCH, 64), not one per allocation. One at a time,
  // each eviction freed the next block down the oldest chain, so a new sequence got its pages in descending order
  // (X, X-1, X-2, ...) and attn_mpp never found a contiguous 4-page group: 12-22% slower verify steps at 64k
  // (decode_test BENCH_SHUFFLE) and 64k x 4/8 decode 305/352 -> 263/287 tok/s after a 128k run filled the pool.
  // A batch frees a run whose lowest block is popped first, so the hints walk it upwards.
  int evict_batch = (int)env_knob("EVICT_BATCH", 64, 1, 1 << 24);
  int alloc_block(int prefer = -1) {
    if (n_free == 0) {
      evict(std::min<size_t>(evict_batch, m.n_blocks), 0);  // as many as it can
      if (n_free == 0) return -1;
    }
    int b = prefer >= 0 && prefer < m.n_blocks && is_free[prefer] ? prefer : -1;
    while (b < 0 && prefer < 0 && !free_groups.empty()) {  // a group start: take a whole free group
      int gi = free_groups.back();
      free_groups.pop_back();
      if (grp_free[gi] == 4) b = gi * 4;
    }
    while (b < 0) {
      int c = free_blocks.back();
      free_blocks.pop_back();
      if (is_free[c]) b = c;
    }
    is_free[b] = 0;
    n_free--;
    grp_free[b / 4]--;
    block_ref[b] = 1;
    return b;
  }
  // The block to prefer for logical page i of a sequence: right after page i - 1, within groups of 4.
  static int next_block_hint(const std::vector<int>& blocks) {
    return !blocks.empty() && blocks.size() % 4 ? blocks.back() + 1 : -1;
  }

  int fanout_min = getenv("FANOUT_MIN") ? atoi(getenv("FANOUT_MIN")) : 1024;  // shared tokens worth waiting for
  bool wait_for_leader(const Seq& s, int P, int cached_blocks) {
    for (auto& r : running) {
      if (!r.prefilling()) continue;
      const int rp = (int)r.tokens.size();
      int c = 0;
      while ((c + 1) * KV_BLOCK < P && (c + 1) * KV_BLOCK <= rp &&
             std::equal(s.tokens.begin() + c * KV_BLOCK, s.tokens.begin() + (c + 1) * KV_BLOCK, r.tokens.begin() + c * KV_BLOCK))
        c++;
      const int ct = c * KV_BLOCK;
      if (ct < fanout_min || c <= cached_blocks + 2 || r.computed >= ct) continue;
      r.add_snap(ct);
      if (debug) fprintf(stderr, "fan-out: id%llu waits for id%llu to snapshot at %d\n", (unsigned long long)s.req.id,
                         (unsigned long long)r.req.id, ct);
      return true;
    }
    return false;
  }

  // Branch points: the trie's KV for a prompt only resumes where a GDN state snapshot exists, and snapshots are taken
  // near prompt ends. Two agents with one system prompt and different questions shared 0 cached tokens (examples/,
  // 2026-11-01). A sequence whose KV match runs >= SNAP_BRANCH_MIN tokens past its deepest snapshot snapshots at the
  // branch point (the next sibling resumes there), and front-end hints of at least that many tokens are honoured.
  int branch_min = (int)env_knob("SNAP_BRANCH_MIN", 256, 0, 1 << 30);  // 0: off
  // Decode snapshots (2026-11-02): a sequence whose request has gen_snap (the front end sets it with thinking off)
  // snapshots while it decodes: whenever a step ends on a block boundary at least SNAP_GEN / 2
  // positions after its last one, and (SNAP_GEN_FORCE) optionally by shortening blocks to land every SNAP_GEN. Before this, a turn's
  // own answer was never cached: the next turn resumed at the previous prompt's end (or, once that snapshot was
  // evicted, the document's) and re-read the answer and tool output (M5 Pro, 16 agents x 4k, 250-word answers: 31%
  // of follow-up prompt tokens cached with 32 snapshots).
  int snap_gen = (int)env_knob("SNAP_GEN", 128, 0, 1 << 24) / KV_BLOCK * KV_BLOCK;
  // Positions to the next decode snapshot boundary for s (0: none).
  // Landing on the boundary (shortening the block that would cross it) only with SNAP_GEN_FORCE=N > 0 and N or more
  // decoders. Off by default: it cost a lone stream 2-6% (M5 Pro fixed prompts, 4k: 76.0 -> 71.6 tok/s) and the
  // Studio fleet 4-6% (conc 16k x16: 571 -> 535 tok/s at 128, 549 at 256), while snapshots taken only when a step
  // happens to end on a block boundary (at most every SNAP_GEN / 2 positions) cost nothing measurable (569 / 577).
  int snap_force = getenv("SNAP_GEN_FORCE") ? atoi(getenv("SNAP_GEN_FORCE")) : 0;
  int cur_ndec = 0;  // decoders in the step being planned
  int gen_need(const Seq& s) const {
    if (!s.req.gen_snap || snap_gen <= 0 || !s.decoding() || snap_force <= 0 || cur_ndec < snap_force) return 0;
    return (s.computed / snap_gen + 1) * snap_gen - s.computed;
  }
  // A snapshot made redundant by a deeper one on the same unbranched chain (an agent's earlier turn ends, or the end
  // of its document once a turn has extended it). When a slot is needed these go first, least recently used first:
  // LRU over leaves took agents' newest snapshots (leaves) while their older inner ones survived, so follow-ups
  // resumed from the document's end. Branch points (2+ children) are never redundant: other agents share them.
  int redundant_victim(int keep) {
    int victim = -1;
    uint64_t best = UINT64_MAX;
    for (auto& [id, n] : cache.nodes) {
      if (n.snapshot < 0) continue;
      for (int x = n.parent; x >= 0;) {
        auto& a = cache.nodes.at(x);
        if (a.children != 1) break;
        if (a.snapshot >= 0 && !a.pin_snap && !a.sess && x != keep && !pending_nodes.count(x) && a.last_used < best) { best = a.last_used; victim = x; }
        x = a.parent;
      }
    }
    return victim;
  }

  int admit_reserve = (int)env_knob("ADMIT_RESERVE", 512, 0, 1 << 24);  // generated tokens to budget per turn
  bool fits(const Seq& s, const std::vector<int>& path, int depth) {
    long avail = (long)n_free;
    for (auto& [id, n] : cache.nodes)
      if (block_ref[n.block] == 1 && !n.pin) avail++;  // held by the trie alone (pinned contexts stay)
    for (int i = 0; i < depth; i++)
      if (block_ref[cache.nodes[path[i]].block] == 1 && !cache.nodes[path[i]].pin) avail--;  // the prefix this sequence will reuse
    long need = std::max(0, s.target_blocks(admit_reserve) - depth);
    for (auto& r : running) need += r.target_blocks(admit_reserve) - (long)r.blocks.size();
    return need <= avail;
  }

  // Preempt running sequence i when the pool runs dry: give back its blocks and slot and queue it first in line. Its
  // prompt's snapshot stays in the prefix cache (and goes to the SSD tier if evicted), so on re-admission it restores
  // that and recomputes only the tail: the prompt's last few tokens plus what it has generated so far.
  // The sequence to preempt among those not yet planned this step (indices after `planned`): lowest priority, and
  // among equals the newest (the old rule: the last one).
  size_t victim(size_t planned) const {
    size_t v = running.size() - 1;
    for (size_t j = running.size() - 1; j > planned; j--)
      if (running[j].req.priority < running[v].req.priority) v = j;
    return v;
  }
  void preempt(size_t i) {
    Seq s = std::move(running[i]);
    running.erase(running.begin() + i);
    if (debug || getenv("SPILL_DEBUG"))
      fprintf(stderr, "preempt id%llu at %d tokens (%d generated)\n", (unsigned long long)s.req.id, s.computed, s.generated);
    for (int b : s.blocks) unref_block(b);
    free_slots.push_back(s.slot);
    s.blocks.clear();
    s.slot = -1;
    s.pend.clear();
    s.mtp_fold = false;
    s.snap_pts.clear();
    if (s.generated > 0) {
      s.tokens.resize(s.computed + 1);  // drop draft placeholders; keep the staged token
      s.resumed = true;
    }
    s.computed = 0;
    stat_preempted++;
    waiting.push_front(std::move(s));
  }

  // Admit waiting requests: match the prefix cache, take a slot, plan the snapshot point.
  void admit(std::vector<std::pair<int, int>>& restores, std::vector<int>& zeros) {
    // ADMIT_LOOKAHEAD (engine-pf 2026-11-02, Codex review S05): when the request at the head cannot be admitted yet (KV
    // budget, a slot, or it waits for a fan-out leader), up to this many requests behind it may be admitted if they
    // fit; a waiting request bypassed ADMIT_AGE times blocks further bypassing (aging), so a large head is not starved.
    size_t qi = 0;
    auto blocked = [&]() {  // the candidate at qi cannot be admitted now: try the next one, or stop
      if (qi + 1 < waiting.size() && (int)qi < admit_lookahead) {
        for (size_t j = 0; j <= qi; j++) if (waiting[j].bypassed >= admit_age) return false;
        qi++;
        return true;
      }
      return false;
    };
    while (qi < waiting.size() && (int)running.size() < max_running) {
      Seq& s = waiting[qi];
      int P = (int)s.tokens.size();
      // The original contract, prompt + output limit (after a preemption P also holds the output so far: Codex R05),
      // against the exact limit (Codex R02: the block-rounded test admitted more than the block table holds).
      const long limit = ctx_limit > 0 ? std::min<long>(ctx_limit, (long)(m.max_blocks - 1) * KV_BLOCK) : (long)(m.max_blocks - 1) * KV_BLOCK;
      // A prompt that fits but whose max_tokens would pass the window gets the room that is left (generation stops
      // there with "length"), as llama.cpp and most OpenAI-style servers do: refusing it failed every feedback round of
      // languages-benchmark at a 30k cap in a 32k window (2026-11-02). CLAMP_MAX_TOKENS=0 restores the refusal.
      static const bool clamp = !getenv("CLAMP_MAX_TOKENS") || atoi(getenv("CLAMP_MAX_TOKENS")) != 0;
      if (clamp && (long)s.req.prompt.size() < limit && (long)s.req.prompt.size() + std::max(0, s.req.max_tokens) > limit)
        s.req.max_tokens = (int)(limit - (long)s.req.prompt.size());
      const long total = (long)s.req.prompt.size() + std::max(0, s.req.max_tokens);
      const char* why = total > limit ? "error: context too long"
                        : m.cfg.qwen4 && !m.qsa_on && !qwen4_long && total > qwen4_exact ? "error: context too long (Flash-Next with QSA=0: 2,048 tokens without the sparse indexer; FN_ALLOW_LONG=1 runs it inexactly)"
                        : nullptr;
      if (why) {
        emit(Event{s.req.id, {}, true, why, P, 0, 0});
        waiting.erase(waiting.begin() + qi);
        continue;
      }
      // Walk the trie over the prompt's full blocks (never the whole prompt: the last token must be computed
      // to produce logits).
      int best_node = -1, best_depth = 0;
      std::vector<int> path;
      auto walk = [&]() {
        int parent = -1, depth = 0;
        best_node = -1, best_depth = 0;
        path.clear();
        // (A saved context may match whole: it computes nothing and samples nothing.)
        while ((depth + 1) * KV_BLOCK < P + (int)s.req.prefill_only) {
          int id = cache.find(parent, s.tokens.data() + depth * KV_BLOCK);
          if (id < 0) break;
          path.push_back(id);
          parent = id;
          depth++;
          if (stateless || cache.nodes[id].snapshot >= 0) { best_node = id; best_depth = depth; }
        }
      };
      walk();
      // A context save whose chain is cached already (or was just restored from SSD): register / pin it, no compute.
      auto ctx_hit = [&] { return s.req.prefill_only && best_node >= 0 && best_depth * KV_BLOCK == P; };
      if (ctx_hit()) {
        s.cached = P;
        ctx_finish(s, best_node);
        waiting.erase(waiting.begin() + qi);
        continue;
      }
      // KV budget: admit only when the blocks this turn will need fit beside what the running sequences will still
      // take, counting free blocks plus prefix-cache blocks nobody is using (evictable, to the SSD tier when on).
      // Without it, admission was by count alone: on the 32 GB M6 (3 GiB pool, ~96k tokens) four 33k agents were
      // all admitted, each grew until the pool was empty, and no running sequence could free a block for another:
      // 0.02 tok/s and two-hour timeouts (2026-10-31). The rest wait, their contexts parked in the trie or on SSD.
      if (!running.empty() && !fits(s, path, best_depth)) { if (blocked()) continue; break; }
      // A slot for the sequence, sparing the snapshot it will start from. (Until 2026-10-30 these evictions could take
      // the matched or just-restored snapshot itself, often the only leaf with one: on the M5 Pro 16:16 fleet, agents
      // primed moments earlier re-prefilled their 16k contexts, 190k new tokens per cell instead of ~60k.)
      if (free_slots.empty()) {
        if (!evict(0, 1, best_node)) { if (blocked()) continue; break; }
        walk();
      }
      if (spill) {
        size_t before = spill->n_restored;
        restore_spilled(s, path, best_depth);
        if (spill->n_restored != before) walk();
        if (free_slots.empty()) {  // the restore took the last free slot: make room again, sparing what it restored
          if (!evict(0, 1, best_node)) { if (blocked()) continue; break; }
          walk();
        }
      }
      // Fan-out: when this prompt shares a long prefix with a sequence still prefilling it, have that leader
      // snapshot at the end of the shared part and wait for it, so the prefix is computed once (8 agents on one
      // 32k document: 188 s -> one prefill). FANOUT_MIN=0 turns it off.
      if (ctx_hit()) {
        s.cached = P;
        ctx_finish(s, best_node);
        waiting.erase(waiting.begin() + qi);
        continue;
      }
      if (fanout_min > 0 && wait_for_leader(s, P, best_depth)) { if (blocked()) continue; break; }
      s.slot = free_slots.back();
      free_slots.pop_back();
      const int was_cached = s.cached;
      if (best_node >= 0) {
        for (int i = 0; i < best_depth; i++) {
          auto& n = cache.nodes[path[i]];
          n.last_used = ++clock;
          block_ref[n.block]++;
          s.blocks.push_back(n.block);
        }
        if (cache.nodes[best_node].snapshot >= 0) restores.push_back({cache.nodes[best_node].snapshot, s.slot});
        s.computed = s.cached = best_depth * KV_BLOCK;
        if (s.resumed) s.cached = was_cached;  // report the first admission's hit, not the resume's
        if (!s.req.session.empty()) mark_session(s, best_node, best_depth);  // until its own turn snapshots deeper
      } else {
        zeros.push_back(s.slot);
      }
      static const bool sdebug = getenv("SPILL_DEBUG") != nullptr;
      if (sdebug)
        fprintf(stderr, "admit id%llu: %d tokens, cached %d, %d running, %zu free slots, %zu free blocks\n",
                (unsigned long long)s.req.id, P, s.computed, (int)running.size(), free_slots.size(), n_free);
      int sp = s.req.prefill_only ? P : (P - snap_margin) / KV_BLOCK * KV_BLOCK;
      if (sp > s.computed && !s.req.no_cache) s.add_snap(sp);
      if (branch_min > 0 && !s.req.no_cache) {
        const int kv = (int)path.size() * KV_BLOCK;  // the trie's KV runs this far, its state only to s.computed
        if (kv - s.computed >= branch_min && kv < sp) s.add_snap(kv);
        for (int h : s.req.snap_hints) {
          const int a = h / KV_BLOCK * KV_BLOCK;
          if (a >= branch_min && a > s.computed && a < sp) s.add_snap(a);
        }
      }
      for (size_t j = 0; j < qi; j++) waiting[j].bypassed++;
      running.push_back(std::move(s));
      waiting.erase(waiting.begin() + qi);
      qi = 0;  // the head again: what blocked it may have changed
    }
  }

  // After a step: snapshot sequences that just reached their snapshot point, and enter their blocks in the trie.
  void take_snapshots(Seq& s, std::vector<std::pair<int, int>>& snaps) {
    const int sa = s.next_snap();
    if (sa < 0 || s.computed != sa) return;
    s.snap_pts.erase(s.snap_pts.begin());
    snapshot_at(s, sa, snaps);
  }
  // Enter s's blocks up to block-aligned position sa (where its slot's state now is) in the trie and snapshot there:
  // a copy into a free slot (snaps, encoded by the caller), or with donate the sequence's own slot (it is finishing).
  // Returns true when the slot was donated.
  bool snapshot_at(Seq& s, int sa, std::vector<std::pair<int, int>>& snaps, bool donate = false) {
    int nb = sa / KV_BLOCK, parent = -1;
    for (int i = 0; i < nb; i++) {
      int id = cache.find(parent, s.tokens.data() + i * KV_BLOCK);
      if (id < 0) {
        id = cache.insert(parent, s.tokens.data() + i * KV_BLOCK, s.blocks[i]);
        block_ref[s.blocks[i]]++;
      }
      cache.nodes[id].last_used = ++clock;
      if (!donate) pending_nodes.insert(id);  // its KV / snapshot is written by this step's command buffer (Codex R01)
      parent = id;
    }
    if (parent < 0) return false;
    auto& node = cache.nodes[parent];
    if (stateless) {  // the blocks are the whole resume state: mark the point for the SSD tier, take no slot
      if (node.snapshot == -1) node.snapshot = kNoState;
      mark_session(s, parent, nb);
      return false;
    }
    if (node.snapshot >= 0) { mark_session(s, parent, nb); return false; }  // an identical prefix already has one
    if (donate) {
      node.snapshot = s.slot;
      snapshot_count++;
      mark_session(s, parent, nb);
      return true;
    }
    if (free_slots.empty() && !evict(0, 1, parent)) return false;
    if (!cache.nodes.count(parent)) return false;  // evicted while making room
    node.snapshot = free_slots.back();
    free_slots.pop_back();
    snapshot_count++;
    snaps.push_back({s.slot, node.snapshot});
    pending_snaps.push_back(parent);
    mark_session(s, parent, nb);
    return false;
  }
  // ---- Agent contexts and sessions (docs/CONTEXTS.md) ----
  // A context is a named, block-aligned token prefix computed once (prefill_only), with a snapshot at its end. Pinned,
  // its chain is exempt from eviction (KV blocks counted out of the admission budget, snapshot never dropped); with
  // the SSD tier it can also be written out (persist "ssd") or moved there (offload), and a launch restores it. A
  // session is an agent's name: the deepest snapshot its latest turn took is marked, and eviction takes every
  // unmarked prefix first (the SSD tier, when on, receives it before it leaves memory).
  struct Context {
    std::vector<int> tokens;  // whole KV blocks
    int node = -1;            // the pinned terminal node, when pinned
    bool pinned = false;
    int cached = 0;           // tokens that were already cached when it was saved
    uint64_t h = 0;           // its chain's prefix hash (the SSD tier's key)
  };
  std::unordered_map<std::string, Context> contexts;
  // A session's latest snapshot node (several sessions may mark the same node: Node::sess counts them) and the SSD
  // entry its latest snapshot was written to. Ownership is derived from these records (Codex review 2026-11-03, R06:
  // a node -> one session reverse map lost every owner but the last, and drops deleted entries others still used).
  struct Session { int node = -1, depth = 0; uint64_t ssd = 0, turns = 0; };
  std::unordered_map<std::string, Session> sessions;
  // Whether an SSD entry h (whose chain ends at trie node `node`, or -1: looked up from the entry) is owned by a named
  // context or session other than skip_ctx / skip_sess; drops delete an entry only when nobody else owns it (R06).
  bool ssd_shared(uint64_t h, int node, const std::string* skip_ctx, const std::string* skip_sess) const {
    for (auto& [n, c] : contexts)
      if ((!skip_ctx || n != *skip_ctx) && c.h == h) return true;
    if (node < 0 && spill)
      if (auto it = spill->entries.find(h); it != spill->entries.end()) node = find_chain(it->second.tokens, it->second.depth);
    for (auto& [n, se] : sessions) {
      if (skip_sess && n == *skip_sess) continue;
      if (se.ssd == h) return true;
      if (node >= 0 && se.node == node && cache.nodes.count(node) && cache.nodes.at(node).sess > 0) return true;
    }
    return false;
  }
  // Pins may hold at most PIN_MAX_SLOTS snapshot slots (default: half of the slots beyond max_running, at least 1) and
  // PIN_MAX_FRAC of the KV pool (0.5): the rest must keep serving running agents and the ordinary prefix cache.
  int pin_max_slots() const {
    return getenv("PIN_MAX_SLOTS") ? (int)env_knob("PIN_MAX_SLOTS", 0, 0, 1 << 24) : std::max(1, (m.n_slots - max_running) / 2);
  }
  double pin_max_frac = env_knob_f("PIN_MAX_FRAC", 0.5, 0, 1);

  void pin_chain(int id, int d) {
    cache.nodes.at(id).pin_snap += d;
    for (int x = id; x >= 0; x = cache.nodes.at(x).parent) cache.nodes.at(x).pin += d;
  }
  // The trie node ending tokens[0, depth * KV_BLOCK), or -1.
  int find_chain(const std::vector<int>& t, int depth) const {
    int parent = -1;
    for (int i = 0; i < depth; i++) {
      parent = cache.find(parent, t.data() + (size_t)i * KV_BLOCK);
      if (parent < 0) return -1;
    }
    return parent;
  }
  static uint64_t tokens_hash(const std::vector<int>& t, int depth) {
    uint64_t h = SpillTier::kSeed;
    for (int i = 0; i < depth; i++) h = SpillTier::extend(h, t.data() + (size_t)i * KV_BLOCK);
    return h;
  }
  uint64_t node_hash(int id) const {
    std::vector<int> chain;
    for (int x = id; x >= 0; x = cache.nodes.at(x).parent) chain.push_back(x);
    uint64_t h = SpillTier::kSeed;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) h = SpillTier::extend(h, cache.nodes.at(*it).tokens.data());
    return h;
  }
  void mark_session(const Seq& s, int id, int depth) {
    if (s.req.session.empty()) return;
    Session& se = sessions[s.req.session];
    if (se.node == id) return;
    if (se.node >= 0 && cache.nodes.count(se.node) && cache.nodes[se.node].sess > 0) cache.nodes[se.node].sess--;
    se.node = id;
    se.depth = depth;
    se.turns++;
    cache.nodes.at(id).sess++;
  }
  // Free the unshared tail of the chain ending at `id`: upward while a node is a leaf that no pin, session, running
  // sequence (block_ref 1: the trie's own reference) or in-flight step uses. A node that must stay loses only its
  // snapshot (when it is not pinned there).
  void drop_chain(int id) {
    for (int x = id; x >= 0;) {
      auto& n = cache.nodes.at(x);
      const bool busy = pending_nodes.count(x) != 0;
      if (n.children || n.pin || n.sess || busy || block_ref[n.block] != 1) {
        if (x == id && n.snapshot >= 0 && !n.pin_snap && !n.sess && !busy) {
          free_slots.push_back(n.snapshot);
          n.snapshot = -1;
          snapshot_count--;
        }
        return;
      }
      const int p = n.parent;
      if (n.snapshot >= 0) { free_slots.push_back(n.snapshot); snapshot_count--; }
      unref_block(n.block);
      cache.erase(x);
      x = p;
    }
  }
  static std::string jstr(const std::string& v) {
    std::string o = "\"";
    for (char c : v) {
      if (c == '"' || c == '\\') o += '\\';
      if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; continue; }
      o += c;
    }
    return o + "\"";
  }
  std::string ctx_json(const std::string& name, const Context& c) const {
    const int depth = (int)c.tokens.size() / KV_BLOCK;
    const int id = find_chain(c.tokens, depth);
    const bool gpu = id >= 0 && cache.nodes.at(id).snapshot != -1;
    const bool ssd = spill && spill->has(tokens_hash(c.tokens, depth));
    char b[256];
    snprintf(b, sizeof b, ",\"tokens\":%zu,\"cached\":%d,\"pinned\":%s,\"gpu\":%s,\"ssd\":%s,\"kv_mib\":%.1f}", c.tokens.size(),
             c.cached, c.pinned ? "true" : "false", gpu ? "true" : "false", ssd ? "true" : "false",
             depth * (double)m.cfg.kv_block_bytes(m.kv8, m.kv4) / 1048576.0);
    return "{\"name\":" + jstr(name) + b;
  }
  // A context save finished (or was already cached): register it under its name, pin it, write it to the SSD tier.
  void ctx_finish(const Seq& s, int id) {
    const int depth = (int)s.tokens.size() / KV_BLOCK;
    if (id < 0) id = find_chain(s.tokens, depth);
    std::string why = "stop";
    if (id < 0 || cache.nodes.at(id).snapshot == -1)
      why = "error: no snapshot slot for the context (more snapshots, or fewer pinned contexts)";
    Context c;
    c.tokens = s.tokens;
    c.cached = std::min(s.cached, (int)s.tokens.size());
    const std::string& name = s.req.ctx_name;
    if (why == "stop" && !name.empty()) {
      auto old = contexts.find(name);
      const bool same = old != contexts.end() && old->second.pinned && old->second.node == id;
      bool pinned_now = false;
      if (s.req.ctx_pin && same) {
        c.pinned = true;
        c.node = id;
      } else if (s.req.ctx_pin) {
        int slots = 0;
        long blocks = 0, add = 0;
        for (auto& [k, x] : contexts) slots += x.pinned && !(k == name);
        for (auto& [k, n] : cache.nodes) blocks += n.pin > 0;
        for (int x = id; x >= 0; x = cache.nodes.at(x).parent) add += cache.nodes.at(x).pin == 0;
        if (slots >= pin_max_slots())
          why = "error: pinned context limit reached (" + std::to_string(slots) + " snapshot slots; PIN_MAX_SLOTS)";
        else if (blocks + add > (long)(pin_max_frac * m.n_blocks))
          why = "error: pinned contexts would hold more than PIN_MAX_FRAC of the KV pool";
        else {
          pin_chain(id, 1);
          c.pinned = true;
          c.node = id;
          pinned_now = true;
        }
      }
      // persist "ssd" is acknowledged only once the chain is on the tier (R05): a failed write (quota, I/O) or no tier
      // fails the save, which is rolled back (this pin undone, any earlier context of the name kept); memory stays.
      if (why == "stop" && s.req.ctx_ssd && !spill_chain(id, 1)) {
        why = spill ? "error: SSD write failed (quota or I/O); the context was not saved"
                    : "error: persist ssd needs the SSD tier (SPILL_DIR)";
        if (pinned_now) pin_chain(id, -1);
        c.pinned = false;
        c.node = -1;
      }
      if (why == "stop") {
        if (old != contexts.end() && old->second.pinned && !same && cache.nodes.count(old->second.node))
          pin_chain(old->second.node, -1);
        c.h = tokens_hash(c.tokens, depth);
        contexts[name] = c;
      }
    }
    Event e{s.req.id, {}, true, why, (int)s.tokens.size(), 0, c.cached};
    e.extra = ",\"ctx\":" + ctx_json(name, c);
    emit(e);
  }
  std::string ctx_drop(const std::string& name) {
    auto it = contexts.find(name);
    if (it == contexts.end()) return "error: unknown context";
    Context& c = it->second;
    const int depth = (int)c.tokens.size() / KV_BLOCK;
    if (c.pinned && cache.nodes.count(c.node)) pin_chain(c.node, -1);
    const int id = find_chain(c.tokens, depth);
    const uint64_t h = tokens_hash(c.tokens, depth);
    const bool shared = ssd_shared(h, id, &name, nullptr);  // before drop_chain may erase the node
    if (id >= 0) drop_chain(id);
    if (spill && spill->has(h) && spill->pinned != h && !shared) spill->drop(h);
    contexts.erase(it);
    return "ok";
  }
  // Move a context to the SSD tier and out of memory (unpinned; the next launch or save restores it).
  std::string ctx_offload(const std::string& name) {
    auto it = contexts.find(name);
    if (it == contexts.end()) return "error: unknown context";
    if (!spill) return "error: no SSD tier (SPILL_DIR)";
    Context& c = it->second;
    const int depth = (int)c.tokens.size() / KV_BLOCK;
    const int id = find_chain(c.tokens, depth);
    if (id < 0 || cache.nodes.at(id).snapshot == -1)
      return spill->has(tokens_hash(c.tokens, depth)) ? "ok" : "error: context is neither in memory nor on SSD";
    if (!spill_chain(id, 1)) return "error: SSD write failed (quota or I/O); the context stays in memory";
    if (c.pinned) pin_chain(id, -1);
    c.pinned = false;
    c.node = -1;
    drop_chain(id);
    return "ok";
  }
  // offload: move the session's latest snapshot to the SSD tier and out of memory. It succeeds only if the entry is on
  // the tier; on a failed write the session keeps its memory state (R05). Drop (offload false) deletes the session's
  // entries unless another context or session owns them too (R06).
  std::string session_drop(const std::string& name, bool offload) {
    if (offload && !spill) return "error: no SSD tier (SPILL_DIR)";
    auto it = sessions.find(name);
    if (it == sessions.end()) return offload ? "error: unknown session" : "ok";
    Session& se = it->second;
    const bool live = se.node >= 0 && cache.nodes.count(se.node) && cache.nodes[se.node].sess > 0;
    if (offload) {
      if (live) {
        if (!spill_chain(se.node, 1)) return "error: SSD write failed (quota or I/O); the session stays in memory";
        cache.nodes[se.node].sess--;
        drop_chain(se.node);
      } else if (!se.ssd || !spill->has(se.ssd)) {
        return "error: session is neither in memory nor on SSD";
      }
      se.node = -1;
      return "ok";
    }
    if (live) {
      if (spill)
        if (uint64_t h = node_hash(se.node); spill->has(h) && spill->pinned != h && !ssd_shared(h, se.node, nullptr, &name))
          spill->drop(h);
      cache.nodes[se.node].sess--;
      drop_chain(se.node);
    }
    if (spill && se.ssd && spill->has(se.ssd) && spill->pinned != se.ssd && !ssd_shared(se.ssd, -1, nullptr, &name))
      spill->drop(se.ssd);
    sessions.erase(it);
    return "ok";
  }
  // Memory and cache occupancy for GET /v1/contexts and the benchmarks.
  std::string stats_json() const {
    long trie = 0, trie_only = 0, pinned = 0, sess_snaps = 0;
    for (auto& [id, n] : cache.nodes) {
      trie++;
      trie_only += block_ref[n.block] == 1;
      pinned += n.pin > 0;
      sess_snaps += n.sess > 0 && n.snapshot != -1;
    }
    std::unordered_set<int> act;
    for (auto& r : running) act.insert(r.blocks.begin(), r.blocks.end());
    const double bm = m.cfg.kv_block_bytes(m.kv8, m.kv4) / 1048576.0;
    char b[768];
    snprintf(b, sizeof b, "{\"kv_blocks\":%d,\"kv_free\":%zu,\"kv_running\":%zu,\"kv_cached\":%ld,\"kv_cached_only\":%ld,"
             "\"kv_pinned\":%ld,\"block_tokens\":%d,\"block_mib\":%.3f,\"slots\":%d,\"free_slots\":%zu,\"snapshots\":%d,"
             "\"session_snapshots\":%ld,\"sessions\":%zu,\"running\":%zu,\"waiting\":%zu,\"preempted\":%llu",
             m.n_blocks, n_free, act.size(), trie, trie_only, pinned, KV_BLOCK, bm, m.n_slots, free_slots.size(),
             snapshot_count, sess_snaps, sessions.size(), running.size(), waiting.size(), (unsigned long long)stat_preempted);
    std::string o = b;
    if (spill) {
      snprintf(b, sizeof b, ",\"ssd\":{\"entries\":%zu,\"gib\":%.2f,\"quota_gib\":%.1f,\"spilled\":%llu,\"restored\":%llu,"
               "\"spill_s\":%.2f,\"restore_s\":%.2f}", spill->entries.size(), spill->used / 1073741824.0,
               spill->quota / 1073741824.0, (unsigned long long)spill->n_spilled, (unsigned long long)spill->n_restored,
               spill->spill_s, spill->restore_s);
      o += b;
    }
    o += ",\"contexts\":[";
    bool first = true;
    for (auto& [k, c] : contexts) { o += (first ? "" : ",") + ctx_json(k, c); first = false; }
    return o + "]}";
  }


  int sample(const uint16_t* lg, Seq& s) {
    const int V = m.cfg.vocab;
    if (s.req.grammar)
      return s.req.grammar->sample(lg, V, s.req.temperature, s.req.top_p, s.req.top_k, s.rng, s.req.stop_ids);
    auto f = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float x; memcpy(&x, &u, 4); return x; };
    if (s.req.temperature <= 0) {
      int best = 0;
      float bv = -INFINITY;
      for (int i = 0; i < V; i++) {
        float v = f(lg[i]);
        if (v > bv) { bv = v; best = i; }
      }
      return best;
    }
    int k = s.req.top_k > 0 ? std::min(s.req.top_k, V) : std::min(V, 1024);
    std::vector<std::pair<float, int>> c(V);
    for (int i = 0; i < V; i++) c[i] = {f(lg[i]), i};
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    c.resize(k);
    double mx = c[0].first, sum = 0;
    std::vector<double> pr(k);
    for (int i = 0; i < k; i++) sum += pr[i] = std::exp((c[i].first - mx) / s.req.temperature);
    double cut = s.req.top_p * sum, acc = 0;
    int n = 0;
    while (n < k) { acc += pr[n++]; if (acc >= cut) break; }
    std::uniform_real_distribution<double> u(0, acc);
    double r = u(s.rng);
    for (int i = 0; i < n; i++) { if ((r -= pr[i]) <= 0) return c[i].second; }
    return c[n - 1].second;
  }

  void finish(size_t i, const std::string& reason) {
    Seq& s = running[i];
    if (s.req.prefill_only && reason == "stop") ctx_finish(s, -1);  // a context save: register / pin, then its event
    else emit(Event{s.req.id, {}, true, reason, (int)s.req.prompt.size(), s.generated, s.cached});
    // A turn that ends on a block boundary hands its slot to the trie as the snapshot there (no copy): the next turn
    // resumes at the end of this one's output. Only when every computed position holds a delivered token (a stop token
    // inside an accepted block leaves state past it).
    bool donated = false;
    if (s.req.gen_snap && snap_gen > 0 && (reason == "stop" || reason == "length") && s.decoding() && s.computed > 0 &&
        s.computed % KV_BLOCK == 0 && (int)s.tokens.size() == s.computed + 1 && s.slot >= 0) {
      std::vector<std::pair<int, int>> none;
      donated = snapshot_at(s, s.computed, none, true);
    }
    for (int b : s.blocks) unref_block(b);
    if (!donated) free_slots.push_back(s.slot);
    running.erase(running.begin() + i);
  }

  // Emit generated tokens for sequence i, stopping at a stop token or max_tokens. Returns true when finished
  // (the finish reason goes to `why`).
  bool deliver(Seq& s, const int* t, int n, std::string& why) {
    for (int k = 0; k < n; k++) {
      s.tokens.push_back(t[k]);
      s.generated++;
      if (std::find(s.req.stop_ids.begin(), s.req.stop_ids.end(), t[k]) != s.req.stop_ids.end()) { why = "stop"; return true; }
      emit(Event{s.req.id, {t[k]}});
      if (s.generated >= s.req.max_tokens) { why = "length"; return true; }
    }
    return false;
  }

  // One engine step. Returns false when there was nothing to do.
  bool step() {
    std::vector<std::pair<int, int>> restores, snaps, gsnaps;
    std::vector<int> zeros;
    // Decode snapshots due from the last step: the slots still hold the state at that boundary; copied in this step's
    // first blit, before anything computes.
    for (auto& r : running)
      if (r.snap_due) {
        r.snap_due = false;
        if (r.computed % KV_BLOCK == 0 && (int)r.tokens.size() == r.computed + 1) {
          snapshot_at(r, r.computed, gsnaps);
          r.last_snap = r.computed;
        }
      }
    admit(restores, zeros);
    if (running.empty()) return false;
    // Priority also orders the work (PREFILL_PRIORITY, default on): prefill is planned in running order, so a stable
    // sort by priority (highest first) serves an agent's prompt before lower-priority ones that arrived earlier. Within
    // one priority arrival order is kept, so preemption (lowest priority, then newest) picks the same victim.
    if (prefill_priority && running.size() > 1 &&
        std::any_of(running.begin(), running.end(), [&](const Seq& r) { return r.req.priority != running[0].req.priority; }))
      std::stable_sort(running.begin(), running.end(), [](const Seq& a, const Seq& b) { return a.req.priority > b.req.priority; });

    // Plan: decoding sequences first (a draft block of d->block rows for greedy ones, else 1 row), then
    // prefill chunks under the remaining token budget.
    std::vector<Qwen35::SeqWork> work;
    std::vector<int> who, counts;
    int budget = step_tokens;
    // PREFILL_STEP (server): while nothing decodes, prefill chunks may fill a larger step (the model's activations are
    // sized for it). Flash-Next reads ~64 GB of expert weights per step whatever its rows, so 8,192-row steps
    // prefill a lone 32k prompt at 4.1k tok/s against 3.6k at 2,048 (2026-11-02 night).
    bool prio = false;  // PREFILL_PRIO: decoders' rows do not come out of the prefill budget
    {
      int ndec = 0;
      bool pf = false;
      for (auto& r : running) { ndec += r.decoding(); pf |= !r.decoding(); }
      if (prefill_step > step_tokens && !ndec) budget = prefill_step;
      if (pf && ndec > 0 && ndec <= prefill_prio) {
        prio = true;
        budget = std::max(step_tokens, prefill_step);
      }
    }
    // Speculation block size by load: a verify row costs a target row of compute, so as more sequences decode
    // together the extra drafts stop paying for themselves. SPEC_BLOCKS="n1:L1,n2:L2,..." overrides.
    int n_dec = 0;
    for (auto& r : running) n_dec += r.decoding() && r.spec_ok;
    cur_ndec = n_dec;
    const auto t_plan = std::chrono::steady_clock::now();
    int L = d ? d->block : mtp ? mtp->block : 1;
    int tune_key = -1, tune_i = -1;
    // Wide blocks (8) for up to spec_wide agents; above that the block-5 tuning measured on the Studio (5 rows, or 4
    // when that saves a W4A8 tile) until block 8 is re-tuned under load.
    if (d) L = std::min(L, max_block_for(n_dec));
    if (d && L > 5 && n_dec > spec_wide) L = 5;
    // No neural accelerators (M3/M4): verify rows cost much more there (M3 Ultra, one stream: 5 rows 37 ms, 8 rows 57 ms;
    // spec_test block 5 vs 8: 91.8 vs 83.6 and 86.3 vs 70.5 tok/s), so start at 5; SPEC_AUTO still tries 4/5/8.
    if (d && L > 5 && !m.g.nax) L = 5;
    if (!m.g.nax) tuner.far_every = 4;
    for (auto [n, l] : spec_blocks) if (n_dec > n) L = l;
    if (mtp) L = std::max(2, std::min(L, mtp->block));
    // Without SPEC_BLOCKS: drop to L-1 when that saves a 32-row W4A8 tile in every linear layer (a tile pass costs
    // ~13-17 ms; the last draft position ~13% of the accepted tokens). 8 agents: 40 -> 32 rows, 16: 80 -> 64.
    // At 16k this took 8 agents 375 -> 406 tok/s total and 16 agents 464 -> 488.
    if (spec_blocks.empty() && d && m.use_w4a8 && L == std::min(d->block, 5) && L > 2 && n_dec * L < m.w4a8_max &&
        (n_dec * L + 31) / 32 > (n_dec * (L - 1) + 31) / 32)
      L--;
    // Token trees (TREE) for up to tree_seqs greedy decoders (DFlash2 only), in place of the tuner's chain block.
    bool any_greedy = false;
    for (auto& r : running) any_greedy |= r.decoding() && r.spec_ok && r.req.temperature <= 0;
    const bool tree_ok = d && tree_n > 1 && n_dec >= 1 && n_dec <= tree_seqs && any_greedy && m.tpath.buf;
    int tree_nodes = tree_ok ? tree_n : 0;
    if (tree_nodes) {  // every verify row needs an output row (Qwen35::max_out): trees only when theirs fit
      int out_rows = 0;
      for (auto& r : running)
        if (r.decoding()) out_rows += !r.spec_ok ? 1 : r.req.temperature <= 0 ? tree_nodes : d->block;
      if (out_rows > m.max_out) tree_nodes = 0;
    }
    if (spec_auto && spec_blocks.empty() && d && n_dec > 0 && !tree_nodes) {
      double ctx = 0;
      for (auto& r : running)
        if (r.decoding() && r.spec_ok) ctx += r.computed;
      tune_key = BlockTuner::key(n_dec, ctx / n_dec);
      tune_i = tuner.choose(tune_key, n_dec, max_block_for(n_dec), L);
      L = std::min(d->block, tuner.cand[tune_i]);
    }
    // Guard: chain blocks within the verify scratch (spec_rows_cap already covers the policy; SPEC_BLOCKS may not)
    if (d && !tree_nodes && n_dec > 0)
      while (L > 2 && n_dec * (L <= 5 ? 5 : L <= 8 ? 8 : L <= 12 ? 12 : 16) > d->max_rows) L--;
    // Token trees for a few greedy decoders (the drafter then drafts its full block for every sequence).
    const bool use_tree = tree_nodes > 1;
    auto tree_for = [&](const Seq& s) { return use_tree && s.spec_ok && s.req.temperature <= 0 && !s.req.grammar; };
    if (use_tree) L = d->block;  // the drafter drafts its whole block; the tree takes what it needs
    for (auto& r : running) r.cdraft.clear();
    if (mtp && copy_on && (!copy_solo || n_dec == 1)) {  // context-copy chains (src/copydraft.h)
      static const int ng = getenv("COPY_NG") ? atoi(getenv("COPY_NG")) : 3;
      for (auto& r : running) {
        if (!r.decoding() || !r.spec_ok || r.req.temperature > 0) continue;
        r.copy.ng = ng;
        int match = 0;
        const int n = r.copy.propose(r.tokens, r.computed, std::min(copy_k, kMaxBlock - 1), r.cdraft, match);
        bool use = copy_on == 1 ? n >= L - 1 : n > L - 1;
        if (use && copy_check)
          use = r.mtp_fold ? r.cdraft[0] == mtp->d1s.ptr<int>()[r.slot] : match >= copy_trust;
        if (use && copy_on == 1) {  // adaptive: the length with the best expected tokens per ms, or MTP
          const int k = copy_pol.choose(match, n, L);
          use = k > 0;
          if (use) r.cdraft.resize(k);
        }
        r.cmatch = match;
        if (!use) r.cdraft.clear();
      }
    }
    auto rows_for = [&](const Seq& s) {
      int r = tree_for(s) ? tree_nodes : s.spec_ok ? (s.cdraft.empty() ? L : 1 + (int)s.cdraft.size()) : 1;
      // land on the next decode snapshot boundary (a tree commits up to kMaxBlock tokens: a chain of `need` rows then)
      if (d && s.spec_ok && s.cdraft.empty())
        if (const int need = gen_need(s); need > 0 && need < (tree_for(s) ? kMaxBlock : r)) r = need;
      return r;
    };
    if (mtp) {  // every verify row needs an output row (Qwen35::max_out): drop the longest copy chains until they fit
      int out_rows = 0;
      for (auto& r : running) if (r.decoding()) out_rows += r.spec_ok ? rows_for(r) : 1;
      while (out_rows > m.max_out) {
        Seq* lg = nullptr;
        for (auto& r : running) if (!r.cdraft.empty() && (!lg || r.cdraft.size() > lg->cdraft.size())) lg = &r;
        if (!lg) break;
        out_rows -= (int)lg->cdraft.size() + 1 - L;
        lg->cdraft.clear();
      }
    }
    if (!prio)
      for (size_t i = 0; i < running.size(); i++)
        if (running[i].decoding()) budget -= rows_for(running[i]);
    if (prefill_min > 0) budget = std::max(budget, prefill_min);
    // Short prompts first (PREFILL_SHORT tokens, 0 off; agent-contexts prototype): in arrival order a few long prefills
    // took every step's budget and short agents admitted beside them waited for all of it (agent scenario e: helpers'
    // first token ~100 s behind three 16k researchers). Prefills with at most PREFILL_SHORT tokens left get first claim
    // on the budget; long ones take what remains, so they are delayed by at most the short work actually present.
    auto chunk_want = [&](const Seq& r) {
      int w = (int)r.tokens.size() - r.computed;
      if (r.next_snap() > r.computed) w = std::min(w, r.next_snap() - r.computed);
      return w;
    };
    int short_left = 0;
    if (prefill_short > 0)
      for (auto& r : running)
        if (!r.decoding() && r.prefilling() && (int)r.tokens.size() - r.computed <= prefill_short) short_left += chunk_want(r);
    for (size_t i = 0; i < running.size(); i++) {
      Seq& s = running[i];
      int n;
      bool spec = false;
      if (!s.decoding()) {
        if (budget <= 0) continue;
        const bool sh = prefill_short > 0 && (int)s.tokens.size() - s.computed <= prefill_short;
        const int avail = sh ? budget : budget - std::min(short_left, budget * 3 / 4);  // long ones keep >= 1/4
        if (avail <= 0) continue;
        n = std::min(avail, chunk_want(s));
        budget -= n;
        if (sh) short_left -= n;
      } else {
        n = rows_for(s);
        spec = s.spec_ok;
        // A verify block must stay inside the block table (Codex review R02): near the end, decode one row.
        if (spec && s.computed + n > (m.max_blocks - 1) * KV_BLOCK) { n = 1; spec = false; s.cdraft.clear(); }
      }
      // Make sure the KV blocks cover the new positions.
      int need = (s.computed + n + KV_BLOCK - 1) / KV_BLOCK;
      if (need > m.max_blocks) {  // cannot happen after the admission and width checks; never write past the table
        finish(i, "error: context too long");
        i--;
        continue;
      }
      bool ok = true;
      while ((int)s.blocks.size() < need) {
        int b = alloc_block(next_block_hint(s.blocks));
        if (b < 0 && running.size() > i + 1) {  // out of blocks: preempt the newest sequence not yet planned
          preempt(victim(i));
          continue;
        }
        if (b < 0) { ok = false; break; }
        s.blocks.push_back(b);
      }
      if (!ok && running.size() == 1) {  // alone, and evicting the whole prefix cache did not make room
        finish(i, "error: KV pool exhausted");
        break;
      }
      if (!ok) { if (s.decoding()) budget += n; continue; }
      if (spec) s.tokens.resize(s.computed + n, 0);  // the staged token, then placeholders the drafter fills
      if (spec && !s.cdraft.empty()) std::copy(s.cdraft.begin(), s.cdraft.end(), s.tokens.begin() + s.computed + 1);
      who.push_back((int)i);
      counts.push_back(n);
      Qwen35::SeqWork sw{s.slot, &s.blocks, s.computed, s.tokens.data() + s.computed, n, spec};
      sw.tree = spec && tree_for(s) && n == tree_nodes && !(gen_need(s) > 0 && gen_need(s) < kMaxBlock);
      if (spec && s.req.temperature > 0) {
        sw.temp = s.req.temperature;
        sw.top_p = s.req.top_p;
        sw.top_k = s.req.top_k;
        sw.seed = (uint32_t)s.rng();
      }
      sw.mrope = s.req.mrope.data();
      sw.mrope_len = (int)s.req.mrope.size() / 3;
      sw.rope_delta = s.req.rope_delta;
      sw.emb = s.req.embeds.data();
      sw.ext_idx = s.ext_idx.data();
      sw.ext_len = (int)s.ext_idx.size();
      work.push_back(sw);
    }
    if (work.empty()) return false;
    // Verify blocks first: the GDN tapes hold only the first rows of a step (Qwen35::enable_spec). Rows are
    // independent of their position in the step, so this changes no output.
    if (d && m.tape_rows < m.max_tokens) {
      std::vector<size_t> ord(work.size());
      for (size_t j = 0; j < ord.size(); j++) ord[j] = j;
      std::stable_partition(ord.begin(), ord.end(), [&](size_t j) { return work[j].spec; });
      std::vector<Qwen35::SeqWork> w2;
      std::vector<int> who2, counts2;
      for (size_t j : ord) { w2.push_back(work[j]); who2.push_back(who[j]); counts2.push_back(counts[j]); }
      work.swap(w2);
      who.swap(who2);
      counts.swap(counts2);
    }

    int T = m.set_batch(work);
    if (debug) {
      fprintf(stderr, "step: T=%d S=%zu restores=%zu zeros=%zu |", T, work.size(), restores.size(), zeros.size());
      for (size_t j = 0; j < work.size(); j++)
        fprintf(stderr, " [id%llu pos%d n%d%s]", (unsigned long long)running[who[j]].req.id, work[j].pos0, work[j].n,
                work[j].spec ? " spec" : "");
      fprintf(stderr, "\n");
    }
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLCommandBuffer> dcb = nil;          // tree steps: the drafter's command buffer
    std::vector<int> dblk(who.size(), -1);   // tree steps: each sequence's block index in the drafter's rows
    if (!restores.empty() || !zeros.empty() || !gsnaps.empty()) {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      for (auto [src, dst] : gsnaps) { m.encode_copy_slot(b, src, dst); if (d) d->encode_copy_slot(b, src, dst); }
      for (auto [src, dst] : restores) { m.encode_copy_slot(b, src, dst); if (d) d->encode_copy_slot(b, src, dst); }
      for (int z : zeros) { m.encode_zero_slot(b, z); if (d) d->encode_zero_slot(b, z); }
      [b endEncoding];
    }
    if (d) {
      // Drafter: enter last step's committed rows into its context, then draft every greedy decoding sequence.
      std::vector<Drafter::CtxRow> ctx;
      std::vector<Drafter::BlockSeq> blocks;
      for (size_t j = 0; j < who.size(); j++) {
        Seq& s = running[who[j]];
        ctx.insert(ctx.end(), s.pend.begin(), s.pend.end());
        s.pend.clear();
        if (work[j].spec) dblk[j] = (int)blocks.size();
        if (work[j].spec)
          blocks.push_back({s.slot, s.computed, s.tokens[s.computed], m.seq_start.ptr<int>()[j], work[j].tree, work[j].n});
      }
      Enc e(g, cb);
      d->encode_context(e, ctx, 0);
      d->encode_block(e, blocks, L);
      if (m.n_tree && tree_gpu) {
        std::vector<std::array<int, 4>> ts;
        for (size_t j = 0; j < who.size(); j++)
          if (work[j].tree) ts.push_back({dblk[j] * L, m.seq_start.ptr<int>()[j], work[j].pos0, work[j].n});
        d->encode_tree(e, ts, tree_params, L);
      }
      e.end();
      const bool host_tree = m.n_tree && !tree_gpu;
      if (m.cb_split || host_tree) m.commit_early(cb);  // draft on the GPU while the target step is encoded
      if (host_tree) dcb = m.early.back();
    }
    bool any_spec = false;
    // Tree steps keep the verify in one command buffer, committed only once the trees are in its input rows.
    const int saved_split = m.cb_split;
    if (m.n_tree && !tree_gpu) m.cb_split = 0;
    if (mtp) {
      // MTP head: draft L - 1 tokens per speculated sequence into its verify rows, then the PLE n-gram ids of those
      // rows from the drafts. Context rows' next tokens: the prompt's for prefill rows (the GPU fills verify rows').
      std::vector<Mtp::DraftSeq> ds;
      std::vector<std::array<int, 2>> h2;
      int maxn = 1;
      bool folded = mtp->fold;
      last_prompt.assign(who.size(), 0);
      for (size_t j = 0; j < who.size(); j++) {
        Seq& s = running[who[j]];
        const int r0 = m.seq_start.ptr<int>()[j], p0 = work[j].pos0;
        if (work[j].spec && s.cdraft.empty()) ds.push_back({s.slot, &s.blocks, s.computed, s.tokens[s.computed], r0});
        if (work[j].spec && s.cdraft.empty()) folded &= s.mtp_fold;
        last_prompt[j] = !work[j].spec && s.computed + counts[j] == (int)s.tokens.size() && s.req.temperature <= 0;
        any_spec |= work[j].spec;
        maxn = std::max(maxn, counts[j]);
        h2.push_back({p0 >= 1 ? s.tokens[p0 - 1] : 0, p0 >= 2 ? s.tokens[p0 - 2] : 0});
        for (int k = 0; k < counts[j]; k++) {
          const int p = p0 + k;
          mtp->next.ptr<int>()[r0 + k] = !work[j].spec && p + 1 < (int)s.tokens.size() ? s.tokens[p + 1] : 0;
        }
      }
      Enc e(g, cb);
      static const bool skip_draft = getenv("MTP_SKIP_DRAFT") && atoi(getenv("MTP_SKIP_DRAFT"));  // test: junk drafts
      if (!skip_draft) mtp->encode_draft(e, cb, ds, L - 1, folded);
      mtp->ids_fix(e, h2, maxn, any_spec);
      if (mtp->ids_check) mtp->ids_all(e, (int)who.size(), maxn);
      e.end();
      if (m.cb_split && any_spec) m.commit_early(cb);
    }
    {
      static const bool split_pf = getenv("FN_SPLIT_PF") && atoi(getenv("FN_SPLIT_PF"));
      bool pf = false;
      for (auto& w : work) pf |= !w.spec && w.n > 1;
      m.no_split_step = pf && !split_pf;
    }
    if (m.n_out > m.max_out) {  // would write logits past their buffer (see the server's enable_spec)
      fprintf(stderr, "step: %d output rows > %d\n", m.n_out, m.max_out);
      abort();
    }
    m.encode_step(cb, T, (int)work.size());
    if (mtp) {
      Enc e(g, cb);
      mtp->encode_context(e, cb, T, (int)work.size(), any_spec, last_prompt);
      e.end();
    }
    if (d) {
      // Prefill rows are all committed: enter them into the drafter context now, so snapshots include them.
      std::vector<Drafter::CtxRow> ctx;
      for (size_t j = 0; j < who.size(); j++) {
        Seq& s = running[who[j]];
        if (!s.spec_ok || work[j].spec) continue;
        int r0 = m.seq_start.ptr<int>()[j];
        for (int k = 0; k < counts[j]; k++) ctx.push_back({r0 + k, s.computed + k, s.slot});
      }
      Enc e(g, cb);
      d->encode_context(e, ctx, 1);
      e.end();
    }
    // Snapshots are decided after planning: a sequence snapshots when this step ends exactly at its point.
    for (size_t j = 0; j < who.size(); j++) {
      Seq& s = running[who[j]];
      if (work[j].spec) continue;
      s.computed += counts[j];
      take_snapshots(s, snaps);
      s.computed -= counts[j];
    }
    if (!snaps.empty()) {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      for (auto [src, dst] : snaps) { m.encode_copy_slot(b, src, dst); if (d) d->encode_copy_slot(b, src, dst); }
      [b endEncoding];
    }
    m.cb_split = saved_split;
    if (m.n_tree && !tree_gpu) {
      [dcb waitUntilCompleted];
      for (size_t j = 0; j < who.size(); j++) {
        if (!work[j].tree) continue;
        Seq& s = running[who[j]];
        const int r = dblk[j];
        DraftOut o{d->cand.ptr<int>(), d->unary.ptr<float>(), d->hp.ptr<uint16_t>(), d->pred_cb.ptr<uint16_t>(),
                   d->succ_cb.ptr<uint16_t>(), r * L, L};
        Tree t;
        build_tree(o, s.tokens[s.computed], tree_params, t);
        m.set_tree((int)j, s.computed, t);
      }
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (debug) fprintf(stderr, "[%.3f s] ", std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
    if (debug)
      fprintf(stderr, "  step %.1f ms since planning (last command buffer GPU %.1f ms)\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_plan).count(),
              (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    if (debug && !m.early.empty()) {
      fprintf(stderr, "    early command buffers (GPU start..end ms, relative to the first):");
      const double g0 = m.early[0].GPUStartTime;
      for (id<MTLCommandBuffer> c : m.early) fprintf(stderr, " %.1f..%.1f", (c.GPUStartTime - g0) * 1e3, (c.GPUEndTime - g0) * 1e3);
      fprintf(stderr, " | last %.1f..%.1f\n", (cb.GPUStartTime - g0) * 1e3, (cb.GPUEndTime - g0) * 1e3);
    }
    NSError* err = cb.error;
    for (id<MTLCommandBuffer> c : m.early)
      if (c.error) err = c.error;
    m.early.clear();
    if (err) {  // snapshots published this step hold no valid state: drop them (Codex review R01)
      for (int id : pending_snaps)
        if (cache.nodes.count(id) && cache.nodes[id].snapshot >= 0) {
          free_slots.push_back(cache.nodes[id].snapshot);
          cache.nodes[id].snapshot = -1;
          snapshot_count--;
        }
    }
    if (err && stateless) {  // every node is a resume point: drop those this step was writing (and their subtrees)
      std::vector<int> bad(pending_nodes.begin(), pending_nodes.end());
      std::sort(bad.begin(), bad.end(), std::greater<int>());  // children have larger ids than their parents
      for (int id : bad) drop_subtree(id);
    }
    pending_nodes.clear();
    pending_snaps.clear();
    if (err) {
      fprintf(stderr, "step failed: %s\n", err.localizedDescription.UTF8String);
      for (int i = (int)running.size() - 1; i >= 0; i--) finish(i, "error");
      return true;
    }

    if (mtp && mtp->ids_check && m.ple_ids_buf.buf) {  // MTP_IDS_CHECK: the GPU n-gram ids against the host's hash
      const int nh = m.ngram.heads;
      long bad_fix = 0, bad_all = 0, rows = 0;
      for (size_t j = 0; j < who.size(); j++) {
        Seq& s = running[who[j]];
        const int r0 = m.seq_start.ptr<int>()[j], p0 = work[j].pos0, n = counts[j];
        std::vector<int> t{p0 >= 2 ? s.tokens[p0 - 2] : 0, p0 >= 1 ? s.tokens[p0 - 1] : 0};
        for (int k = 0; k < n; k++) t.push_back(m.tok.ptr<int>()[r0 + k]);  // the drafts, for verify rows
        std::vector<int> ref((size_t)n * nh);
        m.ngram.ids(t.data() + 2, p0, n, ref.data());
        for (int k = 0; k < n * nh; k++) {
          bad_fix += m.ple_ids_buf.ptr<int>()[(size_t)r0 * nh + k] != ref[k];
          bad_all += mtp->ids_chk.ptr<int>()[(size_t)r0 * nh + k] != ref[k];
        }
        rows += n;
      }
      fprintf(stderr, "mtp ids check: %ld rows, %ld wrong ids in the step, %ld wrong from the GPU hash\n", rows, bad_fix, bad_all);
    }
    // Commit: speculative blocks keep their accepted prefix plus the bonus token; everything else keeps all rows
    // and samples one token after its last row.
    std::vector<std::pair<int, std::string>> done;
    std::vector<uint64_t> rollback;
    for (size_t j = 0; j < who.size(); j++) {
      Seq& s = running[who[j]];
      std::string why;
      if (work[j].spec) {
        int c = m.commit.ptr<int>()[j], r0 = m.seq_start.ptr<int>()[j];
        const int* tp = m.tpath.buf ? m.tpath.ptr<int>() + j * kMaxBlock : nullptr;
        if (d) for (int k = 0; k < c; k++) s.pend.push_back({r0 + (work[j].tree ? tp[k] : k), s.computed + k, s.slot});
        if (work[j].tree) {
          tree_cycles++;
          tree_committed += c;
          tree_hist[std::min(c, kMaxBlock)]++;
          if (debug && tree_cycles % 64 == 0) {
            fprintf(stderr, "tree: %llu cycles, %.2f tokens per cycle, hist", (unsigned long long)tree_cycles,
                    (double)tree_committed / tree_cycles);
            for (int k = 1; k <= kMaxBlock; k++) if (tree_hist[k]) fprintf(stderr, " %d:%llu", k, (unsigned long long)tree_hist[k]);
            fprintf(stderr, "\n");
          }
        }
        s.mtp_fold = mtp && mtp->fold;
        s.tokens.resize(s.computed + 1);  // drop the placeholders; keep the staged token
        s.computed += c;
        stat_cycles++;
        stat_committed += c;
        if (!s.cdraft.empty()) {
          stat_copy_cycles++;
          stat_copy_committed += c;
          copy_pol.copy_done(s.cmatch, (int)s.cdraft.size(), c);
          s.cdraft.clear();
        } else if (mtp && counts[j] == L) {
          copy_pol.mtp_done(c);
        }
        if (counts[j] > 1 && !work[j].tree) tuner.accepted(counts[j], c);
        if (debug) {
          fprintf(stderr, "  id%llu commit %d:", (unsigned long long)s.req.id, c);
          for (int k = 0; k < c; k++) fprintf(stderr, " %d", m.emit.ptr<int>()[j * kMaxBlock + k]);
          fprintf(stderr, " | drafts:");
          for (int k = 0; k < counts[j]; k++) fprintf(stderr, " %d", m.tok.ptr<int>()[r0 + k]);
          fprintf(stderr, "\n");
        }
        if (s.req.grammar) {  // structured output: the committed tokens must pass the grammar (src/grammar.h)
          const int* e = m.emit.ptr<int>() + j * kMaxBlock;
          const bool sampled = s.req.temperature > 0;
          const int ok = s.req.grammar->advance(e, c, s.req.stop_ids, sampled);
          if (sampled && s.req.grammar->active) s.spec_ok = false;  // from here one token per step, Matcher::sample
          if (ok < c) {  // token `ok` is rejected (sampled: or follows </think>): its row, under the grammar, decides
            std::vector<int> fix(e, e + ok);
            fix.push_back(sample(m.logits.ptr<uint16_t>() + (size_t)(m.out_base.ptr<int>()[j] + ok) * m.cfg.vocab, s));
            stat_grammar_fix++;
            s.mtp_fold = false;  // the GPU folded its next draft from the token just replaced
            if (deliver(s, fix.data(), (int)fix.size(), why)) done.push_back({who[j], why});
            else if (ok < c - 1) rollback.push_back(s.req.id);  // drafts past `ok` are in the state: re-prefill
            continue;
          }
        }
        if (deliver(s, m.emit.ptr<int>() + j * kMaxBlock, c, why)) done.push_back({who[j], why});
        continue;
      }
      s.computed += counts[j];
      s.mtp_fold = mtp && mtp->fold && !last_prompt.empty() && last_prompt[j];
      if (s.prefilling()) continue;
      s.resumed = false;
      if (s.req.prefill_only) { done.push_back({who[j], "stop"}); continue; }  // a context: computed, nothing to sample
      int out = m.out_base.ptr<int>()[j];
      int t = ((d || mtp) && s.req.temperature <= 0 && !s.req.grammar) ? m.pred.ptr<int>()[out]
                                            : sample(m.logits.ptr<uint16_t>() + (size_t)out * m.cfg.vocab, s);
      if (deliver(s, &t, 1, why)) done.push_back({who[j], why});
    }
    // Decode-only steps (no prefill rows, no restores or snapshot copies) time the block choice.
    if (tune_i >= 0 && restores.empty() && snaps.empty() && zeros.empty()) {
      bool pure = true;
      for (size_t j = 0; j < who.size(); j++) pure &= work[j].spec;
      if (pure) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_plan).count();
        tuner.timed(tune_key, tune_i, ms);
        if (debug && tuner.cells[tune_key].steps % 64 == 0) {
          auto& c = tuner.cells[tune_key];
          fprintf(stderr, "tuner: %d decoders, key %d: block 4/5/8/12/16 %.1f/%.1f/%.1f/%.1f/%.1f ms, "
                  "E %.2f/%.2f/%.2f/%.2f/%.2f, chose %d\n", n_dec, tune_key, c.t[0], c.t[1], c.t[2], c.t[3], c.t[4],
                  tuner.expect(4), tuner.expect(5), tuner.expect(8), tuner.expect(12), tuner.expect(16), [&] {
                    int b = 0;  // the best candidate (this step itself is always a re-test step: steps % 16 == 0)
                    for (int i = 1; i < BlockTuner::kC; i++)
                      if (c.n[i] && (!c.n[b] || tuner.expect(tuner.cand[i]) / c.t[i] > tuner.expect(tuner.cand[b]) / c.t[b])) b = i;
                    return tuner.cand[b];
                  }());
        }
      }
    }
    for (size_t j = 0; j < who.size(); j++) {  // decode snapshots: a step that ended on a boundary (finishers: finish)
      Seq& s = running[who[j]];
      if (s.req.gen_snap && snap_gen > 0 && s.decoding() && s.computed % KV_BLOCK == 0 &&
          (s.computed % snap_gen == 0 || s.computed - s.last_snap >= snap_gen / 2))
        s.snap_due = true;
    }
    std::sort(done.begin(), done.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (auto& [i, why] : done) finish(i, why);
    for (uint64_t id : rollback)  // structured output: re-queue to recompute the state up to the last valid token
      for (size_t i = 0; i < running.size(); i++)
        if (running[i].req.id == id) {
          Seq& s = running[i];
          s.snap_due = false;
          s.computed = (int)s.tokens.size() - 1;  // preempt keeps tokens[0, computed] (the last is staged)
          preempt(i);
          stat_preempted--;
          stat_grammar_rollback++;
          break;
        }
    return true;
  }
};
