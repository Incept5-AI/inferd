// inferd: the engine as a local server. Listens on a Unix socket for newline-delimited JSON:
//   {"op":"gen","id":N,"tokens":[...],"max_tokens":N,"temperature":T,"top_p":P,"top_k":K,"stop":[...],"seed":S,
//    "max_context":C,"priority":R,"snap_at":[pos...],"gen_snap":1}   (optional: context cap, preemption priority, positions
//    worth a prefix-cache snapshot, e.g. the end of the system prompt)
//   multimodal prompts add "mrope":[t,h,w per prompt token...],"rope_delta":D,"embeds":"<base64 bf16 [n, hidden]>"
//   (rows with token id >= vocab take the next embeds row; see frontend/vision.py)
//   {"op":"cancel","id":N}
//   agent contexts (docs/CONTEXTS.md): "gen" with "ctx":"<name>" (+ "pin":1, "ssd":1) computes the prompt (whole KV
//   blocks) and registers it, finishing without output (done event with "ctx":{...}); "gen" with "session":"<agent>"
//   marks that agent's latest snapshot, "no_cache":1 takes no snapshots. Ops answered with
//   {"id":N,"done":"ok|error: ...",...}: {"op":"ctx_drop"|"ctx_offload","id":N,"ctx":"<name>"},
//   {"op":"session_drop"|"session_offload","id":N,"session":"<agent>"}, {"op":"stats","id":N} (+ "stats":{...}).
//   {"op":"vocab","hex":["<token bytes as hex>",...],"special":[ids...]}   (once per connection, before any
//   "json_schema" request: the vocabulary as bytes for structured output; "gen" then takes "json_schema":{...}
//   (the token stream is constrained to it, src/grammar.h) and "think_end":ID (unconstrained until that token))
// and writes events: {"id":N,"t":[tok]} per token, then {"id":N,"done":"stop|length|...","p":P,"c":C,"cached":K}.
// One client (the HTTP front end) at a time; it multiplexes requests by id.
// usage: inferd <model_dir> <socket> [max_running=64] [step_tokens=2048] [kv_gib=96] [snapshots=96] [max_ctx=131072]
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>
#include "../src/scheduler.h"

// Image features arrive as base64 little-endian bf16. Strict (Codex review 2026-11-03, R03): anything but the alphabet
// and trailing '=' padding, or an odd byte count, is a bad request (stray bytes used to be skipped and a trailing odd
// byte dropped, so a malformed payload still produced rows).
static std::vector<uint16_t> base64_u16(const std::string& s) {
  static int8_t lut[256];
  static bool init = false;
  if (!init) {
    memset(lut, -1, sizeof lut);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) lut[(uint8_t)a[i]] = (int8_t)i;
    init = true;
  }
  std::vector<uint8_t> out;
  out.reserve(s.size() * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  size_t n = s.size();
  while (n && s[n - 1] == '=' && s.size() - n < 2) n--;
  for (size_t i = 0; i < n; i++) {
    const unsigned char c = s[i];
    if (lut[c] < 0) throw std::runtime_error("embeds: not base64");
    acc = (acc << 6) | lut[c];
    bits += 6;
    if (bits >= 8) { bits -= 8; out.push_back((acc >> bits) & 0xff); }
  }
  if (out.size() % 2) throw std::runtime_error("embeds: odd byte count");
  std::vector<uint16_t> r(out.size() / 2);
  memcpy(r.data(), out.data(), r.size() * 2);
  return r;
}

static std::mutex mu;
static std::mutex sch_mu;  // held by the main thread while it works on the scheduler (the shutdown flush takes it)
static std::atomic<bool> stopping{false};
static std::condition_variable cv;
// A request line, parsed (and, for structured output, its schema compiled) on the reader thread, off the GPU loop.
struct Msg {
  json::Value v;
  std::shared_ptr<const gram::Grammar> grammar;
  std::string grammar_err;
  size_t bytes = 0;     // what it holds against the inbox budget
  std::string reject;   // refused by the reader (oversized line, inbox full): fail request reject_id with this
  uint64_t reject_id = 0;
};
static std::deque<Msg> inbox;
// Bounds on what the reader buffers (Codex review 2026-11-03, R02): one request line (INFERD_MAX_LINE_MIB; default:
// a max_ctx prompt of image rows, so every request the engine could serve fits) and the parsed requests waiting for
// the GPU loop (INFERD_INBOX_MIB; at least one maximal line). A line over its limit is discarded as it streams in and
// its request fails (the front end writes "id" last, so it is read from the line's tail); a "gen" that would overfill
// the inbox fails at once with "error: overloaded". Cancels and control ops always pass. Both are per-request errors.
static size_t max_line = 256u << 20, max_inbox = 512u << 20, inbox_bytes = 0;

// Compiled schemas, keyed by their compact JSON text (agents repeat their schema every turn); LRU, 64 entries.
// Compilation is bounded (gram::kMax*); any error becomes the request's error, never a crash of the engine.
static void compile_schema(Msg& m) {
  static std::mutex cmu;
  static std::list<std::pair<std::string, std::shared_ptr<const gram::Grammar>>> lru;
  try {
    std::string key = gram::dump(m.v["json_schema"]);
    if (key.size() > gram::kMaxSchemaBytes) throw gram::Error("schema is larger than 64 KiB");
    {
      std::lock_guard<std::mutex> l(cmu);
      for (auto it = lru.begin(); it != lru.end(); ++it)
        if (it->first == key) { lru.splice(lru.begin(), lru, it); m.grammar = it->second; return; }
    }
    auto g = std::make_shared<const gram::Grammar>(m.v["json_schema"]);
    std::lock_guard<std::mutex> l(cmu);
    lru.emplace_front(std::move(key), g);
    if (lru.size() > 64) lru.pop_back();
    m.grammar = g;
  } catch (std::exception& e) {  // the event line is JSON built with %s in 256 bytes: plain text, short
    for (const char* p = e.what(); *p && m.grammar_err.size() < 140; p++)
      m.grammar_err += (std::isalnum((unsigned char)*p) || strchr(" .,:;_()+*?[]{}^$|/-", *p)) ? *p : '?';
  }
}
static bool client_gone = false;

// The request id at the end of a (discarded) line: the last "id": N in its tail, or 0.
static uint64_t tail_id(const std::string& t) {
  size_t k = t.rfind("\"id\"");
  if (k == std::string::npos) return 0;
  k += 4;
  while (k < t.size() && (t[k] == ' ' || t[k] == ':')) k++;
  uint64_t id = 0;
  for (int d = 0; k < t.size() && isdigit((unsigned char)t[k]) && d < 19; k++, d++) id = id * 10 + (t[k] - '0');
  return id;
}

static void push_msg(Msg m) {
  {
    std::lock_guard<std::mutex> l(mu);
    if (m.reject.empty() && m.v["op"].s == "gen" && !inbox.empty() && inbox_bytes + m.bytes > max_inbox) {
      Msg r;
      r.reject = "error: overloaded: the engine's request inbox is full (INFERD_INBOX_MIB)";
      try { r.reject_id = (uint64_t)m.v["id"].i64(); } catch (...) {}
      m = std::move(r);
    }
    inbox_bytes += m.bytes;
    inbox.push_back(std::move(m));
  }
  cv.notify_one();
}

static void handle_line(const std::string& line) {
  if (line.empty()) return;
  try {
    Msg m{json::parse(line)};
    // Parsed requests are larger than their text: every array element is a json::Value.
    m.bytes = line.size();
    for (const char* k : {"tokens", "mrope", "stop", "snap_at"}) m.bytes += m.v[k].a.size() * sizeof(json::Value);
    if (m.v["op"].s == "gen" && m.v.has("json_schema")) compile_schema(m);
    push_msg(std::move(m));
  } catch (std::exception& e) {
    fprintf(stderr, "bad request: %s\n", e.what());
  }
}

static void reader(int fd) {
  std::string buf, tail;
  bool skipping = false;  // discarding an oversized line; tail keeps its last bytes (the id)
  char tmp[65536];
  for (;;) {
    ssize_t n = read(fd, tmp, sizeof tmp);  // the socket is non-blocking (flush_out): wait for input with poll
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      pollfd p{fd, POLLIN, 0};
      poll(&p, 1, -1);
      continue;
    }
    if (n <= 0) break;
    for (size_t i = 0; i < (size_t)n;) {
      const char* nl = (const char*)memchr(tmp + i, '\n', n - i);
      const size_t end = nl ? nl - tmp : (size_t)n;
      if (!skipping && buf.size() + (end - i) > max_line) {
        skipping = true;
        tail = buf.substr(buf.size() > 64 ? buf.size() - 64 : 0);
        std::string().swap(buf);
      }
      if (skipping) {
        tail.append(tmp + i, end - i);
        if (tail.size() > 64) tail.erase(0, tail.size() - 64);
      } else {
        buf.append(tmp + i, end - i);
      }
      i = end;
      if (!nl) break;
      i++;
      if (skipping) {
        Msg r;
        r.reject = "error: bad request: request line exceeds INFERD_MAX_LINE_MIB (" + std::to_string(max_line >> 20) + " MiB)";
        r.reject_id = tail_id(tail);
        fprintf(stderr, "inferd: discarded an oversized request line (id%llu)\n", (unsigned long long)r.reject_id);
        push_msg(std::move(r));
        skipping = false;
        tail.clear();
      } else {
        handle_line(buf);
        buf.clear();
      }
    }
  }
  std::lock_guard<std::mutex> l(mu);
  client_gone = true;
  cv.notify_one();
}

// Event output (R02): the GPU loop never blocks on the client. Events queue in `out` and go out on the non-blocking
// socket (O_NONBLOCK: MSG_DONTWAIT alone still blocked in sendto on macOS AF_UNIX while the reader sat in read) after
// every step; a client that lets more than INFERD_OUT_MIB (64) queue up, or takes nothing for INFERD_OUT_STALL_S (30)
// while events wait, is dropped: its requests are cancelled as on a disconnect. Returns false when the client is gone.
static uint64_t out_sent = 0;  // bytes the client has taken (progress for the stall deadline)
static bool flush_out(int fd, std::string& out, size_t& off) {
  while (off < out.size()) {
    ssize_t n = send(fd, out.data() + off, out.size() - off, 0);
    if (n > 0) { off += (size_t)n; out_sent += (uint64_t)n; continue; }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    return false;
  }
  if (off == out.size()) out.clear(), off = 0;
  else if (off > (1u << 20)) out.erase(0, off), off = 0;
  return true;
}

// Typed request fields (R03): ".n" read a string or an object as 0 and ".s" a number as "".
static double num_field(const json::Value& v, const char* k, double lo, double hi) {
  const json::Value& x = v[k];
  if (x.type != json::Value::Num || !std::isfinite(x.n) || x.n < lo || x.n > hi)
    throw std::runtime_error(std::string(k) + " must be a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return x.n;
}
static const std::string& str_field(const json::Value& v, const char* k, size_t max = 256) {
  const json::Value& x = v[k];
  if (x.type != json::Value::Str || x.s.size() > max)
    throw std::runtime_error(std::string(k) + " must be a string of at most " + std::to_string(max) + " bytes");
  return x.s;
}
static const std::vector<json::Value>& arr_field(const json::Value& v, const char* k) {
  static const std::vector<json::Value> none;
  if (!v.has(k)) return none;
  if (v[k].type != json::Value::Arr) throw std::runtime_error(std::string(k) + " must be an array");
  return v[k].a;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) { fprintf(stderr, "usage: inferd <model_dir> <socket> [max_running] [step_tokens] [kv_gib] [snapshots] [max_ctx]\n"); return 1; }
  // Launch arguments are range-checked (Codex review 2026-11-03, R03: atoi/atof fed unchecked values into the
  // allocation sizes and int arithmetic below).
  int max_running = argc > 3 ? (int)knob_int("max_running", argv[3], 1, 4096) : 64;
  int step_tokens = argc > 4 ? (int)knob_int("step_tokens", argv[4], 1, 1 << 20) : 2048;
  double kv_gib = argc > 5 ? knob_num("kv_gib", argv[5], -1, 1 << 20) : 96;
  int snapshots = argc > 6 ? (int)knob_int("snapshots", argv[6], 0, 1 << 16) : 96;
  int max_ctx = argc > 7 ? (int)knob_int("max_ctx", argv[7], KV_BLOCK, 1 << 24) : 131072;
  if (strlen(argv[2]) >= sizeof(sockaddr_un::sun_path)) { fprintf(stderr, "inferd: socket path too long\n"); return 1; }
  signal(SIGPIPE, SIG_IGN);
  sigset_t term;  // SIGTERM / SIGINT go to the shutdown thread (blocked here, before any thread exists)
  sigemptyset(&term);
  sigaddset(&term, SIGTERM);
  sigaddset(&term, SIGINT);
  pthread_sigmask(SIG_BLOCK, &term, nullptr);
  @autoreleasepool {
    Gpu g(std::string(getenv("INFERD_HOME") ? getenv("INFERD_HOME") : ".") + "/build/core.metallib");
    Checkpoint ck(argv[1]);
    Qwen35Config cfg(ck.config);
    bool kv8 = getenv("KV8") ? atoi(getenv("KV8")) != 0 : !cfg.gemma;  // must match Qwen35::kv8: int8 values + a float scale
    const bool kv4 = cfg.kv4_wanted(kv8);  // must match Qwen35::kv4 (KV4=1): signed nibbles + KV8's scales
    size_t block_bytes = cfg.kv_block_bytes(kv8, kv4);
    // kv_gib <= 0: size the pool from what the model, drafter, state slots and scratch leave of Metal's working-set
    // budget (recommendedMaxWorkingSetSize, the stock GPU limit), less a margin (KV_MARGIN_GIB, 1.5), as Splash does.
    const bool auto_kv = kv_gib <= 0;
    if (!auto_kv && kv_gib * (1ull << 30) / block_bytes > (1 << 28)) { fprintf(stderr, "inferd: kv_gib too large\n"); return 1; }
    int n_blocks = auto_kv ? 4 : (int)(kv_gib * (1ull << 30) / block_bytes);
    int max_blocks = max_ctx / KV_BLOCK + 1;
    const double gib = 1ull << 30;
    // PREFILL_STEP: rows a step may hold while nothing decodes (Scheduler::prefill_step); default 8,192 for Flash-Next
    const int prefill_step = (int)env_knob("PREFILL_STEP", cfg.qwen4 ? 8192 : 0, 0, 1 << 20);
    Qwen35::pf8_defer = true;  // PF8 auto: decided below, once the KV need is known
    Qwen35 m(g, ck, std::max(step_tokens, prefill_step) + max_running * kMaxBlock, max_running, max_running + snapshots,
             n_blocks, max_blocks);
    const double after_model = g.dev.currentAllocatedSize / gib - n_blocks * (double)block_bytes / gib;  // less the KV pool
    const double pf8_in_model = m.pf8_bytes / gib;  // copies made at construction (PF8=1 forced)
    Drafter* drafter = nullptr;
    Mtp* mtp = nullptr;
    if (m.has_mtp) {  // MTP=1 or DRAFT=mtp on a model with an MTP head (Flash-Next): it drafts instead of DFlash2
      mtp = new Mtp(g, m);
      printf("inferd: MTP head drafter (block %d: %d drafts per cycle)\n", mtp->block, mtp->block - 1);
      // Verify outputs for the widest block a sequence may take: context-copy chains (COPY_DRAFT, src/copydraft.h)
      // verify up to kMaxBlock rows. Sized for MTP's block, 8 agents on 12-row copy chains (96 output rows) wrote
      // logits past the 64-row buffer and corrupted other sequences' predictions (2026-11-02).
      m.enable_spec({}, !getenv("COPY_DRAFT") || atoi(getenv("COPY_DRAFT")) ? kMaxBlock : mtp->block);
    } else if (getenv("DRAFT") && std::string(getenv("DRAFT")) == "mtp") {
      fprintf(stderr, "inferd: DRAFT=mtp but this model has no MTP head; speculation off\n");
    } else if (getenv("DRAFT") && *getenv("DRAFT")) {
      static Checkpoint dck(getenv("DRAFT"));
      // Verify rows the block policy can plan in a step: sizes the drafter's block scratch, the logits and the GDN
      // tapes (SPEC_ROWS overrides; 0 = the old sizing, every sequence at block 16 and tapes for every step row).
      m.spec_rows = getenv("SPEC_ROWS") ? (int)env_knob("SPEC_ROWS", 0, 0, max_running * kMaxBlock)
                                        : Scheduler::spec_rows_cap(max_running, Drafter::env_block());
      drafter = new Drafter(g, m, dck);
      printf("inferd: DFlash2 drafter loaded (block %d)\n", drafter->block);
      // The verify scratch (logits for every block row, hidden captures, GDN tapes) before the KV pool is sized, so
      // auto sizing counts it (Codex review F03; it was allocated later by the Scheduler, on top of the budget).
      const size_t before_spec = g.dev.currentAllocatedSize;
      m.enable_spec(drafter->target_layers, drafter->block, m.spec_rows);
      printf("inferd: drafter memory: %.2f GiB own (context rings %.2f for %d slots%s), %.2f GiB verify scratch (GDN tapes "
             "%.2f for %d rows; %d output rows)\n", drafter->mem_total / gib, drafter->mem_rings / gib, m.n_slots,
             drafter->ring8 ? ", 8-bit" : "",
             (g.dev.currentAllocatedSize - before_spec) / gib, m.mem_tapes / gib, m.tape_rows, m.max_out);
    }
    const double budget = g.dev.recommendedMaxWorkingSetSize / gib;
    // The Neural Engine split's surfaces and the ANE service's own buffers are outside Metal's count (ANE_MARGIN_GIB,
    // 0.5: on the M6 wired memory with the split was ~0.3 GiB above Metal's count; 2.5 starved the KV pool, 4x slower).
    const double ane_margin = env_knob_f("ANE_MARGIN_GIB", 0.5, 0, 1 << 20);
    // Headroom: 1.5 GiB on small machines (their budgets fit as measured); 12% of the budget above 64 GiB, because
    // filling the Studio's 224 GiB to 222.5 made requests stall (a 20-token request took 37 s; a first request
    // hung on 2026-11-01) as the OS, file cache and front end compete for the rest (Codex review F03).
    const double base_margin = budget > 64 ? 0.12 * budget : 1.5;
    // Images (VISION=1, set when the server will take image requests): the front end runs the vision tower in its own
    // MLX process, outside this budget, so leave room for its weights and activations (VISION_MARGIN_GIB: 1 on
    // budgets up to 64 GiB, 0 above, where the 12% margin covers it; measured 1.15 GiB on the M6). Without it the M4
    // Pro's KV pool grew into the space the drafter cut freed and 4 concurrent OCR agents hit a GPU out-of-memory
    // step failure (2026-11-02). Opt-in, because the front end loads the tower only on the first image request: a
    // reserve keyed on the checkpoint alone cost text-only fleets on the 32 GB M6 ~40% of their KV pool.
    const bool vision_on = getenv("VISION") && atoi(getenv("VISION")) != 0 && ck.config.has("vision_config");
    const double vision_margin = env_knob_f("VISION_MARGIN_GIB", !vision_on ? 0.0 : budget > 64 ? 0.0 : 1.0, 0, 1 << 20);
    const double margin = env_knob_f("KV_MARGIN_GIB", base_margin, 0, 1 << 20) + (m.ane ? ane_margin : 0) + vision_margin;
    // PF8 (group-free int8 prefill GEMMs, Qwen35::pf8): forced by PF8=0/1; otherwise on when the GPU has neural
    // accelerators and the KV pool left after the int8 copies still holds PF8_KV_FRAC (1.0) x max_running x max_ctx
    // tokens (auto KV), or the budget keeps 8% headroom beside an explicit pool. 64 GB Macs: the 27B's 22.7 GiB of
    // copies would take most of a ~28 GiB pool, so it stays off there unless forced.
    // QA8 (int8 activations x the packed 4-bit weights, Qwen35::qa8; no copy): QA8=0/1 forces; auto = on with neural
    // accelerators unless PF8=1 forces PF8, off for Flash-Next. Decided before PF8 (an auto PF8 stays off where QA8 is on:
    // QA8 is faster on the Studio and needs no copies) and before the KV pool, so its activation scratch is counted.
    std::string qa8_why;
    {
      const bool forced = m.qa8_on >= 0;
      if (!forced) m.qa8_on = m.qa8_auto();
      if (m.qa8_on > 0) m.qa8_prepare();
      char buf[192];
      if (forced) snprintf(buf, sizeof buf, "%s (QA8=%d)", m.qa8_on > 0 ? "on" : "off", m.qa8_on);
      else snprintf(buf, sizeof buf, "%s (auto: %s)", m.qa8_on > 0 ? "on" : "off",
                    !m.g.nax ? "no neural accelerators" : m.cfg.qwen4 ? "not for Flash-Next" : m.pf8_on > 0 ? "PF8=1 forces PF8"
                    : m.qa8_on > 0 ? "neural accelerators, no weight copy" : "no flipped 4-bit linears (W4A8=0 or not 4-bit)");
      qa8_why = buf;
    }
    std::string pf8_why;
    {
      const double copies = m.pf8_estimate() / gib;
      const double need = (double)((long)max_running * max_ctx / KV_BLOCK + 1) * block_bytes / gib *
                          env_knob_f("PF8_KV_FRAC", 1.0, 0, 64);
      const double used0 = g.dev.currentAllocatedSize / gib - n_blocks * (double)block_bytes / gib;
      char buf[256];
      if (m.pf8_on < 0 && m.cfg.qwen4) {  // Flash-Next: opt-in (32k tail KL 0.0229 vs 0.0205 off, top-1 94.8 vs 95.5%)
        m.pf8_on = 0;
        snprintf(buf, sizeof buf, "off (Flash-Next default; PF8=1 to enable)");
      } else if (m.pf8_on < 0 && m.qa8_on > 0) {  // QA8 does the large steps without the copies
        m.pf8_on = 0;
        snprintf(buf, sizeof buf, "off (auto: QA8 is on; PF8=1 to force)");
      } else if (m.pf8_on < 0) {
        // the same margin the auto KV sizing below keeps (KV_MARGIN_GIB, the ANE split, VISION_MARGIN_GIB for a vision
        // tower in the front end), so PF8 never takes room a vision checkpoint needs
        const double room = auto_kv ? budget - used0 - margin - copies
                                    : budget - g.dev.currentAllocatedSize / gib - copies - 0.08 * budget - vision_margin;
        const bool ok = g.nax && (auto_kv ? room >= need : room >= 0);
        m.pf8_on = ok;
        snprintf(buf, sizeof buf, "%s (auto: %s; KV room after copies %.1f GiB, need %.1f for %d x %d tokens)", ok ? "on" : "off",
                 !g.nax ? "no neural accelerators" : ok ? "fits" : "the copies would shrink the KV pool", room, need, max_running, max_ctx);
      } else {
        snprintf(buf, sizeof buf, "%s (PF8=%d)", m.pf8_on > 0 ? "on" : "off", m.pf8_on);
      }
      pf8_why = buf;
      if (m.pf8_on > 0) m.pf8_prepare(m.pf8_list);
    }
    const double used = g.dev.currentAllocatedSize / gib - n_blocks * (double)block_bytes / gib;
    if (auto_kv) {
      n_blocks = (int)std::min<double>((budget - used - margin) * gib / block_bytes, 1 << 28);
      if (n_blocks < 64) {  // Codex review F03: never force a pool the budget does not hold
        fprintf(stderr, "inferd: no room for a KV pool: budget %.1f GiB, %.1f used, %.1f margin (lower max_running/snapshots, "
                        "or set kv_gib explicitly)\n", budget, used, margin);
        return 1;
      }
      m.alloc_kv(n_blocks);
      if (vision_margin > 0) printf("inferd: KV sizing leaves %.1f GiB for the front end's vision tower (VISION=1 / VISION_MARGIN_GIB)\n", vision_margin);
    }
    printf("inferd: memory: GPU budget %.1f GiB; model + state + scratch %.1f, drafter %.1f, PF8 %.1f, KV %.1f%s -> %.1f allocated\n",
           budget, after_model - pf8_in_model, used - after_model - (m.pf8_bytes / gib - pf8_in_model), m.pf8_bytes / gib, n_blocks * (double)block_bytes / gib, auto_kv ? " (auto)" : "",
           g.dev.currentAllocatedSize / gib);
    printf("inferd: %s KV, %d KV blocks (%.0f GiB, %.1fM tokens), %d slots (%.1f GiB state), max ctx %d\n", kv4 ? "4-bit" : kv8 ? "8-bit" : "bf16", n_blocks,
           n_blocks * (double)block_bytes / (1ull << 30), n_blocks * (double)KV_BLOCK / 1e6, m.n_slots,
           m.n_slots * (double)m.slot_bytes() / (1ull << 30), max_ctx);
    printf("inferd: PF8: %s, copies %.2f GiB%s, KV %.1f GiB\n", pf8_why.c_str(), (m.pf8_on > 0 ? m.pf8_bytes : m.pf8_estimate()) / gib,
           m.pf8_on > 0 ? "" : " (not made)", n_blocks * (double)block_bytes / gib);
    printf("inferd: QA8 (int8-activation prefill GEMM on the packed 4-bit weights): %s, scratch %.2f GiB, steps >= %d rows\n",
           qa8_why.c_str(), m.qa8_on > 0 ? m.qa8_bytes() / gib : 0.0, m.qa8_min);
    if (!getenv("CB_SPLIT")) m.cb_split = 16;
    Scheduler sch(m, g, max_running, step_tokens, 16, drafter, mtp);
    sch.ctx_limit = max_ctx;
    {  // request line and inbox bounds (R02, see Msg): a max_ctx prompt of tokens, as image rows when there is a tower
      const double row = 64 + (ck.config.has("vision_config") ? cfg.hidden * 2 * 4.0 / 3 : 0);
      const long long dflt = (long long)std::clamp(max_ctx * row / (1 << 20) + 16, 64.0, 4096.0);
      max_line = (size_t)env_knob("INFERD_MAX_LINE_MIB", dflt, 1, 1 << 16) << 20;
      max_inbox = (size_t)env_knob("INFERD_INBOX_MIB", std::max<long long>(512, dflt), 1, 1 << 16) << 20;
    }
    const size_t max_out = (size_t)env_knob("INFERD_OUT_MIB", 64, 1, 1 << 16) << 20;
    const double out_stall = env_knob_f("INFERD_OUT_STALL_S", 30, 0.1, 86400);
    sch.prefill_step = prefill_step;
    if (getenv("SPILL_DIR") && *getenv("SPILL_DIR") && (m.cfg.qwen4 || m.cfg.gemma)) {
      // Codex review R09: the tier serialises neither Flash-Next's PLE conv state nor Gemma-4's per-layer KV shapes
      fprintf(stderr, "inferd: SPILL_DIR ignored: the SSD tier does not support this model yet\n");
    } else if (getenv("SPILL_DIR") && *getenv("SPILL_DIR")) {
      double gib = env_knob_f("SPILL_GIB", 64, 0, 1 << 20);
      std::string id = std::string("model ") + argv[1] + "\ndraft " + (drafter ? getenv("DRAFT") : "none") +
                       " bits " + (getenv("DRAFT_BITS") ? getenv("DRAFT_BITS") : "4") +
                       "\ncontent " + SpillTier::content_id(argv[1]) + " " +  // replaced weights: a new identity (S07)
                       (drafter ? SpillTier::content_id(getenv("DRAFT")) : "none");
      sch.spill = new SpillTier(m, drafter, getenv("SPILL_DIR"), gib, id);
      if (sch.spill->persist)
        printf("inferd: SSD tier is persistent: %zu contexts recovered (%.1f GiB)\n", sch.spill->n_recovered,
               sch.spill->used / 1073741824.0);
      printf("inferd: SSD tier in %s (%.0f GiB, %.1f MiB per block + %.0f MiB state per context)\n", sch.spill->dir.c_str(), gib,
             sch.spill->block_bytes() / 1048576.0, sch.spill->state_bytes() / 1048576.0);
    }

    // Warm-up (WARMUP, default 1; 0 off), before the socket listens. The first request after a start paid for the
    // process's first use of its buffers on the GPU: the first command buffers that bind the weights (with PF8 on, also
    // the 22.7 GiB of int8 copies) wait ~0.3 s each on the CPU before the GPU starts (macOS making the allocations
    // resident: wired memory went 2.3 -> 29.7 GiB in that request). M5 Pro, PF8 on: TTFT 3.8-4.1 s for a 0.5k prompt
    // against 1.0 s for the second request (2026-11-03). It is once per process: a request after 5 minutes idle took
    // 0.98 s. Pipeline creation is not it (57 pipelines in 10 ms, from Metal's shader cache). Two throwaway greedy
    // requests on a scratch Scheduler run the paths a turn takes: a prompt that prefills in one step of >= PF8_MIN rows
    // (the PF8 or MLX qmm GEMMs, flash attention, GDN prefill) and a short tail chunk (W4A8) as the prompt-end snapshot
    // splits it, the snapshot copy, speculative verify steps up to a decode snapshot (SNAP_GEN, 128-token boundary),
    // then a 24-token prompt. The scratch Scheduler (its prefix trie, snapshots, block tuner, statistics) is discarded:
    // the real one starts with every slot and KV block free and an empty prefix cache. The state slots and KV blocks
    // the warm-up wrote are overwritten before they are read, as after any finished request.
    if (env_knob("WARMUP", 1, 0, 1)) {
      const auto t0 = std::chrono::steady_clock::now();
      const int p0 = g.pso_created;
      const double pms0 = g.pso_ms;
      Scheduler w(m, g, max_running, step_tokens, 16, drafter, mtp);
      w.ctx_limit = max_ctx;
      w.prefill_step = prefill_step;
      w.emit = [](const Event&) {};
      const int vocab_ok = std::min(m.cfg.vocab, 150000);  // ordinary text token ids
      // 637 at PF8_MIN 512: a 608-row step, a 29-row tail, a decode snapshot at 640 within the 8 generated tokens
      const int len0 = std::min((std::max(m.pf8_min, 32) + 32 + 127) / 128 * 128 - 3, std::max(32, max_ctx - 64));
      const int lens[2] = {len0, 24};
      int steps = 0;
      for (int k = 0; k < 2; k++) {
        Request r;
        r.id = k + 1;
        for (int i = 0; i < lens[k]; i++) r.prompt.push_back(1000 + (int)((i * 7919u + k * 104729u) % (unsigned)(vocab_ok - 1000)));
        r.max_tokens = 8;
        r.gen_snap = true;
        w.submit(std::move(r));
        while (!w.idle()) {
          @autoreleasepool { w.step(); }
          if (++steps > 256) { fprintf(stderr, "inferd: warm-up did not finish in 256 steps\n"); break; }
        }
      }
      while (!w.running.empty()) w.finish(w.running.size() - 1, "cancelled");
      w.waiting.clear();
      printf("inferd: warm-up: %d + %d-token prompts, 8 tokens each (%d steps; PF8 %s, %s), %d pipelines created "
             "(%.0f ms), %.2f s (WARMUP=0 disables)\n", lens[0], lens[1], steps, m.pf8_on > 0 ? "on" : "off",
             drafter ? "DFlash2 verify" : mtp ? "MTP verify" : "no drafter", g.pso_created - p0, g.pso_ms - pms0,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // Shutdown: with a persistent SSD tier, write every cached context it lacks first, so a restart resumes them.
    std::thread([&sch, term] {
      int sig = 0;
      sigwait(&term, &sig);
      stopping = true;
      std::lock_guard<std::mutex> l(sch_mu);
      if (sch.spill && sch.spill->persist) {
        auto t0 = std::chrono::steady_clock::now();
        size_t n = sch.spill_all();
        printf("inferd: flushed %zu contexts to the SSD tier in %.1f s (%zu on disk, %.1f GiB)\n", n,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), sch.spill->entries.size(),
               sch.spill->used / 1073741824.0);
      }
      fflush(stdout);
      fflush(stderr);
      _exit(0);
    }).detach();

    unlink(argv[2]);
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, argv[2], sizeof addr.sun_path - 1);
    if (bind(ls, (sockaddr*)&addr, sizeof addr) || listen(ls, 4)) { perror("socket"); return 1; }
    chmod(argv[2], 0600);  // the engine socket is for this user only (Codex review F13)
    printf("inferd: listening on %s\n", argv[2]);

    for (;;) {
      int fd = accept(ls, nullptr, nullptr);
      if (fd < 0) continue;
      fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);  // writes never block the GPU loop (flush_out)
      {
        std::lock_guard<std::mutex> l(mu);
        client_gone = false;
        inbox.clear();
        inbox_bytes = 0;
      }
      std::thread rd(reader, fd);
      std::string out;
      size_t out_off = 0;
      uint64_t seen_sent = out_sent;
      auto out_progress = std::chrono::steady_clock::now();
      sch.emit = [&](const Event& e) {
        char b[384];
        if (!e.done) {
          snprintf(b, sizeof b, "{\"id\":%llu,\"t\":[%d]}\n", (unsigned long long)e.id, e.tokens[0]);
        } else {
          snprintf(b, sizeof b, "{\"id\":%llu,\"done\":%s,\"p\":%d,\"c\":%d,\"cached\":%d",
                   (unsigned long long)e.id, Scheduler::jstr(e.reason).c_str(), e.prompt_tokens, e.completion_tokens, e.cached_tokens);
          out += b;
          out += e.extra;
          snprintf(b, sizeof b, "}\n");
        }
        out += b;
      };
      for (;;) {
        std::deque<Msg> msgs;
        bool gone;
        {
          std::unique_lock<std::mutex> l(mu);
          if (sch.idle() && out.empty()) cv.wait(l, [] { return !inbox.empty() || client_gone; });
          else if (sch.idle()) cv.wait_for(l, std::chrono::milliseconds(20), [] { return !inbox.empty() || client_gone; });
          msgs.swap(inbox);
          inbox_bytes = 0;
          gone = client_gone;
        }
        while (stopping) std::this_thread::sleep_for(std::chrono::seconds(1));  // the shutdown thread owns it now
        std::lock_guard<std::mutex> sl(sch_mu);
        for (auto& msg : msgs) try {  // a malformed field fails its request, never the engine (strict json.h throws)
          if (!msg.reject.empty()) {  // refused by the reader (R02)
            if (msg.reject_id) sch.emit(Event{msg.reject_id, {}, true, msg.reject, 0, 0, 0});
            continue;
          }
          const json::Value& v = msg.v;
          uint64_t id = (uint64_t)v["id"].i64();
          if (v["op"].s == "cancel") { sch.cancel(id); continue; }
          {  // agent contexts and sessions (Scheduler: "Agent contexts and sessions")
            const std::string& op = v["op"].s;
            std::string why, extra;
            if (op == "ctx_drop") why = sch.ctx_drop(str_field(v, "ctx"));
            else if (op == "ctx_offload") why = sch.ctx_offload(str_field(v, "ctx"));
            else if (op == "session_drop") why = sch.session_drop(str_field(v, "session"), false);
            else if (op == "session_offload") why = sch.session_drop(str_field(v, "session"), true);
            else if (op == "stats") why = "ok";
            if (!why.empty()) {
              extra = ",\"stats\":" + sch.stats_json();
              out += "{\"id\":" + std::to_string(id) + ",\"done\":" + Scheduler::jstr(why) + extra + "}\n";
              continue;
            }
          }
          if (v["op"].s == "vocab") {  // structured output: the vocabulary as bytes (src/grammar.h)
            auto vb = std::make_shared<gram::Vocab>();
            for (auto& h : v["hex"].a) {
              std::string b;
              for (size_t k = 0; k + 1 < h.s.size(); k += 2) b += (char)strtol(h.s.substr(k, 2).c_str(), nullptr, 16);
              vb->bytes.push_back(std::move(b));
            }
            vb->text.assign(vb->bytes.size(), 1);
            for (auto& t : v["special"].a) if (t.i() >= 0 && t.i() < (int)vb->text.size()) vb->text[t.i()] = 0;
            gram::vocab() = vb;
            printf("inferd: vocabulary for structured output: %zu tokens\n", vb->bytes.size());
            continue;
          }
          if (v.has("op") && v["op"].s != "gen") throw std::runtime_error("unknown op");
          // Typed fields (R03); ranges and token ids are checked by Scheduler::submit (Scheduler::invalid).
          Request r;
          r.id = id;
          if (!v.has("tokens")) throw std::runtime_error("tokens missing");
          for (auto& t : arr_field(v, "tokens")) r.prompt.push_back(t.i());
          if (v.has("max_tokens")) r.max_tokens = v["max_tokens"].i();
          if (v.has("temperature")) r.temperature = (float)num_field(v, "temperature", 0, 1e4);
          if (v.has("top_p")) r.top_p = (float)num_field(v, "top_p", 0, 1);
          if (v.has("top_k")) r.top_k = v["top_k"].i();
          if (v.has("seed")) { r.seed = (uint64_t)v["seed"].i64(); r.has_seed = true; }
          if (v.has("max_context")) r.max_context = v["max_context"].i();
          if (v.has("priority")) r.priority = std::clamp(v["priority"].i(), -10, 10);  // a bounded scale (Codex R06)
          for (auto& t : arr_field(v, "stop")) r.stop_ids.push_back(t.i());
          for (auto& t : arr_field(v, "snap_at")) r.snap_hints.push_back(t.i());
          if (v.has("gen_snap")) r.gen_snap = v["gen_snap"].i() != 0;  // decode snapshots (Scheduler::snap_gen)
          if (v.has("ctx")) {  // save a context: compute, snapshot, register (no output)
            r.prefill_only = true;
            r.ctx_name = str_field(v, "ctx");
            r.ctx_pin = v.has("pin") && v["pin"].i() != 0;
            r.ctx_ssd = v.has("ssd") && v["ssd"].i() != 0;
          }
          if (v.has("session")) r.session = str_field(v, "session");
          if (v.has("no_cache")) r.no_cache = v["no_cache"].i() != 0;
          for (auto& t : arr_field(v, "mrope")) r.mrope.push_back(t.i());
          if (v.has("rope_delta")) r.rope_delta = v["rope_delta"].i();
          if (v.has("embeds")) r.embeds = base64_u16(str_field(v, "embeds", SIZE_MAX));
          if (v.has("json_schema")) {  // structured output (src/grammar.h); a schema needs the "vocab" op first
            if (!gram::vocab()) {
              sch.emit(Event{id, {}, true, "error: json_schema needs the vocab op first", (int)r.prompt.size(), 0, 0});
              continue;
            }
            if (!msg.grammar) {
              sch.emit(Event{id, {}, true, "error: json_schema: " + msg.grammar_err, (int)r.prompt.size(), 0, 0});
              continue;
            }
            const auto& g = msg.grammar;
            const int think_end = v.has("think_end") ? v["think_end"].i() : -1;
            if (think_end < -1 || think_end >= m.cfg.vocab) throw std::runtime_error("think_end out of range");
            r.grammar = std::make_shared<gram::Matcher>(g, think_end);
            if (!g->loose.empty() && getenv("INFERD_DEBUG")) {
              fprintf(stderr, "id%llu json_schema: not enforced:", (unsigned long long)id);
              for (auto& l : g->loose) fprintf(stderr, " [%s]", l.c_str());
              fprintf(stderr, "\n");
            }
          }
          sch.submit(std::move(r));
        } catch (std::exception& ex) {
          uint64_t eid = 0;
          try { eid = (uint64_t)msg.v["id"].i64(); } catch (...) {}
          fprintf(stderr, "bad request id%llu: %s\n", (unsigned long long)eid, ex.what());
          std::string why = "error: bad request: ";  // plain text, as for json_schema errors
          for (const char* p = ex.what(); *p && why.size() < 160; p++)
            why += (std::isalnum((unsigned char)*p) || strchr(" .,:;_()+*?[]{}^$|/-", *p)) ? *p : '?';
          if (eid && msg.v["op"].s != "cancel" && msg.v["op"].s != "vocab") sch.emit(Event{eid, {}, true, why, 0, 0, 0});
        }
        if (!gone && !flush_out(fd, out, out_off)) gone = true;  // events go out around each step, never blocking it (R02)
        if (!gone) {
          @autoreleasepool { sch.step(); }
          if (!flush_out(fd, out, out_off)) gone = true;
          const auto now = std::chrono::steady_clock::now();
          if (out.empty() || out_sent != seen_sent) out_progress = now, seen_sent = out_sent;
          if (!gone && !out.empty() &&
              (out.size() - out_off > max_out || std::chrono::duration<double>(now - out_progress).count() > out_stall)) {
            fprintf(stderr, "inferd: client is not reading its events (%.1f MiB queued): dropping it, its requests cancelled\n",
                    (out.size() - out_off) / 1048576.0);
            gone = true;
          }
        }
        if (gone) {
          shutdown(fd, SHUT_RDWR);  // the reader returns; nothing more is written to this client
          while (!sch.running.empty()) sch.finish(sch.running.size() - 1, "cancelled");
          sch.waiting.clear();
          break;
        }
      }
      rd.join();
      close(fd);
      printf("inferd: client disconnected (%d cached prefix blocks, %d snapshots; %.2f tokens per spec cycle; %llu preempted)\n",
             (int)sch.cache.nodes.size(), sch.snapshot_count,
             sch.stat_cycles ? (double)sch.stat_committed / sch.stat_cycles : 0.0, (unsigned long long)sch.stat_preempted);
      if (sch.stat_grammar_fix)
        printf("inferd: structured output: %llu block tokens replaced by the grammar, %llu re-prefills\n",
               (unsigned long long)sch.stat_grammar_fix, (unsigned long long)sch.stat_grammar_rollback);
      if (sch.stat_copy_cycles)
        printf("inferd: copy drafts: %llu cycles, %.2f tokens per copy cycle\n", (unsigned long long)sch.stat_copy_cycles,
               (double)sch.stat_copy_committed / sch.stat_copy_cycles);
      if (sch.spill)
        printf("inferd: SSD tier: %llu spilled (%.1f GiB, %.1f s), %llu restored (%.1f GiB, %.1f s), %zu on disk\n",
               (unsigned long long)sch.spill->n_spilled, sch.spill->bytes_w / 1073741824.0, sch.spill->spill_s,
               (unsigned long long)sch.spill->n_restored, sch.spill->bytes_r / 1073741824.0, sch.spill->restore_s,
               sch.spill->entries.size());
    }
  }
}
