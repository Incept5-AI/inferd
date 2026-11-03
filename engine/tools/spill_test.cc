// Host test of the SSD tier (src/spill.h) with stub model buffers: round trip, checksums, quota accounting with token
// files, persistence/recovery, malformed names, the directory lock and content identity (Codex review 2026-11-02, S07);
// failed writes (quota exhaustion, a read-only directory, a disabled tier, a short write) are reported as failures and
// leave no entry, files or quota behind, so callers that acknowledge persistence can trust the status (2026-11-03, R05).
//   clang++ -std=c++20 -g -O1 -fsanitize=address,undefined -Isrc tools/spill_test.cc -o build/spill_test && build/spill_test
#include <sys/resource.h>
#include <sys/wait.h>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

constexpr int KV_BLOCK = 32;
struct Buf {
  std::vector<char> b;
  template <class T> T* ptr() { return (T*)b.data(); }
};
struct Cfg {
  int hkv = 2, dh = 8, conv_k = 4, lhv = 2, ldv = 4, ldk = 4;
  int conv_dim() const { return 16; }
};
struct Qwen35 {
  Cfg cfg;
  bool kv8 = false;
  std::vector<Buf> kc, vc, ks, vs, conv_state, ssm_state;
  static constexpr int NS = 8 * KV_BLOCK;  // 8 blocks
  size_t kv_elem() const { return 2; }
  size_t kv_ns() const { return NS; }
  Qwen35() {
    for (int l = 0; l < 2; l++) {
      kc.push_back({std::vector<char>((size_t)cfg.hkv * NS * cfg.dh * 2)});
      vc.push_back({std::vector<char>((size_t)cfg.hkv * NS * cfg.dh * 2)});
    }
    for (int l = 0; l < 3; l++) {
      conv_state.push_back({std::vector<char>(4 * (size_t)(cfg.conv_k - 1) * cfg.conv_dim() * 2)});
      ssm_state.push_back({std::vector<char>(4 * (size_t)cfg.lhv * cfg.ldv * cfg.ldk * 4)});
    }
  }
  void fill(int seed) {
    for (auto* v : {&kc, &vc, &conv_state, &ssm_state})
      for (auto& b : *v)
        for (size_t i = 0; i < b.b.size(); i++) b.b[i] = (char)(i * 31 + seed);
  }
  void wipe() {
    for (auto* v : {&kc, &vc, &conv_state, &ssm_state})
      for (auto& b : *v) memset(b.b.data(), 0, b.b.size());
  }
};
struct Drafter {
  int ring = 1, Hkv = 1, D = 1, nl = 0;
  std::vector<Buf> kr, vr;
};
#define SPILL_HOST_TEST
#include "../src/spill.h"

extern "C" const char* __asan_default_options() { return "exitcode=99"; }
static int fails = 0, total = 0;
#define CHECK(c, msg) do { total++; if (!(c)) { fails++; printf("FAIL: %s (line %d)\n", msg, __LINE__); } } while (0)

static std::vector<int> toks(int nblocks, int seed) {
  std::vector<int> t((size_t)nblocks * KV_BLOCK);
  for (size_t i = 0; i < t.size(); i++) t[i] = (int)(i * 7 + seed);
  return t;
}
static uint64_t chain_hash(const std::vector<int>& t) {
  uint64_t h = SpillTier::kSeed;
  for (size_t i = 0; i < t.size() / KV_BLOCK; i++) h = SpillTier::extend(h, t.data() + i * KV_BLOCK);
  return h;
}
static bool same_block(Qwen35& a, Qwen35& b, int blk_a, int blk_b) {
  const size_t per = (size_t)KV_BLOCK * a.cfg.dh * 2;
  for (size_t l = 0; l < a.kc.size(); l++)
    for (int h = 0; h < a.cfg.hkv; h++) {
      size_t oa = ((size_t)h * Qwen35::NS + (size_t)blk_a * KV_BLOCK) * a.cfg.dh * 2;
      size_t ob = ((size_t)h * Qwen35::NS + (size_t)blk_b * KV_BLOCK) * a.cfg.dh * 2;
      if (memcmp(a.kc[l].b.data() + oa, b.kc[l].b.data() + ob, per) || memcmp(a.vc[l].b.data() + oa, b.vc[l].b.data() + ob, per))
        return false;
    }
  return true;
}
static void flip_byte(const std::string& p, long off) {
  FILE* f = fopen(p.c_str(), "r+b");
  fseek(f, off, SEEK_SET);
  int c = fgetc(f);
  fseek(f, off, SEEK_SET);
  fputc(c ^ 0x5a, f);
  fclose(f);
}

int main() {
  char tmpl[] = "/tmp/spill_test.XXXXXX";
  std::string base = mkdtemp(tmpl);
  Qwen35 m, ref;
  m.fill(3), ref.fill(3);
  const auto t3 = toks(3, 1);
  const uint64_t h3 = chain_hash(t3);

  {  // 1. round trip, non-persistent
    unsetenv("SPILL_PERSIST");
    SpillTier st(m, nullptr, base, 1.0, "id A");
    CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "spill");
    CHECK(st.used == 3 * st.bfile() + st.sfile(), "quota counts block and state files with trailers");
    CHECK(SpillTier::file_size(st.path('b', st.entries[h3].bh[0])) == st.block_bytes() + SpillTier::kTrailer, "trailer");
    m.wipe();
    CHECK(st.load(st.entries.at(h3), {4, 5, 6}, 2), "load");
    CHECK(same_block(m, ref, 4, 0) && same_block(m, ref, 6, 2), "restored KV equals spilled KV");
    CHECK(!memcmp(m.ssm_state[1].b.data() + 2 * st.ssm_bytes(), ref.ssm_state[1].b.data() + 1 * st.ssm_bytes(),
                  st.ssm_bytes()), "restored state equals spilled state");
    // 2. a corrupted block file fails the restore and drops the entry
    flip_byte(st.path('b', st.entries[h3].bh[1]), 5);
    CHECK(!st.load(st.entries.at(h3), {4, 5, 6}, 2), "corrupt block rejected");
    CHECK(st.entries.empty() && st.n_corrupt == 1 && st.used == 0, "corrupt entry dropped, quota released");
    // a corrupted state file / truncated file
    m.fill(3);
    CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "re-spill");
    flip_byte(st.path('s', h3), 0);
    CHECK(!st.load(st.entries.at(h3), {-1, -1, -1}, 2), "corrupt state rejected");
    CHECK(st.entries.empty(), "dropped");
    CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "re-spill 2");
    truncate(st.path('s', h3).c_str(), (off_t)st.state_bytes());  // trailer cut off
    CHECK(!st.load(st.entries.at(h3), {-1, -1, -1}, 2), "truncated state rejected");
  }
  {  // 3. lock: a second engine on the same SPILL_DIR is refused (in a child: the tier exits)
    SpillTier st(m, nullptr, base, 1.0, "id A");
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
      freopen("/dev/null", "w", stderr);
      SpillTier other(m, nullptr, base, 1.0, "id A");
      _exit(0);
    }
    int ws = 0;
    waitpid(pid, &ws, 0);
    CHECK(WIFEXITED(ws) && WEXITSTATUS(ws) == 1, "second tier on the same directory refused");
  }
  {  // 4. persistence: token files counted, recovery, identity change, corrupt token file, junk names
    setenv("SPILL_PERSIST", "1", 1);
    m.fill(3);
    {
      SpillTier st(m, nullptr, base, 1.0, "id P");
      CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "persistent spill");
      CHECK(st.used == 3 * st.bfile() + st.sfile() + st.tfile(3), "quota counts the token file");
      CHECK(SpillTier::file_size(st.path('t', h3)) == st.tfile(3), "token file size");
    }
    {
      FILE* f = fopen((base + "/inferd-spill/tzzzzzzzzzzzzzzzz").c_str(), "w");  // non-hex: used to throw in stoull
      fclose(f);
      f = fopen((base + "/inferd-spill/b0123").c_str(), "w");
      fclose(f);
      SpillTier st(m, nullptr, base, 1.0, "id P");
      CHECK(st.n_recovered == 1 && st.used == 3 * st.bfile() + st.sfile() + st.tfile(3), "recovered with the same quota");
      m.wipe();
      CHECK(st.load(st.entries.at(h3), {4, 5, 6}, 2) && same_block(m, ref, 5, 1), "recovered entry restores");
    }
    flip_byte(base + "/inferd-spill/t" + [&] { char b[17]; snprintf(b, 17, "%016llx", (unsigned long long)h3); return std::string(b); }(), 4);
    {
      SpillTier st(m, nullptr, base, 1.0, "id P");
      CHECK(st.n_recovered == 0 && st.used == 0, "corrupt token file not recovered, its files removed");
      m.fill(3);
      CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "spill again");
    }
    {
      SpillTier st(m, nullptr, base, 1.0, "id P, weights replaced");
      CHECK(st.n_recovered == 0 && st.entries.empty(), "a different identity empties the tier");
    }
    // quota: a budget that fits the blocks + state but not the token file refuses the spill
    {
      SpillTier probe(m, nullptr, base, 1.0, "q");
      const double gib = (3 * probe.bfile() + probe.sfile() + SpillTier::kTrailer) / (double)(1ull << 30);
      probe.~SpillTier();
      new (&probe) SpillTier(m, nullptr, base, gib, "q2");
      CHECK(!probe.spill(h3, t3, {0, 1, 2}, 1) && probe.used == 0, "token file bytes count against the quota");
    }
    unsetenv("SPILL_PERSIST");
  }
  {  // 5. content identity
    std::string d = base + "/model";
    std::filesystem::create_directories(d);
    auto write = [&](const std::string& n, const std::string& s) { FILE* f = fopen((d + "/" + n).c_str(), "wb"); fwrite(s.data(), 1, s.size(), f); fclose(f); };
    std::string hdr = R"({"w":{"dtype":"U8","shape":[200000],"data_offsets":[0,200000]}})";
    uint64_t hl = hdr.size();
    std::string file(8, '\0');
    memcpy(file.data(), &hl, 8);
    file += hdr + std::string(200000, 'x');
    write("config.json", "{\"a\":1}");
    write("model.safetensors", file);
    std::string a = SpillTier::content_id(d);
    CHECK(a == SpillTier::content_id(d), "content id is stable");
    struct stat sb;
    stat((d + "/model.safetensors").c_str(), &sb);
    file[8 + hdr.size() + 199999] = 'y';  // last byte of the data: in the final sampled window
    write("model.safetensors", file);
    struct timespec ts[2] = {sb.st_atimespec, sb.st_mtimespec};
    utimensat(AT_FDCWD, (d + "/model.safetensors").c_str(), ts, 0);  // same size and mtime: only the content differs
    CHECK(a != SpillTier::content_id(d), "replaced data at the same size and mtime changes the id");
    write("config.json", "{\"a\":2}");
    CHECK(a != SpillTier::content_id(d), "config change changes the id");
  }
  {  // 6. Sum64 is independent of chunking
    std::vector<char> data(1000);
    for (size_t i = 0; i < data.size(); i++) data[i] = (char)(i * 13);
    Sum64 one, many;
    one.add(data.data(), data.size());
    for (size_t i = 0; i < data.size();) { size_t c = std::min<size_t>(1 + i % 37, data.size() - i); many.add(data.data() + i, c); i += c; }
    CHECK(one.done() == many.done(), "Sum64 chunking-independent");
  }
  {  // 7. R05: failed writes report false and leave nothing behind (entries, data files, quota)
    unsetenv("SPILL_PERSIST");
    m.fill(3);
    auto data_files = [&](const std::string& dir) {
      int n = 0;
      std::error_code ec;
      for (auto& f : std::filesystem::directory_iterator(dir, ec)) {
        const char c = f.path().filename().string()[0];
        n += c == 'b' || c == 's' || c == 't';
      }
      return n;
    };
    const auto t3b = toks(3, 2);
    const uint64_t h3b = chain_hash(t3b);
    {  // quota exhaustion: one chain fits; a second cannot evict the entry being restored (pinned) and is refused
      SpillTier probe(m, nullptr, base, 1.0, "q");
      const double gib = (3 * probe.bfile() + probe.sfile()) / (double)(1ull << 30);
      probe.~SpillTier();
      new (&probe) SpillTier(m, nullptr, base, gib, "q");
      CHECK(probe.spill(h3, t3, {0, 1, 2}, 1), "first chain fits the quota");
      probe.pinned = h3;
      CHECK(!probe.spill(h3b, t3b, {3, 4, 5}, 1), "quota exhausted: second chain refused");
      CHECK(probe.entries.size() == 1 && probe.has(h3) && !probe.has(h3b), "refused chain has no entry");
      CHECK(probe.used == 3 * probe.bfile() + probe.sfile() && data_files(probe.dir) == 4, "refused chain left no files or quota");
      probe.pinned = 0;
    }
    {  // a quota smaller than one chain
      SpillTier st(m, nullptr, base, 1e-9, "q");
      CHECK(!st.spill(h3, t3, {0, 1, 2}, 1) && st.entries.empty() && st.used == 0 && data_files(st.dir) == 0,
            "chain larger than the quota refused");
    }
    {  // disabled tier (SPILL_GIB=0): every write refused
      SpillTier st(m, nullptr, base, 0.0, "q");
      CHECK(!st.spill(h3, t3, {0, 1, 2}, 1) && st.entries.empty() && st.used == 0 && data_files(st.dir) == 0,
            "zero quota refuses every chain");
    }
    if (getuid() != 0) {  // permission error: a read-only tier directory (root ignores the mode)
      SpillTier st(m, nullptr, base, 1.0, "q");
      chmod(st.dir.c_str(), 0500);
      CHECK(!st.spill(h3, t3, {0, 1, 2}, 1), "read-only directory: spill fails");
      CHECK(st.entries.empty() && st.used == 0 && st.block_refs.empty(), "read-only directory: no entry or quota");
      chmod(st.dir.c_str(), 0700);
      CHECK(data_files(st.dir) == 0, "read-only directory: no files");
      CHECK(st.spill(h3, t3, {0, 1, 2}, 1), "writable again: spill works");
    }
    {  // short write: the state file is cut by RLIMIT_FSIZE after the block files were written; all are rolled back
      SpillTier st(m, nullptr, base, 1.0, "q");
      struct rlimit old, lim;
      getrlimit(RLIMIT_FSIZE, &old);
      signal(SIGXFSZ, SIG_IGN);
      lim = old;
      lim.rlim_cur = (rlim_t)std::min(st.bfile(), st.sfile()) - 1;
      setrlimit(RLIMIT_FSIZE, &lim);
      const bool ok = st.spill(h3, t3, {0, 1, 2}, 1);
      setrlimit(RLIMIT_FSIZE, &old);
      CHECK(!ok && st.entries.empty() && st.used == 0 && st.block_refs.empty(), "short write: spill fails, no entry");
      CHECK(data_files(st.dir) == 0, "short write: partial files removed");
    }
  }
  std::filesystem::remove_all(base);
  printf("spill_test: %d/%d passed\n", total - fails, total);
  return fails != 0;
}
