// SSD tier for the prefix cache (SPILL_DIR=<dir>, SPILL_GIB=<quota, default 64>). When the cache evicts a leaf that
// carries a GDN snapshot, the chain from the root is saved so a later prompt that extends it resumes in a fraction
// of a second instead of re-prefilling it. Content-addressed (2026-10-30): a KV block's contents depend only on the
// tokens up to its end, so each block is one file named by that prefix hash ("b<hash>": every attention layer's K, V
// and 8-bit scales), written once and shared by every chain through it; a snapshot's state (conv + GDN state and the
// drafter's context ring) is one file per chain ("s<hash>"). A spill writes the state plus only the blocks not on
// disk yet (each agent turn adds a few), a restore reads the state plus only the blocks memory no longer holds. Files
// are reference-counted by the entries using them and deleted oldest-entry-first above the quota. All copies run on
// the CPU between steps (shared-storage buffers, GPU idle). The index lives in memory. (The first version rewrote the
// whole chain per spill: a 16-agent fleet on the M5 Pro wrote 96 GiB of chains whose KV never left memory.)
// SPILL_PERSIST=1 keeps the directory across restarts: each entry also has a token file ("t<hash>", written last and
// renamed into place, so it implies a complete state file and blocks), and start-up rebuilds the index from the
// token files whose state and blocks are present at the right sizes. A MANIFEST holds the identity (model, drafter,
// KV precision, block and state geometry); any difference empties the directory. Otherwise it is emptied at start-up
// and on exit.
// Hardening (Codex review 2026-11-02, S07): every file ends in a 16-byte trailer with a checksum of its contents, written
// at spill time and verified on restore (token files on recovery), so a torn or corrupted file costs a re-prefill, never
// wrong KV/state; the quota counts token files and trailers; writes and reads loop over short transfers and EINTR; an
// exclusive flock on <SPILL_DIR>/inferd-spill.lock keeps a second engine out of the directory; and the manifest's
// identity includes content_id() of the model and drafter (config bytes, safetensors headers, sizes, mtimes and sampled
// data), so replaced weights at the same path empty the tier instead of restoring stale state. Durability contract:
// no fsync; a crash may leave torn files, which the checksums reject at restore (the entry is dropped).
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#ifndef SPILL_HOST_TEST  // tools/spill_test.cc supplies host stubs
#include "dflash.h"
#endif

// Streaming 64-bit checksum (xxHash64-style: four lanes over 32-byte stripes, ~10 GB/s); the result does not depend
// on how the input is split into add() calls.
struct Sum64 {
  static constexpr uint64_t P1 = 11400714785074694791ull, P2 = 14029467366897019727ull, P3 = 1609587929392839161ull;
  uint64_t v[4] = {P1 + P2, P2, 0, 0 - P1}, total = 0;
  unsigned char tail[32];
  size_t nt = 0;
  static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
  void stripe(const unsigned char* p) {
    for (int i = 0; i < 4; i++) {
      uint64_t x;
      memcpy(&x, p + 8 * i, 8);
      v[i] = rotl(v[i] + x * P2, 31) * P1;
    }
  }
  void add(const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    total += n;
    if (nt) {
      size_t c = std::min(n, 32 - nt);
      memcpy(tail + nt, p, c);
      nt += c, p += c, n -= c;
      if (nt < 32) return;
      stripe(tail);
      nt = 0;
    }
    for (; n >= 32; p += 32, n -= 32) stripe(p);
    memcpy(tail, p, n);
    nt = n;
  }
  uint64_t done() const {
    uint64_t h = rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18) + total;
    for (size_t i = 0; i < nt; i++) h = rotl(h ^ (tail[i] * P3), 11) * P1;
    h ^= h >> 33, h *= P2, h ^= h >> 29, h *= P3, h ^= h >> 32;
    return h;
  }
};

struct SpillTier {
  Qwen35& m;
  Drafter* d;
  std::string dir;
  size_t quota, used = 0;
  struct Entry {
    int depth;
    std::vector<int> tokens;      // depth * KV_BLOCK
    std::vector<uint64_t> bh;     // prefix hash at the end of each block
    uint64_t h;                   // prefix hash of the whole chain (the state file's name)
    uint64_t last;
    int prio = 0;  // 1: a named context or an agent session's latest turn (dropped after every prio-0 entry)
  };
  std::unordered_map<uint64_t, Entry> entries;   // prefix hash -> entry
  std::unordered_map<uint64_t, int> block_refs;  // block file -> entries using it
  uint64_t clock = 0, n_spilled = 0, n_restored = 0;
  double spill_s = 0, restore_s = 0;
  size_t bytes_w = 0, bytes_r = 0;

  bool persist = false;
  size_t n_recovered = 0;
  int lock_fd = -1;
  static constexpr size_t kTrailer = 16;  // "inferdS1" + Sum64 of the file's contents
  static constexpr char kMagic[9] = "inferdS1";
  size_t n_corrupt = 0;

  // Content identity of a checkpoint directory for the manifest: config.json's bytes, and per *.safetensors file (by
  // name) its size, mtime, header and four sampled 64 KiB windows of its data. Cheap (a few MiB read); catches
  // re-quantised or replaced weights at the same path, which the path alone did not (Codex review 2026-11-02, S07).
  static std::string content_id(const std::string& d) {
    Sum64 h;
    std::string cfg = slurp(d + "/config.json");
    h.add(cfg.data(), cfg.size());
    std::vector<std::string> names;
    if (DIR* dp = opendir(d.c_str())) {
      while (dirent* de = readdir(dp)) {
        std::string n = de->d_name;
        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) names.push_back(n);
      }
      closedir(dp);
    }
    std::sort(names.begin(), names.end());
    std::vector<char> buf(64 << 10);
    for (auto& n : names) {
      h.add(n.data(), n.size() + 1);
      int fd = open((d + "/" + n).c_str(), O_RDONLY | O_CLOEXEC);
      struct stat st;
      if (fd < 0 || fstat(fd, &st) != 0) { if (fd >= 0) close(fd); h.add("?", 1); continue; }
      const uint64_t meta[3] = {(uint64_t)st.st_size, (uint64_t)st.st_mtimespec.tv_sec, (uint64_t)st.st_mtimespec.tv_nsec};
      h.add(meta, sizeof meta);
      uint64_t hlen = 0;
      if (pread(fd, &hlen, 8, 0) == 8 && hlen <= (100ull << 20) && hlen + 8 <= (uint64_t)st.st_size) {
        std::string hdr(hlen, '\0');
        if (pread(fd, hdr.data(), hlen, 8) == (ssize_t)hlen) h.add(hdr.data(), hdr.size());
        const uint64_t data = 8 + hlen, span = (uint64_t)st.st_size - data;
        for (int k = 0; k < 4 && span; k++) {
          uint64_t off = data + (span > buf.size() ? (span - buf.size()) * k / 3 : 0);
          ssize_t r = pread(fd, buf.data(), std::min<uint64_t>(buf.size(), span), off);
          if (r > 0) h.add(buf.data(), (size_t)r);
        }
      }
      close(fd);
    }
    char out[24];
    snprintf(out, sizeof out, "%016llx", (unsigned long long)h.done());
    return out;
  }

  // The tier lives in <SPILL_DIR>/inferd-spill, which it owns: only that subdirectory is ever cleared or removed, so a
  // mistaken SPILL_DIR (a home or project directory) cannot be deleted (Codex review F04, 2026-11-01).
  SpillTier(Qwen35& m, Drafter* d, const std::string& base, double gib, const std::string& identity = "")
      : m(m), d(d), dir(base + "/inferd-spill"), quota((size_t)(gib * (1ull << 30))) {
    persist = getenv("SPILL_PERSIST") && atoi(getenv("SPILL_PERSIST"));
    // One engine per tier: an exclusive lock beside the directory (inside it would be deleted by a clear), held for the
    // engine's lifetime (the kernel releases it if the process dies).
    std::error_code lec;
    std::filesystem::create_directories(base, lec);
    const std::string lock = base + "/inferd-spill.lock";
    lock_fd = open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
      fprintf(stderr, "spill: %s is in use by another inferd (cannot lock %s: %s)\n", dir.c_str(), lock.c_str(),
              strerror(errno));
      exit(1);
    }
    char geo[200];
    snprintf(geo, sizeof geo, "kv8 %d block %zu state %zu kv_block %d planes format 2 (checksummed) w4a8 %s", (int)m.kv8,
             block_bytes(), state_bytes(), KV_BLOCK, getenv("W4A8") ? getenv("W4A8") : "default");
    if (m.kv4) snprintf(geo + strlen(geo), sizeof geo - strlen(geo), " kv4 rot %d", m.kv4_rot);  // (KV8 manifests unchanged)
    const std::string manifest = identity + "\n" + geo + "\n";
    if (persist && slurp(dir + "/MANIFEST") == manifest) {
      recover();
      return;
    }
    // Only clear a directory this engine created (its owner marker), or an empty one (Codex review R07).
    std::error_code ec;
    if (std::filesystem::exists(dir) && !std::filesystem::exists(dir + "/.inferd-owner") &&
        !std::filesystem::is_empty(dir, ec)) {
      fprintf(stderr, "spill: %s exists, is not empty and has no .inferd-owner marker: refusing to clear it\n", dir.c_str());
      exit(1);
    }
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::filesystem::permissions(dir, std::filesystem::perms::owner_all, ec);  // prompts and state: private
    std::ofstream(dir + "/.inferd-owner") << "inferd SSD tier\n";
    if (persist) {
      std::ofstream f(dir + "/MANIFEST");
      f << manifest;
    }
  }
  ~SpillTier() {
    if (!persist) std::filesystem::remove_all(dir);
    if (lock_fd >= 0) close(lock_fd);  // releases the flock
  }

  static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f ? std::string(std::istreambuf_iterator<char>(f), {}) : std::string();
  }
  static size_t file_size(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (size_t)st.st_size : SIZE_MAX;
  }
  // "<kind><16 hex digits>" -> the hash; false for anything else (recovery used to std::stoull, which throws).
  static bool parse_name(const std::string& n, char kind, uint64_t* h) {
    if (n.size() != 17 || n[0] != kind) return false;
    uint64_t v = 0;
    for (size_t i = 1; i < 17; i++) {
      int c = n[i], x = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
      if (x < 0) return false;
      v = v << 4 | (uint64_t)x;
    }
    *h = v;
    return true;
  }
  // On-disk sizes, trailers included; token files exist only with SPILL_PERSIST.
  size_t bfile() const { return block_bytes() + kTrailer; }
  size_t sfile() const { return state_bytes() + kTrailer; }
  size_t tfile(int depth) const { return persist ? (size_t)depth * KV_BLOCK * 4 + kTrailer : 0; }
  // A whole small file (a token file) with a valid trailer -> its contents; empty on any mismatch.
  static std::string slurp_checked(const std::string& p) {
    std::string raw = slurp(p);
    if (raw.size() < kTrailer || memcmp(raw.data() + raw.size() - kTrailer, kMagic, 8)) return {};
    Sum64 h;
    h.add(raw.data(), raw.size() - kTrailer);
    uint64_t want;
    memcpy(&want, raw.data() + raw.size() - 8, 8);
    if (h.done() != want) return {};
    raw.resize(raw.size() - kTrailer);
    return raw;
  }

  // Rebuild the index from the token files; drop entries with a missing or short file, then files no entry uses.
  void recover() {
    std::vector<std::pair<uint64_t, std::vector<int>>> found;
    std::error_code ec;
    for (auto& f : std::filesystem::directory_iterator(dir, ec)) {
      const std::string n = f.path().filename().string();
      uint64_t th;
      if (!parse_name(n, 't', &th)) continue;
      std::string raw = slurp_checked(f.path().string());  // checksum-verified, or empty
      if (raw.empty() || raw.size() % (KV_BLOCK * 4)) continue;
      std::vector<int> toks(raw.size() / 4);
      memcpy(toks.data(), raw.data(), raw.size());
      found.push_back({th, std::move(toks)});
    }
    std::sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.second.size() < b.second.size(); });
    for (auto& [h, toks] : found) {
      Entry e{(int)(toks.size() / KV_BLOCK), std::move(toks), {}, h, 0};
      uint64_t bh = kSeed;
      bool ok = file_size(path('s', h)) == sfile();
      for (int i = 0; i < e.depth && ok; i++) {
        bh = extend(bh, e.tokens.data() + (size_t)i * KV_BLOCK);
        e.bh.push_back(bh);
        ok = block_refs.count(bh) || file_size(path('b', bh)) == bfile();
      }
      if (!ok || bh != h) {
        unlink(path('t', h).c_str());
        continue;
      }
      used += sfile() + tfile(e.depth);
      for (uint64_t b : e.bh)
        if (block_refs[b]++ == 0) used += bfile();
      e.last = ++clock;  // shortest first: the deepest chains are the most recently useful guess
      entries[h] = std::move(e);
    }
    for (auto& f : std::filesystem::directory_iterator(dir, ec)) {
      const std::string n = f.path().filename().string();
      if (n.size() == 21 && n.compare(17, 4, ".tmp") == 0) { unlink(f.path().c_str()); continue; }
      uint64_t h;
      if (!parse_name(n, n.empty() ? '?' : n[0], &h)) continue;
      if ((n[0] == 'b' && !block_refs.count(h)) || ((n[0] == 's' || n[0] == 't') && !entries.count(h)))
        unlink(f.path().c_str());
    }
    n_recovered = entries.size();
    while (used > quota && drop_oldest()) {}
  }

  // Hash of the first `depth` blocks of tokens, extended one block at a time by `extend`.
  static uint64_t extend(uint64_t h, const int* t) {
    for (int i = 0; i < KV_BLOCK; i++) h = (h ^ (uint32_t)t[i]) * 1099511628211ull;
    return h;
  }
  static constexpr uint64_t kSeed = 1469598103934665603ull;

  size_t kv_bytes() const { return m.kv_bytes((size_t)KV_BLOCK * m.cfg.hkv * m.cfg.dh); }
  size_t sc_bytes() const { return m.kv8 ? (size_t)KV_BLOCK * m.cfg.hkv * 4 : 0; }
  size_t conv_bytes() const { return (size_t)(m.cfg.conv_k - 1) * m.cfg.conv_dim() * 2; }
  size_t ssm_bytes() const { return (size_t)m.cfg.lhv * m.cfg.ldv * m.cfg.ldk * 4; }
  size_t ring_bytes() const { return d ? d->slot_ring_bytes() : 0; }
  size_t block_bytes() const { return m.kc.size() * 2 * (kv_bytes() + sc_bytes()); }
  size_t state_bytes() const { return m.conv_state.size() * (conv_bytes() + ssm_bytes()) + (d ? d->nl * 2 * ring_bytes() : 0); }

  bool has(uint64_t h) const { return entries.count(h) != 0; }
  std::string path(char kind, uint64_t h) const {
    char b[32];
    snprintf(b, sizeof b, "/%c%016llx", kind, (unsigned long long)h);
    return dir + b;
  }

  // The regions of one block (all layers' K, V, scales) and of one state slot, in file order: f(ptr, bytes).
  template <class F> void block_regions(int b, F&& f) {
    // Head planes (Qwen35::kv_ns): a block is one contiguous run per kv head in each tensor.
    const int H = m.cfg.hkv;
    const size_t kb = kv_bytes() / H, sb = sc_bytes() / H, ns = m.kv_ns();
    for (size_t l = 0; l < m.kc.size(); l++)
      for (int h = 0; h < H; h++) {
        const size_t slot = h * ns + (size_t)b * KV_BLOCK;
        f(m.kc[l].ptr<char>() + slot * kb / KV_BLOCK, kb);
        f(m.vc[l].ptr<char>() + slot * kb / KV_BLOCK, kb);
        if (sb) {
          f(m.ks[l].ptr<char>() + slot * 4, sb);
          f(m.vs[l].ptr<char>() + slot * 4, sb);
        }
      }
  }
  template <class F> void state_regions(int slot, F&& f) {
    const size_t cs = conv_bytes(), ss = ssm_bytes();
    for (size_t i = 0; i < m.conv_state.size(); i++) {
      f(m.conv_state[i].ptr<char>() + (size_t)slot * cs, cs);
      f(m.ssm_state[i].ptr<char>() + (size_t)slot * ss, ss);
    }
    if (d) {
      const size_t rb = ring_bytes();
      for (int l = 0; l < d->nl; l++) {
        f(d->kr[l].ptr<char>() + (size_t)slot * rb, rb);
        f(d->vr[l].ptr<char>() + (size_t)slot * rb, rb);
      }
    }
  }

  // Write / read a file of regions through a 64 MiB staging buffer (large sequential I/O, no page cache). The file
  // ends in a trailer (magic + Sum64 of the contents); read_file verifies it and fails on a mismatch.
  static bool write_all(int fd, const char* p, size_t n) {
    while (n) {
      ssize_t w = write(fd, p, n);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) return false;
      p += w, n -= (size_t)w;
    }
    return true;
  }
  static ssize_t read_some(int fd, char* p, size_t n) {
    for (;;) {
      ssize_t r = read(fd, p, n);
      if (r < 0 && errno == EINTR) continue;
      return r;
    }
  }
  template <class R> bool write_file(const std::string& p, R&& regions) {
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    fcntl(fd, F_NOCACHE, 1);  // keep the page cache out of the way (on a 32 GB machine it competes with the GPU)
    static std::vector<char> buf(64 << 20);
    size_t fill = 0;
    bool ok = true;
    Sum64 sum;
    auto flush = [&]() {
      if (fill && ok) ok = write_all(fd, buf.data(), fill);
      fill = 0;
    };
    regions([&](const char* q, size_t n) {
      sum.add(q, n);
      while (n) {
        size_t c = std::min(n, buf.size() - fill);
        memcpy(buf.data() + fill, q, c);
        fill += c; q += c; n -= c;
        if (fill == buf.size()) flush();
      }
    });
    if (fill + kTrailer > buf.size()) flush();
    const uint64_t h = sum.done();
    memcpy(buf.data() + fill, kMagic, 8);
    memcpy(buf.data() + fill + 8, &h, 8);
    fill += kTrailer;
    flush();
    if (close(fd) != 0) ok = false;
    if (!ok) unlink(p.c_str());
    return ok;
  }
  template <class R> bool read_file(const std::string& p, R&& regions) {
    int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    fcntl(fd, F_NOCACHE, 1);
    static std::vector<char> buf(64 << 20);
    size_t have = 0, pos = 0;
    bool ok = true;
    Sum64 sum;
    auto take = [&](char* q, size_t n, bool hash) {
      while (n && ok) {
        if (pos == have) {
          pos = 0;
          ssize_t r = read_some(fd, buf.data(), buf.size());
          have = r > 0 ? (size_t)r : 0;
          if (r <= 0) { ok = false; return; }
        }
        size_t c = std::min(n, have - pos);
        memcpy(q, buf.data() + pos, c);
        if (hash) sum.add(q, c);
        q += c; pos += c; n -= c;
      }
    };
    regions([&](char* q, size_t n) { take(q, n, true); });
    char tr[kTrailer];
    take(tr, kTrailer, false);
    uint64_t want = 0;
    memcpy(&want, tr + 8, 8);
    if (ok && (memcmp(tr, kMagic, 8) || want != sum.done())) {
      fprintf(stderr, "spill: checksum mismatch in %s: discarding it\n", p.c_str());
      n_corrupt++;
      ok = false;
    }
    close(fd);
    return ok;
  }

  void drop(uint64_t h) {
    Entry& e = entries.at(h);
    unlink(path('t', h).c_str());  // first: without it a restart ignores the rest
    unlink(path('s', h).c_str());
    used -= sfile() + tfile(e.depth);
    for (uint64_t b : e.bh)
      if (--block_refs[b] == 0) {
        unlink(path('b', b).c_str());
        block_refs.erase(b);
        used -= bfile();
      }
    entries.erase(h);
  }
  uint64_t pinned = 0;  // an entry being restored (its hash): never dropped (Codex review F01)
  bool drop_oldest() {  // false when nothing can be dropped (empty, or only the pinned entry left)
    auto victim = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it)
      if (it->first != pinned && (victim == entries.end() || it->second.prio < victim->second.prio ||
                                  (it->second.prio == victim->second.prio && it->second.last < victim->second.last)))
        victim = it;
    if (victim == entries.end()) return false;
    drop(victim->first);
    return true;
  }

  // Save a chain: tokens of blocks.size() blocks, their KV (blocks[i], in memory now), the state in `slot`.
  bool spill(uint64_t h, const std::vector<int>& tokens, const std::vector<int>& blocks, int slot, int prio = 0) {
    if (has(h)) { entries[h].last = ++clock; entries[h].prio = std::max(entries[h].prio, prio); return true; }
    auto t0 = std::chrono::steady_clock::now();
    Entry e{(int)blocks.size(), tokens, {}, h, 0};
    uint64_t bh = kSeed;
    size_t fresh = 0;
    for (size_t i = 0; i < blocks.size(); i++) {
      bh = extend(bh, tokens.data() + i * KV_BLOCK);
      e.bh.push_back(bh);
      if (!block_refs.count(bh)) fresh++;
    }
    (void)fresh;
    // What this chain still needs, recounted after every eviction: dropping an entry can delete block files this
    // chain was going to share (Codex review F07/R07: 262 bytes written against a 200-byte quota).
    auto need = [&] {
      size_t f = 0;
      for (uint64_t b : e.bh) f += !block_refs.count(b);
      return f * bfile() + sfile() + tfile(e.depth);
    };
    while (used + need() > quota && drop_oldest()) {}
    if (used + need() > quota) return false;
    size_t written = 0;
    std::vector<uint64_t> created;  // block files this call wrote, removed again if a later write fails
    auto rollback = [&] {
      for (uint64_t b : created) {
        unlink(path('b', b).c_str());
        block_refs.erase(b);
        used -= bfile();
      }
    };
    for (size_t i = 0; i < blocks.size(); i++) {
      if (block_refs.count(e.bh[i])) continue;
      if (!write_file(path('b', e.bh[i]), [&](auto f) { block_regions(blocks[i], f); })) { rollback(); return false; }
      block_refs[e.bh[i]] = 0;
      created.push_back(e.bh[i]);
      used += bfile();
      written += bfile();
    }
    if (!write_file(path('s', h), [&](auto f) { state_regions(slot, f); })) { rollback(); return false; }
    used += sfile();
    written += sfile();
    if (persist) {  // the token file last, renamed into place: its presence implies the state and blocks are complete
      const std::string t = path('t', h), tmp = t + ".tmp";
      if (!write_file(tmp, [&](auto f) { f((const char*)tokens.data(), tokens.size() * 4); }) || rename(tmp.c_str(), t.c_str())) {
        unlink(tmp.c_str());
        unlink(path('s', h).c_str());
        used -= sfile();
        rollback();
        return false;
      }
      used += tfile(e.depth);
      written += tfile(e.depth);
    }
    for (uint64_t b : e.bh) block_refs[b]++;
    e.last = ++clock;
    e.prio = prio;
    entries[h] = std::move(e);
    n_spilled++;
    bytes_w += written;
    spill_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
  }

  // The deepest spilled prefix of `t` with min_depth < depth <= max_depth, or nullptr.
  const Entry* lookup(const std::vector<int>& t, int min_depth, int max_depth, uint64_t* hash_out) {
    uint64_t h = kSeed;
    const Entry* best = nullptr;
    for (int dpt = 1; dpt <= max_depth; dpt++) {
      h = extend(h, t.data() + (size_t)(dpt - 1) * KV_BLOCK);
      if (dpt <= min_depth) continue;
      auto it = entries.find(h);
      if (it != entries.end() && it->second.depth == dpt &&
          std::equal(it->second.tokens.begin(), it->second.tokens.end(), t.begin())) {
        best = &it->second;
        *hash_out = h;
      }
    }
    return best;
  }

  // Read an entry into `blocks` (one per depth; -1 = still in memory, skip) and state slot `slot`.
  bool load(const Entry& e, const std::vector<int>& blocks, int slot) {
    auto t0 = std::chrono::steady_clock::now();
    size_t read_bytes = 0;
    const uint64_t eh = e.h;
    for (size_t i = 0; i < blocks.size(); i++) {
      if (blocks[i] < 0) continue;
      if (!read_file(path('b', e.bh[i]), [&](auto f) { block_regions(blocks[i], f); })) {
        drop_users(e.bh[i]);  // a bad block file: every entry through it would restore wrong KV
        return false;
      }
      read_bytes += bfile();
    }
    if (!read_file(path('s', eh), [&](auto f) { state_regions(slot, f); })) {
      if (entries.count(eh)) drop(eh);
      return false;
    }
    read_bytes += sfile();
    entries.at(eh).last = ++clock;
    n_restored++;
    bytes_r += read_bytes;
    restore_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
  }
  // Drop every entry whose chain uses block file `b` (after a failed read of it).
  void drop_users(uint64_t b) {
    std::vector<uint64_t> users;
    for (auto& [h, e] : entries)
      if (std::find(e.bh.begin(), e.bh.end(), b) != e.bh.end()) users.push_back(h);
    for (uint64_t h : users) drop(h);
  }
};
