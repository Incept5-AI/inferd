// Safetensors loader. Indexes every tensor across a checkpoint's shards and reads tensors
// (or row-concatenations of several) straight into Metal buffers with pread.
#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include "gpu.h"
#include "json.h"

// Launch-time numeric knobs (argv and environment) that size allocations or feed integer arithmetic: the whole value
// must parse and lie in [lo, hi], or the engine stops at start-up with a message (Codex review 2026-11-03, R03: atoi
// and atof read "64k" as 64, "x" as 0 and "-1" as a huge allocation once cast to size_t).
inline long long knob_int(const char* name, const char* v, long long lo, long long hi) {
  char* end = nullptr;
  errno = 0;
  const long long x = v ? strtoll(v, &end, 10) : 0;
  if (!v || !*v || *end || errno || x < lo || x > hi) {
    fprintf(stderr, "%s=%s: expected an integer in [%lld, %lld]\n", name, v ? v : "", lo, hi);
    exit(1);
  }
  return x;
}
inline double knob_num(const char* name, const char* v, double lo, double hi) {
  char* end = nullptr;
  const double x = v ? strtod(v, &end) : 0;
  if (!v || !*v || *end || !std::isfinite(x) || x < lo || x > hi) {
    fprintf(stderr, "%s=%s: expected a number in [%g, %g]\n", name, v ? v : "", lo, hi);
    exit(1);
  }
  return x;
}
inline long long env_knob(const char* name, long long dflt, long long lo, long long hi) {
  const char* v = getenv(name);
  return v ? knob_int(name, v, lo, hi) : dflt;
}
inline double env_knob_f(const char* name, double dflt, double lo, double hi) {
  const char* v = getenv(name);
  return v ? knob_num(name, v, lo, hi) : dflt;
}

struct TensorInfo {
  int fd;
  std::string dtype;
  std::vector<int64_t> shape;
  size_t off, nbytes;  // absolute file offset of the data
};

struct Checkpoint {
  std::string dir;
  std::map<std::string, TensorInfo> t;
  json::Value config;

  static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }

  [[noreturn]] static void bad(const std::string& path, const std::string& what) {
    fprintf(stderr, "invalid safetensors file %s: %s\n", path.c_str(), what.c_str());
    exit(1);
  }
  static size_t dtype_size(const std::string& d) {
    static const std::map<std::string, size_t> sz = {
        {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1}, {"U16", 2}, {"I16", 2}, {"F16", 2},
        {"BF16", 2}, {"U32", 4}, {"I32", 4}, {"F32", 4}, {"U64", 8}, {"I64", 8}, {"F64", 8}};
    auto it = sz.find(d);
    return it == sz.end() ? 0 : it->second;
  }
  // Whole-range pread (EINTR and short reads retried); false at EOF or on error.
  static bool pread_all(int fd, void* dst, size_t n, size_t off) {
    for (size_t done = 0; done < n;) {
      ssize_t r = pread(fd, (char*)dst + done, std::min<size_t>(n - done, 1 << 28), off + done);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) return false;
      done += r;
    }
    return true;
  }

  // The header is untrusted input (Codex review 2026-11-02, S07): its length is checked against the file, and every
  // tensor's dtype, shape and data_offsets against each other and the data section before anything is read.
  void index_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) bad(path, strerror(errno));
    // INFERD_NOCACHE=1: read the checkpoint past the unified buffer cache (F_NOCACHE), so its pages are not held a
    // second time beside the Metal buffers they were copied into (2026-11-03, after TensorFold #1885).
    if (const char* nc = getenv("INFERD_NOCACHE"); nc && atoi(nc)) fcntl(fd, F_NOCACHE, 1);
    struct stat st;
    if (fstat(fd, &st) != 0) bad(path, strerror(errno));
    const uint64_t fsize = (uint64_t)st.st_size;
    uint64_t hlen = 0;
    if (fsize < 8 || !pread_all(fd, &hlen, 8, 0)) bad(path, "shorter than its 8-byte header length");
    if (hlen < 2 || hlen > (100ull << 20) || hlen > fsize - 8)
      bad(path, "header length " + std::to_string(hlen) + " does not fit the file (" + std::to_string(fsize) + " bytes)");
    std::string h(hlen, '\0');
    if (!pread_all(fd, h.data(), hlen, 8)) bad(path, "short read of the header");
    json::Value hv;
    try {
      hv = json::parse(h);
    } catch (std::exception& e) {
      bad(path, std::string("header: ") + e.what());
    }
    if (hv.type != json::Value::Obj) bad(path, "header is not a JSON object");
    const uint64_t data = fsize - 8 - hlen;
    for (auto& [k, v] : hv.o) {
      if (k == "__metadata__") continue;
      try {
        if (v.type != json::Value::Obj || v["dtype"].type != json::Value::Str || v["shape"].type != json::Value::Arr ||
            v["data_offsets"].type != json::Value::Arr || v["data_offsets"].a.size() != 2)
          throw std::runtime_error("needs dtype, shape and two data_offsets");
        TensorInfo ti{fd, v["dtype"].s, {}, 0, 0};
        uint64_t elems = 1;
        for (auto& s : v["shape"].a) {
          int64_t d = s.i64();
          if (d < 0 || (d && elems > UINT64_MAX / (uint64_t)d)) throw std::runtime_error("bad shape");
          ti.shape.push_back(d);
          elems *= (uint64_t)d;
        }
        uint64_t b = v["data_offsets"][0].u64(), e = v["data_offsets"][1].u64();
        if (b > e || e > data) throw std::runtime_error("data_offsets [" + std::to_string(b) + ", " + std::to_string(e) +
                                                        ") outside the " + std::to_string(data) + "-byte data section");
        size_t es = dtype_size(ti.dtype);
        if (!es) throw std::runtime_error("unsupported dtype " + ti.dtype);
        if (elems > UINT64_MAX / es || elems * es != e - b)
          throw std::runtime_error(ti.dtype + " shape holds " + std::to_string(elems) + " elements but the range is " +
                                   std::to_string(e - b) + " bytes");
        ti.off = 8 + hlen + b;
        ti.nbytes = e - b;
        t[k] = ti;
      } catch (std::exception& ex) {
        bad(path, "tensor " + k + ": " + ex.what());
      }
    }
  }

  explicit Checkpoint(const std::string& d) : dir(d) {
    try {
      config = json::parse(slurp(dir + "/config.json"));
    } catch (std::exception& e) {
      fprintf(stderr, "invalid %s/config.json: %s\n", dir.c_str(), e.what());
      exit(1);
    }
    DIR* dp = opendir(dir.c_str());
    if (!dp) { fprintf(stderr, "cannot open %s\n", dir.c_str()); exit(1); }
    while (dirent* de = readdir(dp)) {
      std::string n = de->d_name;
      if (n.size() < 12 || n.substr(n.size() - 12) != ".safetensors") continue;
      index_file(dir + "/" + n);
    }
    closedir(dp);
  }

  const TensorInfo& info(const std::string& name) const {
    auto it = t.find(name);
    if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    return it->second;
  }
  bool has(const std::string& name) const { return t.count(name) > 0; }

  // Every copy names its destination's capacity (Codex review 2026-11-03, R01: a self-consistent I64 [32] tensor read
  // into NGram::mult's 24 bytes was a 256-byte heap overflow). A tensor larger than the destination is a malformed
  // checkpoint: refused before anything is copied.
  void read_into(const TensorInfo& ti, void* dst, size_t cap) const {
    if (ti.nbytes > cap) {
      fprintf(stderr, "invalid checkpoint %s: a %s tensor of %zu bytes does not fit its %zu-byte destination\n", dir.c_str(),
              ti.dtype.c_str(), ti.nbytes, cap);
      exit(1);
    }
    if (!pread_all(ti.fd, dst, ti.nbytes, ti.off)) { perror("pread (checkpoint truncated or unreadable?)"); exit(1); }
  }
  [[noreturn]] void reject(const std::string& name, const std::string& what) const {
    fprintf(stderr, "invalid checkpoint %s: tensor %s: %s\n", dir.c_str(), name.c_str(), what.c_str());
    exit(1);
  }
  // Dimension `axis` of a tensor that must have rank `rank`, in [lo, hi] (int-sized: loaders do int arithmetic on it).
  int dim(const std::string& name, int rank, int axis, int64_t lo = 1, int64_t hi = 1 << 30) const {
    auto& ti = info(name);
    if ((int)ti.shape.size() != rank) reject(name, "expected rank " + std::to_string(rank) + ", got " + std::to_string(ti.shape.size()));
    const int64_t v = ti.shape[axis];
    if (v < lo || v > hi) reject(name, "dimension " + std::to_string(axis) + " is " + std::to_string(v));
    return (int)v;
  }
  void need_dtype(const std::string& name, const char* dt) const {
    if (info(name).dtype != dt) reject(name, std::string("expected ") + dt + ", got " + info(name).dtype);
  }

  BufRef load(Gpu& g, const std::string& name) const {
    auto& ti = info(name);
    id<MTLBuffer> b = g.alloc(ti.nbytes);
    read_into(ti, b.contents, b.length);
    return BufRef(b);
  }
  // A 2-byte (bf16) vector of exactly `n` elements (norm weights and the like: kernels read n values, R01).
  BufRef load(Gpu& g, const std::string& name, int64_t n) const {
    auto& ti = info(name);
    if (ti.nbytes != (size_t)n * 2 || dtype_size(ti.dtype) != 2)
      reject(name, "expected " + std::to_string(n) + " 16-bit values, got " + ti.dtype + " of " + std::to_string(ti.nbytes) + " bytes");
    return load(g, name);
  }
  // A float tensor as bf16 (F32 sources are rounded; e.g. Qwen3.5-9B stores A_log in F32).
  BufRef load_bf16(Gpu& g, const std::string& name) const {
    auto& ti = info(name);
    if (ti.dtype != "F32") return load(g, name);
    std::vector<float> f(ti.nbytes / 4);
    read_into(ti, f.data(), f.size() * 4);
    id<MTLBuffer> b = g.alloc(f.size() * 2);
    uint16_t* d = (uint16_t*)b.contents;
    for (size_t i = 0; i < f.size(); i++) {
      uint32_t u; memcpy(&u, &f[i], 4);
      u += 0x7fff + ((u >> 16) & 1);
      d[i] = (uint16_t)(u >> 16);
    }
    return BufRef(b);
  }
};

// An affine-quantized (4- or 8-bit, group 64) linear layer [N, K]: one buffer holding [Wq | scales | biases].
struct QLinear {
  BufRef wq, s, b;
  BufRef lp, lsb;  // lane4 (LANE): 1 KB code blocks per 32-column tile and 64-group, (s, b) pairs group-major
  int sbn = 0;     // lane4: the (s, b) table's row stride (the full matrix's N; a row view keeps it)
  int ldk = 0;     // a K-slice view (kslice_of): the full matrix's row length (MLX qmm only); 0 = K
  BufRef rp, rsb;  // qmst (QMST=1): 4-bit payloads per 32-column tile and 32-input step, (s, b) per tile and group
  bool flipped = false;  // W4A8/W8A8: values flipped in place to signed (4-bit q ^ 8, 8-bit q ^ 0x80); readers flip back
  BufRef r8, r8s;  // PF8 (large prefill steps): Hadamard-rotated int8 copy [N][K] and its per-row scales (kernels/pf8.metal)
  int N = 0, K = 0, bits = 4;
};

// Output rows [lo, lo + n) of a linear, sharing its buffers (a view: nothing is copied or flipped again).
inline QLinear rows_of(const QLinear& q, int lo, int n) {
  QLinear r = q;
  r.rp = r.rsb = BufRef();
  r.N = n;
  if (q.lp.buf != nil && lo % 32 == 0) {  // lane4 copy: tile-aligned views keep it
    r.lp = q.lp.at((size_t)lo / 32 * (q.K / 64) * 1024);
    r.lsb = q.lsb.at((size_t)lo * 4);
  } else {
    r.lp = r.lsb = BufRef();
  }
  if (q.r8.buf != nil) { r.r8 = q.r8.at((size_t)lo * q.K); r.r8s = q.r8s.at((size_t)lo * 4); }
  if (q.bits == 16) { r.wq = q.wq.at((size_t)lo * q.K * 2); r.s = r.b = r.wq; return r; }
  r.wq = q.wq.at((size_t)lo * q.K * q.bits / 8);
  r.s = q.s.at((size_t)lo * (q.K / 64) * 2);
  r.b = q.b.at((size_t)lo * (q.K / 64) * 2);
  return r;
}

// Input columns [k0, k0 + kn) of a 4-bit/8-bit linear (k0, kn multiples of 64): a view only MLX's qmm reads (its rows
// keep the full matrix's length, `ldk`). The GPU's share of an FFN split with the Neural Engine (src/ane.h).
inline QLinear kslice_of(const QLinear& q, int k0, int kn) {
  QLinear r = q;
  r.rp = r.rsb = r.lp = r.lsb = BufRef();
  r.ldk = q.ldk ? q.ldk : q.K;
  r.K = kn;
  r.wq = q.wq.at((size_t)k0 * q.bits / 8);
  r.s = q.s.at((size_t)k0 / 64 * 2);
  r.b = q.b.at((size_t)k0 / 64 * 2);
  return r;
}

// Load several quantized linears that share K as one fused [sum N, K] layer (rows concatenated).
inline QLinear load_qlinear(Gpu& g, const Checkpoint& ck, const std::vector<std::string>& prefixes) {
  QLinear q;
  if (ck.info(prefixes[0] + ".weight").dtype == "BF16") {
    // Unquantized checkpoint: bits = 16, wq holds the bf16 rows [N, K] (s, b unused).
    q.bits = 16;
    size_t bytes = 0;
    for (auto& p : prefixes) {
      auto& w = ck.info(p + ".weight");
      const int n = ck.dim(p + ".weight", 2, 0), k = ck.dim(p + ".weight", 2, 1);
      if (w.dtype != "BF16" || (q.K && q.K != k)) { fprintf(stderr, "fused bf16 mismatch at %s\n", p.c_str()); exit(1); }
      if (q.N > (1 << 30) - n) ck.reject(p + ".weight", "fused rows overflow");
      q.K = k;
      q.N += n;
      bytes += w.nbytes;
    }
    q.wq = BufRef(g.alloc(bytes));
    size_t o = 0;
    for (auto& p : prefixes) {
      auto& w = ck.info(p + ".weight");
      ck.read_into(w, q.wq.ptr<char>() + o, bytes - o);
      o += w.nbytes;
    }
    q.s = q.b = q.wq;
    return q;
  }
  size_t wq_bytes = 0, s_bytes = 0;
  for (auto& p : prefixes) {
    auto& w = ck.info(p + ".weight");
    if (w.dtype != "U32") { fprintf(stderr, "%s: expected quantized U32 weights, got %s\n", p.c_str(), w.dtype.c_str()); exit(1); }
    auto &sc = ck.info(p + ".scales"), &bi = ck.info(p + ".biases");
    if (w.shape.size() != 2 || sc.shape.size() != 2 || sc.shape[1] < 1 || sc.shape[0] != w.shape[0] ||
        bi.shape != sc.shape || bi.nbytes != sc.nbytes) {
      fprintf(stderr, "%s: weight/scales/biases shapes do not match a group-64 quantized [N, K]\n", p.c_str());
      exit(1);
    }
    // The kernels read scales and biases as bf16 (R01: only byte lengths and shapes were checked).
    ck.need_dtype(p + ".scales", "BF16");
    ck.need_dtype(p + ".biases", "BF16");
    const int n = ck.dim(p + ".weight", 2, 0), wk = ck.dim(p + ".weight", 2, 1), G = ck.dim(p + ".scales", 2, 1, 1, (1 << 30) / 64);
    // group 64: K from the scales, bits from how many K values each uint32 row packs
    int K = G * 64, bits = (int)((int64_t)wk * 32 / K);
    if ((bits != 4 && bits != 8) || (int64_t)wk * 32 != (int64_t)K * bits) {
      fprintf(stderr, "%s: unsupported %d-bit weights\n", p.c_str(), bits);
      exit(1);
    }
    if (q.K && (q.K != K || q.bits != bits)) { fprintf(stderr, "fused K/bits mismatch at %s\n", p.c_str()); exit(1); }
    if (q.N > (1 << 30) - n) ck.reject(p + ".weight", "fused rows overflow");
    q.K = K;
    q.bits = bits;
    q.N += n;
    wq_bytes += w.nbytes;
    s_bytes += ck.info(p + ".scales").nbytes;
  }
  id<MTLBuffer> buf = g.alloc(wq_bytes + 2 * s_bytes);
  q.wq = BufRef(buf, 0);
  q.s = BufRef(buf, wq_bytes);
  q.b = BufRef(buf, wq_bytes + s_bytes);
  size_t ow = 0, os = 0;
  for (auto& p : prefixes) {
    auto& w = ck.info(p + ".weight");
    auto& s = ck.info(p + ".scales");
    auto& b = ck.info(p + ".biases");
    ck.read_into(w, q.wq.ptr<char>() + ow, wq_bytes - ow);
    ck.read_into(s, q.s.ptr<char>() + os, s_bytes - os);
    ck.read_into(b, q.b.ptr<char>() + os, s_bytes - os);
    ow += w.nbytes;
    os += s.nbytes;
  }
  return q;
}

static inline float bf16_to_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f_to_bf16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
  return (uint16_t)(u >> 16);
}

// Load bf16 linears [N_i, K] and quantize them (rows concatenated) to MLX affine 8-bit (or 4-bit), group 64:
// w = s*q + b with q in [0,2^bits-1]; element j of each uint32 in bits (bits*j).
inline QLinear load_qlinear_bf16_q8(Gpu& g, const Checkpoint& ck, const std::vector<std::string>& names, int bits = 8) {
  QLinear q;
  q.bits = bits;
  if (bits != 4 && bits != 8) { fprintf(stderr, "%s: %d-bit quantization is not supported\n", names[0].c_str(), bits); exit(1); }
  for (auto& n : names) {
    auto& t = ck.info(n);
    if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[1] % 64) {
      fprintf(stderr, "%s: expected a 2-D BF16 tensor with K a multiple of 64\n", n.c_str());
      exit(1);
    }
    const int rows = ck.dim(n, 2, 0), k = ck.dim(n, 2, 1);
    if (q.K && q.K != k) { fprintf(stderr, "fused K mismatch at %s\n", n.c_str()); exit(1); }
    if (q.N > (1 << 30) - rows) ck.reject(n, "fused rows overflow");
    q.K = k;
    q.N += rows;
  }
  const float levels = (float)((1 << bits) - 1);
  size_t wq_bytes = (size_t)q.N * q.K * bits / 8, s_bytes = (size_t)q.N * (q.K / 64) * 2;
  id<MTLBuffer> buf = g.alloc(wq_bytes + 2 * s_bytes);
  q.wq = BufRef(buf, 0);
  q.s = BufRef(buf, wq_bytes);
  q.b = BufRef(buf, wq_bytes + s_bytes);
  uint8_t* W = q.wq.ptr<uint8_t>();
  uint16_t *S = q.s.ptr<uint16_t>(), *Bz = q.b.ptr<uint16_t>();
  size_t row = 0;
  for (auto& n : names) {
    auto& t = ck.info(n);
    std::vector<uint16_t> src((size_t)t.shape[0] * q.K);
    ck.read_into(t, src.data(), src.size() * 2);
    int rows = (int)t.shape[0], K = q.K;
    const uint16_t* srcp = src.data();
    dispatch_apply(rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
      const uint16_t* w = srcp + r * K;
      size_t R = row + r;
      for (int g0 = 0; g0 < K; g0 += 64) {
        float lo = INFINITY, hi = -INFINITY;
        for (int i = 0; i < 64; i++) { float v = bf16_to_f(w[g0 + i]); lo = std::min(lo, v); hi = std::max(hi, v); }
        float sc = (hi - lo) / levels;
        if (sc == 0) sc = 1e-8f;
        uint16_t sb = f_to_bf16(sc), bb = f_to_bf16(lo);
        float scr = bf16_to_f(sb), lor = bf16_to_f(bb);
        S[R * (K / 64) + g0 / 64] = sb;
        Bz[R * (K / 64) + g0 / 64] = bb;
        for (int i = 0; i < 64; i++) {
          float v = (bf16_to_f(w[g0 + i]) - lor) / scr;
          uint8_t u = (uint8_t)std::min(levels, std::max(0.f, std::round(v)));
          if (bits == 8) W[R * K + g0 + i] = u;
          else if (i % 2 == 0) W[(R * K + g0 + i) / 2] = u;  // low nibble first
          else W[(R * K + g0 + i) / 2] |= u << 4;
        }
      }
    });
    row += rows;
  }
  return q;
}
