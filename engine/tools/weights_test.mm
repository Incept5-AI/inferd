// Safetensors header validation in src/weights.h (Codex review 2026-11-02, S07). Each malformed file is indexed in a
// forked child (the loader exits on bad input) under AddressSanitizer/UBSan; the parent checks it was refused cleanly.
// Loader families (Codex review 2026-11-03, R01): checkpoints whose headers are valid and self-consistent but whose
// tensors do not fit what a loader expects (NGram multipliers, standard / mixed-width / bf16-quantized linears, the
// MoE stack, the PLE table, bf16 router rows, config geometry) must be refused before anything is copied. These need
// a Metal device and build/core.metallib (WEIGHTS_TEST_METALLIB), created in the child only.
//   clang++ -std=c++20 -g -O1 -fsanitize=address,undefined -fobjc-arc -framework Metal -framework Foundation \
//     -framework IOKit -framework IOSurface tools/weights_test.mm -o build/weights_test
//   build/weights_test                 # synthetic cases
//   build/weights_test <model dir>...  # index real checkpoints (each must load)
#include <sys/wait.h>
#include <cstdlib>
#include <string>
#include "../src/qwen35.h"

// A sanitizer finding must not look like a clean refusal (exit 1).
extern "C" const char* __asan_default_options() { return "exitcode=99"; }
extern "C" const char* __ubsan_default_options() { return "halt_on_error=1:exitcode=98"; }

static std::string tmp_dir() {
  char t[] = "/tmp/weights_test.XXXXXX";
  if (!mkdtemp(t)) { perror("mkdtemp"); exit(2); }
  return t;
}

static void write_case(const std::string& dir, const std::string& header, size_t data_bytes, int64_t hlen_override = -1,
                       bool truncate_header = false) {
  FILE* f = fopen((dir + "/config.json").c_str(), "w");
  fputs("{}", f);
  fclose(f);
  f = fopen((dir + "/model.safetensors").c_str(), "wb");
  uint64_t hlen = hlen_override >= 0 ? (uint64_t)hlen_override : header.size();
  fwrite(&hlen, 8, 1, f);
  fwrite(header.data(), 1, truncate_header ? header.size() / 2 : header.size(), f);
  std::string data(data_bytes, '\x11');
  fwrite(data.data(), 1, data.size(), f);
  fclose(f);
}

// 0: indexed, 1: refused with exit(1), other: crashed/sanitizer (ASan exits with 1 too, so stderr is checked by eye).
static int index_in_child(const std::string& dir) {
  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    freopen("/dev/null", "w", stderr);
    Checkpoint ck(dir);
    _exit(ck.t.empty() ? 3 : 0);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

// A synthetic checkpoint: tensors with valid headers and data (bytes 0x11), and a config.json.
struct T { std::string name, dtype; std::vector<int64_t> shape; };
static void write_ckpt(const std::string& dir, const std::string& config, const std::vector<T>& ts) {
  std::string h = "{", data;
  for (auto& t : ts) {
    size_t n = Checkpoint::dtype_size(t.dtype);
    std::string sh;
    for (auto d : t.shape) { sh += (sh.empty() ? "" : ",") + std::to_string(d); n *= (size_t)d; }
    h += (h.size() > 1 ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + sh +
         "],\"data_offsets\":[" + std::to_string(data.size()) + "," + std::to_string(data.size() + n) + "]}";
    data.append(n, '\x11');
  }
  h += "}";
  FILE* f = fopen((dir + "/config.json").c_str(), "w");
  fputs(config.c_str(), f);
  fclose(f);
  f = fopen((dir + "/model.safetensors").c_str(), "wb");
  uint64_t hl = h.size();
  fwrite(&hl, 8, 1, f);
  fwrite(h.data(), 1, h.size(), f);
  fwrite(data.data(), 1, data.size(), f);
  fclose(f);
}

// Run a loader on a checkpoint in a forked child: 0 loaded, 1 refused (exit(1)), else crashed / sanitizer.
static int load_in_child(const std::string& dir, const std::string& what) {
  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    if (!getenv("WEIGHTS_TEST_VERBOSE")) freopen("/dev/null", "w", stderr);
    Checkpoint ck(dir);
    if (what == "config") { Qwen35Config c(ck.config); _exit(0); }
    if (what == "ngram") { auto* ng = new NGram; ng->init(ck.config, ck, "mult", 0); delete ng; _exit(0); }
    if (what == "bf16_rows") { std::vector<uint16_t> rows; append_bf16_rows(ck, "l", rows, 128); _exit(0); }
    const char* lib = getenv("WEIGHTS_TEST_METALLIB");
    Gpu g(lib ? lib : "build/core.metallib");
    if (what == "qlinear") load_qlinear(g, ck, {"l"});
    else if (what == "qlinear_any") load_qlinear_any(g, ck, {"l"});
    else if (what == "qlinear_w") load_qlinear_w(g, ck, {"l"});
    else if (what == "bf16_q8") load_qlinear_bf16_q8(g, ck, {"l.weight"});
    else if (what == "moe") load_moe(g, ck, {"e"});
    else if (what == "ple") { PleTable t; t.load(g, ck, "p."); }
    else _exit(4);
    _exit(0);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

static int loader_cases() {
  const std::string q4 = R"({"quantization":{"bits":4,"group_size":64}})", q5 = R"({"quantization":{"bits":5,"group_size":64}})";
  const std::string ng = R"({"ngram_size":3,"heads_per_ngram":1,"eos_token_id":0,"ngram_vocab_size_base":3})";
  auto lin = [](const std::string& wd, std::vector<int64_t> w, const std::string& sd, std::vector<int64_t> sc,
                std::vector<int64_t> bi = {}) {
    return std::vector<T>{{"l.weight", wd, w}, {"l.scales", sd, sc}, {"l.biases", sd, bi.empty() ? sc : bi}};
  };
  auto moe = [](std::vector<int64_t> w, std::vector<int64_t> sc, const std::string& bd = "BF16") {
    return std::vector<T>{{"e.weight", "U32", w}, {"e.scales", "BF16", sc}, {"e.biases", bd, sc}};
  };
  auto ple = [](int bad_shard, std::vector<int64_t> bad_scales, int64_t rows3 = 2) {
    std::vector<T> v;
    for (int i = 0; i < 4; i++) {
      const std::string p = "p." + std::to_string(i);
      const int64_t r = i == 3 ? rows3 : 2;
      v.push_back({p + ".weight", "U32", {r, 8}});
      v.push_back({p + ".scales", "BF16", i == bad_shard ? bad_scales : std::vector<int64_t>{r, 2}});
      v.push_back({p + ".biases", "BF16", i == bad_shard ? bad_scales : std::vector<int64_t>{r, 2}});
    }
    return v;
  };
  const std::string pleq = R"({"quantization":{"bits":4,"group_size":64,"p.0":{"bits":4,"group_size":32},)"
                           R"("p.1":{"bits":4,"group_size":32},"p.2":{"bits":4,"group_size":32},"p.3":{"bits":4,"group_size":32}}})";
  // A small Qwen3.5-style config with one key overridden (or added): "key", "json value".
  auto cfg = [](const std::string& key = "", const std::string& val = "") {
    std::map<std::string, std::string> c = {
        {"hidden_size", "128"}, {"intermediate_size", "256"}, {"num_hidden_layers", "4"}, {"vocab_size", "1000"},
        {"full_attention_interval", "4"}, {"num_attention_heads", "4"}, {"num_key_value_heads", "2"}, {"head_dim", "64"},
        {"rms_norm_eps", "1e-6"}, {"rope_theta", "1000000"}, {"partial_rotary_factor", "0.25"},
        {"linear_num_key_heads", "2"}, {"linear_num_value_heads", "2"}, {"linear_key_head_dim", "64"},
        {"linear_value_head_dim", "64"}, {"linear_conv_kernel_dim", "4"}};
    if (!key.empty()) c[key] = val;
    std::string o = "{";
    for (auto& [k, v] : c) o += (o.size() > 1 ? "," : "") + std::string("\"") + k + "\":" + v;
    return o + "}";
  };
  struct LC { const char* name; std::string config; std::vector<T> ts; const char* what; int want; };
  std::vector<LC> cases = {
      {"ngram I64[3]", ng, {{"mult", "I64", {3}}}, "ngram", 0},
      {"ngram I64[32] (R01 repro)", ng, {{"mult", "I64", {32}}}, "ngram", 1},
      {"ngram F64[3]", ng, {{"mult", "F64", {3}}}, "ngram", 1},
      {"ngram I64[2] short", ng, {{"mult", "I64", {2}}}, "ngram", 1},
      {"ngram_size 5", R"({"ngram_size":5,"heads_per_ngram":1,"eos_token_id":0})", {{"mult", "I64", {5}}}, "ngram", 1},
      {"ngram heads 0", R"({"ngram_size":3,"heads_per_ngram":0,"eos_token_id":0})", {{"mult", "I64", {3}}}, "ngram", 1},
      {"qlinear 4-bit", "{}", lin("U32", {4, 16}, "BF16", {4, 2}), "qlinear", 0},
      {"qlinear 8-bit", "{}", lin("U32", {4, 32}, "BF16", {4, 2}), "qlinear", 0},
      {"qlinear F32 scales", "{}", lin("U32", {4, 16}, "F32", {4, 2}), "qlinear", 1},
      {"qlinear biases shape", "{}", lin("U32", {4, 16}, "BF16", {4, 2}, {4, 3}), "qlinear", 1},
      {"qlinear rank-3 weight", "{}", lin("U32", {1, 4, 16}, "BF16", {4, 2}), "qlinear", 1},
      {"qlinear 5-bit", "{}", lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear", 1},
      {"qlinear I32 weight", "{}", lin("I32", {4, 16}, "BF16", {4, 2}), "qlinear", 1},
      {"qlinear bf16", "{}", {{"l.weight", "BF16", {4, 128}}}, "qlinear", 0},
      {"qlinear bf16 rank 1", "{}", {{"l.weight", "BF16", {512}}}, "qlinear", 1},
      {"qlinear_any 5-bit", q5, lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_any", 0},
      {"qlinear_w 5-bit", q5, lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_w", 0},
      {"qlinear_any 4-bit g32", R"({"quantization":{"bits":4,"group_size":32}})", lin("U32", {4, 16}, "BF16", {4, 4}), "qlinear_any", 0},
      {"qlinear_any scales too wide", q5, lin("U32", {4, 20}, "BF16", {4, 8}), "qlinear_any", 1},
      {"qlinear_any scales rows", q5, lin("U32", {4, 20}, "BF16", {8, 2}), "qlinear_any", 1},
      {"qlinear_any bits 0", R"({"quantization":{"bits":0,"group_size":64}})", lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_any", 1},
      {"qlinear_any group 0", R"({"quantization":{"bits":5,"group_size":0}})", lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_any", 1},
      {"qlinear_any group 48", R"({"quantization":{"bits":5,"group_size":48}})", lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_any", 1},
      {"qlinear_any bits string", R"({"quantization":{"bits":"5","group_size":64}})", lin("U32", {4, 20}, "BF16", {4, 2}), "qlinear_any", 1},
      {"qlinear_any 0 rows", q5, lin("U32", {0, 20}, "BF16", {0, 2}), "qlinear_any", 1},
      {"qlinear_any 3-bit partial row", R"({"quantization":{"bits":3,"group_size":64}})", lin("U32", {4, 1}, "BF16", {4, 1}), "qlinear_any", 1},
      {"qlinear_any F32 biases", q5, std::vector<T>{{"l.weight", "U32", {4, 20}}, {"l.scales", "BF16", {4, 2}}, {"l.biases", "F32", {4, 2}}}, "qlinear_any", 1},
      {"bf16 rows", "{}", {{"l.weight", "BF16", {2, 128}}}, "bf16_rows", 0},
      {"bf16 rows K mismatch", "{}", {{"l.weight", "BF16", {2, 64}}}, "bf16_rows", 1},
      {"bf16 rows quantized", q4, lin("U32", {2, 16}, "BF16", {2, 2}), "bf16_rows", 0},
      {"bf16 rows scales too wide", q4, lin("U32", {2, 16}, "BF16", {2, 4}), "bf16_rows", 1},
      {"bf16_q8", "{}", {{"l.weight", "BF16", {4, 128}}}, "bf16_q8", 0},
      {"bf16_q8 K 100", "{}", {{"l.weight", "BF16", {4, 100}}}, "bf16_q8", 1},
      {"bf16_q8 rank 3", "{}", {{"l.weight", "BF16", {1, 4, 128}}}, "bf16_q8", 1},
      {"moe", "{}", moe({2, 4, 16}, {2, 4, 2}), "moe", 0},
      {"moe 2-D", "{}", moe({4, 16}, {4, 2}), "moe", 0},
      {"moe rank-1 weight", "{}", moe({64}, {4, 2}), "moe", 1},
      {"moe scales K (2-bit)", "{}", moe({2, 4, 16}, {2, 4, 4}), "moe", 1},
      {"moe scales N", "{}", moe({2, 4, 16}, {2, 5, 2}), "moe", 1},
      {"moe scales E", "{}", moe({2, 4, 16}, {3, 4, 2}), "moe", 1},
      {"moe scales rank", "{}", moe({2, 4, 16}, {8, 2}), "moe", 1},
      {"moe F32 biases", "{}", moe({2, 4, 16}, {2, 4, 2}, "F32"), "moe", 1},
      {"ple", pleq, ple(-1, {}), "ple", 0},
      {"ple shard scales too wide", pleq, ple(2, {2, 4}), "ple", 1},
      {"ple uneven shards", pleq, ple(-1, {}, 3), "ple", 1},
      {"config", cfg(), {{"x", "U8", {1}}}, "config", 0},
      {"config kv heads 0", cfg("num_key_value_heads", "0"), {{"x", "U8", {1}}}, "config", 1},
      {"config heads 3/2", cfg("num_attention_heads", "3"), {{"x", "U8", {1}}}, "config", 1},
      {"config layers 0", cfg("num_hidden_layers", "0"), {{"x", "U8", {1}}}, "config", 1},
      {"config linear heads 0", cfg("linear_num_key_heads", "0"), {{"x", "U8", {1}}}, "config", 1},
      {"config head_dim string", cfg("head_dim", "\"64\""), {{"x", "U8", {1}}}, "config", 1},
      {"config vocab 2^31", cfg("vocab_size", "2147483648"), {{"x", "U8", {1}}}, "config", 1},
      {"config eps 0", cfg("rms_norm_eps", "0"), {{"x", "U8", {1}}}, "config", 1},
      {"config layer_types short", cfg("layer_types", "[\"linear_attention\"]"), {{"x", "U8", {1}}}, "config", 1},
  };
  int fails = 0;
  for (auto& c : cases) {
    std::string d = tmp_dir();
    const std::string& conf = c.config;
    write_ckpt(d, conf, c.ts);
    int got = load_in_child(d, c.what);
    bool pass = got == c.want;
    fails += !pass;
    printf("%s %-30s exit %d (want %d)\n", pass ? "ok  " : "FAIL", c.name, got, c.want);
    std::string rm = "rm -rf " + d;
    system(rm.c_str());
  }
  printf("weights_test loaders: %zu/%zu passed\n", cases.size() - fails, cases.size());
  return fails;
}

int main(int argc, char** argv) {
  if (argc > 1) {
    int bad = 0;
    for (int i = 1; i < argc; i++) {
      Checkpoint ck(argv[i]);
      size_t bytes = 0;
      for (auto& [k, v] : ck.t) bytes += v.nbytes;
      printf("ok %s: %zu tensors, %.2f GiB\n", argv[i], ck.t.size(), bytes / 1073741824.0);
    }
    return bad;
  }
  struct Case { const char* name; std::string header; size_t data; int64_t hlen; bool trunc; int want; };
  const std::string ok = R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}})";
  std::vector<Case> cases = {
      {"valid", ok, 12, -1, false, 0},
      {"valid with metadata", R"({"__metadata__":{"format":"mlx"},"w":{"dtype":"U32","shape":[4,2],"data_offsets":[0,32]}})", 32, -1, false, 0},
      {"header length past EOF", ok, 12, 1 << 20, false, 1},
      {"header length 2^64-1", ok, 12, -2 & 0x7fffffffffffffffll, false, 1},
      {"header length 0", ok, 12, 0, false, 1},
      {"truncated header", ok, 0, -1, true, 1},
      {"header not JSON", "{\"w\":", 12, -1, false, 1},
      {"header not an object", "[1,2]", 12, -1, false, 1},
      {"offsets past data", R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}})", 8, -1, false, 1},
      {"offsets reversed", R"({"w":{"dtype":"BF16","shape":[0],"data_offsets":[8,4]}})", 12, -1, false, 1},
      {"offset 2^64-1", R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[18446744073709551615,12]}})", 12, -1, false, 1},
      {"negative offset", R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[-4,8]}})", 12, -1, false, 1},
      {"shape/range mismatch", R"({"w":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,12]}})", 12, -1, false, 1},
      {"dtype size mismatch", R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,12]}})", 12, -1, false, 1},
      {"shape product overflow", R"({"w":{"dtype":"BF16","shape":[4294967296,4294967296,4],"data_offsets":[0,0]}})", 12, -1, false, 1},
      {"negative dim", R"({"w":{"dtype":"BF16","shape":[-2,-3],"data_offsets":[0,12]}})", 12, -1, false, 1},
      {"fractional dim", R"({"w":{"dtype":"BF16","shape":[1.5,4],"data_offsets":[0,12]}})", 12, -1, false, 1},
      {"unknown dtype", R"({"w":{"dtype":"Q7","shape":[2,3],"data_offsets":[0,12]}})", 12, -1, false, 1},
      {"missing offsets", R"({"w":{"dtype":"BF16","shape":[2,3]}})", 12, -1, false, 1},
      {"three offsets", R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,6,12]}})", 12, -1, false, 1},
      {"tensor not an object", R"({"w":[1]})", 12, -1, false, 1},
  };
  int fails = 0;
  for (auto& c : cases) {
    std::string d = tmp_dir();
    write_case(d, c.header, c.data, c.hlen, c.trunc);
    int got = index_in_child(d);
    bool pass = got == c.want;
    fails += !pass;
    printf("%s %-24s exit %d (want %d)\n", pass ? "ok  " : "FAIL", c.name, got, c.want);
    std::string rm = "rm -rf " + d;
    system(rm.c_str());
  }
  printf("weights_test: %zu/%zu passed\n", cases.size() - fails, cases.size());
  fails += loader_cases();
  return fails != 0;
}
