// Host tests for src/json.h (strict RFC 8259 syntax, range-checked integers; Codex review 2026-11-02, S08).
//   clang++ -std=c++20 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc tools/json_test.cc \
//     -o build/json_test && build/json_test
#include <cstdio>
#include <string>
#include "../src/json.h"

static int fails = 0, total = 0;

static void accept(const std::string& s) {
  total++;
  try { json::parse(s); }
  catch (std::exception& e) { fails++; printf("FAIL: rejected %s: %s\n", s.substr(0, 40).c_str(), e.what()); }
}
static void reject(const std::string& s) {
  total++;
  try { json::parse(s); fails++; printf("FAIL: accepted %s\n", s.substr(0, 40).c_str()); }
  catch (std::exception&) {}
}
template <class F> static void throws(const char* what, F f) {
  total++;
  try { f(); fails++; printf("FAIL: no throw: %s\n", what); } catch (std::exception&) {}
}
template <class T> static void eq(const char* what, T got, T want) {
  total++;
  if (!(got == want)) { fails++; printf("FAIL: %s\n", what); }
}

int main() {
  // The review's cases.
  reject("\"" + std::string(124, 'a') + "\\u");
  reject("true trailing");
  reject("+1");
  reject("01");
  reject("\"raw\nnewline\"");
  throws("u64 max as i64", [] { json::parse("18446744073709551615").i64(); });
  eq("u64 max as u64", json::parse("18446744073709551615").u64(), (uint64_t)18446744073709551615ull);
  // Numbers.
  for (auto s : {"0", "-0", "1", "-1", "10", "1.5", "-0.25", "1e5", "1E+5", "1e-5", "0.0", "123456789012345678",
                 "9223372036854775807", "-9223372036854775808", "1.0e10"})
    accept(s);
  for (auto s : {"-", "+0", "00", "-01", "1.", ".5", "1e", "1e+", "--1", "0x10", "1.5.2", "1e5e5", "Infinity", "NaN",
                 "1e999", "-", "- 1", "[1,]", "[,1]", "{\"a\":1,}", "{\"a\" 1}", "{a:1}", "[1 2]", "", " ",
                 "tru", "nul", "'x'", "[\"a\"", "{\"a\":"})
    reject(s);
  // Strings: control characters must be escaped; escapes are validated; input must be UTF-8.
  accept("\"tab\\tnl\\n\\u0041\\ud83d\\ude80\"");
  eq("surrogate pair", json::parse("\"\\ud83d\\ude80\"").s, std::string("\xF0\x9F\x9A\x80"));
  eq("raw utf8", json::parse("\"\xC3\xA9\"").s, std::string("\xC3\xA9"));
  reject("\"a\tb\"");
  reject(std::string("\"a\0b\"", 5));
  reject("\"\x1f\"");
  reject("\"\\x\"");
  reject("\"\\ud83d\"");
  reject("\"\\ude80\"");
  reject("\"\xC3\"");               // truncated sequence
  reject("\"\xC0\xAF\"");           // overlong
  reject("\"\xED\xA0\x80\"");       // encoded surrogate
  reject("\"\xF4\x90\x80\x80\"");   // above U+10FFFF
  reject("\xEF\xBB\xBF{}");         // BOM is not JSON whitespace
  // Integer accessors.
  eq("i small", json::parse("42").i(), 42);
  eq("i neg", json::parse("-7").i(), -7);
  eq("i integral double", json::parse("4096.0").i(), 4096);
  eq("i64 exact 2^62+1", json::parse("4611686018427387905").i64(), (int64_t)4611686018427387905ll);
  eq("i64 min", json::parse("-9223372036854775808").i64(), (int64_t)INT64_MIN);
  eq("null as 0", json::parse("{}")["missing"].i(), 0);
  eq("true as 1", json::parse("true").i(), 1);
  throws("i fraction", [] { json::parse("1.5").i(); });
  throws("i above int32", [] { json::parse("2147483648").i(); });
  throws("i below int32", [] { json::parse("-2147483649").i(); });
  throws("i64 above", [] { json::parse("9223372036854775808").i64(); });
  throws("i64 1e19", [] { json::parse("1e19").i64(); });
  throws("i of string", [] { json::parse("\"5\"").i(); });
  throws("u64 negative", [] { json::parse("-1").u64(); });
  throws("u64 2^64", [] { json::parse("18446744073709551616").u64(); });
  // Depth guard and a realistic header.
  reject(std::string(300, '[') + std::string(300, ']'));
  accept(std::string(200, '[') + std::string(200, ']'));
  auto h = json::parse("{\"__metadata__\":{\"format\":\"mlx\"},\"w\":{\"dtype\":\"BF16\",\"shape\":[2,3],"
                       "\"data_offsets\":[0,12]}}");
  eq("header offsets", h["w"]["data_offsets"][1].u64(), (uint64_t)12);
  printf("json_test: %d/%d passed\n", total - fails, total);
  return fails != 0;
}
