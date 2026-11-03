// Minimal JSON reader: enough for safetensors headers, HF config files and inferd's request lines. Strict RFC 8259
// syntax (Codex review 2026-11-02, S08): no '+' or leading zeros in numbers, no raw control characters in strings,
// UTF-8 input only, and integer accessors that throw instead of truncating or saturating.
#pragma once
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

struct Value {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> a;
  std::map<std::string, Value> o;

  const Value& operator[](const std::string& k) const {
    static const Value null;
    auto it = o.find(k);
    return it == o.end() ? null : it->second;
  }
  const Value& operator[](size_t i) const { return a.at(i); }
  bool has(const std::string& k) const { return o.count(k) > 0; }
  long long ll = 0;     // integers written without a fraction or exponent, exactly (ids and seeds above 2^53)
  bool exact = false;   // ll holds the value (false for fractions, exponents and integers outside int64)
  // Range-checked integers: null/false read as 0 (absent config keys), true as 1; a number must be integral and in
  // range, or this throws (2^64 - 1 used to saturate to INT64_MAX, and 1.5 to truncate to 1).
  int64_t i64() const {
    if (type == Bool) return b;
    if (type != Num) {
      if (type == Null) return 0;
      throw std::runtime_error("json: expected an integer");
    }
    if (exact) return ll;
    if (!(n == std::floor(n)) || n < -9223372036854775808.0 || n >= 9223372036854775808.0)
      throw std::runtime_error("json: integer out of range or not integral");
    return (int64_t)n;
  }
  int i() const {
    int64_t v = i64();
    if (v < INT32_MIN || v > INT32_MAX) throw std::runtime_error("json: integer out of 32-bit range");
    return (int)v;
  }
  unsigned long long ull = 0;  // integers in (INT64_MAX, UINT64_MAX], exactly
  bool uexact = false;
  uint64_t u64() const {  // non-negative integers up to 2^64 - 1 (byte offsets)
    if (type == Num && uexact) return ull;
    int64_t v = i64();
    if (v < 0) throw std::runtime_error("json: expected a non-negative integer");
    return (uint64_t)v;
  }
};

// RFC 8259 texts are UTF-8: reject malformed, overlong and surrogate encodings.
inline bool valid_utf8(const char* s, size_t n) {
  const unsigned char* p = (const unsigned char*)s;
  for (size_t i = 0; i < n;) {
    unsigned c = p[i];
    if (c < 0x80) { i++; continue; }
    int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 ? 2 : 0;
    if (!len || c > 0xF4 || i + len > n) return false;
    unsigned cp = c & (0x3F >> (len - 1));
    for (int k = 1; k < len; k++) {
      if ((p[i + k] & 0xC0) != 0x80) return false;
      cp = cp << 6 | (p[i + k] & 0x3F);
    }
    if ((len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp < 0xE000))) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)))
      return false;
    i += len;
  }
  return true;
}

struct Parser {
  const char* p;
  const char* e;
  int depth = 0;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what); }
  // Bounds-checked: every read is guarded by p < e (Codex review F02: an incomplete \u escape read past the input).
  char peek() { if (p >= e) fail("unexpected end"); return *p; }
  char next() { char c = peek(); p++; return c; }
  static int hex(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
  }
  unsigned u4() {
    if (e - p < 4) fail("short \\u escape");
    unsigned cp = 0;
    for (int i = 0; i < 4; i++) {
      int h = hex(p[i]);
      if (h < 0) fail("bad \\u escape");
      cp = cp * 16 + h;
    }
    p += 4;
    return cp;
  }
  static void utf8(std::string& r, unsigned cp) {
    if (cp < 0x80) r += (char)cp;
    else if (cp < 0x800) { r += (char)(0xC0 | (cp >> 6)); r += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { r += (char)(0xE0 | (cp >> 12)); r += (char)(0x80 | ((cp >> 6) & 0x3F)); r += (char)(0x80 | (cp & 0x3F)); }
    else { r += (char)(0xF0 | (cp >> 18)); r += (char)(0x80 | ((cp >> 12) & 0x3F)); r += (char)(0x80 | ((cp >> 6) & 0x3F)); r += (char)(0x80 | (cp & 0x3F)); }
  }
  std::string str() {
    if (next() != '"') fail("expected string");
    std::string r;
    for (;;) {
      char c = next();
      if (c == '"') return r;
      if ((unsigned char)c < 0x20) fail("raw control character in string");
      if (c != '\\') { r += c; continue; }
      switch (char x = next()) {
        case 'n': r += '\n'; break;
        case 't': r += '\t'; break;
        case 'r': r += '\r'; break;
        case 'b': r += '\b'; break;
        case 'f': r += '\f'; break;
        case '"': case '\\': case '/': r += x; break;
        case 'u': {
          unsigned cp = u4();
          if (cp >= 0xD800 && cp < 0xDC00) {  // a surrogate pair
            if (e - p < 6 || p[0] != '\\' || p[1] != 'u') fail("lone high surrogate");
            p += 2;
            unsigned lo = u4();
            if (lo < 0xDC00 || lo >= 0xE000) fail("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp < 0xE000) {
            fail("lone low surrogate");
          }
          utf8(r, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  bool word(const char* w, size_t n) {
    if ((size_t)(e - p) < n || strncmp(p, w, n)) return false;
    p += n;
    return true;
  }
  Value val() {
    ws();
    Value v;
    if (++depth > 256) fail("nesting too deep");
    char c = peek();
    if (c == '{') {
      v.type = Value::Obj;
      p++;
      ws();
      if (peek() == '}') { p++; depth--; return v; }
      for (;;) {
        ws();
        std::string k = str();
        ws();
        if (next() != ':') fail("expected :");
        v.o[k] = val();
        ws();
        char d = next();
        if (d == ',') continue;
        if (d != '}') fail("expected }");
        depth--;
        return v;
      }
    }
    if (c == '[') {
      v.type = Value::Arr;
      p++;
      ws();
      if (peek() == ']') { p++; depth--; return v; }
      for (;;) {
        v.a.push_back(val());
        ws();
        char d = next();
        if (d == ',') continue;
        if (d != ']') fail("expected ]");
        depth--;
        return v;
      }
    }
    depth--;
    if (c == '"') { v.type = Value::Str; v.s = str(); return v; }
    if (word("true", 4)) { v.type = Value::Bool; v.b = true; return v; }
    if (word("false", 5)) { v.type = Value::Bool; return v; }
    if (word("null", 4)) return v;
    // numbers, RFC 8259: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?  (no '+', no leading zeros, no "1.")
    const char* q = p;
    auto digits = [&] { const char* s0 = q; while (q < e && *q >= '0' && *q <= '9') q++; return q > s0; };
    if (q < e && *q == '-') q++;
    if (q >= e || *q < '0' || *q > '9') fail("bad value");
    if (*q == '0') {
      q++;
      if (q < e && *q >= '0' && *q <= '9') fail("leading zero in number");
    } else {
      digits();
    }
    bool integral = true;
    if (q < e && *q == '.') { q++; integral = false; if (!digits()) fail("bad number fraction"); }
    if (q < e && (*q == 'e' || *q == 'E')) {
      q++;
      integral = false;
      if (q < e && (*q == '+' || *q == '-')) q++;
      if (!digits()) fail("bad number exponent");
    }
    if (q - p > 512) fail("number too long");
    // strtod/strtoll need a terminated buffer; copy the (short) token
    const std::string tok(p, q);
    v.type = Value::Num;
    v.n = std::strtod(tok.c_str(), nullptr);
    if (!std::isfinite(v.n)) fail("number out of range");
    if (integral) {
      errno = 0;
      long long x = std::strtoll(tok.c_str(), nullptr, 10);
      if (errno != ERANGE) {
        v.ll = x, v.exact = true;
      } else if (tok[0] != '-') {  // outside int64: kept as a double, and exactly for u64() up to 2^64 - 1
        errno = 0;
        unsigned long long u = std::strtoull(tok.c_str(), nullptr, 10);
        if (errno != ERANGE) v.ull = u, v.uexact = true;
      }
    }
    p = q;
    return v;
  }
};

inline Value parse(const std::string& s) {
  if (!valid_utf8(s.data(), s.size())) throw std::runtime_error("json: input is not valid UTF-8");
  Parser ps{s.data(), s.data() + s.size()};
  Value v = ps.val();
  ps.ws();
  if (ps.p != ps.e) ps.fail("trailing data");
  return v;
}

}  // namespace json
