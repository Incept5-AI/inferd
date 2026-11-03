// Grammar-constrained decoding for structured output (response_format json_schema / json_object).
//
// A JSON Schema is compiled to a tree of value nodes (Grammar), and a byte-level pushdown matcher (Matcher) walks
// the generated text: a token is allowed when every byte of it is. The sampler never builds a vocabulary mask: it
// takes candidates in descending logit order and keeps the first one the matcher accepts (greedy: exactly the
// argmax of the masked logits; sampling: top-k/top-p over the allowed candidates). Usually the model's own top token
// is allowed, so a step costs one matcher copy plus the argmax that the plain sampler does anyway.
//
// Enforced: JSON syntax (no NaN, no raw control characters), type (incl. lists, nullable), enum/const (as literal
// texts), properties/required/additionalProperties (no unknown keys when false, no duplicate keys, '}' only once
// every required key is present), items/prefixItems/minItems/maxItems, minLength/maxLength, pattern (an ASCII regex
// subset: classes, groups, |, * + ? {m,n}, ^ $ at the ends), format date/date-time/uuid, integer vs number,
// minimum >= 0 (no '-'), small integer ranges (enumerated), local $ref (recursion included).
// Loosened (the front end validates afterwards and retries): anyOf/oneOf/allOf (branches merged into their union),
// other numeric bounds, multipleOf, uniqueItems, patternProperties, unsupported patterns, string escapes in keys.
// Whitespace between tokens is capped (kWsMax in a row) so a model cannot loop on newlines.
#pragma once
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "json.h"

namespace gram {

// Budgets. A schema over any of them is refused (gram::Error: the request fails with that message), never weakened.
// The front end (frontend/structured.py, check_schema) applies the same limits and the same regex subset first.
constexpr size_t kMaxSchemaBytes = 64 << 10;  // the schema as compact JSON text
constexpr size_t kMaxPatternBytes = 512;      // one pattern
constexpr int kMaxPatternDepth = 32;          // nested groups in a pattern
constexpr int kMaxRepeat = 64;                // {m,n} bounds
constexpr size_t kMaxNfaStates = 2000;        // one pattern's NFA
constexpr int kMaxSchemaDepth = 48;           // nested subschemas ($ref cycles are back-edges, not depth)
constexpr size_t kMaxNodes = 4096;            // compiled value nodes
constexpr size_t kMaxLits = 20000;            // enum / const / integer-range literals, all nodes together
constexpr size_t kMaxWork = 1000000;          // compile steps (subschemas, merges, literals)
constexpr size_t kMaxFrames = 64;             // nesting of the generated JSON value
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The model's vocabulary as bytes (sent once by the front end: inferd op "vocab"). text[t] = 0 for special/added
// tokens, which never appear inside JSON.
struct Vocab {
  std::vector<std::string> bytes;
  std::vector<uint8_t> text;
};
inline std::shared_ptr<const Vocab>& vocab() {
  static std::shared_ptr<const Vocab> v;
  return v;
}

// ---------------------------------------------------------------- regex subset -> Thompson NFA
// Characters: 0..127 ASCII, 128 = any non-ASCII character (a pattern's own non-ASCII characters are unsupported).
using CSet = std::bitset<129>;

struct Nfa {
  struct St {
    uint8_t kind;  // 0: character in set -> a; 1: split a | b; 2: accept
    CSet set;
    int a = -1, b = -1;
  };
  std::vector<St> st;
  int start = -1;
  using Set = std::vector<uint64_t>;
  int add(uint8_t kind, CSet set = {}, int a = -1, int b = -1) {
    st.push_back({kind, set, a, b});
    return (int)st.size() - 1;
  }
  void closure(Set& s, int i) const {
    int stack[512], n = 0;
    std::vector<int> big;
    auto push = [&](int k) { if (n < 512) stack[n++] = k; else big.push_back(k); };
    push(i);
    while (n || !big.empty()) {
      int k;
      if (!big.empty()) { k = big.back(); big.pop_back(); } else k = stack[--n];
      if (k < 0 || (s[k >> 6] >> (k & 63) & 1)) continue;
      s[k >> 6] |= 1ull << (k & 63);
      if (st[k].kind == 1) { push(st[k].a); push(st[k].b); }
    }
  }
  Set init() const {
    Set s((st.size() + 63) / 64, 0);
    closure(s, start);
    return s;
  }
  bool step(Set& s, int c) const {  // false: no state survives c
    Set n(s.size(), 0);
    bool any = false;
    for (size_t k = 0; k < st.size(); k++)
      if ((s[k >> 6] >> (k & 63) & 1) && st[k].kind == 0 && st[k].set[c]) { closure(n, st[k].a); any = true; }
    s.swap(n);
    return any;
  }
  bool accepting(const Set& s) const {
    for (size_t k = 0; k < st.size(); k++)
      if ((s[k >> 6] >> (k & 63) & 1) && st[k].kind == 2) return true;
    return false;
  }
};

struct Re {
  int kind = 0;  // 0 set, 1 concatenation, 2 alternation, 3 repeat [lo, hi] (hi -1: unbounded), 4 empty
  CSet set;
  std::vector<Re> k;
  int lo = 0, hi = 0;
};

struct ReParser {
  const std::string& p;
  size_t i = 0, end = 0;
  bool bad = false;
  int depth = 0;  // nested groups (kMaxPatternDepth): the parser recurses per group
  static CSet cls_digit() { CSet s; for (int c = '0'; c <= '9'; c++) s.set(c); return s; }
  static CSet cls_word() {
    CSet s = cls_digit();
    for (int c = 'a'; c <= 'z'; c++) s.set(c);
    for (int c = 'A'; c <= 'Z'; c++) s.set(c);
    s.set('_');
    return s;
  }
  static CSet cls_space() { CSet s; for (char c : std::string(" \t\n\r\f\v")) s.set((unsigned char)c); return s; }
  CSet esc() {  // after a backslash
    if (i >= end) { bad = true; return {}; }
    char c = p[i++];
    CSet s;
    switch (c) {
      case 'd': return cls_digit();
      case 'D': return ~cls_digit();
      case 'w': return cls_word();
      case 'W': return ~cls_word();
      case 's': return cls_space();
      case 'S': return ~cls_space();
      case 'n': s.set('\n'); return s;
      case 't': s.set('\t'); return s;
      case 'r': s.set('\r'); return s;
      case 'f': s.set('\f'); return s;
      case 'v': s.set('\v'); return s;
      default:
        if ((unsigned char)c < 128 && !std::isalnum((unsigned char)c)) { s.set(c); return s; }
        bad = true;  // \b, \p{..}, \u...., back-references: unsupported
        return s;
    }
  }
  CSet cls() {  // after '['
    CSet s;
    bool neg = i < end && p[i] == '^';
    if (neg) i++;
    bool first = true;
    while (i < end && (p[i] != ']' || first)) {
      first = false;
      unsigned char c = p[i++];
      if (c >= 128) { bad = true; return s; }
      if (c == '\\') { CSet e = esc(); if (bad) return s; if (e.count() != 1 || i >= end || p[i] != '-') { s |= e; continue; }
        for (int k = 0; k < 128; k++) if (e[k]) c = (unsigned char)k; }
      if (i + 1 < end && p[i] == '-' && p[i + 1] != ']') {
        unsigned char d = p[i + 1];
        i += 2;
        if (d == '\\') { CSet e = esc(); if (bad || e.count() != 1) { bad = true; return s; } for (int k = 0; k < 128; k++) if (e[k]) d = (unsigned char)k; }
        if (d >= 128 || d < c) { bad = true; return s; }
        for (int k = c; k <= d; k++) s.set(k);
      } else {
        s.set(c);
      }
    }
    if (i >= end) { bad = true; return s; }
    i++;  // ']'
    return neg ? ~s : s;
  }
  Re atom() {
    unsigned char c = p[i++];
    Re r;
    if (c == '(') {
      if (++depth > kMaxPatternDepth) { bad = true; return r; }
      if (p.compare(i, 2, "?:") == 0) i += 2;
      else if (i < end && p[i] == '?') { bad = true; return r; }
      Re x = alt();
      if (bad || i >= end || p[i] != ')') { bad = true; return r; }
      i++;
      depth--;
      return x;
    }
    if (c == '[') { r.set = cls(); return r; }
    if (c == '.') { r.set.set(); r.set.reset('\n'); return r; }
    if (c == '\\') { r.set = esc(); return r; }
    if (c >= 128 || c == '^' || c == '$' || c == ')' || c == '*' || c == '+' || c == '?' || c == '|') { bad = true; return r; }
    r.set.set(c);
    return r;
  }
  bool num(int& v) {
    size_t j = i;
    v = 0;
    while (i < end && std::isdigit((unsigned char)p[i]) && v < 100000) v = v * 10 + (p[i++] - '0');
    return i > j;
  }
  void quant(Re& a) {
    while (i < end && !bad) {
      char c = p[i];
      int lo, hi;
      if (c == '*') { lo = 0; hi = -1; i++; }
      else if (c == '+') { lo = 1; hi = -1; i++; }
      else if (c == '?') { lo = 0; hi = 1; i++; }
      else if (c == '{') {
        size_t save = i++;
        if (!num(lo)) { i = save; return; }
        hi = lo;
        if (i < end && p[i] == ',') { i++; if (!num(hi)) hi = -1; }
        if (i >= end || p[i] != '}') { i = save; return; }
        i++;
      } else return;
      if (i < end && (p[i] == '?' || p[i] == '+')) i++;  // lazy / possessive: same language
      if (lo > kMaxRepeat || hi > kMaxRepeat || (hi >= 0 && hi < lo)) { bad = true; return; }
      Re r;
      r.kind = 3;
      r.k.push_back(std::move(a));
      r.lo = lo;
      r.hi = hi;
      a = std::move(r);
    }
  }
  Re cat() {
    Re r;
    r.kind = 1;
    while (i < end && p[i] != '|' && p[i] != ')' && !bad) {
      Re a = atom();
      if (bad) return r;
      quant(a);
      r.k.push_back(std::move(a));
    }
    return r;
  }
  Re alt() {
    Re r;
    r.kind = 2;
    r.k.push_back(cat());
    while (i < end && p[i] == '|' && !bad) { i++; r.k.push_back(cat()); }
    if (r.k.size() == 1) return std::move(r.k[0]);
    return r;
  }
};

inline int nfa_comp(Nfa& n, const Re& r, int out, bool& bad) {
  if (bad || n.st.size() > kMaxNfaStates) { bad = true; return out; }
  switch (r.kind) {
    case 0: return n.add(0, r.set, out);
    case 1:
      for (int j = (int)r.k.size() - 1; j >= 0; j--) out = nfa_comp(n, r.k[j], out, bad);
      return out;
    case 2: {
      int s = nfa_comp(n, r.k.back(), out, bad);
      for (int j = (int)r.k.size() - 2; j >= 0; j--) {
        int a = nfa_comp(n, r.k[j], out, bad);
        s = n.add(1, {}, a, s);
      }
      return s;
    }
    case 3: {
      int t = out;
      if (r.hi < 0) {
        int s = n.add(1, {}, -1, out);
        int a = nfa_comp(n, r.k[0], s, bad);
        n.st[s].a = a;
        t = s;
      } else {
        for (int j = 0; j < r.hi - r.lo; j++) {
          int a = nfa_comp(n, r.k[0], t, bad);
          t = n.add(1, {}, a, out);
        }
      }
      for (int j = 0; j < r.lo; j++) t = nfa_comp(n, r.k[0], t, bad);
      return t;
    }
    default: return out;
  }
}

// -> false for a pattern outside the subset (the caller leaves the string unconstrained by it).
inline bool compile_regex(const std::string& pat, Nfa& n) {
  if (pat.size() > kMaxPatternBytes) return false;
  ReParser ps{pat};
  ps.end = pat.size();
  bool a0 = !pat.empty() && pat[0] == '^';
  if (a0) ps.i = 1;
  bool a1 = false;
  if (ps.end > ps.i && pat[ps.end - 1] == '$') {
    size_t bs = 0;
    for (size_t j = ps.end - 1; j > ps.i && pat[j - 1] == '\\'; j--) bs++;
    if (bs % 2 == 0) { a1 = true; ps.end--; }
  }
  Re re = ps.alt();
  if (ps.bad || ps.i != ps.end) return false;
  CSet any;
  any.set();
  bool bad = false;
  int out = n.add(2);
  if (!a1) {
    int e = n.add(1, {}, -1, out);
    n.st[e].a = n.add(0, any, e);
    out = e;
  }
  int s = nfa_comp(n, re, out, bad);
  if (!a0) {
    int l = n.add(1, {}, -1, s);
    n.st[l].a = n.add(0, any, l);
    s = l;
  }
  n.start = s;
  return !bad && n.st.size() <= kMaxNfaStates;
}

// ---------------------------------------------------------------- schema -> value nodes

enum : uint8_t { T_NULL = 1, T_BOOL = 2, T_INT = 4, T_NUM = 8, T_STR = 16, T_ARR = 32, T_OBJ = 64, T_ALL = 127 };

struct Node {
  uint8_t types = T_ALL;  // T_NUM: any number; T_INT: integers only
  bool has_lits = false;  // only these JSON texts (enum / const / small integer ranges)
  std::vector<std::string> lits;
  int min_len = 0, max_len = -1, nfa = -1;
  bool nonneg = false;
  int items = 0;  // node for items past prefix; -1: none allowed
  std::vector<int> prefix;
  int min_items = 0, max_items = -1;
  std::vector<std::string> keys;
  std::vector<int> kprop;
  std::vector<uint8_t> kreq;
  int additional = 0;  // node for keys not in `keys`; -1: none allowed
};

inline std::string dump(const json::Value& v) {
  switch (v.type) {
    case json::Value::Null: return "null";
    case json::Value::Bool: return v.b ? "true" : "false";
    case json::Value::Num: {
      if (v.exact) return std::to_string(v.ll);
      char b[40];
      if (std::isfinite(v.n) && v.n == std::floor(v.n) && std::fabs(v.n) < 1e15) snprintf(b, sizeof b, "%.1f", v.n);
      else snprintf(b, sizeof b, "%.17g", v.n);
      return b;
    }
    case json::Value::Str: {
      std::string r = "\"";
      for (unsigned char c : v.s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\t') r += "\\t";
        else if (c == '\r') r += "\\r";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); r += b; }
        else r += (char)c;
      }
      return r + "\"";
    }
    case json::Value::Arr: {
      std::string r = "[";
      for (size_t i = 0; i < v.a.size(); i++) r += (i ? "," : "") + dump(v.a[i]);
      return r + "]";
    }
    case json::Value::Obj: {
      std::string r = "{";
      bool first = true;
      for (auto& [k, x] : v.o) {
        json::Value ks;
        ks.type = json::Value::Str;
        ks.s = k;
        r += (first ? "" : ",") + dump(ks) + ":" + dump(x);
        first = false;
      }
      return r + "}";
    }
  }
  return "null";
}

inline uint8_t lit_type(const std::string& l) {
  switch (l.empty() ? 0 : l[0]) {
    case '"': return T_STR;
    case '{': return T_OBJ;
    case '[': return T_ARR;
    case 't': case 'f': return T_BOOL;
    case 'n': return T_NULL;
    default: return l.find_first_of(".eE") == std::string::npos ? (T_INT | T_NUM) : T_NUM;
  }
}

struct Grammar {
  std::vector<Node> nodes;
  std::vector<Nfa> nfas;
  int root = 0;
  std::unordered_map<std::string, int> refs;
  const json::Value* doc = nullptr;
  std::vector<std::string> loose;  // constructs not enforced (logged; the front end validates)

  size_t work = 0, n_lits = 0;

  // Throws gram::Error for a schema outside the supported subset or over a budget.
  explicit Grammar(const json::Value& schema) {
    if (schema.type != json::Value::Obj && schema.type != json::Value::Bool) throw Error("schema must be an object");
    if (dump(schema).size() > kMaxSchemaBytes) throw Error("schema is larger than 64 KiB");
    nodes.emplace_back();  // 0: any JSON value
    doc = &schema;
    root = compile(schema, 0);
    doc = nullptr;
  }

  void tick(size_t k = 1) {
    if ((work += k) > kMaxWork) throw Error("schema is too complex to compile");
  }
  int push(Node n) {
    if (nodes.size() >= kMaxNodes) throw Error("schema has too many subschemas");
    if ((n_lits += n.lits.size()) > kMaxLits) throw Error("schema has too many enum/const values");
    tick(1 + n.lits.size());
    nodes.push_back(std::move(n));
    return (int)nodes.size() - 1;
  }

  const json::Value* resolve(const std::string& ref) const {
    if (ref.empty() || ref[0] != '#') return nullptr;
    const json::Value* v = doc;
    size_t i = 1;
    while (i < ref.size()) {
      if (ref[i] != '/') return nullptr;
      size_t j = ref.find('/', i + 1);
      std::string part = ref.substr(i + 1, (j == std::string::npos ? ref.size() : j) - i - 1);
      for (size_t k; (k = part.find("~1")) != std::string::npos;) part.replace(k, 2, "/");
      for (size_t k; (k = part.find("~0")) != std::string::npos;) part.replace(k, 2, "~");
      if (v->type == json::Value::Obj && v->has(part)) v = &(*v)[part];
      else if (v->type == json::Value::Arr && !part.empty() && std::isdigit((unsigned char)part[0]) &&
               (size_t)atoi(part.c_str()) < v->a.size()) v = &v->a[atoi(part.c_str())];
      else return nullptr;
      i = j == std::string::npos ? ref.size() : j;
    }
    return v;
  }

  int compile(const json::Value& s, int depth) {
    tick();
    if (depth > kMaxSchemaDepth) throw Error("schema is nested too deeply");
    if (s.type == json::Value::Bool) {
      if (!s.b) throw Error("false subschemas are not supported");
      return 0;
    }
    if (s.type != json::Value::Obj) throw Error("a subschema must be an object");
    if (s.has("$ref")) {
      if (s["$ref"].type != json::Value::Str) throw Error("$ref must be a string");
      const std::string& ref = s["$ref"].s;
      auto it = refs.find(ref);
      if (it != refs.end()) return it->second;
      const json::Value* t = resolve(ref);
      if (!t) throw Error("$ref " + ref + " is not a local reference that resolves");
      int idx = push(Node{});
      refs[ref] = idx;
      Node n = build(*t, depth + 1);
      nodes[idx] = std::move(n);
      return idx;
    }
    Node n = build(s, depth);
    return push(std::move(n));
  }

  Node build(const json::Value& s, int depth) {
    if (s.type != json::Value::Obj) return Node{};
    if (s.has("$ref")) return nodes[compile(s, depth)];
    Node b = base(s, depth);
    const bool plain = !(s.has("type") || s.has("enum") || s.has("const") || s.has("properties") || s.has("items") ||
                         s.has("prefixItems"));
    for (const char* k : {"anyOf", "oneOf", "allOf"}) {
      if (!s.has(k) || s[k].type != json::Value::Arr || s[k].a.empty()) continue;
      int m = compile(s[k].a[0], depth + 1);
      for (size_t i = 1; i < s[k].a.size(); i++) m = merge(m, compile(s[k].a[i], depth + 1), 0);
      if (s[k].a.size() > 1) loose.push_back(std::string(k) + " (union)");
      if (plain) {
        Node r = nodes[m];
        if (s.has("nullable") && s["nullable"].b) {
          if (r.has_lits) r.lits.push_back("null");
          else r.types |= T_NULL;
        }
        return r;
      }
      loose.push_back(std::string(k) + " beside other keywords");
    }
    return b;
  }

  static double num(const json::Value& v, double d) { return v.type == json::Value::Num ? v.n : d; }

  Node base(const json::Value& s, int depth) {
    Node n;
    const bool nullable = s.has("nullable") && s["nullable"].b;
    if (s.has("enum") && s["enum"].type == json::Value::Arr) {
      n.has_lits = true;
      for (auto& v : s["enum"].a) n.lits.push_back(dump(v));
      if (nullable) n.lits.push_back("null");
      return n;
    }
    if (s.has("const")) {
      n.has_lits = true;
      n.lits.push_back(dump(s["const"]));
      return n;
    }
    if (s.has("type")) {
      n.types = 0;
      std::vector<const json::Value*> ts;
      if (s["type"].type == json::Value::Arr) for (auto& t : s["type"].a) ts.push_back(&t);
      else ts.push_back(&s["type"]);
      for (auto* t : ts) {
        const std::string& x = t->s;
        const uint8_t b = x == "null" ? T_NULL : x == "boolean" ? T_BOOL : x == "integer" ? T_INT : x == "number" ? T_NUM
                        : x == "string" ? T_STR : x == "array" ? T_ARR : x == "object" ? T_OBJ : 0;
        if (!b || t->type != json::Value::Str) throw Error("unknown type " + dump(*t));
        n.types |= b;
      }
      if (!n.types) throw Error("type must name at least one type");
    } else if (s.has("properties") || s.has("required") || s.has("additionalProperties")) {
      n.types = T_OBJ;
    } else if (s.has("items") || s.has("prefixItems")) {
      n.types = T_ARR;
    }
    if (nullable) n.types |= T_NULL;
    // strings
    n.min_len = (int)num(s["minLength"], 0);
    n.max_len = (int)num(s["maxLength"], -1);
    std::string pat = s["pattern"].type == json::Value::Str ? s["pattern"].s : "";
    if (pat.empty() && s["format"].type == json::Value::Str) {
      const std::string& f = s["format"].s;
      if (f == "date") pat = "^\\d{4}-\\d{2}-\\d{2}$";
      else if (f == "date-time") pat = "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$";
      else if (f == "time") pat = "^\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?$";
      else if (f == "uuid") pat = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$";
    }
    if (!pat.empty() && (n.types & T_STR)) {
      Nfa a;
      if (!compile_regex(pat, a)) throw Error("pattern outside the supported subset or over its limits: " + pat);
      tick(a.st.size());
      nfas.push_back(std::move(a));
      n.nfa = (int)nfas.size() - 1;
    }
    // numbers
    double lo = -INFINITY, hi = INFINITY;
    const json::Value &mn = s["minimum"], &mx = s["maximum"], &xmn = s["exclusiveMinimum"], &xmx = s["exclusiveMaximum"];
    if (mn.type == json::Value::Num) lo = mn.n;
    if (mx.type == json::Value::Num) hi = mx.n;
    if (xmn.type == json::Value::Num) lo = std::max(lo, std::nextafter(xmn.n, INFINITY));
    if (xmx.type == json::Value::Num) hi = std::min(hi, std::nextafter(xmx.n, -INFINITY));
    if (xmn.type == json::Value::Bool && xmn.b && mn.type == json::Value::Num) lo = std::nextafter(mn.n, INFINITY);
    if (xmx.type == json::Value::Bool && xmx.b && mx.type == json::Value::Num) hi = std::nextafter(mx.n, -INFINITY);
    n.nonneg = lo >= 0;
    if ((n.types & ~T_NULL) == T_INT && std::isfinite(lo) && std::isfinite(hi) && std::ceil(lo) <= std::floor(hi) &&
        std::floor(hi) - std::ceil(lo) <= 1000 && std::fabs(lo) < 1e9 && std::fabs(hi) < 1e9) {
      for (long long v = (long long)std::ceil(lo); v <= (long long)std::floor(hi); v++) n.lits.push_back(std::to_string(v));
      if (n.types & T_NULL) n.lits.push_back("null");
      n.has_lits = true;
      return n;
    }
    if (std::isfinite(lo) || std::isfinite(hi)) {
      if (lo < 0 || std::isfinite(hi)) loose.push_back("numeric bounds");
    }
    if (s.has("multipleOf")) loose.push_back("multipleOf");
    // arrays
    if (s.has("prefixItems") && s["prefixItems"].type == json::Value::Arr)
      for (auto& x : s["prefixItems"].a) n.prefix.push_back(compile(x, depth + 1));
    const json::Value& it = s["items"];
    if (it.type == json::Value::Arr) {  // draft 4-7 tuple form
      for (auto& x : it.a) n.prefix.push_back(compile(x, depth + 1));
      const json::Value& ai = s["additionalItems"];
      n.items = ai.type == json::Value::Bool && !ai.b ? -1 : ai.type == json::Value::Obj ? compile(ai, depth + 1) : 0;
    } else if (it.type == json::Value::Bool) {
      n.items = it.b ? 0 : -1;
    } else if (it.type == json::Value::Obj) {
      n.items = compile(it, depth + 1);
    }
    n.min_items = (int)num(s["minItems"], 0);
    n.max_items = (int)num(s["maxItems"], -1);
    if (n.items < 0) n.max_items = n.max_items < 0 ? (int)n.prefix.size() : std::min(n.max_items, (int)n.prefix.size());
    if (s.has("uniqueItems") && s["uniqueItems"].b) loose.push_back("uniqueItems");
    // objects
    if (s["properties"].type == json::Value::Obj)
      for (auto& [k, v] : s["properties"].o) {
        n.keys.push_back(k);
        n.kprop.push_back(compile(v, depth + 1));
        n.kreq.push_back(0);
      }
    const json::Value& ap = s["additionalProperties"];
    n.additional = ap.type == json::Value::Bool && !ap.b ? -1 : ap.type == json::Value::Obj ? compile(ap, depth + 1) : 0;
    if (s.has("patternProperties")) { n.additional = std::max(n.additional, 0); loose.push_back("patternProperties"); }
    if (s["required"].type == json::Value::Arr)
      for (auto& r : s["required"].a) {
        auto k = std::find(n.keys.begin(), n.keys.end(), r.s);
        if (k != n.keys.end()) n.kreq[k - n.keys.begin()] = 1;
        else if (n.additional >= 0) {  // required but not described: a property of the additional schema
          n.keys.push_back(r.s);
          n.kprop.push_back(n.additional);
          n.kreq.push_back(1);
        }
      }
    return n;
  }

  // The union of two nodes (anyOf/oneOf/allOf): every value either allows is allowed (looser than the schema).
  int merge(int a, int b, int depth) {
    tick();
    if (a == b) return a;
    if (a == 0 || b == 0 || depth > 6) return 0;
    const Node x = nodes[a], y = nodes[b];
    Node r;
    if (x.has_lits && y.has_lits) {
      r.has_lits = true;
      r.lits = x.lits;
      for (auto& l : y.lits) if (std::find(r.lits.begin(), r.lits.end(), l) == r.lits.end()) r.lits.push_back(l);
      return push(std::move(r));
    }
    auto lt = [](const Node& n) { uint8_t t = 0; if (!n.has_lits) return n.types; for (auto& l : n.lits) t |= lit_type(l); return t; };
    auto side = [](const Node& n, uint8_t bit) { return !n.has_lits && (n.types & bit); };
    auto litside = [&](const Node& n, uint8_t bit) { return n.has_lits && (lt(n) & bit); };
    r.types = lt(x) | lt(y);
    // strings
    if (!litside(x, T_STR) && !litside(y, T_STR)) {
      const bool p = side(x, T_STR), q = side(y, T_STR);
      if (p && q) {
        r.min_len = std::min(x.min_len, y.min_len);
        r.max_len = x.max_len < 0 || y.max_len < 0 ? -1 : std::max(x.max_len, y.max_len);
        r.nfa = x.nfa == y.nfa ? x.nfa : -1;
      } else if (p || q) {
        const Node& o = p ? x : y;
        r.min_len = o.min_len, r.max_len = o.max_len, r.nfa = o.nfa;
      }
    }
    // numbers
    const uint8_t NB = T_INT | T_NUM;
    r.nonneg = !litside(x, NB) && !litside(y, NB) && (!side(x, NB) || x.nonneg) && (!side(y, NB) || y.nonneg);
    // arrays
    if (!litside(x, T_ARR) && !litside(y, T_ARR)) {
      const bool p = side(x, T_ARR), q = side(y, T_ARR);
      if (p && q) {
        if (x.prefix.size() == y.prefix.size())
          for (size_t i = 0; i < x.prefix.size(); i++) r.prefix.push_back(merge(x.prefix[i], y.prefix[i], depth + 1));
        r.items = x.items < 0 ? y.items : y.items < 0 ? x.items : merge(x.items, y.items, depth + 1);
        if (x.prefix.size() != y.prefix.size()) r.items = 0;
        r.min_items = std::min(x.min_items, y.min_items);
        r.max_items = x.max_items < 0 || y.max_items < 0 ? -1 : std::max(x.max_items, y.max_items);
        if (r.items < 0 && r.max_items < 0) r.max_items = (int)r.prefix.size();
      } else if (p || q) {
        const Node& o = p ? x : y;
        r.items = o.items, r.prefix = o.prefix, r.min_items = o.min_items, r.max_items = o.max_items;
      }
    }
    // objects
    if (!litside(x, T_OBJ) && !litside(y, T_OBJ)) {
      const bool p = side(x, T_OBJ), q = side(y, T_OBJ);
      if (p && q) {
        for (size_t i = 0; i < x.keys.size(); i++) {
          auto k = std::find(y.keys.begin(), y.keys.end(), x.keys[i]);
          r.keys.push_back(x.keys[i]);
          if (k == y.keys.end()) { r.kprop.push_back(x.kprop[i]); r.kreq.push_back(0); continue; }
          size_t j = k - y.keys.begin();
          r.kprop.push_back(merge(x.kprop[i], y.kprop[j], depth + 1));
          r.kreq.push_back(x.kreq[i] && y.kreq[j]);
        }
        for (size_t j = 0; j < y.keys.size(); j++)
          if (std::find(x.keys.begin(), x.keys.end(), y.keys[j]) == x.keys.end()) {
            r.keys.push_back(y.keys[j]);
            r.kprop.push_back(y.kprop[j]);
            r.kreq.push_back(0);
          }
        r.additional = x.additional < 0 ? y.additional : y.additional < 0 ? x.additional
                     : merge(x.additional, y.additional, depth + 1);
      } else if (p || q) {
        const Node& o = p ? x : y;
        r.keys = o.keys, r.kprop = o.kprop, r.kreq = o.kreq, r.additional = o.additional;
      }
    }
    return push(std::move(r));
  }
};

// ---------------------------------------------------------------- the matcher

constexpr int kWsMax = 40;     // whitespace characters in a row outside strings
constexpr int kNumMax = 32;    // characters in one number
constexpr int kKeyMax = 256;   // bytes in one object key

struct Matcher {
  enum Mode : uint8_t { VAL, STR, NUM, LIT, ARR, OBJ };
  enum R : uint8_t { ACCEPT, REJECT, POP, POPRE, PUSHRE };
  struct Frame {
    int node;
    Mode mode = VAL;
    uint8_t sub = 0;   // NUM: phase; ARR/OBJ: position (see step)
    uint8_t esc = 0;   // STR / OBJ key: 0, 1 after '\', 2..5 reading \u hex digits
    uint8_t lits = 0;  // LIT: 0 the node's literals, 1 true/false, 2 null
    int count = 0;     // STR: characters; NUM: characters; ARR: items
    int kprop = -1;    // OBJ: the current key's property index, -1 an additional property
    unsigned ucp = 0;
    std::string buf;   // LIT: text so far; OBJ: key bytes
    Nfa::Set nfa;      // STR with a pattern
    std::vector<uint8_t> seen;  // OBJ
    std::vector<std::string> extra;  // OBJ: additional keys so far (no duplicates)
  };

  std::shared_ptr<const Grammar> g;
  std::vector<Frame> st;
  bool active = true;  // false until think_end (reasoning is unconstrained)
  int think_end = -1;
  int ws = 0;

  Matcher(std::shared_ptr<const Grammar> gr, int think_end_id) : g(std::move(gr)), think_end(think_end_id) {
    active = think_end_id < 0;
    st.push_back(Frame{g->root});
  }

  static bool is_ws(uint8_t c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
  R wsok() { if (ws >= kWsMax) return REJECT; ws++; return ACCEPT; }

  const std::vector<std::string>& lit_list(const Frame& f) const {
    static const std::vector<std::string> tf{"true", "false"}, nl{"null"};
    return f.lits == 1 ? tf : f.lits == 2 ? nl : g->nodes[f.node].lits;
  }

  R lit(Frame& f, uint8_t c) {
    const auto& L = lit_list(f);
    f.buf.push_back((char)c);
    for (auto& l : L)
      if (l.size() >= f.buf.size() && l.compare(0, f.buf.size(), f.buf) == 0) return ACCEPT;
    f.buf.pop_back();
    for (auto& l : L) if (l == f.buf) return POPRE;
    return REJECT;
  }

  R str_char(Frame& f, const Node& n, int code) {
    if (n.max_len >= 0 && f.count >= n.max_len) return REJECT;
    if (n.nfa >= 0 && !g->nfas[n.nfa].step(f.nfa, code)) return REJECT;
    f.count++;
    return ACCEPT;
  }

  static int hexv(uint8_t c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
  }

  static bool key_possible(const Frame& f, const Node& n) {
    if (n.additional >= 0) return true;
    for (size_t i = 0; i < n.keys.size(); i++) if (!f.seen[i]) return true;
    return false;
  }
  static bool req_ok(const Frame& f, const Node& n) {
    for (size_t i = 0; i < n.keys.size(); i++) if (n.kreq[i] && !f.seen[i]) return false;
    return true;
  }
  static int max_items(const Node& n) { return n.max_items; }
  static int child_of(const Frame& f, const Node& n) {
    return f.count < (int)n.prefix.size() ? n.prefix[f.count] : n.items;
  }

  R step(uint8_t c) {
    Frame& f = st.back();
    const Node& n = g->nodes[f.node];
    switch (f.mode) {
      case VAL: {
        if (is_ws(c)) return wsok();
        if (n.has_lits) { f.mode = LIT; f.lits = 0; return lit(f, c); }
        if (c == '"' && (n.types & T_STR)) {
          f.mode = STR;
          if (n.nfa >= 0) f.nfa = g->nfas[n.nfa].init();
          return ACCEPT;
        }
        if ((c == '-' || (c >= '0' && c <= '9')) && (n.types & (T_INT | T_NUM))) {
          if (c == '-' && n.nonneg) return REJECT;
          f.mode = NUM;
          f.count = 1;
          f.sub = c == '-' ? 0 : c == '0' ? 1 : 2;
          return ACCEPT;
        }
        if (c == '{' && (n.types & T_OBJ)) { f.mode = OBJ; f.seen.assign(n.keys.size(), 0); return ACCEPT; }
        if (c == '[' && (n.types & T_ARR)) { f.mode = ARR; return ACCEPT; }
        if ((c == 't' || c == 'f') && (n.types & T_BOOL)) { f.mode = LIT; f.lits = 1; return lit(f, c); }
        if (c == 'n' && (n.types & T_NULL)) { f.mode = LIT; f.lits = 2; return lit(f, c); }
        return REJECT;
      }
      case LIT: return lit(f, c);
      case STR: {
        if (f.esc == 1) {
          static const char E[] = "\"\\/bfnrt";
          static const int D[] = {'"', '\\', '/', 8, 12, 10, 13, 9};
          if (c == 'u') { f.esc = 2; f.ucp = 0; return ACCEPT; }
          const char* p = c ? strchr(E, c) : nullptr;
          if (!p) return REJECT;
          f.esc = 0;
          return str_char(f, n, D[p - E]);
        }
        if (f.esc >= 2) {
          int h = hexv(c);
          if (h < 0) return REJECT;
          f.ucp = f.ucp * 16 + h;
          if (f.esc++ < 5) return ACCEPT;
          f.esc = 0;
          if (f.ucp >= 0xD800 && f.ucp < 0xDC00) return ACCEPT;  // a high surrogate: its pair is one character
          return str_char(f, n, f.ucp < 128 ? (int)f.ucp : 128);
        }
        if (c == '"') {
          if (f.count < n.min_len) return REJECT;
          if (n.nfa >= 0 && !g->nfas[n.nfa].accepting(f.nfa)) return REJECT;
          return POP;
        }
        if (c == '\\') {
          if (n.max_len >= 0 && f.count >= n.max_len) return REJECT;
          f.esc = 1;
          return ACCEPT;
        }
        if (c < 0x20 || c >= 0xF8) return REJECT;
        if (c >= 0x80 && c < 0xC0) return ACCEPT;  // UTF-8 continuation byte: same character
        return str_char(f, n, c < 0x80 ? c : 128);
      }
      case NUM: {
        const bool dig = c >= '0' && c <= '9';
        const bool frac = n.types & T_NUM;
        uint8_t nx = 255;
        switch (f.sub) {
          case 0: nx = c == '0' ? 1 : dig ? 2 : 255; break;
          case 1: nx = c == '.' && frac ? 3 : (c == 'e' || c == 'E') && frac ? 5 : 254; break;
          case 2: nx = dig ? 2 : c == '.' && frac ? 3 : (c == 'e' || c == 'E') && frac ? 5 : 254; break;
          case 3: nx = dig ? 4 : 255; break;
          case 4: nx = dig ? 4 : (c == 'e' || c == 'E') ? 5 : 254; break;
          case 5: nx = c == '+' || c == '-' ? 6 : dig ? 7 : 255; break;
          case 6: nx = dig ? 7 : 255; break;
          case 7: nx = dig ? 7 : 254; break;
        }
        if (nx == 254) return POPRE;  // a complete number ended by a character of the parent's
        if (nx == 255 || ++f.count > kNumMax) return REJECT;
        f.sub = nx;
        return ACCEPT;
      }
      case ARR: {
        const int mx = max_items(n);
        if (f.sub != 2 && is_ws(c)) return wsok();
        if (f.sub == 1) {
          if (c == ',' && (mx < 0 || f.count < mx)) { f.sub = 2; return ACCEPT; }
          if (c == ']' && f.count >= n.min_items) return POP;
          return REJECT;
        }
        if (f.sub == 0 && c == ']') return f.count >= n.min_items ? POP : REJECT;
        if (f.sub == 2 && is_ws(c)) return wsok();
        if (mx >= 0 && f.count >= mx) return REJECT;
        const int ch = child_of(f, n);
        if (ch < 0 || st.size() >= kMaxFrames) return REJECT;
        st.push_back(Frame{ch});
        return PUSHRE;
      }
      case OBJ: {
        if (f.sub == 5) {  // inside a key
          if (c == '"') {
            for (size_t i = 0; i < n.keys.size(); i++)
              if (!f.seen[i] && n.keys[i] == f.buf) { f.kprop = (int)i; f.sub = 1; return ACCEPT; }
            if (n.additional < 0) return REJECT;
            for (size_t i = 0; i < n.keys.size(); i++) if (n.keys[i] == f.buf) return REJECT;  // a duplicate key
            if (std::find(f.extra.begin(), f.extra.end(), f.buf) != f.extra.end()) return REJECT;
            f.extra.push_back(f.buf);
            f.kprop = -1;
            f.sub = 1;
            return ACCEPT;
          }
          if (c == '\\' || c < 0x20 || c >= 0xF8 || f.buf.size() >= (size_t)kKeyMax) return REJECT;
          f.buf.push_back((char)c);
          if (n.additional >= 0) return ACCEPT;
          for (size_t i = 0; i < n.keys.size(); i++)
            if (!f.seen[i] && n.keys[i].size() >= f.buf.size() && n.keys[i].compare(0, f.buf.size(), f.buf) == 0)
              return ACCEPT;
          f.buf.pop_back();
          return REJECT;
        }
        if (is_ws(c)) return wsok();
        switch (f.sub) {
          case 0:
            if (c == '"' && key_possible(f, n)) { f.sub = 5; f.buf.clear(); return ACCEPT; }
            if (c == '}' && req_ok(f, n)) return POP;
            return REJECT;
          case 1:
            if (c == ':') { f.sub = 2; return ACCEPT; }
            return REJECT;
          case 2: {
            const int ch = f.kprop >= 0 ? n.kprop[f.kprop] : n.additional;
            if (ch < 0 || st.size() >= kMaxFrames) return REJECT;
            st.push_back(Frame{ch});
            return PUSHRE;
          }
          case 3:
            if (c == ',' && key_possible(f, n)) { f.sub = 4; return ACCEPT; }
            if (c == '}' && req_ok(f, n)) return POP;
            return REJECT;
          case 4:
            if (c == '"') { f.sub = 5; f.buf.clear(); return ACCEPT; }
            return REJECT;
        }
        return REJECT;
      }
    }
    return REJECT;
  }

  void value_done() {
    if (st.empty()) return;
    Frame& p = st.back();
    if (p.mode == ARR) { p.count++; p.sub = 1; }
    else if (p.mode == OBJ) { if (p.kprop >= 0) p.seen[p.kprop] = 1; p.sub = 3; }
  }

  bool feed(uint8_t c) {
    if (!is_ws(c)) ws = 0;
    for (int guard = 0; guard < 256; guard++) {
      if (st.empty()) return is_ws(c) && wsok() == ACCEPT;
      switch (step(c)) {
        case ACCEPT: return true;
        case REJECT: return false;
        case POP: st.pop_back(); value_done(); return true;
        case POPRE: st.pop_back(); value_done(); continue;
        case PUSHRE: continue;
      }
    }
    return false;
  }

  bool done() const { return st.empty(); }

  // Could generation end here (a stop token)? Pops a trailing number or literal that is complete as it stands.
  bool can_end() const {
    if (!active) return true;
    Matcher m = *this;
    while (!m.st.empty()) {
      Frame& f = m.st.back();
      bool complete = false;
      if (f.mode == NUM) complete = f.sub == 1 || f.sub == 2 || f.sub == 4 || f.sub == 7;
      else if (f.mode == LIT) { const auto& L = m.lit_list(f); complete = std::find(L.begin(), L.end(), f.buf) != L.end(); }
      if (!complete) return false;
      m.st.pop_back();
      m.value_done();
    }
    return true;
  }

  // Would token t be allowed here? With `out`, the matcher after it.
  bool allows(int t, const std::vector<int>& stops, Matcher* out) const {
    if (!active) {
      if (out) { *out = *this; if (t == think_end) out->active = true; }
      return true;
    }
    if (std::find(stops.begin(), stops.end(), t) != stops.end()) {
      if (!can_end()) return false;
      if (out) *out = *this;
      return true;
    }
    const Vocab* vb = vocab().get();
    if (!vb || t < 0 || t >= (int)vb->bytes.size() || !vb->text[t] || vb->bytes[t].empty()) return false;
    Matcher m = *this;
    for (unsigned char c : vb->bytes[t]) if (!m.feed(c)) return false;
    if (out) *out = std::move(m);
    return true;
  }

  // Advance over tokens committed elsewhere (a speculative block): -> how many were taken (n when all were): it
  // stops at the first token the grammar rejects, and with `stop_at_activation` right after the token that
  // activates the grammar (</think>), so that the caller samples the next row under the grammar itself.
  int advance(const int* t, int n, const std::vector<int>& stops, bool stop_at_activation = false) {
    for (int k = 0; k < n; k++) {
      Matcher next = *this;
      if (!allows(t[k], stops, &next)) return k;
      const bool activated = !active && next.active;
      *this = std::move(next);
      if (stop_at_activation && activated) return k + 1;
    }
    return n;
  }

  // Sample one token from bf16 logits under the grammar and advance past it. The contract: greedy = the argmax over
  // the allowed tokens; sampling = the allowed tokens only, the top_k of them by logit (1,024 when top_k is 0, as the
  // unconstrained host sampler), softmax at `temp`, then the top_p nucleus, then a draw. Sampled requests are not
  // speculated while the grammar is active (Scheduler: the GPU's speculative sampler keeps 16 candidates, a different
  // distribution), so this is the only sampler they see.
  int sample(const uint16_t* lg, int V, float temp, float top_p, int top_k, std::mt19937_64& rng,
             const std::vector<int>& stops) {
    auto f = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float x; memcpy(&x, &u, 4); return x; };
    Matcher next = *this;
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < V; i++) { float v = f(lg[i]); if (v > bv) { bv = v; best = i; } }
    const bool greedy = temp <= 0;
    if (greedy && allows(best, stops, &next)) { *this = std::move(next); return best; }
    // Candidates in descending logit order, widening until enough are allowed.
    const int want = greedy ? 1 : (top_k > 0 ? std::min(top_k, V) : std::min(V, 1024));
    std::vector<std::pair<float, int>> c(V);
    for (int i = 0; i < V; i++) c[i] = {f(lg[i]), i};
    auto cmp = [](auto& a, auto& b) { return a.first > b.first; };
    std::vector<std::pair<float, int>> ok;
    size_t sorted = 0;
    for (size_t want_sorted : {(size_t)std::min(V, 4096), (size_t)V}) {
      std::partial_sort(c.begin() + sorted, c.begin() + want_sorted, c.end(), cmp);
      for (size_t i = sorted; i < want_sorted && (int)ok.size() < want; i++)
        if (allows(c[i].second, stops, nullptr)) ok.push_back(c[i]);
      sorted = want_sorted;
      if ((int)ok.size() >= want) break;
    }
    if (ok.empty()) return stops.empty() ? best : stops[0];  // a dead end (should not happen): end the reply
    int t = ok[0].second;
    if (!greedy) {
      double mx = ok[0].first, sum = 0;
      std::vector<double> pr(ok.size());
      for (size_t i = 0; i < ok.size(); i++) sum += pr[i] = std::exp((ok[i].first - mx) / temp);
      double cut = top_p * sum, acc = 0;
      size_t n = 0;
      while (n < ok.size()) { acc += pr[n++]; if (acc >= cut) break; }
      std::uniform_real_distribution<double> u(0, acc);
      double r = u(rng);
      t = ok[n - 1].second;
      for (size_t i = 0; i < n; i++) if ((r -= pr[i]) <= 0) { t = ok[i].second; break; }
    }
    allows(t, stops, &next);
    *this = std::move(next);
    return t;
  }
};

}  // namespace gram
