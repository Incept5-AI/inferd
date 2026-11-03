// Unit test for src/grammar.h (no GPU, no model): schema -> matcher, fed character by character.
//   clang++ -std=c++20 -O2 tools/grammar_test.cc -o build/grammar_test && build/grammar_test
#include <chrono>
#include <cstdio>
#include "../src/grammar.h"

static int fails = 0, runs = 0;

// Feed text; -> 1 when every byte is accepted and the value is complete (could end here), 0 otherwise.
static int full(const char* schema, const std::string& text) {
  json::Value s = json::parse(schema);
  auto g = std::make_shared<const gram::Grammar>(s);
  gram::Matcher m(g, -1);
  for (unsigned char c : text) if (!m.feed(c)) return 0;
  return m.can_end() ? 1 : 0;
}

static void expect(const char* schema, const std::string& text, int want) {
  runs++;
  int got = full(schema, text);
  if (got != want) { fails++; printf("FAIL %s | %s -> %d (want %d)\n", schema, text.c_str(), got, want); }
}

int main() {
  const char* ANY = "{}";
  expect(ANY, "{\"a\": [1, 2.5, -3e+2, true, null, \"x\\n\\u00e9\"]}", 1);
  expect(ANY, "[]", 1);
  expect(ANY, "  42", 1);
  expect(ANY, "{\"a\": 1", 0);           // incomplete
  expect(ANY, "{\"a\": NaN}", 0);
  expect(ANY, "{\"a\": 01}", 0);
  expect(ANY, "\"raw\nnewline\"", 0);    // control character in a string
  expect(ANY, "{\"a\":1}{", 0);          // nothing after the value
  expect(ANY, "{\"a\":1}\n", 1);
  expect(ANY, "{\"a\":1,\"a\":2}", 0);   // duplicate key
  expect(ANY, std::string("[") + std::string(41, ' ') + "1]", 0);  // whitespace run cap

  const char* OBJ = "{\"type\":\"object\"}";
  expect(OBJ, "{}", 1);
  expect(OBJ, "[1]", 0);

  const char* P = R"({"type":"object","properties":{"name":{"type":"string","minLength":1},
      "age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":["string","null"]},
      "tags":{"type":"array","items":{"type":"string"},"maxItems":2}},
      "required":["name","age"],"additionalProperties":false})";
  expect(P, R"({"name":"Ann","age":31})", 1);
  expect(P, R"({ "age": 31, "name": "Ann", "email": null, "tags": ["a","b"] })", 1);  // any order
  expect(P, R"({"name":"Ann"})", 0);              // '}' before a required key
  expect(P, R"({"name":"","age":3})", 0);          // minLength
  expect(P, R"({"name":"A","age":3.5})", 0);       // integer
  expect(P, R"({"name":"A","age":151})", 0);       // enumerated range
  expect(P, R"({"name":"A","age":-1})", 0);
  expect(P, R"({"name":"A","age":3,"nick":"z"})", 0);  // additionalProperties false
  expect(P, R"({"name":"A","age":3,"tags":["a","b","c"]})", 0);  // maxItems
  expect(P, R"({"name":"A","age":3,"email":7})", 0);

  const char* E = R"({"enum":["positive","negative",3,null]})";
  expect(E, "\"positive\"", 1);
  expect(E, "\"pos\"", 0);
  expect(E, "3", 1);
  expect(E, "null", 1);
  expect(E, "4", 0);

  const char* PAT = R"({"type":"string","pattern":"^INV-\\d+$"})";
  expect(PAT, "\"INV-2291\"", 1);
  expect(PAT, "\"INV-\"", 0);
  expect(PAT, "\"inv-1\"", 0);
  const char* PAT2 = R"({"type":"string","pattern":"[a-z]{2}(-[0-9]{1,3})?"})";  // unanchored: search
  expect(PAT2, "\"XXab-12YY\"", 1);
  expect(PAT2, "\"XYZ\"", 0);
  expect(R"({"type":"string","format":"date"})", "\"1969-07-16\"", 1);
  expect(R"({"type":"string","format":"date"})", "\"16/07/1969\"", 0);
  expect(R"({"type":"string","maxLength":3})", "\"\\u00e9\\n\xc3\xa9\"", 1);  // 3 characters
  expect(R"({"type":"string","maxLength":2})", "\"\\u00e9\\n\xc3\xa9\"", 0);

  const char* REF = R"({"$defs":{"Node":{"type":"object","properties":{"name":{"type":"string"},
      "kids":{"type":"array","items":{"$ref":"#/$defs/Node"}}},"required":["name","kids"],
      "additionalProperties":false}},"$ref":"#/$defs/Node"})";
  expect(REF, R"({"name":"a","kids":[{"name":"b","kids":[]},{"name":"c","kids":[{"name":"d","kids":[]}]}]})", 1);
  expect(REF, R"({"name":"a","kids":[{"name":"b"}]})", 0);

  const char* ONE = R"({"oneOf":[{"type":"object","properties":{"kind":{"const":"circle"},"r":{"type":"number"}},
      "required":["kind","r"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"rect"},
      "w":{"type":"number"},"h":{"type":"number"}},"required":["kind","w","h"],"additionalProperties":false}]})";
  expect(ONE, R"({"kind":"circle","r":2})", 1);
  expect(ONE, R"({"kind":"rect","w":1,"h":2})", 1);
  expect(ONE, R"({"kind":"square"})", 0);
  expect(ONE, R"({"kind":"circle","z":1})", 0);

  const char* ANYOF = R"({"type":"array","items":{"anyOf":[{"type":"string"},{"type":"integer"}]},"minItems":2})";
  expect(ANYOF, R"(["jdoe", 1042])", 1);
  expect(ANYOF, R"(["jdoe"])", 0);
  expect(ANYOF, R"(["jdoe", 1.5])", 0);
  expect(ANYOF, R"(["jdoe", true])", 0);

  const char* TUP = R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],"items":false})";
  expect(TUP, R"([1,"a"])", 1);
  expect(TUP, R"([1,"a",2])", 0);
  expect(TUP, R"(["a"])", 0);

  const char* NUMS = R"({"type":"number","minimum":0})";
  expect(NUMS, "0.25", 1);
  expect(NUMS, "1e3", 1);
  expect(NUMS, "-1", 0);
  expect(NUMS, "1.", 0);

  const char* CONSTS = R"({"type":"object","properties":{"servings":{"const":2}},"required":["servings"]})";
  expect(CONSTS, R"({"servings": 2})", 1);
  expect(CONSTS, R"({"servings": 2, "extra": [1]})", 1);  // additionalProperties not false
  expect(CONSTS, R"({"servings": 3})", 0);

  // ---- refused schemas (gram::Error), never weakened, never a crash (Codex sweep 2026-11-02, S01)
  auto refused = [&](const std::string& schema, const char* why) {
    runs++;
    auto t0 = std::chrono::steady_clock::now();
    try {
      json::Value v = json::parse(schema);
      gram::Grammar g(v);
      fails++;
      printf("FAIL not refused: %s\n", why);
    } catch (gram::Error& e) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (ms > 500) { fails++; printf("FAIL slow refusal (%.0f ms): %s\n", ms, why); }
    } catch (std::exception& e) {
      fails++;
      printf("FAIL wrong exception for %s: %s\n", why, e.what());
    }
  };
  {  // the review's repro: 50,000 nested groups, straight into the regex compiler
    runs++;
    gram::Nfa n;
    if (gram::compile_regex(std::string(50000, '(') + "a" + std::string(50000, ')'), n)) { fails++; printf("FAIL deep regex accepted\n"); }
    std::string deep = std::string(40, '(') + "a" + std::string(40, ')');
    runs++;
    gram::Nfa n2;
    if (gram::compile_regex(deep, n2)) { fails++; printf("FAIL 40 nested groups accepted\n"); }
    std::string ok = std::string(30, '(') + "a" + std::string(30, ')');
    runs++;
    gram::Nfa n3;
    if (!gram::compile_regex(ok, n3)) { fails++; printf("FAIL 30 nested groups refused\n"); }
  }
  refused(R"({"type":"string","pattern":")" + std::string(600, 'a') + R"("})", "pattern over 512 bytes");
  refused(R"({"type":"string","pattern":"((a{64}){64}){64}"})", "repetition blow-up");
  refused(R"({"type":"string","pattern":"(a)\\1"})", "back-reference");
  refused(R"({"type":"string","pattern":"(?=a)b"})", "lookahead");
  refused(R"({"type":"string","pattern":"a(b"})", "malformed pattern");
  refused(R"({"type":"strng"})", "unknown type");
  refused(R"({"$ref":"http://example.com/s.json"})", "remote $ref");
  refused(R"({"$ref":"#/$defs/missing"})", "unresolvable $ref");
  {
    std::string deep;
    for (int i = 0; i < 60; i++) deep += R"({"type":"array","items":)";
    deep += "{}";
    for (int i = 0; i < 60; i++) deep += "}";
    refused(deep, "60 nested subschemas");
    std::string big = R"({"enum":[)";
    for (int i = 0; i < 25000; i++) big += (i ? "," : "") + std::to_string(i);
    refused(big + "]}", "25,000 enum values (also over 64 KiB)");
    std::string many = R"({"type":"object","properties":{)";
    for (int i = 0; i < 5000; i++) many += (i ? "," : "") + std::string("\"k") + std::to_string(i) + R"(":{"type":"integer"})";
    refused(many + "}}", "5,000 properties");
  }

  // ---- advance(): the block is cut right after the token that activates the grammar
  {
    auto v = std::make_shared<gram::Vocab>();
    v->bytes = {"x", "", "{", "}", "\"a\""};  // 1: </think> (special)
    v->text = {1, 0, 1, 1, 1};
    gram::vocab() = v;
    auto g = std::make_shared<const gram::Grammar>(json::parse(R"({"type":"object"})"));
    gram::Matcher m(g, 1);
    int toks[] = {0, 0, 1, 2, 3};
    runs++;
    int k = gram::Matcher(m).advance(toks, 5, {}, false);
    if (k != 5) { fails++; printf("FAIL advance greedy %d\n", k); }
    runs++;
    gram::Matcher m2 = m;
    k = m2.advance(toks, 5, {}, true);
    if (k != 3 || !m2.active) { fails++; printf("FAIL advance stop_at_activation %d\n", k); }
    runs++;
    int bad[] = {0, 1, 0};  // "x" after </think> is not JSON
    if (gram::Matcher(m).advance(bad, 3, {}, false) != 2) { fails++; printf("FAIL advance rejection\n"); }
  }

  // ---- sampling frequencies (S03): Matcher::sample must draw from the masked distribution, per its contract
  auto bf = [](float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); };
  auto ff = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float x; memcpy(&x, &u, 4); return x; };
  // expected: allowed tokens by logit, top_k, softmax(temp), nucleus top_p (include the token that crosses it)
  auto expected = [&](const std::vector<uint16_t>& lg, const std::vector<int>& allowed, float temp, float top_p, int top_k) {
    std::vector<std::pair<float, int>> c;
    for (int t : allowed) c.push_back({ff(lg[t]), t});
    std::stable_sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    if (top_k > 0 && (int)c.size() > top_k) c.resize(top_k);
    std::vector<double> p(c.size());
    double sum = 0;
    for (size_t i = 0; i < c.size(); i++) sum += p[i] = std::exp((c[i].first - c[0].first) / temp);
    double acc = 0;
    size_t n = 0;
    while (n < c.size()) { acc += p[n++]; if (acc >= top_p * sum) break; }
    std::vector<double> out(lg.size(), 0);
    for (size_t i = 0; i < n; i++) out[c[i].second] = p[i] / acc;
    return out;
  };
  auto freq = [&](const char* name, const char* schema, const std::vector<std::string>& bytes,
                  const std::vector<float>& logits, const std::vector<int>& allowed, float temp, float top_p, int top_k) {
    auto v = std::make_shared<gram::Vocab>();
    v->bytes = bytes;
    v->text.assign(bytes.size(), 1);
    gram::vocab() = v;
    auto g = std::make_shared<const gram::Grammar>(json::parse(schema));
    std::vector<uint16_t> lg;
    for (float x : logits) lg.push_back(bf(x));
    const int V = (int)lg.size(), N = 200000;
    std::vector<int> cnt(V, 0);
    std::mt19937_64 rng(7);
    for (int i = 0; i < N; i++) { gram::Matcher m(g, -1); cnt[m.sample(lg.data(), V, temp, top_p, top_k, rng, {})]++; }
    auto e = expected(lg, allowed, temp, top_p, top_k);
    double chi = 0;
    int df = -1;
    bool leak = false;
    for (int t = 0; t < V; t++) {
      if (e[t] == 0) { leak |= cnt[t] > 0; continue; }
      chi += (cnt[t] - N * e[t]) * (cnt[t] - N * e[t]) / (N * e[t]);
      df++;
    }
    // chi-square with df degrees of freedom: mean df, sd sqrt(2 df); 6 sd is far beyond chance
    runs++;
    if (leak || chi > df + 6 * std::sqrt(2.0 * std::max(df, 1)) + 10) {
      fails++;
      printf("FAIL freq %s: chi2 %.1f (df %d)%s\n", name, chi, df, leak ? ", mass outside the support" : "");
    } else {
      printf("freq %-34s chi2 %6.1f df %d ok\n", name, chi, df);
    }
  };
  {  // the review's 17-token case: legal "a" in the top 16, legal "b" just outside, 15 illegal "x"
    std::vector<std::string> b{"\"a\""};
    for (int i = 1; i < 16; i++) b.push_back("\"x\"");
    b.push_back("\"b\"");
    std::vector<float> lg(17, -0.001f);
    lg[0] = 0;
    lg[16] = -0.01f;
    freq("codex 17-token (P(a) ~ 0.5025)", R"({"enum":["a","b"]})", b, lg, {0, 16}, 1.0f, 1.0f, 0);
  }
  {  // eight digits, odd ones illegal; top_k and top_p boundaries land on legal and illegal tokens
    std::vector<std::string> b;
    std::vector<float> lg;
    for (int i = 0; i < 8; i++) { b.push_back(std::to_string(i)); lg.push_back(2.0f - 0.35f * i); }
    const char* S = R"({"enum":[0,2,4,6]})";
    std::vector<int> al{0, 2, 4, 6};
    freq("digits, full", S, b, lg, al, 1.0f, 1.0f, 0);
    freq("digits, top_k 2", S, b, lg, al, 1.0f, 1.0f, 2);
    freq("digits, top_k 3, temp 0.7", S, b, lg, al, 0.7f, 1.0f, 3);
    freq("digits, top_p 0.6", S, b, lg, al, 1.0f, 0.6f, 0);
    freq("digits, top_p 0.9, top_k 3, temp 1.3", S, b, lg, al, 1.3f, 0.9f, 3);
  }

  printf("%d/%d passed\n", runs - fails, runs);
  return fails != 0;
}
