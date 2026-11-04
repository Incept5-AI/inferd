// Engine socket protocol helpers shared by tools/server.mm and its host tests (tools/json_test.cc): no Metal.
#pragma once
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>

namespace proto {

// The request id at the end of a line (the front end writes "id" last): the last "id": N in its last 64 bytes, or 0.
// Used for lines that cannot be parsed (Codex review 2026-11-04, R05) and oversized lines discarded as they stream in.
inline uint64_t tail_id(const std::string& line) {
  const std::string t = line.size() > 64 ? line.substr(line.size() - 64) : line;
  size_t k = t.rfind("\"id\"");
  if (k == std::string::npos) return 0;
  k += 4;
  while (k < t.size() && (t[k] == ' ' || t[k] == ':')) k++;
  uint64_t id = 0;
  int d = 0;
  for (; k < t.size() && isdigit((unsigned char)t[k]) && d < 19; k++, d++) id = id * 10 + (t[k] - '0');
  if (!d) return 0;
  while (k < t.size() && t[k] == ' ') k++;
  return k < t.size() && t[k] == '}' ? id : 0;  // "id" must be the line's last member: never a value inside a string
}

// Plain printable text for a "done" reason (the event line is JSON built from it).
inline std::string plain(const char* what, size_t max = 160) {
  std::string s;
  for (const char* p = what; *p && s.size() < max; p++)
    s += (std::isalnum((unsigned char)*p) || strchr(" .,:;_()+*?[]{}^$|/-", *p)) ? *p : '?';
  return s;
}

}  // namespace proto
