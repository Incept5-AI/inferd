# Structured output (OpenAI `response_format`) for serve.py: Python standard library only, no engine or model needed.
#   {"type": "json_object"}
#   {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": true}}
# The model is told what is required (an instruction appended to the last user turn, so the system prompt and the
# conversation so far stay a cacheable prefix), its answer is extracted (code fences and surrounding prose are
# dropped) and validated against the schema, and on failure serve.py asks again with the validation error.
import functools, json, re

DEFAULT_RETRIES = 2
MAX_RETRIES = 5

# Budgets, the same as the engine's (src/grammar.h, gram::kMax*): a schema over one is refused with HTTP 400.
MAX_SCHEMA_BYTES = 64 << 10
MAX_PATTERN_BYTES = 512
MAX_PATTERN_DEPTH = 32
MAX_REPEAT = 64
MAX_NFA_STATES = 2000
MAX_SCHEMA_DEPTH = 48
MAX_LITS = 20000
MAX_TEXT = 1 << 20           # characters of a reply that are searched for JSON
MATCH_BUDGET = 2_000_000     # NFA state visits per reply checked (patterns); past it the reply counts as invalid


def parse_response_format(rf):
    """-> None (plain text) or {"kind", "schema", "name", "strict"}; ValueError for a malformed response_format."""
    if rf is None:
        return None
    if not isinstance(rf, dict):
        raise ValueError("response_format must be an object")
    kind = rf.get("type", "text")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"kind": "json_object", "schema": {"type": "object"}, "name": None, "strict": False, "checked_only": []}
    if kind == "json_schema":
        js = rf.get("json_schema")
        if not isinstance(js, dict):
            raise ValueError("response_format.json_schema must be an object with a 'schema'")
        schema = js.get("schema", {})
        if not isinstance(schema, dict):
            raise ValueError("response_format.json_schema.schema must be a JSON Schema object")
        checked_only = check_schema(schema)
        return {"kind": "json_schema", "schema": schema, "name": js.get("name"), "strict": bool(js.get("strict")),
                "checked_only": checked_only}
    raise ValueError(f"response_format.type must be 'text', 'json_object' or 'json_schema', not {kind!r}")


def instruction(spec):
    """The text that tells the model what to produce."""
    if spec["kind"] == "json_object":
        return ("Respond with a single JSON object and nothing else: no code fences, no explanation before or after "
                "it.")
    name = f" ({spec['name']})" if spec.get("name") else ""
    return ("Respond with a single JSON value that conforms to the JSON Schema below" + name + ", and nothing else: "
            "no code fences, no explanation before or after it. Include every required property and no property "
            "the schema does not allow.\n\nJSON Schema:\n" + json.dumps(spec["schema"], ensure_ascii=False))


def add_instruction(messages, text):
    """-> messages with text appended to the last user message (a new user message if the last one is not a user's).
    The request's own list and dicts are not modified."""
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "user":
        m = dict(msgs[-1])
        c = m.get("content")
        if isinstance(c, list):  # multimodal parts
            m["content"] = c + [{"type": "text", "text": text}]
        else:
            m["content"] = ((c or "") + "\n\n" + text).lstrip("\n")
        msgs[-1] = m
    else:
        msgs.append({"role": "user", "content": text})
    return msgs


def retry_message(error):
    return (f"That reply was not acceptable: {error}. Reply again with only the corrected JSON, following the "
            "instructions above exactly.")


# ---------------------------------------------------------------- extraction

def _no_constants(name):  # json.loads accepts NaN/Infinity, JSON does not
    raise ValueError(f"{name} is not valid JSON")


_DEC = json.JSONDecoder(parse_constant=_no_constants)
_FENCE = re.compile(r"```[ \t]*(?:json|JSON|json5)?[ \t]*\n?(.*?)```", re.S)


def candidates(text, limit=64):
    """Yield (value, json text) for each JSON value the reply plausibly holds, best first: the whole reply, the inside
    of each code fence, then each '{' or '[' that starts a complete value (leading prose skipped)."""
    s = text[:MAX_TEXT].strip()
    seen = set()

    def one(t):
        t = t.strip()
        if t in seen or not t:
            return None
        seen.add(t)
        try:
            v, end = _DEC.raw_decode(t)
        except (ValueError, RecursionError):  # RecursionError: absurdly deep nesting
            return None
        return (v, t) if not t[end:].strip() else None

    if (c := one(s)) is not None:
        yield c
    for m in _FENCE.finditer(s):
        if (c := one(m.group(1))) is not None:
            yield c
    n = 0
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        n += 1
        if n > limit:
            return
        try:
            v, end = _DEC.raw_decode(s, i)
        except (ValueError, RecursionError):
            continue
        t = s[i:end]
        if t not in seen:
            seen.add(t)
            yield v, t


def check(text, spec):
    """-> (json text, None) for the first candidate that validates, else (None, error message). Bounded work: pattern
    checks share MATCH_BUDGET; serve.py runs this off the event loop."""
    first_err = None
    budget = [MATCH_BUDGET]
    for value, raw in candidates(text or ""):
        try:
            errs = validate(value, spec["schema"], budget=budget)
        except _OverBudget:
            errs = ["$: the reply is too expensive to check against the schema's patterns"]
        if not errs:
            return raw, None
        first_err = first_err or "; ".join(errs[:5])
    if first_err:
        return None, "the JSON does not match the schema: " + first_err
    if not (text or "").strip():
        return None, "the reply was empty (no JSON in the answer)"
    return None, "the reply contains no parseable JSON value"


# ---------------------------------------------------------------- JSON Schema (the common subset)
# type, enum, const, properties, required, additionalProperties, patternProperties, min/maxProperties, items (schema
# or tuple), prefixItems, min/maxItems, uniqueItems, contains, min/maxLength, pattern, minimum, maximum,
# exclusiveMinimum/Maximum (draft 4 booleans too), multipleOf, anyOf, oneOf, allOf, not, local $ref (#/$defs/...,
# #/definitions/...), OpenAPI's nullable, boolean schemas. format and other annotations are ignored.

_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")


def _type_of(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _is(v, t):
    a = _type_of(v)
    if t == "number":
        return a in ("integer", "number")
    if t == "integer":
        return a == "integer" or (a == "number" and float(v).is_integer())
    return a == t


def _eq(a, b):
    """JSON equality: true is not 1, 1 equals 1.0."""
    ta, tb = _type_of(a), _type_of(b)
    if {ta, tb} <= {"integer", "number"}:
        return a == b
    if ta != tb:
        return False
    if ta == "array":
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if ta == "object":
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


def _resolve(ref, root):
    if not ref.startswith("#"):
        raise ValueError(f"only local $ref is supported, not {ref!r}")
    node = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ValueError(f"$ref {ref!r} does not resolve")
    return node


def _key(v):
    """A hashable key with JSON equality (true is not 1, 1 equals 1.0), for uniqueItems in linear time."""
    t = _type_of(v)
    if t in ("integer", "number"):
        return ("n", int(v) if float(v).is_integer() and abs(v) < 2 ** 63 else v)
    if t == "array":
        return ("a", tuple(_key(x) for x in v))
    if t == "object":
        return ("o", frozenset((k, _key(x)) for k, x in v.items()))
    return (t, v)


def _short(v, n=40):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n - 3] + "..."


def validate(v, schema, root=None, path="$", depth=0, budget=None):
    """-> list of error strings (empty when v conforms). Work is bounded: every call and every pattern step spends
    from `budget` (a one-item list; a fresh MATCH_BUDGET when None) and _OverBudget is raised when it runs out."""
    root = schema if root is None else root
    if budget is None:
        budget = [MATCH_BUDGET]
    budget[0] -= 1
    if budget[0] < 0:
        raise _OverBudget()
    if depth > 128:
        return [f"{path}: schema nesting too deep (recursive $ref?)"]
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    if not isinstance(schema, dict):
        return []
    if "$ref" in schema:
        try:
            target = _resolve(schema["$ref"], root)
        except (ValueError, IndexError) as e:
            return [f"{path}: {e}"]
        errs = validate(v, target, root, path, depth + 1, budget)
        rest = {k: x for k, x in schema.items() if k != "$ref"}
        return errs + (validate(v, rest, root, path, depth + 1, budget) if rest else [])
    errs = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if schema.get("nullable"):
            ts = ts + ["null"]
        if not any(_is(v, x) for x in ts):
            return [f"{path}: expected {' or '.join(ts)}, got {_type_of(v)}"]
    elif schema.get("nullable") is False and v is None:
        return [f"{path}: null is not allowed"]
    if v is None and schema.get("nullable"):
        return []
    if "enum" in schema and not any(_eq(v, e) for e in schema["enum"]):
        errs.append(f"{path}: {_short(v)} is not one of {_short(schema['enum'], 120)}")
    if "const" in schema and not _eq(v, schema["const"]):
        errs.append(f"{path}: must be {_short(schema['const'])}")

    if isinstance(v, str):
        n = len(v)
        if "minLength" in schema and n < schema["minLength"]:
            errs.append(f"{path}: string is {n} characters, minimum is {schema['minLength']}")
        if "maxLength" in schema and n > schema["maxLength"]:
            errs.append(f"{path}: string is {n} characters, maximum is {schema['maxLength']}")
        if "pattern" in schema and not pattern(schema["pattern"]).search(v, budget):
            errs.append(f"{path}: {_short(v)} does not match the pattern {schema['pattern']!r}")
    elif _type_of(v) in ("integer", "number"):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if xlo is True:  # draft 4
            xlo, lo = lo, None
        if xhi is True:
            xhi, hi = hi, None
        if isinstance(lo, (int, float)) and not isinstance(lo, bool) and v < lo:
            errs.append(f"{path}: {v} is less than the minimum {lo}")
        if isinstance(hi, (int, float)) and not isinstance(hi, bool) and v > hi:
            errs.append(f"{path}: {v} is greater than the maximum {hi}")
        if isinstance(xlo, (int, float)) and not isinstance(xlo, bool) and v <= xlo:
            errs.append(f"{path}: {v} must be greater than {xlo}")
        if isinstance(xhi, (int, float)) and not isinstance(xhi, bool) and v >= xhi:
            errs.append(f"{path}: {v} must be less than {xhi}")
        m = schema.get("multipleOf")
        if isinstance(m, (int, float)) and m > 0:
            q = v / m
            if abs(q - round(q)) > 1e-9 * max(1.0, abs(q)):
                errs.append(f"{path}: {v} is not a multiple of {m}")
    elif isinstance(v, list):
        n = len(v)
        if "minItems" in schema and n < schema["minItems"]:
            errs.append(f"{path}: {n} items, minimum is {schema['minItems']}")
        if "maxItems" in schema and n > schema["maxItems"]:
            errs.append(f"{path}: {n} items, maximum is {schema['maxItems']}")
        prefix = schema.get("prefixItems")
        items = schema.get("items")
        if isinstance(items, list):  # draft 4-7 tuple form
            prefix, items = items, schema.get("additionalItems", True)
        k = 0
        if isinstance(prefix, list):
            for k, (x, s) in enumerate(zip(v, prefix)):
                errs += validate(x, s, root, f"{path}[{k}]", depth + 1, budget)
            k = min(len(prefix), n)
        if items is not None:
            for i in range(k, n):
                errs += validate(v[i], items, root, f"{path}[{i}]", depth + 1, budget)
        if schema.get("uniqueItems"):
            seen = set()
            for x in v:
                k = _key(x)
                if k in seen:
                    errs.append(f"{path}: items must be unique ({_short(x)} repeats)")
                    break
                seen.add(k)
        if "contains" in schema and not any(not validate(x, schema["contains"], root, path, depth + 1, budget) for x in v):
            errs.append(f"{path}: no item matches 'contains'")
    elif isinstance(v, dict):
        props = schema.get("properties") or {}
        for r in schema.get("required") or []:
            if r not in v:
                errs.append(f"{path}: missing required property {r!r}")
        if "minProperties" in schema and len(v) < schema["minProperties"]:
            errs.append(f"{path}: {len(v)} properties, minimum is {schema['minProperties']}")
        if "maxProperties" in schema and len(v) > schema["maxProperties"]:
            errs.append(f"{path}: {len(v)} properties, maximum is {schema['maxProperties']}")
        pprops = schema.get("patternProperties") or {}
        extra = schema.get("additionalProperties", True)
        for key, x in v.items():
            sub = f"{path}.{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else f"{path}[{json.dumps(key)}]"
            matched = False
            if key in props:
                matched = True
                errs += validate(x, props[key], root, sub, depth + 1, budget)
            for pat, s in pprops.items():
                if pattern(pat).search(key, budget):
                    matched = True
                    errs += validate(x, s, root, sub, depth + 1, budget)
            if not matched:
                if extra is False:
                    allowed = ", ".join(map(repr, props)) or "none"
                    errs.append(f"{path}: property {key!r} is not allowed (allowed: {allowed})")
                elif isinstance(extra, dict):
                    errs += validate(x, extra, root, sub, depth + 1, budget)

    for k in ("allOf",):
        for s in schema.get(k) or []:
            errs += validate(v, s, root, path, depth + 1, budget)
    if "anyOf" in schema:
        subs = [validate(v, s, root, path, depth + 1, budget) for s in schema["anyOf"]]
        if all(subs):
            errs.append(f"{path}: matches none of anyOf ({'; '.join(e[0] for e in subs if e)})")
    if "oneOf" in schema:
        subs = [validate(v, s, root, path, depth + 1, budget) for s in schema["oneOf"]]
        ok = sum(1 for e in subs if not e)
        if ok == 0:
            errs.append(f"{path}: matches none of oneOf ({'; '.join(e[0] for e in subs if e)})")
        elif ok > 1:
            errs.append(f"{path}: matches {ok} of oneOf, exactly one is required")
    if "not" in schema and not validate(v, schema["not"], root, path, depth + 1, budget):
        errs.append(f"{path}: must not match the 'not' schema")
    return errs


# ---------------------------------------------------------------- patterns: one bounded regex subset
# The same subset and limits as the engine's grammar (src/grammar.h, compile_regex), so a pattern means the same in
# both places: literals; . (not newline); classes [a-z] [^...]; escapes \d \w \s \D \W \S \n \t \r \f \v and escaped
# punctuation; groups ( ) and (?: ); |; quantifiers * + ? {m} {m,} {m,n} (m, n <= 64; lazy/possessive suffixes allowed,
# same language); ^ only at the start, $ only at the end; ASCII only. Unanchored ends search, as JSON Schema's pattern
# does. Anything else (back-references, lookaround, \b, \p{..}, inline flags, non-ASCII) is refused.
# Matching is a Thompson NFA simulation: linear in the text, no backtracking.

class _OverBudget(Exception):
    pass


_ALL = (1 << 129) - 1  # 0..127 ASCII, 128 any non-ASCII character
_DIGIT = sum(1 << c for c in range(48, 58))
_WORD = _DIGIT | sum(1 << c for c in range(65, 91)) | sum(1 << c for c in range(97, 123)) | (1 << 95)
_SPACE = sum(1 << ord(c) for c in " \t\n\r\f\v")


class _ReParser:
    def __init__(self, p, i, end):
        self.p, self.i, self.end, self.depth = p, i, end, 0

    def bad(self, what="unsupported construct"):
        raise ValueError(f"{what} at offset {self.i}")

    def esc(self):
        if self.i >= self.end:
            self.bad("dangling backslash")
        c = self.p[self.i]
        self.i += 1
        cls = {"d": _DIGIT, "D": _ALL ^ _DIGIT, "w": _WORD, "W": _ALL ^ _WORD, "s": _SPACE, "S": _ALL ^ _SPACE,
               "n": 1 << 10, "t": 1 << 9, "r": 1 << 13, "f": 1 << 12, "v": 1 << 11}
        if c in cls:
            return cls[c]
        if ord(c) < 128 and not c.isalnum():
            return 1 << ord(c)
        self.bad(f"escape \\{c}")

    @staticmethod
    def single(m):
        return m.bit_length() - 1 if m and m & (m - 1) == 0 else None

    def cls(self):
        p, s = self.p, 0
        neg = self.i < self.end and p[self.i] == "^"
        if neg:
            self.i += 1
        first = True
        while self.i < self.end and (p[self.i] != "]" or first):
            first = False
            c = ord(p[self.i])
            self.i += 1
            if c >= 128:
                self.bad("non-ASCII character")
            if c == 92:  # backslash
                e = self.esc()
                one = self.single(e)
                if one is None or one >= 128 or self.i >= self.end or p[self.i] != "-":
                    s |= e
                    continue
                c = one
            if self.i + 1 < self.end and p[self.i] == "-" and p[self.i + 1] != "]":
                d = ord(p[self.i + 1])
                self.i += 2
                if d == 92:
                    one = self.single(self.esc())
                    if one is None:
                        self.bad("class range")
                    d = one
                if d >= 128 or d < c:
                    self.bad("class range")
                s |= sum(1 << k for k in range(c, d + 1))
            else:
                s |= 1 << c
        if self.i >= self.end:
            self.bad("unterminated class")
        self.i += 1
        return (_ALL ^ s) if neg else s

    def atom(self):
        p = self.p
        c = p[self.i]
        self.i += 1
        if c == "(":
            self.depth += 1
            if self.depth > MAX_PATTERN_DEPTH:
                self.bad("groups nested too deeply")
            if p.startswith("?:", self.i):
                self.i += 2
            elif self.i < self.end and p[self.i] == "?":
                self.bad("(? construct")
            x = self.alt()
            if self.i >= self.end or p[self.i] != ")":
                self.bad("unbalanced (")
            self.i += 1
            self.depth -= 1
            return x
        if c == "[":
            return ("set", self.cls())
        if c == ".":
            return ("set", _ALL ^ (1 << 10))
        if c == "\\":
            return ("set", self.esc())
        if ord(c) >= 128 or c in "^$)*+?|":
            self.bad(f"{c!r}")
        return ("set", 1 << ord(c))

    def num(self):
        j, v = self.i, 0
        while self.i < self.end and "0" <= self.p[self.i] <= "9" and v < 100000:
            v = v * 10 + ord(self.p[self.i]) - 48
            self.i += 1
        return v if self.i > j else None

    def quant(self, a):
        p = self.p
        while self.i < self.end:
            c = p[self.i]
            if c == "*":
                lo, hi = 0, -1
                self.i += 1
            elif c == "+":
                lo, hi = 1, -1
                self.i += 1
            elif c == "?":
                lo, hi = 0, 1
                self.i += 1
            elif c == "{":
                save = self.i
                self.i += 1
                lo = self.num()
                if lo is None:
                    self.i = save
                    return a
                hi = lo
                if self.i < self.end and p[self.i] == ",":
                    self.i += 1
                    hi = self.num()
                    hi = -1 if hi is None else hi
                if self.i >= self.end or p[self.i] != "}":
                    self.i = save
                    return a
                self.i += 1
            else:
                return a
            if self.i < self.end and p[self.i] in "?+":
                self.i += 1
            if lo > MAX_REPEAT or hi > MAX_REPEAT or (hi >= 0 and hi < lo):
                self.bad(f"repetition bounds over {MAX_REPEAT}")
            a = ("rep", a, lo, hi)
        return a

    def cat(self):
        k = []
        while self.i < self.end and self.p[self.i] not in "|)":
            k.append(self.quant(self.atom()))
        return ("cat", k)

    def alt(self):
        k = [self.cat()]
        while self.i < self.end and self.p[self.i] == "|":
            self.i += 1
            k.append(self.cat())
        return k[0] if len(k) == 1 else ("alt", k)


class Pattern:
    """A compiled pattern: states are [kind, set, a, b] (0 character in set -> a, 1 split a | b, 2 accept)."""

    def __init__(self, src):
        if not isinstance(src, str):
            raise ValueError("a pattern must be a string")
        if len(src) > MAX_PATTERN_BYTES:
            raise ValueError(f"longer than {MAX_PATTERN_BYTES} characters")
        ps = _ReParser(src, 0, len(src))
        a0 = src.startswith("^")
        if a0:
            ps.i = 1
        a1 = False
        if ps.end > ps.i and src[ps.end - 1] == "$":
            bs, j = 0, ps.end - 1
            while j > ps.i and src[j - 1] == "\\":
                bs, j = bs + 1, j - 1
            if bs % 2 == 0:
                a1, ps.end = True, ps.end - 1
        tree = ps.alt()
        if ps.i != ps.end:
            ps.bad("unbalanced )")
        self.st = []
        out = self._add(2)
        if not a1:
            e = self._add(1, 0, -1, out)
            self.st[e][2] = self._add(0, _ALL, e)
            out = e
        start = self._comp(tree, out)
        if not a0:
            lp = self._add(1, 0, -1, start)
            self.st[lp][2] = self._add(0, _ALL, lp)
            start = lp
        self.start = start
        self._clo = {}
        self._dfa = {}

    def _add(self, kind, cset=0, a=-1, b=-1):
        if len(self.st) >= MAX_NFA_STATES:
            raise ValueError(f"compiles to more than {MAX_NFA_STATES} states")
        self.st.append([kind, cset, a, b])
        return len(self.st) - 1

    def _comp(self, r, out):
        kind = r[0]
        if kind == "set":
            return self._add(0, r[1], out)
        if kind == "cat":
            for x in reversed(r[1]):
                out = self._comp(x, out)
            return out
        if kind == "alt":
            s = self._comp(r[1][-1], out)
            for x in reversed(r[1][:-1]):
                s = self._add(1, 0, self._comp(x, out), s)
            return s
        _, x, lo, hi = r
        t = out
        if hi < 0:
            sp = self._add(1, 0, -1, out)
            self.st[sp][2] = self._comp(x, sp)
            t = sp
        else:
            for _ in range(hi - lo):
                t = self._add(1, 0, self._comp(x, t), out)
        for _ in range(lo):
            t = self._comp(x, t)
        return t

    def _closure(self, i):  # -> bitmask of the states reachable from i without input
        c = self._clo.get(i)
        if c is None:
            c, stack = 0, [i]
            while stack:
                k = stack.pop()
                if k < 0 or c >> k & 1:
                    continue
                c |= 1 << k
                if self.st[k][0] == 1:
                    stack += (self.st[k][2], self.st[k][3])
            self._clo[i] = c
        return c

    def search(self, text, budget=None):
        """True when the pattern matches somewhere in text (JSON Schema semantics). Spends one budget unit per active
        state per character (bounded: no backtracking); raises _OverBudget when the budget runs out."""
        cur, st, dfa = self._closure(self.start), self.st, self._dfa
        for ch in text:
            o = ord(ch)
            code = o if o < 128 else 128
            nxt = dfa.get((cur, code))
            if nxt is None:
                nxt, x, n = 0, cur, 0
                while x:
                    low = x & -x
                    k = low.bit_length() - 1
                    x ^= low
                    n += 1
                    if st[k][0] == 0 and st[k][1] >> code & 1:
                        nxt |= self._closure(st[k][2])
                if budget is not None:
                    budget[0] -= n
                    if budget[0] < 0:
                        raise _OverBudget()
                if len(dfa) < 4096:
                    dfa[(cur, code)] = nxt
            elif budget is not None:
                budget[0] -= 1
                if budget[0] < 0:
                    raise _OverBudget()
            cur = nxt
            if not cur:
                return False
        x = cur
        while x:
            low = x & -x
            if st[low.bit_length() - 1][0] == 2:
                return True
            x ^= low
        return False


@functools.lru_cache(maxsize=256)
def pattern(src):
    """-> Pattern (cached); ValueError for a pattern outside the subset or its limits."""
    return Pattern(src)


# ---------------------------------------------------------------- schema check (before anything runs)

# Keywords the validator implements. The grammar enforces most of them while decoding; the rest are checked after
# decoding only, and a strict reply's "inferd" block lists those that appear in the schema (CHECKED_ONLY).
_SUPPORTED = {"type", "enum", "const", "properties", "required", "additionalProperties", "patternProperties",
              "minProperties", "maxProperties", "items", "prefixItems", "additionalItems", "minItems", "maxItems",
              "uniqueItems", "contains", "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
              "exclusiveMaximum", "multipleOf", "anyOf", "oneOf", "allOf", "not", "$ref", "nullable", "$defs",
              "definitions"}
CHECKED_ONLY = ("anyOf", "oneOf", "allOf", "not", "contains", "minProperties", "maxProperties", "multipleOf",
                "uniqueItems", "patternProperties", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
# Validation keywords of JSON Schema that this subset does not implement: refused rather than ignored.
_UNSUPPORTED = {"if", "then", "else", "dependentRequired", "dependentSchemas", "dependencies", "propertyNames",
                "unevaluatedProperties", "unevaluatedItems", "minContains", "maxContains", "$dynamicRef",
                "$recursiveRef", "$dynamicAnchor", "$recursiveAnchor", "contentSchema"}
_TYPE_NAMES = {"null", "boolean", "integer", "number", "string", "array", "object"}


def check_schema(schema):
    """Refuse a schema outside the supported subset or over a budget (ValueError, a 400 for the client), and return
    the CHECKED_ONLY keywords it uses. Other keywords (title, description, default, examples, format, x-...) are
    annotations, ignored as JSON Schema requires (format date/date-time/time/uuid is also enforced while decoding)."""
    try:
        text = json.dumps(schema, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as e:
        raise ValueError(f"schema is not valid JSON: {e}") from None
    if len(text.encode()) > MAX_SCHEMA_BYTES:
        raise ValueError(f"schema is larger than {MAX_SCHEMA_BYTES >> 10} KiB")
    used, lits = set(), [0]

    def nonneg_int(v, where):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"{where} must be a non-negative integer")

    def number(v, where):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{where} must be a number")

    def walk(s, path, depth, allow_false=False):
        if depth > MAX_SCHEMA_DEPTH:
            raise ValueError(f"schema is nested deeper than {MAX_SCHEMA_DEPTH} levels")
        if s is True:
            return
        if s is False:
            if allow_false:
                return
            raise ValueError(f"{path}: false subschemas are only supported for additionalProperties and items")
        if not isinstance(s, dict):
            raise ValueError(f"{path}: a subschema must be an object")
        for k, v in s.items():
            at = f"{path}.{k}"
            if k in _UNSUPPORTED:
                raise ValueError(f"{at}: the keyword {k!r} is not supported")
            if k not in _SUPPORTED:
                continue
            if k in CHECKED_ONLY:
                used.add(k)
            if k == "type":
                ts = v if isinstance(v, list) else [v]
                if not ts or not all(isinstance(t, str) and t in _TYPE_NAMES for t in ts):
                    raise ValueError(f"{at}: unknown type {v!r}")
            elif k == "enum":
                if not isinstance(v, list) or not v:
                    raise ValueError(f"{at}: enum must be a non-empty list")
                lits[0] += len(v)
            elif k == "const":
                lits[0] += 1
            elif k in ("properties", "patternProperties", "$defs", "definitions"):
                if not isinstance(v, dict):
                    raise ValueError(f"{at}: must be an object of subschemas")
                for name, sub in v.items():
                    if k == "patternProperties":
                        try:
                            pattern(name)
                        except ValueError as e:
                            raise ValueError(f"{at}: pattern {name!r}: {e}") from None
                    walk(sub, f"{at}[{json.dumps(name)}]", depth + 1)
            elif k == "required":
                if not isinstance(v, list) or not all(isinstance(r, str) for r in v):
                    raise ValueError(f"{at}: required must be a list of strings")
            elif k in ("additionalProperties", "additionalItems", "contains", "not"):
                walk(v, at, depth + 1, allow_false=k in ("additionalProperties", "additionalItems"))
            elif k == "items":
                if isinstance(v, list):
                    for i, sub in enumerate(v):
                        walk(sub, f"{at}[{i}]", depth + 1)
                else:
                    walk(v, at, depth + 1, allow_false=True)
            elif k in ("prefixItems", "anyOf", "oneOf", "allOf"):
                if not isinstance(v, list) or not v:
                    raise ValueError(f"{at}: must be a non-empty list of subschemas")
                for i, sub in enumerate(v):
                    walk(sub, f"{at}[{i}]", depth + 1)
            elif k in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
                nonneg_int(v, at)
            elif k in ("minimum", "maximum"):
                number(v, at)
            elif k in ("exclusiveMinimum", "exclusiveMaximum"):
                if not isinstance(v, bool):
                    number(v, at)
            elif k == "multipleOf":
                number(v, at)
                if v <= 0:
                    raise ValueError(f"{at}: multipleOf must be positive")
            elif k in ("uniqueItems", "nullable"):
                if not isinstance(v, bool):
                    raise ValueError(f"{at}: must be true or false")
            elif k == "pattern":
                try:
                    pattern(v)
                except ValueError as e:
                    raise ValueError(f"{at}: {e}") from None
            elif k == "$ref":
                if not isinstance(v, str) or not v.startswith("#"):
                    raise ValueError(f"{at}: only local references (#/...) are supported")
                try:
                    _resolve(v, schema)
                except (ValueError, IndexError, KeyError, TypeError):
                    raise ValueError(f"{at}: {v!r} does not resolve") from None
        if lits[0] > MAX_LITS:
            raise ValueError(f"schema has more than {MAX_LITS} enum/const values")

    walk(schema, "$", 0)
    return sorted(used)


# ---------------------------------------------------------------- the vocabulary for constrained decoding (inferd)

def _bytes_to_unicode():
    """GPT-2 byte-level BPE: the printable character that stands for each byte -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def vocab_op(tok):
    """-> the inferd "vocab" op line (bytes): every token id's bytes as hex, and the added/special ids (never JSON)."""
    dec, n = _bytes_to_unicode(), len(tok)
    added = set(getattr(tok, "added_tokens_decoder", {}) or {})
    pieces = tok.convert_ids_to_tokens(list(range(n)))
    hexes, special = [], []
    for i, p in enumerate(pieces):
        if p is None or i in added:
            hexes.append("")
            special.append(i)
            continue
        if all(ch in dec for ch in p):  # byte-level BPE (Qwen, GPT-2 style)
            b = bytes(dec[ch] for ch in p)
        elif (m := re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", p)):  # sentencepiece byte fallback
            b = bytes([int(m.group(1), 16)])
        else:  # sentencepiece piece
            b = p.replace("▁", " ").encode()
        hexes.append(b.hex())
    return (json.dumps({"op": "vocab", "hex": hexes, "special": special}) + "\n").encode()
