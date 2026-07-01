# Technical Lessons

A running log of technical lessons learned while working through the assignments.
Organized by topic. Entries are meant to capture the *why* behind a concept, not
just the fix, so they stay useful later.

> How to use this file: when a debugging session or discussion surfaces a
> reusable concept, add it under the right category (create a new category if
> needed). Prefer explaining the underlying mechanism and the mental model over
> pasting assignment solution code. Keep snippets small and generic/illustrative.

## Table of Contents

- [Python / Language](#python--language)
- [Regex](#regex)
- [Multiprocessing / Concurrency](#multiprocessing--concurrency)
- [BPE / Tokenizer](#bpe--tokenizer)
- [PyTorch](#pytorch) *(placeholder)*
- [Automatic Differentiation](#automatic-differentiation) *(placeholder)*
- [Transformer](#transformer) *(placeholder)*
- [Adam / Optimization](#adam--optimization) *(placeholder)*

---

## Python / Language

### `str` vs `bytes`, and `encode`/`decode`

- `bytes` is raw bytes; `str` is Unicode text. They are **not** interchangeable,
  and most APIs require you to pick one consistently.
- Convert with the mirror-image methods:
  - `bytes.decode("utf-8")` -> `str` (bytes interpreted as text via an encoding)
  - `str.encode("utf-8")` -> `bytes`
- Match each function's expected type **at the call site**, converting at the
  boundary. Example that bit us: `find_chunk_boundaries` reads a file in binary
  mode and *asserts* its special token is `bytes`, while `re.split` on a decoded
  `str` chunk needs a `str` pattern. Keep a `bytes` copy and derive a `str` copy
  (`token.decode("utf-8")`) where needed instead of forcing one type everywhere.
- For ASCII tokens the round-trip is lossless and trivial.

### The bytes-in-f-string trap

- Formatting a `bytes` object into an f-string calls `str()` on it, which inserts
  the **repr** including the `b'...'` prefix and quotes.
- Symptom we hit: `f"({re.escape(b'<|endoftext|>')})"` produced the pattern
  `(b'<\\|endoftext\\|>')` — literally trying to match the characters `b`, `'`,
  ... so the regex matched nothing (a split returned exactly 1 element = no
  matches).
- Debug technique: `print(repr(pattern))` and look for a stray `b'...'` wrapper
  or doubled backslashes. That immediately reveals type contamination.
- Fix: keep the value as `str` before putting it in a `str` f-string.

### Raw strings `r"..."` and triple quotes `"""..."""`

- `r"..."` = **raw string**: backslash escapes are *not* processed by Python, so
  the characters (e.g. `\p`, `\s`, `\n`) are handed to the regex engine literally.
  Always use the `r` prefix for regex patterns to avoid double-backslashing.
- `"""..."""` = triple quotes: lets the string span lines and contain quote
  characters without escaping. Combined `r"""..."""` = raw + quote-safe.
- Key distinction: the `r` prefix matters for the **pattern** you write. The
  **text being searched** is a separate string; if it's a normal (non-raw)
  literal, its escapes are processed normally (so `"...\n"` genuinely contains a
  single newline character, not `\` + `n`).

### Escape sequences are single characters

- In a normal string literal, `\n` is **one** character (newline, code point 10),
  not a backslash followed by `n`. `repr`/`print(list(...))` *displays* it as
  `\n` as a visibility convention.
- Verify: `len(s)`, `s[-1] == "\n"`, `ord(s[-1]) == 10`.
- You only get two characters (`\` and `n`) if the data actually contains a
  backslash, e.g. a raw literal `r"...\n"` or an escaped `"...\\n"`.

### `dict`: ordering, methods, and the map concept

- `dict` **is** Python's hash map / associative array (there's no separate
  `std::map`-style class). The builtin `map()` function is unrelated — it lazily
  applies a function over an iterable.
- Since Python 3.7, `dict` **preserves insertion order** on iteration. But
  insertion-ordered != sorted; to rank by value you still call
  `sorted(d.items(), key=lambda pair: pair[1], reverse=True)`.
- Useful methods: `d.get(key, default)`, `d.setdefault(key, default)`,
  `d.pop(key, default)`, `d.update(other)`, `d.keys()/.values()/.items()`,
  `key in d`, `len(d)`.
- `.keys()/.values()/.items()` return **views** (live windows, not lists). Wrap
  in `list(...)` to snapshot. Iterating a dict directly yields **keys** (that's
  why `for k, v in d:` fails — use `d.items()`).

### Counting patterns: `.get`, `defaultdict`, `Counter`

- `d[k] += 1` on a plain dict raises `KeyError` on the first occurrence because
  the right-hand read happens before the key exists.
- Three idiomatic fixes, roughly increasing elegance for counting:
  1. `d[k] = d.get(k, 0) + 1` — no imports; note you must **assign back**
     (`d.get(...) + 1` alone computes and discards).
  2. `collections.defaultdict(int)` then `d[k] += 1`.
  3. `collections.Counter` then `cnt[k] += 1`, or `cnt.update(iterable)` to count
     a whole iterable at once, and `cnt.most_common(n)` for top-N for free.
- `collections` is **standard library** (no install) and idiomatic — also common
  in competitive programming precisely because it makes code *shorter*
  (`Counter`, `defaultdict`, `deque`, `heapq`, `bisect`).

### `defaultdict` — the argument is a value factory, not a key type

- `defaultdict(factory)` calls `factory()` (no args) to produce the **default
  value** for a missing key, then inserts and returns it.
  - `int` -> `0`, `list` -> `[]`, `set` -> `set()`, `lambda: "n/a"` -> custom.
- It says nothing about **key** types; keys can be any hashable, same as `dict`.
- Mechanism: stored as `default_factory`; the magic is the `__missing__` method,
  which plain `dict` lacks (hence its `KeyError`).
- Gotcha: **reading** a missing key *inserts* it (side effect). Use `k in d` or
  `d.get(k)` to probe without inserting.

### Type enforcement (there is none at runtime)

- Python does **not** enforce annotations at runtime for any dict, `defaultdict`
  included. `pretoken_cnt: defaultdict[bytes, int] = defaultdict(int)` documents
  intent and lets static checkers (mypy/pyright) flag mismatches — but running
  code with wrong types still "works".
- The subscripted `int` in an annotation (`defaultdict[bytes, int]`) is the
  *value type* for the checker; the `int` passed to the constructor is the
  runtime *value factory*. Same word, different roles.
- Runtime enforcement is possible by subclassing and overriding `__setitem__`,
  but it's rarely worth the overhead; Python leans on duck typing + static checks.

### Iterator protocol (iterable vs iterator)

- **Iterable**: has `__iter__` (can *produce* an iterator). E.g. `list`, `dict`,
  `str` — each `__iter__` call makes a *fresh* iterator, so they're re-iterable.
- **Iterator**: has both `__next__` (produce next / raise `StopIteration`) and
  `__iter__` (returns `self`). E.g. generators, `enumerate`, `re.finditer`'s
  Scanner.
- `for x in obj:` needs `obj` iterable; it calls `iter(obj)` then `next()`
  repeatedly. Hallmark of an iterator: `iter(it) is it` -> `True`.
- **Iterators are one-shot**: once exhausted, a second loop yields nothing. To
  reuse, materialize with `list(...)` or recreate. Laziness saves memory at the
  cost of single use.

### `map` and calling a method per element

- `map(func, iterable)` needs `func` to be callable taking one element. To call a
  *method* on each element you can't pass the bare method name (`group` is not a
  free function -> `NameError`). Use:
  - `map(lambda m: m.group(), it)`, or
  - `from operator import methodcaller; map(methodcaller("group"), it)`, or
  - the usually-cleaner comprehension `[m.group() for m in it]`.
- `map` is lazy (returns an iterator) — wrap in `list(...)` to see/reuse.

---

## Regex

### `re.escape`: for literal data, never for a pattern you wrote

- `re.escape(s)` backslash-escapes every regex metacharacter so `s` matches
  **literally**. Use it for *data you want matched verbatim* (e.g. a special
  token `<|endoftext|>`, whose `|` would otherwise mean alternation).
- **Do not** `re.escape` a pattern you authored (like the GPT-2 `PAT`): escaping
  turns its intentional metacharacters (`?`, `+`, `|`, `[]`, `\p`) into literals,
  so it only matches the literal text of the pattern and finds nothing.

### `re.split` behavior

- `re.split(pattern, s)` splits on matches and returns the pieces.
- **Capturing group keeps the delimiter**: `re.split(r"(,)", "a,b")` ->
  `['a', ',', 'b']`; without the group the delimiter is dropped. This is the
  lever for "split but still see what I split on" (e.g. to count special tokens).
- Leading/trailing/adjacent matches yield empty-string pieces — often filter them.
- Multiple delimiters via alternation: `re.split(r"X|Y", s)`. Build dynamically
  and safely with `"|".join(re.escape(t) for t in tokens)`.

### Builtin `re` vs the `regex` package (`\p{...}`)

- The **builtin `re`** module does **not** support Unicode-property escapes like
  `\p{L}` (any letter) or `\p{N}` (any number) -> raises `re.error: bad escape \p`.
- The GPT-2 pre-tokenizer pattern uses `\p{L}`/`\p{N}`, so it requires the
  third-party **`regex`** package (commonly `import regex as re`). `regex` is a
  superset of `re`.

### `finditer`, Scanner, and match objects

- `re.finditer(pattern, s)` is **lazy**: it returns an *iterator* (called a
  `callable_iterator` in builtin `re`, a `Scanner` in `regex`), not the matches.
  Iterating it yields **match objects** one at a time.
- So `pretokens = re.finditer(...)` is the iterator; in `for m in pretokens:`
  each `m` **is** a match object. Get the matched text with `m.group()` (a match
  object is not the string — keying a dict on the match object is a bug).
- Being an iterator, it's one-shot (see iterator protocol above).

---

## Multiprocessing / Concurrency

### Processes vs threads (no shared memory, no GIL serialization)

- The "shared global dict + lock" model is a **threading** intuition. With
  `multiprocessing`, each worker is a separate OS process with its **own** memory
  — nothing shared to lock, and the GIL does not serialize them (the main reason
  to use processes for CPU-bound work).
- Consequence: data to/from workers is **pickled** and copied across the process
  boundary. Model = each worker builds a **local** result and returns it; the
  **parent merges**.

### `Pool` API

- `Pool.map(func, iterable)` — one func over each item, results in order.
- `Pool.starmap(func, iterable_of_tuples)` — unpacks tuples for multi-arg funcs
  (build args with `zip(...)`).
- `Pool.imap` / `imap_unordered` — lazy streaming variants.
- Low-level `Process(target=..., args=...)` + `.start()`/`.join()` for manual
  control.
- The worker **function must be top-level** (picklable) — lambdas / nested
  functions can't be pickled for a `Pool`.

### Passing offsets vs bytes (and file descriptors)

- Don't pass a giant chunk (it gets pickled + copied into the worker). Pass the
  **file path + `(start, end)`** and let each worker `open` and read its own slice
  — only a few ints/strings cross the boundary.
- Don't try to pass an open file object / `f`: file objects generally aren't
  picklable, and a raw fd integer refers to a slot in the *parent's* fd table —
  invalid in the child.
- Concurrent **reads** are safe with no lock: each worker's own `open()` gives an
  **independent file handle with its own offset**, so seeks don't interfere, and
  reads don't mutate the file. (Contention/locking concerns are about writes or a
  *shared* handle.)

### `fork` vs `spawn` and the `if __name__ == "__main__"` guard

- Start methods:
  - **`fork`** (historical Linux default): child is a memory clone; it does **not**
    re-import/re-run your module. Top-level code runs once (in the parent).
  - **`spawn`** (macOS/Windows default; Linux is moving toward it in 3.14 because
    `fork` interacts badly with threads): child is a fresh interpreter that
    **re-imports** your module to rebuild the worker function.
- The guard matters under **spawn**: re-importing re-runs top-level code; without
  the guard that re-creates a `Pool` -> recursive process explosion. Under `fork`
  it's not strictly needed but is the portable, robust habit.
- `__name__` in the child:
  - **fork**: unchanged from the parent (memory copy) — but moot, since top level
    never re-runs.
  - **spawn**: the re-imported module is loaded under the special name
    **`__mp_main__`** (not `"__main__"`), so `if __name__ == "__main__":` is
    `False` in the child and the driver block is skipped. In the parent (the entry
    script) it's `"__main__"` -> `True`.

### `fork` return value and how `Pool` uses it

- `os.fork()` is called once, returns twice: **child sees `0`**, **parent sees the
  child's PID** (positive); failure raises `OSError`.
- `multiprocessing` (in `popen_fork.py`) uses exactly this:
  `pid = os.fork(); if pid == 0: run worker bootstrap; os._exit(...)`. The child
  is diverted into the worker loop and exits there — never returning to your
  top-level code (that's *why* fork children don't re-run the driver).
- You rarely branch on parent/child yourself with `Pool`. To identify a process:
  `os.getpid()`, `os.getppid()`, or `multiprocessing.current_process().name`
  (`"MainProcess"` vs `"ForkPoolWorker-N"`).

### Merging per-worker dicts

- `Counter.update(other)` **adds** counts (as does `Counter + Counter` and
  `sum(counters, Counter())`). This is what you want for merging partial counts.
- **Gotcha**: plain `dict.update` **overwrites** shared keys (loses counts). If
  using plain dicts, merge with an explicit `for k, v in d.items(): total[k] += v`
  loop, not `update`.

---

## BPE / Tokenizer

### Chunk boundaries for parallel pre-tokenization

- `find_chunk_boundaries` only **chooses byte offsets**; it never trims or deletes
  anything. Boundaries are used as half-open `[start, end)` ranges.
- It snaps each boundary to the **start** of a full special-token occurrence
  (found via `bytes.find`, which returns the start index of the first match, or
  `-1`). So the preceding chunk excludes the token and the following chunk begins
  with it.
- The guarantee this buys: a chunk break never lands **inside** a special token,
  so no chunk sees a partial token split across a boundary. That's what makes the
  chunks independently processable in parallel.
- Edge cases worth checking: a token straddling a `mini_chunk` read boundary; and
  duplicate boundaries collapsing via `sorted(set(...))`.

### GPT-2 pre-tokenizer pattern (`PAT`)

- It's a big **alternation** (`|`) tried left-to-right at each position:
  - `'(?:[sdmt]|ll|ve|re)` — contractions (`'s`, `'t`, `'ll`, `'ve`, `'re`).
  - ` ?\p{L}+` — optional leading space + letters (a word).
  - ` ?\p{N}+` — optional leading space + digits (a number).
  - ` ?[^\s\p{L}\p{N}]+` — optional leading space + "other" (punctuation/symbols),
    where `[^...]` is a negated class.
  - `\s+(?!\S)` — trailing whitespace run (negative lookahead `(?!\S)` = "not
    followed by a non-space"), leaving the single space that belongs to the next
    word.
  - `\s+` — any remaining whitespace run.
- Building blocks: `\p{L}`/`\p{N}` (Unicode letter/number), `+` (one or more),
  `(?:...)` (non-capturing group), `(?!...)` (negative lookahead), ` ?` (optional
  literal space).

### Leading-space tokens (why ` the` not `the`)

- ` ?\p{L}+` intentionally attaches **one leading space** to the following word.
  So `"the cat"` -> `the`, then `␣cat`. This makes the same word distinguishable
  at line/sentence start (`b'the'`) vs mid-sentence (`b' the'`), and preserves
  whitespace losslessly (concatenating tokens reconstructs the original text).
- Sanity check output looks like: `b'.'`, `b','`, `b' the'`, `b' and'`, `b'\n'`
  at the top of a frequency ranking — a good signal pre-tokenization is correct.

### Special tokens must be held out of merge statistics

- Splitting a chunk on a **capturing** group (`re.split(f"({escaped_token})", s)`)
  keeps the special token as its own element in the result, interleaved with the
  text pieces.
- Do **not** run the pre-tokenizer `PAT` over the special-token elements: `PAT`
  would shred `<|endoftext|>` into fragments (`<`, `|`, `endoftext`, `|>`) and
  count them. For BPE, special tokens should be **atomic** — added to the vocab
  directly and excluded from merge counts so the model never learns to merge
  their fragments. Skip those elements (you know exactly which string they equal).

### Pre-token counts feed BPE training

- The pre-token frequency dict (each pre-token as a byte sequence, with its count)
  is the input to BPE merge training — the trainer works over these frequencies
  rather than rescanning text. Getting this dict correct and fast directly
  benefits the next stage.

---

## PyTorch

*(No lessons yet — add as they come up.)*

---

## Automatic Differentiation

*(No lessons yet — add as they come up.)*

---

## Transformer

*(No lessons yet — add as they come up.)*

---

## Adam / Optimization

*(No lessons yet — add as they come up.)*
