"""Builds native/src/kernels/draft/draft_shortlist.bin, the token ranking of the drafter's shortlist head.

The drafter scores its base logits over the first K tokens of this ranking (plus the special tokens and the
tokens of the current context) instead of the whole 200,058-token unembedding. The ranking is token frequency
over a corpus of code and English text (no model outputs, so evaluations on captured agent traces are not
biased), then the remaining tokens by id (the BPE merge order). Format: 65,536 little-endian uint32 token ids.

    tools/.venv/bin/python native/tools/build_draft_shortlist.py --tokenizer <model folder>/tokenizer.json

The shipped file was built from 25 MB of code (/usr/lib/python3.14 *.py, /usr/include *.h, Rust/TypeScript/
C++/JavaScript sources under ~/Projects) and 25 MB of English text (/usr/share/doc *.md/*.txt/*.rst/README/
*.html, ~/Projects *.md), shuffled with seed 0, files truncated at 200 KB: 17.8M tokens, 41,390 distinct
(2026-11-21; the corpus files under ~/Projects change over time, so a rebuild gives a slightly different order).
"""
import argparse
import collections
import glob
import os
import random
import struct

from tokenizers import Tokenizer

ROWS = 65536


def collect(patterns, budget):
    files = []
    for p in patterns:
        files += glob.glob(os.path.expanduser(p), recursive=True)
    random.shuffle(files)
    out, n = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if not t.strip():
            continue
        out.append(t[:200000])
        n += len(out[-1])
        if n > budget:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../src/kernels/draft/draft_shortlist.bin"))
    a = ap.parse_args()
    random.seed(0)
    tok = Tokenizer.from_file(a.tokenizer)
    code = collect(["/usr/lib/python3.14/**/*.py", "/usr/include/**/*.h", "~/Projects/**/*.rs", "~/Projects/**/*.ts",
                    "~/Projects/**/*.cpp", "~/Projects/**/*.js"], 25_000_000)
    eng = collect(["/usr/share/doc/**/*.md", "/usr/share/doc/**/*.txt", "/usr/share/doc/**/*.rst",
                   "/usr/share/doc/**/README*", "/usr/share/doc/**/*.html", "~/Projects/**/*.md"], 25_000_000)
    cnt = collections.Counter()
    for batch in (code, eng):
        for e in tok.encode_batch(batch, add_special_tokens=False):
            cnt.update(e.ids)
    ranked = [i for i, _ in cnt.most_common()]
    seen = set(ranked)
    ranked += [i for i in range(tok.get_vocab_size()) if i not in seen]
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{ROWS}I", *ranked[:ROWS]))
    print(f"{sum(cnt.values())} tokens, {len(cnt)} distinct -> {a.out}")


if __name__ == "__main__":
    main()
