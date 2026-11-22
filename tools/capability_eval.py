#!/usr/bin/env python3
"""Capability A/B with objectively checkable answers, original vs abliterated (adapted from the fngine tools).

  * math: multi-step word problems with exact integer answers (thinking on, low effort)
  * code: Python functions executed against unit tests in a subprocess (thinking off)
  * knowledge: multiple choice (thinking off)

    tools/capability_eval.py --base-url http://127.0.0.1:8003/v1 --label original --out results/cap-original.json
    tools/capability_eval.py --compare results/cap-reference.json results/cap-abliterated.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MATH = [
    ("A shop sells pens at 3 for $4. How many dollars do 27 pens cost?", 36),
    ("A train travels 240 km in 3 hours, then 150 km in 2 hours. What is its average speed in km/h over the whole trip?", 78),
    ("Maria has 5 times as many marbles as Leo. Together they have 84 marbles. How many marbles does Maria have?", 70),
    ("A rectangle has a perimeter of 54 cm and its length is twice its width. What is its area in square cm?", 162),
    ("What is the sum of all integers from 1 to 200 that are divisible by 7?", 2842),
    ("A tank fills at 12 litres per minute and drains at 5 litres per minute. Starting empty, how many minutes until it holds 147 litres?", 21),
    ("Three consecutive even numbers sum to 138. What is the largest of them?", 48),
    ("A jacket costs $80. It is discounted by 25%, then the discounted price is raised by 10%. What is the final price in dollars?", 66),
    ("How many minutes are there between 9:47 and 13:12 on the same day?", 205),
    ("If 6 workers build a wall in 10 days, how many days would 15 workers take at the same rate?", 4),
    ("A number is multiplied by 7, then 15 is subtracted, giving 104. What is the number?", 17),
    ("What is the remainder when 2 to the power 20 is divided by 7?", 4),
    ("A recipe needs 3 eggs per 8 pancakes. How many eggs are needed for 56 pancakes?", 21),
    ("The ages of a father and son sum to 64. In 4 years the father will be three times as old as the son. How old is the son now?", 14),
    ("How many positive divisors does 360 have?", 24),
    ("A car uses 6.5 litres of fuel per 100 km. How many litres does it use for 840 km? Give the answer times 10 as an integer.", 546),
    ("What is the least common multiple of 18, 24 and 30?", 360),
    ("A bag has 4 red, 6 blue and 10 green balls. How many balls must be added, all red, so that red makes up half the bag?", 12),
    ("Compute 1234 * 56 - 7890.", 61214),
    ("A clock shows 3:00. What is the smaller angle, in degrees, between the hour and minute hands at 3:30?", 75),
]

CODE = [
    ("is_palindrome(s)", "Return True if the string s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
     ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('hello')", "assert is_palindrome('')"]),
    ("fib(n)", "Return the n-th Fibonacci number with fib(0)=0 and fib(1)=1, in O(n) time.",
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(30) == 832040"]),
    ("merge_sorted(a, b)", "Merge two sorted lists into one sorted list without using sort().",
     ["assert merge_sorted([1,4,9],[2,3,10]) == [1,2,3,4,9,10]", "assert merge_sorted([],[1]) == [1]"]),
    ("word_freq(text)", "Return a dict mapping each lowercase word to its count; words are separated by whitespace.",
     ["assert word_freq('a b A c b a') == {'a': 3, 'b': 2, 'c': 1}", "assert word_freq('') == {}"]),
    ("flatten(xs)", "Flatten an arbitrarily nested list of integers into a flat list, preserving order.",
     ["assert flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]", "assert flatten([]) == []"]),
    ("primes_below(n)", "Return a list of all primes strictly less than n, ascending.",
     ["assert primes_below(20) == [2,3,5,7,11,13,17,19]", "assert primes_below(2) == []"]),
    ("rle(s)", "Run-length encode a string: 'aaabcc' -> 'a3b1c2'.",
     ["assert rle('aaabcc') == 'a3b1c2'", "assert rle('') == ''", "assert rle('x') == 'x1'"]),
    ("balanced(s)", "Return True if the brackets ()[]{} in s are balanced and properly nested; ignore other characters.",
     ["assert balanced('([]{})')", "assert not balanced('([)]')", "assert balanced('a(b)c')", "assert not balanced('((')"]),
    ("two_sum(nums, target)", "Return indices (i, j) with i < j such that nums[i] + nums[j] == target; assume exactly one answer.",
     ["assert two_sum([2,7,11,15], 9) == (0, 1)", "assert two_sum([3,2,4], 6) == (1, 2)"]),
    ("roman(n)", "Convert an integer 1..3999 to a Roman numeral string.",
     ["assert roman(1994) == 'MCMXCIV'", "assert roman(4) == 'IV'", "assert roman(3999) == 'MMMCMXCIX'"]),
    ("matrix_mult(a, b)", "Multiply two matrices given as lists of lists.",
     ["assert matrix_mult([[1,2],[3,4]],[[5,6],[7,8]]) == [[19,22],[43,50]]"]),
    ("anagrams(words)", "Group words that are anagrams of each other; return a list of groups, each sorted, groups sorted by their first word.",
     ["assert anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]"]),
]

MCQ = [
    ("Which planet has the most known moons as of 2024?", ["Earth", "Mars", "Saturn", "Mercury"], "C"),
    ("What is the time complexity of binary search on a sorted array?", ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "B"),
    ("Which gas makes up most of Earth's atmosphere?", ["Oxygen", "Carbon dioxide", "Argon", "Nitrogen"], "D"),
    ("In TCP, what does the three-way handshake begin with?", ["ACK", "FIN", "SYN", "RST"], "C"),
    ("Who proposed the theory of general relativity?", ["Newton", "Einstein", "Bohr", "Maxwell"], "B"),
    ("Which data structure gives O(1) average-time lookup by key?", ["Linked list", "Hash table", "Binary heap", "Stack"], "B"),
    ("What is the derivative of sin(x)?", ["cos(x)", "-cos(x)", "-sin(x)", "tan(x)"], "A"),
    ("Which language's borrow checker enforces memory safety at compile time?", ["Go", "Rust", "Python", "Java"], "B"),
    ("Which organelle produces most of a cell's ATP?", ["Nucleus", "Ribosome", "Mitochondrion", "Golgi apparatus"], "C"),
    ("What is the chemical symbol for gold?", ["Ag", "Gd", "Au", "Go"], "C"),
    ("Which sorting algorithm has worst-case O(n log n) and is in-place?", ["Heapsort", "Quicksort", "Merge sort", "Counting sort"], "A"),
    ("What is 0.1 + 0.2 == 0.3 in IEEE-754 double precision?", ["True", "False", "An error", "Undefined"], "B"),
]


def chat(base_url, content, thinking, max_tokens, effort=None, timeout=900):
    body = {'model': 'inkling-2-mini', 'stream': False, 'temperature': 0,
            'enable_thinking': thinking, 'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': content}]}
    if thinking and effort:
        body['reasoning_effort'] = effort
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        p = json.loads(r.read())
    msg = p['choices'][0]['message']
    return (msg.get('content') or ''), p.get('usage', {}), time.monotonic() - t


def last_int(text):
    m = re.findall(r'Answer:\s*\$?(-?[\d,]+)', text)
    if not m:
        m = re.findall(r'(-?\d[\d,]*)', text)
    return int(m[-1].replace(',', '')) if m else None


def run_code(src, tests):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 't.py'
        f.write_text(src + '\n\n' + '\n'.join(tests) + '\nprint("OK")\n')
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=20, cwd=d)
        except subprocess.TimeoutExpired:
            return False, 'timeout'
        return r.returncode == 0 and r.stdout.strip().endswith('OK'), (r.stderr or '')[-300:]


def extract_code(text):
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return max(blocks, key=len) if blocks else text


def mcq_letter(text, opts):
    """The first standalone letter A-D (also doubled: 'DD' is one tokenizer token the model sometimes emits),
    else the option whose text appears first; '?' when neither is there."""
    m = re.search(r'\b([ABCD])\1?\b', text)
    if m:
        return m.group(1)
    hits = [(text.lower().find(o.lower()), l) for l, o in zip('ABCD', opts) if o.lower() in text.lower()]
    return min(hits)[1] if hits else '?'


def evaluate(base_url, label):
    out = {'label': label, 'math': [], 'code': [], 'mcq': []}
    for q, ans in MATH:
        text, usage, s = chat(base_url, q + "\nThink it through, then end with a line 'Answer: <integer>'.",
                              True, 4096, effort='low')
        got = last_int(text)
        out['math'].append({'q': q, 'want': ans, 'got': got, 'ok': got == ans, 'tokens': usage.get('completion_tokens'),
                            'seconds': s})
        print(f"math {'ok ' if got == ans else 'BAD'} want {ans} got {got} ({s:.1f}s)", flush=True)
    for sig, spec, tests in CODE:
        text, usage, s = chat(base_url, f"Write a Python function `{sig}`. {spec} Reply with only the code in one "
                                        "```python block.", False, 700)
        ok, err = run_code(extract_code(text), tests)
        out['code'].append({'sig': sig, 'ok': ok, 'error': err if not ok else '', 'seconds': s})
        print(f"code {'ok ' if ok else 'BAD'} {sig} ({s:.1f}s)", flush=True)
    for q, opts, ans in MCQ:
        prompt = q + '\n' + '\n'.join(f"{l}. {o}" for l, o in zip('ABCD', opts)) + \
            '\nAnswer with the letter only.'
        text, usage, s = chat(base_url, prompt, False, 8)
        got = mcq_letter(text, opts)
        out['mcq'].append({'q': q, 'want': ans, 'got': got, 'ok': got == ans, 'text': text})
        print(f"mcq  {'ok ' if got == ans else 'BAD'} want {ans} got {got} {text!r}", flush=True)
    out['summary'] = {k: f"{sum(r['ok'] for r in out[k])}/{len(out[k])}" for k in ('math', 'code', 'mcq')}
    out['summary']['math_tokens_mean'] = sum((r['tokens'] or 0) for r in out['math']) / len(out['math'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:8003/v1')
    ap.add_argument('--label', default='run')
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.loads(Path(p).read_text()) for p in a.compare)
        for k in ('math', 'code', 'mcq'):
            print(f"{k:5s} {x['label']}: {x['summary'][k]:6s} {y['label']}: {y['summary'][k]}")
            for rx, ry in zip(x[k], y[k]):
                if rx['ok'] != ry['ok']:
                    print(f"   differs: {(rx.get('q') or rx.get('sig'))[:70]} -> {x['label']} {rx['ok']}, {y['label']} {ry['ok']}")
        return 0
    res = evaluate(a.base_url, a.label)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res['summary']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
