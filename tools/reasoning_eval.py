#!/usr/bin/env python3
"""Reasoning stability, original vs abliterated, through an ingine server: does the reasoning end, stay free of
loops and reach the right answer, at several efforts and seeds, and is the engine deterministic?

  * gsm8k: grade-school word problems (openai/gsm8k test, a fixed random subset), efforts low and high
  * aime: AIME 2024 (HuggingFaceH4/aime_2024), effort high, long reasoning
  * determinism: the same request twice, greedy and sampled with a fixed seed, must return the same text

Sampling at temperature 1 (the model's generation_config) with a seed per sample. A sample counts as
  finished  when the reasoning closed and an answer followed (finish_reason "stop", non-empty content),
  looping   when under half of the word 8-grams in the last 400 words are distinct (a repeated passage),
  correct   when the integer after the last 'Answer:' (else the last integer) equals the reference.

    tools/reasoning_eval.py --base-url http://127.0.0.1:8017/v1 --label abliterated --out results/reasoning-abliterated.json
    tools/reasoning_eval.py --compare results/reasoning-original.json results/reasoning-abliterated.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

INSTRUCTION = "\nThink it through, then end with a line 'Answer: <integer>'."


def chat(base_url, content, effort, max_tokens, temperature=1.0, seed=None, timeout=3600):
    body = {'model': 'inkling-2-mini', 'stream': False, 'temperature': temperature, 'max_tokens': max_tokens,
            'reasoning_effort': effort, 'messages': [{'role': 'user', 'content': content}]}
    if seed is not None:
        body['seed'] = seed
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        p = json.loads(r.read())
    choice = p['choices'][0]
    msg = choice['message']
    return {'content': msg.get('content') or '', 'reasoning': msg.get('reasoning_content') or '',
            'finish': choice.get('finish_reason'), 'tokens': p.get('usage', {}).get('completion_tokens'),
            'seconds': time.monotonic() - t}


def last_int(text):
    m = re.findall(r'Answer:\s*\**\s*\$?(-?[\d,]+)', text)
    if not m:
        m = re.findall(r'(-?\d[\d,]*)', text)
    try:
        return int(m[-1].replace(',', '')) if m else None
    except ValueError:
        return None


def distinct_ratio(text, n=8, window=400):
    words = text.split()[-window:]
    if len(words) < 64:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def sample(base_url, task, idx, question, want, effort, max_tokens, seed):
    r = chat(base_url, question + INSTRUCTION, effort, max_tokens, seed=seed)
    got = last_int(r['content'])
    ratio = min(distinct_ratio(r['reasoning']), distinct_ratio(r['content']))
    rec = {'task': task, 'idx': idx, 'effort': effort, 'seed': seed, 'want': want, 'got': got, 'ok': got == want,
           'finish': r['finish'], 'finished': r['finish'] == 'stop' and bool(r['content'].strip()),
           'loop': ratio < 0.5, 'distinct': round(ratio, 3), 'tokens': r['tokens'],
           'reasoning_chars': len(r['reasoning']), 'seconds': round(r['seconds'], 1)}
    if rec['loop'] or not rec['finished']:
        rec['tail'] = (r['reasoning'] + '\n---\n' + r['content'])[-600:]
    print(f"{task:5s} {effort:5s} #{idx:<4d} {'ok ' if rec['ok'] else 'BAD'} want {want} got {got} "
          f"{r['finish']:6s} {r['tokens']} tok{' LOOP' if rec['loop'] else ''} ({r['seconds']:.0f}s)", flush=True)
    return rec


def determinism(base_url, questions):
    out = []
    for q in questions:
        for temperature, seed in ((0.0, None), (1.0, 1234)):
            a, b = (chat(base_url, q + INSTRUCTION, 'low', 2048, temperature, seed) for _ in range(2))
            same = (a['reasoning'], a['content']) == (b['reasoning'], b['content'])
            out.append({'temperature': temperature, 'seed': seed, 'same': same, 'tokens': [a['tokens'], b['tokens']]})
            print(f"determinism T={temperature} seed={seed}: {'same' if same else 'DIFFERENT'}", flush=True)
    return out


def summarize(samples):
    groups = {}
    for r in samples:
        groups.setdefault(f"{r['task']}/{r['effort']}", []).append(r)
    out = {}
    for key, rs in groups.items():
        toks = sorted(r['tokens'] or 0 for r in rs)
        out[key] = {'n': len(rs), 'correct': sum(r['ok'] for r in rs), 'finished': sum(r['finished'] for r in rs),
                    'looping': sum(r['loop'] for r in rs), 'truncated': sum(r['finish'] == 'length' for r in rs),
                    'tokens_median': statistics.median(toks), 'tokens_p90': toks[int(0.9 * (len(toks) - 1))]}
    return out


def evaluate(a):
    from datasets import load_dataset  # tools/.venv
    gsm = load_dataset('openai/gsm8k', 'main', split='test')
    aime = load_dataset('HuggingFaceH4/aime_2024', split='train')
    picks = random.Random(0).sample(range(len(gsm)), a.gsm8k)
    gsm_items = [(i, gsm[i]['question'], int(gsm[i]['answer'].split('####')[-1].strip().replace(',', '')))
                 for i in picks]
    aime_items = [(i, aime[i]['problem'], int(aime[i]['answer'])) for i in range(min(a.aime, len(aime)))]
    out = {'label': a.label, 'samples': []}
    out['determinism'] = determinism(a.base_url, [q for _, q, _ in gsm_items[:3]])
    for effort in ('low', 'high'):
        for i, q, want in gsm_items:
            out['samples'].append(sample(a.base_url, 'gsm8k', i, q, want, effort, a.max_tokens, seed=1000 + i))
    for seed in range(a.aime_seeds):
        for i, q, want in aime_items:
            out['samples'].append(sample(a.base_url, 'aime', i, q, want, 'high', a.max_tokens, seed=2000 + 97 * seed + i))
    out['summary'] = summarize(out['samples'])
    out['summary']['deterministic'] = f"{sum(d['same'] for d in out['determinism'])}/{len(out['determinism'])}"
    return out


def compare(*runs):
    print(f"{'task/effort':12s} {'run':24s} {'correct':>8s} {'finished':>9s} {'looping':>8s} {'truncated':>10s} "
          f"{'median tokens':>14s}")
    for key in sorted({k for r in runs for k in r['summary'] if '/' in k}):
        for r in runs:
            s = r['summary'].get(key)
            if s:
                print(f"{key:12s} {r['label']:24s} {s['correct']:>4d}/{s['n']:<3d} {s['finished']:>5d}/{s['n']:<3d} "
                      f"{s['looping']:>8d} {s['truncated']:>10d} {s['tokens_median']:>14.0f}")
    for r in runs:
        print(f"deterministic ({r['label']}): {r['summary']['deterministic']} identical repeats")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:8003/v1')
    ap.add_argument('--label', default='run')
    ap.add_argument('--out')
    ap.add_argument('--gsm8k', type=int, default=40, help='GSM8K problems (each at efforts low and high)')
    ap.add_argument('--aime', type=int, default=15, help='AIME 2024 problems (effort high)')
    ap.add_argument('--aime-seeds', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--compare', nargs='+', metavar='RESULT')
    a = ap.parse_args()
    if a.compare:
        compare(*(json.loads(Path(p).read_text()) for p in a.compare))
        return 0
    res = evaluate(a)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res['summary']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
