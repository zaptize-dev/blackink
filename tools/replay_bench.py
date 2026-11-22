#!/usr/bin/env python3
"""Replay recorded agent requests (tools/trace_proxy.py) against a server and measure what a real client feels.

Each trace file is one POST /v1/chat/completions body of a real agent loop: system prompt, tool schemas, the
files read so far, the conversation. The files are replayed in name order (the trace sets are named so that this
is scenario by scenario, each loop in its captured order), every request with stream=true, timed on the wire:

  ttft s       seconds from sending the request to the first delta that carries output (prefill + first token)
  decode s     last delta time - first delta time; the last delta is the chunk that carries finish_reason, so
               trailing tokens without visible text are inside the interval
  decode tok/s (completion_tokens - 1) / decode s
  prefix       how much of the prompt the server did not have to prefill, inferred from ttft: a prompt that is
               prefilled from scratch takes prompt_tokens / R seconds (R: the cold prefill rate, calibrated on the
               requests of this run that were certainly cold, or --prefill-rate); "full" = the whole prompt was
               prefilled, "reused 93%" = ttft was 7% of that
  model s      ttft + decode: the seconds the model worked on the request

Two orders:
  default      natural order: request N+1 is sent as it was captured, i.e. it contains the assistant turn the
               model produced in the capture, not what the server generates for request N in this replay. After
               replaying request N the server holds prompt N plus its own new output, so prompt N+1 leaves the
               server state where the assistant turn starts: the common prefix is prompt N, never more. A server
               that reuses the longest common prefix prefills only the new turn and tool result; a server that
               reuses only a state that is entirely a prefix of the new prompt prefills everything. (In live use
               such a server sometimes hits, when the re-rendered assistant turn equals the generated tokens;
               a replay cannot reproduce those hits.) One unrelated request is sent before the first one, so a run
               does not depend on what the server did before.
  --cold       an unrelated tiny request before every replayed one: no reuse at all, ttft is the whole prefill.

Totals: token-weighted decode tok/s (sum of tokens / sum of decode seconds), the median over requests, the sums
of ttft, decode and model seconds (the cost of the loops), and the same split by prompt size (<8K, 8-32K,
32-128K, >128K tokens) and by what the output mostly is (tool call, text, reasoning-heavy: at least half of the
output characters are reasoning).

    tools/replay_bench.py --traces traces/run1 --temperature 0 --max-tokens 400 --cold --out a.json
    tools/replay_bench.py --compare a.json b.json

For an A/B use --temperature 0 (greedy: both sides produce the same tokens unless the change alters the model's
output; --compare counts the requests whose output differs).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

BUCKETS = [('<8K', 8_000), ('8-32K', 32_000), ('32-128K', 128_000), ('>128K', float('inf'))]
KINDS = ['tool call', 'text', 'reasoning-heavy']


def bucket_of(prompt_tokens):
    return next(name for name, limit in BUCKETS if prompt_tokens < limit)


def kind_of(reasoning_chars, content_chars, tool_chars, has_tool):
    total = reasoning_chars + content_chars + tool_chars
    if total and 2 * reasoning_chars >= total:
        return 'reasoning-heavy'
    return 'tool call' if has_tool else 'text'


def recorded_kind(trace):
    r = trace.get('response') or {}
    calls = r.get('tool_calls') or []
    tool_chars = sum(len(c['function']['name']) + len(c['function']['arguments']) for c in calls)
    return kind_of(len(r.get('reasoning_content') or ''), len(r.get('content') or ''), tool_chars, bool(calls))


def load_traces(directory, only):
    out = []
    for p in sorted(Path(directory).glob('*.json')):
        if only and not re.search(only, p.name):
            continue
        try:
            t = json.loads(p.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if isinstance(t, dict) and isinstance(t.get('request'), dict) and 'messages' in t['request']:
            out.append((p.name, t))
    return out


def post(base_url, body, timeout):
    u = urllib.parse.urlsplit(base_url.rstrip('/') + '/chat/completions')
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    data = json.dumps(body).encode()
    t0 = time.monotonic()
    conn.request('POST', u.path, body=data, headers={'Content-Type': 'application/json',
                                                     'Accept': 'text/event-stream'})
    return conn, conn.getresponse(), t0


def flush_prefix(base_url, model, nonce, timeout):
    """A tiny unrelated request: afterwards the server holds no prefix of the next replayed prompt."""
    conn, resp, _ = post(base_url, {'model': model, 'stream': False, 'temperature': 0, 'max_tokens': 1,
                                    'messages': [{'role': 'user', 'content': f'ping {nonce}'}]}, timeout)
    resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f'cold-start request failed with HTTP {resp.status}')


def run_request(base_url, body, timeout):
    """Stream one request; returns its timings, token counts and what was generated."""
    conn, resp, t0 = post(base_url, body, timeout)
    if resp.status != 200:
        detail = resp.read().decode('utf-8', 'replace')[:300]
        conn.close()
        return {'error': f'HTTP {resp.status}: {detail}'}
    t_first = t_last = None
    usage, finish, error = None, None, None
    reasoning, content, calls = [], [], {}
    for line in resp:
        now = time.monotonic()
        if not line.startswith(b'data:'):
            continue
        payload = line[5:].strip()
        if payload == b'[DONE]':
            resp.read()
            break
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if obj.get('usage'):
            usage = obj['usage']
        if obj.get('error'):
            error = json.dumps(obj['error'])[:300]
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            produced = False
            if delta.get('content'):
                content.append(delta['content'])
                produced = True
            if delta.get('reasoning_content'):
                reasoning.append(delta['reasoning_content'])
                produced = True
            for tc in delta.get('tool_calls') or []:
                slot = calls.setdefault(tc.get('index', 0), ['', ''])
                fn = tc.get('function') or {}
                slot[0] += fn.get('name') or ''
                slot[1] += fn.get('arguments') or ''
                produced = True
            if produced and t_first is None:
                t_first = now
            if produced or choice.get('finish_reason'):
                t_last = now
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
    conn.close()
    if error or usage is None or t_first is None:
        return {'error': error or ('no usage in the stream' if usage is None else 'no output delta')}
    reasoning, content = ''.join(reasoning), ''.join(content)
    tools = [calls[i] for i in sorted(calls)]
    tool_chars = sum(len(name) + len(args) for name, args in tools)
    tokens = usage['completion_tokens']
    decode_s = t_last - t_first
    return {'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': tokens, 'ttft_s': t_first - t0,
            'decode_s': decode_s, 'decode_tps': (tokens - 1) / decode_s if tokens > 1 and decode_s > 0 else None,
            'total_s': t_last - t0, 'finish_reason': finish,
            'kind': kind_of(len(reasoning), len(content), tool_chars, bool(tools)),
            'chars': {'reasoning': len(reasoning), 'content': len(content), 'tool': tool_chars},
            'output_sha': hashlib.sha256(json.dumps([reasoning, content, tools]).encode()).hexdigest()[:16],
            'output': {'reasoning_content': reasoning, 'content': content,
                       'tool_calls': [{'name': name, 'arguments': args} for name, args in tools]}}


def prepare(request, a):
    body = dict(request)
    body['stream'] = True
    body['stream_options'] = {**(body.get('stream_options') or {}), 'include_usage': True}
    if a.temperature is not None:
        body['temperature'] = a.temperature
    if a.seed is not None:
        body['seed'] = a.seed
    if a.top_p is not None:
        body['top_p'] = a.top_p
    if a.max_tokens is not None:
        fields = [f for f in ('max_tokens', 'max_completion_tokens') if body.get(f) is not None] or ['max_tokens']
        for f in fields:
            body[f] = min(body.get(f) or a.max_tokens, a.max_tokens)
    return body


def median_run(runs):
    """The repeat with the median decode rate stands for the request; ttft is the median over repeats."""
    rated = sorted((r for r in runs if r['decode_tps'] is not None), key=lambda r: r['decode_tps'])
    rep = dict(rated[(len(rated) - 1) // 2] if rated else runs[0])
    rep['ttft_s'] = statistics.median(r['ttft_s'] for r in runs)
    rep['total_s'] = rep['ttft_s'] + rep['decode_s']
    return rep


def cold_rate(rows, certainly_cold):
    """Cold prefill rate in tokens/s: over the requests known to be cold and those as slow per token as they."""
    known = [r for r in rows if r['name'] in certainly_cold and 'error' not in r]
    if not known:
        return None
    base = sum(r['prompt_tokens'] for r in known) / sum(r['ttft_s'] for r in known)
    cold = [r for r in rows if 'error' not in r and r['prompt_tokens'] / r['ttft_s'] < 1.5 * base]
    return sum(r['prompt_tokens'] for r in cold) / sum(r['ttft_s'] for r in cold)


def mark_reuse(rows, rate):
    for r in rows:
        if 'error' in r or not rate:
            continue
        frac = min(1.0, max(0.0, 1.0 - r['ttft_s'] * rate / r['prompt_tokens']))
        r['prefix_reused_frac'] = round(frac, 3)
        r['prefix_reused'] = frac >= 0.5
        r['prefill_rate_assumed'] = round(rate, 1)


def summarize(rows):
    """rows: per-request results (without errors) -> totals of one group."""
    rated = [r for r in rows if r.get('decode_tps')]
    tokens = sum(r['completion_tokens'] - 1 for r in rated)
    seconds = sum(r['decode_s'] for r in rated)
    return {'requests': len(rows), 'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
            'completion_tokens': sum(r['completion_tokens'] for r in rows),
            'decode_tps_weighted': tokens / seconds if seconds > 0 else None,
            'decode_tps_median': statistics.median(r['decode_tps'] for r in rated) if rated else None,
            'ttft_s_median': statistics.median(r['ttft_s'] for r in rows) if rows else None,
            'ttft_s_sum': sum(r['ttft_s'] for r in rows), 'decode_s_sum': sum(r['decode_s'] for r in rows),
            'model_s_sum': sum(r['total_s'] for r in rows),
            'prefix_reused_requests': sum(bool(r.get('prefix_reused')) for r in rows)}


def totals_of(rows):
    ok = [r for r in rows if 'error' not in r]
    return {'all': summarize(ok),
            'by_bucket': {b: summarize([r for r in ok if r['bucket'] == b]) for b, _ in BUCKETS},
            'by_kind': {k: summarize([r for r in ok if r['kind'] == k]) for k in KINDS},
            'by_scenario': {s: summarize([r for r in ok if r.get('scenario') == s])
                            for s in dict.fromkeys(r.get('scenario') for r in ok) if s}}


def num(x, spec):
    return format(x, spec) if x is not None else '-'


def prefix_label(r):
    frac = r.get('prefix_reused_frac')
    if frac is None:
        return '-'
    return f'reused {100 * frac:.0f}%' if frac >= 0.2 else 'full'


REQUEST_HEADER = (f"{'request':34s} {'prompt':>7s} {'compl':>6s} {'ttft s':>7s} {'decode s':>8s} {'dec tok/s':>9s} "
                  f"{'model s':>8s}  {'prefix':11s} {'output kind':15s} finish")


def print_request(r, live=False):
    if 'error' in r:
        print(f"{r['name'][:34]:34s} ERROR {r['error']}", flush=True)
        return
    prefix = f"{r['prompt_tokens'] / r['ttft_s']:.0f} t/s" if live else prefix_label(r)
    print(f"{r['name'][:34]:34s} {r['prompt_tokens']:7d} {r['completion_tokens']:6d} {r['ttft_s']:7.2f} "
          f"{r['decode_s']:8.2f} {num(r['decode_tps'], '9.1f'):>9s} {r['total_s']:8.2f}  {prefix:11s} "
          f"{r['kind']:15s} {r['finish_reason'] or '-'}", flush=True)


def print_totals(totals):
    print(f"\n{'group':18s} {'n':>3s} {'prompt':>8s} {'compl':>7s} {'weighted':>9s} {'median':>8s} {'ttft med':>9s} "
          f"{'ttft sum':>9s} {'decode sum':>10s} {'model sum':>9s} {'prefix':>7s}")
    print(f"{'':18s} {'':3s} {'tokens':>8s} {'tokens':>7s} {'tok/s':>9s} {'tok/s':>8s} {'s':>9s} {'s':>9s} "
          f"{'s':>10s} {'s':>9s} {'reused':>7s}")
    groups = [('all', totals['all'])] + [('prompt ' + b, totals['by_bucket'][b]) for b, _ in BUCKETS] + \
             [(k, totals['by_kind'][k]) for k in KINDS] + \
             [(s[:18], t) for s, t in totals.get('by_scenario', {}).items()]
    for name, s in groups:
        if not s['requests']:
            print(f"{name:18s} {0:3d}")
            continue
        print(f"{name:18s} {s['requests']:3d} {s['prompt_tokens']:8d} {s['completion_tokens']:7d} "
              f"{num(s['decode_tps_weighted'], '9.1f'):>9s} {num(s['decode_tps_median'], '8.1f'):>8s} "
              f"{s['ttft_s_median']:9.2f} {s['ttft_s_sum']:9.1f} {s['decode_s_sum']:10.1f} {s['model_s_sum']:9.1f} "
              f"{s['prefix_reused_requests']:3d}/{s['requests']:<3d}")


def replay(a):
    traces = load_traces(a.traces, a.only)
    if not traces:
        print(f'no trace files match in {a.traces}', file=sys.stderr)
        return 2
    runs = {name: [] for name, _ in traces}
    errors = {}
    print(f"replaying {len(traces)} requests x {a.repeat} against {a.base_url} "
          f"({'cold: unrelated request before each' if a.cold else 'natural order, as captured'}); "
          f"live column 'prefix' = prompt tokens / ttft")
    print(REQUEST_HEADER, flush=True)
    for rep in range(a.repeat):
        for i, (name, trace) in enumerate(traces):
            body = prepare(trace['request'], a)
            try:
                if a.cold or (rep == 0 and i == 0):
                    flush_prefix(a.base_url, body.get('model', 'inkling-2-mini'), f'{rep}-{i}-{time.time_ns()}',
                                 a.timeout)
                r = run_request(a.base_url, body, a.timeout)
            except (OSError, RuntimeError, http.client.HTTPException) as e:
                r = {'error': f'{type(e).__name__}: {e}'}
            if 'error' in r:
                errors[name] = r['error']
                print_request({'name': name, **r})
                continue
            if not a.save_text:
                del r['output']
            runs[name].append(r)
            print_request({'name': f'[{rep + 1}/{a.repeat}] {name}' if a.repeat > 1 else name, **r}, live=True)
    rows = []
    for name, trace in traces:
        if name in errors or not runs[name]:
            rows.append({'name': name, 'error': errors.get(name, 'no run')})
            continue
        row = {'name': name, **median_run(runs[name])}
        row['bucket'] = bucket_of(row['prompt_tokens'])
        row['scenario'] = (trace.get('capture') or {}).get('scenario') or trace.get('tag') or ''
        row['recorded_kind'] = recorded_kind(trace)
        row['recorded_completion_tokens'] = ((trace.get('response') or {}).get('usage') or {}).get('completion_tokens')
        if a.repeat > 1:
            row['runs'] = runs[name]
        rows.append(row)
    # With repeats the first request of a later pass follows the last request of the previous pass: still unrelated.
    rate = a.prefill_rate or cold_rate(rows, {n for n, _ in traces} if a.cold else {traces[0][0]})
    mark_reuse(rows, rate)
    print(f"\n{'median of ' + str(a.repeat) + ' repeats; ' if a.repeat > 1 else ''}prefix reuse inferred with a cold "
          f"prefill rate of {num(rate, '.0f')} tokens/s ({'--prefill-rate' if a.prefill_rate else 'calibrated'})")
    print(REQUEST_HEADER)
    for r in rows:
        print_request(r)
    totals = totals_of(rows)
    print_totals(totals)
    if a.out:
        result = {'label': a.label or Path(a.out).stem, 'base_url': a.base_url, 'traces': str(a.traces),
                  'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'options': {'temperature': a.temperature, 'top_p': a.top_p, 'seed': a.seed,
                              'max_tokens': a.max_tokens, 'only': a.only, 'repeat': a.repeat, 'cold': a.cold},
                  'requests': rows, 'totals': totals}
        if a.append and Path(a.out).exists():
            old = json.loads(Path(a.out).read_text())
            keep = [r for r in old['requests'] if r['name'] not in {x['name'] for x in rows}]
            result['requests'] = sorted(keep + rows, key=lambda r: r['name'])
            result['totals'] = totals_of(result['requests'])
            result['options']['only'] = None
            print(f"\n{a.out}: {len(result['requests'])} requests after merging")
            print_totals(result['totals'])
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
    return 1 if errors else 0


def show(path):
    result = json.loads(Path(path).read_text())
    opts = result['options']
    print(f"{result['label']}: {result['base_url']}, {result['traces']}, "
          f"{'cold' if opts['cold'] else 'natural order'}, temperature {opts['temperature']}, "
          f"max_tokens {opts['max_tokens']}, repeat {opts['repeat']}")
    print(REQUEST_HEADER)
    for r in result['requests']:
        print_request(r)
    print_totals(result['totals'])
    return 0


def ratio(b, a):
    return b / a if a and b else None


def compare(path_a, path_b):
    A, B = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    la, lb = A['label'], B['label']
    if la == lb:
        la, lb = 'A', 'B'
    bx = {r['name']: r for r in B['requests'] if 'error' not in r}
    pairs = [(r, bx[r['name']]) for r in A['requests'] if 'error' not in r and r['name'] in bx]
    if not pairs:
        print('the two result files have no request in common', file=sys.stderr)
        return 2
    print(f'A = {la} ({path_a})\nB = {lb} ({path_b})\n'
          f'x = speedup of B over A (decode: tok/s B / tok/s A; ttft and model seconds: A / B)\n')
    print(f"{'request':34s} {'prompt':>7s} {'compl A':>7s} {'compl B':>7s} {'tok/s A':>8s} {'tok/s B':>8s} "
          f"{'x':>6s} {'ttft A':>7s} {'ttft B':>7s} {'x':>6s} {'model A':>7s} {'model B':>7s} {'x':>6s}  "
          f"{'prefix A':11s} {'prefix B':11s} output")
    same = 0
    for x, y in pairs:
        identical = x['output_sha'] == y['output_sha']
        same += identical
        print(f"{x['name'][:34]:34s} {x['prompt_tokens']:7d} {x['completion_tokens']:7d} {y['completion_tokens']:7d} "
              f"{num(x['decode_tps'], '8.1f'):>8s} {num(y['decode_tps'], '8.1f'):>8s} "
              f"{num(ratio(y['decode_tps'], x['decode_tps']), '6.3f'):>6s} {x['ttft_s']:7.2f} {y['ttft_s']:7.2f} "
              f"{num(ratio(x['ttft_s'], y['ttft_s']), '6.2f'):>6s} {x['total_s']:7.2f} {y['total_s']:7.2f} "
              f"{num(ratio(x['total_s'], y['total_s']), '6.2f'):>6s}  {prefix_label(x):11s} {prefix_label(y):11s} "
              f"{'same' if identical else 'DIFFERS'}")
    print(f'\noutputs identical: {same}/{len(pairs)}'
          + ('' if same == len(pairs) else '  (where they differ the speedup also reflects different text)'))
    print(f"\n{'group':18s} {'n':>3s} {'weighted tok/s':>19s} {'x':>6s} {'median tok/s':>17s} {'x':>6s} "
          f"{'ttft sum s':>15s} {'x':>6s} {'decode sum s':>15s} {'x':>6s} {'model sum s':>15s} {'x':>6s} "
          f"{'prefix reused':>13s}")
    print(f"{'':18s} {'':3s} {'A':>9s} {'B':>9s} {'':6s} {'A':>8s} {'B':>8s} {'':6s} {'A':>7s} {'B':>7s} {'':6s} "
          f"{'A':>7s} {'B':>7s} {'':6s} {'A':>7s} {'B':>7s} {'':6s} {'A':>6s} {'B':>6s}")
    groups = [('all', lambda r: True)] + [('prompt ' + b, lambda r, b=b: r['bucket'] == b) for b, _ in BUCKETS] + \
             [(k + ' (A)', lambda r, k=k: r['kind'] == k) for k in KINDS] + \
             [(s[:18], lambda r, s=s: r.get('scenario') == s)
              for s in dict.fromkeys(x.get('scenario') for x, _ in pairs) if s]
    for name, pred in groups:
        sel = [(x, y) for x, y in pairs if pred(x)]
        if not sel:
            continue
        sa, sb = summarize([x for x, _ in sel]), summarize([y for _, y in sel])
        print(f"{name:18s} {len(sel):3d} {num(sa['decode_tps_weighted'], '9.1f'):>9s} "
              f"{num(sb['decode_tps_weighted'], '9.1f'):>9s} "
              f"{num(ratio(sb['decode_tps_weighted'], sa['decode_tps_weighted']), '6.3f'):>6s} "
              f"{num(sa['decode_tps_median'], '8.1f'):>8s} {num(sb['decode_tps_median'], '8.1f'):>8s} "
              f"{num(ratio(sb['decode_tps_median'], sa['decode_tps_median']), '6.3f'):>6s} "
              f"{sa['ttft_s_sum']:7.1f} {sb['ttft_s_sum']:7.1f} "
              f"{num(ratio(sa['ttft_s_sum'], sb['ttft_s_sum']), '6.2f'):>6s} "
              f"{sa['decode_s_sum']:7.1f} {sb['decode_s_sum']:7.1f} "
              f"{num(ratio(sa['decode_s_sum'], sb['decode_s_sum']), '6.2f'):>6s} "
              f"{sa['model_s_sum']:7.1f} {sb['model_s_sum']:7.1f} "
              f"{num(ratio(sa['model_s_sum'], sb['model_s_sum']), '6.2f'):>6s} "
              f"{sa['prefix_reused_requests']:6d} {sb['prefix_reused_requests']:6d}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', help='directory of trace files written by tools/trace_proxy.py')
    ap.add_argument('--base-url', default='http://127.0.0.1:8003/v1')
    ap.add_argument('--temperature', type=float, help='override the recorded temperature (0: greedy, for A/B)')
    ap.add_argument('--top-p', type=float, help='override the recorded top_p')
    ap.add_argument('--seed', type=int, help='override the recorded seed')
    ap.add_argument('--max-tokens', type=int, help='cap the completion length of every request')
    ap.add_argument('--only', metavar='PATTERN', help='replay only trace files whose name matches this regex')
    ap.add_argument('--repeat', type=int, default=1, help='passes over the trace set; the report is the median')
    ap.add_argument('--cold', action='store_true',
                    help='send a tiny unrelated request before each replayed one, so the server cannot reuse a '
                         'prompt prefix and ttft includes the whole prefill')
    ap.add_argument('--prefill-rate', type=float, metavar='TOKENS_PER_S',
                    help='cold prefill rate for the prefix-reuse inference (default: calibrated in the run)')
    ap.add_argument('--out', help='write the results as JSON')
    ap.add_argument('--append', action='store_true',
                    help='merge into an existing --out file (replay in chunks with --only, one file at the end)')
    ap.add_argument('--save-text', action='store_true', help='keep the generated text of every request in --out')
    ap.add_argument('--label', help='name of this run in --compare (default: the --out file name)')
    ap.add_argument('--timeout', type=float, default=1800, help='seconds per request')
    ap.add_argument('--compare', nargs=2, metavar=('A.json', 'B.json'),
                    help='print per-request and total speedups of B over A and exit')
    ap.add_argument('--show', metavar='RESULTS.json', help='print the tables of a results file and exit')
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.show:
        return show(a.show)
    if not a.traces:
        ap.error('--traces DIR is required to replay')
    return replay(a)


if __name__ == '__main__':
    sys.exit(main())
