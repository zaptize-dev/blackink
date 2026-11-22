#!/usr/bin/env python3
"""Logging reverse proxy: record the chat requests a real client (e.g. the coding agent omp) sends to ingine.

Every request is forwarded to the upstream server and its response is streamed back byte for byte. Each
POST /v1/chat/completions additionally leaves one JSON file in the trace directory:

  seq, tag, time (wall clock)        when and in which capture run
  request                            the request body, byte for byte as the client sent it
  response                           the stream reassembled: content, reasoning_content, tool_calls,
                                     finish_reason, usage
  timing                             seconds from the request to the first delta, to the last delta and in total

    tools/trace_proxy.py --trace-dir traces/run1 --tag fix-test
    tools/replay_bench.py --traces traces/run1 --base-url http://127.0.0.1:8017/v1

The usage block is only on the wire when the client asks for it (stream_options.include_usage); --force-usage adds
that request field on the way up (the only change the proxy ever makes) and is off by default.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP_BY_HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers',
              'transfer-encoding', 'upgrade', 'content-length', 'host'}
SECRET_HEADERS = {'authorization', 'x-api-key', 'api-key', 'cookie'}


class Assembler:
    """Reassembles an OpenAI chat completion from its SSE stream (or from the non-streamed JSON body)."""

    def __init__(self, t0):
        self.t0 = t0
        self.buf = b''
        self.content, self.reasoning = [], []
        self.tool_calls = {}
        self.finish_reason = None
        self.usage = None
        self.error = None
        self.t_first = self.t_last = None
        self.chunks = 0

    def feed(self, data, now):
        self.buf += data
        while True:
            m = re.search(rb'\r?\n\r?\n', self.buf)
            if not m:
                return
            event, self.buf = self.buf[:m.start()], self.buf[m.end():]
            payload = b'\n'.join(line[5:].lstrip() for line in event.splitlines() if line.startswith(b'data:'))
            if payload and payload != b'[DONE]':
                try:
                    self.chunk(json.loads(payload), now)
                except ValueError:
                    pass

    def chunk(self, obj, now):
        if obj.get('usage'):
            self.usage = obj['usage']
        if obj.get('error'):
            self.error = obj['error']
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            produced = False
            if delta.get('content'):
                self.content.append(delta['content'])
                produced = True
            if delta.get('reasoning_content'):
                self.reasoning.append(delta['reasoning_content'])
                produced = True
            for tc in delta.get('tool_calls') or []:
                slot = self.tool_calls.setdefault(tc.get('index', 0), {'id': None, 'type': 'function',
                                                                       'function': {'name': '', 'arguments': ''}})
                fn = tc.get('function') or {}
                slot['id'] = tc.get('id') or slot['id']
                slot['function']['name'] += fn.get('name') or ''
                slot['function']['arguments'] += fn.get('arguments') or ''
                produced = True
            if produced:
                self.chunks += 1
                if self.t_first is None:
                    self.t_first = now - self.t0
                self.t_last = now - self.t0
            if choice.get('finish_reason'):
                self.finish_reason = choice['finish_reason']

    def whole(self, body, now):
        """A non-streamed response: one JSON document."""
        try:
            obj = json.loads(body)
        except ValueError:
            return
        self.usage, self.error = obj.get('usage'), obj.get('error')
        for choice in obj.get('choices') or []:
            msg = choice.get('message') or {}
            self.content.append(msg.get('content') or '')
            self.reasoning.append(msg.get('reasoning_content') or '')
            for i, tc in enumerate(msg.get('tool_calls') or []):
                self.tool_calls[i] = tc
            self.finish_reason = choice.get('finish_reason')
        self.t_first = self.t_last = now - self.t0

    def response(self):
        out = {'content': ''.join(self.content), 'reasoning_content': ''.join(self.reasoning),
               'tool_calls': [self.tool_calls[i] for i in sorted(self.tool_calls)],
               'finish_reason': self.finish_reason, 'usage': self.usage}
        if self.error:
            out['error'] = self.error
        return out


class Recorder:
    def __init__(self, trace_dir, tag):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = re.sub(r'[^A-Za-z0-9_.-]+', '-', tag).strip('-') if tag else ''
        self.lock = threading.Lock()
        seen = [int(m.group(1)) for p in self.dir.glob('*.json') if (m := re.match(r'(\d+)', p.name))]
        self.seq = max(seen, default=0)

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def write(self, seq, meta, raw_request):
        """The request body goes into the file byte for byte when it is JSON (it is the value of "request")."""
        name = f"{seq:04d}{'-' + self.tag if self.tag else ''}.json"
        try:
            json.loads(raw_request)
            request = raw_request.decode('utf-8').strip()
        except ValueError:
            request = json.dumps(raw_request.decode('utf-8', 'replace'))
        head = json.dumps(meta['head'], ensure_ascii=False)[:-1]
        tail = json.dumps(meta['tail'], ensure_ascii=False, indent=1)[1:]
        tmp = self.dir / (name + '.tmp')
        tmp.write_text(f'{head}, "request": {request},{tail}\n', encoding='utf-8')
        tmp.rename(self.dir / name)
        return name


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):  # a client hanging up is not news
            super().handle_error(request, client_address)


def make_handler(upstream, recorder, force_usage, quiet):
    target = urllib.parse.urlsplit(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            pass

        def do_any(self):
            t0, wall = time.monotonic(), time.time()
            length = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(length) if length else b''
            path = urllib.parse.urlsplit(self.path).path.rstrip('/')
            traced = self.command == 'POST' and path.endswith('/chat/completions')
            body = raw
            if traced and force_usage:
                try:
                    obj = json.loads(raw)
                    if obj.get('stream'):
                        obj.setdefault('stream_options', {})['include_usage'] = True
                        body = json.dumps(obj).encode()
                except ValueError:
                    pass
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            if body:
                headers['Content-Length'] = str(len(body))
            conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
            try:
                conn.request(self.command, (target.path.rstrip('/') + self.path) if target.path else self.path,
                             body=body or None, headers=headers)
                resp = conn.getresponse()
            except OSError as e:
                conn.close()
                msg = json.dumps({'error': {'message': f'trace_proxy: upstream {upstream} unreachable: {e}',
                                            'type': 'proxy_error'}}).encode()
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            seq = recorder.next_seq() if traced else None
            asm = Assembler(t0)
            streamed = 'text/event-stream' in (resp.getheader('Content-Type') or '')
            fixed = resp.getheader('Content-Length')
            if fixed is None and (self.command == 'HEAD' or resp.status in (204, 304)):
                fixed = '0'
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            if fixed is not None:
                self.send_header('Content-Length', fixed)
            else:
                self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            aborted, whole = False, []
            try:
                while True:
                    data = resp.read1(65536)
                    if not data:
                        break
                    now = time.monotonic()
                    self.wfile.write(data if fixed is not None else b'%x\r\n%s\r\n' % (len(data), data))
                    self.wfile.flush()
                    if traced:
                        asm.feed(data, now) if streamed else whole.append(data)
                if fixed is None:
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
            except OSError:  # the client went away: closing the upstream connection cancels the generation
                aborted = True
                self.close_connection = True
            finally:
                conn.close()
            total = time.monotonic() - t0
            if not traced:
                return
            if not streamed:
                asm.whole(b''.join(whole), time.monotonic())
            head = {'seq': seq, 'tag': recorder.tag,
                    'time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(wall)) + f'{wall % 1:.3f}'[1:],
                    'time_unix': round(wall, 3), 'path': self.path, 'upstream': upstream,
                    'headers': {k: v for k, v in self.headers.items() if k.lower() not in SECRET_HEADERS}}
            tail = {'status': resp.status, 'aborted': aborted, 'response': asm.response(),
                    'timing': {'first_delta_s': asm.t_first and round(asm.t_first, 4),
                               'last_delta_s': asm.t_last and round(asm.t_last, 4),
                               'total_s': round(total, 4), 'delta_chunks': asm.chunks}}
            name = recorder.write(seq, {'head': head, 'tail': tail}, raw)
            if not quiet:
                r, u = tail['response'], asm.usage or {}
                kinds = '+'.join(k for k, v in (('reasoning', r['reasoning_content']), ('text', r['content']),
                                                ('tool', r['tool_calls'])) if v) or 'empty'
                print(f"{name}: status={resp.status} prompt={u.get('prompt_tokens', '?')} "
                      f"completion={u.get('completion_tokens', '?')} finish={r['finish_reason']} {kinds} "
                      f"first={asm.t_first or 0:.2f}s total={total:.2f}s{' ABORTED' if aborted else ''}", flush=True)

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = do_any

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8003, help='port to listen on (default 8003: where omp expects ingine)')
    ap.add_argument('--upstream', default='http://127.0.0.1:8017', help='server the requests are forwarded to')
    ap.add_argument('--trace-dir', default='traces', help='directory for the per-request JSON files')
    ap.add_argument('--tag', default='', help='label appended to the file names of this run (e.g. the scenario)')
    ap.add_argument('--force-usage', action='store_true',
                    help='add stream_options.include_usage to streamed requests on the way up (the recorded request '
                         'stays as the client sent it)')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    recorder = Recorder(a.trace_dir, a.tag)
    server = Server((a.host, a.port), make_handler(a.upstream.rstrip('/'), recorder, a.force_usage, a.quiet))
    print(f'trace_proxy: http://{a.host}:{a.port} -> {a.upstream}, traces in {recorder.dir} (next seq '
          f'{recorder.seq + 1})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
