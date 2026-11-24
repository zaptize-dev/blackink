#!/usr/bin/env node
// Unified reverse proxy for BlackInk:
//   /chat, /chat/*  -> ingine-serve (strips /chat)
//   /metrics, /api/metrics, /v1/*, /props, /slots, /health, /tools, / -> ingine-serve
//   /editor, /editor/* -> dsh web (strips /editor, supports HTTP + WebSocket upgrades)
import http from 'node:http';
import net from 'node:net';

const LISTEN_HOST = process.env.PROXY_HOST || '127.0.0.1';
const LISTEN_PORT = Number(process.env.PROXY_PORT || 8090);
const INGINE_HOST = process.env.INGINE_HOST || '127.0.0.1';
const INGINE_PORT = Number(process.env.INGINE_PORT || 18090);
const DSH_HOST = process.env.DSH_HOST || '127.0.0.1';
const DSH_PORT = Number(process.env.DSH_PORT || 3080);

function resolveTarget(url) {
  if (url === '/editor' || url.startsWith('/editor?') || url.startsWith('/editor/')) {
    let rest = url.slice('/editor'.length) || '/';
    if (!rest.startsWith('/')) rest = '/' + rest;
    return { kind: 'dsh', host: DSH_HOST, port: DSH_PORT, path: rest };
  }
  if (url === '/chat' || url.startsWith('/chat?') || url.startsWith('/chat/')) {
    let rest = url.slice('/chat'.length) || '/';
    if (!rest.startsWith('/')) rest = '/' + rest;
    return { kind: 'ingine', host: INGINE_HOST, port: INGINE_PORT, path: rest };
  }
  return { kind: 'ingine', host: INGINE_HOST, port: INGINE_PORT, path: url || '/' };
}

const server = http.createServer((req, res) => {
  const url = req.url || '/';
  if (url === '/editor') {
    res.writeHead(307, { Location: '/editor/' });
    res.end();
    return;
  }
  if (url === '/chat') {
    res.writeHead(307, { Location: '/chat/' });
    res.end();
    return;
  }

  const target = resolveTarget(url);
  const headers = { ...req.headers };
  if (!headers.host) headers.host = `${LISTEN_HOST}:${LISTEN_PORT}`;

  const proxyReq = http.request(
    {
      hostname: target.host,
      port: target.port,
      path: target.path,
      method: req.method,
      headers,
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
      proxyRes.pipe(res);
    },
  );

  proxyReq.on('error', (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
    }
    res.end(`Bad Gateway (${target.kind}): ${err.message}\n`);
  });

  req.pipe(proxyReq);
});

server.on('upgrade', (req, clientSocket, head) => {
  const target = resolveTarget(req.url || '/');
  const upstream = net.connect(target.port, target.host, () => {
    const rawHeaders = [];
    for (let i = 0; i < req.rawHeaders.length; i += 2) {
      rawHeaders.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
    }
    const reqLine = `${req.method} ${target.path} HTTP/${req.httpVersion}\r\n${rawHeaders.join('\r\n')}\r\n\r\n`;
    upstream.write(reqLine);
    if (head && head.length) upstream.write(head);
    upstream.pipe(clientSocket);
    clientSocket.pipe(upstream);
  });
  upstream.on('error', () => clientSocket.destroy());
  clientSocket.on('error', () => upstream.destroy());
});

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  console.log(`blackink-proxy listening on http://${LISTEN_HOST}:${LISTEN_PORT}`);
});
