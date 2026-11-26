#!/usr/bin/env python3
"""Host-side HTTP bridge for gen-3d so sandboxed shells can trigger GPU handoff."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_3d  # noqa: E402


class Handler3D(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8"))
            result = gen_3d.generate_3d(
                out=str(req["out"]),
                image=req.get("image"),
                prompt=req.get("prompt"),
                steps=req.get("steps"),
                octree_res=req.get("octree_res"),
                max_faces=req.get("max_faces"),
                texture=req.get("texture"),
                seed=req.get("seed"),
            )
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> int:
    host = os.environ.get("INGINE_3D_HOST", "127.0.0.1")
    port = int(os.environ.get("INGINE_3D_PORT", "8093"))
    server = ThreadingHTTPServer((host, port), Handler3D)
    print(f"blackink-3d-server listening on http://{host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
