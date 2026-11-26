#!/usr/bin/env python3
"""Host-side HTTP bridge for gen-image so sandboxed shells can trigger GPU handoff."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_image  # noqa: E402


class ImageHandler(BaseHTTPRequestHandler):
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
            result = gen_image.generate_image(
                prompt=str(req["prompt"]),
                out=str(req["out"]),
                size=req.get("size"),
                steps=req.get("steps"),
                guidance=req.get("guidance"),
                seed=req.get("seed"),
                transparent=bool(req.get("transparent", False)),
            )
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> int:
    host = os.environ.get("INGINE_IMAGE_HOST", "127.0.0.1")
    port = int(os.environ.get("INGINE_IMAGE_PORT", "8091"))
    server = ThreadingHTTPServer((host, port), ImageHandler)
    print(f"blackink-image-server listening on http://{host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
