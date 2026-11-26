#!/usr/bin/env python3
"""Host-side HTTP bridge for gen-video so sandboxed shells can trigger GPU handoff."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_video  # noqa: E402


class VideoHandler(BaseHTTPRequestHandler):
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
            result = gen_video.generate_video(
                prompt=str(req["prompt"]),
                out=str(req["out"]),
                init_img=req.get("init_img"),
                end_img=req.get("end_img"),
                size=req.get("size"),
                frames=req.get("frames"),
                fps=req.get("fps"),
                steps=req.get("steps"),
                guidance=req.get("guidance"),
                flow_shift=req.get("flow_shift"),
                seed=req.get("seed"),
                negative_prompt=req.get("negative_prompt"),
                loop=bool(req.get("loop", False)),
            )
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> int:
    host = os.environ.get("INGINE_VIDEO_HOST", "127.0.0.1")
    port = int(os.environ.get("INGINE_VIDEO_PORT", "8092"))
    server = ThreadingHTTPServer((host, port), VideoHandler)
    print(f"blackink-video-server listening on http://{host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
