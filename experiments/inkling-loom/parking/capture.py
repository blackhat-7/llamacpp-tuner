"""A recording proxy: :6891 -> :6890. Saves each /v1/messages request and its SSE reply."""

import http.client
import itertools
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
counter = itertools.count(1)


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _relay(self, method: str) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        n = next(counter)
        if self.path.startswith("/v1/messages") and "count_tokens" not in self.path:
            (OUT / f"{n:04d}-req.json").write_bytes(body or b"")
        upstream = http.client.HTTPConnection("127.0.0.1", 6890, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        upstream.request(method, self.path, body=body, headers=headers)
        response = upstream.getresponse()
        self.send_response(response.status)
        for k, v in response.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.end_headers()
        saved = bytearray()
        for piece in iter(lambda: response.read1(65536), b""):
            saved += piece
            self.wfile.write(piece)
            self.wfile.flush()
        if self.path.startswith("/v1/messages") and "count_tokens" not in self.path:
            (OUT / f"{n:04d}-resp.txt").write_bytes(bytes(saved))
        upstream.close()

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")


ThreadingHTTPServer(("127.0.0.1", 6891), Proxy).serve_forever()
