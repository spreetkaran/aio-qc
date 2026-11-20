# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A fake AI service for the tests: OpenAI-compatible, Gemini and Anthropic endpoints on localhost, answering with
whatever the test puts in `answer` (a function of the request), counting calls and simulating rate limits."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Fake:
    def __init__(self):
        self.calls = []
        self.fail_next = []          # status codes to return first (e.g. [429, 500])
        self.key = "sk-test"
        self.models = ["gemini-2.5-flash", "gpt-4o", "qwen2.5-vl-72b"]
        self.answer = lambda body: {"detections": []}
        self.usage = (1000, 100)
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _auth(self):
                h = self.headers
                k = (h.get("Authorization") or "").replace("Bearer ", "") or h.get("x-api-key") or \
                    h.get("x-goog-api-key") or ""
                return k == srv.key

            def do_GET(self):
                if not self._auth():
                    return self._send(401, {"error": {"message": "bad key"}})
                if self.path.startswith("/v1/models"):
                    return self._send(200, {"data": [{"id": m} for m in srv.models]})
                if self.path.startswith("/v1beta/models"):
                    return self._send(200, {"models": [{"name": f"models/{m}", "supportedGenerationMethods":
                                                        ["generateContent"]} for m in srv.models]})
                self._send(404, {"error": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                srv.calls.append((self.path, body))
                if not self._auth():
                    return self._send(401, {"error": {"message": "bad key"}})
                if srv.fail_next:
                    code = srv.fail_next.pop(0)
                    return self._send(code, {"error": {"message": "slow down"}}, {"Retry-After": "0"})
                text = json.dumps(srv.answer(body))
                i, o = srv.usage
                if self.path.endswith("/chat/completions"):
                    return self._send(200, {"choices": [{"message": {"content": text}}],
                                            "usage": {"prompt_tokens": i, "completion_tokens": o}})
                if ":generateContent" in self.path:
                    return self._send(200, {"candidates": [{"content": {"parts": [{"text": text}]}}],
                                            "usageMetadata": {"promptTokenCount": i, "candidatesTokenCount": o}})
                if self.path.endswith("/v1/messages"):
                    return self._send(200, {"content": [{"type": "text", "text": text}],
                                            "usage": {"input_tokens": i, "output_tokens": o}})
                self._send(404, {"error": "nope"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
