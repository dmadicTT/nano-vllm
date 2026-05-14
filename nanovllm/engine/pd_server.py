"""PD-disaggregated worker for nanovllm.

Provides:
  * `PDWorker` — holds a single role-bound LLMEngine and serves prefill/decode
    requests over HTTP+JSON.
  * `serve()` — long-running entrypoint (used by the e2e demo).
  * `PDClient` — convenience wrapper for the orchestrator (uses `requests`).

Wire protocol — plain HTTP+JSON:
  POST /prefill   body: {prompt_token_ids, request_id, temperature, max_tokens, ignore_eos}
                  -> {ok, descriptor}      (descriptor goes into the decode request)
  POST /decode    body: {descriptor}
                  -> {ok, completion_token_ids, completion_text, request_id}
  GET  /stats     -> {ok, stats: {...}}
  POST /shutdown  -> {ok: true} then stops the worker

Errors come back as JSON `{"ok": false, "error": "..."}` with an HTTP 4xx/5xx.
"""
from __future__ import annotations

import json
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import requests

from nanovllm import LLM, SamplingParams


class PDWorker:
    """Wraps an LLMEngine in `prefill` or `decode` role."""

    def __init__(self, model: str, role: str, **engine_kwargs):
        assert role in ("prefill", "decode")
        self.role = role
        self.llm = LLM(model=model, role=role, **engine_kwargs)

    # --- request handlers (return a dict; the HTTP layer JSON-encodes it) ---

    def handle_prefill(self, body: Dict[str, Any]) -> Dict[str, Any]:
        sp = SamplingParams(
            temperature=body["temperature"],
            max_tokens=body["max_tokens"],
            ignore_eos=body.get("ignore_eos", False),
        )
        descriptor = self.llm.run_prefill_and_publish(
            prompt=body["prompt_token_ids"],
            sampling_params=sp,
            request_id=body["request_id"],
        )
        return {"ok": True, "descriptor": descriptor}

    def handle_decode(self, body: Dict[str, Any]) -> Dict[str, Any]:
        result = self.llm.run_decode_from_handoff(body["descriptor"])
        return {"ok": True, **result}

    def handle_stats(self) -> Dict[str, Any]:
        tx = self.llm.kv_transport
        if tx is None:
            return {"ok": True, "stats": {}}
        return {"ok": True, "stats": {
            "role": self.role,
            "blocks_pushed": tx.blocks_pushed,
            "blocks_pulled": tx.blocks_pulled,
            "bytes_pushed": tx.bytes_pushed,
            "bytes_pulled": tx.bytes_pulled,
            "bytes_per_block": tx.bytes_per_block,
        }}

    # --- HTTP plumbing ------------------------------------------------------

    def serve(self, host: str, port: int) -> None:
        worker = self

        class Handler(BaseHTTPRequestHandler):
            # Silence the default per-request stderr line; the workers' own
            # logging is more useful and less noisy.
            def log_message(self, fmt, *args):
                return

            def _json(self, status: int, body: Dict[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read_body(self) -> Dict[str, Any]:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n else b""
                return json.loads(raw) if raw else {}

            def do_GET(self):
                if self.path == "/stats":
                    self._json(HTTPStatus.OK, worker.handle_stats())
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"no route {self.path}"})

            def do_POST(self):
                try:
                    body = self._read_body()
                except json.JSONDecodeError as e:
                    self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON: {e}"})
                    return
                if self.path == "/prefill":
                    if worker.role != "prefill":
                        self._json(HTTPStatus.BAD_REQUEST,
                                   {"ok": False, "error": f"this worker has role={worker.role!r}"})
                        return
                    self._serve_or_error(worker.handle_prefill, body)
                elif self.path == "/decode":
                    if worker.role != "decode":
                        self._json(HTTPStatus.BAD_REQUEST,
                                   {"ok": False, "error": f"this worker has role={worker.role!r}"})
                        return
                    self._serve_or_error(worker.handle_decode, body)
                elif self.path == "/shutdown":
                    self._json(HTTPStatus.OK, {"ok": True})
                    # server.shutdown() can't be called from the handler thread
                    # — it blocks waiting for all handler threads to drain,
                    # which includes us. Punt it to another thread.
                    threading.Thread(target=httpd.shutdown, daemon=True).start()
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"no route {self.path}"})

            def _serve_or_error(self, handler, body):
                try:
                    resp = handler(body)
                    self._json(HTTPStatus.OK, resp)
                except Exception as e:
                    traceback.print_exc()
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                               {"ok": False, "error": f"{type(e).__name__}: {e}"})

        httpd = ThreadingHTTPServer((host, port), Handler)
        httpd.daemon_threads = True
        print(f"[PDWorker:{self.role}] listening on http://{host}:{port}", flush=True)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()


class PDClient:
    """Convenience wrapper for an orchestrator to call into prefill/decode workers."""

    def __init__(self, host: str, port: int, *, timeout: float = 300.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(self.base + path, json=body, timeout=self.timeout)
        try:
            return r.json()
        except ValueError:
            r.raise_for_status()
            raise

    def _get(self, path: str) -> Dict[str, Any]:
        r = self.session.get(self.base + path, timeout=self.timeout)
        try:
            return r.json()
        except ValueError:
            r.raise_for_status()
            raise

    def prefill(
        self, prompt_token_ids, request_id, temperature, max_tokens, ignore_eos=False,
    ) -> Dict[str, Any]:
        return self._post("/prefill", {
            "prompt_token_ids": list(prompt_token_ids),
            "request_id": request_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
        })

    def decode(self, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/decode", {"descriptor": descriptor})

    def stats(self) -> Dict[str, Any]:
        return self._get("/stats")

    def shutdown(self) -> Dict[str, Any]:
        try:
            return self._post("/shutdown", {})
        except requests.exceptions.RequestException:
            # The server may close the socket before/while replying — that's
            # the intended shutdown path, not a failure.
            return {"ok": True}


def serve(model: str, role: str, host: str, port: int, **engine_kwargs) -> None:
    """Entry point used by the e2e demo to spawn workers via multiprocessing."""
    worker = PDWorker(model=model, role=role, **engine_kwargs)
    worker.serve(host=host, port=port)
