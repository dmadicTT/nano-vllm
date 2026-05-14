"""PD-disaggregated worker for nanovllm.

Provides:
  * `PDWorker` — holds a single role-bound LLMEngine and serves prefill/decode
    requests over a small length-prefixed pickle protocol.
  * `serve()` — long-running entrypoint (used by the e2e demo).
  * `PDClient` — convenience wrapper for the orchestrator to call into a worker.

Wire protocol on the worker TCP socket:
  [4-byte big-endian length][pickle-serialized dict]
Each request dict has an `action` key. Supported actions:
  * 'prefill'  — body: {prompt_token_ids, request_id, temperature, max_tokens, ignore_eos}
                  -> {ok, descriptor}      (descriptor goes into the decode request)
  * 'decode'   — body: {descriptor}
                  -> {ok, completion_token_ids, completion_text}
  * 'shutdown' — clean exit
"""
from __future__ import annotations

import pickle
import socket
import struct
import threading
import traceback
from dataclasses import asdict
from typing import Any, Dict, Tuple

from nanovllm import LLM, SamplingParams


def _send(conn: socket.socket, payload: Dict[str, Any]) -> None:
    data = pickle.dumps(payload)
    conn.sendall(struct.pack("!I", len(data)) + data)


def _recv(conn: socket.socket) -> Dict[str, Any]:
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            raise ConnectionError("connection closed before length received")
        hdr += chunk
    (n,) = struct.unpack("!I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf += chunk
    return pickle.loads(buf)


class PDWorker:
    """Wraps an LLMEngine in `prefill` or `decode` role."""

    def __init__(self, model: str, role: str, **engine_kwargs):
        assert role in ("prefill", "decode")
        self.role = role
        self.llm = LLM(model=model, role=role, **engine_kwargs)

    def handle_prefill(self, req: Dict[str, Any]) -> Dict[str, Any]:
        sp = SamplingParams(
            temperature=req["temperature"],
            max_tokens=req["max_tokens"],
            ignore_eos=req.get("ignore_eos", False),
        )
        descriptor = self.llm.run_prefill_and_publish(
            prompt=req["prompt_token_ids"],
            sampling_params=sp,
            request_id=req["request_id"],
        )
        return {"ok": True, "descriptor": descriptor}

    def handle_decode(self, req: Dict[str, Any]) -> Dict[str, Any]:
        result = self.llm.run_decode_from_handoff(req["descriptor"])
        return {"ok": True, **result}

    def handle_stats(self, req: Dict[str, Any]) -> Dict[str, Any]:
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

    def serve(self, host: str, port: int) -> None:
        """Blocking listener. Handles one connection at a time (the orchestrator
        is single-client, so we don't need concurrent connections)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(8)
            print(f"[PDWorker:{self.role}] listening on {host}:{port}", flush=True)
            while True:
                conn, addr = sock.accept()
                try:
                    while True:
                        try:
                            req = _recv(conn)
                        except ConnectionError:
                            break
                        action = req.get("action")
                        if action == "shutdown":
                            _send(conn, {"ok": True})
                            return
                        try:
                            if action == "prefill" and self.role == "prefill":
                                resp = self.handle_prefill(req)
                            elif action == "decode" and self.role == "decode":
                                resp = self.handle_decode(req)
                            elif action == "stats":
                                resp = self.handle_stats(req)
                            else:
                                resp = {"ok": False, "error": f"action {action!r} not supported in role {self.role}"}
                        except Exception as e:
                            traceback.print_exc()
                            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        _send(conn, resp)
                finally:
                    conn.close()


class PDClient:
    """Convenience wrapper for an orchestrator to call into prefill/decode."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def _request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            _send(sock, body)
            return _recv(sock)

    def prefill(
        self, prompt_token_ids, request_id, temperature, max_tokens, ignore_eos=False,
    ) -> Dict[str, Any]:
        return self._request({
            "action": "prefill",
            "prompt_token_ids": list(prompt_token_ids),
            "request_id": request_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
        })

    def decode(self, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        return self._request({"action": "decode", "descriptor": descriptor})

    def stats(self) -> Dict[str, Any]:
        return self._request({"action": "stats"})

    def shutdown(self) -> Dict[str, Any]:
        return self._request({"action": "shutdown"})


def serve(model: str, role: str, host: str, port: int, **engine_kwargs) -> None:
    """Entry point used by the e2e demo to spawn workers via multiprocessing."""
    worker = PDWorker(model=model, role=role, **engine_kwargs)
    worker.serve(host=host, port=port)
