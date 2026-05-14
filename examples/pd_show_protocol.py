"""Run two turns through the PD stack and dump every request / response on the wire.

Output shows:
  * what the orchestrator sends to the prefill worker,
  * the descriptor the prefill worker returns,
  * what the orchestrator sends to the decode worker,
  * the completion the decode worker returns,
  * KVTransfer counters at the end.

This is useful when you want to see the actual contract between the parts.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _short(v, max_len=80):
    s = repr(v)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _dump_value(v, indent: int):
    pad = " " * indent
    if isinstance(v, dict):
        print("{")
        for k2, v2 in v.items():
            print(f"{pad}  {k2!r}: ", end="")
            _dump_value(v2, indent + 2)
        print(f"{pad}}}", end="")
    elif isinstance(v, list) and len(v) > 6:
        print(f"[{v[0]}, {v[1]}, {v[2]}, ..., {v[-2]}, {v[-1]}]  (list of {len(v)})", end="")
    elif isinstance(v, str) and len(v) > 90:
        print(f"{v[:87]!r}...  (str of {len(v)})", end="")
    else:
        print(repr(v), end="")
    if indent == 4:  # top-level field — terminate the line
        print()


def _dump_dict(label: str, d: dict, indent: int = 2):
    print(f"\n  {label}:")
    pad = " " * (indent + 2)
    for k, v in d.items():
        print(f"{pad}{k!r}: ", end="")
        _dump_value(v, indent + 2)
        print()


def _wait_for_port(host, port, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"{host}:{port} not open within {timeout}s")


def _worker_entry(model, role, host, port, engine_kwargs):
    sys.path.insert(0, str(ROOT))
    from nanovllm.engine.pd_server import serve
    serve(model=model, role=role, host=host, port=port, **engine_kwargs)


def main():
    model = str(ROOT / "models/Qwen3-0.6B")
    master_bin = shutil.which("mooncake_master")
    meta_bin = shutil.which("mooncake_http_metadata_server")
    if not (master_bin and meta_bin):
        sys.exit("mooncake_master / mooncake_http_metadata_server not on PATH "
                 "(did you `pip install .`?)")

    procs = []
    try:
        meta = subprocess.Popen([meta_bin, "--port=8082"],
                                stdout=open("/tmp/proto_meta.log", "wb"),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        procs.append(meta)
        _wait_for_port("127.0.0.1", 8082, timeout=10)
        master = subprocess.Popen([master_bin, "--port=50052", "--metrics_port=9023"],
                                  stdout=open("/tmp/proto_master.log", "wb"),
                                  stderr=subprocess.STDOUT,
                                  start_new_session=True)
        procs.append(master)
        _wait_for_port("127.0.0.1", 50052, timeout=10)
        print("[setup] mooncake_master and metadata server are up")

        common = dict(
            device="cpu", tensor_parallel_size=1, enforce_eager=True,
            max_num_seqs=2, max_num_batched_tokens=512, max_model_len=512,
            num_kvcache_blocks=8,
            mooncake_master_addr="127.0.0.1:50052",
            mooncake_metadata_server="http://127.0.0.1:8082/metadata",
            mooncake_protocol="tcp",
            mooncake_rdma_devices="",
        )
        ctx = mp.get_context("spawn")
        prefill_proc = ctx.Process(
            target=_worker_entry,
            args=(model, "prefill", "127.0.0.1", 18101,
                  {**common, "mooncake_local_hostname": "127.0.0.1:14101"}),
        )
        decode_proc = ctx.Process(
            target=_worker_entry,
            args=(model, "decode", "127.0.0.1", 18102,
                  {**common, "mooncake_local_hostname": "127.0.0.1:14102"}),
        )
        prefill_proc.start(); decode_proc.start()
        procs.append(prefill_proc); procs.append(decode_proc)
        _wait_for_port("127.0.0.1", 18101, timeout=120)
        _wait_for_port("127.0.0.1", 18102, timeout=120)
        print("[setup] prefill + decode workers are up")

        from nanovllm.engine.pd_server import PDClient
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        prefill = PDClient("127.0.0.1", 18101)
        decode = PDClient("127.0.0.1", 18102)

        prompts = [
            "What is the capital of France?",
            "Write a haiku about RDMA.",
        ]
        for i, user_msg in enumerate(prompts):
            print("\n" + "=" * 78)
            print(f"=== TURN {i+1}: {user_msg!r}")
            print("=" * 78)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
            prompt_ids = tok.encode(prompt)
            rid = f"demo_req_{i+1}"

            prefill_req = {
                "prompt_token_ids": prompt_ids,
                "request_id": rid,
                "temperature": 0.7,
                "max_tokens": 30,
                "ignore_eos": False,
            }
            print(f"\n>>> orchestrator → prefill worker  POST http://127.0.0.1:18101/prefill")
            _dump_dict("REQUEST body", prefill_req)
            t0 = time.perf_counter()
            prefill_resp = prefill.prefill(
                prompt_token_ids=prompt_ids, request_id=rid,
                temperature=0.7, max_tokens=30,
            )
            dt_p = time.perf_counter() - t0
            print(f"\n<<< prefill worker → orchestrator   ({dt_p:.2f}s)")
            _dump_dict("RESPONSE", prefill_resp)

            hashes = prefill_resp["descriptor"]["block_hashes"]
            keys_preview = ", ".join(f"nanovllm/kv/{h & ((1<<64)-1):016x}" for h in hashes[:2])
            if len(hashes) > 2:
                keys_preview += f", ... ({len(hashes)} total)"
            print(f"\n  → prefill also `is_exist`-probed and (where missing) pushed "
                  f"{len(hashes)} content-addressed block(s) into Mooncake:")
            print(f"      {keys_preview}")

            decode_req = {"descriptor": prefill_resp["descriptor"]}
            print(f"\n>>> orchestrator → decode worker   POST http://127.0.0.1:18102/decode")
            _dump_dict("REQUEST body", decode_req)
            t0 = time.perf_counter()
            decode_resp = decode.decode(prefill_resp["descriptor"])
            dt_d = time.perf_counter() - t0
            print(f"\n<<< decode worker → orchestrator   ({dt_d:.2f}s)")
            _dump_dict("RESPONSE", decode_resp)

            print(f"\n  → decode first pulled the KV bytes for those keys (28 MiB / block),")
            print(f"    placed them in fresh local block ids, ran the decode loop,")
            print(f"    then called Mooncake remove() on each key to free segment space.")

        print("\n" + "=" * 78)
        print("=== KVTransfer counters (in-process, authoritative)")
        print("=" * 78)
        p_stats = prefill.stats()["stats"]
        d_stats = decode.stats()["stats"]
        _dump_dict("prefill worker stats", p_stats)
        _dump_dict("decode worker stats", d_stats)

        prefill.shutdown(); decode.shutdown()
    finally:
        for p in reversed(procs):
            try:
                if hasattr(p, "terminate"):
                    p.terminate()
                if hasattr(p, "join"):
                    p.join(timeout=5)
                elif isinstance(p, subprocess.Popen):
                    import signal as _signal
                    try:
                        os.killpg(p.pid, _signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, _signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            p.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            pass
                else:
                    p.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
