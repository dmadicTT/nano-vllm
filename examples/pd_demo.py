"""End-to-end PD-disaggregated multi-turn chat with nanovllm + Mooncake.

What this does:
  1. Spawns `mooncake_http_metadata_server` and `mooncake_master` subprocesses.
  2. Spawns two nanovllm worker processes — one in `prefill` role, one in
     `decode` role — both holding their own Qwen3-0.6B model and KV cache.
  3. Issues a sequence of multi-turn chat requests:
       * each turn's prompt is the full conversation so far,
       * the prefill worker runs the forward pass + writes KV blocks to Mooncake,
       * the decode worker pulls those blocks via Mooncake and generates the reply.
  4. Verifies that the replies are coherent and that the Mooncake stats show
     bytes actually traversed the transfer engine.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Resolve the repo root so subprocess spawns can find the nanovllm package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"port {host}:{port} not open within {timeout}s")


def _worker_entry(model: str, role: str, host: str, port: int, engine_kwargs: Dict,
                  log_level: str = "WARNING") -> None:
    """Sub-process entrypoint. Builds the engine and starts a TCP listener."""
    # Re-add the repo root inside the spawned process.
    sys.path.insert(0, str(ROOT))
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f"[{role}] %(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from nanovllm.engine.pd_server import serve
    serve(model=model, role=role, host=host, port=port, **engine_kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    p.add_argument("--master-port", type=int, default=50051)
    p.add_argument("--master-metrics-port", type=int, default=9013,
                   help="Mooncake master admin/metrics port (9003 default often collides)")
    p.add_argument("--meta-port", type=int, default=8081)
    p.add_argument("--prefill-port", type=int, default=18001)
    p.add_argument("--decode-port", type=int, default=18002)
    p.add_argument("--prefill-mc-port", type=int, default=14001,
                   help="prefill node's TransferEngine port (Mooncake)")
    p.add_argument("--decode-mc-port", type=int, default=14002)
    p.add_argument("--protocol", default="tcp", choices=["tcp", "rdma"],
                   help="Mooncake transfer protocol")
    p.add_argument("--rdma-devices", default="",
                   help="RDMA device list (e.g. 'rxe0') — only used when --protocol=rdma")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose-mooncake", action="store_true",
                   help="Enable Mooncake logging at all layers: per-put/get lines from "
                        "KVTransfer (Python INFO), Mooncake C++ glog verbose level, "
                        "TransferEngine metrics, and stream master+metadata-server "
                        "stderr to this process instead of /tmp logs.")
    p.add_argument("--mooncake-glog-v", type=int, default=0,
                   help="GLOG_v level for Mooncake C++ (0=quiet, 1=info, 2=debug). "
                        "Implied >=1 when --verbose-mooncake is set.")
    args = p.parse_args()

    # Apply Mooncake-side environment knobs BEFORE we spawn anything that
    # links to the Mooncake shared object (master / metadata server / workers
    # all inherit this env).
    if args.verbose_mooncake:
        os.environ["MC_TE_METRIC"] = "1"                    # transfer engine periodic metrics
        os.environ["MC_STORE_CLIENT_METRIC_BANDWIDTH"] = "1"  # explicit bandwidth summary
        os.environ.setdefault("GLOG_v", str(max(1, args.mooncake_glog_v)))
    elif args.mooncake_glog_v:
        os.environ["GLOG_v"] = str(args.mooncake_glog_v)
    worker_log_level = "INFO" if args.verbose_mooncake else "WARNING"

    # The mooncake-transfer-engine pip wheel installs these into the venv's
    # bin/ on `pip install`, so they're on PATH inside an activated venv.
    master_bin = shutil.which("mooncake_master")
    meta_bin = shutil.which("mooncake_http_metadata_server")
    if not (master_bin and meta_bin):
        sys.exit("mooncake_master / mooncake_http_metadata_server not on PATH "
                 "(did you `pip install .` or `pip install mooncake-transfer-engine`?)")

    procs = []
    try:
        # 1. metadata server (must be up before any client connects)
        if args.verbose_mooncake:
            meta_stdout, meta_stderr = None, None  # inherit -> printed on this terminal
        else:
            meta_stdout = open("/tmp/pd_demo_meta.log", "wb")
            meta_stderr = subprocess.STDOUT
        meta = subprocess.Popen(
            [meta_bin, f"--port={args.meta_port}"],
            stdout=meta_stdout, stderr=meta_stderr,
            start_new_session=True,  # so we can killpg the whole tree on shutdown
        )
        procs.append(("meta", meta))
        _wait_for_port("127.0.0.1", args.meta_port)
        print(f"[demo] metadata server up on :{args.meta_port}")

        # 2. master
        if args.verbose_mooncake:
            master_stdout, master_stderr = None, None
            master_cmd = [master_bin, f"--port={args.master_port}",
                          f"--metrics_port={args.master_metrics_port}",
                          "--alsologtostderr=true"]
        else:
            master_stdout = open("/tmp/pd_demo_master.log", "wb")
            master_stderr = subprocess.STDOUT
            master_cmd = [master_bin, f"--port={args.master_port}",
                          f"--metrics_port={args.master_metrics_port}"]
        master = subprocess.Popen(
            master_cmd, stdout=master_stdout, stderr=master_stderr,
            start_new_session=True,  # master_bin is a Python wrapper that execs
            # the C++ binary in a child; without our own session, terminating
            # the wrapper leaves that child reparented to init.
        )
        procs.append(("master", master))
        _wait_for_port("127.0.0.1", args.master_port)
        print(f"[demo] master up on :{args.master_port}")

        # 3. prefill + decode workers
        master_addr = f"127.0.0.1:{args.master_port}"
        meta_addr = f"http://127.0.0.1:{args.meta_port}/metadata"
        common_kwargs = dict(
            device="cpu",
            tensor_parallel_size=1,
            enforce_eager=True,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            max_model_len=1024,
            num_kvcache_blocks=16,
            mooncake_master_addr=master_addr,
            mooncake_metadata_server=meta_addr,
            mooncake_protocol=args.protocol,
            mooncake_rdma_devices=args.rdma_devices,
        )
        ctx = mp.get_context("spawn")
        prefill_proc = ctx.Process(
            target=_worker_entry,
            args=(args.model, "prefill", "127.0.0.1", args.prefill_port,
                  {**common_kwargs, "mooncake_local_hostname": f"127.0.0.1:{args.prefill_mc_port}"},
                  worker_log_level),
        )
        decode_proc = ctx.Process(
            target=_worker_entry,
            args=(args.model, "decode", "127.0.0.1", args.decode_port,
                  {**common_kwargs, "mooncake_local_hostname": f"127.0.0.1:{args.decode_mc_port}"},
                  worker_log_level),
        )
        prefill_proc.start()
        decode_proc.start()
        procs.append(("prefill", prefill_proc))
        procs.append(("decode", decode_proc))
        # Workers each take ~30s to load the model on CPU.
        _wait_for_port("127.0.0.1", args.prefill_port, timeout=120)
        _wait_for_port("127.0.0.1", args.decode_port, timeout=120)
        print("[demo] workers up. Building client...")

        from nanovllm.engine.pd_server import PDClient
        prefill = PDClient("127.0.0.1", args.prefill_port)
        decode = PDClient("127.0.0.1", args.decode_port)

        # 4. Drive multi-turn conversations. Two separate conversations alternating.
        import torch
        torch.manual_seed(args.seed)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

        def chat_template(messages):
            return tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )

        conversations = [
            [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "user", "content": "Write a haiku about RDMA."},
            ],
        ]
        # 3 turns of conversation 0, 2 turns of conversation 1, interleaved.
        turns_plan = [
            (0, "What is the capital of France?"),
            (1, "Write a haiku about RDMA."),
            (0, "Now tell me one famous landmark there."),
            (1, "Now explain what RDMA stands for in one sentence."),
            (0, "And what's the country's currency?"),
        ]

        transcripts: List[List[Dict]] = [list(conversations[0]), list(conversations[1])]
        # First turn is in `conversations`; later turns get appended below.
        # Reset turns_plan to start fresh
        transcripts = [[], []]

        for i, (conv_idx, user_msg) in enumerate(turns_plan):
            transcripts[conv_idx].append({"role": "user", "content": user_msg})
            prompt = chat_template(transcripts[conv_idx])
            prompt_ids = tok.encode(prompt)
            rid = f"conv{conv_idx}_turn{len(transcripts[conv_idx])}"
            print(f"\n=== [{rid}] user: {user_msg!r} ({len(prompt_ids)} tokens) ===")
            t0 = time.perf_counter()
            r = prefill.prefill(prompt_ids, rid, args.temperature, args.max_tokens)
            t_prefill = time.perf_counter() - t0
            if not r.get("ok"):
                print("PREFILL FAILED:", r); break
            desc = r["descriptor"]
            print(f"    prefill: {t_prefill:.2f}s  blocks={desc['block_count']}  "
                  f"first_tok={desc['first_token']}")
            t0 = time.perf_counter()
            r = decode.decode(desc)
            t_decode = time.perf_counter() - t0
            if not r.get("ok"):
                print("DECODE FAILED:", r); break
            answer = r["completion_text"].split("<|im_end|>")[0].strip()
            print(f"    decode:  {t_decode:.2f}s  -> {answer!r}")
            transcripts[conv_idx].append({"role": "assistant", "content": answer})

        # Pretty-print final conversations
        print("\n\n" + "=" * 70)
        for i, t in enumerate(transcripts):
            print(f"\n--- Conversation {i} ---")
            for m in t:
                print(f"  [{m['role']:9}] {m['content']!r}")

        # Pull authoritative counters straight from each worker's KVTransfer.
        p_stats = prefill.stats().get("stats", {})
        d_stats = decode.stats().get("stats", {})
        print("\n" + "-" * 70)
        print("Mooncake KV transfer summary (in-process counters):")
        for name, st in [("prefill", p_stats), ("decode", d_stats)]:
            if not st:
                continue
            mb_pushed = st["bytes_pushed"] / (1 << 20)
            mb_pulled = st["bytes_pulled"] / (1 << 20)
            print(f"  [{name}]  pushed: {st['blocks_pushed']} blocks / {mb_pushed:.1f} MiB"
                  f"   pulled: {st['blocks_pulled']} blocks / {mb_pulled:.1f} MiB"
                  f"   ({st['bytes_per_block'] / (1<<20):.1f} MiB/block)")

        # Scrape the master's Prometheus /metrics endpoint. Works regardless
        # of where master output is routed (verbose vs quiet mode).
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.master_metrics_port}/metrics", timeout=2
            ) as r:
                body = r.read().decode("utf-8", errors="replace")
            interesting = {
                "master_put_start_requests_total":  "puts",
                "master_get_replica_list_requests_total": "gets",
                "master_remove_requests_total":     "removes",
                "master_exist_key_requests_total":  "exists",
                "master_key_count":                 "keys (now)",
                "master_active_clients":            "clients",
                "master_allocated_bytes":           "alloc bytes",
            }
            counters = {}
            for line in body.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                name, _, val = line.partition(" ")
                if name in interesting:
                    counters[interesting[name]] = val.strip()
            if counters:
                print(f"  [master /metrics] " + "  ".join(
                    f"{label}={v}" for label, v in counters.items()))
        except Exception as e:
            print(f"  [master /metrics] scrape failed: {e}")

        prefill.shutdown()
        decode.shutdown()
    finally:
        for name, p in reversed(procs):
            try:
                if isinstance(p, subprocess.Popen):
                    # mooncake_master / mooncake_http_metadata_server are
                    # Python wrappers that exec a C++ binary in a child.
                    # We launched them with start_new_session=True so we
                    # can kill the whole process group cleanly here.
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            p.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            pass
                else:
                    # mp.Process worker
                    if hasattr(p, "terminate"):
                        p.terminate()
                    if hasattr(p, "join"):
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=3)
            except Exception:
                pass
        print("[demo] shut down all subprocesses")


if __name__ == "__main__":
    main()
