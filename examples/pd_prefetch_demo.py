"""Show off the cross-node KV reuse the content-addressed Mooncake cache enables.

Two prefill workers, same 300-token prompt issued first to one then to the
other (different request IDs, different processes). With block_size=256
the prompt is one full block + one partial — the full block is eligible
for `_prefetch_from_store`.

Expected output (counters from each worker's /stats):
  node 0   request #1   pushed=2 skipped-put=0 pulled=0 prefetched=0
  node 1   request #2   pushed=0 skipped-put=2 pulled=1 prefetched=1

i.e. node 1 *pulled* the prompt's full block from Mooncake (the store-tier
prefix cache) instead of recomputing it locally, and its push step then
skipped both blocks because the keys were already present.

This stresses the cross-prefill-node coordination, which is exactly the
case the cluster of multiple prefill workers in `pd_demo.py --num-prefill N`
is built to exploit.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _worker_entry(role: str, port: int, mc_port: int, mooncake_master_port: int,
                  meta_port: int, model: str) -> None:
    sys.path.insert(0, str(ROOT))
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{role}@{port}] %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from nanovllm.engine.pd_server import serve
    serve(
        model=model,
        role=role, host="127.0.0.1", port=port,
        device="cpu", tensor_parallel_size=1, enforce_eager=True,
        max_num_seqs=2, max_num_batched_tokens=1024, max_model_len=1024,
        num_kvcache_blocks=8,
        mooncake_master_addr=f"127.0.0.1:{mooncake_master_port}",
        mooncake_metadata_server=f"http://127.0.0.1:{meta_port}/metadata",
        mooncake_protocol="tcp",
        mooncake_local_hostname=f"127.0.0.1:{mc_port}",
    )


def _wait_port(p: int, t: int = 120) -> None:
    for _ in range(t):
        try:
            socket.create_connection(("127.0.0.1", p), timeout=0.5).close()
            return
        except OSError:
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    ap.add_argument("--master-port", type=int, default=50058)
    ap.add_argument("--meta-port", type=int, default=8088)
    args = ap.parse_args()

    master_bin = shutil.which("mooncake_master")
    meta_bin = shutil.which("mooncake_http_metadata_server")
    if not (master_bin and meta_bin):
        sys.exit("mooncake_master / mooncake_http_metadata_server not on PATH")

    meta = subprocess.Popen([meta_bin, f"--port={args.meta_port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)
    master = subprocess.Popen([master_bin, f"--port={args.master_port}",
                               f"--metrics_port={args.master_port + 1000}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                              start_new_session=True)
    time.sleep(2)

    ctx = mp.get_context("spawn")
    procs = []
    try:
        p0 = ctx.Process(target=_worker_entry,
                         args=("prefill", 19001, 14911, args.master_port, args.meta_port,
                               args.model))
        p1 = ctx.Process(target=_worker_entry,
                         args=("prefill", 19002, 14912, args.master_port, args.meta_port,
                               args.model))
        p0.start(); p1.start()
        procs.extend([p0, p1])
        _wait_port(19001); _wait_port(19002)

        from nanovllm.engine.pd_server import PDClient
        node0 = PDClient("127.0.0.1", 19001)
        node1 = PDClient("127.0.0.1", 19002)

        # 300 tokens = 1 full block (block_size=256) + 1 partial.
        prompt = list(range(300))
        print(f"prompt: {len(prompt)} arbitrary tokens (one full block + 44-token tail)\n")
        for i, node in enumerate((node0, node1)):
            t0 = time.perf_counter()
            r = node.prefill(prompt, f"req{i}", 0.7, 2)
            dt = time.perf_counter() - t0
            assert r["ok"], r
            s = node.stats()["stats"]
            print(f"node{i} after its request ({dt:.2f}s):  "
                  f"pushed={s['blocks_pushed']}  skipped-put={s['blocks_skipped_push']}  "
                  f"pulled={s['blocks_pulled']}  prefetched={s['blocks_prefetched']}")
        print()
        print("Expected: node 0 pushed=2/skipped=0, node 1 pushed=0/skipped=2/prefetched=1.")
        node0.shutdown(); node1.shutdown()
    finally:
        for p in procs:
            try:
                if p.is_alive(): p.terminate(); p.join(timeout=5)
                if p.is_alive(): p.kill(); p.join(timeout=3)
            except Exception: pass
        for proc in (master, meta):
            try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=3)
            except Exception:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except Exception: pass


if __name__ == "__main__":
    main()
