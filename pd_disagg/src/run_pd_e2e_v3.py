"""End-to-end test for PD disaggregation with Mooncake KV transfer.

This test demonstrates:
1. Prefill server processing a prompt
2. KV cache transfer from prefill to decode
3. Decode server generating completion tokens
"""

import os
import sys
import time
import pickle
import struct
import socket
import subprocess
import signal

print("=" * 60)
print("PD Disaggregation End-to-End Test")
print("=" * 60)

model_path = "/tmp/qwen3-0.6b-full"
prefill_port = 8030
decode_port = 8040

# Use a longer prompt to fill multiple blocks
prompt = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain."""

print(f"\n[1/5] Starting prefill server on port {prefill_port}...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

prefill_proc = subprocess.Popen(
    [
        sys.executable, "-u", "pd_server_v2.py",
        "--role", "prefill",
        "--model", model_path,
        "--port", str(prefill_port),
    ],
    cwd="/root/.openclaw/workspace/nano-vllm-pd",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)

# Wait for prefill server
for _ in range(60):
    line = prefill_proc.stdout.readline()
    if line:
        print(f"  [Prefill] {line.strip()}")
        if "listening on" in line:
            break
    time.sleep(0.5)

print(f"\n[2/5] Starting decode server on port {decode_port}...")
decode_proc = subprocess.Popen(
    [
        sys.executable, "-u", "pd_server_v2.py",
        "--role", "decode",
        "--model", model_path,
        "--port", str(decode_port),
        "--prefill-port", str(prefill_port),
    ],
    cwd="/root/.openclaw/workspace/nano-vllm-pd",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)

# Wait for decode server
for _ in range(60):
    line = decode_proc.stdout.readline()
    if line:
        print(f"  [Decode] {line.strip()}")
        if "listening on" in line:
            break
    time.sleep(0.5)

time.sleep(3)

print(f"\n[3/5] Sending prefill request (prompt length: {len(prompt)} chars)...")

def send_request(host, port, request):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    data = pickle.dumps(request)
    sock.sendall(struct.pack("!I", len(data)) + data)
    
    len_bytes = sock.recv(4)
    msg_len = struct.unpack("!I", len_bytes)[0]
    response_data = b""
    while len(response_data) < msg_len:
        chunk = sock.recv(min(65536, msg_len - len(response_data)))
        if not chunk:
            break
        response_data += chunk
    
    sock.close()
    return pickle.loads(response_data)

prefill_response = send_request("127.0.0.1", prefill_port, {
    "action": "prefill",
    "prompt": prompt,
    "max_tokens": 30,
    "temperature": 0.7,
})

print(f"  Request ID: {prefill_response.get('request_id')}")
print(f"  Prompt tokens: {len(prefill_response.get('prompt_token_ids', []))}")
print(f"  Block table: {prefill_response.get('block_table')}")
print(f"  Cached tokens: {prefill_response.get('num_cached_tokens')}")

if "error" in prefill_response:
    print(f"ERROR: Prefill failed: {prefill_response['error']}")
    prefill_proc.kill()
    decode_proc.kill()
    sys.exit(1)

print(f"\n[4/5] Sending decode request with KV cache transfer...")

decode_response = send_request("127.0.0.1", decode_port, {
    "action": "decode",
    "request_id": prefill_response["request_id"],
    "prompt_token_ids": prefill_response["prompt_token_ids"],
    "block_table": prefill_response["block_table"],
    "num_cached_tokens": prefill_response["num_cached_tokens"],
    "max_tokens": prefill_response["max_tokens"],
    "temperature": prefill_response["temperature"],
    "prefill_rpc_port": prefill_response["prefill_rpc_port"],
    "block_size_bytes": prefill_response["block_size_bytes"],
})

if "error" in decode_response:
    print(f"ERROR: Decode failed: {decode_response['error']}")
else:
    print(f"\n[5/5] SUCCESS!")
    print(f"  Tokens generated: {decode_response.get('num_tokens_generated', 0)}")
    print(f"  Completion: {decode_response.get('completion', 'N/A')[:200]}...")

# Print remaining output
print("\n--- Prefill server output ---")
prefill_proc.send_signal(signal.SIGTERM)
remaining, _ = prefill_proc.communicate(timeout=5)
for line in remaining.splitlines():
    if line.strip() and "PDServerV2" in line:
        print(f"  {line.strip()}")

print("\n--- Decode server output ---")
decode_proc.send_signal(signal.SIGTERM)
remaining, _ = decode_proc.communicate(timeout=5)
for line in remaining.splitlines():
    if line.strip() and "PDServerV2" in line:
        print(f"  {line.strip()}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
