"""End-to-end test for PD disaggregation with Mooncake KV transfer."""

import os
import sys
import time
import pickle
import struct
import socket
import subprocess
import signal
import select

# Start prefill server
print("=" * 60)
print("Starting PD End-to-End Test")
print("=" * 60)

model_path = "/tmp/qwen3-0.6b-full"
prefill_port = 8030
decode_port = 8040

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

# Wait for prefill server to be ready
prefill_ready = False
for _ in range(60):
    line = prefill_proc.stdout.readline()
    if line:
        print(f"  [Prefill] {line.strip()}")
        if "listening on" in line:
            prefill_ready = True
            break
    time.sleep(0.5)

if not prefill_ready:
    print("ERROR: Prefill server failed to start")
    prefill_proc.kill()
    sys.exit(1)

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

# Wait for decode server to be ready
decode_ready = False
for _ in range(60):
    line = decode_proc.stdout.readline()
    if line:
        print(f"  [Decode] {line.strip()}")
        if "listening on" in line:
            decode_ready = True
            break
    time.sleep(0.5)

if not decode_ready:
    print("ERROR: Decode server failed to start")
    prefill_proc.kill()
    decode_proc.kill()
    sys.exit(1)

# Give servers time to fully initialize
time.sleep(3)

print(f"\n[3/5] Sending prefill request...")

def send_request(host, port, request):
    """Send a request to a PD server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    
    data = pickle.dumps(request)
    sock.sendall(struct.pack("!I", len(data)) + data)
    
    # Read response
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

# Send prefill request
prefill_response = send_request("127.0.0.1", prefill_port, {
    "action": "prefill",
    "prompt": "Hello, how are you today?",
    "max_tokens": 50,
    "temperature": 0.7,
})

print(f"  Prefill response: {prefill_response}")

if "error" in prefill_response:
    print(f"ERROR: Prefill failed: {prefill_response['error']}")
    prefill_proc.kill()
    decode_proc.kill()
    sys.exit(1)

print(f"\n[4/5] Sending decode request with KV cache transfer...")

# Send decode request with prefill results
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

print(f"  Decode response keys: {list(decode_response.keys())}")

if "error" in decode_response:
    print(f"ERROR: Decode failed: {decode_response['error']}")
else:
    print(f"\n[5/5] SUCCESS!")
    print(f"  Completion: {decode_response.get('completion', 'N/A')}")
    print(f"  Tokens generated: {decode_response.get('num_tokens_generated', 0)}")

# Print remaining output from both servers
print("\n--- Prefill server output ---")
prefill_proc.send_signal(signal.SIGTERM)
remaining, _ = prefill_proc.communicate(timeout=5)
for line in remaining.splitlines():
    if line.strip():
        print(f"  [Prefill] {line.strip()}")

print("\n--- Decode server output ---")
decode_proc.send_signal(signal.SIGTERM)
remaining, _ = decode_proc.communicate(timeout=5)
for line in remaining.splitlines():
    if line.strip():
        print(f"  [Decode] {line.strip()}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
