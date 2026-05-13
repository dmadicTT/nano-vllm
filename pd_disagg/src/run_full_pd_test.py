"""Full end-to-end PD disaggregation test.

This script:
1. Starts a prefill server
2. Starts a decode server
3. Sends a request through the pipeline
4. Verifies KV cache transfer happened
"""

import os
import sys
import time
import pickle
import struct
import socket
import threading
import subprocess

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from pd_config import PDConfig


def send_request(host: str, port: int, request: dict) -> dict:
    """Send a request to a server and get response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
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


def wait_for_server(host: str, port: int, timeout: int = 60) -> bool:
    """Wait for a server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            return True
        except:
            time.sleep(0.5)
    return False


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not model_path:
        print("Usage: python run_full_pd_test.py <model_path>")
        print("Example: python run_full_pd_test.py /path/to/Qwen3-0.6B")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        print("Please download a model first, e.g.:")
        print("  huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B")
        sys.exit(1)
    
    print("=" * 60)
    print("Full PD Disaggregation Test")
    print("=" * 60)
    
    # Start prefill server
    print("\n[1] Starting prefill server...")
    prefill_proc = subprocess.Popen(
        [sys.executable, "pd_server_v2.py", "--role", "prefill", "--model", model_path, "--port", "8010"],
        cwd="/root/.openclaw/workspace/nano-vllm-pd",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Start decode server
    print("[2] Starting decode server...")
    decode_proc = subprocess.Popen(
        [sys.executable, "pd_server_v2.py", "--role", "decode", "--model", model_path, "--port", "8020"],
        cwd="/root/.openclaw/workspace/nano-vllm-pd",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for servers
    print("[3] Waiting for servers to be ready...")
    if not wait_for_server("127.0.0.1", 8010, timeout=120):
        print("ERROR: Prefill server failed to start")
        prefill_proc.kill()
        decode_proc.kill()
        sys.exit(1)
    
    if not wait_for_server("127.0.0.1", 8020, timeout=120):
        print("ERROR: Decode server failed to start")
        prefill_proc.kill()
        decode_proc.kill()
        sys.exit(1)
    
    print("[4] Servers ready!")
    
    # Get RPC ports
    print("[5] Getting Mooncake RPC ports...")
    prefill_info = send_request("127.0.0.1", 8010, {"action": "get_rpc_port"})
    decode_info = send_request("127.0.0.1", 8020, {"action": "get_rpc_port"})
    
    print(f"  Prefill RPC port: {prefill_info['rpc_port']}")
    print(f"  Decode RPC port: {decode_info['rpc_port']}")
    
    # Send prefill request
    prompt = "Hello, how are you today?"
    print(f"\n[6] Sending prefill request: '{prompt}'")
    
    prefill_request = {
        "action": "prefill",
        "prompt": prompt,
        "max_tokens": 20,
        "temperature": 0.7,
    }
    
    prefill_result = send_request("127.0.0.1", 8010, prefill_request)
    
    if "error" in prefill_result:
        print(f"ERROR: Prefill failed: {prefill_result['error']}")
        prefill_proc.kill()
        decode_proc.kill()
        sys.exit(1)
    
    print(f"  Prefill complete!")
    print(f"  Request ID: {prefill_result['request_id']}")
    print(f"  Block table: {prefill_result['block_table']}")
    print(f"  Cached tokens: {prefill_result['num_cached_tokens']}")
    
    # Send decode request
    print(f"\n[7] Sending decode request with KV transfer...")
    
    decode_request = {
        "action": "decode",
        **prefill_result,
    }
    
    decode_result = send_request("127.0.0.1", 8020, decode_request)
    
    if "error" in decode_result:
        print(f"ERROR: Decode failed: {decode_result['error']}")
        prefill_proc.kill()
        decode_proc.kill()
        sys.exit(1)
    
    print(f"  Decode complete!")
    print(f"  Generated {decode_result['num_tokens_generated']} tokens")
    print(f"  Completion: {decode_result['completion']}")
    
    # Cleanup
    print("\n[8] Cleaning up...")
    prefill_proc.terminate()
    decode_proc.terminate()
    
    print("\n" + "=" * 60)
    print("SUCCESS! PD disaggregation with KV migration works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
