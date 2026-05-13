"""Simple client to test PD disaggregation."""

import pickle
import struct
import socket
import sys


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


def main():
    # Step 1: Get RPC ports
    print("[Client] Getting RPC ports...")
    prefill_info = send_request("127.0.0.1", 8010, {"action": "get_rpc_port"})
    decode_info = send_request("127.0.0.1", 8020, {"action": "get_rpc_port"})
    print(f"  Prefill RPC: {prefill_info['rpc_port']}")
    print(f"  Decode RPC: {decode_info['rpc_port']}")
    
    # Step 2: Send prefill request
    prompt = "Hello, how are you today?"
    print(f"\n[Client] Sending prefill request: '{prompt}'")
    
    prefill_request = {
        "action": "prefill",
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.7,
    }
    
    prefill_result = send_request("127.0.0.1", 8010, prefill_request)
    
    if "error" in prefill_result:
        print(f"ERROR: Prefill failed: {prefill_result['error']}")
        sys.exit(1)
    
    print(f"  Prefill complete!")
    print(f"  Request ID: {prefill_result['request_id']}")
    print(f"  Block table: {prefill_result['block_table']}")
    print(f"  Cached tokens: {prefill_result['num_cached_tokens']}")
    
    # Step 3: Send decode request
    print(f"\n[Client] Sending decode request with KV transfer...")
    
    decode_request = {
        "action": "decode",
        **prefill_result,
    }
    
    decode_result = send_request("127.0.0.1", 8020, decode_request)
    
    if "error" in decode_result:
        print(f"ERROR: Decode failed: {decode_result['error']}")
        sys.exit(1)
    
    print(f"  Decode complete!")
    print(f"  Generated {decode_result['num_tokens_generated']} tokens")
    print(f"  Completion: {decode_result['completion']}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! PD disaggregation with KV migration works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
