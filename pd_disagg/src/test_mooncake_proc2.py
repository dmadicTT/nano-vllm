"""Process 2: Receiver"""
import sys
import time
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")
from mooncake_kv_transfer import MooncakeKVTransfer

arr = np.zeros((1024 * 1024,), dtype=np.float32)

engine = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
port = engine.get_rpc_port()
engine.register_kv_cache(arr)

# Write port to file
with open("/tmp/mooncake_port2.txt", "w") as f:
    f.write(str(port))

print(f"PORT={port}", flush=True)

# Wait for sender port
sender_port = None
for _ in range(60):
    try:
        with open("/tmp/mooncake_port1.txt", "r") as f:
            sender_port = int(f.read().strip())
        break
    except:
        time.sleep(0.5)

if sender_port is None:
    print("Timeout waiting for sender", flush=True)
    sys.exit(1)

print(f"Sender port: {sender_port}", flush=True)

# Receive
engine.receive_kv_cache("127.0.0.1", sender_port, [0], [0], arr.nbytes)
print(f"Receive complete. arr[0] = {arr[0]}", flush=True)

engine.unregister_all()
