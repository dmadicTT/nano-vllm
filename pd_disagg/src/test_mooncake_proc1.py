"""Process 1: Sender"""
import sys
import time
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")
from mooncake_kv_transfer import MooncakeKVTransfer

arr = np.zeros((1024 * 1024,), dtype=np.float32)
arr[:] = 42.0

engine = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
port = engine.get_rpc_port()
engine.register_kv_cache(arr)

# Write port to file
with open("/tmp/mooncake_port1.txt", "w") as f:
    f.write(str(port))

print(f"PORT={port}", flush=True)

# Wait for receiver port
receiver_port = None
for _ in range(60):
    try:
        with open("/tmp/mooncake_port2.txt", "r") as f:
            receiver_port = int(f.read().strip())
        break
    except:
        time.sleep(0.5)

if receiver_port is None:
    print("Timeout waiting for receiver", flush=True)
    sys.exit(1)

print(f"Receiver port: {receiver_port}", flush=True)

# Send
engine.send_kv_cache("127.0.0.1", receiver_port, [0], [0], arr.nbytes)
print(f"Send complete. arr[0] = {arr[0]}", flush=True)

engine.unregister_all()
