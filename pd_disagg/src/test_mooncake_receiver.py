"""Mooncake receiver process."""

import sys
import numpy as np
import time

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")
from mooncake_kv_transfer import MooncakeKVTransfer

# Create array
arr = np.zeros((1024,), dtype=np.float32)

engine = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
port = engine.get_rpc_port()
engine.register_kv_cache(arr)

print(f"RECEIVER_PORT={port}")
print(f"RECEIVER_ADDR={arr.ctypes.data}")

# Wait for sender to send
time.sleep(10)

sender_port = int(sys.argv[1])
success = engine.receive_kv_cache("127.0.0.1", sender_port, [0], [0], arr.nbytes)
print(f"Receive result: {success}")
print(f"arr[0] = {arr[0]}")

engine.unregister_all()
