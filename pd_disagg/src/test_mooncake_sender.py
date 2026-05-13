"""Mooncake sender process."""

import sys
import numpy as np
import time

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")
from mooncake_kv_transfer import MooncakeKVTransfer

# Create array
arr = np.zeros((1024,), dtype=np.float32)
arr[:] = 42.0

engine = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
port = engine.get_rpc_port()
engine.register_kv_cache(arr)

print(f"SENDER_PORT={port}")
print(f"SENDER_ADDR={arr.ctypes.data}")

# Wait for receiver to be ready
time.sleep(5)

receiver_port = int(sys.argv[1])
success = engine.send_kv_cache("127.0.0.1", receiver_port, [0], [0], arr.nbytes)
print(f"Send result: {success}")

engine.unregister_all()
