"""Simple test of Mooncake transfer between two processes."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from mooncake_kv_transfer import MooncakeKVTransfer


def test_transfer():
    """Test transfer with two engines in same process (sequentially)."""
    
    # Create arrays
    arr1 = np.zeros((1024 * 1024,), dtype=np.float32)  # 4MB
    arr1[:] = 42.0
    
    arr2 = np.zeros((1024 * 1024,), dtype=np.float32)
    
    print(f"arr1 addr: {arr1.ctypes.data:#x}, size: {arr1.nbytes}")
    print(f"arr2 addr: {arr2.ctypes.data:#x}, size: {arr2.nbytes}")
    
    # Engine 1
    engine1 = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
    port1 = engine1.get_rpc_port()
    engine1.register_kv_cache(arr1)
    print(f"Engine1 RPC port: {port1}")
    
    # Engine 2
    engine2 = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
    port2 = engine2.get_rpc_port()
    engine2.register_kv_cache(arr2)
    print(f"Engine2 RPC port: {port2}")
    
    # Give time for metadata exchange
    time.sleep(2)
    
    # Transfer from engine1 to engine2
    print(f"\nTransferring {arr1.nbytes} bytes from engine1 to engine2...")
    success = engine1.send_kv_cache("127.0.0.1", port2, [0], [0], arr1.nbytes)
    
    if success:
        print(f"Transfer successful!")
        print(f"arr2[0] = {arr2[0]}")
        print(f"arr2[100] = {arr2[100]}")
        print(f"arr2[-1] = {arr2[-1]}")
        
        if arr2[0] == 42.0:
            print("\n✓ Data verification PASSED")
        else:
            print("\n✗ Data verification FAILED")
    else:
        print("Transfer failed!")
    
    engine1.unregister_all()
    engine2.unregister_all()


if __name__ == "__main__":
    test_transfer()
