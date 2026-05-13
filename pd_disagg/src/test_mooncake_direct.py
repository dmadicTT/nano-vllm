"""Test Mooncake transfer directly to debug the issue."""

import os
import sys
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from mooncake_kv_transfer import MooncakeKVTransfer


def test_small_transfer():
    """Test with a very small array."""
    print("Testing small transfer...")
    
    # Create small arrays
    arr1 = np.zeros((1024,), dtype=np.float32)
    arr1[:] = 42.0
    
    arr2 = np.zeros((1024,), dtype=np.float32)
    
    print(f"arr1 addr: {arr1.ctypes.data}, size: {arr1.nbytes}")
    print(f"arr2 addr: {arr2.ctypes.data}, size: {arr2.nbytes}")
    
    # Initialize engines
    engine1 = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
    engine2 = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
    
    port1 = engine1.get_rpc_port()
    port2 = engine2.get_rpc_port()
    
    # Register
    engine1.register_kv_cache(arr1)
    engine2.register_kv_cache(arr2)
    
    # Transfer
    success = engine1.send_kv_cache("127.0.0.1", port2, [0], [0], arr1.nbytes)
    
    if success:
        print(f"arr2[0] after transfer: {arr2[0]}")
        if arr2[0] == 42.0:
            print("SUCCESS: Small transfer works!")
        else:
            print("FAIL: Data mismatch")
    else:
        print("FAIL: Transfer failed")
    
    engine1.unregister_all()
    engine2.unregister_all()


def test_numpy_aligned():
    """Test with page-aligned numpy array."""
    print("\nTesting page-aligned transfer...")
    
    # Allocate page-aligned memory
    page_size = 4096
    size = 4096 * 10  # 10 pages
    
    arr1 = np.zeros(size, dtype=np.float32)
    arr1[:] = 99.0
    
    arr2 = np.zeros(size, dtype=np.float32)
    
    print(f"arr1 addr: {arr1.ctypes.data:#x}, aligned: {arr1.ctypes.data % page_size == 0}")
    print(f"arr2 addr: {arr2.ctypes.data:#x}, aligned: {arr2.ctypes.data % page_size == 0}")
    
    engine1 = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
    engine2 = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
    
    port1 = engine1.get_rpc_port()
    port2 = engine2.get_rpc_port()
    
    engine1.register_kv_cache(arr1)
    engine2.register_kv_cache(arr2)
    
    success = engine1.send_kv_cache("127.0.0.1", port2, [0], [0], arr1.nbytes)
    
    if success:
        print(f"arr2[0] after transfer: {arr2[0]}")
        if arr2[0] == 99.0:
            print("SUCCESS: Page-aligned transfer works!")
        else:
            print("FAIL: Data mismatch")
    else:
        print("FAIL: Transfer failed")
    
    engine1.unregister_all()
    engine2.unregister_all()


def test_large_transfer():
    """Test with larger array."""
    print("\nTesting large transfer...")
    
    size = 1024 * 1024 * 100  # 100MB
    
    arr1 = np.zeros(size, dtype=np.float32)
    arr1[0] = 123.0
    arr1[-1] = 456.0
    
    arr2 = np.zeros(size, dtype=np.float32)
    
    print(f"arr1 size: {arr1.nbytes / 1024 / 1024:.1f} MB")
    
    engine1 = MooncakeKVTransfer("prefill", "127.0.0.1", "tcp")
    engine2 = MooncakeKVTransfer("decode", "127.0.0.1", "tcp")
    
    port1 = engine1.get_rpc_port()
    port2 = engine2.get_rpc_port()
    
    engine1.register_kv_cache(arr1)
    engine2.register_kv_cache(arr2)
    
    success = engine1.send_kv_cache("127.0.0.1", port2, [0], [0], arr1.nbytes)
    
    if success:
        print(f"arr2[0] after transfer: {arr2[0]}")
        print(f"arr2[-1] after transfer: {arr2[-1]}")
        if arr2[0] == 123.0 and arr2[-1] == 456.0:
            print("SUCCESS: Large transfer works!")
        else:
            print("FAIL: Data mismatch")
    else:
        print("FAIL: Transfer failed")
    
    engine1.unregister_all()
    engine2.unregister_all()


if __name__ == "__main__":
    test_small_transfer()
    test_numpy_aligned()
    test_large_transfer()
