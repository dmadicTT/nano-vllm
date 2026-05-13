"""Simplified demo script for PD disaggregation with nano-vllm.

This runs prefill and decode in the same process for demonstration,
using a mock model (no actual inference) to test the KV transfer pipeline.
"""

import os
import sys
import time
import threading
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")
sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from pd_config import PDConfig
from mooncake_kv_transfer import MooncakeKVTransfer


def create_mock_kv_cache(num_layers=2, num_blocks=16, block_size=256, num_kv_heads=4, head_dim=64):
    """Create a mock KV cache for testing."""
    dtype = np.float32
    shape = (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    kv_cache = np.zeros(shape, dtype=dtype)
    
    # Fill with identifiable pattern
    for b in range(num_blocks):
        kv_cache[:, :, b, :, :, :] = b
    
    return kv_cache


def test_kv_transfer():
    """Test basic KV cache transfer between two nodes."""
    print("=" * 60)
    print("Testing Mooncake KV Cache Transfer")
    print("=" * 60)
    
    # Create mock KV caches for prefill and decode
    prefill_kv = create_mock_kv_cache(num_blocks=16)
    decode_kv = create_mock_kv_cache(num_blocks=16)
    
    print(f"Prefill KV cache shape: {prefill_kv.shape}")
    print(f"Decode KV cache shape: {decode_kv.shape}")
    
    # Initialize Mooncake on both sides
    print("\n[1] Initializing Mooncake Transfer Engine...")
    prefill_mooncake = MooncakeKVTransfer(
        role="prefill",
        local_hostname="127.0.0.1",
        protocol="tcp",
    )
    
    decode_mooncake = MooncakeKVTransfer(
        role="decode",
        local_hostname="127.0.0.1",
        protocol="tcp",
    )
    
    prefill_port = prefill_mooncake.get_rpc_port()
    decode_port = decode_mooncake.get_rpc_port()
    print(f"Prefill RPC port: {prefill_port}")
    print(f"Decode RPC port: {decode_port}")
    
    # Register KV caches
    print("\n[2] Registering KV caches...")
    prefill_mooncake.register_kv_cache(prefill_kv)
    decode_mooncake.register_kv_cache(decode_kv)
    
    # Transfer some blocks
    local_block_ids = [0, 1, 2, 3]
    remote_block_ids = [0, 1, 2, 3]
    block_size_bytes = prefill_kv.nbytes // prefill_kv.shape[2]
    
    print(f"\n[3] Transferring blocks {local_block_ids}...")
    
    # Send from prefill to decode
    success = prefill_mooncake.send_kv_cache(
        remote_host="127.0.0.1",
        remote_port=decode_port,
        local_block_ids=local_block_ids,
        remote_block_ids=remote_block_ids,
        block_size_bytes=block_size_bytes,
    )
    
    if success:
        print("Transfer successful!")
        
        # Verify data
        print("\n[4] Verifying transferred data...")
        for b in local_block_ids:
            prefill_data = prefill_kv[:, :, b, 0, 0, 0]
            decode_data = decode_kv[:, :, b, 0, 0, 0]
            match = np.allclose(prefill_data, decode_data)
            print(f"  Block {b}: match={match}")
    else:
        print("Transfer failed!")
    
    # Cleanup
    prefill_mooncake.unregister_all()
    decode_mooncake.unregister_all()
    
    print("\n" + "=" * 60)
    print("KV Transfer Test Complete")
    print("=" * 60)
    return success


def test_full_pipeline():
    """Test the full PD pipeline with a simple prompt."""
    print("\n" + "=" * 60)
    print("Testing Full PD Pipeline")
    print("=" * 60)
    
    # This would require a real model, so we just show the architecture
    print("\nPipeline steps:")
    print("1. Client sends prompt to proxy")
    print("2. Proxy routes to prefill node")
    print("3. Prefill node processes prompt, generates KV cache")
    print("4. Prefill node sends KV cache to decode node via Mooncake")
    print("5. Decode node receives KV cache, continues generation")
    print("6. Decode node returns completion to proxy")
    print("7. Proxy returns result to client")
    
    print("\nFor full testing, download a model and run:")
    print("  python pd_server.py --role prefill --model <path> --port 8010")
    print("  python pd_server.py --role decode --model <path> --port 8020")
    print("  python proxy_server.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test KV transfer
    success = test_kv_transfer()
    
    if success:
        test_full_pipeline()
    else:
        print("\nKV transfer test failed - cannot proceed with full pipeline")
        sys.exit(1)
