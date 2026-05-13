"""Prefill/Decode server for nano-vllm PD disaggregation - CPU version.

This version uses CPUModelRunner for testing without CUDA.
"""

import os
import sys
import pickle
import struct
import socket
import threading
import argparse
from typing import Dict, Optional

sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")
sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from transformers import AutoTokenizer

from pd_config import PDConfig
from mooncake_kv_transfer import MooncakeKVTransfer
from cpu_model_runner import CPULLMEngine, SequenceStatus
# from nanovllm.sampling_params import SamplingParams
from cpu_model_runner import SamplingParams


class PDServerV2:
    """Server that handles prefill or decode requests."""
    
    def __init__(self, pd_config: PDConfig):
        self.pd_config = pd_config
        
        # Initialize CPU engine
        print(f"[PDServerV2] Loading model from {pd_config.model_path}...")
        self.engine = CPULLMEngine(pd_config.model_path)
        self.tokenizer = self.engine.tokenizer
        
        # Initialize Mooncake with RDMA
        local_host = pd_config.prefill_host if pd_config.is_prefill() else pd_config.decode_host
        self.mooncake = MooncakeKVTransfer(
            role=pd_config.role,
            local_hostname=local_host,
            protocol="rdma",
        )
        
        # Note: We don't register the numpy KV cache with Mooncake because
        # it causes memory overlap issues with managed buffers.
        # Instead, we use managed buffers for RDMA transfers.
        self.block_size_bytes = self.engine.model_runner.kv_cache.nbytes // self.engine.model_runner.kv_cache.shape[2]
        
        # Server socket
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        host = pd_config.prefill_host if pd_config.is_prefill() else pd_config.decode_host
        port = pd_config.prefill_port if pd_config.is_prefill() else pd_config.decode_port
        self.server_sock.bind((host, port))
        self.server_sock.listen(10)
        
        # Track sequences
        self.sequences = {}  # request_id -> seq
        self.pending_transfer = {}  # request_id -> seq info
        
        print(f"[PDServerV2] {pd_config.role} node listening on {host}:{port}")
        print(f"[PDServerV2] Mooncake RPC port: {self.mooncake.get_rpc_port()}")
    
    def run(self):
        """Main server loop."""
        while True:
            conn, addr = self.server_sock.accept()
            print(f"[PDServerV2] Connection from {addr}")
            
            handler = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True
            )
            handler.start()
    
    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        try:
            # Read message length
            len_bytes = conn.recv(4)
            if len(len_bytes) < 4:
                return
            msg_len = struct.unpack("!I", len_bytes)[0]
            
            # Read message
            data = b""
            while len(data) < msg_len:
                chunk = conn.recv(min(65536, msg_len - len(data)))
                if not chunk:
                    break
                data += chunk
            
            request = pickle.loads(data)
            action = request.get("action")
            
            if action == "prefill":
                response = self._do_prefill(request)
            elif action == "decode":
                response = self._do_decode(request)
            elif action == "get_rpc_port":
                response = {"rpc_port": self.mooncake.get_rpc_port()}
            elif action == "get_kv_cache":
                response = self._do_get_kv_cache(request)
            else:
                response = {"error": f"Unknown action: {action}"}
            
            # Send response
            response_data = pickle.dumps(response)
            conn.sendall(struct.pack("!I", len(response_data)) + response_data)
            
        except Exception as e:
            print(f"[PDServerV2] Error handling connection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
    
    def _do_prefill(self, request: Dict) -> Dict:
        """Run prefill on a prompt."""
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)
        
        print(f"[PDServerV2] Prefilling: {str(prompt)[:50]}...")
        
        # Tokenize
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = prompt
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Add request and run prefill
        seq = self.engine.add_request(prompt_tokens, sampling_params)
        request_id = f"req_{seq.seq_id}"
        self.sequences[request_id] = seq
        
        # Run until prefill is complete
        prefill_done = False
        while not prefill_done:
            outputs, num_tokens = self.engine.step()
            
            # Check if our sequence has finished prefill
            if seq.num_cached_tokens >= seq.num_prompt_tokens:
                prefill_done = True
            
            # Check if it's in running state
            if seq.status == SequenceStatus.RUNNING:
                prefill_done = True
        
        print(f"[PDServerV2] Prefill complete for {request_id}")
        print(f"[PDServerV2] Block table: {seq.block_table}")
        print(f"[PDServerV2] Cached tokens: {seq.num_cached_tokens}")
        
        # Store for transfer
        self.pending_transfer[request_id] = {
            "seq": seq,
            "prompt_token_ids": prompt_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        return {
            "request_id": request_id,
            "prompt_token_ids": prompt_tokens,
            "block_table": seq.block_table,
            "num_cached_tokens": seq.num_cached_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prefill_rpc_port": self.mooncake.get_rpc_port(),
            "block_size_bytes": self.block_size_bytes,
        }
    
    def _receive_kv_cache_via_mooncake(
        self,
        remote_host: str,
        remote_port: int,
        local_block_ids: list,
        block_size_bytes: int,
        remote_buf_addrs: list = None,
    ) -> bool:
        """Receive KV cache blocks from prefill node using Mooncake RDMA.
        
        Uses Mooncake's RDMA transfer engine for zero-copy KV cache migration.
        For CPU memory, we use managed buffers which Mooncake can properly map.
        """
        print(f"[PDServerV2] Receiving KV cache via RDMA from {remote_host}:{remote_port}...")
        
        # For RDMA with CPU memory, we need to use managed buffers
        total_size = len(local_block_ids) * block_size_bytes
        
        # Allocate managed buffer for receive
        recv_buf = self.mooncake.engine.allocate_managed_buffer(total_size)
        if recv_buf == 0:
            print("[PDServerV2] Failed to allocate managed buffer")
            return False
        
        try:
            # Managed buffers are already registered by Mooncake
            # No need to call batch_register_memory
            
            # Give time for segment descriptor propagation
            import time
            time.sleep(1)
            
            # The prefill node has already written data to our recv_buf via RDMA.
            # We just need to wait a bit for the transfer to complete, then copy data.
            import time
            time.sleep(2)  # Wait for RDMA write to complete
            
            # Copy received data from managed buffer to KV cache
            all_data = self.mooncake.engine.read_bytes_from_buffer(recv_buf, total_size)
            self.engine.model_runner.set_kv_cache_bytes(local_block_ids, all_data, block_size_bytes)
            
            print(f"[PDServerV2] KV cache received via RDMA for {len(local_block_ids)} blocks")
            return True
            
        finally:
            self.mooncake.engine.free_managed_buffer(recv_buf, total_size)
    
    def _do_get_kv_cache(self, request: Dict) -> Dict:
        """Handle KV cache data request from decode node.
        
        This triggers an RDMA send from prefill to decode using managed buffers.
        """
        block_ids = request.get("block_ids", [])
        decode_rpc_port = request.get("decode_rpc_port")
        block_size_bytes = request.get("block_size_bytes", self.block_size_bytes)
        decode_buf_addrs = request.get("decode_buf_addrs", None)
        
        print(f"[PDServerV2] Serving KV cache for blocks: {block_ids} via RDMA to decode port {decode_rpc_port}")
        
        if decode_rpc_port is None:
            return {"error": "decode_rpc_port not provided"}
        
        # For RDMA with CPU memory, copy KV cache blocks to managed buffer and send
        total_size = len(block_ids) * block_size_bytes
        
        # Allocate managed buffer
        send_buf = self.mooncake.engine.allocate_managed_buffer(total_size)
        if send_buf == 0:
            return {"error": "Failed to allocate managed buffer"}
        
        try:
            # Copy KV cache data to managed buffer
            kv_data = self.engine.model_runner.get_kv_cache_bytes(block_ids)
            self.mooncake.engine.write_bytes_to_buffer(send_buf, kv_data, len(kv_data))
            
            # Managed buffers are already registered by Mooncake
            # No need to call batch_register_memory
            
            # Give time for segment descriptor propagation
            import time
            time.sleep(1)
            
            # Send via RDMA using the managed buffer address directly
            remote_session = f"{self.pd_config.decode_host}:{decode_rpc_port}"
            src_ptrs = [send_buf + i * block_size_bytes for i in range(len(block_ids))]
            if decode_buf_addrs:
                dst_ptrs = decode_buf_addrs
            else:
                dst_ptrs = [send_buf + i * block_size_bytes for i in range(len(block_ids))]
            lengths = [block_size_bytes] * len(block_ids)
            
            print(f"[PDServerV2] RDMA transfer to {remote_session}...")
            ret = self.mooncake.engine.batch_transfer_sync_write(
                remote_session, src_ptrs, dst_ptrs, lengths
            )
            
            if ret == 0:
                print(f"[PDServerV2] RDMA send successful")
                return {"status": "rdma_send_complete"}
            else:
                print(f"[PDServerV2] RDMA send failed: {ret}")
                return {"error": f"RDMA send failed: {ret}"}
        finally:
            self.mooncake.engine.free_managed_buffer(send_buf, total_size)
    
    def _do_decode(self, request: Dict) -> Dict:
        """Run decode with transferred KV cache."""
        request_id = request["request_id"]
        prompt_token_ids = request["prompt_token_ids"]
        block_table = request["block_table"]
        num_cached_tokens = request["num_cached_tokens"]
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)
        prefill_rpc_port = request["prefill_rpc_port"]
        block_size_bytes = request.get("block_size_bytes", self.block_size_bytes)
        
        print(f"[PDServerV2] Decoding request {request_id}...")
        print(f"[PDServerV2] Receiving KV cache from prefill node (port {prefill_rpc_port})...")
        
        # First, notify prefill node to send KV cache via RDMA
        print(f"[PDServerV2] Notifying prefill node to send KV cache via RDMA...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.pd_config.prefill_host, self.pd_config.prefill_port))
            
            # Allocate receive buffer and get its address
            total_size = len(block_table) * block_size_bytes
            recv_buf = self.mooncake.engine.allocate_managed_buffer(total_size)
            recv_buf_addrs = [recv_buf + i * block_size_bytes for i in range(len(block_table))]
            
            notify_data = pickle.dumps({
                "action": "get_kv_cache",
                "block_ids": block_table,
                "decode_rpc_port": self.mooncake.get_rpc_port(),
                "block_size_bytes": block_size_bytes,
                "decode_buf_addrs": recv_buf_addrs,
            })
            sock.sendall(struct.pack("!I", len(notify_data)) + notify_data)
            
            # Read response
            len_bytes = sock.recv(4)
            msg_len = struct.unpack("!I", len_bytes)[0]
            response_data = b""
            while len(response_data) < msg_len:
                chunk = sock.recv(min(65536, msg_len - len(response_data)))
                if not chunk:
                    break
                response_data += chunk
            
            response = pickle.loads(response_data)
            sock.close()
            
            if "error" in response:
                print(f"[PDServerV2] Prefill notification error: {response['error']}")
                self.mooncake.engine.free_managed_buffer(recv_buf, total_size)
                return {"error": f"Prefill notification failed: {response['error']}"}
            
            print(f"[PDServerV2] Prefill node acknowledged RDMA send")
        except Exception as e:
            print(f"[PDServerV2] Failed to notify prefill node: {e}")
            self.mooncake.engine.free_managed_buffer(recv_buf, total_size)
            return {"error": f"Failed to notify prefill node: {e}"}
        
        # Receive KV cache from prefill node via RDMA
        success = self._receive_kv_cache_via_mooncake(
            remote_host=self.pd_config.prefill_host,
            remote_port=prefill_rpc_port,
            local_block_ids=block_table,
            block_size_bytes=block_size_bytes,
            remote_buf_addrs=recv_buf_addrs,
        )
        
        if not success:
            return {"error": "Failed to receive KV cache from prefill node"}
        
        print(f"[PDServerV2] KV cache received successfully")
        
        # Verify KV cache data
        print(f"[PDServerV2] Verifying KV cache for blocks: {block_table}")
        for block_id in block_table:
            expected = float(block_id)
            actual = float(self.engine.model_runner.kv_cache[0, 0, block_id, 0, 0, 0])
            print(f"[PDServerV2] Block {block_id}: expected={expected}, actual={actual}")
            if abs(actual - expected) > 0.1:
                print(f"[PDServerV2] WARNING: Block {block_id} mismatch: expected {expected}, got {actual}")
        
        # Create sequence with received KV cache
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        seq = self.engine.add_request(prompt_token_ids, sampling_params)
        seq.block_table = list(block_table)
        seq.num_cached_tokens = num_cached_tokens
        seq.status = SequenceStatus.RUNNING  # Mark as running (decode phase)
        
        # Move from waiting to running in scheduler
        self.engine.scheduler.waiting.remove(seq)
        self.engine.scheduler.running.append(seq)
        
        # Run decode steps until completion
        print(f"[PDServerV2] Running decode...")
        completion_tokens = []
        steps = 0
        max_steps = max_tokens
        
        while steps < max_steps:
            outputs, num_tokens = self.engine.step()
            steps += 1
            
            for seq_id, tokens in outputs:
                if seq_id == seq.seq_id:
                    completion_tokens = tokens
                    break
            
            if seq.is_finished:
                break
        
        # Decode tokens
        completion_text = self.tokenizer.decode(completion_tokens)
        
        print(f"[PDServerV2] Decode complete. Generated {len(completion_tokens)} tokens")
        print(f"[PDServerV2] Completion: {completion_text[:100]}...")
        
        return {
            "request_id": request_id,
            "completion": completion_text,
            "completion_tokens": completion_tokens,
            "num_tokens_generated": len(completion_tokens),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["prefill", "decode"])
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--prefill-host", default="127.0.0.1")
    parser.add_argument("--prefill-port", type=int, default=8010)
    parser.add_argument("--decode-host", default="127.0.0.1")
    parser.add_argument("--decode-port", type=int, default=8020)
    args = parser.parse_args()
    
    # Build config
    config = PDConfig(
        role=args.role,
        model_path=args.model,
        prefill_host=args.prefill_host,
        prefill_port=args.prefill_port,
        decode_host=args.decode_host,
        decode_port=args.decode_port,
    )
    
    if args.port:
        if args.role == "prefill":
            config.prefill_port = args.port
        else:
            config.decode_port = args.port
    
    if args.host:
        if args.role == "prefill":
            config.prefill_host = args.host
        else:
            config.decode_host = args.host
    
    server = PDServerV2(config)
    server.run()


if __name__ == "__main__":
    main()
