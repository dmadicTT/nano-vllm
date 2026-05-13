"""Prefill/Decode server for nano-vllm PD disaggregation.

Each instance runs either as prefill or decode node.
Communicates with proxy via TCP sockets.
"""

import os
import sys
import pickle
import struct
import socket
import threading
import argparse
from typing import Dict, Optional

# Add paths
sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm")
sys.path.insert(0, "/root/.openclaw/workspace/nano-vllm-pd")

from transformers import AutoTokenizer

from pd_config import PDConfig
from pd_engine import PDEngine
from nanovllm.sampling_params import SamplingParams


class PDServer:
    """Server that handles prefill or decode requests."""
    
    def __init__(self, pd_config: PDConfig):
        self.pd_config = pd_config
        self.engine = PDEngine(pd_config)
        self.tokenizer = AutoTokenizer.from_pretrained(pd_config.model_path, use_fast=True)
        
        # Server socket
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        host = pd_config.prefill_host if pd_config.is_prefill() else pd_config.decode_host
        port = pd_config.prefill_port if pd_config.is_prefill() else pd_config.decode_port
        self.server_sock.bind((host, port))
        self.server_sock.listen(10)
        
        print(f"[PDServer] {pd_config.role} node listening on {host}:{port}")
    
    def run(self):
        """Main server loop."""
        while True:
            conn, addr = self.server_sock.accept()
            print(f"[PDServer] Connection from {addr}")
            
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
            else:
                response = {"error": f"Unknown action: {action}"}
            
            # Send response
            response_data = pickle.dumps(response)
            conn.sendall(struct.pack("!I", len(response_data)) + response_data)
            
        except Exception as e:
            print(f"[PDServer] Error handling connection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
    
    def _do_prefill(self, request: Dict) -> Dict:
        """Run prefill on a prompt."""
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)
        
        print(f"[PDServer] Prefilling: {prompt[:50]}...")
        
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
        
        # Add request
        request_id = self.engine.add_request(prompt_tokens, sampling_params)
        
        # Run prefill steps until prefill is complete
        prefill_complete = False
        seq = None
        while not prefill_complete:
            outputs, num_tokens = self.engine.step()
            
            # Check if our request has finished prefill
            for s in self.engine.scheduler.running:
                if isinstance(s, self.engine.scheduler.__class__.__mro__[0]):
                    pass
            
            # Check pending transfers
            for s in self.engine.get_pending_transfers():
                if s.request_id == request_id:
                    seq = s
                    prefill_complete = True
                    break
            
            # Also check if it's still in waiting/running
            if not prefill_complete:
                all_done = True
                for q in [self.engine.scheduler.waiting, self.engine.scheduler.running]:
                    for s in q:
                        if hasattr(s, 'request_id') and s.request_id == request_id:
                            all_done = False
                            if s.num_cached_tokens >= s.num_prompt_tokens:
                                # Prefill done, should be in pending
                                seq = s
                                prefill_complete = True
                
                if all_done and seq is None:
                    # Request might have finished completely (short prompt)
                    break
        
        if seq is None:
            return {"error": "Prefill failed - sequence not found"}
        
        # Return metadata for decode node
        return {
            "request_id": request_id,
            "prompt_token_ids": prompt_tokens,
            "block_table": seq.block_table,
            "num_cached_tokens": seq.num_cached_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prefill_rpc_port": self.engine.mooncake.get_rpc_port(),
        }
    
    def _do_decode(self, request: Dict) -> Dict:
        """Run decode with transferred KV cache."""
        request_id = request["request_id"]
        prompt_token_ids = request["prompt_token_ids"]
        block_table = request["block_table"]
        num_cached_tokens = request["num_cached_tokens"]
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)
        prefill_rpc_port = request["prefill_rpc_port"]
        
        print(f"[PDServer] Decoding request {request_id}...")
        
        # First, receive KV cache from prefill node
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        success = self.engine.receive_sequence(
            request_id=request_id,
            prefill_rpc_port=prefill_rpc_port,
            block_table=block_table,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        
        if not success:
            return {"error": "Failed to receive KV cache from prefill node"}
        
        # Run decode steps until completion
        completion_tokens = []
        while not self.engine.is_finished():
            outputs, num_tokens = self.engine.step()
            
            for seq_id, tokens in outputs:
                completion_tokens = tokens
        
        # Decode tokens
        completion_text = self.tokenizer.decode(completion_tokens)
        
        return {
            "request_id": request_id,
            "completion": completion_text,
            "completion_tokens": completion_tokens,
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
    
    server = PDServer(config)
    server.run()


if __name__ == "__main__":
    main()
