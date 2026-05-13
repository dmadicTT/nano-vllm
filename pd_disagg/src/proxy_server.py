"""Simple proxy server for PD disaggregation.

Routes requests to prefill node first, then decode node.
Uses HTTP for simplicity (in production, this would be more robust).
"""

import json
import pickle
import socket
import struct
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
import urllib.parse

from pd_config import PDConfig


class PDProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes requests through prefill -> decode."""
    
    def __init__(self, pd_config: PDConfig, *args, **kwargs):
        self.pd_config = pd_config
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass
    
    def do_POST(self):
        if self.path == "/v1/completions" or self.path == "/v1/chat/completions":
            self._handle_completion()
        else:
            self.send_error(404)
    
    def _handle_completion(self):
        """Handle a completion request through PD pipeline."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        # Extract prompt
        if "messages" in request_data:
            # Chat completion format - extract last user message
            messages = request_data["messages"]
            prompt = messages[-1]["content"] if messages else ""
        else:
            prompt = request_data.get("prompt", "")
        
        max_tokens = request_data.get("max_tokens", 256)
        temperature = request_data.get("temperature", 0.7)
        
        # Step 1: Send to prefill node
        print(f"[Proxy] Routing request to prefill: {prompt[:50]}...")
        prefill_result = self._send_to_prefill(prompt, max_tokens, temperature)
        
        if prefill_result is None:
            self.send_error(500, "Prefill failed")
            return
        
        # Step 2: Send to decode node with transfer metadata
        print(f"[Proxy] Routing to decode with KV transfer...")
        decode_result = self._send_to_decode(prefill_result)
        
        if decode_result is None:
            self.send_error(500, "Decode failed")
            return
        
        # Return result
        response = {
            "id": f"pd-{prefill_result['request_id']}",
            "object": "text_completion",
            "created": 0,
            "model": request_data.get("model", "nano-vllm-pd"),
            "choices": [{
                "text": decode_result.get("completion", ""),
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _send_to_prefill(self, prompt: str, max_tokens: int, temperature: float) -> Optional[Dict]:
        """Send request to prefill node."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(300)
            sock.connect((self.pd_config.prefill_host, self.pd_config.prefill_port))
            
            request = {
                "action": "prefill",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            data = pickle.dumps(request)
            sock.sendall(struct.pack("!I", len(data)) + data)
            
            # Receive response
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
        except Exception as e:
            print(f"[Proxy] Prefill error: {e}")
            return None
    
    def _send_to_decode(self, prefill_result: Dict) -> Optional[Dict]:
        """Send request to decode node with prefill metadata."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(300)
            sock.connect((self.pd_config.decode_host, self.pd_config.decode_port))
            
            request = {
                "action": "decode",
                **prefill_result,
            }
            data = pickle.dumps(request)
            sock.sendall(struct.pack("!I", len(data)) + data)
            
            # Receive response
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
        except Exception as e:
            print(f"[Proxy] Decode error: {e}")
            return None


def make_handler(pd_config: PDConfig):
    """Create a handler class with pd_config bound."""
    def handler(*args, **kwargs):
        return PDProxyHandler(pd_config, *args, **kwargs)
    return handler


def run_proxy(pd_config: PDConfig):
    """Run the proxy server."""
    handler = make_handler(pd_config)
    server = HTTPServer(("0.0.0.0", pd_config.proxy_port), handler)
    print(f"[Proxy] Listening on port {pd_config.proxy_port}")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python proxy_server.py <config_file>")
        sys.exit(1)
    
    # Load config
    config = PDConfig(role="proxy")
    run_proxy(config)
