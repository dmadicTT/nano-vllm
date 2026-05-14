import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # PD disaggregation hook — lazily built so the import path stays free of
        # the Mooncake dependency for vanilla colocated use.
        self.kv_transport = None
        if config.role != "colocated":
            from nanovllm.engine.kv_transfer import KVTransfer
            self.kv_transport = KVTransfer(
                kv_cache=self.model_runner.kv_cache,
                role=config.role,
                local_hostname=config.mooncake_local_hostname,
                metadata_server=config.mooncake_metadata_server,
                master_addr=config.mooncake_master_addr,
                protocol=config.mooncake_protocol,
                rdma_devices=config.mooncake_rdma_devices,
                global_segment_size=config.mooncake_global_segment_size,
                local_buffer_size=config.mooncake_local_buffer_size,
            )
        atexit.register(self.exit)

    def exit(self):
        if self.kv_transport is not None:
            self.kv_transport.close()
            self.kv_transport = None
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    # ----------------------------- PD disaggregation -----------------------------
    # The two methods below are entered by an external orchestrator (see
    # nanovllm/engine/pd_server.py) — they assume the engine has been built
    # with role='prefill' or role='decode' respectively.

    def run_prefill_and_publish(
        self,
        prompt: list[int],
        sampling_params: SamplingParams,
        request_id: str,
    ) -> dict:
        """Prefill on this node, publish KV blocks to Mooncake, return a handoff descriptor.

        The seq is fully advanced until it sits in the running queue with one
        generated token, then evicted from the scheduler (its KV stays in the
        local cache only until the next request reuses those block ids; the
        receiver doesn't depend on that).
        """
        assert self.config.role == "prefill", "called run_prefill_and_publish in non-prefill role"
        seq = Sequence(list(prompt), sampling_params)
        self.scheduler.add(seq)
        # Iterate until the seq leaves prefill (status RUNNING and the next token is appended).
        while not seq.is_finished and seq.status != SequenceStatus.RUNNING:
            self.step()
        # Drive one more step ONLY if the prefill state hasn't produced its first
        # token yet (chunked prefill path). After this, the scheduler will move
        # on to decode for this seq if we don't pull it.
        if seq.num_completion_tokens == 0:
            self.step()

        block_ids = list(seq.block_table)
        descriptor = {
            "request_id": request_id,
            "prompt_token_ids": list(prompt),
            "first_token": seq.last_token,
            "block_count": len(block_ids),
            "num_prompt_tokens": seq.num_prompt_tokens,
            "num_cached_tokens": seq.num_cached_tokens,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        }
        # Publish KV blocks before tearing down the seq so the data is still
        # in the local cache and Mooncake can read it through put().
        self.kv_transport.push(request_id, block_ids)
        self._evict_seq(seq)
        return descriptor

    def _evict_seq(self, seq: Sequence):
        """Remove a sequence from the scheduler and release its blocks."""
        self.scheduler.block_manager.deallocate(seq)
        if seq in self.scheduler.running:
            self.scheduler.running.remove(seq)
        if seq in self.scheduler.waiting:
            self.scheduler.waiting.remove(seq)
        seq.status = SequenceStatus.FINISHED

    def run_decode_from_handoff(self, descriptor: dict) -> dict:
        """Resume a sequence whose KV cache was prefilled on another node.

        Allocates fresh local blocks (the decode block-manager has its own id
        space), pulls the prefilled bytes from Mooncake into those blocks, and
        runs the decode loop until completion.
        """
        assert self.config.role == "decode", "called run_decode_from_handoff in non-decode role"
        sp = SamplingParams(
            temperature=descriptor["temperature"],
            max_tokens=descriptor["max_tokens"],
            ignore_eos=descriptor.get("ignore_eos", False),
        )
        # Build the seq state: prompt + first generated token. The first
        # generated token's KV will be computed on the *first* decode step
        # of this node — only the prompt's KV bytes come from the wire.
        full_tokens = list(descriptor["prompt_token_ids"]) + [descriptor["first_token"]]
        seq = Sequence(full_tokens, sp)
        seq.num_prompt_tokens = descriptor["num_prompt_tokens"]
        seq.num_cached_tokens = descriptor["num_cached_tokens"]
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False

        # Allocate decode-local block ids — bypass the hash-based prefix cache
        # so we get fresh blocks the migrated KV bytes can land in.
        bm = self.scheduler.block_manager
        decode_block_ids = [bm._allocate_block() for _ in range(descriptor["block_count"])]
        seq.block_table = decode_block_ids

        # Pull KV blocks via Mooncake
        self.kv_transport.pull(descriptor["request_id"], decode_block_ids)

        # Put the seq into the running queue so the scheduler picks it up.
        self.scheduler.running.append(seq)

        # Decode loop
        completion_tokens = []
        while seq.status == SequenceStatus.RUNNING:
            outputs, _ = self.step()
            for seq_id, tokens in outputs:
                if seq_id == seq.seq_id:
                    completion_tokens = tokens
                    break

        # Cleanup: tell prefill's Mooncake store it can drop the keys.
        self.kv_transport.remove(descriptor["request_id"], descriptor["block_count"])
        text = self.tokenizer.decode(completion_tokens)
        return {
            "request_id": descriptor["request_id"],
            "completion_token_ids": completion_tokens,
            "completion_text": text,
        }
