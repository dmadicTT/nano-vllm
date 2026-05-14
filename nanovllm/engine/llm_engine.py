import atexit
import logging
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

log = logging.getLogger("nanovllm.engine")

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager
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
        # Build a relaxed SamplingParams for the prefill-side seq: the
        # scheduler's postprocess finishes a seq when num_completion_tokens
        # equals max_tokens (or it hits EOS), and on FINISH it deallocates —
        # which would wipe out the block_table and num_cached_tokens before
        # we get a chance to capture them for the handoff. The actual
        # max_tokens / ignore_eos belong to the decode side and are recorded
        # in the descriptor below.
        prefill_sp = SamplingParams(
            temperature=sampling_params.temperature,
            max_tokens=max(sampling_params.max_tokens, 2),
            ignore_eos=True,
        )
        seq = Sequence(list(prompt), prefill_sp)
        # Try to satisfy the prompt's prefix from Mooncake's shared cache
        # before we burn local compute on it. This is the "remote tier" of
        # nano-vllm's prefix cache and turns Mooncake into a fleet-wide
        # KV cache that any prefill node can warm.
        self._prefetch_from_store(seq)
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
        block_hashes = self._compute_block_hashes(seq)
        assert len(block_hashes) == len(block_ids), (
            "block hash count must match block_table length — got "
            f"{len(block_hashes)} hashes vs {len(block_ids)} blocks"
        )
        descriptor = {
            "request_id": request_id,
            "prompt_token_ids": list(prompt),
            "first_token": seq.last_token,
            "block_hashes": block_hashes,
            "num_prompt_tokens": seq.num_prompt_tokens,
            "num_cached_tokens": seq.num_cached_tokens,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        }
        # Publish KV blocks before tearing down the seq so the data is still
        # in the local cache and Mooncake can read it through put(). The push
        # itself is a no-op for any block whose hash is already in Mooncake
        # — that's the cross-request prefix cache.
        self.kv_transport.push(block_hashes, block_ids)
        self._evict_seq(seq)
        return descriptor

    def _prefetch_from_store(self, seq: Sequence) -> int:
        """Pull any prompt-prefix blocks that are already in Mooncake into the
        local KV cache, so the model only forward-passes over the suffix.

        Algorithm:
          * Walk the prompt full block-by-full block, computing the same xxh64
            chain hash `BlockManager.compute_hash` uses.
          * Local hit (`hash_to_block_id`)  → already covered by the existing
            nano-vllm prefix cache; nothing to do.
          * Mooncake hit                   → peek at `free_block_ids[0]`, pull
            the bytes into that slot, write the block's hash + tokens, and
            advertise it in `hash_to_block_id`. We don't pop the block from
            the free list — the scheduler's subsequent `allocate()` does that
            via its normal "cached block found by hash" path, which gets the
            ref counts right.
          * Miss                          → stop (we only honor a contiguous
            prefix; the suffix will be computed locally).

        Returns the number of blocks newly pulled from Mooncake (local hits
        don't count — those were already covered before this call).
        """
        if self.kv_transport is None:
            return 0
        bs = self.config.kvcache_block_size
        bm = self.scheduler.block_manager
        # Only full blocks are eligible. BlockManager.can_allocate also skips
        # the last (partial) block, so anything we advertise beyond what
        # can_allocate walks would be ignored.
        n_full = seq.num_prompt_tokens // bs
        if n_full == 0:
            return 0
        h = -1
        local_hits = 0
        remote_hits = 0
        for i in range(n_full):
            tokens = seq.token_ids[i * bs:(i + 1) * bs]
            h = BlockManager.compute_hash(tokens, h)
            # Local hit?
            local_bid = bm.hash_to_block_id.get(h, -1)
            if local_bid != -1 and bm.blocks[local_bid].token_ids == tokens:
                local_hits += 1
                continue
            # Mooncake hit?
            key = self.kv_transport.KEY_FMT.format(h=h & ((1 << 64) - 1))
            if self.kv_transport.store.is_exist(key) != 1:
                break  # contiguous prefix only — first miss ends the walk
            if not bm.free_block_ids:
                break  # capacity-bound — defer the rest to local compute
            bid = bm.free_block_ids[0]  # peek; scheduler.allocate will pop
            # The slot we're about to repurpose may have advertised a stale
            # hash from a previous tenant; drop that advert before we replace.
            old_hash = bm.blocks[bid].hash
            if old_hash != -1 and bm.hash_to_block_id.get(old_hash) == bid:
                del bm.hash_to_block_id[old_hash]
            # Pull the KV bytes into kv_cache[bid] and advertise.
            self.kv_transport.pull([h], [bid])
            self.kv_transport.blocks_prefetched += 1
            bm.blocks[bid].update(h, tokens)
            bm.hash_to_block_id[h] = bid
            remote_hits += 1
        if local_hits or remote_hits:
            log.info(
                "prefill prefetch: %d local-hit + %d Mooncake pull (out of %d full prompt blocks)",
                local_hits, remote_hits, n_full,
            )
        return remote_hits

    def _compute_block_hashes(self, seq: Sequence) -> list[int]:
        """xxh64 chain hash over the prompt tokens, block by block.

        Matches BlockManager.compute_hash so full blocks have the same hash
        whether they're computed here or by the local block manager. The last
        block may be partial (fewer than block_size tokens) — we hash whatever
        the prompt actually fills, which is exactly what's in the KV cache at
        push time. The trailing first-generated-token has no KV bytes yet, so
        it's not part of any hash.
        """
        bs = self.config.kvcache_block_size
        num_cached = seq.num_cached_tokens
        h = -1
        hashes: list[int] = []
        for i in range(len(seq.block_table)):
            start = i * bs
            end = min((i + 1) * bs, num_cached)
            if end <= start:
                break  # no KV data for this block — shouldn't happen post-prefill
            tokens = seq.token_ids[start:end]
            h = BlockManager.compute_hash(tokens, h)
            hashes.append(h)
        return hashes

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

        block_hashes = descriptor["block_hashes"]
        # Allocate decode-local block ids — bypass the hash-based prefix cache
        # so we get fresh blocks the migrated KV bytes can land in.
        bm = self.scheduler.block_manager
        decode_block_ids = [bm._allocate_block() for _ in block_hashes]
        seq.block_table = decode_block_ids

        # Pull KV blocks via Mooncake, keyed by content hash.
        self.kv_transport.pull(block_hashes, decode_block_ids)

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

        # No explicit remove: keys are content-addressed and may be reused by
        # future requests with the same prefix. Mooncake's lease/TTL handles
        # eviction.
        text = self.tokenizer.decode(completion_tokens)
        return {
            "request_id": descriptor["request_id"],
            "completion_token_ids": completion_tokens,
            "completion_text": text,
        }
