import atexit
import logging
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

log = logging.getLogger("nanovllm.engine")
# `nanovllm.flow` is a separate logger that emits one INFO line per phase of
# a PD request (hashing -> prefetch -> prefill -> push -> decode -> ...).
# Demos that want a clean narrative can enable just this logger at INFO and
# silence everything else.
flow = logging.getLogger("nanovllm.flow")


def _short_hashes(hashes, n=3, width=16):
    """Render a list of int hashes as a compact comma-separated string."""
    hexed = [f"{h & ((1 << 64) - 1):0{width}x}" for h in hashes]
    if len(hexed) > n:
        return ", ".join(hexed[:n]) + f", ... (+{len(hexed) - n} more)"
    return ", ".join(hexed)

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

        # ---- 1) Hashing: compute every block's content hash upfront -------
        bs = self.config.kvcache_block_size
        full_block_hashes, partial_block_hash = self._hash_prompt_blocks(seq)
        n_total = len(full_block_hashes) + (1 if partial_block_hash is not None else 0)
        flow.info("Hashing: %d block(s) (%d full + %d partial) for request %s",
                  n_total, len(full_block_hashes),
                  1 if partial_block_hash is not None else 0, request_id)

        # ---- 2) Local cache lookup + 3) store lookup ----------------------
        local_hits, remote_hits = self._prefetch_from_store(seq, full_block_hashes)
        flow.info("Found %d block(s) locally", local_hits)
        n_to_fetch = len(full_block_hashes) - local_hits
        if n_to_fetch > 0:
            flow.info("Fetching from store: %d block(s)", n_to_fetch)
            flow.info("Found %d in store", remote_hits)

        # ---- 4) Prefill: the scheduler now sees a seq whose num_cached_tokens
        #        already covers the prefetched prefix; model forward only
        #        runs over the suffix.
        n_already_cached = (local_hits + remote_hits) * bs
        n_to_compute = seq.num_prompt_tokens - n_already_cached
        flow.info("Prefilling: %d token(s)", n_to_compute)
        self.scheduler.add(seq)
        while not seq.is_finished and seq.status != SequenceStatus.RUNNING:
            self.step()
        if seq.num_completion_tokens == 0:
            self.step()

        block_ids = list(seq.block_table)
        all_hashes = list(full_block_hashes)
        if partial_block_hash is not None:
            all_hashes.append(partial_block_hash)
        assert len(all_hashes) == len(block_ids), (
            "block hash count must match block_table length — got "
            f"{len(all_hashes)} hashes vs {len(block_ids)} blocks"
        )

        descriptor = {
            "request_id": request_id,
            "prompt_token_ids": list(prompt),
            "first_token": seq.last_token,
            "block_hashes": all_hashes,
            "num_prompt_tokens": seq.num_prompt_tokens,
            "num_cached_tokens": seq.num_cached_tokens,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        }

        # ---- 5) Push: skip blocks already in the store via is_exist -------
        prev_pushed = self.kv_transport.blocks_pushed
        prev_skipped = self.kv_transport.blocks_skipped_push
        self.kv_transport.push(all_hashes, block_ids)
        n_pushed_now = self.kv_transport.blocks_pushed - prev_pushed
        n_skipped_now = self.kv_transport.blocks_skipped_push - prev_skipped
        # Figure out exactly which hashes were the new pushes for the log line.
        # The push order is identical to all_hashes, and skips happen in-place,
        # so we can reconstruct: a hash was pushed iff it wasn't in the store
        # before we called push. We do a fresh is_exist round here (cheap)
        # only for the log message — these keys are now ALL present.
        # In practice, "pushed" = all_hashes minus those already in store
        # before push. We approximate by saying: of the n_pushed_now newest,
        # the trailing block (the partial) is most likely the new one;
        # for accuracy we'd track per-call inside push(). To keep the log
        # honest, push the count + show hash IDs of the ones in this request.
        flow.info("Pushing %d block(s) to store (skipped %d already cached): hashes=[%s]",
                  n_pushed_now, n_skipped_now, _short_hashes(all_hashes))

        self._evict_seq(seq)
        return descriptor

    def _hash_prompt_blocks(self, seq: Sequence):
        """Return (full_block_hashes, partial_block_hash_or_None).

        The chain hash matches BlockManager.compute_hash. Full blocks are
        the ones that participate in the prefix cache (`can_allocate` only
        looks at full blocks too). The partial last block gets a hash for
        push purposes only.
        """
        bs = self.config.kvcache_block_size
        num_prompt = seq.num_prompt_tokens
        n_full = num_prompt // bs
        has_partial = (num_prompt % bs) != 0
        full_hashes = []
        h = -1
        for i in range(n_full):
            tokens = seq.token_ids[i * bs:(i + 1) * bs]
            h = BlockManager.compute_hash(tokens, h)
            full_hashes.append(h)
        partial_hash = None
        if has_partial:
            tokens = seq.token_ids[n_full * bs:num_prompt]
            partial_hash = BlockManager.compute_hash(tokens, h)
        return full_hashes, partial_hash

    def _prefetch_from_store(self, seq: Sequence, full_block_hashes: list[int]):
        """Pull any prompt-prefix blocks that are already in Mooncake into the
        local KV cache so the model only forward-passes over the suffix.

        For each precomputed full-block hash:
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

        Returns (local_hits, remote_hits).
        """
        if self.kv_transport is None or not full_block_hashes:
            return 0, 0
        bs = self.config.kvcache_block_size
        bm = self.scheduler.block_manager
        local_hits = 0
        remote_hits = 0
        for i, h in enumerate(full_block_hashes):
            tokens = seq.token_ids[i * bs:(i + 1) * bs]
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
            old_hash = bm.blocks[bid].hash
            if old_hash != -1 and bm.hash_to_block_id.get(old_hash) == bid:
                del bm.hash_to_block_id[old_hash]
            self.kv_transport.pull([h], [bid])
            self.kv_transport.blocks_prefetched += 1
            bm.blocks[bid].update(h, tokens)
            bm.hash_to_block_id[h] = bid
            remote_hits += 1
        return local_hits, remote_hits

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
        flow.info("Received %d block hash(es) from prefill for request %s",
                  len(block_hashes), descriptor["request_id"])
        # Allocate decode-local block ids — bypass the hash-based prefix cache
        # so we get fresh blocks the migrated KV bytes can land in.
        bm = self.scheduler.block_manager
        decode_block_ids = [bm._allocate_block() for _ in block_hashes]
        seq.block_table = decode_block_ids

        # Pull KV blocks via Mooncake, keyed by content hash.
        flow.info("Fetching from store: %d block(s)", len(block_hashes))
        self.kv_transport.pull(block_hashes, decode_block_ids)
        flow.info("Found %d in store", len(block_hashes))

        # Put the seq into the running queue so the scheduler picks it up.
        self.scheduler.running.append(seq)
        flow.info("Decoding...")

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
        flow.info("Decoded %d token(s)", len(completion_tokens))
        return {
            "request_id": descriptor["request_id"],
            "completion_token_ids": completion_tokens,
            "completion_text": text,
        }
