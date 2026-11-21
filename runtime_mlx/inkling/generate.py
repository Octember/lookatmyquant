"""Streaming serial and DSpark block-8 speculative decoding for packed Inkling-2-Mini on Apple Silicon."""
from __future__ import annotations

import statistics
import time
from typing import Any, Iterator, Sequence

import mlx.core as mx
import numpy as np

from .drafter import DSparkDrafterMLX
from .model import Inkling2LoopPackedMLX, Inkling2MLXModel


def _unwrap_model(model: Inkling2LoopPackedMLX | Inkling2MLXModel) -> Inkling2LoopPackedMLX:
    return model.inner if isinstance(model, Inkling2MLXModel) else model


def _sample_next(logits_1d: mx.array, temp: float = 0.0, top_p: float = 1.0) -> int:
    if temp <= 0.0:
        return int(mx.argmax(logits_1d))
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p)
    logprobs = logits_1d - mx.logsumexp(logits_1d, axis=-1, keepdims=True)
    return int(sampler(logprobs[None, :])[0])


def stream_serial(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: set[int] | Sequence[int] = (200006,),
    temp: float = 0.0,
    top_p: float = 1.0,
    stats_out: dict[str, Any] | None = None,
) -> Iterator[list[int]]:
    """Stream single-token serial generation from a packed Inkling-2-Mini model."""
    m = _unwrap_model(model)
    eos = set(int(x) for x in eos_ids)
    greedy = temp <= 0.0
    old_wire = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    try:
        cache = m.make_cache()
        t0 = time.perf_counter()
        if greedy:
            nxt_mx, cache = m.forward_argmax(list(prompt_ids), cache=cache)
            nxt_mx = nxt_mx[0, -1]
            nxt = int(nxt_mx)
        else:
            logits, cache = m.forward(list(prompt_ids), cache=cache)
            nxt = _sample_next(logits[0, -1], temp=temp, top_p=top_p)
            mx.eval(logits)
        t1 = time.perf_counter()

        emitted = 1
        yield [nxt]
        step_ms: list[float] = []
        if nxt not in eos and max_tokens > 1:
            if greedy:
                # 1-step async_eval pipeline overlaps Python enqueue of step k+1 with GPU execution of step k!
                cur_tok_mx = mx.array([nxt], dtype=mx.int32)
                pending_mx, cache = m.forward_argmax(cur_tok_mx, cache=cache)
                pending_mx = pending_mx[0, -1]
                mx.async_eval(pending_mx)
                ts = time.perf_counter()
                for step_i in range(max_tokens - 1):
                    # Schedule step+1 BEFORE reading step's scalar if we might continue
                    next_pending = None
                    if step_i + 1 < max_tokens - 1:
                        next_pending, cache = m.forward_argmax(pending_mx.reshape(1), cache=cache)
                        next_pending = next_pending[0, -1]
                        mx.async_eval(next_pending)
                    nxt = int(pending_mx)
                    now = time.perf_counter()
                    step_ms.append((now - ts) * 1e3)
                    ts = now
                    emitted += 1
                    yield [nxt]
                    if nxt in eos:
                        break
                    pending_mx = next_pending
            else:
                for _ in range(max_tokens - 1):
                    ts = time.perf_counter()
                    logits, cache = m.forward([nxt], cache=cache)
                    nxt = _sample_next(logits[0, -1], temp=temp, top_p=top_p)
                    mx.eval(logits)
                    step_ms.append((time.perf_counter() - ts) * 1e3)
                    emitted += 1
                    yield [nxt]
                    if nxt in eos:
                        break
        t2 = time.perf_counter()
        if stats_out is not None:
            decode_s = max(t2 - t1, 1e-9)
            decode_toks = max(emitted - 1, 0)
            stats_out.update(
                {
                    "mode": "serial",
                    "prompt_tokens": len(prompt_ids),
                    "tokens": decode_toks,
                    "generated_tokens": emitted,
                    "prompt_s": round(t1 - t0, 4),
                    "rounds_s": round(decode_s, 4),
                    "decode_tok_s": round(decode_toks / decode_s, 2) if decode_toks else 0.0,
                    "median_step_ms": round(statistics.median(step_ms), 2) if step_ms else 0.0,
                    "active_gb": round(mx.get_active_memory() / 1e9, 3),
                    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
    finally:
        mx.set_wired_limit(old_wire)


def stream_dspark(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    drafter: DSparkDrafterMLX,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: set[int] | Sequence[int] = (200006,),
    adaptive: bool = True,
    draft_len: int | None = None,
    stats_out: dict[str, Any] | None = None,
) -> Iterator[list[int]]:
    """Stream exact greedy DSpark block speculative generation with zero-sync propose+verify and adaptive fallback."""
    m = _unwrap_model(model)
    eos = set(int(x) for x in eos_ids)
    tap_ids = drafter.target_layer_ids  # (5, 23, 41, 48)
    fixed_block = None if draft_len is None else max(2, min(int(draft_len), drafter.block_size))
    cur_block = fixed_block if fixed_block is not None else min(6, drafter.block_size)
    draft_head = m.lm_head.get_draft_head_argmax()
    old_wire = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    try:
        drafter.reset()
        cache = m.make_cache()
        t0 = time.perf_counter()
        first_mx, cache, taps = m.forward_argmax(list(prompt_ids), cache=cache, tap_visits=tap_ids)
        first_mx = first_mx[0, -1]
        mx.eval(first_mx, taps)
        first = int(first_mx)
        t1 = time.perf_counter()

        drafter.push_context(taps, start_pos=0)
        mx.eval(drafter.f_rnn)
        t2 = time.perf_counter()

        emitted_total = 1
        anchor = first
        yield [first]

        accepted_per_round: list[int] = []
        round_ms: list[float] = []
        serial_steps = 0
        ewma_accept = 3.0
        serial_cooldown = 0
        serial_burst_len = 8

        while emitted_total < max_tokens and anchor not in eos:
            if adaptive and serial_cooldown > 0:
                nxt_mx, cache, taps = m.forward_argmax([anchor], cache=cache, tap_visits=tap_ids)
                nxt_mx = nxt_mx[0, -1]
                mx.eval(nxt_mx, taps)
                nxt = int(nxt_mx)
                drafter.push_context(taps, start_pos=cache.seq_len - 1)
                mx.async_eval(drafter.f_rnn, *drafter.ctx_k, *drafter.ctx_v)
                emitted_total += 1
                anchor = nxt
                serial_steps += 1
                serial_cooldown -= 1
                yield [nxt]
                continue

            tr0 = time.perf_counter()
            block_size = cur_block
            # Pipelined zero-sync GPU stream: async_eval(draft) runs drafter while CPU encodes target verify!
            draft_gpu = drafter.propose_gpu(anchor, m.embed, draft_head, draft_len=block_size)
            mx.async_eval(draft_gpu)
            base_pos = cache.seq_len
            cache.begin_spec_verify()
            ver_preds_mx, cache, ver_taps = m.forward_all_argmax(draft_gpu, cache=cache, tap_visits=tap_ids)
            ver_preds_mx = ver_preds_mx[0]
            mx.eval(draft_gpu, ver_preds_mx, ver_taps)

            draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
            preds = np.array(ver_preds_mx, dtype=np.int32).tolist()

            k_match = 0
            for j in range(1, block_size):
                if preds[j - 1] == draft_block[j]:
                    k_match += 1
                else:
                    break
            k_commit = k_match + 1
            cache.commit_spec_verify(k_commit, block_size)
            drafter.push_context(ver_taps[:k_commit], start_pos=base_pos)
            mx.async_eval(drafter.f_rnn, *drafter.ctx_k, *drafter.ctx_v)

            raw = draft_block[1 : k_match + 1] + [preds[k_match]]
            chunk: list[int] = []
            for tok in raw:
                if emitted_total >= max_tokens:
                    break
                chunk.append(tok)
                emitted_total += 1
                anchor = tok
                if tok in eos:
                    break
            if chunk:
                yield chunk

            accepted_per_round.append(len(raw))
            round_ms.append((time.perf_counter() - tr0) * 1e3)
            ewma_accept = 0.7 * ewma_accept + 0.3 * len(raw)
            if adaptive:
                if fixed_block is None:
                    # Adapt draft block length to MoE active-expert cost vs acceptance rate (~20.4ms T=4, ~22.9ms T=6, ~26.1ms T=8)
                    if ewma_accept >= 4.0:
                        cur_block = drafter.block_size
                    elif ewma_accept >= 2.0:
                        cur_block = min(6, drafter.block_size)
                    else:
                        cur_block = min(4, drafter.block_size)
                if len(accepted_per_round) >= 5 and ewma_accept < 1.35:
                    serial_cooldown = serial_burst_len
                    serial_burst_len = min(16, serial_burst_len * 2)
                    ewma_accept = 2.0

        t3 = time.perf_counter()
        if stats_out is not None:
            decode_s = max(t3 - t2, 1e-9)
            decode_toks = max(emitted_total - 1, 0)
            mean_acc = sum(accepted_per_round) / max(len(accepted_per_round), 1)
            stats_out.update(
                {
                    "mode": "dspark_adaptive" if adaptive else "dspark",
                    "prompt_tokens": len(prompt_ids),
                    "tokens": decode_toks,
                    "generated_tokens": emitted_total,
                    "prompt_s": round(t1 - t0, 4),
                    "drafter_prompt_s": round(t2 - t1, 4),
                    "rounds_s": round(decode_s, 4),
                    "decode_tok_s": round(decode_toks / decode_s, 2) if decode_toks else 0.0,
                    "rounds": len(accepted_per_round),
                    "tokens_per_check": round(mean_acc, 2),
                    "median_round_ms": round(statistics.median(round_ms), 2) if round_ms else 0.0,
                    "serial_steps": serial_steps,
                    "active_gb": round(mx.get_active_memory() / 1e9, 3),
                    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
    finally:
        mx.set_wired_limit(old_wire)


def generate_serial(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    prompt_ids: Sequence[int],
    max_tokens: int = 128,
    eos_ids: Sequence[int] = (200006,),
    temp: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[int], dict[str, Any]]:
    stats: dict[str, Any] = {}
    out: list[int] = []
    for chunk in stream_serial(
        model, prompt_ids, max_tokens=max_tokens, eos_ids=eos_ids, temp=temp, top_p=top_p, stats_out=stats
    ):
        out.extend(chunk)
    return out, stats


def generate_dspark(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    drafter: DSparkDrafterMLX,
    prompt_ids: Sequence[int],
    max_tokens: int = 128,
    eos_ids: Sequence[int] = (200006,),
    adaptive: bool = True,
    draft_len: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    stats: dict[str, Any] = {}
    out: list[int] = []
    for chunk in stream_dspark(
        model, drafter, prompt_ids, max_tokens=max_tokens, eos_ids=eos_ids, adaptive=adaptive, draft_len=draft_len, stats_out=stats
    ):
        out.extend(chunk)
    return out, stats
