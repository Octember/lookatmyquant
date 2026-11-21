"""Standalone Apple Silicon MLX implementation of bit-packed Inkling-2-Mini (Inkling2Loop).

Matches Hugging Face PR #51050 `modeling_inkling2_loop.py` line-for-line:
  - 24 physical layers with L6..17 looped 3x (48 logical visits), 48 KV + sconv cache slots
  - Block Attention Residuals (`attention_input`, `mlp_input`, `final_input`) & cross-pass `message_input`
  - 48 Loop Message Gates (`{q,k,v,mlp}_message_gate`) on passes 1 & 2
  - Static (`ShortConvolution`) & Dynamic (`DynamicShortConvolution`) causal 4-tap convolutions
  - Scaled/LogScaled L2Clip on Q/K/relpos, UnweightedRMSNorm on V, relative position bias + per-token sinks
  - Gated attention output (`sigmoid(gate_proj(h)) * attn_out -> linear`)
  - FP32 Sigmoid Top-8 router (`e_score_correction_bias`, `route_scale=8.0`) + SoftcappedSwiGLU (4.0, 25.0)
  - All 6,144 MoE experts + 144 attention linears + untied embed/lm_head executed directly from
    exact bit-packed 4-8 KB SRAM half-warp Metal kernels (`kernels.py`) at ~8.85 GB resident RAM.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .kernels import (
    HASH_MULT_U32,
    lm_dense_matmul,
    rht_apply_in,
    rht_apply_out,
    trellis_dense_matmul,
    trellis_matvec,
    trellis_moe_grouped,
)

ID_TO_FMT = {0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4"}
FMT_PARAMS = {
    "ht1": (1, 4, 512, HASH_MULT_U32),
    "ht2": (2, 4, 512, HASH_MULT_U32),
    "ht3": (3, 2, 1024, HASH_MULT_U32),
    "rtr1": (1, 1, 4096, 0),
    "rtr2": (2, 1, 4096, 0),
    "rtr3": (3, 1, 4096, 0),
    "rtr4": (4, 1, 4096, 0),
}


@dataclass
class Inkling2Config:
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    local_layer_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
    sconv_kernel_size: int = 4
    conv_kernel_size: int = 4
    final_logit_softcapping: float = 30.0
    sliding_window_size: int = 1024
    n_layers_per_attn_res_block: int = 6
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    intermediate_size: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    loop_gate_rank: int = 64

    @classmethod
    def from_json(cls, path: Path) -> "Inkling2Config":
        raw = json.loads(path.read_text())
        tc = raw.get("text_config", raw)
        lc = tc.get("loop_config", {})
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            vocab_size=tc["vocab_size"],
            unpadded_vocab_size=tc.get("unpadded_vocab_size", tc["vocab_size"]),
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            d_rel=tc["d_rel"],
            rel_extent=tc["rel_extent"],
            log_scaling_n_floor=tc["log_scaling_n_floor"],
            log_scaling_alpha=tc["log_scaling_alpha"],
            rms_norm_eps=tc["rms_norm_eps"],
            qk_l2_clip_threshold=tc["qk_l2_clip_threshold"],
            rel_l2_clip_threshold=tc["rel_l2_clip_threshold"],
            local_layer_ids=tuple(tc["local_layer_ids"]),
            sconv_kernel_size=tc.get("sconv_kernel_size", 4),
            conv_kernel_size=tc.get("conv_kernel_size", tc.get("sconv_kernel_size", 4)),
            final_logit_softcapping=tc["final_logit_softcapping"],
            sliding_window_size=tc["sliding_window_size"],
            n_layers_per_attn_res_block=tc["n_layers_per_attn_res_block"],
            n_routed_experts=tc["n_routed_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            intermediate_size=tc["intermediate_size"],
            mlp_gate_softcap=tc["mlp_gate_softcap"],
            mlp_up_softcap=tc["mlp_up_softcap"],
            route_scale=tc["route_scale"],
            loop_start=lc.get("core_start", 6),
            loop_end=lc.get("core_end", 18),
            loop_passes=lc.get("num_passes", 3),
            loop_gate_rank=lc.get("gate_rank", 64),
        )

    @property
    def total_visits(self) -> int:
        return self.num_hidden_layers + (self.loop_passes - 1) * (self.loop_end - self.loop_start)


# ----------------------------------------------------------------------------- Primitives (exact FP32 matching HF)

@lru_cache(maxsize=64)
def _f32(val: float) -> mx.array:
    return mx.array(val, dtype=mx.float32)


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), None, eps)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps)


@mx.compile
def _l2_clip_compiled(x: mx.array, threshold: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    return xf * mx.minimum(threshold / (norm + 1e-6), 1.0)


def l2_clip(x: mx.array, threshold: float) -> mx.array:
    return _l2_clip_compiled(x, _f32(threshold))


def log_scale(x: mx.array, position_ids: mx.array, alpha: float, n_floor: int) -> mx.array:
    # position_ids: [B, T]
    tau = 1.0 + alpha * mx.log(mx.maximum((position_ids.astype(mx.float32) + 1.0) / float(n_floor), 1.0))
    return x.astype(mx.float32) * tau[..., None, None]


@mx.compile
def _softcap_compiled(x: mx.array, cap: mx.array) -> mx.array:
    return mx.tanh(x / cap) * cap


def softcap(x: mx.array, cap: float) -> mx.array:
    return _softcap_compiled(x, _f32(cap))


@mx.compile
def _softcapped_swiglu_compiled(gate_up: mx.array, gate_cap: mx.array, up_cap: mx.array) -> mx.array:
    g, u = mx.split(gate_up.astype(mx.float32), 2, axis=-1)
    return (mx.tanh(g / gate_cap) * gate_cap) * mx.sigmoid(g) * (mx.tanh(u / up_cap) * up_cap)


def softcapped_swiglu(gate_up: mx.array, gate_cap: float = 4.0, up_cap: float = 25.0) -> mx.array:
    return _softcapped_swiglu_compiled(gate_up, _f32(gate_cap), _f32(up_cap))


# ----------------------------------------------------------------------------- 48-Visit KV + Conv Cache

class Inkling2Cache:
    """KV + 4-slot causal conv state cache across all 48 logical layer visits, with exact speculative rollback."""

    def __init__(self, cfg: Inkling2Config):
        self.cfg = cfg
        n_slots = cfg.total_visits
        self.keys: list[mx.array | None] = [None] * n_slots
        self.values: list[mx.array | None] = [None] * n_slots
        self.conv_states: list[list[mx.array | None]] = [[None, None, None, None] for _ in range(n_slots)]
        self.seq_len: int = 0
        self._spec_recording: bool = False
        self._spec_base_seq: int = 0
        self._spec_kv_full: list[tuple[mx.array, mx.array, bool] | None] = [None] * n_slots
        self._spec_conv_full: list[list[mx.array | None]] = [[None, None, None, None] for _ in range(n_slots)]

    def begin_spec_verify(self):
        self._spec_recording = True
        self._spec_base_seq = self.seq_len

    def commit_spec_verify(self, k_accept: int, block_len: int):
        """Roll back KV & conv states after verifying `block_len` tokens and accepting the first `k_accept` (`1..block_len`)."""
        self._spec_recording = False
        drop = block_len - k_accept
        self.seq_len = self._spec_base_seq + k_accept
        for slot in range(len(self.keys)):
            kv = self._spec_kv_full[slot]
            if kv is not None:
                K, V, is_sliding = kv
                if drop > 0:
                    K = K[:, :, :-drop, :]
                    V = V[:, :, :-drop, :]
                if is_sliding and K.shape[2] > self.cfg.sliding_window_size:
                    K = K[:, :, -self.cfg.sliding_window_size :, :]
                    V = V[:, :, -self.cfg.sliding_window_size :, :]
                self.keys[slot] = K
                self.values[slot] = V
                self._spec_kv_full[slot] = None
            for ci in range(4):
                cf = self._spec_conv_full[slot][ci]
                if cf is not None:
                    need = cf.shape[1] - block_len
                    self.conv_states[slot][ci] = cf[:, k_accept : k_accept + need, :]
                    self._spec_conv_full[slot][ci] = None

    def update_kv(self, slot: int, k: mx.array, v: mx.array, is_sliding: bool) -> tuple[mx.array, mx.array]:
        # k, v: [B, n_kv, T, head_dim]
        if self.keys[slot] is None:
            K, V = k, v
        else:
            K = mx.concatenate([self.keys[slot], k], axis=2)
            V = mx.concatenate([self.values[slot], v], axis=2)
        if self._spec_recording:
            self._spec_kv_full[slot] = (K, V, is_sliding)
        if is_sliding and K.shape[2] > self.cfg.sliding_window_size:
            self.keys[slot] = K[:, :, -self.cfg.sliding_window_size :, :]
            self.values[slot] = V[:, :, -self.cfg.sliding_window_size :, :]
        else:
            self.keys[slot] = K
            self.values[slot] = V
        return K, V

    def update_conv(self, slot: int, conv_idx: int, x_btc: mx.array, kernel_size: int) -> mx.array:
        # x_btc: [B, T, C] float32. Returns context [B, T + kernel_size - 1, C] with exact causal left-padding (zero transposes!).
        B, T, C = x_btc.shape
        need = kernel_size - 1
        prev = self.conv_states[slot][conv_idx]
        if prev is None:
            prev = mx.zeros((B, need, C), dtype=mx.float32)
        full = mx.concatenate([prev, x_btc], axis=1)
        if self._spec_recording:
            self._spec_conv_full[slot][conv_idx] = full
        self.conv_states[slot][conv_idx] = full[:, -need:, :]
        return full

    @property
    def offset(self) -> int:
        return self.seq_len

    @property
    def state(self) -> list[mx.array]:
        out: list[mx.array] = []
        for k, v in zip(self.keys, self.values):
            if k is not None:
                out.append(k)
            if v is not None:
                out.append(v)
        for slot_c in self.conv_states:
            for c in slot_c:
                if c is not None:
                    out.append(c)
        return out


# ----------------------------------------------------------------------------- Packed Linear & MoE Modules

class PackedLinear:
    """Executes one bit-packed attention linear (`ht*`, `rtr*`, `lm6`, or `lm8`) directly from packed bytes."""

    def __init__(self, meta: dict[str, Any], arrays: dict[str, Any], luts: dict[str, mx.array]):
        self.fmt = meta["fmt"]
        self.m = int(meta["m"])
        self.n = int(meta["n"])
        self.k = int(meta["k"])
        safe = meta["safe"]
        self.codes = mx.array(arrays[f"lin_{safe}_codes"])
        self.scales = mx.array(arrays[f"lin_{safe}_scales"])  # float16
        self.signs_in = mx.array(arrays[f"lin_{safe}_signs_in"].astype(np.float32))
        self.out_rot = bool(meta.get("out_rot", False))
        if self.out_rot:
            self.signs_out = mx.array(arrays[f"lin_{safe}_signs_out"].astype(np.float32))
        else:
            self.signs_out = None
        if self.fmt.startswith("lm"):
            self.levels = mx.array(arrays[f"lin_{safe}_levels"].astype(np.float32))
            self.kind = "lm"
        else:
            self.kind = "trellis"
            self.g = int(meta["g"])
            self.K = int(meta["K"])
            self.hash_mult = int(meta["hash_mult"])
            self.lut = luts[self.fmt]

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.shape
        T = int(np.prod(orig[:-1]))
        x_rot = rht_apply_in(x.reshape(T, self.n), self.signs_in)
        if self.kind == "lm":
            y = lm_dense_matmul(x_rot, self.codes, self.scales, self.levels, self.m, self.n, self.k)
        else:
            y = trellis_dense_matmul(
                x_rot, self.codes, self.scales, self.lut,
                self.m, self.n, self.k, self.g, self.K, self.hash_mult,
            )
        if self.out_rot:
            y = rht_apply_out(y, self.signs_out)
        return y.reshape(*orig[:-1], self.m)


class PackedMoE:
    """Executes all 128 MoE experts from 4-8 KB SRAM half-warp packed slabs with shared layer input RHT.

    100% GPU-resident and zero CPU↔GPU syncs: routes all `T` tokens in one batch and dispatches
    each active format slab once with `0xFFFFFFFF` sentinel masking so the entire 48-visit model
    runs in one unbroken asynchronous Metal command stream.
    """

    def __init__(self, arrays: dict[str, Any], luts: dict[str, mx.array], router_w: mx.array, router_bias: mx.array, cfg: Inkling2Config):
        self.cfg = cfg
        self.router_w = router_w.astype(mx.float32)          # [128, 2304]
        self.router_bias = router_bias.astype(mx.float32)    # [128]
        self.gu_signs_in = mx.array(arrays["gu_signs_in"].astype(np.float32))
        self.dn_signs_in = mx.array(arrays["dn_signs_in"].astype(np.float32))
        gu_map = np.asarray(arrays["gu_map"], dtype=np.int32)  # [128, 2] -> (fmt_id, local_idx)
        dn_map = np.asarray(arrays["dn_map"], dtype=np.int32)
        self.gu_map_fid = mx.array(gu_map[:, 0].astype(np.int32))
        self.gu_map_loc = mx.array(gu_map[:, 1].astype(np.uint32))
        self.dn_map_fid = mx.array(dn_map[:, 0].astype(np.int32))
        self.dn_map_loc = mx.array(dn_map[:, 1].astype(np.uint32))
        self._sentinel = mx.array(0xFFFFFFFF, dtype=mx.uint32)
        self.gu_slabs: list[tuple[int, mx.array, mx.array, mx.array, tuple[int, int, int, int]]] = []
        self.dn_slabs: list[tuple[int, mx.array, mx.array, mx.array, tuple[int, int, int, int]]] = []
        for fid, fmt in ID_TO_FMT.items():
            gk = f"gu_{fmt}_codes"
            if gk in arrays:
                self.gu_slabs.append((fid, mx.array(arrays[gk]), mx.array(arrays[f"gu_{fmt}_scales"]), luts[fmt], FMT_PARAMS[fmt]))
            dk = f"dn_{fmt}_codes"
            if dk in arrays:
                self.dn_slabs.append((fid, mx.array(arrays[dk]), mx.array(arrays[f"dn_{fmt}_scales"]), luts[fmt], FMT_PARAMS[fmt]))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # hidden_states: [B, T, 2304]
        B, T, H = hidden_states.shape
        N = B * T
        K_top = self.cfg.num_experts_per_tok
        flat = hidden_states.reshape(N, H).astype(mx.float32)

        logits = flat @ self.router_w.T  # [N, 128]
        scores = mx.sigmoid(logits)
        biased = scores + self.router_bias[None, :]
        idx_top = mx.argpartition(-biased, K_top - 1, axis=-1)[:, :K_top]  # [N, 8]
        w_raw = mx.take_along_axis(scores, idx_top, axis=-1)               # [N, 8]
        w_norm = (w_raw / (mx.sum(w_raw, axis=-1, keepdims=True) + 1e-20)) * self.cfg.route_scale

        eids_flat = idx_top.reshape(-1)  # [N * 8]
        gu_fids = self.gu_map_fid[eids_flat]
        gu_locs = self.gu_map_loc[eids_flat]
        dn_fids = self.dn_map_fid[eids_flat]
        dn_locs = self.dn_map_loc[eids_flat]

        # Shared input RHT computed ONCE for all N tokens; x_div=K_top maps slot s -> x_rot[s // K_top]
        x_rot = rht_apply_in(flat, self.gu_signs_in)  # [N, H]
        use_grouped = 2 <= N <= 8
        moe_fn = trellis_moe_grouped if use_grouped else trellis_matvec
        single_gu = len(self.gu_slabs) == 1
        y_gu_total = None
        for fid, codes, scales, lut, (k, g, K, hmult) in self.gu_slabs:
            locs_f = gu_locs if single_gu else mx.where(gu_fids == fid, gu_locs, self._sentinel)
            y_f = moe_fn(
                x_rot, codes, scales, lut, locs_f,
                2 * self.cfg.intermediate_size, self.cfg.hidden_size, k, g, K, hmult, x_div=K_top,
            )
            y_gu_total = y_f if y_gu_total is None else (y_gu_total + y_f)

        act = softcapped_swiglu(y_gu_total, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap)  # [N * 8, 1152]
        act_rot = rht_apply_in(act, self.dn_signs_in)  # [N * 8, 1152]

        single_dn = len(self.dn_slabs) == 1
        y_dn_total = None
        for fid, codes, scales, lut, (k, g, K, hmult) in self.dn_slabs:
            locs_f = dn_locs if single_dn else mx.where(dn_fids == fid, dn_locs, self._sentinel)
            y_f = moe_fn(
                act_rot, codes, scales, lut, locs_f,
                self.cfg.hidden_size, self.cfg.intermediate_size, k, g, K, hmult, x_div=1,
            )
            y_dn_total = y_f if y_dn_total is None else (y_dn_total + y_f)

        out = mx.sum(y_dn_total.reshape(N, K_top, H) * w_norm[:, :, None], axis=1)
        return out.reshape(B, T, H)


# ----------------------------------------------------------------------------- Packed Embed & LM Head

class PackedEmbed:
    """Decodes individual token rows on the fly from `embed_packed.npz` (~150 MB RAM)."""

    def __init__(self, packed_dir: Path, luts: dict[str, mx.array], embed_norm_w: mx.array, eps: float):
        self.meta = json.loads((packed_dir / "embed_meta.json").read_text())
        npz = np.load(packed_dir / "embed_packed.npz")
        order = npz["order"]
        self.inv_order = np.empty_like(order)
        self.inv_order[order] = np.arange(len(order), dtype=np.int32)
        self.embed_norm_w = embed_norm_w.astype(mx.float32)
        self.hidden_size = int(self.embed_norm_w.shape[0])
        self.eps = eps
        self.luts_np = {k: np.array(v, dtype=np.float32) for k, v in luts.items()}
        self.buckets: list[dict[str, Any]] = []
        lo = 0
        for bm in self.meta:
            kb = bm["kb"]
            rows = bm["rows"]
            opt = bm["opt"]
            binfo: dict[str, Any] = {"lo": lo, "hi": lo + rows, "opt": opt, "k": bm["k"]}
            binfo["codes"] = np.array(npz[f"b{kb}_codes"])
            binfo["scales"] = np.array(npz[f"b{kb}_scales"], dtype=np.float32)
            binfo["signs_in"] = mx.array(np.array(npz[f"b{kb}_signs_in"], dtype=np.float32))
            if opt.startswith("lm"):
                binfo["levels"] = np.array(npz[f"b{kb}_levels"], dtype=np.float32)
            else:
                binfo["lut"] = self.luts_np[opt]
            self.buckets.append(binfo)
            lo += rows

    def _decode_row(self, pos: int) -> mx.array:
        n = self.hidden_size
        for b in self.buckets:
            if b["lo"] <= pos < b["hi"]:
                r = pos - b["lo"]
                k = b["k"]
                scale = float(b["scales"][r])
                if b["opt"].startswith("lm"):
                    if k == 8:
                        idx = b["codes"].reshape(-1, n)[r]
                    else:  # k == 6
                        bpr = (n * 6) // 8
                        raw = b["codes"][r * bpr : (r + 1) * bpr].reshape(-1, 3).astype(np.uint32)
                        v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
                        idx = np.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], axis=-1).reshape(-1)
                    q = b["levels"][idx] * scale
                else:
                    # RowTrellisCode (g=1, L=12) — vectorized numpy unpack
                    wpr = (n * k) // 32
                    row_u32 = b["codes"][r].astype(np.uint64)
                    bp = (n - 1 - np.arange(n, dtype=np.int64)) * k
                    wi = bp >> 5
                    off = (bp & 31).astype(np.uint64)
                    w0 = row_u32[wi]
                    wi1 = np.minimum(wi + 1, wpr - 1)
                    w1 = np.where(wi + 1 < wpr, row_u32[wi1], np.uint64(0))
                    hi = np.where(off > 0, w1 << (np.uint64(32) - off), np.uint64(0))
                    s = ((w0 >> off) | hi) & np.uint64(4095)
                    q = b["lut"][s] * scale
                return rht_apply_out(mx.array(q[None, :]), b["signs_in"])[0]
        raise IndexError(f"Token position {pos} out of range")

    def __call__(self, input_ids: list[int] | np.ndarray) -> mx.array:
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        rows = [self._decode_row(int(self.inv_order[int(t)])) for t in ids]
        h = mx.stack(rows, axis=0)[None, :, :]  # [1, T, 2304]
        return rms_norm(h, self.embed_norm_w, self.eps)


class PackedLMHead:
    """Executes unpadded `[200058, 2304]` lm_head directly from `head_packed.npz` (~176 MB RAM)."""

    def __init__(self, packed_dir: Path, luts: dict[str, mx.array], cap: float):
        self.meta = json.loads((packed_dir / "head_meta.json").read_text())
        self.cap = cap
        npz = np.load(packed_dir / "head_packed.npz")
        order = np.array(npz["order"], dtype=np.int32)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order), dtype=np.int32)
        self.inv_order = mx.array(inv)
        self.buckets: list[dict[str, Any]] = []
        for bm in self.meta:
            kb = bm["kb"]
            opt = bm["opt"]
            b: dict[str, Any] = {
                "opt": opt,
                "rows": int(bm["rows"]),
                "k": int(bm["k"]),
                "codes": mx.array(npz[f"b{kb}_codes"]),
                "scales": mx.array(npz[f"b{kb}_scales"]),
                "signs_in": mx.array(npz[f"b{kb}_signs_in"].astype(np.float32)),
            }
            if opt.startswith("lm"):
                b["levels"] = mx.array(npz[f"b{kb}_levels"].astype(np.float32))
            else:
                b["g"] = int(bm["g"])
                b["K"] = int(bm["K"])
                b["hash_mult"] = int(bm["hash_mult"])
                b["lut"] = luts[opt]
            self.buckets.append(b)

    def __call__(self, h: mx.array) -> mx.array:
        # h: [B, T, 2304]
        B, T, H = h.shape
        x2 = h.reshape(B * T, H)
        parts = []
        for b in self.buckets:
            x_rot = rht_apply_in(x2, b["signs_in"])
            if b["opt"].startswith("lm"):
                y = lm_dense_matmul(x_rot, b["codes"], b["scales"], b["levels"], b["rows"], H, b["k"])
            else:
                y = trellis_dense_matmul(
                    x_rot, b["codes"], b["scales"], b["lut"],
                    b["rows"], H, b["k"], b["g"], b["K"], b["hash_mult"],
                )
            parts.append(y)
        ordered = mx.concatenate(parts, axis=-1)  # [B*T, 200058]
        logits = ordered[:, self.inv_order]
        return softcap(logits, self.cap).reshape(B, T, -1)


# ----------------------------------------------------------------------------- Decoder Layer & Full Looped Model

@mx.compile
def _block_attn_res_compiled(q: mx.array, stacked: mx.array, inv_sqrt_h: mx.array) -> mx.array:
    st_n = mx.fast.rms_norm(stacked, None, 1e-6)
    scores = mx.einsum("qh,btsh->btqs", q, st_n) * inv_sqrt_h
    w = mx.softmax(scores, axis=-1)
    return mx.einsum("btqs,btsh->btqh", w, st_n)


def block_attn_res(query: mx.array, block_states: list[mx.array], partial: mx.array) -> tuple[mx.array, ...]:
    # block_states: list of [B, T, H], partial: [B, T, H], query: [H] or [num_q, H]
    stacked = mx.stack(block_states + [partial.astype(mx.float32)], axis=-2)  # [B, T, S, H]
    H = stacked.shape[-1]
    q = query.astype(mx.float32).reshape(-1, H)  # [num_q, H]
    out = _block_attn_res_compiled(q, stacked, _f32(H ** -0.5))  # [B, T, num_q, H]
    return tuple(out[:, :, i, :] for i in range(q.shape[0]))


@mx.compile
def _message_gate_compiled(h: mx.array, msg: mx.array, down_w: mx.array, forget_w: mx.array, msg_w: mx.array) -> mx.array:
    cat = mx.concatenate([mx.fast.rms_norm(h, None, 1e-6), mx.fast.rms_norm(msg, None, 1e-6)], axis=-1).astype(down_w.dtype)
    g = cat @ down_w.T
    fg = mx.tanh((g @ forget_w.T).astype(mx.float32))
    mg = mx.tanh((g @ msg_w.T).astype(mx.float32))
    return (1.0 - fg) * h + mg * msg


def message_gate(h: mx.array, msg: mx.array, down_w: mx.array, forget_w: mx.array, msg_w: mx.array, eps: float) -> mx.array:
    return _message_gate_compiled(h.astype(mx.float32), msg.astype(mx.float32), down_w, forget_w, msg_w)


@mx.compile
def _short_conv4_t1(ctx: mx.array, res: mx.array, w_kc: mx.array, inv_s: mx.array) -> mx.array:
    # ctx: [B, 4, C], w_kc: [4, C] -> [B, 1, C]
    return (mx.sum(ctx * w_kc[None, :, :], axis=1, keepdims=True) + res) * inv_s


@mx.compile
def _short_conv4_tn(ctx: mx.array, res: mx.array, w_kc: mx.array, inv_s: mx.array) -> mx.array:
    # ctx: [B, T + 3, C], res: [B, T, C], w_kc: [4, C]
    T = res.shape[1]
    out = (
        ctx[:, 0:T, :] * w_kc[None, 0:1, :]
        + ctx[:, 1 : T + 1, :] * w_kc[None, 1:2, :]
        + ctx[:, 2 : T + 2, :] * w_kc[None, 2:3, :]
        + ctx[:, 3 : T + 3, :] * w_kc[None, 3:4, :]
    )
    return (out + res) * inv_s


def short_conv(x: mx.array, w_kc: mx.array, cache: Inkling2Cache, slot: int, conv_idx: int) -> mx.array:
    # x: [B, T, C], w_kc: [K, C] (w_kc[0] oldest tap, w_kc[K-1] current tap!)
    K = w_kc.shape[0]
    xf = x.astype(mx.float32)
    ctx = cache.update_conv(slot, conv_idx, xf, K)  # [B, T + K - 1, C]
    inv_s = _f32(1.0 / math.sqrt(K + 1))
    if K == 4:
        return _short_conv4_t1(ctx, xf, w_kc, inv_s) if xf.shape[1] == 1 else _short_conv4_tn(ctx, xf, w_kc, inv_s)
    T = xf.shape[1]
    acc = mx.zeros_like(xf)
    for lag in range(K):
        acc = acc + ctx[:, lag : lag + T, :] * w_kc[None, lag : lag + 1, :]
    return (acc + xf) * inv_s


@mx.compile
def _dyn_conv4_compiled(ctx: mx.array, res: mx.array, dyn_in: mx.array, st_hk: mx.array, proj_w: mx.array, inv_s: mx.array) -> mx.array:
    # ctx: [B, T+3, C], res: [B, T, C], st_hk: [1, 1, num_heads, head_dim, 4]
    B, T, C = res.shape
    num_heads = st_hk.shape[2]
    head_dim = st_hk.shape[3]
    dyn = (dyn_in.astype(mx.float32) @ proj_w.T).reshape(B, T, num_heads, 1, 4)
    w = st_hk + dyn  # [B, T, num_heads, head_dim, 4]
    c0 = ctx[:, 3 : T + 3, :].reshape(B, T, num_heads, head_dim)
    c1 = ctx[:, 2 : T + 2, :].reshape(B, T, num_heads, head_dim)
    c2 = ctx[:, 1 : T + 1, :].reshape(B, T, num_heads, head_dim)
    c3 = ctx[:, 0 : T, :].reshape(B, T, num_heads, head_dim)
    out = (c0 * w[..., 0] + c1 * w[..., 1] + c2 * w[..., 2] + c3 * w[..., 3]).reshape(B, T, C)
    return (out + res) * inv_s


def dynamic_short_conv(
    x: mx.array,
    dyn_in: mx.array,
    static_w: mx.array,
    proj_w: mx.array,
    num_heads: int,
    cache: Inkling2Cache,
    slot: int,
    conv_idx: int,
) -> mx.array:
    # static_w: [K, C] where static_w[0] weights current token (lag=0), static_w[lag] weights `lag` steps back!
    B, T, C = x.shape
    K = static_w.shape[0]
    head_dim = C // num_heads
    xf = x.astype(mx.float32)
    ctx = cache.update_conv(slot, conv_idx, xf, K)  # [B, T + K - 1, C]
    inv_s = _f32(1.0 / math.sqrt(2 * K + 1))
    st = static_w.T.reshape(1, 1, num_heads, head_dim, K)
    if K == 4:
        return _dyn_conv4_compiled(ctx, xf, dyn_in, st, proj_w, inv_s)
    dyn = (dyn_in.astype(mx.float32) @ proj_w.T).reshape(B, T, num_heads, 1, K)
    weights = st + dyn
    acc = mx.zeros((B, T, num_heads, head_dim), dtype=mx.float32)
    for lag in range(K):
        sl = ctx[:, (K - 1 - lag) : (K - 1 - lag + T), :].reshape(B, T, num_heads, head_dim)
        acc = acc + sl * weights[:, :, :, :, lag]
    return (acc.reshape(B, T, C) + xf) * inv_s


@mx.compile
def _gqa_sdpa_sinks_compiled(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    pos_bias: mx.array,
    allowed: mx.array,
    sinks: mx.array,
    inv_sqrt_d: mx.array,
) -> mx.array:
    # Zero-copy GQA without mx.repeat: q [B, n_kv, n_rep, T, D], k/v [B, n_kv, 1, kv_len, D]
    scores = mx.matmul(q, k.transpose(0, 1, 2, 4, 3)) * inv_sqrt_d + pos_bias
    scores = mx.where(allowed, scores, -1e9)
    combined = mx.concatenate([scores, sinks], axis=-1)
    probs = mx.softmax(combined, axis=-1)[..., :-1]
    return mx.matmul(probs, v)


def _gqa_sdpa_sinks(
    q: mx.array,
    K_all: mx.array,
    V_all: mx.array,
    pos_bias: mx.array,
    allowed: mx.array,
    sinks: mx.array,
    n_kv: int,
    n_rep: int,
    head_dim: int,
) -> mx.array:
    B, _, T, D = q.shape
    kv_len = K_all.shape[2]
    qg = q.reshape(B, n_kv, n_rep, T, D)
    kg = K_all[:, :, None, :, :]
    vg = V_all[:, :, None, :, :]
    pb = pos_bias.reshape(B, n_kv, n_rep, T, kv_len)
    al = allowed[None, None, None, :, :]
    sk = sinks.reshape(B, n_kv, n_rep, T, 1)
    out = _gqa_sdpa_sinks_compiled(qg, kg, vg, pb, al, sk, _f32(head_dim ** -0.5))
    return out.reshape(B, n_kv * n_rep, T, D).transpose(0, 2, 1, 3).reshape(B, T, n_kv * n_rep * D)


class PackedDecoderLayer:
    def __init__(self, layer_idx: int, cfg: Inkling2Config, packed_dir: Path, luts: dict[str, mx.array], st: dict[str, mx.array]):
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.is_sliding = layer_idx in cfg.local_layer_ids
        self.is_loop = cfg.loop_start <= layer_idx < cfg.loop_end
        self.is_block_boundary = (layer_idx % cfg.n_layers_per_attn_res_block) == 0

        p = f"model.layers.{layer_idx}."
        meta = json.loads((packed_dir / f"layer_{layer_idx:02d}_meta.json").read_text())
        npz = np.load(packed_dir / f"layer_{layer_idx:02d}_packed.npz")
        arrays = {k: npz[k] for k in npz.files}

        # Packed attention linears
        lin_meta = meta["linears"]
        self.q_lin = PackedLinear(lin_meta["self_attn.q_proj.linear"], arrays, luts)
        self.k_lin = PackedLinear(lin_meta["self_attn.k_proj.linear"], arrays, luts)
        self.v_lin = PackedLinear(lin_meta["self_attn.v_proj.linear"], arrays, luts)
        self.o_lin = PackedLinear(lin_meta["self_attn.o_proj.linear"], arrays, luts)
        self.o_gate = PackedLinear(lin_meta["self_attn.o_proj.gate_proj"], arrays, luts)
        self.rel_lin = PackedLinear(lin_meta["self_attn.relative_position_bias.linear"], arrays, luts)

        # Native steering weights (pre-cast to FP32 once at load to eliminate ~960 cast kernels/token)
        self.attn_in_q = st[p + "attention_input.query"].astype(mx.float32) if (p + "attention_input.query") in st else None
        self.mlp_in_q = st[p + "mlp_input.query"].astype(mx.float32)
        if self.is_sliding:
            self.in_ln = st[p + "input_layernorm.weight"].astype(mx.float32)
        else:
            self.q_in_ln = st[p + "q_input_layernorm.weight"].astype(mx.float32)
            self.k_in_ln = st[p + "k_input_layernorm.weight"].astype(mx.float32)
            self.v_in_ln = st[p + "v_input_layernorm.weight"].astype(mx.float32)
        self.post_attn_ln = st[p + "post_attention_layernorm.weight"].astype(mx.float32)

        self.q_scale = st[p + "self_attn.q_proj.norm.scale"].astype(mx.float32)
        self.rel_scale = st[p + "self_attn.relative_position_bias.norm.scale"].astype(mx.float32)
        self.rel_proj = st[p + "self_attn.relative_position_bias.proj"].astype(mx.float32)
        self.sink_w = st[p + "self_attn.sink_proj.weight"].astype(mx.float32)

        if self.is_sliding:
            self.k_conv_w = st[p + "self_attn.k_proj.sconv.conv1d.weight"][:, 0, :].T.astype(mx.float32)
            self.v_conv_w = st[p + "self_attn.v_proj.sconv.conv1d.weight"][:, 0, :].T.astype(mx.float32)
        else:
            self.k_dyn_st = st[p + "self_attn.k_proj.sconv.static_weight"].astype(mx.float32)
            self.k_dyn_pr = st[p + "self_attn.k_proj.sconv.weight_proj.weight"].astype(mx.float32)
            self.v_dyn_st = st[p + "self_attn.v_proj.sconv.static_weight"].astype(mx.float32)
            self.v_dyn_pr = st[p + "self_attn.v_proj.sconv.weight_proj.weight"].astype(mx.float32)

        self.post_attn_conv_w = st[p + "post_attention_convolution.conv1d.weight"][:, 0, :].T.astype(mx.float32)
        self.post_mlp_conv_w = st[p + "post_mlp_convolution.conv1d.weight"][:, 0, :].T.astype(mx.float32)

        if self.is_loop:
            self.msg_gates = {}
            for gn in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                # Keep 2D message gate weights in native bfloat16 for 2x bandwidth + Neural Accelerator speed
                self.msg_gates[gn] = (
                    st[p + f"{gn}.down_proj.weight"].astype(mx.bfloat16),
                    st[p + f"{gn}.forget_gate_proj.weight"].astype(mx.bfloat16),
                    st[p + f"{gn}.message_gate_proj.weight"].astype(mx.bfloat16),
                )

        self.moe = PackedMoE(
            arrays, luts,
            st[p + "mlp.gate.weight"],
            st[p + "mlp.gate.e_score_correction_bias"],
            cfg,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: list[mx.array],
        position_ids: mx.array,
        cache: Inkling2Cache,
        cache_slot: int,
        messages: tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
    ) -> mx.array:
        cfg = self.cfg
        B, T, H = hidden_states.shape
        block_residual = hidden_states

        if self.is_sliding:
            h = block_attn_res(self.attn_in_q, block_states, hidden_states)[0] if self.attn_in_q is not None else hidden_states
            if messages is None:
                q_in = k_in = v_in = rms_norm(h, self.in_ln, cfg.rms_norm_eps)
            else:
                q_m, k_m, v_m, _ = messages
                q_in = rms_norm(message_gate(h, q_m, *self.msg_gates["q_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
                k_in = rms_norm(message_gate(h, k_m, *self.msg_gates["k_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
                v_in = rms_norm(message_gate(h, v_m, *self.msg_gates["v_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
        else:
            if self.attn_in_q is None:
                q_h = k_h = v_h = hidden_states
            else:
                q_h, k_h, v_h = block_attn_res(self.attn_in_q, block_states, hidden_states)
            if messages is not None:
                q_m, k_m, v_m, _ = messages
                q_h = message_gate(q_h, q_m, *self.msg_gates["q_message_gate"], cfg.rms_norm_eps)
                k_h = message_gate(k_h, k_m, *self.msg_gates["k_message_gate"], cfg.rms_norm_eps)
                v_h = message_gate(v_h, v_m, *self.msg_gates["v_message_gate"], cfg.rms_norm_eps)
            q_in = rms_norm(q_h, self.q_in_ln, cfg.rms_norm_eps)
            k_in = rms_norm(k_h, self.k_in_ln, cfg.rms_norm_eps)
            v_in = rms_norm(v_h, self.v_in_ln, cfg.rms_norm_eps)

        # Q, K, V projections
        q = self.q_lin(q_in).reshape(B, T, cfg.num_attention_heads, cfg.head_dim)
        q = l2_clip(q * self.q_scale, cfg.qk_l2_clip_threshold)
        if not self.is_sliding:
            q = log_scale(q, position_ids, cfg.log_scaling_alpha, cfg.log_scaling_n_floor)
        q = q.transpose(0, 2, 1, 3)  # [B, 24, T, 128]

        k_raw = self.k_lin(k_in)
        v_raw = self.v_lin(v_in)
        if self.is_sliding:
            k_raw = short_conv(k_raw, self.k_conv_w, cache, cache_slot, 0)
            v_raw = short_conv(v_raw, self.v_conv_w, cache, cache_slot, 1)
        else:
            k_raw = dynamic_short_conv(k_raw, k_in, self.k_dyn_st, self.k_dyn_pr, cfg.num_key_value_heads, cache, cache_slot, 0)
            v_raw = dynamic_short_conv(v_raw, v_in, self.v_dyn_st, self.v_dyn_pr, cfg.num_key_value_heads, cache, cache_slot, 1)

        k = l2_clip(k_raw.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim), cfg.qk_l2_clip_threshold).transpose(0, 2, 1, 3)
        v = unweighted_rms_norm(v_raw.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim), cfg.rms_norm_eps).transpose(0, 2, 1, 3)

        K_all, V_all = cache.update_kv(cache_slot, k, v, self.is_sliding)
        kv_len = K_all.shape[2]

        # Relative position bias
        rel_st = self.rel_lin(q_in).reshape(B, T, cfg.num_attention_heads, cfg.d_rel)
        rel_st = l2_clip(rel_st * self.rel_scale, cfg.rel_l2_clip_threshold)
        if not self.is_sliding:
            rel_st = log_scale(rel_st, position_ids, cfg.log_scaling_alpha, cfg.log_scaling_n_floor)
        rel_logits = (rel_st @ self.rel_proj).transpose(0, 2, 1, 3)  # [B, 24, T, rel_extent]

        # Exact causal relative distance [T, kv_len] (zero CPU sync!)
        q_pos = position_ids[0]  # [T]
        kv_end = cache.seq_len + T
        kv_start = kv_end - kv_len
        kv_pos = mx.arange(kv_start, kv_end, dtype=mx.int32)
        rel_dist = q_pos[:, None] - kv_pos[None, :]  # [T, kv_len]
        valid_rel = (rel_dist >= 0) & (rel_dist < cfg.rel_extent)
        g_idx = mx.clip(rel_dist, 0, cfg.rel_extent - 1)
        pos_bias = mx.where(valid_rel[None, None, :, :], rel_logits[:, :, mx.arange(T)[:, None], g_idx], 0.0)

        # Causal (+ sliding window) mask + zero-copy GQA SDPA with per-token sinks
        if self.is_sliding:
            allowed = (rel_dist >= 0) & (rel_dist < cfg.sliding_window_size)
        else:
            allowed = rel_dist >= 0
        sinks = (q_in @ self.sink_w.T).transpose(0, 2, 1)[..., None]  # [B, 24, T, 1]
        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        attn_out = _gqa_sdpa_sinks(
            q, K_all, V_all, pos_bias, allowed, sinks,
            cfg.num_key_value_heads, n_rep, cfg.head_dim,
        )

        # Gated output projection + post-attention conv
        gate = mx.sigmoid(self.o_gate(q_in))
        attn_out = self.o_lin(attn_out * gate)
        attn_out = short_conv(attn_out, self.post_attn_conv_w, cache, cache_slot, 2)

        # Residual & block boundary
        if self.is_block_boundary:
            local_blocks = block_states + [block_residual.astype(mx.float32)]
            residual = attn_out
            h_post = attn_out
        else:
            local_blocks = block_states
            residual = block_residual.astype(mx.float32) + attn_out
            h_post = residual

        # MLP input AttnRes + optional mlp message gate + layernorm
        h_mlp = block_attn_res(self.mlp_in_q, local_blocks, h_post)[0]
        if messages is not None:
            h_mlp = message_gate(h_mlp, messages[3], *self.msg_gates["mlp_message_gate"], cfg.rms_norm_eps)
        h_mlp = rms_norm(h_mlp, self.post_attn_ln, cfg.rms_norm_eps)

        # MoE + post-MLP conv + residual
        mlp_out = self.moe(h_mlp)
        mlp_out = short_conv(mlp_out, self.post_mlp_conv_w, cache, cache_slot, 3)
        return residual + mlp_out


class Inkling2LoopPackedMLX:
    """Complete standalone Inkling-2-Mini text model running directly from packed 8.85 GB artifacts on Apple Silicon."""

    def __init__(self, packed_dir: str | Path, steering_override: str | Path | None = None):
        self.packed_dir = Path(packed_dir)
        self.cfg = Inkling2Config.from_json(self.packed_dir / "config.json")
        lut_npz = np.load(self.packed_dir / "luts.npz")
        self.luts = {k: mx.array(lut_npz[k]) for k in lut_npz.files}
        st_raw = mx.load(str(self.packed_dir / "steering.safetensors"))
        if steering_override is not None:
            ov = mx.load(str(steering_override))
            st_raw.update(ov)
        self.st = st_raw

        self.embed = PackedEmbed(self.packed_dir, self.luts, self.st["model.embed_tokens.embed_norm.weight"], self.cfg.rms_norm_eps)
        self.final_in_q = self.st["model.final_input.query"].astype(mx.float32)
        self.final_norm_w = self.st["model.norm.weight"].astype(mx.float32)
        self.message_in_q = self.st["model.message_input.query"].astype(mx.float32)  # [12, 4, 2304]
        self.layers = [PackedDecoderLayer(i, self.cfg, self.packed_dir, self.luts, self.st) for i in range(self.cfg.num_hidden_layers)]
        self.lm_head = PackedLMHead(self.packed_dir, self.luts, self.cfg.final_logit_softcapping)

    def make_cache(self) -> Inkling2Cache:
        return Inkling2Cache(self.cfg)

    def _forward_core(
        self,
        input_ids: list[int] | np.ndarray,
        cache: Inkling2Cache | None = None,
        all_logits: bool = False,
        tap_visits: tuple[int, ...] = (),
    ):
        if cache is None:
            cache = self.make_cache()
        cfg = self.cfg
        h = self.embed(input_ids)  # [1, T, H]
        T = h.shape[1]
        position_ids = mx.arange(cache.seq_len, cache.seq_len + T, dtype=mx.int32)[None, :]
        block_states: list[mx.array] = []
        tap_set = set(tap_visits)
        captured: dict[int, mx.array] = {}
        visit = 0

        # 1. Pre-loop layers (L0..5)
        for i in range(cfg.loop_start):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        pre_loop_h = h
        pre_loop_blocks = list(block_states)
        num_pre = len(pre_loop_blocks)

        # 2. Pass 0 over loop layers (L6..17)
        for i in range(cfg.loop_start, cfg.loop_end):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        # 3. Passes 1..loop_passes-1 over loop layers with cross-pass messages
        num_loop = cfg.loop_end - cfg.loop_start
        for loop_idx in range(1, cfg.loop_passes):
            prev_blocks = block_states[num_pre + 1 :]
            prev_h = h
            h = pre_loop_h
            block_states = list(pre_loop_blocks)
            cache_offset = cfg.num_hidden_layers + (loop_idx - 1) * num_loop - cfg.loop_start
            for loop_layer_idx, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
                if visit in tap_set:
                    captured[visit] = h[0].astype(mx.float32)
                L = self.layers[i]
                msgs = block_attn_res(self.message_in_q[loop_layer_idx], prev_blocks, prev_h)
                new_h = L(h, block_states, position_ids, cache, i + cache_offset, messages=msgs)
                if L.is_block_boundary:
                    block_states = block_states + [h.astype(mx.float32)]
                h = new_h
                visit += 1

        # 4. Post-loop layers (L18..23)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        # 5. Final AttnRes + norm + unpadded lm_head + softcap
        h = block_attn_res(self.final_in_q, block_states, h)[0]
        h = rms_norm(h, self.final_norm_w, cfg.rms_norm_eps)
        if visit in tap_set:
            captured[visit] = h[0].astype(mx.float32)
        cache.seq_len += T
        logits = self.lm_head(h if all_logits else h[:, -1:, :])
        if tap_visits:
            taps = mx.stack([captured[v] for v in tap_visits], axis=1)  # [T, len(taps), H]
            return logits, cache, taps
        return logits, cache

    def forward(self, input_ids: list[int] | np.ndarray, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = ()):
        return self._forward_core(input_ids, cache=cache, all_logits=False, tap_visits=tap_visits)

    def forward_all_logits(self, input_ids: list[int] | np.ndarray, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = ()):
        return self._forward_core(input_ids, cache=cache, all_logits=True, tap_visits=tap_visits)


class Inkling2MLXModel:
    """Thin `mlx_lm`-compatible adapter around `Inkling2LoopPackedMLX`."""

    def __init__(self, inner: Inkling2LoopPackedMLX):
        self.inner = inner
        self.cfg = inner.cfg
        self.layers = inner.layers

    def make_cache(self) -> list[Inkling2Cache]:
        return [self.inner.make_cache()]

    def __call__(self, inputs: mx.array, cache: list[Inkling2Cache] | Inkling2Cache | None = None) -> mx.array:
        c = cache[0] if isinstance(cache, list) and cache else cache
        logits, _ = self.inner.forward(inputs, cache=c)
        return logits


def load_inkling(packed_dir: str | Path, steering_override: str | Path | None = None):
    """Load packed Inkling-2-Mini as an `(mlx_lm-compatible model, tokenizer)` pair."""
    from mlx_lm.utils import load_tokenizer

    path = Path(packed_dir)
    inner = Inkling2LoopPackedMLX(path, steering_override=steering_override)
    tok = load_tokenizer(path)
    return Inkling2MLXModel(inner), tok
