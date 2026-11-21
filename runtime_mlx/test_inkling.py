"""Self-contained Apple Silicon MLX tests for packed Inkling-2-Mini runtime and CLI integration."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_mlx.inkling.drafter import resolve_drafter_files
from runtime_mlx.inkling.kernels import trellis_matvec, trellis_moe_grouped
from runtime_mlx.inkling.model import (
    FMT_PARAMS,
    ID_TO_FMT,
    Inkling2LoopPackedMLX,
    Inkling2MLXModel,
)
from runtime_mlx.runtime import detect_format, inspect_model


def _pack_u32_random(rng, shape_m_wpr):
    return rng.integers(0, 2**32, size=shape_m_wpr, dtype=np.uint32)


def build_mini_packed_dir(out_dir: Path) -> None:
    """Create a miniature 6-layer / 10-visit looped MoE packed Inkling-2-Loop directory exercising every code path."""
    rng = np.random.default_rng(2026)
    H = 256
    I = 256
    n_q, n_kv, hd = 4, 4, 64
    d_rel, rel_ext = 16, 32
    V_emb, V_head = 512, 500
    n_layers = 6
    n_exp = 8
    top_k = 2
    cfg = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": {
            "model_type": "inkling2_loop",
            "hidden_size": H,
            "num_hidden_layers": n_layers,
            "vocab_size": V_emb,
            "unpadded_vocab_size": V_head,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "d_rel": d_rel,
            "rel_extent": rel_ext,
            "log_scaling_n_floor": 4,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": [0, 2, 4],
            "sconv_kernel_size": 4,
            "conv_kernel_size": 4,
            "final_logit_softcapping": 30.0,
            "sliding_window_size": 8,
            "n_layers_per_attn_res_block": 2,
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "intermediate_size": I,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 32},
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    luts = {
        "ht2": rng.standard_normal((512, 4)).astype(np.float16) * 0.5,
        "ht3": rng.standard_normal((1024, 2)).astype(np.float16) * 0.5,
        "rtr1": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr2": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr3": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr4": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
    }
    np.savez(out_dir / "luts.npz", **luts)

    st: dict[str, mx.array] = {
        "model.embed_tokens.embed_norm.weight": mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.final_input.query": mx.array(rng.standard_normal((H,)).astype(np.float32) * 0.1),
        "model.norm.weight": mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.message_input.query": mx.array(rng.standard_normal((2, 4, H)).astype(np.float32) * 0.1),
    }

    emb_order = rng.permutation(V_emb).astype(np.int32)
    emb_meta = [
        {"kb": 0, "opt": "lm8", "rows": 128, "k": 8},
        {"kb": 1, "opt": "rtr2", "rows": V_emb - 128, "k": 2, "g": 1, "K": 4096, "hash_mult": 0},
    ]
    np.savez(
        out_dir / "embed_packed.npz",
        order=emb_order,
        b0_codes=rng.integers(0, 256, size=(128 * H,), dtype=np.uint8),
        b0_scales=rng.uniform(0.01, 0.03, size=(128,)).astype(np.float16),
        b0_levels=np.sort(rng.standard_normal(256)).astype(np.float32),
        b0_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b1_codes=_pack_u32_random(rng, (V_emb - 128, (H * 2) // 32)),
        b1_scales=rng.uniform(0.01, 0.03, size=(V_emb - 128,)).astype(np.float16),
        b1_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
    )
    (out_dir / "embed_meta.json").write_text(json.dumps(emb_meta, indent=2))

    hd_order = rng.permutation(V_head).astype(np.int32)
    hd_meta = [
        {"kb": 0, "opt": "lm8", "rows": 100, "k": 8},
        {"kb": 1, "opt": "lm6", "rows": 150, "k": 6},
        {"kb": 2, "opt": "rtr1", "rows": V_head - 250, "k": 1, "g": 1, "K": 4096, "hash_mult": 0},
    ]
    np.savez(
        out_dir / "head_packed.npz",
        order=hd_order,
        b0_codes=rng.integers(0, 256, size=(100 * H,), dtype=np.uint8),
        b0_scales=rng.uniform(0.01, 0.03, size=(100,)).astype(np.float16),
        b0_levels=np.sort(rng.standard_normal(256)).astype(np.float32),
        b0_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b1_codes=rng.integers(0, 256, size=((150 * H * 6) // 8,), dtype=np.uint8),
        b1_scales=rng.uniform(0.01, 0.03, size=(150,)).astype(np.float16),
        b1_levels=np.sort(rng.standard_normal(64)).astype(np.float32),
        b1_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b2_codes=_pack_u32_random(rng, (V_head - 250, (H * 1) // 32)),
        b2_scales=rng.uniform(0.01, 0.03, size=(V_head - 250,)).astype(np.float16),
        b2_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
    )
    (out_dir / "head_meta.json").write_text(json.dumps(hd_meta, indent=2))

    lin_specs = [
        ("self_attn.q_proj.linear", n_q * hd, H, "lm8"),
        ("self_attn.k_proj.linear", n_kv * hd, H, "lm6"),
        ("self_attn.v_proj.linear", n_kv * hd, H, "rtr4"),
        ("self_attn.o_proj.linear", H, n_q * hd, "ht3"),
        ("self_attn.o_proj.gate_proj", n_q * hd, H, "lm6"),
        ("self_attn.relative_position_bias.linear", n_q * d_rel, H, "ht2"),
    ]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        is_sliding = i in (0, 2, 4)
        is_loop = 2 <= i < 4
        if i > 0:
            st[p + "attention_input.query"] = mx.array(
                rng.standard_normal((H,) if is_sliding else (3, H)).astype(np.float32) * 0.1
            )
        st[p + "mlp_input.query"] = mx.array(rng.standard_normal((H,)).astype(np.float32) * 0.1)
        if is_sliding:
            st[p + "input_layernorm.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        else:
            for ln in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                st[p + f"{ln}.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "post_attention_layernorm.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "self_attn.q_proj.norm.scale"] = mx.array(rng.uniform(0.8, 1.2, size=(n_q, hd)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.norm.scale"] = mx.array(rng.uniform(0.8, 1.2, size=(n_q, d_rel)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.proj"] = mx.array(rng.standard_normal((d_rel, rel_ext)).astype(np.float32) * 0.1)
        st[p + "self_attn.sink_proj.weight"] = mx.array(rng.standard_normal((n_q, H)).astype(np.float32) * 0.05)
        if is_sliding:
            st[p + "self_attn.k_proj.sconv.conv1d.weight"] = mx.array(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.conv1d.weight"] = mx.array(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
        else:
            st[p + "self_attn.k_proj.sconv.static_weight"] = mx.array(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.k_proj.sconv.weight_proj.weight"] = mx.array(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
            st[p + "self_attn.v_proj.sconv.static_weight"] = mx.array(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.weight_proj.weight"] = mx.array(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
        st[p + "post_attention_convolution.conv1d.weight"] = mx.array(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        st[p + "post_mlp_convolution.conv1d.weight"] = mx.array(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        if is_loop:
            for gn in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                st[p + f"{gn}.down_proj.weight"] = mx.array(rng.standard_normal((32, 2 * H)).astype(np.float32) * 0.05)
                st[p + f"{gn}.forget_gate_proj.weight"] = mx.array(rng.standard_normal((H, 32)).astype(np.float32) * 0.05)
                st[p + f"{gn}.message_gate_proj.weight"] = mx.array(rng.standard_normal((H, 32)).astype(np.float32) * 0.05)
        st[p + "mlp.gate.weight"] = mx.array(rng.standard_normal((n_exp, H)).astype(np.float32) * 0.1)
        st[p + "mlp.gate.e_score_correction_bias"] = mx.array(rng.standard_normal((n_exp,)).astype(np.float32) * 0.02)

        l_arrs: dict[str, np.ndarray] = {}
        l_meta = {"layer": i, "linears": {}, "experts": {}}
        for name, m, n, opt in lin_specs:
            safe = name.replace(".", "_")
            l_arrs[f"lin_{safe}_signs_in"] = rng.choice([-1, 1], size=(n,)).astype(np.int8)
            l_arrs[f"lin_{safe}_scales"] = rng.uniform(0.01, 0.03, size=(m,)).astype(np.float16)
            if opt.startswith("lm"):
                k = int(opt[2:])
                l_arrs[f"lin_{safe}_codes"] = rng.integers(0, 256, size=((m * n * k) // 8,), dtype=np.uint8)
                l_arrs[f"lin_{safe}_levels"] = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
                l_arrs[f"lin_{safe}_signs_out"] = rng.choice([-1, 1], size=(m,)).astype(np.int8)
                l_meta["linears"][name] = {"fmt": opt, "m": m, "n": n, "k": k, "out_rot": True, "safe": safe}
            else:
                k, g, K, hmult = FMT_PARAMS[opt]
                l_arrs[f"lin_{safe}_codes"] = _pack_u32_random(rng, (m, (n * k) // 32))
                l_meta["linears"][name] = {"fmt": opt, "m": m, "n": n, "k": k, "g": g, "K": K, "hash_mult": hmult, "safe": safe}

        l_arrs["gu_signs_in"] = rng.choice([-1, 1], size=(H,)).astype(np.int8)
        l_arrs["dn_signs_in"] = rng.choice([-1, 1], size=(I,)).astype(np.int8)
        gu_map = np.zeros((n_exp, 2), dtype=np.int16)
        dn_map = np.zeros((n_exp, 2), dtype=np.int16)
        exp_fmts = ["ht2", "ht3", "rtr1", "rtr4"]
        fmt_to_id = {v: k for k, v in ID_TO_FMT.items()}
        for fi, fmt in enumerate(exp_fmts):
            k, g, K, hmult = FMT_PARAMS[fmt]
            l_arrs[f"gu_{fmt}_codes"] = _pack_u32_random(rng, (2, 2 * I, (H * k) // 32))
            l_arrs[f"gu_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(2, 2 * I)).astype(np.float16)
            l_arrs[f"dn_{fmt}_codes"] = _pack_u32_random(rng, (2, H, (I * k) // 32))
            l_arrs[f"dn_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(2, H)).astype(np.float16)
            for loc in range(2):
                e = fi * 2 + loc
                gu_map[e] = [fmt_to_id[fmt], loc]
                dn_map[e] = [fmt_to_id[fmt], loc]
        l_arrs["gu_map"] = gu_map
        l_arrs["dn_map"] = dn_map
        np.savez(out_dir / f"layer_{i:02d}_packed.npz", **l_arrs)
        (out_dir / f"layer_{i:02d}_meta.json").write_text(json.dumps(l_meta, indent=2))

    mx.save_safetensors(str(out_dir / "steering.safetensors"), st)
    manifest = {
        "format": "inkling2-packed-v1-sqyfr",
        "packed_text_gb": 0.01,
        "steering_tensors": len(st),
        "luts": {k: list(v.shape) for k, v in luts.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_mini_model_and_runtime_integration() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        build_mini_packed_dir(pdir)

        assert detect_format(pdir) == "inkling2-packed"
        info = inspect_model(pdir)
        assert info["format"] == "inkling2-packed"
        assert info["num_hidden_layers"] == 6
        assert info["n_routed_experts"] == 8
        assert info["loop_passes"] == 3
        assert info["separate_lm_head"] is True

        model = Inkling2LoopPackedMLX(pdir)
        toks = [5, 19, 42, 128, 301, 7, 255, 12, 400, 88]

        logits_full, _ = model.forward_all_logits(toks)
        mx.eval(logits_full)
        lf = np.array(logits_full)[0]
        assert np.isfinite(lf).all()

        # Cached prefill(6) + 4 single-token steps via mlx_lm-compatible wrapper
        wrapped = Inkling2MLXModel(model)
        cache_list = wrapped.make_cache()
        l_pre, _ = model.forward_all_logits(toks[:6], cache=cache_list[0])
        mx.eval(l_pre)
        step_logits = [np.array(l_pre)[0]]
        for t in range(6, 10):
            lt = wrapped(mx.array([[toks[t]]], dtype=mx.int32), cache=cache_list)
            mx.eval(lt)
            step_logits.append(np.array(lt)[0])
        lc = np.concatenate(step_logits, axis=0)
        assert float(np.max(np.abs(lf - lc))) < 5e-3
        assert float(np.mean(np.argmax(lf, axis=-1) == np.argmax(lc, axis=-1))) == 1.0
        assert cache_list[0].offset == 10
        assert len(cache_list[0].state) > 0

        # Speculative block-4 verify + rollback (k_accept=2) + 2 steps
        c_spec = model.make_cache()
        l_p, c_spec, taps_p = model.forward_all_logits(toks[:6], cache=c_spec, tap_visits=(1, 5, 7, 10))
        mx.eval(l_p, taps_p)
        assert taps_p.shape == (6, 4, 256)
        c_spec.begin_spec_verify()
        l_blk, c_spec, taps_blk = model.forward_all_logits([toks[6], toks[7], 0, 1], cache=c_spec, tap_visits=(1, 5, 7, 10))
        mx.eval(l_blk, taps_blk)
        c_spec.commit_spec_verify(k_accept=2, block_len=4)
        l_8, c_spec = model.forward([toks[8]], cache=c_spec)
        l_9, c_spec = model.forward([toks[9]], cache=c_spec)
        mx.eval(l_8, l_9)
        l_spec = np.concatenate([np.array(l_p)[0], np.array(l_blk)[0, :2], np.array(l_8)[0], np.array(l_9)[0]], axis=0)
        assert float(np.max(np.abs(lf - l_spec))) < 5e-3

        # Drafter file resolution test
        ddir = pdir / "drafter"
        ddir.mkdir()
        (ddir / "mtp_4bit.safetensors").write_bytes(b"placeholder")
        (ddir / "manifest.json").write_text(json.dumps({"weights": "mtp_4bit.safetensors"}))
        w_p, c_p = resolve_drafter_files(ddir, fallback_config=pdir / "config.json")
        assert w_p == ddir / "mtp_4bit.safetensors"
        assert c_p == pdir / "config.json"


def test_moe_grouped_large_expert_count_parity() -> None:
    """Stress-test deterministic compaction inside `trellis_moe_grouped` with E=96 (>32 experts) and sentinel slots."""
    rng = np.random.default_rng(123)
    E, m, n, k, g, K, hmult = 96, 256, 256, 2, 4, 512, 0x9E3779B1
    codes = mx.array(_pack_u32_random(rng, (E, m, (n * k) // 32)))
    scales = mx.array(rng.uniform(0.01, 0.03, size=(E, m)).astype(np.float16))
    lut = mx.array(rng.standard_normal((K, g)).astype(np.float16) * 0.5)
    T, topk = 8, 8
    x = mx.array(rng.standard_normal((T, n)).astype(np.float32) * 0.2)
    # Mix valid expert IDs across [0..95] and sentinel 0xFFFFFFFF slots
    lidx_np = rng.integers(0, E, size=(T * topk,), dtype=np.uint32)
    lidx_np[::7] = 0xFFFFFFFF
    lidx = mx.array(lidx_np)

    y_grouped = trellis_moe_grouped(x, codes, scales, lut, lidx, m, n, k, g, K, hmult, x_div=topk)
    mx.eval(y_grouped)
    yg = np.array(y_grouped)

    # Exact slot-by-slot reference via `trellis_matvec`
    for s in range(T * topk):
        eid = int(lidx_np[s])
        if eid == 0xFFFFFFFF:
            assert np.allclose(yg[s], 0.0, atol=1e-6)
        else:
            ref = trellis_matvec(
                x[s // topk : s // topk + 1], codes, scales, lut, mx.array([eid], dtype=mx.uint32), m, n, k, g, K, hmult
            )[0]
            mx.eval(ref)
            err = float(np.max(np.abs(yg[s] - np.array(ref))))
            assert err < 2e-3, f"Slot {s} expert {eid} mismatch: {err}"


def main() -> None:
    test_mini_model_and_runtime_integration()
    test_moe_grouped_large_expert_count_parity()
    print("ALL INKLING MLX RUNTIME TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
