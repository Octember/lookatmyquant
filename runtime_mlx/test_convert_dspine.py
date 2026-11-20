"""convert_dspine.py against Tencent's own MLX model: a small random checkpoint in Tencent's layout, run by their
DSpineDraftModel and, converted, by drafter_q35.Drafter, on the same context and block.

    python test_convert_dspine.py            (skips if bench_models/ref/Tencent-DSpine is not there)

The reference is github.com/Tencent/DSpine's dspine_mlx (model.py, bridge.py, config.py: MLX arithmetic only).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "bench_models/ref/Tencent-DSpine/dspine_mlx"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import mlx.core as mx

import convert_dspine
import drafter_q35

H, HEADS, KV, HD, MID, CODE, RANK, V, LAYERS, TAPS, MASK = 64, 4, 2, 16, 128, 32, 16, 96, 3, (1, 4, 7), 95


def tencent_config(window=8192):
    return {"architectures": ["DFlashDraftModel"], "attention_bias": False, "block_size": 16, "draft_vocab_size": V, "vocab_size": V,
            "code_bridge_second_pass": {"first_exact": LAYERS - 1, "head_topk": 16, "mode": "serial_table"},
            "dflash_config": {"draft_block_causal": True, "mask_token_id": MASK, "selected_prefix_code_bridge": {"code_size": CODE, "eps": 1e-06, "kappa": 10.0, "rank": RANK},
                              "selected_prefix_decode_mode": "serial", "selected_prefix_feature_layers": list(range(1, LAYERS + 1)),
                              "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
                              "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
                              "selected_prefix_terminal": False, "target_layer_ids": list(TAPS)},
            "head_dim": HD, "hidden_act": "silu", "hidden_size": H, "intermediate_size": MID, "is_causal": True,
            "layer_types": ["sliding_attention"] * LAYERS, "max_position_embeddings": 4096, "max_window_layers": LAYERS, "model_type": "qwen3",
            "num_attention_heads": HEADS, "num_hidden_layers": LAYERS, "num_key_value_heads": KV, "num_target_layers": 8, "rms_norm_eps": 1e-06,
            "rope_theta": 10000000, "rope_scaling": None, "sliding_window": window, "tie_word_embeddings": False, "use_sliding_window": True}


def tencent_weights():
    mx.random.seed(7)
    r = lambda *shape, s=0.2: (s * mx.random.normal(shape)).astype(mx.bfloat16)
    one = lambda n: (1.0 + 0.1 * mx.random.normal((n,))).astype(mx.bfloat16)
    w = {"fc.weight": r(H, len(TAPS) * H), "hidden_norm.weight": one(H), "norm.weight": one(H),
         "draft_block_causal_version": mx.array(1, dtype=mx.int64), "last_write_refine_version": mx.array(1, dtype=mx.int64)}
    for i in range(LAYERS):
        p = f"layers.{i}."
        w.update({p + "input_layernorm.weight": one(H), p + "post_attention_layernorm.weight": one(H),
                  p + "self_attn.q_norm.weight": one(HD), p + "self_attn.k_norm.weight": one(HD),
                  p + "self_attn.q_proj.weight": r(HEADS * HD, H), p + "self_attn.k_proj.weight": r(KV * HD, H),
                  p + "self_attn.v_proj.weight": r(KV * HD, H), p + "self_attn.o_proj.weight": r(H, HEADS * HD),
                  p + "mlp.gate_proj.weight": r(MID, H), p + "mlp.up_proj.weight": r(MID, H), p + "mlp.down_proj.weight": r(H, MID)})
    b = "selected_prefix_bridge."
    w.update({b + "codebook": r(V, CODE, s=CODE ** -0.5), b + "read_weight": r(CODE, H), b + "early_read_weights": r(LAYERS - 1, CODE, H),
              b + "feature_write_weights": r(LAYERS, H, RANK), b + "gate.weight": r(RANK, CODE), b + "gate.bias": r(RANK),
              b + "predecessor_gate.weight": r(RANK, CODE), b + "prev.weight": r(RANK, CODE), b + "out.weight": mx.zeros((CODE, RANK), dtype=mx.bfloat16),
              b + "independent_write_version": mx.array(1, dtype=mx.int64), b + "terminal_disabled_version": mx.array(1, dtype=mx.int64)})
    return w


class Table:
    """The target's token table as both models take it."""

    def __init__(self):
        mx.random.seed(11)
        self.weight = mx.random.normal((V, H)).astype(mx.bfloat16)

    def __call__(self, ids):
        return self.weight[ids]


class Stub:
    def __init__(self, table):
        self.table, self.taps, self.embed_tokens = table, TAPS, table


@unittest.skipUnless((REF / "dspine_mlx/model.py").exists(), "Tencent's reference is not at bench_models/ref/Tencent-DSpine")
class Converted(unittest.TestCase):
    def run_both(self, window, context, size, rounds=2):
        sys.path.insert(0, str(REF))
        from dspine_mlx.model import load_draft
        table = Table()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "tencent", Path(tmp) / "ours"
            src.mkdir()
            (src / "config.json").write_text(json.dumps(tencent_config(window)))
            mx.save_safetensors(str(src / "model.safetensors"), tencent_weights())
            convert_dspine.convert(src, dst)
            ref, _ = load_draft(src)
            ref.bind(Stub(table))
            ref.set_block_size(size)
            ours = drafter_q35.Drafter(dst, Stub(table), dtype=mx.float32, int8=False)
            mx.random.seed(3)
            caches, worst = ref.make_cache(), 0.0
            for r in range(rounds):                     # each round: more context, then a block after it
                S = context if r == 0 else 5
                taps = [mx.random.normal((S, H)).astype(mx.bfloat16) for _ in TAPS]
                anchor = 10 + r
                block = mx.array([[anchor] + [MASK] * (size - 1)])
                want, stash = ref(mx.concatenate(taps, axis=-1)[None], block, caches)
                ours.context(taps)
                got, pre = ours.draft(anchor, size)
                a, b = want[0, 1:].astype(mx.float32), got.astype(mx.float32)
                worst = max(worst, float(mx.abs(a - b).max() / mx.abs(a).max()))
                p, q = stash["raw"][1:].astype(mx.float32), pre.astype(mx.float32)
                worst = max(worst, float(mx.abs(p - q).max() / mx.abs(p).max()))
            return worst

    def test_states_match_the_reference(self):
        # the reference rounds to bfloat16 after every step and ours runs in float32 here: the 1 to 2% is that rounding
        self.assertLess(self.run_both(window=8192, context=12, size=16), 5e-2)   # reads 0.016
        self.assertLess(self.run_both(window=8192, context=12, size=8), 5e-2)    # 0.012

    def test_states_match_past_the_sliding_window(self):
        self.assertLess(self.run_both(window=20, context=40, size=16, rounds=3), 5e-2)   # 0.020

    def test_a_wrong_mapping_is_seen(self):
        """The comparison is tight enough to catch a swapped pair of gates."""
        swap = convert_dspine.convert_weights

        def swapped(w, layers):
            out = swap(w, layers)
            out["injection.receiver_gate.weight"], out["injection.predecessor_gate.weight"] = out["injection.predecessor_gate.weight"], out["injection.receiver_gate.weight"]
            return out
        convert_dspine.convert_weights = swapped
        try:
            self.assertGreater(self.run_both(window=8192, context=12, size=16), 0.2)   # 0.51
        finally:
            convert_dspine.convert_weights = swap


if __name__ == "__main__":
    unittest.main()
