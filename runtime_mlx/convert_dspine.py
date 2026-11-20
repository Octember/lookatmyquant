"""Tencent's published DSpine drafter written in the layout drafter_q35.Drafter reads.

Tencent exports the drafter as a DFlashDraftModel whose injection lives under `selected_prefix_bridge.` (their
`dspine_mlx/bridge.py`); drafter_q35 reads SpecForge's DSpineDraftModel export (`injection.`, `transfer_codes`), the form
drafter/ trains. It is one mechanism under two sets of names:

    selected_prefix_bridge.early_read_weights[i], .read_weight   ->  injection.readouts.{i}.weight   (the last layer's is read_weight)
    selected_prefix_bridge.feature_write_weights[i]              ->  injection.writes.{i}.weight
    selected_prefix_bridge.prev.weight                           ->  injection.message.weight
    selected_prefix_bridge.gate.weight, .gate.bias               ->  injection.receiver_gate.weight, injection.bias
    selected_prefix_bridge.predecessor_gate.weight               ->  injection.predecessor_gate.weight
    selected_prefix_bridge.codebook                              ->  transfer_codes

Everything else (fc, the norms, the layers) keeps its name. Tencent's bridge divides a state by its RMS before the read
and ours reads the state as it is; the read is normalized afterwards in both, so the two differ only by the 1e-6 Tencent
adds to each denominator. Its `out` projection is the disabled terminal head and must be zero.

    python convert_dspine.py TENCENT_DIR OUT_DIR

test_convert_dspine.py checks the result against Tencent's own MLX model. convert_weights takes MLX arrays or torch
tensors, so the same renaming can run where MLX is not installed.
"""
import json
import sys
from pathlib import Path

BRIDGE = "selected_prefix_bridge."
FLAGS = ("draft_block_causal_version", "last_write_refine_version", BRIDGE + "independent_write_version", BRIDGE + "terminal_disabled_version")


def convert_config(c):
    """Tencent's config.json as drafter_q35 reads it; raises where the checkpoint is not the mechanism drafter_q35 runs."""
    d = c["dflash_config"]
    need = {"selected_prefix_terminal": False, "selected_prefix_decode_mode": "serial", "selected_prefix_feature_writes": True,
            "selected_prefix_feature_write_mode": "independent", "draft_block_causal": True}
    for key, value in need.items():
        if d.get(key) != value:
            raise ValueError(f"dflash_config.{key} is {d.get(key)!r}; drafter_q35 runs {value!r} only")
    layers = int(c["num_hidden_layers"])
    if [int(x) for x in d["selected_prefix_feature_layers"]] != list(range(1, layers + 1)):
        raise ValueError("drafter_q35 writes after every layer; this checkpoint does not")
    bridge = d["selected_prefix_code_bridge"]
    out = dict(c)
    out["architectures"] = ["DSpineDraftModel"]
    out["is_causal"] = True
    out["dspine_config"] = {"transfer_rank": int(bridge["code_size"]), "message_size": int(bridge["rank"]),
                            "top_k": int(d["selected_prefix_last_write_refine"]["head_topk"])}
    out["converted_from"] = "Tencent DSpine export (selected_prefix_bridge.*), by convert_dspine.py"
    return out


def convert_weights(w, layers):
    """Tencent's tensors by name -> drafter_q35's."""
    w = {k.removeprefix("model."): v for k, v in w.items()}
    for flag in FLAGS:
        if flag in w and int(w[flag].item()) != 1:
            raise ValueError(f"{flag} is {int(w[flag].item())}; only version 1 is known")
    if bool((w[BRIDGE + "out.weight"] != 0).any().item()):
        raise ValueError("the terminal projection is not zero: this checkpoint uses a head drafter_q35 does not have")
    out = {k: v for k, v in w.items() if not k.startswith(BRIDGE) and k not in FLAGS}
    early, writes = w[BRIDGE + "early_read_weights"], w[BRIDGE + "feature_write_weights"]
    assert early.shape[0] == layers - 1 and writes.shape[0] == layers
    for i in range(layers):
        out[f"injection.readouts.{i}.weight"] = early[i] if i < layers - 1 else w[BRIDGE + "read_weight"]
        out[f"injection.writes.{i}.weight"] = writes[i]
    out["injection.message.weight"] = w[BRIDGE + "prev.weight"]
    out["injection.receiver_gate.weight"] = w[BRIDGE + "gate.weight"]
    out["injection.bias"] = w[BRIDGE + "gate.bias"]
    out["injection.predecessor_gate.weight"] = w[BRIDGE + "predecessor_gate.weight"]
    out["transfer_codes"] = w[BRIDGE + "codebook"]
    return out


def convert(src, dst):
    import mlx.core as mx
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    out = convert_weights(weights, int(config["num_hidden_layers"]))
    mx.save_safetensors(str(dst / "model.safetensors"), out)
    (dst / "config.json").write_text(json.dumps(convert_config(config), indent=2))
    return {"tensors": len(out), "weights": int(sum(v.size for v in out.values()))}


if __name__ == "__main__":
    print(json.dumps(convert(sys.argv[1], sys.argv[2])))
