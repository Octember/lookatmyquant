"""Unit tests for proximal-gap equal-byte row reallocation (finetune.py) and Softmax-Fisher token ranking (quantize.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune import solve_equal_byte_reallocation
from quantize import softmax_fisher, token_importance


class TestReallocation(unittest.TestCase):
    def _fixture(self):
        g = torch.Generator().manual_seed(0)
        old_k = {
            "model.layers.0.mlp.down_proj.weight": torch.full((32,), 3, dtype=torch.int64),
            "model.layers.0.mlp.up_proj.weight": torch.full((16,), 3, dtype=torch.int64),
            "model.layers.1.mlp.down_proj.weight": torch.full((32,), 2, dtype=torch.int64),
            "model.layers.1.mlp.up_proj.weight": torch.full((16,), 2, dtype=torch.int64),
        }
        raw_s = {k: torch.rand(len(v), generator=g, dtype=torch.float64) + 0.1 for k, v in old_k.items()}
        cols = {
            "model.layers.0.mlp.down_proj.weight": 64,
            "model.layers.0.mlp.up_proj.weight": 128,
            "model.layers.1.mlp.down_proj.weight": 64,
            "model.layers.1.mlp.up_proj.weight": 128,
        }
        layers = {k: int(k.split(".layers.")[1].split(".")[0]) for k in old_k}
        return old_k, raw_s, cols, layers

    def test_equal_bytes_and_depth_shift(self):
        old_k, raw_s, cols, layers = self._fixture()
        gap_kl = {0: 0.01, 1: 0.10}  # deep layer hurts 10x more than early layer
        new_k, stats = solve_equal_byte_reallocation(old_k, raw_s, cols, layers, gap_kl, max_delta=1)

        self.assertLessEqual(stats["new_bits"], stats["base_bits"])
        self.assertEqual(stats["new_bits"], stats["base_bits"])  # divisible column widths fill exact budget
        self.assertGreater(stats["rows_changed"], 0)
        for name in old_k:
            diff = (new_k[name] - old_k[name]).abs()
            self.assertLessEqual(int(diff.max()), 1)
            self.assertGreaterEqual(int(new_k[name].min()), 1)
            self.assertLessEqual(int(new_k[name].max()), 4)

        l0_before = sum(int(old_k[n].sum()) * cols[n] for n in old_k if layers[n] == 0)
        l0_after = sum(int(new_k[n].sum()) * cols[n] for n in old_k if layers[n] == 0)
        l1_before = sum(int(old_k[n].sum()) * cols[n] for n in old_k if layers[n] == 1)
        l1_after = sum(int(new_k[n].sum()) * cols[n] for n in old_k if layers[n] == 1)
        self.assertLess(l0_after, l0_before)
        self.assertGreater(l1_after, l1_before)

    def test_invariant_to_jacobian_layer_scaling(self):
        old_k, raw_s, cols, layers = self._fixture()
        gap_kl = {0: 0.02, 1: 0.08}
        ref_k, _ = solve_equal_byte_reallocation(old_k, raw_s, cols, layers, gap_kl, max_delta=1)

        # Scaling layer 0's raw backward Fisher by 10,000x must not change a single row's bit width.
        skewed_s = {k: (v * 1e4 if layers[k] == 0 else v) for k, v in raw_s.items()}
        got_k, _ = solve_equal_byte_reallocation(old_k, skewed_s, cols, layers, gap_kl, max_delta=1)
        for name in old_k:
            self.assertTrue(torch.equal(ref_k[name], got_k[name]), f"mismatch on {name}")

    def test_nonpositive_gap_floor_and_uncapped(self):
        old_k, raw_s, cols, layers = self._fixture()
        new_k, stats = solve_equal_byte_reallocation(old_k, raw_s, cols, layers, {0: -0.005, 1: 0.05}, max_delta=0)
        self.assertLessEqual(stats["new_bits"], stats["base_bits"])
        # Uncapped allows a 2-bit jump on layer 1 (from 2 to 4) while dropping layer 0 to 1.
        self.assertEqual(int(new_k["model.layers.1.mlp.down_proj.weight"].max()), 4)
        self.assertEqual(int(new_k["model.layers.0.mlp.down_proj.weight"].min()), 1)


class _TinyLM(nn.Module):
    def __init__(self, logits_table: torch.Tensor):
        super().__init__()
        V, D = logits_table.shape[1], 4
        self.head = nn.Linear(D, V, bias=False)
        self._table = logits_table

    def get_output_embeddings(self):
        return self.head

    def forward(self, x, use_cache=False):
        out = self._table[x]
        return type("Out", (), {"logits": out})()


class TestSoftmaxFisherRanking(unittest.TestCase):
    def test_softmax_fisher_exact_formula(self):
        logits_table = torch.tensor([
            [10.0, -10.0, -10.0, -10.0],  # deterministic: p ≈ [1, 0, 0, 0] -> p(1-p) ≈ 0
            [0.0, 0.0, -10.0, -10.0],     # uncertain over tokens 0 & 1: p ≈ [0.5, 0.5, 0, 0] -> p(1-p) ≈ 0.25 each
            [-10.0, -10.0, 1.0, 1.0],     # uncertain over tokens 2 & 3: p ≈ [0, 0, 0.5, 0.5]
        ])
        model = _TinyLM(logits_table)
        chunks = [torch.tensor([[0, 1, 2]])]  # next-token positions are indices 0 and 1 (last position ignored)
        got = softmax_fisher(model, chunks)
        p = torch.softmax(logits_table[:2].float(), dim=-1)
        want = (p * (1.0 - p)).double().sum(0)
        want = want / want.sum()
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_token_importance_modes(self):
        freq = torch.tensor([100.0, 10.0, 0.0, 0.0])  # tokens 2 and 3 unseen in corpus
        sf = torch.tensor([0.05, 0.15, 0.60, 0.20])   # token 2 has high output Fisher
        f_only = token_importance(freq, sf, "freq")
        s_only = token_importance(freq, sf, "sfisher")
        mixed = token_importance(freq, sf, "sfisher-mix")

        self.assertEqual(int(torch.argmax(f_only)), 0)
        self.assertEqual(int(torch.argmax(s_only)), 2)
        self.assertGreater(float(mixed[2]), float(mixed[3]))  # unseen tokens ranked by Fisher
        self.assertAlmostEqual(float(mixed.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
