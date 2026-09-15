from __future__ import annotations

import tempfile
import unittest

import numpy as np

from qaq_pq import QAQPQIndex
from qaq_ravr import QAQNestedIndex
from quip_pq import QUIPConfig, compute_covariance, train_quip_covariance


class QUIPCodecTests(unittest.TestCase):
    def test_noncentred_covariance(self) -> None:
        rng = np.random.default_rng(1)
        queries = rng.normal(size=(13, 12)).astype(np.float32)
        actual = compute_covariance(queries, 3)
        blocks = queries.reshape(13, 3, 4)
        expected = np.stack([block.T @ block / len(block) for block in blocks.transpose(1, 0, 2)])
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_permuted_physical_decode_round_trip(self) -> None:
        rng = np.random.default_rng(2)
        codebooks = rng.normal(size=(3, 4, 2)).astype(np.float32)
        codes = rng.integers(0, 4, size=(19, 3), dtype=np.uint8)
        permutation = np.array([4, 0, 5, 2, 1, 3], dtype=np.int64)
        index = QAQPQIndex.from_codes(
            codebooks, codes, {"dimension_permutation": permutation.tolist()}
        )
        codec_reconstruction = np.concatenate([
            codebooks[stage, codes[:, stage]] for stage in range(3)
        ], axis=1)
        expected = np.empty_like(codec_reconstruction)
        expected[:, permutation] = codec_reconstruction
        np.testing.assert_allclose(index.decode_raw(), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/permuted.quipix"
            index.save(path)
            loaded = QAQPQIndex.load(path)
            np.testing.assert_allclose(loaded.decode_raw(), expected)
            loaded.close()

    def test_permuted_full_prefix_matches_source_codec(self) -> None:
        rng = np.random.default_rng(22)
        codebooks = rng.normal(size=(3, 4, 2)).astype(np.float32)
        codes = rng.integers(0, 4, size=(31, 3), dtype=np.uint8)
        permutation = np.array([4, 0, 5, 2, 1, 3], dtype=np.int64)
        source = QAQPQIndex.from_codes(
            codebooks,
            codes,
            {
                "algorithm": "QUIP-cov(z)",
                "config": {"seed": 22},
                "dimension_permutation": permutation.tolist(),
            },
        )
        raw = source.decode_raw()
        prefix_norms = np.linalg.norm(raw, axis=1)[:, None]
        order = np.array([2, 0, 1], dtype=np.uint16)
        nested = QAQNestedIndex.from_allocation(
            source,
            order,
            prefix_norms,
            (3,),
            np.zeros(source.n_items, dtype=np.int64),
            strategy="full",
        )
        np.testing.assert_allclose(nested.decode_raw(), raw)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/permuted-full.quiprvix"
            nested.save(path)
            loaded = QAQNestedIndex.load(path)
            np.testing.assert_allclose(loaded.decode_raw(), raw)
            loaded.close()

    def test_cuda_training_decreases_quip_objective(self) -> None:
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            available = False
        if not available:
            self.skipTest("CUDA PyTorch unavailable")
        rng = np.random.default_rng(3)
        data = rng.normal(size=(128, 16)).astype(np.float32)
        queries = rng.normal(size=(31, 16)).astype(np.float32)
        config = QUIPConfig(
            m=4, k=8, seed=7, max_iterations=4,
            relative_tolerance=0.0, encode_batch_size=64,
        )
        codebooks, codes, covariance, permutation, report = train_quip_covariance(
            data, queries, config
        )
        self.assertEqual(codebooks.shape, (4, 8, 4))
        self.assertEqual(codes.shape, (128, 4))
        self.assertEqual(covariance.shape, (4, 4, 4))
        self.assertEqual(sorted(permutation.tolist()), list(range(16)))
        losses = [report["initial_encoding"]["loss"]] + [
            row["loss"] for row in report["iterations"]
        ]
        self.assertTrue(np.all(np.diff(losses) <= 5e-4))


if __name__ == "__main__":
    unittest.main()
