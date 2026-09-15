"""Equation-level tests for the QAQ accelerated implementation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qaq_pq import (
    QAQPQIndex,
    encode_cuda,
    pack_codes,
    unpack_codes,
    update_codebook_cuda,
)
from qaq_ravr import QAQNestedIndex


def _official_scalar_encode(points, codebook, matrices, labels, iterations):
    n, d = points.shape
    m, k, ds = codebook.shape
    codes = np.zeros((n, m), dtype=np.int64)
    losses = np.zeros(n, dtype=np.float64)
    for row in range(n):
        reconstruction = codebook[:, 0, :].reshape(d).astype(np.float64)
        matrix = matrices[labels[row]].astype(np.float64)
        point = points[row].astype(np.float64)
        for _ in range(iterations):
            for stage in range(m):
                start, stop = stage * ds, (stage + 1) * ds
                values = []
                for symbol in range(k):
                    reconstruction[start:stop] = codebook[stage, symbol]
                    residual = point - reconstruction
                    values.append(residual @ matrix @ residual)
                codes[row, stage] = int(np.argmin(values))
                reconstruction[start:stop] = codebook[stage, codes[row, stage]]
        residual = point - reconstruction
        losses[row] = residual @ matrix @ residual
    return codes, float(losses.sum())


def _official_normal_equations(points, codes, matrices, labels, k, regularization):
    n, d = points.shape
    m = codes.shape[1]
    ds = d // m
    system = np.zeros((k * d, k * d), dtype=np.float64)
    rhs = np.zeros(k * d, dtype=np.float64)
    for row in range(n):
        matrix = matrices[labels[row]].astype(np.float64)
        selected = np.empty(d, dtype=np.int64)
        for stage in range(m):
            start = (stage * k + int(codes[row, stage])) * ds
            selected[stage * ds:(stage + 1) * ds] = np.arange(start, start + ds)
        system[np.ix_(selected, selected)] += matrix
        rhs[selected] += matrix @ points[row]
    solution = np.linalg.solve(system + regularization * np.eye(k * d), rhs)
    return solution.reshape(m, k, ds).astype(np.float32)


class QAQPQTests(unittest.TestCase):
    def test_pack_round_trip(self):
        rng = np.random.default_rng(7)
        for bits, stages in ((2, 32), (3, 24), (4, 16)):
            codes = rng.integers(0, 1 << bits, size=(19, stages), dtype=np.uint8)
            packed = pack_codes(codes, bits)
            np.testing.assert_array_equal(unpack_codes(packed, stages, bits), codes)

    def test_cuda_encode_matches_official_scalar_equation(self):
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            available = False
        if not available:
            self.skipTest("CUDA PyTorch unavailable")
        rng = np.random.default_rng(11)
        points = rng.normal(size=(13, 8)).astype(np.float32)
        codebook = rng.normal(size=(2, 4, 4)).astype(np.float32)
        base = rng.normal(size=(3, 8, 8)).astype(np.float32)
        matrices = np.einsum("cdi,cei->cde", base, base).astype(np.float32)
        labels = rng.integers(0, 3, size=len(points), dtype=np.int32)
        expected_codes, expected_loss = _official_scalar_encode(
            points, codebook, matrices, labels, 3
        )
        codes, loss, _ = encode_cuda(points, codebook, matrices, labels, iterations=3)
        np.testing.assert_array_equal(codes, expected_codes)
        self.assertAlmostEqual(loss, expected_loss, delta=max(1e-3, abs(expected_loss) * 2e-5))

    def test_cuda_update_matches_official_dense_system(self):
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            available = False
        if not available:
            self.skipTest("CUDA PyTorch unavailable")
        rng = np.random.default_rng(23)
        points = rng.normal(size=(17, 8)).astype(np.float32)
        codes = rng.integers(0, 4, size=(17, 2), dtype=np.uint8)
        base = rng.normal(size=(3, 8, 8)).astype(np.float32)
        matrices = np.einsum("cdi,cei->cde", base, base).astype(np.float32)
        labels = rng.integers(0, 3, size=len(points), dtype=np.int32)
        expected = _official_normal_equations(points, codes, matrices, labels, 4, 1e-3)
        actual, _ = update_codebook_cuda(points, codes, matrices, labels, 4, 1e-3)
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

    def test_physical_index_round_trip(self):
        rng = np.random.default_rng(31)
        codebook = rng.normal(size=(4, 8, 3)).astype(np.float32)
        codes = rng.integers(0, 8, size=(37, 4), dtype=np.uint8)
        index = QAQPQIndex.from_codes(codebook, codes, {"test": True})
        expected = index.decode_normalized()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.qaqix"
            index.save(path)
            loaded = QAQPQIndex.load(path)
            np.testing.assert_allclose(loaded.decode_normalized(), expected, rtol=0, atol=0)
            self.assertEqual(path.stat().st_size, index.persistent_bytes)
            self.assertEqual(loaded.storage_breakdown()["persistent_bytes"], path.stat().st_size)
            loaded.close()

    def test_full_qaq_nested_prefix_preserves_fixed_codec(self):
        rng = np.random.default_rng(37)
        codebook = rng.normal(size=(4, 8, 3)).astype(np.float32)
        codes = rng.integers(0, 8, size=(41, 4), dtype=np.uint8)
        fixed = QAQPQIndex.from_codes(codebook, codes)
        order = np.array([2, 0, 3, 1], dtype=np.uint16)
        prefix_norms = np.ones((len(codes), 1), dtype=np.float32)
        raw = fixed.decode_normalized()
        # The variable index stores the same FP16 inverse norms as the fixed
        # index when the complete prefix is selected.
        full_raw = np.empty_like(raw)
        for stage in range(4):
            start, stop = stage * 3, (stage + 1) * 3
            full_raw[:, start:stop] = codebook[stage, codes[:, stage]]
        prefix_norms[:, 0] = np.linalg.norm(full_raw, axis=1)
        nested = QAQNestedIndex.from_allocation(
            fixed, order, prefix_norms, (4,), np.zeros(len(codes), dtype=np.int64),
            strategy="full",
        )
        np.testing.assert_allclose(nested.decode_normalized(), raw, rtol=0, atol=0)

    def test_fp16_codebook_round_trip_and_nested_storage(self):
        rng = np.random.default_rng(43)
        codebook = rng.normal(size=(4, 8, 3)).astype(np.float32)
        codes = rng.integers(0, 8, size=(41, 4), dtype=np.uint8)
        fp32 = QAQPQIndex.from_codes(codebook, codes)
        fp16 = fp32.to_precision("fp16")
        self.assertEqual(fp16.codebooks.dtype, np.float16)
        self.assertLess(fp16.persistent_bytes, fp32.persistent_bytes)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny-fp16.qaqix"
            fp16.save(path)
            loaded = QAQPQIndex.load(path)
            self.assertEqual(loaded.codebooks.dtype, np.float16)
            np.testing.assert_array_equal(
                unpack_codes(loaded.packed_codes, loaded.stages, loaded.nbits), codes
            )
            order = np.arange(4, dtype=np.uint16)
            raw = loaded.decode_raw()
            prefix_norms = np.linalg.norm(raw, axis=1)[:, None]
            nested = QAQNestedIndex.from_allocation(
                loaded, order, prefix_norms, (4,),
                np.zeros(len(codes), dtype=np.int64), strategy="full",
            )
            nested_path = Path(temporary) / "tiny-fp16.qaqrvix"
            nested.save(nested_path)
            nested_loaded = QAQNestedIndex.load(nested_path)
            self.assertEqual(nested_loaded.codebooks.dtype, np.float16)
            self.assertEqual(
                nested_loaded.storage_breakdown()["codebooks"],
                loaded.codebooks.nbytes,
            )
            np.testing.assert_allclose(
                nested_loaded.decode_normalized(), loaded.decode_normalized(),
                rtol=0, atol=0,
            )
            nested_loaded.close()
            loaded.close()


if __name__ == "__main__":
    unittest.main()
