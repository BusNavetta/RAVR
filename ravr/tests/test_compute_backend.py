import numpy as np
import pytest

from compute_backend import configure_compute_backend, topk_inner_products


def _fixture():
    rng = np.random.default_rng(20260825)
    gallery = rng.normal(size=(4096, 64)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    queries = rng.normal(size=(47, 64)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    exclude = rng.integers(0, len(gallery), size=len(queries), dtype=np.int64)
    return queries, gallery, exclude


def test_cpu_backend_is_sorted_and_respects_exclusions():
    queries, gallery, exclude = _fixture()
    configure_compute_backend("cpu")
    ids, scores, report = topk_inner_products(queries, gallery, 10, exclude)
    assert report["resolved"] == "cpu"
    assert ids.shape == scores.shape == (len(queries), 10)
    assert np.all(scores[:, :-1] >= scores[:, 1:])
    assert not np.any(ids == exclude[:, None])


def test_cuda_backend_matches_cpu_topk_and_scores():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is unavailable")
    queries, gallery, exclude = _fixture()
    configure_compute_backend("cpu")
    cpu_ids, cpu_scores, _ = topk_inner_products(queries, gallery, 10, exclude)
    configure_compute_backend("cuda")
    cuda_ids, cuda_scores, report = topk_inner_products(queries, gallery, 10, exclude)
    assert report["resolved"] == "cuda"
    assert np.array_equal(cuda_ids, cpu_ids)
    assert np.max(np.abs(cuda_scores - cpu_scores)) < 2e-6
