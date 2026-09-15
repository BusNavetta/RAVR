from pathlib import Path
import tempfile

import numpy as np

from additive_coded_search import faiss_additive_codebooks, unpack_stage
from faiss_additive_codecs import AdditiveCodecConfig, train_additive_bundle
from nested_additive_ravr import NestedAdditiveIndex, codec_sha256, query_ip_distortion
from benchmark_nested_additive_ravr import ALLOWED_LENGTHS, STRATEGIES, _selections


def _fixture():
    rng = np.random.default_rng(19)
    gallery = rng.normal(size=(1024, 8)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    config = AdditiveCodecConfig(
        family="rq",
        stages=6,
        nbits=4,
        seed=19,
        rq_kmeans_iterations=2,
        rq_refine_iterations=1,
        rq_beam_size=2,
    )
    bundle = train_additive_bundle(gallery, np.arange(len(gallery)), config)
    codebooks, nbits = faiss_additive_codebooks(bundle)
    norms = np.empty((len(gallery), 3), dtype=np.float32)
    reconstruction = np.zeros_like(gallery)
    for stage in range(6):
        reconstruction += codebooks[stage, unpack_stage(bundle.codes, stage, nbits)]
        if stage + 1 in (2, 4, 6):
            norms[:, (stage + 1) // 2 - 1] = np.linalg.norm(reconstruction, axis=1)
    return gallery, bundle, norms


def test_nested_bundle_is_exact_cap_reloadable_and_dense_free():
    gallery, bundle, norms = _fixture()
    n = len(gallery)
    uniform = np.full(n, 1, dtype=np.int64)
    mixed = np.concatenate((
        np.zeros(n // 2, dtype=np.int64),
        np.full(n - n // 2, 2, dtype=np.int64),
    ))
    uniform_index = NestedAdditiveIndex.from_allocation(
        bundle, norms, (2, 4, 6), uniform, strategy="uniform"
    )
    mixed_index = NestedAdditiveIndex.from_allocation(
        bundle, norms, (2, 4, 6), mixed, strategy="ravr"
    )
    assert uniform_index.code_only_bytes == mixed_index.code_only_bytes == n * 2
    assert uniform_index.persistent_bytes == mixed_index.persistent_bytes
    assert "decoded" not in uniform_index._layout
    assert "one_hot" not in uniform_index._layout

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mixed.navx"
        mixed_index.save(path)
        loaded = NestedAdditiveIndex.load(path)
        assert path.stat().st_size == mixed_index.persistent_bytes
        assert loaded.metadata["codec_sha256"] == codec_sha256(bundle.codec)
        assert np.array_equal(loaded.assigned_lengths_by_id(), np.where(mixed == 0, 2, 6))
        loaded.close()


def test_direct_code_scores_equal_decoded_cosine_scores():
    _, bundle, norms = _fixture()
    selected = np.arange(bundle.n_items, dtype=np.int64) % 3
    index = NestedAdditiveIndex.from_allocation(
        bundle, norms, (2, 4, 6), selected, strategy="mixed"
    )
    reconstruction = index.decode_normalized()
    rng = np.random.default_rng(23)
    queries = rng.normal(size=(5, reconstruction.shape[1])).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    direct = index.score_queries(queries)
    dense = queries @ reconstruction.T
    assert np.max(np.abs(direct - dense)) < 1e-5
    for row in range(len(queries)):
        assert np.array_equal(np.argsort(-direct[row])[:10], np.argsort(-dense[row])[:10])


def test_query_ip_distortion_matches_brute_force_query_scores():
    gallery, bundle, norms = _fixture()
    rng = np.random.default_rng(29)
    queries = rng.normal(size=(13, gallery.shape[1])).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    measured = query_ip_distortion(
        bundle, gallery, queries, norms, (2, 4, 6), block_size=127
    )
    codebooks, nbits = faiss_additive_codebooks(bundle)
    reconstruction = np.zeros_like(gallery)
    option = 0
    for stage in range(6):
        reconstruction += codebooks[stage, unpack_stage(bundle.codes, stage, nbits)]
        if stage + 1 not in (2, 4, 6):
            continue
        normalized = reconstruction / norms[:, option, None]
        brute = np.mean(
            (queries @ gallery.T - queries @ normalized.T) ** 2, axis=0
        )
        assert np.max(np.abs(measured[:, option] - brute)) < 2e-6
        option += 1


def test_allocator_controls_use_legal_options_and_respect_symbol_cap():
    rng = np.random.default_rng(31)
    n_items = 100
    coverage = rng.integers(0, 20, size=n_items, dtype=np.int32)
    remaining = np.arange(len(ALLOWED_LENGTHS) - 1, -1, -1, dtype=np.float64)
    state = {
        "coverage": coverage,
        "distortion": rng.random((n_items, 1)) * remaining[None, :],
        "prior_distortion": rng.random((n_items, 1)) * remaining[None, :],
        "query_ip_distortion": rng.random((n_items, 1)) * remaining[None, :],
    }
    selected = _selections(state, n_items, seed=31, target_length=12, nbits=4)
    assert set(selected) == set(STRATEGIES)
    option_costs = np.arange(len(ALLOWED_LENGTHS), dtype=np.int64)
    expected_cost = n_items * ALLOWED_LENGTHS.index(12)
    for values in selected.values():
        assert values.shape == (n_items,)
        assert np.all((values >= 0) & (values < len(ALLOWED_LENGTHS)))
        assert int(option_costs[values].sum()) <= expected_cost
    assert int(option_costs[selected["uniform"]].sum()) == expected_cost


def test_unseen_rd_fallback_preserves_seen_ravr_and_uses_available_slack():
    n_items = 20
    coverage = np.r_[np.ones(8, dtype=np.int32), np.zeros(12, dtype=np.int32)]
    remaining = np.arange(len(ALLOWED_LENGTHS) - 1, -1, -1, dtype=np.float64)
    state = {
        "coverage": coverage,
        "distortion": coverage[:, None] * remaining[None, :],
        "prior_distortion": np.linspace(1.0, 2.0, n_items)[:, None] * remaining[None, :],
        "query_ip_distortion": np.ones((n_items, 1)) * remaining[None, :],
    }
    selected = _selections(state, n_items, seed=37, target_length=14, nbits=4)
    seen = coverage > 0
    assert np.array_equal(
        selected["ravr_unseen_rd"][seen], selected["ravr"][seen]
    )
    costs = np.arange(len(ALLOWED_LENGTHS), dtype=np.int64)
    assert costs[selected["ravr_unseen_rd"]].sum() >= costs[selected["ravr"]].sum()
    assert costs[selected["ravr_unseen_rd"]].sum() <= n_items * 3
