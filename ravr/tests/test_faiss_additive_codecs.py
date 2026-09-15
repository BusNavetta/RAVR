from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from faiss_additive_codecs import AdditiveCodecConfig, train_additive_bundle
from faiss_fixed_rate import FaissCodecBundle
from faiss_fp16_additive import FP16AdditiveBundle, from_faiss_additive_bundle
from nested_additive_ravr import NestedAdditiveIndex
from additive_coded_search import (
    faiss_additive_codebooks,
    score_packed_additive_codes,
)


def _gallery(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(512, 16)).astype(np.float32)
    return value / np.linalg.norm(value, axis=1, keepdims=True)


def test_rq_non_byte_aligned_codes_round_trip(tmp_path: Path) -> None:
    gallery = _gallery()
    config = AdditiveCodecConfig(
        family="rq", stages=2, nbits=4, seed=7,
        rq_kmeans_iterations=2, rq_refine_iterations=1,
    )
    bundle = train_additive_bundle(gallery, np.arange(256), config)
    assert bundle.codes.shape == (512, 1)
    path = tmp_path / "rq.fcix"
    bundle.save(path)
    loaded = FaissCodecBundle.load(path)
    reconstruction = loaded.decode_normalized()
    np.testing.assert_allclose(np.linalg.norm(reconstruction, axis=1), 1.0, atol=2e-3)
    assert loaded.metadata["nominal_code_bits_per_item"] == 8
    assert loaded.persistent_bytes == path.stat().st_size
    loaded.close()


def test_factory_names_and_validation() -> None:
    assert AdditiveCodecConfig("lsq", 4, 6).factory == "LSQ4x6"
    assert AdditiveCodecConfig("prq", 3, 6, splits=2).factory == "PRQ2x3x6"


def test_lsq_packed_code_scores_equal_dense_decode() -> None:
    gallery = _gallery(9)
    config = AdditiveCodecConfig(
        family="lsq", stages=4, nbits=4, seed=3,
        lsq_train_iterations=2, lsq_train_ils_iterations=1,
        lsq_encode_ils_iterations=1, lsq_icm_iterations=1,
        lsq_perturbations=1,
    )
    bundle = train_additive_bundle(gallery, np.arange(256), config)
    codebooks, nbits = faiss_additive_codebooks(bundle)
    coded = score_packed_additive_codes(
        gallery[:3], codebooks, bundle.codes, bundle.inverse_norms, nbits
    )
    dense = gallery[:3] @ bundle.decode_normalized().T
    np.testing.assert_allclose(coded, dense, atol=6e-7)


def test_fp16_additive_bundle_and_nested_index_round_trip(tmp_path: Path) -> None:
    gallery = _gallery(19)
    config = AdditiveCodecConfig(
        family="rq", stages=2, nbits=4, seed=5,
        rq_kmeans_iterations=2, rq_refine_iterations=1,
    )
    source = train_additive_bundle(gallery, np.arange(256), config)
    converted = from_faiss_additive_bundle(source)
    path = tmp_path / "rq.fa16"
    converted.save(path)
    loaded = FP16AdditiveBundle.load(path)
    assert loaded.codebooks.dtype == np.float16
    np.testing.assert_array_equal(loaded.codes, source.codes)
    np.testing.assert_allclose(
        np.linalg.norm(loaded.decode_normalized(), axis=1), 1.0, atol=2e-3
    )
    prefix_norms = np.linalg.norm(loaded.decode_raw(), axis=1)[:, None]
    nested = NestedAdditiveIndex.from_allocation(
        loaded,
        prefix_norms,
        (2,),
        np.zeros(loaded.n_items, dtype=np.int64),
        strategy="full",
    )
    nested_path = tmp_path / "rq.navx"
    nested.save(nested_path)
    reloaded = NestedAdditiveIndex.load(nested_path)
    assert reloaded.codebooks.dtype == np.float16
    np.testing.assert_allclose(
        reloaded.decode_normalized(), loaded.decode_normalized(), rtol=0, atol=0
    )
    assert reloaded.storage_breakdown()["codec_and_codebooks"] == converted.codebooks.nbytes
    reloaded.close()
    loaded.close()
