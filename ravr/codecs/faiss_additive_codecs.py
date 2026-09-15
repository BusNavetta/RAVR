"""Auditable Faiss additive-quantizer codecs beyond the 8-bit RQ baseline.

The codec family and its exact physical representation are deliberately kept
separate.  Faiss produces the packed per-item codes; :class:`FaissCodecBundle`
then persists the trained codec, packed codes, explicit uint32 IDs, and float16
inverse reconstruction norms in the same format used by the original
PQ/OPQ/RQ audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import faiss
import numpy as np

from faiss_fixed_rate import EPS, FaissCodecBundle


FAMILIES = {"rq", "lsq", "prq", "plsq"}


@dataclass(frozen=True)
class AdditiveCodecConfig:
    family: str
    stages: int
    nbits: int = 6
    splits: int = 1
    seed: int = 17
    rq_kmeans_iterations: int = 5
    rq_refine_iterations: int = 5
    rq_beam_size: int = 5
    lsq_train_iterations: int = 25
    lsq_train_ils_iterations: int = 8
    lsq_encode_ils_iterations: int = 16
    lsq_icm_iterations: int = 4
    lsq_perturbations: int = 4

    def validate(self, dimension: int) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {sorted(FAMILIES)}")
        if self.stages <= 0 or not 1 <= self.nbits <= 16:
            raise ValueError("stages must be positive and nbits must be in [1, 16]")
        if self.splits <= 0 or dimension % self.splits:
            raise ValueError("splits must be positive and divide the dimension")
        if self.family in {"rq", "lsq"} and self.splits != 1:
            raise ValueError("RQ/LSQ use splits=1; use PRQ/PLSQ for product splits")
        if self.family in {"prq", "plsq"} and self.splits < 2:
            raise ValueError("PRQ/PLSQ require at least two product splits")

    @property
    def factory(self) -> str:
        prefix = self.family.upper()
        if self.family in {"rq", "lsq"}:
            return f"{prefix}{self.stages}x{self.nbits}"
        return f"{prefix}{self.splits}x{self.stages}x{self.nbits}"

    @property
    def display_name(self) -> str:
        return self.factory

    @property
    def nominal_code_bits(self) -> int:
        return self.stages * self.nbits * self.splits

    def file_stem(self) -> str:
        return f"{self.factory.lower()}_seed{self.seed}"


def _configure_rq(rq, config: AdditiveCodecConfig, split: int = 0) -> None:
    rq.cp.niter = config.rq_kmeans_iterations
    rq.cp.seed = config.seed + split
    rq.niter_codebook_refine = config.rq_refine_iterations
    rq.max_beam_size = config.rq_beam_size


def _configure_lsq(lsq, config: AdditiveCodecConfig, split: int = 0) -> None:
    lsq.random_seed = config.seed + split
    lsq.train_iters = config.lsq_train_iterations
    lsq.train_ils_iters = config.lsq_train_ils_iterations
    lsq.encode_ils_iters = config.lsq_encode_ils_iterations
    lsq.icm_iters = config.lsq_icm_iterations
    lsq.nperts = config.lsq_perturbations


def _configure_codec(codec, config: AdditiveCodecConfig) -> None:
    if config.family == "rq":
        _configure_rq(codec.rq, config)
    elif config.family == "lsq":
        _configure_lsq(codec.lsq, config)
    else:
        product = getattr(codec, config.family)
        for split in range(product.nsplits):
            quantizer = faiss.downcast_AdditiveQuantizer(product.subquantizer(split))
            if config.family == "prq":
                _configure_rq(quantizer, config, split)
            else:
                _configure_lsq(quantizer, config, split)


def train_additive_bundle(
    gallery: np.ndarray,
    train_ids: np.ndarray,
    config: AdditiveCodecConfig,
) -> FaissCodecBundle:
    """Train, encode, and return a complete reloadable physical bundle."""
    gallery = np.ascontiguousarray(np.asarray(gallery, dtype=np.float32))
    train_ids = np.asarray(train_ids, dtype=np.int64)
    config.validate(gallery.shape[1])
    codec = faiss.index_factory(gallery.shape[1], config.factory, faiss.METRIC_L2)
    _configure_codec(codec, config)

    started = time.perf_counter()
    codec.train(np.ascontiguousarray(gallery[train_ids]))
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codes = np.asarray(codec.sa_encode(gallery), dtype=np.uint8)
    encode_seconds = time.perf_counter() - started
    reconstruction = np.asarray(codec.sa_decode(codes), dtype=np.float32)
    inverse_norms = (
        1.0 / np.maximum(np.linalg.norm(reconstruction, axis=1), EPS)
    ).astype(np.float16)

    dimension = gallery.shape[1]
    codewords = 1 << config.nbits
    # Product variants divide the dimensions between splits, so their total
    # number of codebook floats is still stages * K * dimension.
    nominal_codebook_bytes = config.stages * codewords * dimension * 4
    metadata = {
        "config": asdict(config),
        "factory": config.factory,
        "display_name": config.display_name,
        "family": config.family,
        "dimension": dimension,
        "items": len(gallery),
        "training_items": len(train_ids),
        "training_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(train_ids).view(np.uint8)
        ).hexdigest(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "train_seconds": train_seconds,
        "encode_seconds": encode_seconds,
        "nominal_code_bits_per_item": config.nominal_code_bits,
        "packed_code_bytes_per_item": int(codec.sa_code_size()),
        "rotation_matrix_nominal_bytes": 0,
        "codebooks_nominal_bytes": nominal_codebook_bytes,
        "normalisation": "float16 inverse decoded-vector norm",
        "ids": "uint32 explicit row IDs",
    }
    return FaissCodecBundle(
        codec=codec,
        codes=codes,
        item_ids=np.arange(len(gallery), dtype=np.uint32),
        inverse_norms=inverse_norms,
        metadata=metadata,
    )


__all__ = ["AdditiveCodecConfig", "train_additive_bundle"]
