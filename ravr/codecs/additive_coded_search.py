"""Exact asymmetric cosine scoring from packed RQ/LSQ additive codes.
"""
from __future__ import annotations

import faiss
import numpy as np

from rq_coded_search import _normalise_queries, _rows_for_ids


def faiss_additive_codebooks(bundle) -> tuple[np.ndarray, int]:
    config = bundle.metadata["config"]
    family = config.get("family", config.get("method"))
    if family not in {"rq", "lsq"}:
        raise ValueError("coded additive search currently supports RQ and LSQ")
    stages = int(config.get("stages", config.get("m")))
    nbits = int(config["nbits"])
    size = 1 << nbits
    dimension = int(bundle.metadata["dimension"])
    stored = getattr(bundle, "codebooks", None)
    if stored is not None:
        stored = np.asarray(stored)
        expected_shape = (stages, size, dimension)
        if stored.shape != expected_shape:
            raise ValueError(
                f"unexpected stored codebook shape: {stored.shape} != {expected_shape}"
            )
        if stored.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("stored additive codebooks must be float16 or float32")
        return np.ascontiguousarray(stored, dtype=np.float32), nbits
    quantizer = getattr(bundle.codec, family)
    flat = faiss.vector_to_array(quantizer.codebooks)
    expected = stages * size * dimension
    if len(flat) != expected:
        raise ValueError(f"unexpected codebook size: {len(flat)} != {expected}")
    return (
        np.ascontiguousarray(
            flat.reshape(stages, size, dimension), dtype=np.float32
        ),
        nbits,
    )


def unpack_stage(packed_codes: np.ndarray, stage: int, nbits: int) -> np.ndarray:
    """Extract one stage's index from Faiss little-endian packed codes."""
    packed = np.asarray(packed_codes, dtype=np.uint8)
    if packed.ndim != 2 or stage < 0 or nbits <= 0:
        raise ValueError("invalid packed codes, stage, or nbits")
    bit_offset = stage * nbits
    first_byte = bit_offset // 8
    shift = bit_offset % 8
    required = (bit_offset + nbits + 7) // 8
    if required > packed.shape[1]:
        raise ValueError("stage exceeds the packed code width")
    value = packed[:, first_byte].astype(np.uint32) >> shift
    bits_read = 8 - shift
    next_byte = first_byte + 1
    while bits_read < nbits:
        value |= packed[:, next_byte].astype(np.uint32) << bits_read
        bits_read += 8
        next_byte += 1
    return value & ((1 << nbits) - 1)


def score_packed_additive_codes(
    queries: np.ndarray,
    codebooks: np.ndarray,
    packed_codes: np.ndarray,
    inverse_norms: np.ndarray,
    nbits: int,
) -> np.ndarray:
    queries = _normalise_queries(queries)
    codebooks = np.ascontiguousarray(np.asarray(codebooks, dtype=np.float32))
    packed = np.asarray(packed_codes, dtype=np.uint8)
    norms = np.asarray(inverse_norms, dtype=np.float32)
    if codebooks.ndim != 3 or queries.shape[1] != codebooks.shape[2]:
        raise ValueError("queries and codebooks have incompatible shapes")
    expected_bytes = (codebooks.shape[0] * nbits + 7) // 8
    if packed.ndim != 2 or packed.shape[1] != expected_bytes:
        raise ValueError("packed codes do not match stage/bit configuration")
    if norms.shape != (len(packed),):
        raise ValueError("inverse norms do not match item count")
    lookup = np.einsum("qd,mkd->qmk", queries, codebooks, optimize=True)
    scores = np.zeros((len(queries), len(packed)), dtype=np.float32)
    for stage in range(codebooks.shape[0]):
        indices = unpack_stage(packed, stage, nbits)
        scores += lookup[:, stage, :][:, indices]
    scores *= norms[None, :]
    return scores


def search_packed_additive_codes(
    queries: np.ndarray,
    codebooks: np.ndarray,
    packed_codes: np.ndarray,
    inverse_norms: np.ndarray,
    nbits: int,
    *,
    k: int,
    item_ids: np.ndarray | None = None,
    exclude_item_ids: np.ndarray | None = None,
    query_batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    queries = _normalise_queries(queries)
    packed = np.asarray(packed_codes, dtype=np.uint8)
    if not 0 < k <= len(packed):
        raise ValueError("k must be in [1, number of items]")
    ids = (
        np.arange(len(packed), dtype=np.uint32)
        if item_ids is None else np.asarray(item_ids)
    )
    if ids.shape != (len(packed),):
        raise ValueError("item IDs do not match item count")
    excluded_rows = None
    if exclude_item_ids is not None:
        excluded = np.asarray(exclude_item_ids)
        if excluded.ndim == 1:
            excluded = excluded[:, None]
        if excluded.ndim != 2 or excluded.shape[0] != len(queries):
            raise ValueError("excluded IDs do not match queries")
        excluded_rows = _rows_for_ids(ids, excluded.reshape(-1)).reshape(
            excluded.shape
        )

    all_scores = np.empty((len(queries), k), dtype=np.float32)
    all_ids = np.empty((len(queries), k), dtype=ids.dtype)
    for start in range(0, len(queries), query_batch_size):
        stop = min(len(queries), start + query_batch_size)
        scores = score_packed_additive_codes(
            queries[start:stop], codebooks, packed, inverse_norms, nbits
        )
        if excluded_rows is not None:
            local = excluded_rows[start:stop]
            scores[np.arange(stop - start)[:, None], local] = -np.inf
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        values = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-values, axis=1, kind="stable")
        rows = np.take_along_axis(candidates, order, axis=1)
        all_scores[start:stop] = np.take_along_axis(scores, rows, axis=1)
        all_ids[start:stop] = ids[rows]
    return all_scores, all_ids


__all__ = [
    "faiss_additive_codebooks",
    "score_packed_additive_codes",
    "search_packed_additive_codes",
    "unpack_stage",
]
