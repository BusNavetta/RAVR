"""Asymmetric cosine search directly over residual-quantizer codes.

An M-stage, 8-bit RQ item is stored as M uint8 codebook indices.  For a
normalized query q and decoded item x_i = sum_m C[m, code[i,m]], the cosine
numerator is

    q.T x_i = sum_m (q.T C[m, code[i,m]]).

The terms in parentheses form a small query/codeword lookup table.  This
module performs exhaustive search from that table and the stored inverse item
norms; it never materializes the N-by-D reconstructed gallery.
"""
from __future__ import annotations

import faiss
import numpy as np


EPS = 1e-9


def faiss_rq_codebooks(bundle) -> np.ndarray:
    """Extract [stage, codeword, dimension] FP32 codebooks from an RQ bundle."""
    config = bundle.metadata["config"]
    if config["method"] != "rq":
        raise ValueError("coded RQ search requires a residual-quantizer bundle")
    stages = int(config["m"])
    size = 1 << int(config["nbits"])
    dimension = int(bundle.metadata["dimension"])
    flat = faiss.vector_to_array(bundle.codec.rq.codebooks)
    expected = stages * size * dimension
    if len(flat) != expected:
        raise ValueError(f"unexpected RQ codebook size: {len(flat)} != {expected}")
    return np.ascontiguousarray(flat.reshape(stages, size, dimension), dtype=np.float32)


def _normalise_queries(queries: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError("queries must have shape [query, dimension]")
    norm = np.linalg.norm(value, axis=1)
    if np.any(norm <= EPS):
        raise ValueError("queries contain a zero-norm vector")
    return np.ascontiguousarray(value / norm[:, None])


def _validate(
    queries: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    inverse_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = _normalise_queries(queries)
    cb = np.ascontiguousarray(np.asarray(codebooks, dtype=np.float32))
    code = np.ascontiguousarray(np.asarray(codes))
    inv = np.ascontiguousarray(np.asarray(inverse_norms, dtype=np.float32))
    if cb.ndim != 3:
        raise ValueError("codebooks must have shape [stage, codeword, dimension]")
    if code.ndim != 2 or code.shape[1] != cb.shape[0]:
        raise ValueError("codes do not match the RQ stage count")
    if q.shape[1] != cb.shape[2]:
        raise ValueError("query and codebook dimensions do not match")
    if inv.shape != (code.shape[0],):
        raise ValueError("inverse norms do not match the item count")
    if np.issubdtype(code.dtype, np.signedinteger) and np.any(code < 0):
        raise ValueError("codes contain a negative codeword index")
    if code.size and int(np.max(code)) >= cb.shape[1]:
        raise ValueError("codes exceed the codebook size")
    return q, cb, code.astype(np.int64, copy=False), inv


def score_rq_codes(
    queries: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    inverse_norms: np.ndarray,
) -> np.ndarray:
    """Return the exact coded-domain cosine score matrix in FP32 arithmetic."""
    q, cb, code, inv = _validate(queries, codebooks, codes, inverse_norms)
    return _score_validated(q, cb, code, inv)


def _score_validated(
    queries: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    inverse_norms: np.ndarray,
) -> np.ndarray:
    # [query, stage, codeword]; only Q*M*K dot products touch dimension D.
    lookup = np.einsum(
        "qd,mkd->qmk", queries, codebooks, optimize=True
    )
    scores = np.zeros((len(queries), len(codes)), dtype=np.float32)
    for stage in range(codebooks.shape[0]):
        scores += lookup[:, stage, :][:, codes[:, stage]]
    scores *= inverse_norms[None, :]
    return scores


def _rows_for_ids(item_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    order = np.argsort(item_ids, kind="stable")
    sorted_ids = item_ids[order]
    positions = np.searchsorted(sorted_ids, requested)
    valid = positions < len(sorted_ids)
    clipped = np.minimum(positions, max(0, len(sorted_ids) - 1))
    valid &= sorted_ids[clipped] == requested
    if not np.all(valid):
        missing = np.asarray(requested)[~valid]
        raise ValueError(f"excluded item IDs are absent: {missing[:5].tolist()}")
    return order[positions]


def search_rq_codes(
    queries: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    inverse_norms: np.ndarray,
    *,
    k: int,
    item_ids: np.ndarray | None = None,
    exclude_item_ids: np.ndarray | None = None,
    query_batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k exhaustive search without an N-by-D reconstructed gallery.

    At most ``query_batch_size * N`` scores are materialized.  Returned IDs are
    explicit ``item_ids`` when supplied and item rows otherwise.
    """
    q, cb, code, inv = _validate(queries, codebooks, codes, inverse_norms)
    if not 0 < k <= len(code):
        raise ValueError("k must be in [1, number of items]")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    ids = (
        np.arange(len(code), dtype=np.uint32)
        if item_ids is None else np.asarray(item_ids)
    )
    if ids.shape != (len(code),):
        raise ValueError("item_ids do not match the item count")

    excluded_rows = None
    if exclude_item_ids is not None:
        excluded = np.asarray(exclude_item_ids)
        if excluded.ndim == 1:
            if len(excluded) != len(q):
                raise ValueError("one excluded ID is required per query")
            excluded = excluded[:, None]
        if excluded.ndim != 2 or excluded.shape[0] != len(q):
            raise ValueError("exclude_item_ids must have shape [query] or [query, n]")
        excluded_rows = _rows_for_ids(ids, excluded.reshape(-1)).reshape(excluded.shape)

    all_scores = np.empty((len(q), k), dtype=np.float32)
    all_ids = np.empty((len(q), k), dtype=ids.dtype)
    for start in range(0, len(q), query_batch_size):
        stop = min(len(q), start + query_batch_size)
        scores = _score_validated(q[start:stop], cb, code, inv)
        if excluded_rows is not None:
            local = excluded_rows[start:stop]
            scores[np.arange(stop - start)[:, None], local] = -np.inf
        candidate_rows = np.argpartition(scores, -k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidate_rows, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        rows = np.take_along_axis(candidate_rows, order, axis=1)
        all_scores[start:stop] = np.take_along_axis(scores, rows, axis=1)
        all_ids[start:stop] = ids[rows]
    return all_scores, all_ids


__all__ = [
    "faiss_rq_codebooks",
    "score_rq_codes",
    "search_rq_codes",
]
