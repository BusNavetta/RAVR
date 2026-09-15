"""Reference implementation of Retrieval-Aware Variable-Rate RQ (RAVR-RQ).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import struct
import time
from typing import Iterable, Sequence

import numpy as np


MAGIC = b"RAVRRQ1\0"
HEADER_BYTES = 8192
ALIGN_BYTES = 64
FORMAT_VERSION = 2
EPS = 1e-9
OBJECTIVE_BOUNDARY_KL = "boundary_kl"
OBJECTIVE_ORDINAL_KL = "ordinal_kl"
OBJECTIVE_LEGACY = "legacy"

# Frozen only for reproducing the pre-ablation objective.  These values are not
# part of the current RAVR objective or its public configuration.
LEGACY_OBJECTIVE_PARAMETERS = {
    "lambda_prior": 0.01,
    "lambda_score": 0.1,
    "lambda_rank": 0.25,
    "boundary_temperature_ratio": 0.25,
    "gamma_min": 1e-3,
}
LEGACY_UNUSED_CONFIG_FIELDS = {
    "local_density_block_size",
    "local_density_neighbors",
}


def _normalise(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= EPS):
        raise ValueError(f"{name} contains a zero-norm vector")
    return np.ascontiguousarray(x / norms[:, None])


def _align(value: int, alignment: int = ALIGN_BYTES) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Descending top-k with deterministic score/id tie handling."""
    values = np.asarray(values)
    k = min(max(int(k), 0), len(values))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == len(values):
        candidates = np.arange(len(values), dtype=np.int64)
    else:
        candidates = np.argpartition(values, len(values) - k)[-k:]
    return candidates[np.lexsort((candidates, -values[candidates]))]


def _nearest(x: np.ndarray, centres: np.ndarray, block: int = 8192) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int64)
    c2 = np.sum(centres * centres, axis=1)
    for start in range(0, len(x), block):
        xb = x[start:start + block]
        distance = np.sum(xb * xb, axis=1)[:, None] + c2[None, :] - 2 * xb @ centres.T
        out[start:start + len(xb)] = np.argmin(distance, axis=1)
    return out


def _euclidean_kmeans(
    x: np.ndarray, k: int, seed: int, iterations: int
) -> np.ndarray:
    """Small deterministic Lloyd implementation with k-means++ initialisation."""
    if len(x) < k:
        raise ValueError(f"cannot train {k} codewords from {len(x)} samples")
    rng = np.random.default_rng(seed)
    centres = np.empty((k, x.shape[1]), dtype=np.float32)
    centres[0] = x[rng.integers(len(x))]
    closest = np.sum((x - centres[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(closest.sum(dtype=np.float64))
        if total <= EPS:
            pick = int(rng.integers(len(x)))
        else:
            pick = int(rng.choice(len(x), p=closest / total))
        centres[j] = x[pick]
        closest = np.minimum(closest, np.sum((x - centres[j]) ** 2, axis=1))

    for _ in range(iterations):
        assignment = _nearest(x, centres)
        updated = centres.copy()
        for j in range(k):
            mask = assignment == j
            if mask.any():
                updated[j] = x[mask].mean(axis=0)
            else:
                # Deterministically repair an empty cell with the worst represented point.
                distance = np.sum((x - centres[assignment]) ** 2, axis=1)
                updated[j] = x[int(np.argmax(distance))]
        if np.allclose(updated, centres, rtol=0, atol=1e-6):
            centres = updated
            break
        centres = updated
    return centres


def _spherical_kmeans(
    x: np.ndarray, k: int, seed: int, iterations: int
) -> np.ndarray:
    if len(x) < k:
        raise ValueError(f"cannot train {k} IVF centroids from {len(x)} samples")
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)].copy()
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), EPS)
    for _ in range(iterations):
        assignment = np.argmax(x @ centres.T, axis=1)
        updated = centres.copy()
        for j in range(k):
            mask = assignment == j
            if mask.any():
                updated[j] = x[mask].mean(axis=0)
            else:
                # Pick the item with the weakest affinity to its assigned centroid.
                affinity = np.sum(x * centres[assignment], axis=1)
                updated[j] = x[int(np.argmin(affinity))]
        updated /= np.maximum(np.linalg.norm(updated, axis=1, keepdims=True), EPS)
        if np.allclose(updated, centres, rtol=0, atol=1e-6):
            centres = updated
            break
        centres = updated
    return centres


def _code_dtype(codebook_size: int) -> np.dtype:
    if codebook_size <= 256:
        return np.dtype(np.uint8)
    if codebook_size <= 65536:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _code_bits(codebook_size: int) -> int:
    """Minimum fixed width that can represent one codebook index."""
    return max(1, int(math.ceil(math.log2(codebook_size))))


def _packed_nbytes(symbols: int, bits: int) -> int:
    return (int(symbols) * int(bits) + 7) // 8


def _pack_symbols(symbols: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned symbols into a little-endian fixed-width bit stream."""
    values = np.asarray(symbols).reshape(-1).astype(np.uint64, copy=False)
    if np.any(values >= (1 << bits)):
        raise ValueError("symbol does not fit in the configured bit width")
    if not len(values):
        return np.empty(0, dtype=np.uint8)
    bit_planes = ((values[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    return np.packbits(bit_planes.reshape(-1), bitorder="little")


def _unpack_symbols(
    payload: np.ndarray,
    bits: int,
    start_symbol: int,
    count: int,
) -> np.ndarray:
    """Decode a symbol slice without expanding the full bit stream."""
    count = int(count)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    data = np.asarray(payload, dtype=np.uint8).reshape(-1)
    starts = (int(start_symbol) + np.arange(count, dtype=np.int64)) * int(bits)
    result = np.zeros(count, dtype=np.uint64)
    for bit in range(bits):
        positions = starts + bit
        result |= (
            ((data[positions >> 3] >> (positions & 7)) & 1).astype(np.uint64) << bit
        )
    return result.astype(np.int64)


def _balanced_binary_weights(labels: np.ndarray) -> np.ndarray:
    """Give positive and negative candidates equal mass within one query."""
    labels = np.asarray(labels)
    positive = labels > 0
    negative = ~positive
    weights = np.zeros(len(labels), dtype=np.float64)
    if positive.any() and negative.any():
        weights[positive] = 0.5 / int(positive.sum())
        weights[negative] = 0.5 / int(negative.sum())
    elif len(labels):
        weights[:] = 1.0 / len(labels)
    return weights


def _bernoulli_kl_from_logits(
    teacher_logit: np.ndarray,
    student_logit: np.ndarray,
) -> np.ndarray:
    """KL(Bernoulli(teacher) || Bernoulli(student)) from stable logits."""
    teacher_logit = np.asarray(teacher_logit, dtype=np.float64)
    student_logit = np.asarray(student_logit, dtype=np.float64)
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(teacher_logit, -40.0, 40.0))
    )
    teacher_cross_entropy = (
        np.logaddexp(0.0, teacher_logit) - probability * teacher_logit
    )
    while probability.ndim < student_logit.ndim:
        probability = probability[..., None]
        teacher_cross_entropy = teacher_cross_entropy[..., None]
    value = (
        np.logaddexp(0.0, student_logit)
        - probability * student_logit
        - teacher_cross_entropy
    )
    # Roundoff can produce tiny negative values at identical logits.
    return np.maximum(value, 0.0)


@dataclass(frozen=True)
class RAVRRQConfig:
    seed: int = 20260818
    num_centroids: int = 16
    nprobe: int = 16
    stages: int = 8
    codebook_size: int = 16
    allowed_lengths: tuple[int, ...] = (2, 4, 6, 8)
    base_stages: int = 2
    rerank_candidates: int = 256
    retrieval_k: int = 10
    hard_negative_depth: int = 30
    objective: str = OBJECTIVE_ORDINAL_KL
    kmeans_iterations: int = 12
    train_samples: int = 20000
    norm_mode: str = "gram_tables"
    item_id_dtype: str = "uint32"
    allocation_search_steps: int = 60

    def validate(self, n_items: int | None = None) -> None:
        lengths = tuple(self.allowed_lengths)
        if not lengths or tuple(sorted(set(lengths))) != lengths:
            raise ValueError("allowed_lengths must be strictly increasing")
        if self.base_stages not in lengths:
            raise ValueError("base_stages must occur in allowed_lengths")
        if lengths[0] != self.base_stages or lengths[-1] > self.stages:
            raise ValueError("allowed lengths must span from base_stages to at most stages")
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if not (1 <= self.nprobe <= self.num_centroids):
            raise ValueError("nprobe must lie in [1, num_centroids]")
        if self.rerank_candidates < self.retrieval_k:
            raise ValueError("rerank_candidates must be at least retrieval_k")
        if self.objective not in {
            OBJECTIVE_ORDINAL_KL, OBJECTIVE_BOUNDARY_KL, OBJECTIVE_LEGACY,
        }:
            raise ValueError("objective must be ordinal_kl, boundary_kl, or legacy")
        if self.norm_mode not in {"gram_tables", "stored_final_norm"}:
            raise ValueError("norm_mode must be gram_tables or stored_final_norm")
        if np.dtype(self.item_id_dtype).kind != "u":
            raise ValueError("item_id_dtype must be unsigned")
        if n_items is not None:
            if n_items <= max(self.num_centroids, self.codebook_size):
                raise ValueError("gallery is too small for the requested codec")
            if n_items - 1 < self.retrieval_k + 1:
                raise ValueError("gallery is too small for retrieval_k")


@dataclass
class AllocationResult:
    option_ids: np.ndarray
    selected_bytes: int
    objective: float
    budget_bytes: int
    histogram: dict[str, int]
    strategy: str
    selected_bits: int | None = None
    budget_bits: int | None = None


def solve_lagrangian(
    distortion: np.ndarray,
    option_costs: np.ndarray,
    budget_bytes: int,
    search_steps: int = 60,
) -> np.ndarray:
    """Multiple-choice allocation with lower-cost tie breaking and repair."""
    distortion = np.asarray(distortion, dtype=np.float64)
    costs = np.asarray(option_costs, dtype=np.int64)
    if distortion.ndim != 2 or distortion.shape[1] != len(costs):
        raise ValueError("distortion and option_costs have incompatible shapes")
    if np.any(np.diff(costs) < 0) or costs[0] != 0:
        raise ValueError("option costs must increase from zero")
    if budget_bytes < 0:
        raise ValueError("variable byte budget cannot be negative")

    def choose(lam: float) -> tuple[np.ndarray, int]:
        # np.argmin keeps the first option, and costs are increasing.
        selected = np.argmin(distortion + lam * costs[None, :], axis=1)
        return selected, int(costs[selected].sum())

    best, used = choose(0.0)
    if used > budget_bytes:
        low, high = 0.0, 1.0
        while choose(high)[1] > budget_bytes:
            high *= 2.0
            if high > 1e18:
                raise RuntimeError("failed to bracket a feasible Lagrange multiplier")
        best = np.zeros(len(distortion), dtype=np.int64)
        best_used = 0
        for _ in range(search_steps):
            mid = (low + high) / 2.0
            current, current_used = choose(mid)
            if current_used <= budget_bytes:
                high = mid
                if current_used > best_used or (
                    current_used == best_used
                    and distortion[np.arange(len(distortion)), current].sum()
                    < distortion[np.arange(len(distortion)), best].sum()
                ):
                    best, best_used = current, current_used
            else:
                low = mid
        used = best_used

    best = best.copy()
    # Discrete positive-gain repair.  Recompute a small vector of next upgrades
    # after each accepted move; this avoids N*|C| Python objects.
    while True:
        eligible = best + 1 < distortion.shape[1]
        if not eligible.any():
            break
        rows = np.flatnonzero(eligible)
        nxt = best[rows] + 1
        delta_cost = costs[nxt] - costs[best[rows]]
        affordable = used + delta_cost <= budget_bytes
        if not affordable.any():
            break
        rows = rows[affordable]
        nxt = best[rows] + 1
        delta_cost = costs[nxt] - costs[best[rows]]
        gain = distortion[rows, best[rows]] - distortion[rows, nxt]
        positive = gain > 0
        if not positive.any():
            break
        rows, delta_cost, gain = rows[positive], delta_cost[positive], gain[positive]
        ratio = gain / delta_cost
        # Deterministic: best ratio, then largest gain, then lowest row id.
        pick = np.lexsort((rows, -gain, -ratio))[0]
        row = int(rows[pick])
        used += int(delta_cost[pick])
        best[row] += 1
    return best


def solve_exact_small(
    distortion: np.ndarray, option_costs: np.ndarray, budget_bytes: int
) -> np.ndarray:
    """Exact dynamic program for allocator tests and small-scale diagnostics."""
    distortion = np.asarray(distortion, dtype=np.float64)
    costs = np.asarray(option_costs, dtype=np.int64)
    n = len(distortion)
    inf = np.inf
    dp = np.full((n + 1, budget_bytes + 1), inf)
    parent = np.full((n + 1, budget_bytes + 1), -1, dtype=np.int16)
    previous_budget = np.full((n + 1, budget_bytes + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(n):
        reachable = np.flatnonzero(np.isfinite(dp[i]))
        for spent in reachable:
            for option, cost in enumerate(costs):
                target = spent + int(cost)
                if target > budget_bytes:
                    continue
                value = dp[i, spent] + distortion[i, option]
                if value < dp[i + 1, target] - 1e-12:
                    dp[i + 1, target] = value
                    parent[i + 1, target] = option
                    previous_budget[i + 1, target] = spent
    end = int(np.argmin(dp[n]))
    if not np.isfinite(dp[n, end]):
        raise RuntimeError("exact allocator found no feasible solution")
    selected = np.empty(n, dtype=np.int64)
    for i in range(n, 0, -1):
        selected[i - 1] = parent[i, end]
        end = int(previous_budget[i, end])
    return selected


class RAVRRQTrainer:
    """Train one codec/calibration state and build several equal-codec indexes."""

    def __init__(self, config: RAVRRQConfig | None = None):
        self.config = config or RAVRRQConfig()

    def fit(
        self,
        gallery: np.ndarray,
        calibration_queries: np.ndarray,
        calibration_gallery_ids: np.ndarray | None = None,
        validation_queries: np.ndarray | None = None,
        validation_gallery_ids: np.ndarray | None = None,
        codec_train_ids: np.ndarray | None = None,
    ) -> "RAVRRQTrainer":
        started = time.perf_counter()
        x = _normalise(gallery, "gallery")
        qcal = _normalise(calibration_queries, "calibration_queries")
        self.config.validate(len(x))
        if qcal.shape[1] != x.shape[1]:
            raise ValueError("gallery and calibration query dimensions differ")
        if calibration_gallery_ids is not None:
            calibration_gallery_ids = np.asarray(calibration_gallery_ids, dtype=np.int64)
            if calibration_gallery_ids.shape != (len(qcal),):
                raise ValueError("calibration_gallery_ids has the wrong shape")
        if validation_queries is None and validation_gallery_ids is not None:
            raise ValueError("validation_gallery_ids requires validation_queries")
        self.x = x
        self.dimension = x.shape[1]
        self.n_items = len(x)
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        if codec_train_ids is None:
            codec_pool = np.arange(len(x), dtype=np.int64)
        else:
            codec_pool = np.asarray(codec_train_ids, dtype=np.int64)
            if codec_pool.ndim != 1 or not len(codec_pool):
                raise ValueError("codec_train_ids must be a non-empty vector")
            if np.any(codec_pool < 0) or np.any(codec_pool >= len(x)):
                raise ValueError("codec_train_ids contains an out-of-range item")
            if len(np.unique(codec_pool)) != len(codec_pool):
                raise ValueError("codec_train_ids contains duplicates")
        sample_ids = rng.choice(
            codec_pool, min(cfg.train_samples, len(codec_pool)), replace=False
        )
        train = x[sample_ids]

        centroids = _spherical_kmeans(
            train, cfg.num_centroids, cfg.seed, cfg.kmeans_iterations
        )
        # All index-time calculations use the actual stored FP16 codec values.
        self.centroids = centroids.astype(np.float16).astype(np.float32)
        self.cluster_ids = np.argmax(x @ self.centroids.T, axis=1).astype(np.int32)
        residual = x - self.centroids[self.cluster_ids]
        train_residual = residual[sample_ids].copy()
        full_residual = residual.copy()
        codebooks = []
        code_dtype = _code_dtype(cfg.codebook_size)
        full_codes = np.empty((len(x), cfg.stages), dtype=code_dtype)
        for stage in range(cfg.stages):
            cb = _euclidean_kmeans(
                train_residual,
                cfg.codebook_size,
                cfg.seed + 104729 * (stage + 1),
                cfg.kmeans_iterations,
            ).astype(np.float16).astype(np.float32)
            codebooks.append(cb)
            train_assignment = _nearest(train_residual, cb)
            train_residual -= cb[train_assignment]
            assignment = _nearest(full_residual, cb)
            full_codes[:, stage] = assignment.astype(code_dtype)
            full_residual -= cb[assignment]
        self.codebooks = np.stack(codebooks)
        self.full_codes = full_codes

        self._compute_prefix_statistics()
        if validation_queries is not None:
            qval = _normalise(validation_queries, "validation_queries")
            if qval.shape[1] != self.dimension:
                raise ValueError("gallery and validation query dimensions differ")
            if validation_gallery_ids is not None:
                validation_gallery_ids = np.asarray(validation_gallery_ids, dtype=np.int64)
                if validation_gallery_ids.shape != (len(qval),):
                    raise ValueError("validation_gallery_ids has the wrong shape")
            self._select_rerank_candidates(qval, validation_gallery_ids)
        else:
            self.validation_report = {
                "queries": 0,
                "candidate_sweep": {},
                "selected_rerank_candidates": self.config.rerank_candidates,
                "selection_target": 0.99,
            }
        self._calibrate(qcal, calibration_gallery_ids)
        self.gallery_hash = hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
        codec_bytes = self.centroids.tobytes() + self.codebooks.tobytes()
        self.codec_hash = hashlib.sha256(codec_bytes).hexdigest()
        config_json = json.dumps(asdict(self.config), sort_keys=True, separators=(",", ":"))
        self.config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        self.fit_seconds = time.perf_counter() - started
        return self

    def save_state(self, path: str) -> str:
        """Save reusable codec/calibration state; gallery vectors stay external."""
        if not hasattr(self, "distortion"):
            raise RuntimeError("fit must be called before save_state")
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            config_json=np.array(json.dumps(asdict(self.config), sort_keys=True)),
            calibration_json=np.array(json.dumps(self.calibration_report, sort_keys=True)),
            gallery_hash=np.array(self.gallery_hash),
            codec_hash=np.array(self.codec_hash),
            config_hash=np.array(self.config_hash),
            fit_seconds=np.array(self.fit_seconds, dtype=np.float64),
            centroids=self.centroids,
            cluster_ids=self.cluster_ids,
            codebooks=self.codebooks,
            full_codes=self.full_codes,
            prefix_norms=self.prefix_norms,
            prior_distortion=self.prior_distortion,
            base_inverse_norm=self.base_inverse_norm,
            distortion=self.distortion,
            retrieval_distortion=self.retrieval_distortion,
            coverage=self.coverage,
        )
        return path

    @classmethod
    def load_state(cls, path: str, gallery: np.ndarray) -> "RAVRRQTrainer":
        state = np.load(path, allow_pickle=False)
        config_data = json.loads(str(state["config_json"]))
        config_data["allowed_lengths"] = tuple(config_data["allowed_lengths"])
        legacy_parameters = {
            name: config_data.pop(name)
            for name in LEGACY_OBJECTIVE_PARAMETERS
            if name in config_data
        }
        for name in LEGACY_UNUSED_CONFIG_FIELDS:
            config_data.pop(name, None)
        if "objective" not in config_data:
            config_data["objective"] = OBJECTIVE_LEGACY
        trainer = cls(RAVRRQConfig(**config_data))
        trainer.legacy_objective_parameters = {
            **LEGACY_OBJECTIVE_PARAMETERS,
            **legacy_parameters,
        }
        trainer.x = _normalise(gallery, "gallery")
        trainer.n_items, trainer.dimension = trainer.x.shape
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(trainer.x).view(np.uint8)
        ).hexdigest()
        if expected_hash != str(state["gallery_hash"]):
            raise ValueError("gallery does not match the saved trainer state")
        for name in (
            "centroids", "cluster_ids", "codebooks", "full_codes",
            "prefix_norms", "prior_distortion", "base_inverse_norm",
            "distortion", "retrieval_distortion", "coverage",
        ):
            setattr(trainer, name, np.array(state[name]))
        trainer.gallery_hash = expected_hash
        trainer.codec_hash = str(state["codec_hash"])
        trainer.config_hash = str(state["config_hash"])
        trainer.fit_seconds = float(state["fit_seconds"])
        trainer.calibration_report = json.loads(str(state["calibration_json"]))
        trainer.validation_report = trainer.calibration_report.get("validation", {})
        return trainer

    def _compute_prefix_statistics(self) -> None:
        cfg = self.config
        option_for_length = {c: j for j, c in enumerate(cfg.allowed_lengths)}
        n_options = len(cfg.allowed_lengths)
        self.prefix_norms = np.empty((self.n_items, n_options), dtype=np.float32)
        self.prior_distortion = np.empty((self.n_items, n_options), dtype=np.float32)
        reconstruction = self.centroids[self.cluster_ids].copy()
        rows = np.arange(self.n_items)
        for stage in range(cfg.stages):
            reconstruction += self.codebooks[stage, self.full_codes[:, stage].astype(np.int64)]
            length = stage + 1
            if length not in option_for_length:
                continue
            option = option_for_length[length]
            norm = np.maximum(np.linalg.norm(reconstruction, axis=1), EPS)
            self.prefix_norms[:, option] = norm
            cosine = np.sum(self.x * reconstruction, axis=1) / norm
            self.prior_distortion[:, option] = 1.0 - cosine
        base_option = option_for_length[cfg.base_stages]
        self.base_inverse_norm = 1.0 / self.prefix_norms[:, base_option]

    def _base_candidates(
        self,
        query: np.ndarray,
        rerank_candidates: int | None = None,
        exclude_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        centroid_scores = self.centroids @ query
        probed = _top_indices(centroid_scores, cfg.nprobe)
        mask = np.isin(self.cluster_ids, probed)
        item_ids = np.flatnonzero(mask)
        lut = np.einsum("tld,d->tl", self.codebooks, query)
        numerator = centroid_scores[self.cluster_ids[item_ids]].copy()
        for stage in range(cfg.base_stages):
            numerator += lut[stage, self.full_codes[item_ids, stage].astype(np.int64)]
        scores = numerator * self.base_inverse_norm[item_ids]
        if exclude_id is not None:
            scores[item_ids == int(exclude_id)] = -np.inf
        count = cfg.rerank_candidates if rerank_candidates is None else int(rerank_candidates)
        local = _top_indices(scores, min(count, len(item_ids)))
        return item_ids[local], scores[local]

    def _select_rerank_candidates(
        self, queries: np.ndarray, self_ids: np.ndarray | None = None
    ) -> None:
        """Select the smallest validation R reaching mean CandidateRecall 0.99."""
        cfg = self.config
        candidates = sorted(set([
            cfg.retrieval_k,
            min(cfg.rerank_candidates, max(5 * cfg.retrieval_k, cfg.retrieval_k)),
            min(cfg.rerank_candidates, max(10 * cfg.retrieval_k, cfg.retrieval_k)),
            min(cfg.rerank_candidates, max(20 * cfg.retrieval_k, cfg.retrieval_k)),
            max(cfg.retrieval_k, cfg.rerank_candidates // 2),
            cfg.rerank_candidates,
        ]))
        recalls = {count: [] for count in candidates}
        maximum = candidates[-1]
        for qi, query in enumerate(queries):
            excluded = None if self_ids is None else int(self_ids[qi])
            score = self.x @ query
            if excluded is not None and excluded >= 0:
                score[excluded] = -np.inf
            teacher = _top_indices(score, cfg.retrieval_k)
            ranked, _ = self._base_candidates(
                query, rerank_candidates=maximum, exclude_id=excluded
            )
            teacher_set = set(teacher)
            for count in candidates:
                recalls[count].append(
                    len(set(ranked[:count]) & teacher_set) / cfg.retrieval_k
                )
        sweep = {str(count): float(np.mean(recalls[count])) for count in candidates}
        selected = candidates[-1]
        for count in candidates:
            if sweep[str(count)] >= 0.99:
                selected = count
                break
        self.config = replace(cfg, rerank_candidates=int(selected))
        self.validation_report = {
            "queries": int(len(queries)),
            "candidate_sweep": sweep,
            "selected_rerank_candidates": int(selected),
            "selection_target": 0.99,
        }

    def _prefix_scores(
        self, query: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        centroid_scores = self.centroids @ query
        lut = np.einsum("tld,d->tl", self.codebooks, query)
        result = np.empty((len(item_ids), len(cfg.allowed_lengths)), dtype=np.float32)
        numerator = centroid_scores[self.cluster_ids[item_ids]].copy()
        option_for_length = {c: j for j, c in enumerate(cfg.allowed_lengths)}
        for stage in range(cfg.stages):
            numerator += lut[stage, self.full_codes[item_ids, stage].astype(np.int64)]
            length = stage + 1
            if length in option_for_length:
                option = option_for_length[length]
                result[:, option] = numerator / self.prefix_norms[item_ids, option]
        return result

    def _calibrate(
        self, queries: np.ndarray, self_ids: np.ndarray | None
    ) -> None:
        cfg = self.config
        if cfg.objective not in {OBJECTIVE_ORDINAL_KL, OBJECTIVE_BOUNDARY_KL}:
            raise ValueError(
                "new calibration supports ordinal_kl or boundary_kl; "
                "legacy curves remain loadable for ablation reproduction"
            )
        retrieval_component = np.zeros_like(
            self.prior_distortion, dtype=np.float64
        )
        coverage = np.zeros(self.n_items, dtype=np.int32)
        candidate_recall = []
        boundary_misses = 0
        depth = min(
            self.n_items,
            cfg.retrieval_k + max(cfg.hard_negative_depth, 1) + 1,
        )
        for qi, query in enumerate(queries):
            full_score = self.x @ query
            if self_ids is not None and self_ids[qi] >= 0:
                full_score[int(self_ids[qi])] = -np.inf
            teacher = _top_indices(full_score, depth)
            positive = teacher[:cfg.retrieval_k]
            kth = float(full_score[teacher[cfg.retrieval_k - 1]])
            next_score = float(full_score[teacher[cfg.retrieval_k]])
            tau = 0.5 * (kth + next_score)
            scale_index = min(cfg.retrieval_k + cfg.hard_negative_depth - 1, len(teacher) - 1)
            sigma = max(kth - float(full_score[teacher[scale_index]]), EPS)

            excluded = None if self_ids is None else int(self_ids[qi])
            candidates, base_scores = self._base_candidates(query, exclude_id=excluded)
            candidate_is_positive = np.isin(candidates, positive)
            reachable = int(candidate_is_positive.sum())
            candidate_recall.append(reachable / cfg.retrieval_k)
            boundary_misses += cfg.retrieval_k - reachable
            if not len(candidates):
                continue

            # Enhancement can only help reachable candidates, but evaluating every
            # item in a large top-R set is wasteful and gives easy negatives too much
            # influence.  Mine the bounded hard set required by the specification.
            reachable_positive = candidates[candidate_is_positive]
            negative = candidates[~candidate_is_positive]
            negative_base = base_scores[~candidate_is_positive]
            hard_depth = min(cfg.hard_negative_depth, len(negative))
            if hard_depth:
                boundary_order = np.argsort(np.abs(full_score[negative] - tau))[:hard_depth]
                overestimate = negative_base - full_score[negative]
                intruder_order = _top_indices(overestimate, hard_depth)
                high_base_order = _top_indices(negative_base, hard_depth)
                candidates = np.unique(np.concatenate((
                    reachable_positive,
                    negative[boundary_order],
                    negative[intruder_order],
                    negative[high_base_order],
                )))
            else:
                candidates = reachable_positive
            coverage[candidates] += 1
            if not len(candidates):
                continue

            is_positive = np.isin(candidates, positive)
            approximate = self._prefix_scores(query, candidates)
            if cfg.objective == OBJECTIVE_BOUNDARY_KL:
                labels = np.where(is_positive, 1.0, -1.0)
                weights = _balanced_binary_weights(labels)
                teacher_logit = (full_score[candidates] - tau) / sigma
                student_logit = (approximate - tau) / sigma
                loss = _bernoulli_kl_from_logits(teacher_logit, student_logit)
                contribution = weights[:, None] * loss / len(queries)
            else:
                candidate_rank = np.full(
                    len(candidates), cfg.retrieval_k, dtype=np.int64
                )
                for rank, item in enumerate(positive):
                    candidate_rank[candidates == item] = rank
                contribution = np.zeros_like(approximate, dtype=np.float64)
                for cutoff in range(1, cfg.retrieval_k + 1):
                    cutoff_tau = 0.5 * (
                        float(full_score[teacher[cutoff - 1]])
                        + float(full_score[teacher[cutoff]])
                    )
                    labels = np.where(candidate_rank < cutoff, 1.0, -1.0)
                    weights = _balanced_binary_weights(labels)
                    teacher_logit = (
                        full_score[candidates] - cutoff_tau
                    ) / sigma
                    student_logit = (approximate - cutoff_tau) / sigma
                    contribution += weights[:, None] * _bernoulli_kl_from_logits(
                        teacher_logit, student_logit
                    )
                contribution /= cfg.retrieval_k * len(queries)
            np.add.at(retrieval_component, candidates, contribution)

        self.distortion = retrieval_component.astype(np.float32)
        self.retrieval_distortion = retrieval_component.astype(np.float32)
        self.coverage = coverage
        self.calibration_report = {
            "queries": int(len(queries)),
            "candidate_recall_at_k": float(np.mean(candidate_recall)),
            "candidate_recall_min": float(np.min(candidate_recall)),
            "unreachable_teacher_positives": int(boundary_misses),
            "coverage_fraction": float(np.mean(coverage > 0)),
            "objective": cfg.objective,
            "objective_hyperparameters": {},
            "class_balance": "equal positive and negative mass per query and cutoff",
            "ordinal_cutoffs": (
                list(range(1, cfg.retrieval_k + 1))
                if cfg.objective == OBJECTIVE_ORDINAL_KL else [cfg.retrieval_k]
            ),
            "validation": self.validation_report,
        }

    def recalibrate(
        self,
        calibration_queries: np.ndarray,
        calibration_gallery_ids: np.ndarray | None = None,
        objective: str = OBJECTIVE_ORDINAL_KL,
    ) -> "RAVRRQTrainer":
        """Recompute allocation curves while preserving the trained codec."""
        queries = _normalise(calibration_queries, "calibration_queries")
        if queries.shape[1] != self.dimension:
            raise ValueError("gallery and calibration query dimensions differ")
        if calibration_gallery_ids is not None:
            calibration_gallery_ids = np.asarray(
                calibration_gallery_ids, dtype=np.int64
            )
            if calibration_gallery_ids.shape != (len(queries),):
                raise ValueError("calibration_gallery_ids has the wrong shape")
        self.config = replace(self.config, objective=objective)
        self._calibrate(queries, calibration_gallery_ids)
        return self

    def _base_array_specs(self) -> list[tuple[str, np.dtype, tuple[int, ...]]]:
        cfg = self.config
        k = cfg.num_centroids
        buckets = len(cfg.allowed_lengths)
        bits = _code_bits(cfg.codebook_size)
        specs: list[tuple[str, np.dtype, tuple[int, ...]]] = [
            ("centroids", np.dtype(np.float16), (k, self.dimension)),
            ("codebooks", np.dtype(np.float16), (cfg.stages, cfg.codebook_size, self.dimension)),
            ("item_spans", np.dtype(np.uint64), (k, buckets, 2)),
            ("enhancement_spans", np.dtype(np.uint64), (k, buckets, 2)),
            ("item_ids", np.dtype(cfg.item_id_dtype), (self.n_items,)),
            ("base_codes", np.dtype(np.uint8),
             (_packed_nbytes(self.n_items * cfg.base_stages, bits),)),
            ("base_inverse_norm", np.dtype(np.float16), (self.n_items,)),
        ]
        if cfg.norm_mode == "stored_final_norm":
            specs.append(("final_inverse_norm", np.dtype(np.float16), (self.n_items,)))
        return specs

    def minimum_persistent_bytes(self) -> int:
        offset = HEADER_BYTES
        for _, dtype, shape in self._base_array_specs():
            offset = _align(offset)
            offset += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return _align(offset)

    def option_costs(self) -> np.ndarray:
        """Legacy byte-aligned enhancement costs, retained for API compatibility."""
        width = _code_dtype(self.config.codebook_size).itemsize
        return np.array(
            [(c - self.config.base_stages) * width for c in self.config.allowed_lengths],
            dtype=np.int64,
        )

    def option_cost_bits(self) -> np.ndarray:
        bits = _code_bits(self.config.codebook_size)
        return np.array(
            [(c - self.config.base_stages) * bits for c in self.config.allowed_lengths],
            dtype=np.int64,
        )

    @property
    def code_bits(self) -> int:
        return _code_bits(self.config.codebook_size)

    def _enhancement_packed_nbytes(self, selected: np.ndarray) -> int:
        cfg = self.config
        buckets = len(cfg.allowed_lengths)
        keys = self.cluster_ids.astype(np.int64) * buckets + np.asarray(selected)
        counts = np.bincount(
            keys, minlength=cfg.num_centroids * buckets
        ).reshape(cfg.num_centroids, buckets)
        return int(sum(
            _packed_nbytes(
                int(counts[cluster, bucket])
                * (length - cfg.base_stages),
                self.code_bits,
            )
            for cluster in range(cfg.num_centroids)
            for bucket, length in enumerate(cfg.allowed_lengths)
        ))

    def fixed_length_budget(self, length: int) -> int:
        """Exact serialized bytes of an all-items fixed-prefix index."""
        if length not in self.config.allowed_lengths:
            raise ValueError("length must occur in allowed_lengths")
        option = self.config.allowed_lengths.index(length)
        selected = np.full(self.n_items, option, dtype=np.int64)
        return self.minimum_persistent_bytes() + self._enhancement_packed_nbytes(selected)

    def _repair_packing_budget(
        self, selected: np.ndarray, budget_bytes: int
    ) -> np.ndarray:
        """Remove the few lowest-value upgrades needed for byte-aligned groups."""
        selected = np.asarray(selected, dtype=np.int64).copy()
        available = int(budget_bytes) - self.minimum_persistent_bytes()
        if self._enhancement_packed_nbytes(selected) <= available:
            return selected
        lengths = np.asarray(self.config.allowed_lengths)
        while self._enhancement_packed_nbytes(selected) > available:
            starting_bytes = self._enhancement_packed_nbytes(selected)
            candidates = np.flatnonzero(selected > 0)
            if not len(candidates):
                raise AssertionError("base-only bit stream exceeds the persistent budget")
            current = selected[candidates]
            saved = (lengths[current] - lengths[current - 1]) * self.code_bits
            penalty = (
                self.distortion[candidates, current - 1]
                - self.distortion[candidates, current]
            )
            order = candidates[np.lexsort((candidates, penalty / saved))]
            for rank, item in enumerate(order, 1):
                selected[item] -= 1
                if rank % 32 == 0:
                    current_bytes = self._enhancement_packed_nbytes(selected)
                    if current_bytes <= available:
                        return selected
            if self._enhancement_packed_nbytes(selected) >= starting_bytes:
                raise AssertionError("could not repair grouped bit-packing overhead")
        return selected

    def build(
        self,
        budget_bytes: int,
        strategy: str = "boundary",
        random_seed: int | None = None,
    ) -> "RAVRRQIndex":
        if not hasattr(self, "distortion"):
            raise RuntimeError("fit must be called before build")
        minimum = self.minimum_persistent_bytes()
        if budget_bytes < minimum:
            raise ValueError(
                f"budget {budget_bytes} is below the minimum legal index size {minimum}"
            )
        costs = self.option_cost_bits()
        variable_budget = int(budget_bytes - minimum) * 8
        if strategy == "boundary":
            selected = solve_lagrangian(
                self.distortion, costs, variable_budget, self.config.allocation_search_steps
            )
        elif strategy == "prior":
            selected = solve_lagrangian(
                self.prior_distortion, costs, variable_budget, self.config.allocation_search_steps
            )
        elif strategy == "random_histogram":
            selected = solve_lagrangian(
                self.distortion, costs, variable_budget, self.config.allocation_search_steps
            )
            rng = np.random.default_rng(self.config.seed if random_seed is None else random_seed)
            selected = rng.permutation(selected)
        elif strategy == "within_cluster_random":
            selected = solve_lagrangian(
                self.distortion, costs, variable_budget, self.config.allocation_search_steps
            )
            rng = np.random.default_rng(self.config.seed if random_seed is None else random_seed)
            selected = selected.copy()
            for cluster in range(self.config.num_centroids):
                members = np.flatnonzero(self.cluster_ids == cluster)
                selected[members] = rng.permutation(selected[members])
        elif strategy == "cluster_fixed":
            cluster_distortion = np.stack([
                self.distortion[self.cluster_ids == cluster].sum(axis=0)
                for cluster in range(self.config.num_centroids)
            ])
            cluster_sizes = np.bincount(
                self.cluster_ids, minlength=self.config.num_centroids
            ).astype(np.int64)
            cluster_costs = cluster_sizes[:, None] * costs[None, :]

            def choose_clusters(lam: float) -> tuple[np.ndarray, int]:
                choice = np.argmin(cluster_distortion + lam * cluster_costs, axis=1)
                return choice, int(cluster_costs[np.arange(len(choice)), choice].sum())

            cluster_choice, cluster_used = choose_clusters(0.0)
            if cluster_used > variable_budget:
                low, high = 0.0, 1.0
                while choose_clusters(high)[1] > variable_budget:
                    high *= 2.0
                best_choice = np.zeros(self.config.num_centroids, dtype=np.int64)
                best_used = 0
                for _ in range(self.config.allocation_search_steps):
                    mid = (low + high) / 2.0
                    choice, used_now = choose_clusters(mid)
                    if used_now <= variable_budget:
                        high = mid
                        if used_now > best_used:
                            best_choice, best_used = choice, used_now
                    else:
                        low = mid
                cluster_choice, cluster_used = best_choice, best_used
            # Positive-gain, whole-cluster repair.
            while True:
                best_upgrade = None
                for cluster in range(self.config.num_centroids):
                    current = int(cluster_choice[cluster])
                    if current + 1 >= len(costs):
                        continue
                    delta = int(cluster_costs[cluster, current + 1]
                                - cluster_costs[cluster, current])
                    gain = float(cluster_distortion[cluster, current]
                                 - cluster_distortion[cluster, current + 1])
                    if delta <= 0 or gain <= 0 or cluster_used + delta > variable_budget:
                        continue
                    candidate = (gain / delta, gain, -cluster, cluster, delta)
                    if best_upgrade is None or candidate > best_upgrade:
                        best_upgrade = candidate
                if best_upgrade is None:
                    break
                cluster = best_upgrade[3]
                cluster_used += best_upgrade[4]
                cluster_choice[cluster] += 1
            selected = cluster_choice[self.cluster_ids]
        elif strategy == "fixed":
            feasible = np.flatnonzero(self.n_items * costs <= variable_budget)
            if not len(feasible):
                raise RuntimeError("base fixed-rate option should always be feasible")
            totals = self.distortion[:, feasible].sum(axis=0)
            selected = np.full(self.n_items, feasible[int(np.argmin(totals))], dtype=np.int64)
        else:
            raise ValueError(f"unknown allocation strategy: {strategy}")

        selected = self._repair_packing_budget(selected, budget_bytes)
        used = int(costs[selected].sum())
        if used > variable_budget:
            raise AssertionError("allocator exceeded the variable bit budget")
        lengths = np.asarray(self.config.allowed_lengths)[selected]
        histogram = {
            str(length): int(np.sum(lengths == length))
            for length in self.config.allowed_lengths
        }
        allocation = AllocationResult(
            option_ids=selected,
            selected_bytes=(used + 7) // 8,
            objective=float(self.distortion[np.arange(self.n_items), selected].sum()),
            budget_bytes=(variable_budget + 7) // 8,
            histogram=histogram,
            strategy=strategy,
            selected_bits=used,
            budget_bits=variable_budget,
        )
        return self._pack(allocation, budget_bytes)

    def _pack(self, allocation: AllocationResult, budget_bytes: int) -> "RAVRRQIndex":
        cfg = self.config
        k, buckets = cfg.num_centroids, len(cfg.allowed_lengths)
        selected = allocation.option_ids
        keys = self.cluster_ids.astype(np.int64) * buckets + selected
        order = np.argsort(keys, kind="stable")
        counts = np.bincount(keys, minlength=k * buckets).reshape(k, buckets)
        item_spans = np.empty((k, buckets, 2), dtype=np.uint64)
        enhancement_spans = np.empty((k, buckets, 2), dtype=np.uint64)
        item_cursor = 0
        enhancement_cursor = 0
        bits = _code_bits(cfg.codebook_size)
        for cluster in range(k):
            for bucket, length in enumerate(cfg.allowed_lengths):
                count = int(counts[cluster, bucket])
                item_spans[cluster, bucket] = (item_cursor, item_cursor + count)
                width = length - cfg.base_stages
                packed_width = _packed_nbytes(count * width, bits)
                enhancement_spans[cluster, bucket] = (
                    enhancement_cursor,
                    enhancement_cursor + packed_width,
                )
                item_cursor += count
                enhancement_cursor += packed_width

        enhancements = np.empty(enhancement_cursor, dtype=np.uint8)
        for cluster in range(k):
            for bucket, length in enumerate(cfg.allowed_lengths):
                row_start, row_end = item_spans[cluster, bucket]
                enh_start, enh_end = enhancement_spans[cluster, bucket]
                source = order[int(row_start):int(row_end)]
                width = length - cfg.base_stages
                if width:
                    enhancements[int(enh_start):int(enh_end)] = _pack_symbols(
                        self.full_codes[source, cfg.base_stages:length].reshape(-1),
                        bits,
                    )

        arrays: dict[str, np.ndarray] = {
            "centroids": self.centroids.astype(np.float16),
            "codebooks": self.codebooks.astype(np.float16),
            "item_spans": item_spans,
            "enhancement_spans": enhancement_spans,
            "item_ids": order.astype(cfg.item_id_dtype),
            "base_codes": _pack_symbols(
                self.full_codes[order, :cfg.base_stages].reshape(-1), bits
            ),
            "base_inverse_norm": self.base_inverse_norm[order].astype(np.float16),
        }
        if cfg.norm_mode == "stored_final_norm":
            arrays["final_inverse_norm"] = (
                1.0 / self.prefix_norms[np.arange(self.n_items), selected]
            )[order].astype(np.float16)
        arrays["enhancements"] = enhancements
        index = RAVRRQIndex(
            config=cfg,
            arrays=arrays,
            allocation=allocation,
            gallery_hash=self.gallery_hash,
            codec_hash=self.codec_hash,
            config_hash=self.config_hash,
            calibration_report=self.calibration_report,
            target_budget_bytes=int(budget_bytes),
            format_version=FORMAT_VERSION,
        )
        if index.persistent_bytes > budget_bytes:
            raise AssertionError(
                f"packed index uses {index.persistent_bytes} bytes, budget is {budget_bytes}"
            )
        return index


class RAVRRQIndex:
    """Packed serving index with exact lookup/Gram-table cosine scoring."""

    def __init__(
        self,
        config: RAVRRQConfig,
        arrays: dict[str, np.ndarray],
        allocation: AllocationResult,
        gallery_hash: str,
        codec_hash: str,
        config_hash: str,
        calibration_report: dict,
        target_budget_bytes: int,
        path: str | None = None,
        format_version: int = FORMAT_VERSION,
    ):
        self.config = config
        self.arrays = arrays
        self.allocation = allocation
        self.gallery_hash = gallery_hash
        self.codec_hash = codec_hash
        self.config_hash = config_hash
        self.calibration_report = calibration_report
        self.target_budget_bytes = target_budget_bytes
        self.path = path
        self.format_version = int(format_version)
        self._layout, self.persistent_bytes = self._compute_layout(arrays)
        self._build_norm_tables()

    @staticmethod
    def _compute_layout(
        arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, dict], int]:
        layout: dict[str, dict] = {}
        offset = HEADER_BYTES
        for name, array in arrays.items():
            offset = _align(offset)
            layout[name] = {
                "offset": offset,
                "dtype": np.dtype(array.dtype).str,
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            }
            offset += int(array.nbytes)
        return layout, offset

    def _build_norm_tables(self) -> None:
        centroids = np.asarray(self.arrays["centroids"], dtype=np.float32)
        codebooks = np.asarray(self.arrays["codebooks"], dtype=np.float32)
        self._m0 = np.sum(centroids * centroids, axis=1)
        if self.config.norm_mode == "gram_tables":
            self._m1 = np.einsum("kd,tld->ktl", centroids, codebooks)
            self._m2 = np.sum(codebooks * codebooks, axis=2)
            t, l = self.config.stages, self.config.codebook_size
            self._m3 = np.zeros((t, t, l, l), dtype=np.float32)
            for s in range(t):
                for u in range(s + 1, t):
                    self._m3[s, u] = codebooks[s] @ codebooks[u].T

    @property
    def n_items(self) -> int:
        return int(len(self.arrays["item_ids"]))

    @property
    def dimension(self) -> int:
        return int(self.arrays["centroids"].shape[1])

    @property
    def resident_bytes(self) -> int:
        total = sum(int(array.nbytes) for array in self.arrays.values())
        total += int(self._m0.nbytes)
        if self.config.norm_mode == "gram_tables":
            total += int(self._m1.nbytes + self._m2.nbytes + self._m3.nbytes)
        return total

    @property
    def bytes_per_item(self) -> float:
        return self.persistent_bytes / self.n_items

    @property
    def code_only_bytes(self) -> float:
        """Nominal packed symbols, excluding codec and index metadata."""
        lengths = self.assigned_lengths_by_id().astype(np.int64)
        return float(lengths.sum() * _code_bits(self.config.codebook_size) / 8.0)

    @property
    def code_only_bytes_per_item(self) -> float:
        return self.code_only_bytes / self.n_items

    @property
    def packed_code_bytes(self) -> int:
        return int(
            self.arrays["base_codes"].nbytes + self.arrays["enhancements"].nbytes
        )

    def storage_breakdown(self) -> dict[str, int | float]:
        arrays = {name: int(value.nbytes) for name, value in self.arrays.items()}
        payload = sum(arrays.values())
        return {
            "header_and_alignment": int(self.persistent_bytes - payload),
            "ivf_centroids": arrays["centroids"],
            "codebooks": arrays["codebooks"],
            "codes": self.packed_code_bytes,
            "ids": arrays["item_ids"],
            "norms": arrays["base_inverse_norm"]
            + arrays.get("final_inverse_norm", 0),
            "length_metadata": arrays["item_spans"] + arrays["enhancement_spans"],
            "persistent_bytes": int(self.persistent_bytes),
            "code_only_bytes": self.code_only_bytes,
        }

    def _group_codes(
        self, cluster: int, bucket: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cfg = self.config
        start, end = self.arrays["item_spans"][cluster, bucket]
        start, end = int(start), int(end)
        if self.format_version == 1:
            base = np.asarray(self.arrays["base_codes"][start:end], dtype=np.int64)
        else:
            base = _unpack_symbols(
                self.arrays["base_codes"], _code_bits(cfg.codebook_size),
                start * cfg.base_stages, (end - start) * cfg.base_stages,
            ).reshape(end - start, cfg.base_stages)
        length = cfg.allowed_lengths[bucket]
        width = length - cfg.base_stages
        if width:
            es, ee = self.arrays["enhancement_spans"][cluster, bucket]
            payload = self.arrays["enhancements"][int(es):int(ee)]
            if self.format_version == 1:
                enhancement = np.asarray(payload, dtype=np.int64).reshape(
                    end - start, width
                )
            else:
                enhancement = _unpack_symbols(
                    payload, _code_bits(cfg.codebook_size), 0,
                    (end - start) * width,
                ).reshape(end - start, width)
        else:
            enhancement = np.empty((end - start, 0), dtype=np.int64)
        return base, enhancement, np.asarray(self.arrays["item_ids"][start:end]), start

    def _item_codes(self, row: int, cluster: int, bucket: int) -> np.ndarray:
        cfg = self.config
        if self.format_version == 1:
            base = np.asarray(self.arrays["base_codes"][row], dtype=np.int64)
        else:
            base = _unpack_symbols(
                self.arrays["base_codes"], _code_bits(cfg.codebook_size),
                int(row) * cfg.base_stages, cfg.base_stages,
            )
        length = cfg.allowed_lengths[int(bucket)]
        width = length - cfg.base_stages
        if not width:
            return base
        item_start = int(self.arrays["item_spans"][cluster, bucket, 0])
        enhancement_start, enhancement_end = self.arrays["enhancement_spans"][
            cluster, bucket
        ]
        local = int(row) - item_start
        payload = self.arrays["enhancements"][
            int(enhancement_start):int(enhancement_end)
        ]
        if self.format_version == 1:
            enhancement = np.asarray(
                payload[local * width:(local + 1) * width], dtype=np.int64
            )
        else:
            enhancement = _unpack_symbols(
                payload, _code_bits(cfg.codebook_size), local * width, width
            )
        return np.concatenate((base, enhancement))

    def base_candidates(
        self,
        query: np.ndarray,
        nprobe: int | None = None,
        rerank_candidates: int | None = None,
        exclude_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = _normalise(np.asarray(query)[None, :], "query")[0]
        cfg = self.config
        nprobe = cfg.nprobe if nprobe is None else int(nprobe)
        rerank_candidates = (
            cfg.rerank_candidates if rerank_candidates is None else int(rerank_candidates)
        )
        centroids = np.asarray(self.arrays["centroids"], dtype=np.float32)
        codebooks = np.asarray(self.arrays["codebooks"], dtype=np.float32)
        centroid_scores = centroids @ q
        probed = _top_indices(centroid_scores, nprobe)
        lut = np.einsum("tld,d->tl", codebooks, q)
        score_parts, row_parts, cluster_parts, bucket_parts = [], [], [], []
        for cluster in probed:
            for bucket in range(len(cfg.allowed_lengths)):
                base, _, ids, row_start = self._group_codes(int(cluster), bucket)
                if not len(ids):
                    continue
                numerator = np.full(len(ids), centroid_scores[cluster], dtype=np.float32)
                for stage in range(cfg.base_stages):
                    numerator += lut[stage, base[:, stage]]
                rows = np.arange(row_start, row_start + len(ids), dtype=np.int64)
                score = numerator * np.asarray(self.arrays["base_inverse_norm"][rows])
                if exclude_id is not None:
                    score = score.copy()
                    score[ids.astype(np.int64) == int(exclude_id)] = -np.inf
                score_parts.append(score)
                row_parts.append(rows)
                cluster_parts.append(np.full(len(ids), cluster, dtype=np.int32))
                bucket_parts.append(np.full(len(ids), bucket, dtype=np.int16))
        if not score_parts:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float32), empty.astype(np.int32), empty.astype(np.int16)
        scores = np.concatenate(score_parts)
        rows = np.concatenate(row_parts)
        clusters = np.concatenate(cluster_parts)
        buckets = np.concatenate(bucket_parts)
        keep = _top_indices(scores, min(rerank_candidates, len(scores)))
        return rows[keep], scores[keep], clusters[keep], buckets[keep]

    def _norm_from_code(self, cluster: int, codes: np.ndarray) -> float:
        value = float(self._m0[cluster])
        for t, code in enumerate(codes):
            value += 2.0 * float(self._m1[cluster, t, code])
            value += float(self._m2[t, code])
            for s in range(t):
                value += 2.0 * float(self._m3[s, t, codes[s], code])
        return math.sqrt(max(value, EPS))

    def search_one(
        self,
        query: np.ndarray,
        topk: int | None = None,
        nprobe: int | None = None,
        rerank_candidates: int | None = None,
        exclude_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = _normalise(np.asarray(query)[None, :], "query")[0]
        cfg = self.config
        topk = cfg.retrieval_k if topk is None else int(topk)
        rows, _, clusters, buckets = self.base_candidates(
            q, nprobe=nprobe, rerank_candidates=rerank_candidates, exclude_id=exclude_id
        )
        if not len(rows):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        centroids = np.asarray(self.arrays["centroids"], dtype=np.float32)
        codebooks = np.asarray(self.arrays["codebooks"], dtype=np.float32)
        centroid_scores = centroids @ q
        lut = np.einsum("tld,d->tl", codebooks, q)
        final = np.empty(len(rows), dtype=np.float32)
        for j, (row, cluster, bucket) in enumerate(zip(rows, clusters, buckets)):
            codes = self._item_codes(int(row), int(cluster), int(bucket))
            numerator = float(centroid_scores[cluster])
            for stage, code in enumerate(codes):
                numerator += float(lut[stage, code])
            if cfg.norm_mode == "gram_tables":
                final[j] = numerator / self._norm_from_code(int(cluster), codes)
            else:
                final[j] = numerator * float(self.arrays["final_inverse_norm"][row])
        keep = _top_indices(final, min(topk, len(final)))
        ids = np.asarray(self.arrays["item_ids"][rows[keep]], dtype=np.int64)
        return ids, final[keep]

    def search(
        self,
        queries: np.ndarray,
        topk: int | None = None,
        nprobe: int | None = None,
        rerank_candidates: int | None = None,
        exclude_ids: np.ndarray | None = None,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        queries = _normalise(queries, "queries")
        topk = self.config.retrieval_k if topk is None else int(topk)
        if exclude_ids is not None:
            exclude_ids = np.asarray(exclude_ids, dtype=np.int64)
            if exclude_ids.shape != (len(queries),):
                raise ValueError("exclude_ids has the wrong shape")
        ids_out = np.full((len(queries), topk), -1, dtype=np.int64)
        score_out = np.full((len(queries), topk), -np.inf, dtype=np.float32)
        for i, query in enumerate(queries):
            excluded = None if exclude_ids is None else int(exclude_ids[i])
            ids, scores = self.search_one(
                query,
                topk=topk,
                nprobe=nprobe,
                rerank_candidates=rerank_candidates,
                exclude_id=excluded,
            )
            ids_out[i, :len(ids)] = ids
            score_out[i, :len(scores)] = scores
        if topk == 1:
            ids_out = ids_out[:, 0]
            score_out = score_out[:, 0]
        return (ids_out, score_out) if return_scores else ids_out

    def dense_reference_scores(self, query: np.ndarray) -> np.ndarray:
        """Explicit M-dimensional reconstruction, for correctness tests only."""
        q = _normalise(np.asarray(query)[None, :], "query")[0]
        centroids = np.asarray(self.arrays["centroids"], dtype=np.float32)
        codebooks = np.asarray(self.arrays["codebooks"], dtype=np.float32)
        scores = np.empty(self.n_items, dtype=np.float32)
        for cluster in range(self.config.num_centroids):
            for bucket, _ in enumerate(self.config.allowed_lengths):
                base, enhancement, ids, _ = self._group_codes(cluster, bucket)
                if not len(ids):
                    continue
                codes = np.concatenate((base, enhancement), axis=1)
                reconstruction = np.repeat(centroids[cluster][None, :], len(ids), axis=0)
                for stage in range(codes.shape[1]):
                    reconstruction += codebooks[stage, codes[:, stage]]
                reconstruction /= np.maximum(
                    np.linalg.norm(reconstruction, axis=1, keepdims=True), EPS
                )
                scores[ids.astype(np.int64)] = reconstruction @ q
        return scores

    def assigned_lengths_by_id(self) -> np.ndarray:
        lengths = np.empty(self.n_items, dtype=np.int16)
        for cluster in range(self.config.num_centroids):
            for bucket, length in enumerate(self.config.allowed_lengths):
                start, end = self.arrays["item_spans"][cluster, bucket]
                ids = np.asarray(self.arrays["item_ids"][int(start):int(end)], dtype=np.int64)
                lengths[ids] = length
        return lengths

    def save(self, path: str) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "format_version": self.format_version,
            "config": asdict(self.config),
            "arrays": self._layout,
            "allocation": {
                "selected_bytes": self.allocation.selected_bytes,
                "budget_bytes": self.allocation.budget_bytes,
                "objective": self.allocation.objective,
                "histogram": self.allocation.histogram,
                "strategy": self.allocation.strategy,
                "selected_bits": self.allocation.selected_bits,
                "budget_bits": self.allocation.budget_bits,
            },
            "gallery_hash": self.gallery_hash,
            "codec_hash": self.codec_hash,
            "config_hash": self.config_hash,
            "calibration_report": self.calibration_report,
            "target_budget_bytes": self.target_budget_bytes,
            "persistent_bytes": self.persistent_bytes,
            "resident_bytes": self.resident_bytes,
        }
        payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) + 16 > HEADER_BYTES:
            raise ValueError("index metadata exceeds fixed header reservation")
        with open(path, "wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(payload)
            handle.write(b"\0" * (HEADER_BYTES - 16 - len(payload)))
            for name, array in self.arrays.items():
                target = int(self._layout[name]["offset"])
                if handle.tell() < target:
                    handle.write(b"\0" * (target - handle.tell()))
                handle.write(np.ascontiguousarray(array).tobytes())
        actual = os.path.getsize(path)
        if actual != self.persistent_bytes:
            raise AssertionError(f"serialized size {actual} != predicted {self.persistent_bytes}")
        if actual > self.target_budget_bytes:
            raise AssertionError("serialized index exceeds its target byte budget")
        self.path = path
        return path

    def close(self) -> None:
        """Release memory maps so a loaded index does not lock its file."""
        for value in self.arrays.values():
            current = value
            visited = set()
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                mapping = getattr(current, "_mmap", None)
                if mapping is not None:
                    mapping.close()
                    break
                current = getattr(current, "base", None)

    @classmethod
    def load(cls, path: str) -> "RAVRRQIndex":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not a RAVR-RQ index")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        if header["format_version"] not in {1, FORMAT_VERSION}:
            raise ValueError("unsupported RAVR-RQ format version")
        config_data = dict(header["config"])
        config_data["allowed_lengths"] = tuple(config_data["allowed_lengths"])
        # Format-v1 indexes predate the named objective and stored the five
        # legacy mixture knobs directly in the public config.  They are not
        # needed for serving, but accepting them keeps published artifacts
        # readable after the default objective was simplified.
        for name in LEGACY_OBJECTIVE_PARAMETERS:
            config_data.pop(name, None)
        for name in LEGACY_UNUSED_CONFIG_FIELDS:
            config_data.pop(name, None)
        if "objective" not in config_data:
            config_data["objective"] = OBJECTIVE_LEGACY
        config = RAVRRQConfig(**config_data)
        arrays = {}
        # JSON object keys are sorted for deterministic headers, while the binary
        # arrays are laid out in offset order.  Restore that order before the
        # constructor independently verifies the computed layout.
        array_entries = sorted(
            header["arrays"].items(), key=lambda item: int(item[1]["offset"])
        )
        for name, meta in array_entries:
            arrays[name] = np.memmap(
                path,
                dtype=np.dtype(meta["dtype"]),
                mode="r",
                offset=int(meta["offset"]),
                shape=tuple(meta["shape"]),
            )
        allocation_meta = header["allocation"]
        # Serving does not need one option per item; bucket membership is the source
        # of truth.  Keep an empty vector in the report object after loading.
        allocation = AllocationResult(
            option_ids=np.empty(0, dtype=np.int64),
            selected_bytes=int(allocation_meta["selected_bytes"]),
            objective=float(allocation_meta["objective"]),
            budget_bytes=int(allocation_meta["budget_bytes"]),
            histogram={str(k): int(v) for k, v in allocation_meta["histogram"].items()},
            strategy=allocation_meta["strategy"],
            selected_bits=allocation_meta.get("selected_bits"),
            budget_bits=allocation_meta.get("budget_bits"),
        )
        index = cls(
            config=config,
            arrays=arrays,
            allocation=allocation,
            gallery_hash=header["gallery_hash"],
            codec_hash=header["codec_hash"],
            config_hash=header["config_hash"],
            calibration_report=header["calibration_report"],
            target_budget_bytes=int(header["target_budget_bytes"]),
            path=path,
            format_version=int(header["format_version"]),
        )
        if os.path.getsize(path) != index.persistent_bytes:
            raise ValueError("index file size does not match its array layout")
        return index


def exact_teacher_topk(
    gallery: np.ndarray,
    queries: np.ndarray,
    topk: int,
    exclude_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gallery = _normalise(gallery, "gallery")
    queries = _normalise(queries, "queries")
    ids = np.empty((len(queries), topk), dtype=np.int64)
    scores = np.empty((len(queries), topk), dtype=np.float32)
    for i, query in enumerate(queries):
        row = gallery @ query
        if exclude_ids is not None and int(exclude_ids[i]) >= 0:
            row[int(exclude_ids[i])] = -np.inf
        keep = _top_indices(row, topk)
        ids[i], scores[i] = keep, row[keep]
    return ids, scores


def evaluate_index(
    index: RAVRRQIndex,
    gallery: np.ndarray,
    queries: np.ndarray,
    ks: Sequence[int] = (1, 10),
    gallery_labels: np.ndarray | None = None,
    query_labels: np.ndarray | None = None,
    exclude_ids: np.ndarray | None = None,
) -> dict:
    """Evaluate ranking fidelity, candidate recall, bytes, and Python latency."""
    gallery = _normalise(gallery, "gallery")
    queries = _normalise(queries, "queries")
    max_k = min(max(int(k) for k in ks), len(gallery) - (exclude_ids is not None))
    teacher_ids, teacher_scores = exact_teacher_topk(
        gallery, queries, max_k, exclude_ids=exclude_ids
    )
    # Warm the lookup path without including it in the reported latency.
    for i in range(min(3, len(queries))):
        excluded = None if exclude_ids is None else int(exclude_ids[i])
        index.search_one(queries[i], topk=max_k, exclude_id=excluded)
    returned = np.empty((len(queries), max_k), dtype=np.int64)
    elapsed_ms = np.empty(len(queries), dtype=np.float64)
    candidate_recall = np.empty(len(queries), dtype=np.float64)
    for i, query in enumerate(queries):
        excluded = None if exclude_ids is None else int(exclude_ids[i])
        started = time.perf_counter()
        ids, _ = index.search_one(query, topk=max_k, exclude_id=excluded)
        elapsed_ms[i] = (time.perf_counter() - started) * 1000.0
        returned[i] = ids
        rows, _, _, _ = index.base_candidates(query, exclude_id=excluded)
        candidate_ids = np.asarray(index.arrays["item_ids"][rows], dtype=np.int64)
        candidate_recall[i] = len(set(candidate_ids) & set(teacher_ids[i])) / max_k

    result: dict[str, object] = {
        "method": index.allocation.strategy,
        "persistent_bytes": index.persistent_bytes,
        "resident_bytes": index.resident_bytes,
        "persistent_bytes_per_item": index.persistent_bytes / index.n_items,
        "resident_bytes_per_item": index.resident_bytes / index.n_items,
        "candidate_recall": float(candidate_recall.mean()),
        "latency_ms_p50": float(np.percentile(elapsed_ms, 50)),
        "latency_ms_p95": float(np.percentile(elapsed_ms, 95)),
        "qps": float(1000.0 / elapsed_ms.mean()),
        "allocation_histogram": index.allocation.histogram,
    }
    for k in ks:
        k = min(int(k), max_k)
        overlap = [
            len(set(returned[i, :k]) & set(teacher_ids[i, :k])) / k
            for i in range(len(queries))
        ]
        result[f"teacher_recall_at_{k}"] = float(np.mean(overlap))
        result[f"boundary_inversion_rate_at_{k}"] = float(1.0 - np.mean(overlap))

        ndcg = []
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        for i in range(len(queries)):
            relevance = gallery[returned[i, :k]] @ queries[i]
            ideal = teacher_scores[i, :k]
            dcg = np.sum((np.exp2(relevance) - 1.0) * discount)
            idcg = np.sum((np.exp2(ideal) - 1.0) * discount)
            ndcg.append(float(dcg / max(idcg, EPS)))
        result[f"teacher_ndcg_at_{k}"] = float(np.mean(ndcg))

    if gallery_labels is not None and query_labels is not None:
        gallery_labels = np.asarray(gallery_labels)
        query_labels = np.asarray(query_labels)
        result["semantic_recall_at_1"] = float(
            np.mean(gallery_labels[returned[:, 0]] == query_labels)
        )
    return result


def evaluate_stage_a_exhaustive(
    trainer: RAVRRQTrainer,
    index: RAVRRQIndex,
    queries: np.ndarray,
    ks: Sequence[int] = (1, 10),
    gallery_labels: np.ndarray | None = None,
    query_labels: np.ndarray | None = None,
    exclude_ids: np.ndarray | None = None,
) -> dict:
    """Exhaustively score every gallery item for the Stage-A allocation test.

    This removes IVF routing and base-candidate loss from the comparison.  Prefix
    scores still use the stored FP16 codec values and exact reconstruction norms;
    only the packed two-stage access path is bypassed.
    """
    queries = _normalise(queries, "queries")
    if queries.shape[1] != trainer.dimension:
        raise ValueError("query and trainer dimensions differ")
    if index.codec_hash != trainer.codec_hash:
        raise ValueError("index and trainer do not share a codec")
    if exclude_ids is not None:
        exclude_ids = np.asarray(exclude_ids, dtype=np.int64)
        if exclude_ids.shape != (len(queries),):
            raise ValueError("exclude_ids has the wrong shape")
    max_k = max(int(k) for k in ks)
    lengths = index.assigned_lengths_by_id()
    option_lookup = {length: option for option, length in enumerate(trainer.config.allowed_lengths)}
    options = np.array([option_lookup[int(length)] for length in lengths], dtype=np.int16)
    rows = np.arange(trainer.n_items)
    overlap = {int(k): [] for k in ks}
    ndcg = {int(k): [] for k in ks}
    semantic = []
    elapsed = []
    for qi, query in enumerate(queries):
        started = time.perf_counter()
        teacher_score = trainer.x @ query
        approximate_options = trainer._prefix_scores(query, rows)
        approximate = approximate_options[rows, options]
        if exclude_ids is not None and exclude_ids[qi] >= 0:
            excluded = int(exclude_ids[qi])
            teacher_score[excluded] = -np.inf
            approximate[excluded] = -np.inf
        teacher = _top_indices(teacher_score, max_k)
        returned = _top_indices(approximate, max_k)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        for k in ks:
            k = int(k)
            overlap[k].append(len(set(returned[:k]) & set(teacher[:k])) / k)
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            relevance = teacher_score[returned[:k]]
            ideal = teacher_score[teacher[:k]]
            dcg = np.sum((np.exp2(relevance) - 1.0) * discount)
            idcg = np.sum((np.exp2(ideal) - 1.0) * discount)
            ndcg[k].append(float(dcg / max(idcg, EPS)))
        if gallery_labels is not None and query_labels is not None:
            semantic.append(bool(gallery_labels[returned[0]] == query_labels[qi]))
    result: dict[str, object] = {
        "method": index.allocation.strategy,
        "evaluation_mode": "stage_a_exhaustive",
        "queries": len(queries),
        "gallery_items": trainer.n_items,
        "persistent_bytes": index.persistent_bytes,
        "resident_bytes": index.resident_bytes,
        "persistent_bytes_per_item": index.persistent_bytes / trainer.n_items,
        "resident_bytes_per_item": index.resident_bytes / trainer.n_items,
        "score_time_ms_p50": float(np.percentile(elapsed, 50)),
        "score_time_ms_p95": float(np.percentile(elapsed, 95)),
        "allocation_histogram": index.allocation.histogram,
    }
    for k in ks:
        k = int(k)
        result[f"teacher_recall_at_{k}"] = float(np.mean(overlap[k]))
        result[f"teacher_ndcg_at_{k}"] = float(np.mean(ndcg[k]))
    if semantic:
        result["semantic_recall_at_1"] = float(np.mean(semantic))
    return result


__all__ = [
    "AllocationResult",
    "RAVRRQConfig",
    "RAVRRQIndex",
    "RAVRRQTrainer",
    "OBJECTIVE_BOUNDARY_KL",
    "OBJECTIVE_ORDINAL_KL",
    "OBJECTIVE_LEGACY",
    "LEGACY_OBJECTIVE_PARAMETERS",
    "evaluate_index",
    "evaluate_stage_a_exhaustive",
    "exact_teacher_topk",
    "solve_exact_small",
    "solve_lagrangian",
]
