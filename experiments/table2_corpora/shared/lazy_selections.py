from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator

import numpy as np

import benchmark_nested_additive_ravr as panel
from benchmark_nested_additive_ravr import ALLOWED_LENGTHS
from ravr_rq import solve_lagrangian

# The keys `_selections` returns, in its own order.
STRATEGY_KEYS = (
    "uniform", "ravr", "ravr_unseen_rd", "ordinal_per_exposure",
    "candidate_frequency", "query_ip_mse", "random_histogram", "reconstruction",
)


class LazySelections(Mapping):

    def __init__(self, state: dict, n_items: int, seed: int, target_length: int,
                 nbits: int) -> None:
        if target_length not in ALLOWED_LENGTHS:
            raise ValueError("target_length must occur in ALLOWED_LENGTHS")
        self._state = state
        self._n_items = int(n_items)
        self._seed = int(seed)
        self._target_length = int(target_length)
        base_bytes = ALLOWED_LENGTHS[0] * nbits // 8
        self._option_costs = np.array([
            length * nbits // 8 - base_bytes for length in ALLOWED_LENGTHS
        ], dtype=np.int64)
        self._budget = self._n_items * (
            self._target_length * nbits // 8 - base_bytes
        )
        self._coverage = np.asarray(state["coverage"], dtype=np.float64)
        self._cache: dict[str, np.ndarray] = {}
        self.solved: list[str] = []

    # -- Mapping protocol --------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        if key not in STRATEGY_KEYS:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = self._compute(key)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(STRATEGY_KEYS)

    def __len__(self) -> int:
        return len(STRATEGY_KEYS)

    # -- the allocations, verbatim from _selections ------------------------
    def _solve(self, table: np.ndarray, budget: int | None = None) -> np.ndarray:
        return solve_lagrangian(
            table, self._option_costs,
            self._budget if budget is None else budget,
        )

    def _compute(self, key: str) -> np.ndarray:
        self.solved.append(key)
        state, coverage = self._state, self._coverage

        if key == "uniform":
            return np.full(
                self._n_items, ALLOWED_LENGTHS.index(self._target_length),
                dtype=np.int64,
            )
        if key == "ravr":
            return self._solve(state["distortion"])
        if key == "reconstruction":
            return self._solve(state["prior_distortion"])
        if key == "query_ip_mse":
            return self._solve(state["query_ip_distortion"])
        if key == "ordinal_per_exposure":
            return self._solve(
                np.asarray(state["distortion"], dtype=np.float64)
                / np.maximum(coverage[:, None], 1.0)
            )
        if key == "candidate_frequency":
            remaining_symbols = (
                ALLOWED_LENGTHS[-1] - np.asarray(ALLOWED_LENGTHS, dtype=np.int64)
            )
            return self._solve(coverage[:, None] * remaining_symbols[None, :])
        if key == "random_histogram":
            # A fresh generator seeded exactly as the original, used once.
            rng = np.random.default_rng(20260825 + self._seed)
            return rng.permutation(self["ravr"])
        if key == "ravr_unseen_rd":
            ravr = self["ravr"]
            out = ravr.copy()
            unseen = np.flatnonzero(coverage == 0)
            if len(unseen):
                seen = np.flatnonzero(coverage > 0)
                seen_cost = int(self._option_costs[ravr[seen]].sum())
                unseen_budget = self._budget - seen_cost
                if unseen_budget < 0:
                    raise AssertionError("covered RAVR allocation exceeded the cap")
                out[unseen] = self._solve(
                    state["prior_distortion"][unseen], unseen_budget
                )
            return out
        raise KeyError(key)


def verify_against_reference(seed: int = 20260902, n_items: int = 4096,
                             target_length: int = 12, nbits: int = 4) -> None:
    rng = np.random.default_rng(seed)
    options = len(ALLOWED_LENGTHS)
    state = {
        # Decreasing in prefix length, like a real risk curve.
        "distortion": np.sort(rng.random((n_items, options)), axis=1)[:, ::-1],
        "prior_distortion": np.sort(rng.random((n_items, options)), axis=1)[:, ::-1],
        "query_ip_distortion": np.sort(rng.random((n_items, options)), axis=1)[:, ::-1],
        # Integer, heavily tied, with a realistic share of zeros: the shape
        # that makes candidate_frequency pathological at scale.
        "coverage": rng.integers(0, 12, n_items).astype(np.int64)
                    * (rng.random(n_items) > 0.55),
    }
    reference = panel._selections(state, n_items, seed, target_length, nbits)
    lazy = LazySelections(state, n_items, seed, target_length, nbits)
    for key in STRATEGY_KEYS:
        got, want = np.asarray(lazy[key]), np.asarray(reference[key])
        if got.shape != want.shape or not np.array_equal(got, want):
            raise AssertionError(
                f"lazy selection for {key!r} differs from _selections; "
                "refusing to install the override"
            )


def install(verify: bool = True) -> None:
    """Rebind `_selections` for this process. No file is modified."""
    if verify:
        verify_against_reference()
    if getattr(panel._selections, "_is_lazy_override", False):
        return
    original = panel._selections

    def _lazy(state, n_items, seed, target_length, nbits):
        return LazySelections(state, n_items, seed, target_length, nbits)

    _lazy._is_lazy_override = True
    _lazy._original = original
    panel._selections = _lazy
    print("panel  lazy _selections installed (verified identical to the "
          "reference on every strategy); allocations are solved only for the "
          "strategies actually evaluated", flush=True)




def install_aggregate_subset(strategies: tuple[str, ...]) -> None:

    import aggregate_nested_additive_ravr as agg

    requested = tuple(strategies)
    missing = set(requested) - set(agg.STRATEGIES)
    if missing:
        raise ValueError(f"unknown strategies: {sorted(missing)}")
    if "ravr" not in requested or "uniform" not in requested:
        raise ValueError("ravr and uniform are required to aggregate anything")
    comparators = tuple(c for c in agg.COMPARATORS if c in requested)
    dropped = tuple(c for c in agg.COMPARATORS if c not in requested)
    agg.STRATEGIES = requested
    agg.COMPARATORS = comparators
    if dropped:
        print(f"aggregate  comparators dropped (not evaluated): "
              f"{', '.join(dropped)}; the family-wise band now covers "
              f"{len(comparators)} comparators instead of "
              f"{len(comparators) + len(dropped)}", flush=True)
