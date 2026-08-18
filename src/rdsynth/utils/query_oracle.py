from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class QueryOracleStats:
    query_count: int
    query_calls: int
    query_time_sec: float
    query_budget: int | None
    query_over_budget_count: int
    budget_exhausted: bool
    hard_label: bool


class QueryOracle:
    def __init__(
        self,
        score_fn: ScoreFn,
        *,
        max_queries: int | None = None,
        hard_label: bool = False,
        hard_label_threshold: float = 0.5,
        exhausted_fill: float = 1.0,
    ) -> None:
        self._score_fn = score_fn
        self.max_queries = None if max_queries is None else max(0, int(max_queries))
        self.hard_label = bool(hard_label)
        self.hard_label_threshold = float(hard_label_threshold)
        self.exhausted_fill = float(exhausted_fill)

        self.query_count = 0
        self.query_calls = 0
        self.query_time_sec = 0.0
        self.query_over_budget_count = 0
        self.budget_exhausted = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("QueryOracle expects a 2D array [n_samples, n_features].")
        n = int(arr.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float64)

        allowed = n
        if self.max_queries is not None:
            remaining = max(0, self.max_queries - self.query_count)
            allowed = min(n, remaining)

        out = np.full((n,), self.exhausted_fill, dtype=np.float64)
        if allowed > 0:
            start = time.perf_counter()
            raw = np.asarray(self._score_fn(arr[:allowed]), dtype=np.float64).reshape(-1)
            self.query_time_sec += time.perf_counter() - start
            self.query_calls += 1
            self.query_count += allowed
            if raw.shape[0] != allowed:
                raise ValueError("score_fn must return one score per queried sample.")
            if self.hard_label:
                raw = (raw >= self.hard_label_threshold).astype(np.float64)
            out[:allowed] = raw

        if allowed < n:
            self.query_over_budget_count += n - allowed
        if self.max_queries is not None and self.query_count >= self.max_queries:
            self.budget_exhausted = True
        return out

    def stats(self) -> QueryOracleStats:
        return QueryOracleStats(
            query_count=int(self.query_count),
            query_calls=int(self.query_calls),
            query_time_sec=float(self.query_time_sec),
            query_budget=self.max_queries,
            query_over_budget_count=int(self.query_over_budget_count),
            budget_exhausted=bool(self.budget_exhausted),
            hard_label=bool(self.hard_label),
        )
