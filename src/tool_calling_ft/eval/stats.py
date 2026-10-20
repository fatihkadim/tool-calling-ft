"""Eval metrikleri için güven aralıkları (bootstrap + Wilson).

metrics.py tek bir nokta tahmini (ör. %46.25) üretir. Eval seti küçük olduğunda
(80 pozitif / 20 negatif) bu değerin belirsizliği büyüktür; bu modül o belirsizliği
sayısallaştırır.

İki giriş yolu vardır:
- Örnek bazlı değerler (0/1 doğru-yanlış veya 0.0-1.0 argüman skoru): yüzde-bootstrap.
- Yalnızca sayımlar (k doğru / n örnek): ikili metrikler için örnek bazlı vektör
  sayımlardan birebir yeniden kurulabilir, bu yüzden aynı bootstrap uygulanır.
  Argument accuracy gibi kesirli metrikler sayımlardan yeniden kurulamaz.

İki yöntemi karşılaştırırken aynı örnekler üzerinde eşleştirilmiş (paired) bootstrap
en doğrusudur; bu, örnek bazlı tahminlerin kaydedilmesini gerektirir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ConfidenceInterval:
    """Nokta tahmini ve [low, high] güven aralığı (0.0-1.0 ölçeğinde)."""

    estimate: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    def format_pct(self, digits: int = 1) -> str:
        """Örn: '%46.2 [35.0, 57.5]'."""
        return (
            f"%{self.estimate * 100:.{digits}f} "
            f"[{self.low * 100:.{digits}f}, {self.high * 100:.{digits}f}]"
        )

    def overlaps(self, other: ConfidenceInterval) -> bool:
        return self.low <= other.high and other.low <= self.high


def counts_to_values(successes: int, n: int) -> np.ndarray:
    """k/n sayımını örnek bazlı 0/1 vektörüne çevirir (ikili metrikler için kayıpsız)."""
    if n <= 0:
        raise ValueError("n pozitif olmalı")
    if not 0 <= successes <= n:
        raise ValueError(f"successes 0 ile n arasında olmalı: {successes}/{n}")
    return np.concatenate([np.ones(successes), np.zeros(n - successes)])


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """Örnek bazlı değerlerin ortalaması için yüzde-bootstrap güven aralığı."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values boş olamaz")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)

    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return ConfidenceInterval(float(arr.mean()), float(low), float(high), int(arr.size), confidence)


def bootstrap_ci_from_counts(
    successes: int,
    n: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """İkili bir metrik için (k doğru / n örnek) bootstrap güven aralığı."""
    return bootstrap_ci(counts_to_values(successes, n), n_resamples, confidence, seed)


def wilson_ci(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> ConfidenceInterval:
    """Oran için Wilson skor aralığı.

    Bootstrap, k=0 veya k=n olduğunda sıfır genişlikte aralık verir (ör. 20/20 → [%100, %100]);
    Wilson bu uç durumlarda da anlamlı bir aralık üretir.
    """
    if n <= 0:
        raise ValueError("n pozitif olmalı")
    z = _z_score(confidence)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return ConfidenceInterval(p, max(0.0, center - half), min(1.0, center + half), n, confidence)


def paired_bootstrap_diff(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """Aynı örnekler üzerinde (A - B) ortalama farkı için eşleştirilmiş bootstrap.

    Aralık 0'ı içermiyorsa fark bu güven düzeyinde anlamlıdır.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Eşleştirilmiş değerler aynı uzunlukta olmalı: {a.shape} vs {b.shape}")
    return bootstrap_ci(a - b, n_resamples, confidence, seed)


def unpaired_bootstrap_diff_from_counts(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """Yalnızca sayımlar varken iki oranın farkı (A - B) için bağımsız bootstrap.

    Örnekler gerçekte eşleştirilmiş olduğundan bu aralık, paired bootstrap'e göre
    genellikle daha geniştir (temkinli tarafta kalır).
    """
    a = counts_to_values(successes_a, n_a)
    b = counts_to_values(successes_b, n_b)
    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, n_a, size=(n_resamples, n_a))].mean(axis=1)
    means_b = b[rng.integers(0, n_b, size=(n_resamples, n_b))].mean(axis=1)
    diffs = means_a - means_b

    alpha = (1 - confidence) / 2
    low, high = np.quantile(diffs, [alpha, 1 - alpha])
    return ConfidenceInterval(
        float(a.mean() - b.mean()), float(low), float(high), min(n_a, n_b), confidence
    )


def _z_score(confidence: float) -> float:
    """Yaygın güven düzeyleri için iki taraflı z değeri (scipy bağımlılığı olmadan)."""
    table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    for level, z in table.items():
        if abs(confidence - level) < 1e-9:
            return z
    raise ValueError(f"Desteklenen güven düzeyleri: {sorted(table)}")
