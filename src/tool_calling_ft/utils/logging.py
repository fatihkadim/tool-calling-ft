"""VRAM, eğitim süresi, trainable parameter sayısı gibi yan metrikleri

otomatik loglayıp reports/<method>_metrics.json olarak kaydeden yardımcılar.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import torch

logger = logging.getLogger(__name__)


def count_trainable_params(model: torch.nn.Module) -> dict[str, Any]:
    """Modeldeki toplam, eğitilebilir (trainable) parametre sayısını ve oranını hesaplar."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / all_params * 100.0) if all_params > 0 else 0.0

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": round(trainable_pct, 4),
    }


def get_gpu_memory_mb() -> dict[str, float]:
    """Mevcut ve maksimum VRAM kullanımını MB cinsinden döndürür."""
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }

    return {
        "cuda_available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
    }


@contextmanager
def track_vram_and_time(description: str = "Operation") -> Generator[dict[str, Any], None, None]:
    """Bir işlemin çalışma süresini ve pik VRAM kullanımını ölçen context manager."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    stats: dict[str, Any] = {
        "description": description,
        "elapsed_seconds": 0.0,
        "peak_vram_mb": 0.0,
    }

    start_time = time.perf_counter()
    try:
        yield stats
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        stats["elapsed_seconds"] = round(elapsed, 4)

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            stats["peak_vram_mb"] = round(peak_vram, 2)
            logger.info(
                "[%s] Tamamlandı: %.2f sn, Pik VRAM: %.2f MB",
                description,
                elapsed,
                peak_vram,
            )
        else:
            logger.info("[%s] Tamamlandı: %.2f sn (CPU)", description, elapsed)


def save_metrics_report(
    metrics: dict[str, Any],
    report_filename: str,
    reports_dir: Path | str = "reports",
) -> Path:
    """Metrik sonuçlarını reports/ dizini altına JSON olarak kaydeder."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not report_filename.endswith(".json"):
        report_filename += ".json"

    file_path = out_dir / report_filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Metrik raporu kaydedildi: %s", file_path)
    return file_path
