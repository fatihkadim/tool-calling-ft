"""Fine-tuning öncesi (zero-shot) baseline eval'i çalıştırır.

Kullanım:
    uv run python scripts/run_baseline.py
    uv run python scripts/run_baseline.py --max-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tool_calling_ft.eval.harness import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Baseline Eval")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model adı veya yerel yolu",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/eval_subset.jsonl",
        help="Eval veri seti yolu (varsayılan: eval_subset.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Değerlendirilecek maksimum örnek sayısı (hızlı test için)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch boyutu",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" BASELINE (ZERO-SHOT) TOOL CALLING DEĞERLENDİRMESİ BAŞLIYOR")
    print(f" Model: {args.model}")
    print(f" Veri Seti: {args.dataset}")
    print("=" * 60 + "\n")

    results = run_eval(
        model_name_or_path=args.model,
        adapter_path=None,
        dataset_path=args.dataset,
        method_name="baseline",
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print(" BASELINE SONUÇ ÖZETİ")
    print("=" * 60)
    print(" Kalite Metrikleri:")
    for k, v in results["quality_metrics"].items():
        print(f"   • {k:28s}: {v}")
    print("\n Performans & Kaynak Metrikleri:")
    for k, v in results["performance_metrics"].items():
        print(f"   • {k:28s}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
