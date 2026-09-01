"""Token uzunluk dağılımı ölçüm scripti.

Qwen/Qwen2.5-0.5B tokenizer ile tüm processed veri setlerini tarayıp:
- Min, max, p50, p90, p95, p99 token dağılımını çıkarır
- 1024, 1536, 2048, 2560 sınırlarında kaç örneğin sığdığını ölçer
- Pozitif örneklerde <tool_call> hedefinin hangi token pozisyonunda başladığını bulur
  ve kaç örnekte max_seq_len ile kesildiğini raporlar

Kullanım: uv run python scripts/measure_token_lengths.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_FILES = {
    "train": Path("data/processed/train.jsonl"),
    "val": Path("data/processed/val.jsonl"),
    "eval_subset": Path("data/processed/eval_subset.jsonl"),
}
THRESHOLDS = [512, 1024, 1536, 2048, 2560, 3072]


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def measure_lengths(tokenizer, items: list[dict]) -> dict:
    """Token uzunluk istatistiklerini hesaplar."""
    lengths = []
    tool_call_positions = []  # Pozitif örneklerde <tool_call> token pozisyonu

    for item in items:
        text = item.get("text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(token_ids))

        # Pozitif örneklerde <tool_call> pozisyonunu bul
        if item.get("is_tool_call", False):
            tool_call_marker = "<tool_call>"
            marker_pos = text.find(tool_call_marker)
            if marker_pos >= 0:
                prefix_tokens = tokenizer.encode(text[:marker_pos], add_special_tokens=False)
                tool_call_positions.append(len(prefix_tokens))

    lengths_arr = np.array(lengths)

    stats = {
        "count": len(lengths),
        "min": int(lengths_arr.min()),
        "max": int(lengths_arr.max()),
        "mean": round(float(lengths_arr.mean()), 1),
        "p50": int(np.percentile(lengths_arr, 50)),
        "p90": int(np.percentile(lengths_arr, 90)),
        "p95": int(np.percentile(lengths_arr, 95)),
        "p99": int(np.percentile(lengths_arr, 99)),
    }

    # Threshold analizleri
    threshold_analysis = {}
    for t in THRESHOLDS:
        fits = int((lengths_arr <= t).sum())
        truncated = int((lengths_arr > t).sum())
        threshold_analysis[f"<={t}"] = {
            "fits": fits,
            "fits_pct": round(fits / len(lengths) * 100, 1),
            "truncated": truncated,
            "truncated_pct": round(truncated / len(lengths) * 100, 1),
        }

    # tool_call truncation analizi (yalnızca pozitif örnekler)
    tool_call_truncation = {}
    if tool_call_positions:
        tc_arr = np.array(tool_call_positions)
        for t in THRESHOLDS:
            lost = int((tc_arr >= t).sum())
            tool_call_truncation[f"max_seq_len={t}"] = {
                "tool_calls_lost": lost,
                "tool_calls_lost_pct": round(lost / len(tc_arr) * 100, 1),
                "total_positive": len(tc_arr),
            }

    return {
        "length_stats": stats,
        "threshold_analysis": threshold_analysis,
        "tool_call_truncation": tool_call_truncation,
    }


def main():
    print(f"Tokenizer yükleniyor: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    all_results = {}
    for name, path in DATA_FILES.items():
        if not path.exists():
            print(f"  {path} bulunamadi, atliyorum.")
            continue

        items = load_jsonl(path)
        result = measure_lengths(tokenizer, items)
        all_results[name] = result

        print(f"{'='*60}")
        print(f" {name.upper()} ({len(items)} ornek)")
        print(f"{'='*60}")

        stats = result["length_stats"]
        print(f"  Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']}")
        print(f"  P50: {stats['p50']}, P90: {stats['p90']}, P95: {stats['p95']}, P99: {stats['p99']}")

        print("\\n  Threshold Analizi:")
        for k, v in result["threshold_analysis"].items():
            print(f"    {k}: {v['fits']} sigar ({v['fits_pct']}%), {v['truncated']} kesilir ({v['truncated_pct']}%)")

        if result["tool_call_truncation"]:
            print("\\n  <tool_call> Truncation (pozitif ornekler):")
            for k, v in result["tool_call_truncation"].items():
                print(
                    f"    {k}: {v['tool_calls_lost']}/{v['total_positive']} tool_call kaybedilir "
                    f"({v['tool_calls_lost_pct']}%)"
                )
        print()

    # Sonuclari JSON olarak da kaydet
    out_file = Path("reports/token_length_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Sonuclar kaydedildi: {out_file}")


if __name__ == "__main__":
    main()
