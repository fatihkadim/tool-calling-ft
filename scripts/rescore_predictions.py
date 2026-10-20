"""Kaydedilmiş tahminleri GPU olmadan yeniden puanlar.

reports/predictions/<method>_predictions.jsonl dosyalarını (eval/harness.py üretir)
okur ve her yöntem için:
  - normal (gevşek) ve sıkı parser ile tüm kalite metriklerini,
  - örnek bazlı bootstrap ile %95 güven aralıklarını (argument accuracy dahil),
  - aynı örnekler üzerinde yöntemler arası eşleştirilmiş (paired) farkları
hesaplar. Metrik kodu değiştiğinde modelleri yeniden çalıştırmak gerekmez.

Kullanım:
    uv run python scripts/rescore_predictions.py
    uv run python scripts/rescore_predictions.py --predictions-dir reports/predictions
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from tool_calling_ft.eval.metrics import (
    ToolCallExample,
    aggregate_metrics,
    score_example,
)
from tool_calling_ft.eval.stats import bootstrap_ci, paired_bootstrap_diff

METHOD_ORDER = ["baseline", "lora", "dora", "qlora", "full_ft"]

# (örnek bazlı skor anahtarı, etiket, hangi alt küme)
CI_METRICS = [
    ("tool_selection_correct", "Tool Selection (Pos)", "positive"),
    ("argument_accuracy", "Arg. Accuracy (Pos)", "positive"),
    ("json_valid", "JSON Validity (Pos)", "positive"),
    ("tool_selection_correct", "Neg. Rejection", "negative"),
]


def load_predictions(predictions_dir: Path) -> dict[str, list[dict]]:
    found = {}
    for path in sorted(predictions_dir.glob("*_predictions.jsonl")):
        method = path.stem.removesuffix("_predictions")
        with open(path, encoding="utf-8") as f:
            found[method] = [json.loads(line) for line in f if line.strip()]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return dict(sorted(found.items(), key=lambda kv: order.get(kv[0], len(order))))


def rescore(rows: list[dict], strict: bool) -> tuple[dict, list[dict]]:
    examples = [
        ToolCallExample(
            predicted_raw=r["predicted_raw"],
            expected_tool=r["expected_tool"],
            expected_arguments=r["expected_arguments"],
            known_tools=set(r.get("known_tools", [])),
        )
        for r in rows
    ]
    known = set().union(*(ex.known_tools for ex in examples)) if examples else set()
    per_sample = [score_example(ex, known, strict=strict) for ex in examples]
    return aggregate_metrics(examples, known, strict=strict), per_sample


def subset_values(rows: list[dict], scores: list[dict], key: str, subset: str) -> dict[str, float]:
    """id → değer sözlüğü (paired karşılaştırmada örnekleri eşlemek için)."""
    want_positive = subset == "positive"
    return {
        r["id"]: float(s[key])
        for r, s in zip(rows, scores)
        if s["is_positive"] == want_positive
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaydedilmiş tahminleri yeniden puanla")
    parser.add_argument("--predictions-dir", type=str, default="reports/predictions")
    parser.add_argument("--output", type=str, default="reports/rescored_metrics.json")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions_dir))
    if not predictions:
        print(f"[ERROR] {args.predictions_dir} altında *_predictions.jsonl bulunamadı.")
        print("        Önce modelleri eval/harness.py ile çalıştırın (tahminler otomatik kaydedilir).")
        raise SystemExit(1)

    output: dict = {"methods": {}, "paired_comparisons": {}}
    values: dict[tuple[str, str, str], dict[str, float]] = {}

    for method, rows in predictions.items():
        stopped = sum(bool(r.get("stopped")) for r in rows)
        output["methods"][method] = {"n": len(rows), "stopped": stopped}
        print(f"\n{method}  (n={len(rows)}, durdu: {stopped}/{len(rows)})")

        for mode in ("lenient", "strict"):
            metrics, per_sample = rescore(rows, strict=(mode == "strict"))
            cis = {}
            for key, label, subset in CI_METRICS:
                vals = subset_values(rows, per_sample, key, subset)
                values[(method, mode, label)] = vals
                if not vals:
                    continue
                ci = bootstrap_ci(list(vals.values()), n_resamples=args.n_resamples)
                cis[label] = {"estimate": ci.estimate, "bootstrap_95": [ci.low, ci.high], "n": ci.n}
                print(f"  [{mode:7s}] {label:22s} {ci.format_pct()}  (n={ci.n})")
            output["methods"][method][mode] = {"metrics": metrics, "confidence_intervals": cis}

    # Eşleştirilmiş farklar: yalnızca iki yöntemin ortak örnek id'leri üzerinde
    print("\nEşleştirilmiş farklar (B - A, %95 paired bootstrap):")
    for mode in ("lenient", "strict"):
        for _, label, _ in CI_METRICS:
            for a, b in combinations(predictions, 2):
                va, vb = values.get((a, mode, label), {}), values.get((b, mode, label), {})
                common = sorted(va.keys() & vb.keys())
                if not common:
                    continue
                diff = paired_bootstrap_diff(
                    [vb[i] for i in common], [va[i] for i in common], n_resamples=args.n_resamples
                )
                significant = not (diff.low <= 0 <= diff.high)
                output["paired_comparisons"][f"{mode} | {label} | {b} - {a}"] = {
                    "diff": diff.estimate,
                    "bootstrap_95": [diff.low, diff.high],
                    "n_common": len(common),
                    "significant": significant,
                }
                if label == "Tool Selection (Pos)":
                    mark = "ANLAMLI" if significant else "anlamlı değil"
                    print(f"  [{mode:7s}] {b + ' - ' + a:18s} {diff.estimate * 100:+6.1f} pp "
                          f"[{diff.low * 100:+6.1f}, {diff.high * 100:+6.1f}]  {mark}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
