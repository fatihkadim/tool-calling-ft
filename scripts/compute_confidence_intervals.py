"""reports/*_metrics.json dosyalarından %95 güven aralıklarını hesaplar.

Şu an metrik dosyaları yalnızca oranları içerdiği için ikili metrikler
(tool selection, JSON validity, negative rejection) sayımlardan yeniden kurulup
bootstrap'lenir. Argument accuracy kesirli bir metrik olduğundan, örnek bazlı
tahminler kaydedilene kadar hesaplanamaz.

Yöntem karşılaştırmaları bağımsız (unpaired) bootstrap ile yapılır; bu, eşleştirilmiş
karşılaştırmaya göre temkinli (daha geniş) bir aralık verir.

Kullanım:
    uv run python scripts/compute_confidence_intervals.py
    uv run python scripts/compute_confidence_intervals.py --markdown
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from tool_calling_ft.eval.stats import (
    bootstrap_ci_from_counts,
    unpaired_bootstrap_diff_from_counts,
    wilson_ci,
)

METHOD_ORDER = ["baseline", "lora", "dora", "qlora", "full_ft"]

# (metrik anahtarı, etiket, payda anahtarı)
BINARY_METRICS = [
    ("positive_tool_selection_accuracy", "Tool Selection (Pos)", "tool_required_count"),
    ("positive_json_validity", "JSON Validity (Pos)", "tool_required_count"),
    ("negative_rejection_accuracy", "Neg. Rejection", "no_tool_required_count"),
]


def load_reports(reports_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(reports_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        reports[data.get("method", path.stem.replace("_metrics", ""))] = data
    return {m: reports[m] for m in METHOD_ORDER if m in reports}


def rate_to_count(rate: float, n: int) -> int:
    """Oranı tam sayı sayıma çevirir; oran bir k/n değeri değilse hata verir."""
    k = round(rate * n)
    if abs(k / n - rate) > 1e-3:
        raise ValueError(f"{rate} değeri n={n} için bir k/n oranına karşılık gelmiyor")
    return k


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval metrikleri için güven aralıkları")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--markdown", action="store_true", help="README için Markdown tablo yazdır")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports = load_reports(reports_dir)

    full_size = max(r["quality_metrics"].get("total_examples", 0) for r in reports.values())
    for method, report in list(reports.items()):
        size = report["quality_metrics"].get("total_examples", 0)
        if size != full_size:
            print(f"[WARN] {method}: {size} örnekle değerlendirilmiş (diğerleri {full_size}); atlandı")
            del reports[method]

    results: dict[str, dict] = {}
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for method, report in reports.items():
        qm = report["quality_metrics"]
        results[method] = {}
        for key, _, denom_key in BINARY_METRICS:
            n = qm[denom_key]
            k = rate_to_count(qm[key], n)
            counts[(method, key)] = (k, n)
            boot = bootstrap_ci_from_counts(k, n, n_resamples=args.n_resamples)
            wil = wilson_ci(k, n)
            results[method][key] = {
                "successes": k,
                "n": n,
                "estimate": boot.estimate,
                "bootstrap_95": [boot.low, boot.high],
                "wilson_95": [wil.low, wil.high],
            }

    comparisons = {}
    key = "positive_tool_selection_accuracy"
    for a, b in combinations(reports, 2):
        (ka, na), (kb, nb) = counts[(a, key)], counts[(b, key)]
        diff = unpaired_bootstrap_diff_from_counts(kb, nb, ka, na, n_resamples=args.n_resamples)
        comparisons[f"{b} - {a}"] = {
            "diff": diff.estimate,
            "bootstrap_95": [diff.low, diff.high],
            "significant": not (diff.low <= 0 <= diff.high),
        }

    out_path = reports_dir / "confidence_intervals.json"
    out_path.write_text(
        json.dumps(
            {
                "note": (
                    "İkili metrikler sayımlardan yeniden kurulup bootstrap'lendi "
                    f"({args.n_resamples} örnekleme, %95). Karşılaştırmalar bağımsız bootstrap "
                    "(temkinli). Argument accuracy için örnek bazlı tahminler gerekiyor."
                ),
                "metrics": results,
                "tool_selection_comparisons": comparisons,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fmt(r: dict, field: str = "bootstrap_95") -> str:
        lo, hi = r[field]
        return f"{r['estimate'] * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"

    if args.markdown:
        print("| Method | " + " | ".join(label for _, label, _ in BINARY_METRICS) + " |")
        print("|:---|" + ":---:|" * len(BINARY_METRICS))
        for method, res in results.items():
            print(f"| {method} | " + " | ".join(fmt(res[k]) for k, _, _ in BINARY_METRICS) + " |")
        print("\n| Comparison (Tool Selection) | Difference [95% CI] | Significant? |")
        print("|:---|:---:|:---:|")
        for name, c in comparisons.items():
            lo, hi = c["bootstrap_95"]
            print(f"| {name} | {c['diff'] * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}] | "
                  f"{'yes' if c['significant'] else 'no'} |")
    else:
        for method, res in results.items():
            print(f"\n{method}")
            for k, label, _ in BINARY_METRICS:
                r = res[k]
                print(f"  {label:22s} {r['successes']:3d}/{r['n']:<3d} bootstrap {fmt(r)}   "
                      f"wilson {fmt(r, 'wilson_95')}")
        print("\nTool selection farkları (B - A, bağımsız bootstrap):")
        for name, c in comparisons.items():
            lo, hi = c["bootstrap_95"]
            mark = "ANLAMLI" if c["significant"] else "anlamlı değil"
            print(f"  {name:18s} {c['diff'] * 100:+6.1f} pp  [{lo * 100:+6.1f}, {hi * 100:+6.1f}]  {mark}")

    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
