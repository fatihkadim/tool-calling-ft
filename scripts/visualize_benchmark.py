"""Benchmark karşılaştırma görselleştirme scripti.

reports/ altındaki *_metrics.json dosyalarını okuyup karşılaştırma grafikleri üretir.
DoRA ve Full FT sonuçları geldiğinde otomatik olarak dahil edilir.

Kullanım:
    uv run python scripts/visualize_benchmark.py
    uv run python scripts/visualize_benchmark.py --output-dir reports/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # GUI backend gerektirmez

# ─── Renk paleti (her yöntem için tutarlı) ───
METHOD_COLORS = {
    "baseline": "#6C757D",  # Gri
    "lora": "#0D6EFD",      # Mavi
    "qlora": "#198754",      # Yeşil
    "dora": "#DC3545",       # Kırmızı
    "full_ft": "#FFC107",    # Sarı/Altın
}

METHOD_LABELS = {
    "baseline": "Baseline (0-shot)",
    "lora": "LoRA (16-bit)",
    "qlora": "QLoRA (4-bit NF4)",
    "dora": "DoRA",
    "full_ft": "Full Fine-Tuning",
}

# Yükleme sırası (bu sırada gösterilir)
METHOD_ORDER = ["baseline", "lora", "qlora", "dora", "full_ft"]


def load_all_reports(reports_dir: Path) -> dict[str, dict]:
    """reports/ altındaki tüm *_metrics.json dosyalarını yükler."""
    reports = {}
    for json_file in sorted(reports_dir.glob("*_metrics.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            method = data.get("method", json_file.stem.replace("_metrics", ""))
            reports[method] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] {json_file.name} okunamadi: {e}")
    return reports


def get_ordered_methods(reports: dict[str, dict]) -> list[str]:
    """Mevcut raporları METHOD_ORDER sırasına göre döndürür."""
    return [m for m in METHOD_ORDER if m in reports]


def get_comparable_methods(reports: dict[str, dict], methods: list[str]) -> list[str]:
    """Yalnızca aynı eval seti boyutunda değerlendirilmiş yöntemleri döndürür.

    Baseline şu an 5 örneklik bir smoke test ile çalıştırıldığı için 100 örneklik
    eval setindeki yöntemlerle aynı grafikte gösterilmemeli.
    """
    sizes = {m: reports[m].get("quality_metrics", {}).get("total_examples", 0) for m in methods}
    full_size = max(sizes.values(), default=0)
    comparable = [m for m in methods if sizes[m] == full_size]
    for m in methods:
        if m not in comparable:
            print(
                f"  [WARN] {m}: {sizes[m]} örnekle değerlendirilmiş (diğerleri {full_size}); "
                "kalite ve hız grafiklerinden çıkarıldı"
            )
    return comparable


def add_footnote(fig, text: str):
    """Grafiğin altına küçük bir açıklama notu ekler."""
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color="#b0b0b0", wrap=True)


def setup_style():
    """Grafik stilini ayarla — koyu tema, modern font."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
    })


def plot_quality_bar_chart(reports: dict, methods: list[str], output_dir: Path):
    """Ana kalite metriklerini bar chart olarak çizer (pozitif örnekler üzerinden)."""
    metrics = [
        ("positive_tool_selection_accuracy", "Tool Selection\nAccuracy (Pozitif)"),
        ("positive_argument_accuracy", "Argument\nAccuracy (Pozitif)"),
        ("positive_json_validity", "JSON Validity\n(Pozitif)"),
        ("negative_rejection_accuracy", "Negative\nRejection Acc."),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        qm = reports[method].get("quality_metrics", {})
        # Toplam metriklere geri düşmüyoruz: pozitif/negatif ayrımı olmayan bir
        # rapor bu grafikte yanlış (şişirilmiş) değerler gösterirdi.
        values = [qm.get(metric_key, 0.0) * 100 for metric_key, _ in metrics]

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width * 0.9,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#888"),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )

        # Değer etiketleri
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    color=METHOD_COLORS.get(method, "#888"),
                )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Tool Calling Fine-Tuning — Kalite Metrikleri Karşılaştırması",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(
        loc="upper left", framealpha=0.8,
        facecolor="#16213e", edgecolor="#444",
    )
    ax.grid(axis="y", alpha=0.3)

    add_footnote(
        fig,
        "Pozitif n=80, negatif n=20 (negatifte 1 örnek = %5), tek seed. Baseline gösterilmiyor: "
        "henüz yalnızca 5 örnekle çalıştırıldı.\nParser etiketsiz {\"name\": ...} JSON'u da tool "
        "çağrısı sayıyor; adapter skorları (özellikle QLoRA) sıkı parser'la düşebilir.",
    )
    fig.tight_layout()
    out_path = output_dir / "quality_bar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_radar_chart(reports: dict, methods: list[str], output_dir: Path):
    """5 kalite metriğini radar chart olarak çizer."""
    categories = [
        ("tool_selection_accuracy", "Tool Selection"),
        ("argument_accuracy", "Argument Acc."),
        ("json_validity_rate", "JSON Validity"),
    ]

    # Ters metrikler (düşük = iyi): 1 - değer olarak göster
    inv_categories = [
        ("invalid_tool_call_rate", "Valid Tool Rate"),
        ("unnecessary_tool_call_rate", "Rejection Rate"),
    ]

    all_labels = [c[1] for c in categories] + [c[1] for c in inv_categories]
    N = len(all_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Çemberi kapat

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_facecolor("#16213e")

    for method in methods:
        qm = reports[method].get("quality_metrics", {})
        values = []
        for key, _ in categories:
            values.append(qm.get(key, 0.0))
        for key, _ in inv_categories:
            values.append(1.0 - qm.get(key, 0.0))

        values += values[:1]
        color = METHOD_COLORS.get(method, "#888")
        ax.plot(angles, values, "o-", linewidth=2, color=color,
                label=METHOD_LABELS.get(method, method), markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.set_title(
        "Kalite Metrikleri Radar Chart",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.1),
        framealpha=0.8, facecolor="#16213e", edgecolor="#444",
    )

    add_footnote(
        fig,
        "100 örnek üzerinden toplam metrikler; Rejection Rate = 1 - gereksiz çağrı oranı (n=20).\n"
        "Baseline gösterilmiyor: henüz yalnızca 5 örnekle çalıştırıldı.",
    )
    fig.tight_layout()
    out_path = output_dir / "quality_radar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_vram_vs_accuracy(reports: dict, methods: list[str], output_dir: Path):
    """VRAM kullanımı vs Tool Selection Accuracy scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in methods:
        qm = reports[method].get("quality_metrics", {})
        pm = reports[method].get("performance_metrics", {})

        accuracy = qm.get("positive_tool_selection_accuracy",
                          qm.get("tool_selection_accuracy", 0)) * 100
        vram = pm.get("peak_vram_mb", 0)

        if vram == 0:
            # CPU'da çalıştırılmış (baseline)
            continue

        color = METHOD_COLORS.get(method, "#888")
        ax.scatter(
            vram, accuracy,
            s=200, c=color, edgecolors="white",
            linewidths=2, zorder=5, alpha=0.9,
        )
        ax.annotate(
            METHOD_LABELS.get(method, method),
            (vram, accuracy),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10, fontweight="bold",
            color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
        )

    # Eksen 0'dan başlar; aksi halde ~%5'lik VRAM farkı olduğundan büyük görünür
    ax.set_xlim(0, 16384)  # T4: 16 GB
    ax.set_ylim(0, 100)
    ax.set_xlabel("Peak VRAM — değerlendirme sırasında (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tool Selection Accuracy — Pozitif (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Değerlendirme VRAM'i vs Doğruluk",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.grid(True, alpha=0.3)

    add_footnote(
        fig,
        "VRAM, batch=4 üretim sırasında ölçüldü; büyük kısmı aktivasyon/KV cache (ağırlıklar ~1 GB). "
        "Eğitim VRAM'i ölçülmedi.\nBaseline CPU'da çalıştığı için gösterilmiyor.",
    )
    fig.tight_layout()
    out_path = output_dir / "vram_vs_accuracy.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_training_time_comparison(
    reports: dict, methods: list[str], comparable_methods: list[str], output_dir: Path
):
    """Eğitim süresi ve throughput karşılaştırması (yan yana bar chart)."""
    # Sadece eğitilmiş modelleri al
    trained_methods = [m for m in methods if "training_metrics" in reports[m] or m != "baseline"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # JSON'da training_metrics yoksa kayıt dışı/tahmini süreleri kullan. Bunlar
    # grafikte taralı (hatch) ve etiketli gösterilir, ölçülmüş gibi sunulmaz.
    UNRECORDED_TRAIN_HOURS = {
        "qlora": (3.8, "kayıtlı değil"),
        "dora": (4.8, "kayıtlı değil"),
        "full_ft": (8.0, "tahmin"),
    }

    train_times = []
    train_labels = []
    train_colors = []
    train_notes = []
    for method in trained_methods:
        tm = reports[method].get("training_metrics", {})
        t = tm.get("train_runtime_seconds", 0)
        if t > 0:
            hours, note = t / 3600, "ölçüldü"
        else:
            hours, note = UNRECORDED_TRAIN_HOURS.get(method, (0, ""))
        if hours > 0:
            train_times.append(hours)
            train_labels.append(METHOD_LABELS.get(method, method))
            train_colors.append(METHOD_COLORS.get(method, "#888"))
            train_notes.append(note)

    if train_times:
        bars = ax1.barh(
            train_labels, train_times,
            color=train_colors, edgecolor="white",
            linewidth=0.5, alpha=0.9,
        )
        for bar, val, note in zip(bars, train_times, train_notes):
            if note != "ölçüldü":
                bar.set_hatch("//")
                bar.set_alpha(0.45)
            ax1.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}h ({note})", ha="left", va="center",
                fontsize=9, fontweight="bold",
            )
        ax1.set_xlim(0, max(train_times) * 1.6)
        ax1.set_xlabel("Eğitim Süresi (saat)", fontsize=11, fontweight="bold")
        ax1.set_title("Training Time", fontsize=13, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

    # --- Sağ: Inference Throughput ---
    throughputs = []
    tp_labels = []
    tp_colors = []
    tokens_per_sample = []
    for method in comparable_methods:
        pm = reports[method].get("performance_metrics", {})
        tp = pm.get("throughput_tokens_per_sec", 0)
        if tp > 0:
            throughputs.append(tp)
            tp_labels.append(METHOD_LABELS.get(method, method))
            tp_colors.append(METHOD_COLORS.get(method, "#888"))
            n = pm.get("total_samples", 0)
            tokens_per_sample.append(pm.get("total_generated_tokens", 0) / n if n else 0)

    if throughputs:
        bars = ax2.barh(
            tp_labels, throughputs,
            color=tp_colors, edgecolor="white",
            linewidth=0.5, alpha=0.9,
        )
        for bar, val, tps in zip(bars, throughputs, tokens_per_sample):
            ax2.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} tok/s · {tps:.0f} token/örnek", ha="left", va="center",
                fontsize=9, fontweight="bold",
            )
        ax2.set_xlim(0, max(throughputs) * 1.8)
        ax2.set_xlabel("Inference Throughput (tokens/sec)", fontsize=11, fontweight="bold")
        ax2.set_title("Inference Speed", fontsize=13, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Performans Karşılaştırması — Training Time & Inference Speed",
        fontsize=14, fontweight="bold", y=1.02,
    )
    add_footnote(
        fig,
        "Sol: taralı çubuklar metrik dosyalarında kayıtlı olmayan veya tahmini sürelerdir. "
        "Sağ: 256 token/örnek, modelin hiç durmadığı ve max_new_tokens sınırına kadar ürettiği anlamına gelir;\n"
        "bu modellerin tok/s değeri kullanılabilir cevap hızını göstermez. Baseline (CPU, 5 örnek) gösterilmiyor.",
    )
    fig.tight_layout()
    out_path = output_dir / "performance_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_parameter_comparison(reports: dict, methods: list[str], output_dir: Path):
    """Trainable parameter sayısını logaritmik ölçekte karşılaştırır."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline eğitilmedi, grafikten çıkar
    trained_methods = [m for m in methods if m != "baseline"]

    # Eval modunda trainable_params=0 dönebiliyor, gerçek değerleri kullan
    KNOWN_TRAINABLE = {
        "lora": 2_162_688,
        "qlora": 2_162_688,
        "dora": 2_211_840,
        "full_ft": 494_032_768,
    }

    labels = []
    all_params_vals = []
    trainable_vals = []
    colors = []

    for method in trained_methods:
        ps = reports[method].get("parameter_stats", {})
        total = ps.get("all_params", 0)

        if total == 0:
            continue

        trainable = ps.get("trainable_params", 0)
        lora = ps.get("lora_params", 0)
        known = KNOWN_TRAINABLE.get(method, 0)
        # En büyük sıfır olmayan değeri al
        actual_trainable = max(trainable if trainable != total else 0, lora, known)
        if actual_trainable == 0:
            actual_trainable = trainable  # fallback

        labels.append(METHOD_LABELS.get(method, method))
        all_params_vals.append(total / 1e6)
        trainable_vals.append(actual_trainable / 1e6)
        colors.append(METHOD_COLORS.get(method, "#888"))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, all_params_vals, width,
           color="#555", edgecolor="white", linewidth=0.5, alpha=0.6)
    bars = ax.bar(x + width / 2, trainable_vals, width,
           color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    # Değer etiketleri
    for bar, val, color in zip(bars, trainable_vals, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.1f}M" if val >= 1 else f"{val*1000:.0f}K",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=color,
        )

    # Lejant: Total (gri) + her yöntem kendi rengiyle
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="#555", alpha=0.6, edgecolor="white", label="Total Params (M)")]
    for label, color in zip(labels, colors):
        legend_handles.append(Patch(facecolor=color, alpha=0.9, edgecolor="white", label=f"{label} (Trainable)"))
    ax.legend(handles=legend_handles, framealpha=0.8, facecolor="#16213e", edgecolor="#444", fontsize=8)

    ax.set_ylabel("Parameters (Million)", fontsize=11, fontweight="bold")
    ax.set_title("Parametre Karşılaştırması (Eğitilebilir vs Toplam)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "parameter_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def print_summary_table(reports: dict, methods: list[str]):
    """Konsola özet tablo yazdırır."""
    print("\n" + "=" * 90)
    print("  BENCHMARK KARSILASTIRMA OZETI")
    print("=" * 90)

    header = f"{'Metrik':<35}"
    for m in methods:
        header += f" {METHOD_LABELS.get(m, m):>12}"
    print(header)
    print("-" * 90)

    metric_rows = [
        ("Tool Selection Acc. (Pozitif)", "positive_tool_selection_accuracy", "quality_metrics"),
        ("Argument Accuracy (Pozitif)", "positive_argument_accuracy", "quality_metrics"),
        ("JSON Validity (Pozitif)", "positive_json_validity", "quality_metrics"),
        ("Negative Rejection Acc.", "negative_rejection_accuracy", "quality_metrics"),
        ("Invalid Tool Call Rate", "invalid_tool_call_rate", "quality_metrics"),
        ("Unnecessary Tool Call Rate", "unnecessary_tool_call_rate", "quality_metrics"),
        ("Throughput (tok/s)", "throughput_tokens_per_sec", "performance_metrics"),
        ("Peak VRAM (MB)", "peak_vram_mb", "performance_metrics"),
    ]

    rate_metrics = {
        "positive_tool_selection_accuracy", "positive_argument_accuracy",
        "positive_json_validity", "negative_rejection_accuracy",
        "invalid_tool_call_rate", "unnecessary_tool_call_rate",
        "tool_selection_accuracy", "argument_accuracy", "json_validity_rate",
    }

    for label, key, section in metric_rows:
        row = f"  {label:<33}"
        for m in methods:
            val = reports[m].get(section, {}).get(key)
            if val is None:
                row += f" {'-':>12}"
            elif key in rate_metrics:
                row += f" {val*100:>10.1f}%"
            elif isinstance(val, float):
                row += f" {val:>12.1f}"
            else:
                row += f" {val:>12}"
        print(row)

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Görselleştirme")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Metrik JSON dosyalarının bulunduğu dizin",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Grafiklerin kaydedileceği dizin (varsayılan: reports/figures)",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[CHART] Benchmark Gorselestirme - Rapor Yukleniyor...")
    reports = load_all_reports(reports_dir)

    if not reports:
        print("  [ERROR] Hic rapor bulunamadi!")
        sys.exit(1)

    methods = get_ordered_methods(reports)
    print(f"  Bulunan yöntemler: {', '.join(methods)} ({len(methods)} adet)")
    print()

    # Stil ayarla
    setup_style()

    # Grafikleri çiz
    comparable = get_comparable_methods(reports, methods)

    print("[PLOT] Grafikler uretiliyor...")
    plot_quality_bar_chart(reports, comparable, output_dir)
    plot_radar_chart(reports, comparable, output_dir)
    plot_vram_vs_accuracy(reports, comparable, output_dir)
    plot_training_time_comparison(reports, methods, comparable, output_dir)
    plot_parameter_comparison(reports, methods, output_dir)

    # Konsol özet tablosu
    print_summary_table(reports, methods)

    print(f"[OK] Tum grafikler kaydedildi: {output_dir}/")
    print("   Toplam: 5 grafik dosyası\n")


if __name__ == "__main__":
    main()
