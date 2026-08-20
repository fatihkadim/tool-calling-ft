"""Eval harness: bir checkpoint'i veya base modeli yükleyip test seti üzerinde

metrics.py'daki fonksiyonları çalıştırır, sonucu reports/ altına JSON olarak yazar.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_calling_ft.data.tool_schema import extract_tool_names
from tool_calling_ft.eval.metrics import ToolCallExample, aggregate_metrics
from tool_calling_ft.utils.logging import count_trainable_params, save_metrics_report, track_vram_and_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
    adapter_path: str | Path | None = None,
    device: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
) -> tuple[Any, Any]:
    """Base modeli veya adapter yüklenmiş fine-tune modelini ve tokenizer'ı yükler."""
    logger.info("Tokenizer yükleniyor: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Model yükleniyor: %s (device=%s, dtype=%s)", model_name_or_path, device, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype if torch_dtype != "auto" else (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        logger.info("LoRA/QLoRA adapter yükleniyor: %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def build_generation_prompt(item: dict[str, Any]) -> str:
    """Modelin devamını üretmesi için asistan öncesi prompt metnini oluşturur."""
    system_text = item.get("system", "").strip()
    user_text = item.get("user", "").strip()
    return f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def run_inference_on_samples(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> tuple[list[ToolCallExample], dict[str, Any]]:
    """Test örnekleri üzerinde inference çalıştırır ve ToolCallExample listesi üretir."""
    examples: list[ToolCallExample] = []
    total_generated_tokens = 0
    total_generation_time = 0.0

    # Model cihazını güvenli tespit et
    try:
        model_device = getattr(model, "device", None)
        if model_device is None:
            model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        model_device = torch.device("cpu")

    for i in range(0, len(samples), batch_size):
        batch_items = samples[i : i + batch_size]
        prompts = [build_generation_prompt(item) for item in batch_items]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_duration = time.perf_counter() - start_time
        total_generation_time += batch_duration

        # Yeni üretilen token'ları ayrıştır
        for j, (item, out) in enumerate(zip(batch_items, outputs)):
            prompt_len = input_ids[j].shape[0]
            gen_tokens = out[prompt_len:]
            total_generated_tokens += len(gen_tokens)

            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            # <|im_end|> sonrası metni temizle
            if "<|im_end|>" in gen_text:
                gen_text = gen_text.split("<|im_end|>")[0].strip()

            known_tools_set = set(extract_tool_names(item.get("tools", [])))
            example = ToolCallExample(
                predicted_raw=gen_text,
                expected_tool=item.get("expected_tool"),
                expected_arguments=item.get("expected_arguments"),
                known_tools=known_tools_set,
            )
            examples.append(example)

    throughput = (
        round(total_generated_tokens / total_generation_time, 2)
        if total_generation_time > 0
        else 0.0
    )
    latency_per_sample = (
        round((total_generation_time / len(samples)) * 1000, 2) if samples else 0.0
    )

    perf_stats = {
        "total_samples": len(samples),
        "total_generated_tokens": total_generated_tokens,
        "total_generation_seconds": round(total_generation_time, 2),
        "throughput_tokens_per_sec": throughput,
        "latency_ms_per_sample": latency_per_sample,
    }
    return examples, perf_stats


def run_eval(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
    adapter_path: str | Path | None = None,
    dataset_path: str | Path = "data/processed/eval_subset.jsonl",
    method_name: str = "baseline",
    max_samples: int | None = None,
    batch_size: int = 4,
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Tam değerlendirme sürecini çalıştırır ve rapor kaydeder."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Değerlendirme veri seti bulunamadı: {dataset_file}")

    samples: list[dict[str, Any]] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    logger.info(
        "Değerlendirme başlatılıyor: method='%s', dataset='%s' (%d örnek)",
        method_name,
        dataset_file,
        len(samples),
    )

    with track_vram_and_time(f"Eval: {method_name}") as vram_stats:
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
        )
        param_stats = count_trainable_params(model)

        examples, perf_stats = run_inference_on_samples(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            batch_size=batch_size,
        )

        all_known_tools: set[str] = set()
        for s in samples:
            all_known_tools.update(extract_tool_names(s.get("tools", [])))

        quality_metrics = aggregate_metrics(examples, all_known_tools)

    sample_predictions = [
        {
            "predicted_raw": ex.predicted_raw,
            "expected_tool": ex.expected_tool,
            "expected_arguments": ex.expected_arguments,
        }
        for ex in examples[:5]
    ]

    report = {
        "method": method_name,
        "base_model": model_name_or_path,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dataset_path": str(dataset_file),
        "quality_metrics": quality_metrics,
        "performance_metrics": {**perf_stats, **vram_stats},
        "parameter_stats": param_stats,
        "sample_predictions": sample_predictions,
    }

    report_path = save_metrics_report(report, f"{method_name}_metrics.json", output_dir)
    logger.info("Değerlendirme tamamlandı. Sonuçlar: %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool Calling Evaluation Harness")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model adı veya yolu",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA/QLoRA/DoRA adapter dizin yolu",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/eval_subset.jsonl",
        help="Değerlendirme veri seti yolu",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        help="Metod adı (baseline, lora, qlora, dora, full_ft)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maksimum test örneği sayısı (None = tümü)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch boyutu",
    )

    args = parser.parse_args()

    results = run_eval(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        method_name=args.method,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 50)
    print(f" EVALUATION REPORT: {args.method.upper()}")
    print("=" * 50)
    for k, v in results["quality_metrics"].items():
        print(f"  {k:30s}: {v}")
    print("-" * 50)
    for k, v in results["performance_metrics"].items():
        print(f"  {k:30s}: {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
