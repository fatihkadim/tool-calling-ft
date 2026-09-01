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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_calling_ft.data.tool_schema import extract_tool_names
from tool_calling_ft.eval.metrics import ToolCallExample, aggregate_metrics
from tool_calling_ft.utils.logging import (
    count_trainable_params,
    save_metrics_report,
    track_vram_and_time,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_torch_dtype(torch_dtype: str | torch.dtype = "auto") -> torch.dtype:
    """T4 ve bfloat16 uyumluluğu gözeterek en uygun torch.dtype'ı seçer.

    T4 gibi Turing mimarili GPU'lar bfloat16'yı donanımsal olarak desteklemez.
    Bu durumda 'auto' modu güvenli olarak float16'ya geri döner.
    """
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    dtype_str = str(torch_dtype).lower().strip()
    if dtype_str in ("float16", "fp16", "torch.float16"):
        return torch.float16
    elif dtype_str in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    elif dtype_str in ("float32", "fp32", "torch.float32"):
        return torch.float32

    # "auto" veya tanımlanamayan değer:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
    adapter_path: str | Path | None = None,
    device: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Base modeli veya adapter yüklenmiş fine-tune modelini ve tokenizer'ı yükler."""
    logger.info("Tokenizer yükleniyor: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved_dtype = resolve_torch_dtype(torch_dtype)
    logger.info(
        "Model yükleniyor: %s (device=%s, dtype=%s, load_in_4bit=%s)",
        model_name_or_path,
        device,
        resolved_dtype,
        load_in_4bit,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = device

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
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

    total_batches = (len(samples) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(samples), batch_size), total=total_batches, desc="Inference", unit="batch"):
        batch_items = samples[i : i + batch_size]
        prompts = [build_generation_prompt(item) for item in batch_items]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        start_time = time.perf_counter()
        # <|im_end|> token'ını stop token olarak ekle — Qwen'in eos_token'ı
        # <|endoftext|> (151643) ama ChatML mesaj sonu <|im_end|> (151645).
        # Bu olmadan model doğru yanıttan sonra üretmeye devam eder.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_token_ids = [tokenizer.eos_token_id]
        if isinstance(im_end_id, int) and im_end_id != tokenizer.eos_token_id:
            stop_token_ids.append(im_end_id)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
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
    torch_dtype: str | torch.dtype = "auto",
    load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    """Tam değerlendirme sürecini çalıştırır ve rapor kaydeder."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Değerlendirme veri seti bulunamadı: {dataset_file}")

    if load_in_4bit is None:
        load_in_4bit = (method_name.lower() == "qlora")

    samples: list[dict[str, Any]] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    logger.info(
        "Değerlendirme başlatılıyor: method='%s', dataset='%s' (%d örnek, load_in_4bit=%s, dtype=%s)",
        method_name,
        dataset_file,
        len(samples),
        load_in_4bit,
        torch_dtype,
    )

    with track_vram_and_time(f"Eval: {method_name}") as vram_stats:
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
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
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Model ağırlık veri tipi (varsayılan: auto - T4 için float16, A100 için bfloat16 seçer)",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        default=None,
        help="Base modeli 4-bit (NF4) kuantizasyon ile yükle (QLoRA için varsayılan olarak otomatiktir)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="4-bit kuantizasyonu zorla devre dışı bırak",
    )

    args = parser.parse_args()

    results = run_eval(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        method_name=args.method,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
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
