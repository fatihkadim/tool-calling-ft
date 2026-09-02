"""tests/test_harness.py - eval harness testleri."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from tool_calling_ft.eval.harness import (
    build_generation_prompt,
    load_model_and_tokenizer,
    resolve_torch_dtype,
    run_eval,
)
from tool_calling_ft.utils.logging import (
    count_trainable_params,
    save_metrics_report,
    track_vram_and_time,
)


def test_resolve_torch_dtype():
    # Direkt torch.dtype nesnesi
    assert resolve_torch_dtype(torch.float16) == torch.float16
    assert resolve_torch_dtype(torch.bfloat16) == torch.bfloat16

    # String mapping
    assert resolve_torch_dtype("float16") == torch.float16
    assert resolve_torch_dtype("fp16") == torch.float16
    assert resolve_torch_dtype("bfloat16") == torch.bfloat16
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    assert resolve_torch_dtype("float32") == torch.float32
    assert resolve_torch_dtype("fp32") == torch.float32

    # GPU olmayan ortamda auto -> float32
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_torch_dtype("auto") == torch.float32

    # T4 GPU simülasyonu: cuda var ama bf16 desteklenmiyor -> float16
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            assert resolve_torch_dtype("auto") == torch.float16

    # A100 GPU simülasyonu: cuda var ve bf16 destekleniyor -> bfloat16
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            assert resolve_torch_dtype("auto") == torch.bfloat16


def test_build_generation_prompt():
    item = {
        "system": "System prompt with tools",
        "user": "What is the weather?",
    }
    prompt = build_generation_prompt(item)
    assert "<|im_start|>system\nSystem prompt with tools<|im_end|>" in prompt
    assert "<|im_start|>user\nWhat is the weather?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

    # Boş system/user
    item_empty = {"system": "", "user": ""}
    prompt_empty = build_generation_prompt(item_empty)
    assert "<|im_start|>system\n<|im_end|>" in prompt_empty
    assert "<|im_start|>assistant\n" in prompt_empty

    # Eksik key → boş string kullanılmalı
    item_missing = {}
    prompt_missing = build_generation_prompt(item_missing)
    assert "<|im_start|>assistant\n" in prompt_missing


def test_count_trainable_params():
    linear = torch.nn.Linear(10, 5)
    stats = count_trainable_params(linear)
    assert stats["trainable_params"] == 55  # 10*5 + 5
    assert stats["all_params"] == 55
    assert stats["trainable_percentage"] == 100.0

    # Parametreleri dondur (freeze)
    for p in linear.parameters():
        p.requires_grad = False
    stats_frozen = count_trainable_params(linear)
    assert stats_frozen["trainable_params"] == 0
    assert stats_frozen["trainable_percentage"] == 0.0

    # Kısmi freeze (Sequential model)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),  # 55 param
        torch.nn.Linear(5, 3),   # 18 param
    )
    # İlk katmanı dondur
    for p in model[0].parameters():
        p.requires_grad = False
    stats_partial = count_trainable_params(model)
    assert stats_partial["trainable_params"] == 18
    assert stats_partial["all_params"] == 73
    assert 24.0 < stats_partial["trainable_percentage"] < 25.0


def test_track_vram_and_time():
    with track_vram_and_time("Test block") as stats:
        x = sum(range(1000))
    assert stats["elapsed_seconds"] >= 0.0
    assert "peak_vram_mb" in stats
    assert stats["description"] == "Test block"


def test_save_metrics_report(tmp_path: Path):
    metrics = {"accuracy": 0.95, "loss": 0.05}
    report_path = save_metrics_report(metrics, "test_report.json", tmp_path)
    assert report_path.exists()
    assert report_path.name == "test_report.json"

    with open(report_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == metrics

    # .json uzantısı otomatik eklenmeli
    report_path2 = save_metrics_report(metrics, "test_report_2", tmp_path)
    assert report_path2.name == "test_report_2.json"


def test_run_eval_end_to_end_mock(tmp_path: Path):
    # Dummy eval veri seti
    eval_file = tmp_path / "test_eval.jsonl"
    report_dir = tmp_path / "reports"

    sample = {
        "id": "mock-001",
        "system": "System prompt",
        "user": "Check weather in Izmir",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "expected_tool": "get_weather",
        "expected_arguments": {"city": "Izmir"},
    }

    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    # Mock Model ve Tokenizer
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters.side_effect = lambda: iter([dummy_param])

    # mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Model generate mock çıktısı
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Izmir"}}\n</tool_call><|im_end|>'
    )

    with patch(
        "tool_calling_ft.eval.harness.load_model_and_tokenizer",
        return_value=(mock_model, mock_tokenizer),
    ):
        report = run_eval(
            model_name_or_path="dummy-model",
            dataset_path=eval_file,
            method_name="test_method",
            output_dir=report_dir,
        )

    # Temel rapor yapısı
    assert report["method"] == "test_method"
    assert report["base_model"] == "dummy-model"
    assert report["adapter_path"] is None

    # Kalite metrikleri
    assert report["quality_metrics"]["total_examples"] == 1
    assert report["quality_metrics"]["tool_selection_accuracy"] == 1.0
    assert report["quality_metrics"]["argument_accuracy"] == 1.0
    assert report["quality_metrics"]["json_validity_rate"] == 1.0

    # Performans metrikleri mevcut olmalı
    assert "total_samples" in report["performance_metrics"]
    assert "throughput_tokens_per_sec" in report["performance_metrics"]
    assert "elapsed_seconds" in report["performance_metrics"]

    # Parametre istatistikleri mevcut olmalı
    assert "trainable_params" in report["parameter_stats"]
    assert "all_params" in report["parameter_stats"]

    # Rapor dosyası kaydedilmiş olmalı
    assert (report_dir / "test_method_metrics.json").exists()

    # Kaydedilen rapor ile dönen rapor eşleşmeli
    with open(report_dir / "test_method_metrics.json", "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["method"] == "test_method"
    assert saved_report["quality_metrics"]["tool_selection_accuracy"] == 1.0


def test_run_eval_file_not_found():
    """Olmayan dataset dosyasıyla FileNotFoundError fırlatmalı."""
    with pytest.raises(FileNotFoundError):
        run_eval(
            model_name_or_path="dummy-model",
            dataset_path="nonexistent/path.jsonl",
            method_name="test",
        )


def test_load_model_and_tokenizer_mock():
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|endoftext|>"

    mock_model = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_lm:
            model, tok = load_model_and_tokenizer(
                model_name_or_path="dummy-model",
                torch_dtype="float16",
                load_in_4bit=True,
            )
            assert tok.padding_side == "left"
            assert tok.pad_token == "<|endoftext|>"
            mock_lm.assert_called_once()
            call_kwargs = mock_lm.call_args.kwargs
            assert "quantization_config" in call_kwargs
            assert call_kwargs["quantization_config"].load_in_4bit is True


def test_run_eval_recovers_sample_4(tmp_path: Path):
    """Run 1 Sample 4 gibi XML etiketi olmayan ve arkasında açıklama bulunan çıktılar run_eval'de kurtarılmalı."""
    sample_item = {
        "system": "System prompt",
        "user": "Set up webhook",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_task_completed_webhook",
                    "parameters": {
                        "type": "object",
                        "properties": {"planner_id": {"type": "string"}, "task_id": {"type": "string"}},
                    },
                },
            }
        ],
        "expected_tool": "create_task_completed_webhook",
        "expected_arguments": {"planner_id": "abc123", "task_id": "task456"},
    }
    eval_file = tmp_path / "eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_item) + "\n")

    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters.side_effect = lambda: iter([dummy_param])

    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    mock_tok.convert_tokens_to_ids.return_value = 151645
    mock_tok.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    # Sample 4 ham çıktısı (etiketsiz, trailing metinli, padding'li)
    mock_tok.decode.return_value = (
        '\n{"name": "create_task_completed_webhook", "arguments": {"planner_id": "abc123", "task_id": "task456"}}\n'
        '>manual\n(Have a look at documentation)<|endoftext|><|endoftext|>'
    )

    with patch(
        "tool_calling_ft.eval.harness.load_model_and_tokenizer",
        return_value=(mock_model, mock_tok),
    ):
        report = run_eval(
            model_name_or_path="dummy-model",
            dataset_path=eval_file,
            method_name="sample4_test",
            output_dir=report_dir,
        )

    assert report["quality_metrics"]["tool_selection_accuracy"] == 1.0
    assert report["quality_metrics"]["positive_tool_selection_accuracy"] == 1.0
    assert report["quality_metrics"]["argument_accuracy"] == 1.0
    assert report["quality_metrics"]["positive_argument_accuracy"] == 1.0
    assert report["quality_metrics"]["json_validity_rate"] == 1.0
    # Stop token temizliği kontrolü: <|endoftext|> silinmiş olmalı
    assert "<|endoftext|>" not in report["sample_predictions"][0]["predicted_raw"]
