"""tests/test_harness.py - eval harness testleri."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from tool_calling_ft.eval.harness import (
    build_generation_prompt,
    run_eval,
    run_inference_on_samples,
)
from tool_calling_ft.utils.logging import count_trainable_params, track_vram_and_time


def test_build_generation_prompt():
    item = {
        "system": "System prompt with tools",
        "user": "What is the weather?",
    }
    prompt = build_generation_prompt(item)
    assert "<|im_start|>system\nSystem prompt with tools<|im_end|>" in prompt
    assert "<|im_start|>user\nWhat is the weather?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


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


def test_track_vram_and_time():
    with track_vram_and_time("Test block") as stats:
        x = sum(range(1000))
    assert stats["elapsed_seconds"] >= 0.0
    assert "peak_vram_mb" in stats


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

    assert report["method"] == "test_method"
    assert report["quality_metrics"]["total_examples"] == 1
    assert report["quality_metrics"]["tool_selection_accuracy"] == 1.0
    assert report["quality_metrics"]["argument_accuracy"] == 1.0
    assert (report_dir / "test_method_metrics.json").exists()
