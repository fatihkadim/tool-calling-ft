"""tests/test_data.py - data modülü için pytest testleri."""

import json
from pathlib import Path

import pytest

from tool_calling_ft.data.prepare_dataset import (
    build_negative_examples,
    format_chatml,
    parse_raw_item,
    prepare_and_process_dataset,
)
from tool_calling_ft.data.tool_schema import (
    DEFAULT_TOOLS,
    build_system_prompt,
    extract_tool_names,
    parse_tool_calls_from_text,
    parse_tools,
)


def test_parse_tools_and_names():
    tools = parse_tools(DEFAULT_TOOLS)
    assert len(tools) == 5
    names = extract_tool_names(tools)
    assert "get_current_weather" in names
    assert "calculate_math_expression" in names

    # JSON string formatı testi
    json_str = json.dumps(DEFAULT_TOOLS)
    parsed = parse_tools(json_str)
    assert len(parsed) == 5


def test_build_system_prompt():
    prompt = build_system_prompt(DEFAULT_TOOLS[:2])
    assert "<tools>" in prompt
    assert "</tools>" in prompt
    assert "get_current_weather" in prompt
    assert "<tool_call>" in prompt


def test_parse_tool_calls_from_text():
    # Standart XML formatı
    text = (
        '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Istanbul"}}\n</tool_call>'
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_current_weather"
    assert calls[0]["arguments"] == {"location": "Istanbul"}

    # Çoklu çağrı
    multi_text = (
        '<tool_call>\n{"name": "func_a", "arguments": {"x": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "func_b", "arguments": {"y": 2}}\n</tool_call>'
    )
    multi_calls = parse_tool_calls_from_text(multi_text)
    assert len(multi_calls) == 2
    assert multi_calls[0]["name"] == "func_a"
    assert multi_calls[1]["name"] == "func_b"

    # Düz metin (tool call yok)
    no_tool_text = "The capital of France is Paris."
    assert parse_tool_calls_from_text(no_tool_text) == []


def test_format_chatml():
    chatml = format_chatml("System prompt", "User query", "Assistant response")
    assert "<|im_start|>system\nSystem prompt<|im_end|>" in chatml
    assert "<|im_start|>user\nUser query<|im_end|>" in chatml
    assert "<|im_start|>assistant\nAssistant response<|im_end|>" in chatml


def test_parse_raw_item():
    raw_item = {
        "id": "test-123",
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model.\n<tools>[]</tools>"},
            {"from": "human", "value": "Check the weather in London."},
            {
                "from": "gpt",
                "value": '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "London"}}\n</tool_call>',
            },
        ],
        "tools": json.dumps(DEFAULT_TOOLS[:1]),
        "category": "Weather",
    }
    parsed = parse_raw_item(raw_item)
    assert parsed is not None
    assert parsed["id"] == "test-123"
    assert parsed["is_tool_call"] is True
    assert parsed["expected_tool"] == "get_current_weather"
    assert parsed["expected_arguments"] == {"location": "London"}
    assert len(parsed["messages"]) == 3


def test_build_negative_examples():
    negs = build_negative_examples([DEFAULT_TOOLS], count=10, seed=42)
    assert len(negs) == 10
    for neg in negs:
        assert neg["is_tool_call"] is False
        assert neg["expected_tool"] is None
        assert neg["expected_arguments"] is None
        assert "<|im_start|>system" in neg["text"]
        assert "<tool_call>" not in neg["assistant"]


def test_dataset_pipeline_end_to_end(tmp_path: Path):
    # Dummy raw veri oluştur
    raw_file = tmp_path / "dummy_raw.jsonl"
    processed_dir = tmp_path / "processed"

    sample_items = [
        {
            "id": f"dummy-{i}",
            "conversations": [
                {"from": "system", "value": "System prompt\n<tools>[]</tools>"},
                {"from": "human", "value": f"Question {i}"},
                {
                    "from": "gpt",
                    "value": f'<tool_call>\n{{"name": "func_{i}", "arguments": {{"val": {i}}}}}\n</tool_call>',
                },
            ],
            "tools": json.dumps(DEFAULT_TOOLS),
            "category": "Test",
        }
        for i in range(20)
    ]

    with open(raw_file, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item) + "\n")

    summary = prepare_and_process_dataset(
        raw_file=raw_file,
        output_dir=processed_dir,
        val_ratio=0.2,
        num_negatives=5,
        eval_subset_size=5,
        seed=42,
    )

    assert summary["total_samples"] == 25  # 20 pozitif + 5 negatif
    assert (processed_dir / "train.jsonl").exists()
    assert (processed_dir / "val.jsonl").exists()
    assert (processed_dir / "eval_subset.jsonl").exists()
    assert (processed_dir / "dataset_summary.json").exists()
