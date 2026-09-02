"""tests/test_data.py - data modülü için pytest testleri."""

import json
from pathlib import Path

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

    # Tek dict → list[dict] dönüşümü
    single_tool = json.dumps(DEFAULT_TOOLS[0])
    parsed_single = parse_tools(single_tool)
    assert len(parsed_single) == 1

    # Geçersiz string → boş liste
    assert parse_tools("not valid json") == []

    # Boş string → boş liste
    assert parse_tools("") == []


def test_build_system_prompt():
    prompt = build_system_prompt(DEFAULT_TOOLS[:2])
    assert "<tools>" in prompt
    assert "</tools>" in prompt
    assert "get_current_weather" in prompt
    assert "<tool_call>" in prompt
    assert "function calling AI model" in prompt


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

    # Fallback: XML tag olmadan bare JSON
    bare_json = '{"name": "calc", "arguments": {"x": 42}}'
    bare_calls = parse_tool_calls_from_text(bare_json)
    assert len(bare_calls) == 1
    assert bare_calls[0]["name"] == "calc"

    # Fallback: JSON arkasından metin gelmesi (Sample 4 durumu)
    trailing_text = '\n{"name": "calc", "arguments": {"x": 42}}\nThis was calculated automatically.'
    trailing_calls = parse_tool_calls_from_text(trailing_text)
    assert len(trailing_calls) == 1
    assert trailing_calls[0]["name"] == "calc"
    assert trailing_calls[0]["arguments"] == {"x": 42}

    # Fallback: Markdown ```json ... ``` bloğu
    md_text = '```json\n{"name": "fetch_data", "arguments": {"id": 10}}\n```'
    md_calls = parse_tool_calls_from_text(md_text)
    assert len(md_calls) == 1
    assert md_calls[0]["name"] == "fetch_data"

    # Boş string
    assert parse_tool_calls_from_text("") == []


def test_format_chatml():
    chatml = format_chatml("System prompt", "User query", "Assistant response")
    assert "<|im_start|>system\nSystem prompt<|im_end|>" in chatml
    assert "<|im_start|>user\nUser query<|im_end|>" in chatml
    assert "<|im_start|>assistant\nAssistant response<|im_end|>" in chatml

    # Whitespace temizleme
    chatml_ws = format_chatml("  System  ", "  User  ", "  Assistant  ")
    assert "<|im_start|>system\nSystem<|im_end|>" in chatml_ws


def test_parse_raw_item():
    # Pozitif örnek (tool call var)
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
    assert "text" in parsed  # ChatML formatı mevcut

    # Negatif örnek (tool call yok, düz metin)
    raw_neg = {
        "id": "test-neg",
        "conversations": [
            {"from": "system", "value": "System prompt"},
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "The answer is 4."},
        ],
        "tools": "[]",
    }
    parsed_neg = parse_raw_item(raw_neg)
    assert parsed_neg is not None
    assert parsed_neg["is_tool_call"] is False
    assert parsed_neg["expected_tool"] is None
    assert parsed_neg["expected_arguments"] is None

    # Yetersiz conversation (3'ten az) → None döner
    raw_short = {
        "id": "test-short",
        "conversations": [
            {"from": "system", "value": "System"},
            {"from": "human", "value": "Hello"},
        ],
    }
    assert parse_raw_item(raw_short) is None

    # Boş değerler → None döner
    raw_empty = {
        "id": "test-empty",
        "conversations": [
            {"from": "system", "value": ""},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"},
        ],
    }
    assert parse_raw_item(raw_empty) is None


def test_build_negative_examples():
    from tool_calling_ft.data.prepare_dataset import TRAIN_NEGATIVE_TEMPLATES

    negs = build_negative_examples([DEFAULT_TOOLS], templates=TRAIN_NEGATIVE_TEMPLATES, count=10, seed=42)
    assert len(negs) == 10
    for neg in negs:
        assert neg["is_tool_call"] is False
        assert neg["expected_tool"] is None
        assert neg["expected_arguments"] is None
        assert "<|im_start|>system" in neg["text"]
        assert "<tool_call>" not in neg["assistant"]
        assert neg["all_expected_tool_calls"] == []
        assert len(neg["messages"]) == 3

    # Boş tools_pool → DEFAULT_TOOLS kullanılmalı
    negs_default = build_negative_examples([], templates=TRAIN_NEGATIVE_TEMPLATES, count=3, seed=99)
    assert len(negs_default) == 3
    for neg in negs_default:
        assert neg["tools"] is not None

    # Seed determinizmi: aynı seed → aynı sonuç
    negs_a = build_negative_examples([DEFAULT_TOOLS], templates=TRAIN_NEGATIVE_TEMPLATES, count=5, seed=42)
    negs_b = build_negative_examples([DEFAULT_TOOLS], templates=TRAIN_NEGATIVE_TEMPLATES, count=5, seed=42)
    for a, b in zip(negs_a, negs_b):
        assert a["user"] == b["user"]
        assert a["assistant"] == b["assistant"]


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
        f.writelines(json.dumps(item) + "\n" for item in sample_items)

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

    # Dosya içeriklerini doğrula
    train_lines = (processed_dir / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")
    val_lines = (processed_dir / "val.jsonl").read_text(encoding="utf-8").strip().split("\n")
    eval_lines = (processed_dir / "eval_subset.jsonl").read_text(encoding="utf-8").strip().split("\n")

    assert len(train_lines) == summary["train_samples"]
    assert len(val_lines) == summary["val_samples"]
    assert len(eval_lines) <= summary["eval_subset_samples"]

    # Her satır geçerli JSON olmalı
    for line in train_lines[:5]:
        item = json.loads(line)
        assert "user" in item
        assert "assistant" in item
        assert "is_tool_call" in item

    # Summary dosyası doğru içeriğe sahip olmalı
    with open(processed_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        saved_summary = json.load(f)
    assert saved_summary["total_samples"] == 25
    assert saved_summary["seed"] == 42
