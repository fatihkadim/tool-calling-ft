"""eval/metrics.py için pytest testleri."""

import pytest

from tool_calling_ft.eval.metrics import (
    ToolCallExample,
    aggregate_metrics,
    argument_accuracy,
    is_invalid_tool_call,
    is_unnecessary_tool_call,
    is_valid_json,
    tool_selection_correct,
)


def test_is_valid_json():
    # Geçerli JSON
    valid_raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara"}}\n</tool_call>'
    assert is_valid_json(valid_raw) is True

    # Geçersiz/bozuk JSON
    invalid_raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara",}}\n</tool_call>'
    # (Trailing comma standard json parse'ta fail ederse veya malformed ise)
    broken_raw = '<tool_call>\n{name: get_weather, "arguments": {city\n</tool_call>'
    assert is_valid_json(broken_raw) is False

    # Düz metin (tool call yok)
    plain_text = "I cannot fulfill this request without location info."
    assert is_valid_json(plain_text) is True


def test_tool_selection_correct():
    # Doğru seçim
    ex_correct = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "get_weather", "arguments": {"city": "Izmir"}}\n</tool_call>',
        expected_tool="get_weather",
        expected_arguments={"city": "Izmir"},
    )
    assert tool_selection_correct(ex_correct) is True

    # Yanlış tool
    ex_wrong = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "send_email", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
        expected_arguments={"city": "Izmir"},
    )
    assert tool_selection_correct(ex_wrong) is False

    # Tool beklenirken düz metin üretilmişse
    ex_missed = ToolCallExample(
        predicted_raw="I will check the weather for you.",
        expected_tool="get_weather",
        expected_arguments={"city": "Izmir"},
    )
    assert tool_selection_correct(ex_missed) is False

    # Negatif örnek: tool beklenmiyor ve model düz metin dönmüş (DOĞRU)
    ex_neg_correct = ToolCallExample(
        predicted_raw="Paris is the capital of France.",
        expected_tool=None,
        expected_arguments=None,
    )
    assert tool_selection_correct(ex_neg_correct) is True

    # Negatif örnek: tool beklenmiyor ama model tool çağırmış (YANLIŞ)
    ex_neg_wrong = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "search_db", "arguments": {"q": "Paris"}}\n</tool_call>',
        expected_tool=None,
        expected_arguments=None,
    )
    assert tool_selection_correct(ex_neg_wrong) is False


def test_argument_accuracy():
    # Tam eşleşme (1.0)
    ex_full = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "book_flight", "arguments": {"from": "IST", "to": "LHR", "passengers": 2}}\n</tool_call>',
        expected_tool="book_flight",
        expected_arguments={"from": "IST", "to": "LHR", "passengers": 2},
    )
    assert argument_accuracy(ex_full) == 1.0

    # Kısmi eşleşme (1/2 = 0.5)
    ex_partial = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "book_flight", "arguments": {"from": "IST", "to": "JFK"}}\n</tool_call>',
        expected_tool="book_flight",
        expected_arguments={"from": "IST", "to": "LHR"},
    )
    assert argument_accuracy(ex_partial) == 0.5

    # Sayısal tip esnekliği (str '2' vs int 2)
    ex_flexible = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {"count": "5"}}\n</tool_call>',
        expected_tool="calc",
        expected_arguments={"count": 5},
    )
    assert argument_accuracy(ex_flexible) == 1.0

    # Negatif örnek (tool yok -> 1.0)
    ex_neg = ToolCallExample(
        predicted_raw="Direct answer.",
        expected_tool=None,
        expected_arguments=None,
    )
    assert argument_accuracy(ex_neg) == 1.0


def test_is_unnecessary_tool_call():
    ex_unnecessary = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {}}\n</tool_call>',
        expected_tool=None,
        expected_arguments=None,
    )
    assert is_unnecessary_tool_call(ex_unnecessary) is True

    ex_normal_neg = ToolCallExample(
        predicted_raw="Just plain text response.",
        expected_tool=None,
        expected_arguments=None,
    )
    assert is_unnecessary_tool_call(ex_normal_neg) is False

    ex_tool_req = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {}}\n</tool_call>',
        expected_tool="calc",
        expected_arguments={},
    )
    assert is_unnecessary_tool_call(ex_tool_req) is False


def test_is_invalid_tool_call():
    known = {"get_weather", "send_email"}

    ex_known = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_known, known) is False

    ex_unknown = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "hallucinated_tool", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_unknown, known) is True


def test_aggregate_metrics():
    known = {"tool_a", "tool_b"}
    examples = [
        # 1. Başarılı tool çağrısı
        ToolCallExample(
            predicted_raw='<tool_call>\n{"name": "tool_a", "arguments": {"x": 10}}\n</tool_call>',
            expected_tool="tool_a",
            expected_arguments={"x": 10},
        ),
        # 2. Yanlış tool çağrısı (bilinmeyen tool)
        ToolCallExample(
            predicted_raw='<tool_call>\n{"name": "tool_fake", "arguments": {"x": 10}}\n</tool_call>',
            expected_tool="tool_a",
            expected_arguments={"x": 10},
        ),
        # 3. Başarılı negatif örnek
        ToolCallExample(
            predicted_raw="Plain text answer",
            expected_tool=None,
            expected_arguments=None,
        ),
        # 4. Gereksiz tool çağrısı yapılmış negatif örnek
        ToolCallExample(
            predicted_raw='<tool_call>\n{"name": "tool_b", "arguments": {}}\n</tool_call>',
            expected_tool=None,
            expected_arguments=None,
        ),
    ]

    metrics = aggregate_metrics(examples, known)
    assert metrics["total_examples"] == 4
    assert metrics["tool_selection_accuracy"] == 0.5  # 1. ve 3. doğru (2/4)
    assert metrics["invalid_tool_call_rate"] == 0.25  # 1 tanesi (tool_fake) bilinmiyor (1/4)
    assert metrics["unnecessary_tool_call_rate"] == 0.5  # 2 negatif örnekten 1'i gereksiz çağırdı (1/2)
    assert metrics["json_validity_rate"] == 1.0
