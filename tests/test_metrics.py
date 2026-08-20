"""eval/metrics.py için pytest testleri."""

import pytest

from tool_calling_ft.eval.metrics import (
    ToolCallExample,
    aggregate_metrics,
    argument_accuracy,
    extract_raw_tool_call_json,
    is_invalid_tool_call,
    is_unnecessary_tool_call,
    is_valid_json,
    tool_selection_correct,
)


def test_is_valid_json():
    # Geçerli JSON
    valid_raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara"}}\n</tool_call>'
    assert is_valid_json(valid_raw) is True

    # Geçersiz/bozuk JSON (trailing comma — json.loads strict=False bunu kabul edebilir,
    # bu yüzden kesin bozuk bir JSON kullanıyoruz)
    broken_raw = '<tool_call>\n{name: get_weather, "arguments": {city\n</tool_call>'
    assert is_valid_json(broken_raw) is False

    # Düz metin (tool call yok)
    plain_text = "I cannot fulfill this request without location info."
    assert is_valid_json(plain_text) is True

    # Açılış tag'i var ama kapanış yok (malformed)
    malformed_tag = '<tool_call>\n{"name": "get_weather"}'
    assert is_valid_json(malformed_tag) is False

    # Boş string
    assert is_valid_json("") is True

    # Bare JSON (XML tag'siz ama geçerli)
    bare_json = '{"name": "calc", "arguments": {"x": 1}}'
    assert is_valid_json(bare_json) is True


def test_extract_raw_tool_call_json():
    # Standart XML tag'li
    raw = '<tool_call>\n{"name": "func", "arguments": {}}\n</tool_call>'
    blocks = extract_raw_tool_call_json(raw)
    assert len(blocks) == 1
    assert '"func"' in blocks[0]

    # Çoklu tool call
    multi = (
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    assert len(extract_raw_tool_call_json(multi)) == 2

    # Bare JSON (tag'siz)
    bare = '{"name": "calc", "arguments": {"x": 1}}'
    blocks = extract_raw_tool_call_json(bare)
    assert len(blocks) == 1

    # Düz metin (tool call yok)
    assert extract_raw_tool_call_json("Plain text") == []

    # Boş string
    assert extract_raw_tool_call_json("") == []


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

    # Sayısal tip esnekliği (str '5' vs int 5)
    ex_flexible = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {"count": "5"}}\n</tool_call>',
        expected_tool="calc",
        expected_arguments={"count": 5},
    )
    assert argument_accuracy(ex_flexible) == 1.0

    # Negatif örnek: tool yok, düz metin → 1.0
    ex_neg = ToolCallExample(
        predicted_raw="Direct answer.",
        expected_tool=None,
        expected_arguments=None,
    )
    assert argument_accuracy(ex_neg) == 1.0

    # Negatif örnek: tool yok ama model tool çağırmış → 0.0
    ex_neg_called = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>',
        expected_tool=None,
        expected_arguments=None,
    )
    assert argument_accuracy(ex_neg_called) == 0.0

    # Tool doğru ama argümanlar tamamen eksik → 0.0
    ex_no_args = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "book_flight", "arguments": {}}\n</tool_call>',
        expected_tool="book_flight",
        expected_arguments={"from": "IST", "to": "LHR"},
    )
    assert argument_accuracy(ex_no_args) == 0.0

    # Tool doğru, beklenen argüman boş, model da boş → 1.0
    ex_empty_args = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "ping", "arguments": {}}\n</tool_call>',
        expected_tool="ping",
        expected_arguments={},
    )
    assert argument_accuracy(ex_empty_args) == 1.0

    # Tool doğru, beklenen argüman boş ama model fazladan argüman üretmiş → 0.0
    ex_extra_args = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "ping", "arguments": {"extra": "val"}}\n</tool_call>',
        expected_tool="ping",
        expected_arguments={},
    )
    assert argument_accuracy(ex_extra_args) == 0.0

    # Yanlış tool seçilmişse arg accuracy 0.0
    ex_wrong_tool = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "wrong_tool", "arguments": {"from": "IST"}}\n</tool_call>',
        expected_tool="book_flight",
        expected_arguments={"from": "IST"},
    )
    assert argument_accuracy(ex_wrong_tool) == 0.0

    # Model hiç tool çağırmamış ama tool bekleniyorsa → 0.0
    ex_missed = ToolCallExample(
        predicted_raw="I will book the flight for you.",
        expected_tool="book_flight",
        expected_arguments={"from": "IST"},
    )
    assert argument_accuracy(ex_missed) == 0.0


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

    # Tool gerekli ve tool çağrılmış → unnecessary DEĞİL
    ex_tool_req = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "calc", "arguments": {}}\n</tool_call>',
        expected_tool="calc",
        expected_arguments={},
    )
    assert is_unnecessary_tool_call(ex_tool_req) is False


def test_is_invalid_tool_call():
    known = {"get_weather", "send_email"}

    # Bilinen tool → geçerli
    ex_known = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_known, known) is False

    # Bilinmeyen tool → geçersiz
    ex_unknown = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "hallucinated_tool", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_unknown, known) is True

    # Tool çağrılmamış → geçersiz DEĞİL (invalid yok çünkü çağrı yok)
    ex_no_call = ToolCallExample(
        predicted_raw="No tool call here.",
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_no_call, known) is False

    # Boş known_tools → her çağrı geçersiz
    ex_empty_known = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
        expected_tool="get_weather",
    )
    assert is_invalid_tool_call(ex_empty_known, set()) is True


def test_aggregate_metrics():
    known = {"tool_a", "tool_b"}
    examples = [
        # 1. Başarılı tool çağrısı (tool_a, args doğru)
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
        # 3. Başarılı negatif örnek (düz metin, tool çağrılmamış)
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

    # Temel kontroller
    assert metrics["total_examples"] == 4
    assert metrics["tool_required_count"] == 2
    assert metrics["no_tool_required_count"] == 2

    # tool_selection_accuracy: 1. doğru + 3. doğru = 2/4 = 0.5
    assert metrics["tool_selection_accuracy"] == 0.5

    # argument_accuracy: (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
    assert metrics["argument_accuracy"] == 0.5

    # json_validity_rate: hepsi geçerli JSON = 4/4 = 1.0
    assert metrics["json_validity_rate"] == 1.0

    # invalid_tool_call_rate: sadece tool_fake bilinmiyor = 1/4 = 0.25
    assert metrics["invalid_tool_call_rate"] == 0.25

    # unnecessary_tool_call_rate: 2 negatif örnekten 1'i gereksiz çağırdı = 1/2 = 0.5
    assert metrics["unnecessary_tool_call_rate"] == 0.5


def test_aggregate_metrics_empty():
    """Boş örnek listesiyle çağrıldığında sıfır değerler dönmeli."""
    metrics = aggregate_metrics([], set())
    assert metrics["total_examples"] == 0
    assert metrics["tool_selection_accuracy"] == 0.0
    assert metrics["argument_accuracy"] == 0.0
    assert metrics["json_validity_rate"] == 0.0


def test_aggregate_metrics_with_known_tools_override():
    """ToolCallExample'daki known_tools, global known_tools'u override etmeli."""
    global_known = {"tool_a"}
    # known_tools override ile tool_x de bilinen olsun
    ex = ToolCallExample(
        predicted_raw='<tool_call>\n{"name": "tool_x", "arguments": {}}\n</tool_call>',
        expected_tool="tool_x",
        expected_arguments={},
        known_tools={"tool_x"},  # override
    )
    metrics = aggregate_metrics([ex], global_known)
    # tool_x, ex.known_tools'ta var → invalid olmamalı
    assert metrics["invalid_tool_call_rate"] == 0.0
