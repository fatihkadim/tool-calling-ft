"""Tool-calling eval metrikleri.

Bu modül, bir modelin ürettiği tool-call çıktısını ground-truth ile karşılaştırıp
tek tek metrikleri hesaplar. training/ modülünden bağımsız çalışır — baseline,
LoRA, QLoRA, DoRA ve Full FT çıktılarının hepsi aynı fonksiyonlardan geçer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from tool_calling_ft.data.tool_schema import parse_tool_calls_from_text


@dataclass
class ToolCallExample:
    """Tek bir eval örneği: model çıktısı ve beklenen (ground-truth) tool call."""

    predicted_raw: str  # modelin ham string çıktısı
    expected_tool: str | None  # None ise "tool gerekmiyor" durumu
    expected_arguments: dict[str, Any] | None = None
    known_tools: set[str] | None = None


def extract_raw_tool_call_json(raw: str) -> list[str]:
    """Model çıktısındaki <tool_call> bloklarının içindeki ham metinleri çıkarır."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, raw, flags=re.DOTALL)
    if matches:
        return [m.strip() for m in matches]

    # Eğer XML tag'i yoksa ama süslü parantezle başlıyorsa
    trimmed = raw.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return [trimmed]
    return []


def is_valid_json(raw: str) -> bool:
    """Modelin ürettiği tool call JSON'ının sözdizimsel olarak geçerli olup olmadığını kontrol eder.

    Eğer model çıktı üretmemişse veya tool gerekmeyen bir durumda düz metin dönmüşse ve
    içinde malformed tool_call tag'i yoksa geçerli kabul edilir.
    """
    json_blocks = extract_raw_tool_call_json(raw)
    if not json_blocks:
        # Tool call etiketi yoksa ve kullanıcıya düz metin yanıt verilmişse
        if "<tool_call>" in raw:
            # Açılış etiketi var ama kapanış yok veya malformed
            return False
        return True

    for block in json_blocks:
        s = block.strip()
        if "\\n" in s and "\n" not in s:
            s = s.replace("\\n", "\n")
        try:
            json.loads(s, strict=False)
        except Exception:
            return False
    return True


def tool_selection_correct(example: ToolCallExample) -> bool:
    """Doğru tool seçilmiş mi kontrol eder.

    - expected_tool is None: Model tool çağırmamışsa (düz metin) DOĞRU (True).
    - expected_tool is str: Modelin ürettiği ilk tool adı expected_tool ile eşleşiyorsa DOĞRU (True).
    """
    calls = parse_tool_calls_from_text(example.predicted_raw)

    if example.expected_tool is None:
        return len(calls) == 0

    if not calls:
        return False

    predicted_tool = calls[0].get("name")
    return predicted_tool == example.expected_tool


def _values_match(pred_val: Any, exp_val: Any) -> bool:
    """İki argüman değerini esnek ve tipten bağımsız (string/int/float) kontrol eder."""
    if pred_val == exp_val:
        return True
    if str(pred_val).strip() == str(exp_val).strip():
        return True
    # Sayısal karşılaştırma (örn: 10 vs 10.0)
    try:
        if float(pred_val) == float(exp_val):
            return True
    except (ValueError, TypeError):
        pass
    return False


def argument_accuracy(example: ToolCallExample) -> float:
    """Field-level exact match oranı döndürür (0.0 - 1.0).

    - Tool seçimi yanlışsa veya model tool çağırmamışsa (ama bekleniyorsa) 0.0 döner.
    - Negatif örneklerde (expected_tool=None) model tool çağırmamışsa 1.0 döner.
    - Beklenen argümanlar boşsa ({}) ve model boş argüman vermişse 1.0 döner.
    """
    calls = parse_tool_calls_from_text(example.predicted_raw)

    if example.expected_tool is None:
        return 1.0 if len(calls) == 0 else 0.0

    if not calls:
        return 0.0

    pred_tool = calls[0].get("name")
    if pred_tool != example.expected_tool:
        return 0.0

    pred_args = calls[0].get("arguments")
    if not isinstance(pred_args, dict):
        pred_args = {}

    expected_args = example.expected_arguments or {}
    if not expected_args:
        return 1.0 if not pred_args else 1.0

    total_fields = len(expected_args)
    matched_fields = 0

    for key, exp_val in expected_args.items():
        if key in pred_args and _values_match(pred_args[key], exp_val):
            matched_fields += 1

    return matched_fields / total_fields


def is_unnecessary_tool_call(example: ToolCallExample) -> bool:
    """expected_tool None iken modelin yine de bir tool çağırıp çağırmadığını kontrol eder."""
    if example.expected_tool is not None:
        return False

    calls = parse_tool_calls_from_text(example.predicted_raw)
    return len(calls) > 0


def is_invalid_tool_call(example: ToolCallExample, known_tools: set[str]) -> bool:
    """Modelin bilinmeyen (tanımlanmamış / uydurma) bir tool çağırıp çağırmadığını kontrol eder."""
    calls = parse_tool_calls_from_text(example.predicted_raw)
    if not calls:
        return False

    for call in calls:
        tool_name = call.get("name")
        if not tool_name or tool_name not in known_tools:
            return True
    return False


def aggregate_metrics(examples: list[ToolCallExample], known_tools: set[str]) -> dict[str, Any]:
    """Tüm örnekler üzerinde metrikleri toplayıp özet istatistik sözlüğü döndürür.

    Dönen anahtarlar:
      - total_examples
      - tool_selection_accuracy
      - argument_accuracy
      - json_validity_rate
      - invalid_tool_call_rate
      - unnecessary_tool_call_rate
      - tool_required_count
      - no_tool_required_count
    """
    if not examples:
        return {
            "total_examples": 0,
            "tool_selection_accuracy": 0.0,
            "argument_accuracy": 0.0,
            "json_validity_rate": 0.0,
            "invalid_tool_call_rate": 0.0,
            "unnecessary_tool_call_rate": 0.0,
        }

    total = len(examples)
    tool_sel_correct = 0
    arg_acc_sum = 0.0
    valid_json_count = 0
    invalid_tool_count = 0
    unnecessary_tool_count = 0

    tool_req_total = 0
    no_tool_total = 0

    for ex in examples:
        tools_set = ex.known_tools if ex.known_tools is not None else known_tools

        # 1. JSON Validity
        if is_valid_json(ex.predicted_raw):
            valid_json_count += 1

        # 2. Tool Selection
        if tool_selection_correct(ex):
            tool_sel_correct += 1

        # 3. Argument Accuracy
        arg_acc_sum += argument_accuracy(ex)

        # 4. Invalid Tool Call (bilinmeyen fonksiyon çağırma)
        if is_invalid_tool_call(ex, tools_set):
            invalid_tool_count += 1

        # 5. Unnecessary Tool Call (tool gerekmiyorken çağırma)
        if ex.expected_tool is None:
            no_tool_total += 1
            if is_unnecessary_tool_call(ex):
                unnecessary_tool_count += 1
        else:
            tool_req_total += 1

    return {
        "total_examples": total,
        "tool_selection_accuracy": round(tool_sel_correct / total, 4),
        "argument_accuracy": round(arg_acc_sum / total, 4),
        "json_validity_rate": round(valid_json_count / total, 4),
        "invalid_tool_call_rate": round(invalid_tool_count / total, 4),
        "unnecessary_tool_call_rate": (
            round(unnecessary_tool_count / no_tool_total, 4) if no_tool_total > 0 else 0.0
        ),
        "tool_required_count": tool_req_total,
        "no_tool_required_count": no_tool_total,
    }
