"""Tool-calling eval metrikleri.

Bu modul, bir modelin urettigi tool-call ciktisini ground-truth ile karsilastirip
tek tek metrikleri hesaplar. training/ modulunden bagimsiz calisir - baseline,
LoRA, QLoRA, DoRA ve Full FT ciktilarinin hepsi ayni fonksiyonlardan gecer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ToolCallExample:
    """Tek bir eval ornegi: model ciktisi ve beklenen (ground-truth) tool call."""

    predicted_raw: str  # modelin ham string ciktisi
    expected_tool: str | None  # None ise "tool gerekmiyor" durumu
    expected_arguments: dict | None


def is_valid_json(raw: str) -> bool:
    """TODO: raw string'in gecerli JSON olup olmadigini kontrol et."""
    raise NotImplementedError


def tool_selection_correct(example: ToolCallExample) -> bool:
    """TODO: dogru tool secilmis mi kontrol et (JSON parse + 'tool' alani karsilastirmasi)."""
    raise NotImplementedError


def argument_accuracy(example: ToolCallExample) -> float:
    """TODO: field-level exact match orani dondur (0.0 - 1.0)."""
    raise NotImplementedError


def is_unnecessary_tool_call(example: ToolCallExample) -> bool:
    """TODO: expected_tool None iken model yine de tool cagirmis mi."""
    raise NotImplementedError


def is_invalid_tool_call(example: ToolCallExample, known_tools: set[str]) -> bool:
    """TODO: model var olmayan bir tool cagirmis mi."""
    raise NotImplementedError


def aggregate_metrics(examples: list[ToolCallExample], known_tools: set[str]) -> dict:
    """TODO: butun ornekler uzerinde yukaridaki metrikleri toplayip ozet dict dondur.

    Donmesi beklenen anahtarlar:
      tool_selection_accuracy, argument_accuracy, json_validity_rate,
      invalid_tool_call_rate, unnecessary_tool_call_rate
    """
    raise NotImplementedError
