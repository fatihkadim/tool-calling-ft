"""Agent demosu için gerçekten çalışan (ama dış dünyaya dokunmayan) sahte tool'lar.

Şemalar data/tool_schema.py'deki DEFAULT_TOOLS ile birebir aynıdır; model bu
şemaları görür, buradaki Python fonksiyonlarını değil. Her fonksiyon deterministik
sahte veri döndürür: hava durumu API'si, e-posta sunucusu veya veritabanı yoktur.
Gerçek bir entegrasyon için yalnızca fonksiyon gövdelerini değiştirmek yeterlidir.
"""

from __future__ import annotations

import ast
import hashlib
import operator
from collections.abc import Callable
from typing import Any

from tool_calling_ft.data.tool_schema import DEFAULT_TOOLS

# Gönderilen e-postalar gerçekten gönderilmez, burada birikir
OUTBOX: list[dict[str, str]] = []

_FAKE_DB = [
    {"id": 1, "name": "Ada Lovelace", "department": "Engineering", "city": "London"},
    {"id": 2, "name": "Alan Turing", "department": "Research", "city": "Manchester"},
    {"id": 3, "name": "Grace Hopper", "department": "Engineering", "city": "New York"},
    {"id": 4, "name": "Cahit Arf", "department": "Research", "city": "Istanbul"},
]


def _stable_number(text: str, low: float, high: float) -> float:
    """Aynı girdi için hep aynı sahte değeri üretir (demo tekrarlanabilir olsun)."""
    digest = int(hashlib.sha256(text.lower().encode()).hexdigest()[:8], 16)
    return round(low + (digest / 0xFFFFFFFF) * (high - low), 1)


def get_current_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    if unit not in ("celsius", "fahrenheit"):
        raise ValueError(f"unit 'celsius' veya 'fahrenheit' olmalı, '{unit}' geldi")
    temp_c = _stable_number(location, -5, 35)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    # Sıcaklıkla tutarlı bir hava durumu seç (ör. 13°C'de kar yağmasın)
    conditions = ["snowy", "cloudy"] if temp_c < 2 else ["sunny", "cloudy", "rainy", "windy"]
    condition = conditions[int(_stable_number(location + "c", 0, len(conditions) - 0.01))]
    return {"location": location, "temperature": temp, "unit": unit, "condition": condition}


_ALLOWED_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval_node(node.right)) > 100:
            raise ValueError("üs çok büyük")
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"desteklenmeyen ifade: {ast.dump(node)[:60]}")


def calculate_math_expression(expression: str) -> dict[str, Any]:
    """eval() kullanmadan, yalnızca aritmetiğe izin veren güvenli hesaplayıcı."""
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _eval_node(tree.body)}


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """E-postayı göndermez; OUTBOX listesine ekler."""
    if "@" not in to:
        raise ValueError(f"geçersiz e-posta adresi: {to}")
    OUTBOX.append({"to": to, "subject": subject, "body": body})
    return {"status": "queued (demo: not actually sent)", "to": to, "outbox_size": len(OUTBOX)}


def search_database(query: str, limit: int = 10) -> dict[str, Any]:
    terms = query.lower().split()
    hits = [
        row for row in _FAKE_DB
        if any(term in " ".join(str(v) for v in row.values()).lower() for term in terms)
    ]
    return {"query": query, "count": len(hits[:limit]), "records": hits[:limit]}


def get_stock_price(ticker: str, date: str | None = None) -> dict[str, Any]:
    ticker = ticker.upper()
    price = _stable_number(ticker + (date or ""), 10, 500)
    return {"ticker": ticker, "date": date or "latest", "price_usd": price, "source": "demo"}


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_current_weather": get_current_weather,
    "calculate_math_expression": calculate_math_expression,
    "send_email": send_email,
    "search_database": search_database,
    "get_stock_price": get_stock_price,
}

# Modele gösterilen şemalar (fonksiyonların kendisi değil)
TOOL_SCHEMAS = DEFAULT_TOOLS

assert set(TOOL_FUNCTIONS) == {t["function"]["name"] for t in TOOL_SCHEMAS}, (
    "TOOL_FUNCTIONS ile DEFAULT_TOOLS şemaları uyuşmuyor"
)
