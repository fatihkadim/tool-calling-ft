"""Bu projede kullanılan tool/function tanımlarının (schema) merkezi listesi ve yardımcı fonksiyonlar.

Hem dataset formatlarken hem eval'de 'known_tools' seti ve şablon oluşturucu olarak kullanılır.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Hermes Function Calling prompt şablonu
SYSTEM_PROMPT_TEMPLATE = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools> </tools> XML tags. You may call one or more functions to assist with the user query. "
    "Don't make assumptions about what values to plug into functions.\n"
    "<tools>\n"
    "{tools_json}\n"
    "</tools>\n"
    "For each function call return a json object with function name and arguments within "
    "<tool_call> </tool_call> tags with the following schema:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-dict>}}\n'
    "</tool_call>"
)

# Eval ve testlerde referans olarak kullanılabilecek standart tool şemaları
DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather conditions for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA or London, UK",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use.",
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math_expression",
            "description": "Evaluate a mathematical expression and return the numerical result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression string, e.g. '24 * 7 + 15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message to a specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Email address of the recipient."},
                    "subject": {"type": "string", "description": "Subject line of the email."},
                    "body": {"type": "string", "description": "Content body of the email."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search records from internal database using SQL query or keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword or SQL query."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves the current or historical stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, MSFT)."},
                    "date": {"type": "string", "description": "Optional date in YYYY-MM-DD format."},
                },
                "required": ["ticker"],
            },
        },
    },
]


def parse_tools(tools_data: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool verisini (JSON string veya liste) list[dict] olarak parse eder."""
    if isinstance(tools_data, list):
        return tools_data
    if isinstance(tools_data, str):
        try:
            parsed = json.loads(tools_data, strict=False)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(tools_data)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                return []
    return []


def extract_tool_names(tools_data: str | list[dict[str, Any]]) -> list[str]:
    """Verilen tool şemalarından fonksiyon isimlerini çıkarır."""
    tools = parse_tools(tools_data)
    names: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function", {})
            name = fn.get("name") if isinstance(fn, dict) else t.get("name")
            if name:
                names.append(str(name))
    return names


def build_system_prompt(tools_data: str | list[dict[str, Any]]) -> str:
    """Tool listesini Hermes XML şablonuna yerleştirerek sistem promptu üretir."""
    tools = parse_tools(tools_data)
    tools_json = json.dumps(tools, ensure_ascii=False)
    return SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)


def parse_single_tool_call_payload(
    payload_str: str,
    available_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Tek bir tool call metnini JSON veya Python sözlüğü olarak güvenli biçimde ayrıştırır."""
    s = payload_str.strip()
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\n", "\n")
    s = s.strip()

    obj: Any = None
    # 1. json.loads (strict=False)
    try:
        obj = json.loads(s, strict=False)
    except Exception:
        pass

    # 2. ast.literal_eval
    if obj is None:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            pass

    # 3. unescape ve ast.literal_eval
    if obj is None:
        try:
            s_unescaped = s.replace("\\n", "\n").strip()
            obj = ast.literal_eval(s_unescaped)
        except Exception:
            pass

    if isinstance(obj, dict):
        # 'name' arguments içinde ise yukarı taşı
        if (
            "name" not in obj
            and "arguments" in obj
            and isinstance(obj["arguments"], dict)
            and "name" in obj["arguments"]
        ):
            obj["name"] = obj["arguments"].pop("name")

        # 'name' hala eksik ve tek bir tool tanımlıysa o tool adını ata
        if "name" not in obj and available_tools:
            t = available_tools[0]
            obj["name"] = t.get("function", {}).get("name", t.get("name", ""))

        return obj

    return None


def parse_tool_calls_from_text(
    response_text: str,
    available_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Model çıktısındaki <tool_call>...</tool_call> bloklarını ayrıştırır ve JSON sözlük listesi döndürür.

    Bulunan JSON nesneleri [{'name': '...', 'arguments': {...}}, ...] formatında döner.
    Eğer geçerli tool call yoksa boş liste döner.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, response_text, flags=re.DOTALL)
    tool_calls: list[dict[str, Any]] = []

    for match in matches:
        parsed = parse_single_tool_call_payload(match, available_tools=available_tools)
        if parsed and parsed.get("name"):
            tool_calls.append(parsed)

    # 1. Fallback: Markdown kod blokları (```json ... ``` veya ``` ... ```)
    if not tool_calls:
        code_blocks = re.findall(r"```(?:json)?\s*({.*?})\s*```", response_text, flags=re.DOTALL)
        for block in code_blocks:
            parsed = parse_single_tool_call_payload(block, available_tools=available_tools)
            if parsed and parsed.get("name"):
                tool_calls.append(parsed)

    # 2. Fallback: Metin içinde serbest dolaşan JSON sözlüklerini ({...}) tara
    if not tool_calls:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(response_text):
            brace_idx = response_text.find("{", pos)
            if brace_idx == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(response_text[brace_idx:])
                if isinstance(obj, dict) and ("name" in obj or "arguments" in obj or available_tools):
                    parsed = parse_single_tool_call_payload(
                        response_text[brace_idx : brace_idx + end_idx],
                        available_tools=available_tools,
                    )
                    if parsed and parsed.get("name"):
                        tool_calls.append(parsed)
                    pos = brace_idx + end_idx
                else:
                    pos = brace_idx + 1
            except json.JSONDecodeError:
                pos = brace_idx + 1

    # 3. Fallback: Eğer XML tag'i unutulmuş ve Python dict literal ({'name': ...}) kullanılmışsa
    if not tool_calls:
        trimmed = response_text.strip()
        # Eğer tek parça süslü parantez varsa
        first_brace = trimmed.find("{")
        last_brace = trimmed.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = trimmed[first_brace : last_brace + 1]
            parsed = parse_single_tool_call_payload(candidate, available_tools=available_tools)
            if parsed and parsed.get("name"):
                tool_calls.append(parsed)

    return tool_calls
