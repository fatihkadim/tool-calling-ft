"""Model → tool çağrısı → çalıştırma → sonucu modele geri verme döngüsü.

Model yalnızca metin üretir. Bu modül o metindeki <tool_call> bloklarını sıkı
parser ile okur, ilgili Python fonksiyonunu çalıştırır ve sonucu Hermes formatında
bir `tool` mesajı olarak konuşmaya ekler. Model tool çağırmayan bir cevap ürettiğinde
(ya da adım sınırı dolduğunda) döngü biter.

Döngü modelden bağımsızdır: `generate` herhangi bir `prompt -> metin` fonksiyonu
olabilir (gerçek model, test için sahte model, ileride bir API).

Not: Bu projedeki modeller yalnızca tek turlu veriyle eğitildi (soru → tool çağrısı).
`tool` mesajından sonra nihai cevabı yazmayı hiç görmediler; bu yüzden ikinci
adımdaki cevap kalitesi sınırlı olabilir. Döngü bu açığı ölçmek için de kullanılabilir.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tool_calling_ft.data.tool_schema import (
    build_system_prompt,
    parse_tool_calls_strict,
)

GenerateFn = Callable[[str], str]


@dataclass
class ToolExecution:
    """Tek bir tool çağrısının kaydı."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class AgentStep:
    """Modelin bir üretimi ve o üretimde çalıştırılan tool'lar."""

    model_output: str
    executions: list[ToolExecution] = field(default_factory=list)


@dataclass
class AgentResult:
    final_answer: str | None
    steps: list[AgentStep]
    stopped_reason: str  # "answered" | "max_steps"
    prompt: str


def _format_tool_response(execution: ToolExecution) -> str:
    payload: dict[str, Any] = {"name": execution.name}
    if execution.error is not None:
        payload["error"] = execution.error
    else:
        payload["content"] = execution.result
    return f"<tool_response>\n{json.dumps(payload, ensure_ascii=False, default=str)}\n</tool_response>"


def execute_tool_call(
    call: dict[str, Any], tool_functions: dict[str, Callable[..., Any]]
) -> ToolExecution:
    """Bir tool çağrısını çalıştırır. Hatalar yakalanıp modele geri bildirilir."""
    name = call["name"]
    arguments = call.get("arguments", {})
    execution = ToolExecution(name=name, arguments=arguments)

    func = tool_functions.get(name)
    if func is None:
        execution.error = f"unknown tool '{name}'. Available: {sorted(tool_functions)}"
        return execution
    try:
        execution.result = func(**arguments)
    except TypeError as e:  # yanlış/eksik argüman adları
        execution.error = f"invalid arguments for {name}: {e}"
    except Exception as e:  # noqa: BLE001 — tool hatası modeli düşürmemeli
        execution.error = f"{type(e).__name__}: {e}"
    return execution


def run_agent(
    user_message: str,
    generate: GenerateFn,
    tool_functions: dict[str, Callable[..., Any]],
    tool_schemas: list[dict[str, Any]],
    max_steps: int = 4,
    on_step: Callable[[AgentStep], None] | None = None,
) -> AgentResult:
    """Kullanıcı mesajı için agent döngüsünü çalıştırır.

    Args:
        user_message: Kullanıcının isteği.
        generate: ChatML prompt'u alıp asistanın devamını (stop token'a kadar) döndüren fonksiyon.
        tool_functions: tool adı → çalıştırılacak Python fonksiyonu.
        tool_schemas: Modele system prompt'ta gösterilecek şemalar.
        max_steps: En fazla kaç model üretimi yapılacağı (sonsuz döngüye karşı).
        on_step: Her adımdan sonra çağrılır (canlı çıktı için).
    """
    prompt = (
        f"<|im_start|>system\n{build_system_prompt(tool_schemas)}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
    )
    steps: list[AgentStep] = []

    for _ in range(max_steps):
        prompt += "<|im_start|>assistant\n"
        output = generate(prompt).strip()
        prompt += f"{output}<|im_end|>\n"

        step = AgentStep(model_output=output)
        calls = parse_tool_calls_strict(output)
        if not calls:
            steps.append(step)
            if on_step:
                on_step(step)
            return AgentResult(output, steps, "answered", prompt)

        responses = []
        for call in calls:
            execution = execute_tool_call(call, tool_functions)
            step.executions.append(execution)
            responses.append(_format_tool_response(execution))
        prompt += "<|im_start|>tool\n" + "\n".join(responses) + "<|im_end|>\n"

        steps.append(step)
        if on_step:
            on_step(step)

    return AgentResult(None, steps, "max_steps", prompt)
