"""agent/tools.py ve agent/loop.py için pytest testleri."""

import json

import pytest

from tool_calling_ft.agent import tools
from tool_calling_ft.agent.loop import execute_tool_call, run_agent
from tool_calling_ft.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


def test_weather_is_deterministic_and_converts_units():
    a = tools.get_current_weather("Istanbul")
    b = tools.get_current_weather("Istanbul")
    assert a == b
    f = tools.get_current_weather("Istanbul", unit="fahrenheit")
    assert f["temperature"] == pytest.approx(a["temperature"] * 9 / 5 + 32, abs=0.1)
    with pytest.raises(ValueError):
        tools.get_current_weather("Istanbul", unit="kelvin")


def test_calculator_is_safe():
    assert tools.calculate_math_expression("1234 * 56 + 789")["result"] == 1234 * 56 + 789
    assert tools.calculate_math_expression("-(2 ** 3) / 4")["result"] == -2.0
    for bad in ["__import__('os').system('echo hi')", "open('x')", "2 ** 1000"]:
        with pytest.raises(ValueError):
            tools.calculate_math_expression(bad)


def test_send_email_does_not_send(monkeypatch):
    monkeypatch.setattr(tools, "OUTBOX", [])
    result = tools.send_email("a@example.com", "Hi", "Body")
    assert "not actually sent" in result["status"]
    assert tools.OUTBOX == [{"to": "a@example.com", "subject": "Hi", "body": "Body"}]
    with pytest.raises(ValueError):
        tools.send_email("not-an-email", "Hi", "Body")


def test_search_database():
    result = tools.search_database("research")
    assert {r["name"] for r in result["records"]} == {"Alan Turing", "Cahit Arf"}
    assert tools.search_database("research", limit=1)["count"] == 1


def test_functions_match_schemas():
    assert set(TOOL_FUNCTIONS) == {t["function"]["name"] for t in TOOL_SCHEMAS}


def test_execute_tool_call_reports_errors():
    assert execute_tool_call({"name": "nope", "arguments": {}}, TOOL_FUNCTIONS).error.startswith(
        "unknown tool"
    )
    bad_args = execute_tool_call({"name": "get_stock_price", "arguments": {"symbol": "X"}}, TOOL_FUNCTIONS)
    assert "invalid arguments" in bad_args.error
    ok = execute_tool_call({"name": "get_stock_price", "arguments": {"ticker": "aapl"}}, TOOL_FUNCTIONS)
    assert ok.error is None
    assert ok.result["ticker"] == "AAPL"


class ScriptedModel:
    """Sırayla önceden yazılmış cevapları döndüren sahte model; aldığı prompt'ları kaydeder."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def test_run_agent_executes_tool_and_feeds_result_back():
    call = {"name": "calculate_math_expression", "arguments": {"expression": "6 * 7"}}
    model = ScriptedModel([f"<tool_call>\n{json.dumps(call)}\n</tool_call>", "The answer is 42."])

    result = run_agent("What is 6*7?", model, TOOL_FUNCTIONS, TOOL_SCHEMAS)

    assert result.stopped_reason == "answered"
    assert result.final_answer == "The answer is 42."
    assert len(result.steps) == 2
    assert result.steps[0].executions[0].result["result"] == 42

    # İkinci prompt tool sonucunu Hermes formatında içermeli
    second_prompt = model.prompts[1]
    assert "<|im_start|>tool\n<tool_response>" in second_prompt
    assert '"result": 42' in second_prompt
    assert second_prompt.endswith("<|im_start|>assistant\n")


def test_run_agent_answers_directly_without_tools():
    model = ScriptedModel(["Leonardo da Vinci."])
    result = run_agent("Who painted the Mona Lisa?", model, TOOL_FUNCTIONS, TOOL_SCHEMAS)
    assert result.final_answer == "Leonardo da Vinci."
    assert result.steps[0].executions == []


def test_run_agent_ignores_untagged_json_and_stops_at_max_steps():
    # Etiketsiz JSON tool çağrısı sayılmaz (sıkı parser) → doğrudan cevap kabul edilir
    model = ScriptedModel(['{"name": "get_stock_price", "arguments": {"ticker": "X"}}'])
    assert run_agent("q", model, TOOL_FUNCTIONS, TOOL_SCHEMAS).steps[0].executions == []

    # Sürekli tool çağıran model adım sınırında durdurulur
    loop_call = '<tool_call>{"name": "get_stock_price", "arguments": {"ticker": "X"}}</tool_call>'
    model = ScriptedModel([loop_call] * 3)
    result = run_agent("q", model, TOOL_FUNCTIONS, TOOL_SCHEMAS, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert result.final_answer is None
    assert len(result.steps) == 3


def test_run_agent_handles_parallel_calls_and_tool_errors():
    output = (
        '<tool_call>{"name": "get_stock_price", "arguments": {"ticker": "MSFT"}}</tool_call>\n'
        '<tool_call>{"name": "send_email", "arguments": {"to": "bad", "subject": "s", "body": "b"}}</tool_call>'
    )
    model = ScriptedModel([output, "Done."])
    result = run_agent("q", model, TOOL_FUNCTIONS, TOOL_SCHEMAS)
    first, second = result.steps[0].executions
    assert first.error is None
    assert "geçersiz e-posta" in second.error
    assert '"error"' in model.prompts[1]
