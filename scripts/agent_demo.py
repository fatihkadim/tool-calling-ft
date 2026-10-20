"""Uçtan uca tool calling demosu: model çağrıyı yazar, bu script çalıştırır.

Akış: kullanıcı sorusu → model <tool_call> üretir → sahte tool çalıştırılır →
sonuç <tool_response> olarak modele verilir → model nihai cevabı yazar.

Kullanım:
    # Model olmadan, döngüyü görmek için kural tabanlı sahte model
    uv run python scripts/agent_demo.py --mock --query "What's the weather in Istanbul?"

    # Full FT checkpoint ile (Colab'den indirilmiş klasör)
    uv run python scripts/agent_demo.py --model checkpoints/full_ft

    # Adapter ile
    uv run python scripts/agent_demo.py --adapter checkpoints/qlora --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import re

from tool_calling_ft.agent.loop import AgentStep, GenerateFn, run_agent
from tool_calling_ft.agent.tools import OUTBOX, TOOL_FUNCTIONS, TOOL_SCHEMAS

EXAMPLE_QUERIES = [
    "What's the weather like in Istanbul right now?",
    "What is 1234 * 56 + 789?",
    "Find people in the Research department.",
    "Who painted the Mona Lisa?",  # tool gerektirmez
]


def build_model_generate(args: argparse.Namespace) -> GenerateFn:
    import torch

    from tool_calling_ft.eval.harness import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    def generate(prompt: str) -> str:
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=stop_ids,
            )
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        return text.split("<|im_end|>")[0].split("<|endoftext|>")[0]

    return generate


def mock_generate(prompt: str) -> str:
    """Eğitilmiş bir modelin yapacağını taklit eden kural tabanlı sahte model.

    Döngünün ve tool'ların GPU olmadan denenmesi içindir; model kalitesi ölçmez.
    """
    last_user = prompt.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0]
    last_turn = prompt.rsplit("<|im_start|>", 2)[-2]

    # Tool sonucu geldiyse onu özetle
    if last_turn.startswith("tool\n"):
        payload = re.search(r"<tool_response>\n(.*?)\n</tool_response>", last_turn, re.DOTALL)
        data = json.loads(payload.group(1)) if payload else {}
        if "error" in data:
            return f"Sorry, the tool failed: {data['error']}"
        return f"Here is what I found: {json.dumps(data.get('content'), ensure_ascii=False)}"

    q = last_user.lower()
    call = None
    if "weather" in q:
        city = re.search(r"in ([A-Z][a-zA-Z]+)", last_user)
        call = {"name": "get_current_weather", "arguments": {"location": city.group(1) if city else "London"}}
    elif expr := re.search(r"[\d\s+\-*/().]{3,}\d", last_user):
        call = {"name": "calculate_math_expression", "arguments": {"expression": expr.group(0).strip()}}
    elif "department" in q or "find" in q:
        word = re.search(r"the (\w+) department", last_user)
        call = {"name": "search_database", "arguments": {"query": word.group(1) if word else last_user}}

    if call:
        return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"
    return "I can answer that directly without a tool: Leonardo da Vinci painted the Mona Lisa."


def print_step(step: AgentStep) -> None:
    print(f"\n  🤖 model:\n     {step.model_output[:500]}")
    for ex in step.executions:
        args = json.dumps(ex.arguments, ensure_ascii=False)
        print(f"  🔧 çalıştırıldı: {ex.name}({args})")
        if ex.error:
            print(f"     ❌ hata: {ex.error}")
        else:
            print(f"     ✅ sonuç: {json.dumps(ex.result, ensure_ascii=False, default=str)}")


def ask(query: str, generate: GenerateFn, max_steps: int) -> None:
    print(f"\n{'=' * 70}\n👤 {query}")
    result = run_agent(query, generate, TOOL_FUNCTIONS, TOOL_SCHEMAS, max_steps, on_step=print_step)
    if result.stopped_reason == "max_steps":
        print(f"\n  ⚠️  {max_steps} adımda nihai cevap üretilmedi.")
    tool_count = sum(len(s.executions) for s in result.steps)
    print(f"\n  ({len(result.steps)} model adımı, {tool_count} tool çalıştırması)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Uçtan uca tool calling agent demosu")
    parser.add_argument("--mock", action="store_true", help="Model yerine kural tabanlı sahte model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model adı veya yolu")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA/QLoRA/DoRA adapter yolu")
    parser.add_argument("--load-in-4bit", action="store_true", help="QLoRA adapter'ı için")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--query", type=str, default=None, help="Tek soru (yoksa örnekler + etkileşim)")
    parser.add_argument("--examples", action="store_true", help="Hazır örnek soruları çalıştır")
    args = parser.parse_args()

    generate = mock_generate if args.mock else build_model_generate(args)

    if args.query:
        ask(args.query, generate, args.max_steps)
    elif args.examples:
        for q in EXAMPLE_QUERIES:
            ask(q, generate, args.max_steps)
    else:
        print("Soru yazın (çıkmak için boş satır). Tool'lar: " + ", ".join(TOOL_FUNCTIONS))
        while query := input("\n👤 > ").strip():
            ask(query, generate, args.max_steps)

    if OUTBOX:
        print(f"\n📭 Demo outbox ({len(OUTBOX)} e-posta, hiçbiri gerçekten gönderilmedi)")


if __name__ == "__main__":
    main()
