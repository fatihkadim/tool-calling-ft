"""scripts/test_model.py - Fine-tune edilmis QLoRA / LoRA modelini test etme araci.

Kullanim:
    # 1. Hazir orneklerle hizli test:
    uv run python scripts/test_model.py

    # 2. Kendi ozel sorunuzla test:
    uv run python scripts/test_model.py --query "Ankara'da hava kac derece?"

    # 3. Interaktif mod (istediginiz kadar soru sorabilirsiniz):
    uv run python scripts/test_model.py --interactive

    # 4. Farkli bir checkpoint / adapter yolu belirtmek icin:
    uv run python scripts/test_model.py --adapter checkpoints/qlora
"""

import argparse
import json
import os
import sys
import time

# Windows terminalinde Unicode karakterlerin (cp1254 hatası) basılması için UTF-8 ayarı
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_calling_ft.data.tool_schema import (
    DEFAULT_TOOLS,
    build_system_prompt,
    parse_tool_calls_from_text,
)

# Test icin ornek tanimli araclar
DEMO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather information for a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. Istanbul, TR"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math_expression",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical formula to evaluate, e.g. '42 * 18 + 250'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search internal company database for records matching query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword or SQL filter"},
                    "limit": {"type": "integer", "description": "Maximum number of rows to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def load_fine_tuned_model(base_model_name: str, adapter_path: str):
    """Base model ve egitilmis LoRA/QLoRA adaptörünü yukler."""
    print(f"\n[1/3] Tokenizer yukleniyor ({base_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Eger adapter bir QLoRA adapter'i ise taban model mutlaka 4-bit NF4 olarak yuklenmelidir!
    bnb_config = None
    if "qlora" in adapter_path.lower() and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[2/3] Taban model 4-bit NF4 kuantizasyon ile yukleniyor (Cihaz: {device})...")
    else:
        print(f"[2/3] Taban model yukleniyor (Cihaz: {device}, Tip: {torch_dtype})...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")

    print(f"[3/3] Fine-tune edilmis adaptör yukleniyor ({adapter_path})...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print(">> Model basariyla hazirlandi!\n")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, user_query: str, system_prompt: str) -> str:
    """ChatML sablonu olusturur ve modelden yanit uretir."""
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id != tokenizer.eos_token_id:
        stop_ids.append(im_end_id)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )
    elapsed = time.perf_counter() - start

    prompt_len = inputs.input_ids.shape[1]
    gen_tokens = outputs[0][prompt_len:]
    raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

    for stop_tag in ("<|im_end|>", "<|endoftext|>"):
        if stop_tag in raw_text:
            raw_text = raw_text.split(stop_tag)[0]
    
    # Eger </tool_call> etiketi kapatildiysa sonrasindaki tekrar eden metinleri kes
    if "</tool_call>" in raw_text:
        raw_text = raw_text.split("</tool_call>")[0] + "</tool_call>"

    return raw_text.strip(), elapsed


def run_single_test(model, tokenizer, device, user_query: str, system_prompt: str):
    """Tek bir soruyu test edip cıktıyı guzelce formatlar."""
    print("=" * 60)
    print(f"SORU: {user_query}")
    print("-" * 60)

    raw_response, elapsed = generate_response(model, tokenizer, device, user_query, system_prompt)
    print(f"HAM CIKTI ({elapsed:.2f} sn):\n{raw_response}")
    print("-" * 60)

    # Tool call parser'imiz ile analiz
    tool_calls = parse_tool_calls_from_text(raw_response)
    if tool_calls:
        print("PARSER ANALIZI: [TOOL CALL TESPIT EDILDI]")
        for idx, tc in enumerate(tool_calls, 1):
            print(f"  {idx}. Arac: {tc.get('name')}")
            print(f"     Parametreler: {json.dumps(tc.get('arguments', {}), ensure_ascii=False, indent=6)}")
    else:
        print("PARSER ANALIZI: [DUZ METIN YANITI] (Hicbir arac cagrilmadi)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune edilmis QLoRA modelini test etme scripti")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B", help="Taban model adi")
    parser.add_argument("--adapter", default="checkpoints/qlora", help="Adapter checkpoints yolu")
    parser.add_argument("--query", type=str, default=None, help="Test edilecek tek bir soru")
    parser.add_argument("--sample", type=int, default=None, help="eval_subset.jsonl icinden test ornegi numarasi (1-100)")
    parser.add_argument("--interactive", action="store_true", help="Interaktif sohbet modu")
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"HATA: '{args.adapter}' yolu bulunamadi!")
        print("Lutfen checkpoints/qlora klasorunun dogru yerde oldugundan emin olun.")
        sys.exit(1)

    model, tokenizer, device = load_fine_tuned_model(args.base_model, args.adapter)

    if args.sample is not None:
        eval_path = "data/processed/eval_subset.jsonl"
        if not os.path.exists(eval_path):
            print(f"HATA: '{eval_path}' bulunamadi!")
            sys.exit(1)
        with open(eval_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        if 1 <= args.sample <= len(lines):
            item = lines[args.sample - 1]
            print(f"\n[EVAL DATASET ORNEK #{args.sample}]")
            print(f"ID: {item.get('id')}")
            print(f"Beklenen Arac: {item.get('expected_tool')}")
            print(f"Beklenen Argumanlar: {json.dumps(item.get('expected_arguments'), ensure_ascii=False)}")
            run_single_test(model, tokenizer, device, item["user"], item["system"])
        else:
            print(f"Gecersiz ornek numarasi! (1 ile {len(lines)} arasinda olmali)")
        return

    system_prompt = build_system_prompt(DEMO_TOOLS)

    if args.query:
        # Tek soru modu
        run_single_test(model, tokenizer, device, args.query, system_prompt)
    elif args.interactive:
        # Interaktif mod
        print("\n=== INTERAKTIF TEST MODU (Cikmak icin 'q' veya 'exit' yazin) ===")
        print("Mevcut Araclar:")
        print("  1. get_current_weather(location, unit)")
        print("  2. calculate_math_expression(expression)")
        print("  3. search_database(query, limit)\n")

        while True:
            try:
                user_input = input("Sorunuzu girin > ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input or user_input.lower() in ("q", "exit", "quit"):
                print("Cikis yapiliyor...")
                break
            run_single_test(model, tokenizer, device, user_input, system_prompt)
    else:
        # Varsayilan hazir test senaryolari
        print("=== STANDART TEST SENARYOLARI CALISTIRILIYOR ===\n")
        preset_queries = [
            # 1. Hava durumu (Pozitif ornek)
            "What is the current weather in Tokyo in Celsius?",
            # 2. Matematik hesabi (Pozitif ornek)
            "Can you calculate 145 * 24 + 350 for me?",
            # 3. Veritabani aramasi (Pozitif ornek)
            "Search the database for customer orders placed in August 2024.",
            # 4. Genel bilgi / Duz metin (Negatif ornek - Tool cagrilmamali!)
            "What is the capital of France and what is it famous for?",
        ]

        for query in preset_queries:
            run_single_test(model, tokenizer, device, query, system_prompt)

        print("Ipucu: Kendi ozel sorunuzla test etmek icin:")
        print("  uv run python scripts/test_model.py --query \"Ankara'da hava nasil?\"")
        print("Veya interaktif mod icin:")
        print("  uv run python scripts/test_model.py --interactive\n")


if __name__ == "__main__":
    main()
