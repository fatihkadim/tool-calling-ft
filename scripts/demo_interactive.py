"""
🧪 Interaktif Model Test Scripti
Fine-tune edilmiş modelleri (QLoRA, DoRA) interaktif olarak test et.

Kullanım:
    uv run python scripts/demo_interactive.py --adapter qlora
    uv run python scripts/demo_interactive.py --adapter dora
    uv run python scripts/demo_interactive.py --adapter none   (baseline)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent

# tool_schema.py'den import
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tool_calling_ft.data.tool_schema import (
    DEFAULT_TOOLS,
    build_system_prompt,
    parse_tool_calls_from_text,
)


# ─────────────────────────────────────────────────
# Renk kodları (terminal çıktısı için)
# ─────────────────────────────────────────────────
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════╗
║  🧪 Tool Calling Fine-Tuning — Interactive Demo          ║
║  Model: Qwen2.5-0.5B | Hermes Format                    ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def print_tools():
    """Mevcut toolları listele."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}📦 Mevcut Tool'lar:{Colors.RESET}")
    for i, tool in enumerate(DEFAULT_TOOLS, 1):
        fn = tool["function"]
        params = fn["parameters"]["properties"]
        param_list = ", ".join(
            f"{k}: {v.get('type', '?')}" for k, v in params.items()
        )
        print(f"  {Colors.GREEN}{i}. {fn['name']}{Colors.RESET}({param_list})")
        print(f"     {Colors.DIM}{fn['description']}{Colors.RESET}")
    print()


def load_model(adapter_name: str):
    """Model ve tokenizer'ı yükle."""
    base_model_name = "Qwen/Qwen2.5-0.5B"
    
    # Adapter yolunu belirle
    adapter_map = {
        "qlora": PROJECT_ROOT / "checkpoints" / "qlora",
        "dora": PROJECT_ROOT / "checkpoints" / "dora",
        "none": None,
    }
    
    adapter_path = adapter_map.get(adapter_name)
    
    if adapter_name != "none" and adapter_path and not adapter_path.exists():
        print(f"{Colors.RED}❌ Adapter bulunamadı: {adapter_path}{Colors.RESET}")
        sys.exit(1)
    
    # GPU memory bilgisi
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**2
        total_mem = torch.cuda.mem_get_info()[1] / 1024**2
        print(f"{Colors.DIM}GPU: {torch.cuda.get_device_name(0)} | "
              f"Free: {free_mem:.0f} MB / {total_mem:.0f} MB{Colors.RESET}")
    
    print(f"\n{Colors.BLUE}⏳ Model yükleniyor: {base_model_name}{Colors.RESET}")
    if adapter_path:
        print(f"{Colors.BLUE}   Adapter: {adapter_name} ({adapter_path}){Colors.RESET}")
    else:
        print(f"{Colors.BLUE}   Adapter: Yok (baseline/zero-shot){Colors.RESET}")
    
    start = time.time()
    
    # Model yükleme
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Adapter yükleme
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # Daha hızlı inference için merge
    
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    elapsed = time.time() - start
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**2
        print(f"{Colors.GREEN}✅ Model yüklendi! ({elapsed:.1f}s, VRAM: {vram_used:.0f} MB){Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✅ Model yüklendi! ({elapsed:.1f}s, CPU){Colors.RESET}")
    
    return model, tokenizer, device


def build_prompt(user_message: str, tools: list | None = None) -> str:
    """Hermes format ChatML prompt oluştur."""
    if tools is None:
        tools = DEFAULT_TOOLS
    
    system_prompt = build_system_prompt(tools)
    
    # ChatML formatı
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 256):
    """Model ile yanıt üret."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # Deterministik
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    elapsed = time.time() - start
    
    # Sadece yeni üretilen tokenları decode et
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    n_tokens = len(generated_ids)
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
    
    return response, n_tokens, elapsed, tok_per_sec


def analyze_response(response: str):
    """Yanıtı analiz et: tool call var mı, JSON geçerli mi?"""
    tool_calls = parse_tool_calls_from_text(response, available_tools=DEFAULT_TOOLS)
    
    analysis = {
        "has_tool_call": len(tool_calls) > 0,
        "tool_calls": tool_calls,
        "num_tool_calls": len(tool_calls),
    }
    
    return analysis


def print_response(response: str, analysis: dict, n_tokens: int, elapsed: float, tok_per_sec: float):
    """Yanıtı güzel formatta yazdır."""
    # Ham çıktı
    print(f"\n{Colors.BOLD}📝 Ham Çıktı:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    
    # Çıktıyı max 500 karakterle sınırla (okunabilirlik için)
    display = response[:500]
    if len(response) > 500:
        display += f"\n{Colors.DIM}... (+{len(response) - 500} karakter daha){Colors.RESET}"
    print(display)
    
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    
    # Tool call analizi
    if analysis["has_tool_call"]:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🔧 Tool Call Tespit Edildi!{Colors.RESET}")
        for i, tc in enumerate(analysis["tool_calls"], 1):
            print(f"  {Colors.GREEN}#{i} {tc.get('name', '?')}{Colors.RESET}")
            args = tc.get("arguments", {})
            if args:
                print(f"     {Colors.CYAN}{json.dumps(args, ensure_ascii=False, indent=2)}{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}💬 Tool call tespit edilemedi (doğal dilde yanıt veya hata){Colors.RESET}")
    
    # Performans
    print(f"\n{Colors.DIM}⚡ {n_tokens} token | {elapsed:.2f}s | {tok_per_sec:.1f} tok/s{Colors.RESET}")


def run_preset_tests(model, tokenizer, device):
    """Hazır test senaryolarını çalıştır."""
    presets = [
        {
            "label": "☀️ Hava Durumu (Tool gerekli)",
            "query": "What is the weather in Tokyo in celsius?",
            "expected": "get_current_weather(location='Tokyo', unit='celsius')",
        },
        {
            "label": "📧 Email Gönder (Tool gerekli)",
            "query": "Send an email to john@example.com with subject 'Meeting' and body 'See you at 3pm'",
            "expected": "send_email(to='john@example.com', subject='Meeting', body='See you at 3pm')",
        },
        {
            "label": "🧮 Matematik (Tool gerekli)",
            "query": "Calculate 15 * 7 + 23",
            "expected": "calculate_math_expression(expression='15 * 7 + 23')",
        },
        {
            "label": "📊 Hisse Fiyatı (Tool gerekli)",
            "query": "What is the stock price of AAPL?",
            "expected": "get_stock_price(ticker='AAPL')",
        },
        {
            "label": "💬 Genel Sohbet (Tool GEREKMEMELİ)",
            "query": "What is the capital of France and what is it famous for?",
            "expected": "Tool çağırmamalı — doğal dilde yanıt vermeli",
        },
    ]
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}🧪 Hazır Test Senaryoları ({len(presets)} adet){Colors.RESET}")
    print(f"{Colors.DIM}{'═' * 60}{Colors.RESET}")
    
    for i, preset in enumerate(presets, 1):
        print(f"\n{Colors.BOLD}{preset['label']}{Colors.RESET}")
        print(f"{Colors.DIM}Soru: {preset['query']}{Colors.RESET}")
        print(f"{Colors.DIM}Beklenen: {preset['expected']}{Colors.RESET}")
        
        prompt = build_prompt(preset["query"])
        response, n_tokens, elapsed, tok_per_sec = generate_response(
            model, tokenizer, prompt, device, max_new_tokens=256
        )
        analysis = analyze_response(response)
        print_response(response, analysis, n_tokens, elapsed, tok_per_sec)
        
        if i < len(presets):
            print(f"\n{Colors.DIM}{'═' * 60}{Colors.RESET}")


def interactive_mode(model, tokenizer, device):
    """Interaktif sohbet modu."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}💬 İnteraktif Mod — Soru sorun, 'quit' ile çıkın{Colors.RESET}")
    print(f"{Colors.DIM}Komutlar: 'quit' | 'tools' | 'preset' | 'clear'{Colors.RESET}\n")
    
    while True:
        try:
            user_input = input(f"{Colors.BOLD}🧑 Sen > {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.DIM}👋 Çıkış yapılıyor...{Colors.RESET}")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{Colors.DIM}👋 Hoşça kal!{Colors.RESET}")
            break
        
        if user_input.lower() == "tools":
            print_tools()
            continue
        
        if user_input.lower() == "preset":
            run_preset_tests(model, tokenizer, device)
            continue
        
        if user_input.lower() == "clear":
            import os
            os.system("cls" if sys.platform == "win32" else "clear")
            print_banner()
            continue
        
        # Normal soru — model ile yanıt üret
        prompt = build_prompt(user_input)
        response, n_tokens, elapsed, tok_per_sec = generate_response(
            model, tokenizer, prompt, device
        )
        analysis = analyze_response(response)
        print_response(response, analysis, n_tokens, elapsed, tok_per_sec)
        print()


def main():
    parser = argparse.ArgumentParser(description="🧪 Tool Calling Model Test")
    parser.add_argument(
        "--adapter", "-a",
        choices=["qlora", "dora", "none"],
        default="qlora",
        help="Kullanılacak adapter: qlora, dora, veya none (baseline). Varsayılan: qlora",
    )
    parser.add_argument(
        "--preset", "-p",
        action="store_true",
        help="Hazır test senaryolarını çalıştır ve çık",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Tek bir soru sor ve çık (ör: --query 'What is the weather in Tokyo?')",
    )
    args = parser.parse_args()
    
    print_banner()
    
    adapter_label = {
        "qlora": "🥇 QLoRA (4-bit NF4)",
        "dora": "🥈 DoRA (use_dora=true)",
        "none": "📊 Baseline (zero-shot)",
    }
    print(f"{Colors.BOLD}Seçili Adapter: {adapter_label[args.adapter]}{Colors.RESET}")
    
    # Model yükle
    model, tokenizer, device = load_model(args.adapter)
    
    # Tool listesi göster
    print_tools()
    
    if args.query:
        # Tek soru modu
        prompt = build_prompt(args.query)
        response, n_tokens, elapsed, tok_per_sec = generate_response(
            model, tokenizer, prompt, device
        )
        analysis = analyze_response(response)
        print(f"\n{Colors.BOLD}🧑 Soru:{Colors.RESET} {args.query}")
        print_response(response, analysis, n_tokens, elapsed, tok_per_sec)
    elif args.preset:
        # Hazır test modu
        run_preset_tests(model, tokenizer, device)
    else:
        # İnteraktif mod
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
