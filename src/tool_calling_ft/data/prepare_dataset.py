"""Hermes-function-calling-v1 veri setini indirme, formatlama ve train/val split hazırlama modülü.

İş akışı:
1. Ham veri HuggingFace'ten indirilir ve `data/raw/hermes_raw.jsonl` altına kaydedilir.
2. Ham verideki single-turn konuşmalar ayrıştırılır (system, user, assistant).
3. "Tool gerekmeyen" negatif örnekler (negative examples) eklenir.
4. ChatML şablonuna (`<|im_start|>...<|im_end|>`) dönüştürülür.
5. Train (%85) ve Val (%15) olarak `data/processed/` altına kaydedilir.
6. Hızlı baseline eval için 100 örneklik `eval_subset.jsonl` oluşturulur.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from tool_calling_ft.data.tool_schema import (
    DEFAULT_TOOLS,
    build_system_prompt,
    extract_tool_names,
    parse_tool_calls_from_text,
    parse_tools,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "NousResearch/hermes-function-calling-v1"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def download_raw_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: Path | str = DEFAULT_RAW_DIR,
    output_filename: str = "hermes_raw.jsonl",
) -> Path:
    """HuggingFace'ten ham veri setini indirir ve data/raw altına JSONL formatında kaydeder."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_file = output_path / output_filename
    info_file = output_path / "dataset_info.json"

    logger.info("'%s' veri seti HuggingFace'ten indiriliyor...", dataset_name)
    dataset = load_dataset(dataset_name, split="train")
    total_samples = len(dataset)
    logger.info("İndirme tamamlandı. Toplam örnek sayısı: %d", total_samples)

    logger.info("Ham veri '%s' dosyasına yazılıyor...", target_file)
    with open(target_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    info = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "features": dataset.column_names,
        "target_file": str(target_file),
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    file_size_mb = target_file.stat().st_size / (1024 * 1024)
    logger.info(
        "Ham veri başarıyla kaydedildi: %s (%.2f MB, %d satır)",
        target_file,
        file_size_mb,
        total_samples,
    )
    return target_file


def format_chatml(system_prompt: str, user_prompt: str, assistant_response: str) -> str:
    """ChatML standardında metin üretir (Qwen2.5 ve Hermes uyumlu)."""
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response.strip()}<|im_end|>"
    )


def parse_raw_item(raw_item: dict[str, Any]) -> dict[str, Any] | None:
    """Ham bir veri satırını standardize edilmiş eğitim ve eval formatına dönüştürür."""
    conversations = raw_item.get("conversations", [])
    if len(conversations) < 3:
        return None

    system_text = ""
    user_text = ""
    assistant_text = ""

    for turn in conversations:
        role = turn.get("from")
        val = turn.get("value", "")
        if role == "system":
            system_text = val
        elif role in ("human", "user"):
            user_text = val
        elif role in ("gpt", "assistant"):
            assistant_text = val

    if not system_text or not user_text or not assistant_text:
        return None

    tools_raw = raw_item.get("tools", "[]")
    tools_list = parse_tools(tools_raw)

    parsed_calls = parse_tool_calls_from_text(assistant_text, available_tools=tools_list)
    is_tool_call = len(parsed_calls) > 0

    expected_tool = parsed_calls[0].get("name") if is_tool_call else None
    expected_arguments = parsed_calls[0].get("arguments") if is_tool_call else None

    chatml_text = format_chatml(system_text, user_text, assistant_text)

    return {
        "id": raw_item.get("id"),
        "system": system_text,
        "user": user_text,
        "assistant": assistant_text,
        "tools": tools_list,
        "is_tool_call": is_tool_call,
        "expected_tool": expected_tool,
        "expected_arguments": expected_arguments,
        "all_expected_tool_calls": parsed_calls,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "text": chatml_text,
    }


# Çeşitli negatif örnek şablonları (tool şeması verildiği halde tool çağrısı yapılmaması gereken durumlar)
NEGATIVE_TEMPLATES = [
    (
        "Hello! How are you today? Can you give me a quick summary of what you can help me with?",
        "Hello! I am doing well, thank you. I am an AI assistant equipped with various tools to assist you with tasks such as data analysis, information retrieval, math calculations, and more. Feel free to let me know what you need help with!",
    ),
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
    ),
    (
        "Can you explain how photosynthesis works in plants?",
        "Photosynthesis is the biochemical process by which green plants, algae, and some bacteria convert light energy (typically from sunlight) into chemical energy in the form of glucose. During this process, carbon dioxide and water are absorbed and converted into oxygen and carbohydrates using chlorophyll.",
    ),
    (
        "What is the difference between TCP and UDP protocols?",
        "TCP (Transmission Control Protocol) is connection-oriented, reliable, and guarantees the ordered delivery of packets using handshakes and error checking. UDP (User Datagram Protocol) is connectionless, faster, and does not guarantee packet delivery or order, making it ideal for streaming and real-time gaming.",
    ),
    (
        "Can you write a short poem about stars in the night sky?",
        "Silent lanterns in the deep,\nWatching over worlds asleep.\nSilver sparks of ancient light,\nGuiding travelers through the night.",
    ),
    (
        "Can you calculate what 45 multiplied by 12 is?",
        "45 multiplied by 12 is 540.",
    ),
    (
        "What is the boiling point of water at sea level?",
        "The boiling point of water at sea level is 100°C (212°F).",
    ),
    (
        "Explain what a binary search tree is in computer science.",
        "A binary search tree (BST) is a node-based data structure where each node has at most two children. The left subtree contains only nodes with keys less than the parent's key, and the right subtree contains only nodes with keys greater than the parent's key, enabling efficient searching, insertion, and deletion.",
    ),
    (
        "Can you help me send an email?",
        "Certainly! Please provide the recipient's email address, the subject line, and the message content you would like to include.",
    ),
    (
        "What are some key health benefits of drinking sufficient water daily?",
        "Drinking sufficient water helps maintain bodily fluid balance, supports energy levels and brain function, aids digestion, lubricates joints, and helps flush out metabolic waste products.",
    ),
    (
        "Who wrote the play Romeo and Juliet?",
        "Romeo and Juliet was written by William Shakespeare in the late 16th century.",
    ),
    (
        "What does HTTP 404 status code mean?",
        "The HTTP 404 Not Found error status code indicates that the server cannot find the requested resource or webpage.",
    ),
    (
        "Can you provide a simple Python function to check if a string is a palindrome?",
        "Here is a simple Python function:\n\n```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```",
    ),
    (
        "I need to book a hotel room, but I haven't decided on the city yet.",
        "Take your time! Once you decide on the city, check-in and check-out dates, and the number of guests, let me know and I can assist you with the booking.",
    ),
    (
        "Thank you so much for your assistance, that was very helpful!",
        "You're very welcome! If you have any other questions or need further assistance, feel free to ask anytime.",
    ),
]


def build_negative_examples(
    tools_pool: list[list[dict[str, Any]]],
    count: int = 250,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sistem promptunda tool'lar tanımlı olduğu halde kullanıcının tool gerektirmeyen sorularına

    dair negatif örnekler üretir (expected_tool = None).
    """
    rng = random.Random(seed)
    negative_examples: list[dict[str, Any]] = []

    if not tools_pool:
        tools_pool = [DEFAULT_TOOLS]

    for idx in range(count):
        user_q, assistant_ans = rng.choice(NEGATIVE_TEMPLATES)
        tools_sample = rng.choice(tools_pool)
        system_prompt = build_system_prompt(tools_sample)
        chatml_text = format_chatml(system_prompt, user_q, assistant_ans)

        item = {
            "id": f"neg-synthetic-{idx:04d}",
            "system": system_prompt,
            "user": user_q,
            "assistant": assistant_ans,
            "tools": tools_sample,
            "is_tool_call": False,
            "expected_tool": None,
            "expected_arguments": None,
            "all_expected_tool_calls": [],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": assistant_ans},
            ],
            "text": chatml_text,
        }
        negative_examples.append(item)

    return negative_examples


def prepare_and_process_dataset(
    raw_file: Path | str = DEFAULT_RAW_DIR / "hermes_raw.jsonl",
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    val_ratio: float = 0.15,
    num_negatives: int = 250,
    eval_subset_size: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Ham veriyi okur, negatif örnekleri ekler, train/val split oluşturur ve kaydeder."""
    raw_path = Path(raw_file)
    if not raw_path.exists():
        logger.info("Ham veri bulunamadı, indiriliyor: %s", raw_path)
        download_raw_dataset(output_dir=raw_path.parent, output_filename=raw_path.name)

    logger.info("Ham veri okunuyor: %s", raw_path)
    positive_examples: list[dict[str, Any]] = []
    tools_pool: list[list[dict[str, Any]]] = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            raw_item = json.loads(line_str)
            parsed = parse_raw_item(raw_item)
            if parsed:
                positive_examples.append(parsed)
                if parsed["tools"]:
                    tools_pool.append(parsed["tools"])

    logger.info("Ayrıştırılan orijinal örnek sayısı: %d", len(positive_examples))

    # Negatif örnekler üret
    negative_examples = build_negative_examples(
        tools_pool=tools_pool,
        count=num_negatives,
        seed=seed,
    )
    logger.info("Üretilen sentetik negatif örnek sayısı: %d", len(negative_examples))

    # Tüm örnekleri birleştir ve karıştır
    all_examples = positive_examples + negative_examples
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    total_count = len(all_examples)
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count

    train_data = all_examples[:train_count]
    val_data = all_examples[train_count:]

    # Hızlı baseline eval için val setinden dengeli bir eval_subset oluştur
    eval_pos = [x for x in val_data if x["is_tool_call"]]
    eval_neg = [x for x in val_data if not x["is_tool_call"]]

    target_neg = min(len(eval_neg), int(eval_subset_size * 0.2))  # ~%20 negatif
    target_pos = eval_subset_size - target_neg

    eval_subset = eval_pos[:target_pos] + eval_neg[:target_neg]
    rng.shuffle(eval_subset)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    eval_subset_file = out_path / "eval_subset.jsonl"
    summary_file = out_path / "dataset_summary.json"

    logger.info("Train seti kaydediliyor (%d satır) -> %s", len(train_data), train_file)
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Val seti kaydediliyor (%d satır) -> %s", len(val_data), val_file)
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Eval subset kaydediliyor (%d satır) -> %s", len(eval_subset), eval_subset_file)
    with open(eval_subset_file, "w", encoding="utf-8") as f:
        for item in eval_subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "total_samples": total_count,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "eval_subset_samples": len(eval_subset),
        "positive_tool_calls": sum(1 for x in all_examples if x["is_tool_call"]),
        "negative_examples": sum(1 for x in all_examples if not x["is_tool_call"]),
        "val_ratio": val_ratio,
        "seed": seed,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Veri hazırlama tamamlandı. Özet: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Veri indirme ve hazırlama scripti")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="HuggingFace dataset adı",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Ham veri dizini",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="İşlenmiş veri çıktı dizini",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Yalnızca ham veri indirme işlemini gerçekleştir",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=250,
        help="Eklenecek sentetik negatif örnek sayısı",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation oranı (varsayılan: 0.15)",
    )

    args = parser.parse_args()

    if args.download_only:
        download_raw_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.raw_dir,
        )
    else:
        prepare_and_process_dataset(
            raw_file=Path(args.raw_dir) / "hermes_raw.jsonl",
            output_dir=args.processed_dir,
            val_ratio=args.val_ratio,
            num_negatives=args.num_negatives,
        )


if __name__ == "__main__":
    main()
