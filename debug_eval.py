"""Dogrulama: tail-preserving truncation sonrasi eval_loss nan sorunu duzeldi mi?"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from transformers import AutoTokenizer
from tool_calling_ft.training.collator import DataCollatorForCompletionOnlyLM
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_seq_len = 2048
response_template_str = "<|im_start|>assistant\n"
_resp_token_ids = tokenizer.encode(response_template_str, add_special_tokens=False)

def smart_tokenize(text):
    """train.py'deki yeni tokenize_fn ile ayni mantik."""
    full = tokenizer(text, truncation=False, padding=False)
    input_ids = full["input_ids"]

    if len(input_ids) <= max_seq_len:
        return input_ids, full["attention_mask"], False

    resp_len = len(_resp_token_ids)
    resp_pos = None
    for i in range(len(input_ids) - resp_len, -1, -1):
        if input_ids[i:i+resp_len] == _resp_token_ids:
            resp_pos = i
            break

    if resp_pos is not None:
        response_part = input_ids[resp_pos:]
        prompt_budget = max_seq_len - len(response_part)
        if prompt_budget > 0:
            prompt_part = input_ids[:resp_pos][-prompt_budget:]
            truncated = prompt_part + response_part
        else:
            truncated = response_part[:max_seq_len]
    else:
        truncated = input_ids[:max_seq_len]

    return truncated, [1] * len(truncated), True


collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_str,
    tokenizer=tokenizer,
)

for split in ["train", "val"]:
    lines = open(f"data/processed/{split}.jsonl", "r", encoding="utf-8").readlines()
    
    total = len(lines)
    truncated_count = 0
    no_response_count = 0
    all_masked_count = 0
    total_unmasked = 0
    
    # Truncate edilen orneklerden bir ornek decode icin sakla
    sample_truncated = None
    
    for idx, line in enumerate(lines):
        d = json.loads(line)
        input_ids, attn_mask, was_truncated = smart_tokenize(d["text"])
        
        if was_truncated:
            truncated_count += 1
            if sample_truncated is None:
                sample_truncated = (idx, input_ids)
        
        # Response template kontrolu
        resp_len = len(_resp_token_ids)
        found = any(input_ids[i:i+resp_len] == _resp_token_ids for i in range(len(input_ids)-resp_len+1))
        if not found:
            no_response_count += 1
        
        # Collator ile label kontrolu
        tokens = {"input_ids": input_ids, "attention_mask": attn_mask}
        batch = collator([tokens])
        labels = batch["labels"][0]
        unmasked = (labels != -100).sum().item()
        if unmasked == 0:
            all_masked_count += 1
        total_unmasked += unmasked
    
    print(f"\n{'='*60}")
    print(f"{split.upper()} SET:")
    print(f"  Toplam ornek: {total}")
    print(f"  Truncation olan: {truncated_count} ({100*truncated_count/total:.1f}%)")
    print(f"  Response template bulunamayan: {no_response_count}")
    print(f"  Tamamen maskeli ornek: {all_masked_count}")
    print(f"  Ortalama unmasked token/ornek: {total_unmasked/total:.1f}")
    
    if no_response_count == 0 and all_masked_count == 0:
        print(f"  [OK] BASARILI - eval_loss nan sorunu duzeltildi!")
    else:
        print(f"  [FAIL] HALA SORUNLU")
    
    # Truncate edilen bir ornegi decode et
    if sample_truncated:
        idx, ids = sample_truncated
        decoded = tokenizer.decode(ids)
        print(f"\n  Truncate edilen ornek #{idx} (ilk 200 char):")
        print(f"    {decoded[:200]}...")
        print(f"  ... (son 200 char):")
        print(f"    ...{decoded[-200:]}")
