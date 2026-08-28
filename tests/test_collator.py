"""tests/test_collator.py - DataCollatorForCompletionOnlyLM unit testleri."""

import torch
from transformers import AutoTokenizer

from tool_calling_ft.training.collator import DataCollatorForCompletionOnlyLM


def test_collator_masks_prompt_and_pads():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    text1 = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4<|im_end|>"
    )
    text2 = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWrite a sentence about sky.<|im_end|>\n"
        "<|im_start|>assistant\nThe sky is bright and blue today.<|im_end|>"
    )

    sample1 = tokenizer(text1)
    sample2 = tokenizer(text2)

    batch = collator([sample1, sample2])

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape

    # Her iki örnek için assistant kısmının maskelenmediğini doğrula
    for i in range(2):
        unmasked_ids = [tok for tok in batch["labels"][i].tolist() if tok != -100]
        unmasked_text = tokenizer.decode(unmasked_ids)
        # unmasked_text sadece assistant cevabını içermeli, system ve user içermemeli
        assert "You are a helpful assistant" not in unmasked_text
        assert "What is 2+2" not in unmasked_text
        assert "Write a sentence" not in unmasked_text

    # Sample 1 unmasked text '4<|im_end|>' içermeli
    s1_unmasked = tokenizer.decode([tok for tok in batch["labels"][0].tolist() if tok != -100])
    assert "4" in s1_unmasked
    assert "<|im_end|>" in s1_unmasked

    # Pad tokenlarının da -100 olarak maskelendiğini doğrula
    pad_positions = (batch["input_ids"] == tokenizer.pad_token_id)
    assert (batch["labels"][pad_positions] == -100).all()


def test_collator_no_response_template_masks_all():
    """Response template bulunamadığında tüm sequence -100 olmalı."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    truncated_text = "<|im_start|>system\nYou are a bot.<|im_end|>\n<|im_start|>user\nHello"
    sample = tokenizer(truncated_text)
    batch = collator([sample])

    assert (batch["labels"] == -100).all()


def test_collator_with_instruction_template_multi_turn():
    """Çoklu turda instruction template ile assistant dışı kısımların maskelenmesi."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        instruction_template="<|im_start|>user\n",
        tokenizer=tokenizer,
    )

    multi_turn = (
        "<|im_start|>system\nBot<|im_end|>\n"
        "<|im_start|>user\nTurn 1<|im_end|>\n"
        "<|im_start|>assistant\nResp 1<|im_end|>\n"
        "<|im_start|>user\nTurn 2<|im_end|>\n"
        "<|im_start|>assistant\nResp 2<|im_end|>"
    )
    sample = tokenizer(multi_turn)
    batch = collator([sample])

    unmasked_ids = [tok for tok in batch["labels"][0].tolist() if tok != -100]
    unmasked_text = tokenizer.decode(unmasked_ids)

    assert "Resp 1" in unmasked_text
    assert "Resp 2" in unmasked_text
    assert "Turn 1" not in unmasked_text
    assert "Turn 2" not in unmasked_text
