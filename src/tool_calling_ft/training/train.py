"""Yontem-agnostik egitim scripti. configs/*.yaml dosyasindaki 'method' alanina
gore LoRA/QLoRA/DoRA/Full FT arasinda dallanir.

Kullanim: uv run python -m tool_calling_ft.training.train --config configs/qlora.yaml
"""
import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str) -> dict:
    """YAML config dosyasını doğrudan yolundan yükler."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict):
    ft_method = config.get("method", "qlora")

    # GPU ve mimari desteğine göre uygun veri tipini belirle (T4'te float16, A100'de bfloat16)
    model_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    if ft_method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=model_dtype,
        )
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        lora_model = get_peft_model(model, lora_config)
        # base model dondurulmus oldugu icin checkpointing'in backward'i
        # calismasi icin bu gerekli (yoksa "does not require grad" hatasi alinir)
        lora_model.enable_input_require_grads()
        try:
            lora_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            lora_model.gradient_checkpointing_enable()
        lora_model.config.use_cache = False
        lora_model.print_trainable_parameters()
        return lora_model, config

    elif ft_method == "qlora":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map[config["quantization"]["bnb_4bit_compute_dtype"]]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["quantization"]["load_in_4bit"],
            bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        )

        qlora_model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
        )

        qlora_model = prepare_model_for_kbit_training(
            qlora_model, use_gradient_checkpointing=True
        )
        # gradient checkpointing ile cache celisir
        qlora_model.config.use_cache = False

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )

        qlora_model = get_peft_model(qlora_model, lora_config)
        qlora_model.print_trainable_parameters()
        return qlora_model, config

    elif ft_method == "dora":
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=model_dtype,
        )
        dora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            use_dora=config["lora"]["use_dora"],
            task_type=TaskType.CAUSAL_LM,
        )
        dora_model = get_peft_model(model, dora_config)
        dora_model.enable_input_require_grads()
        try:
            dora_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            dora_model.gradient_checkpointing_enable()
        dora_model.config.use_cache = False
        dora_model.print_trainable_parameters()
        return dora_model, config

    elif ft_method == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=model_dtype,
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        return model, config

    else:
        raise ValueError(f"Bilinmeyen fine-tuning yontemi: {ft_method}")


def build_trainer(model, config: dict):
    """Tokenizer, dataset ve Trainer'i kurar.

    NOT: data/processed/train.jsonl ve val.jsonl dosyalarinin onceden
    (config['dataset'] = NousResearch/hermes-function-calling-v1 kaynagindan)
    bir preprocessing adimiyla uretilmis oldugu varsayilir. Bu dosyalarda
    her satirin bir 'text' alani (chat-template uygulanmis, duz string)
    icermesi beklenir.

    Degisiklikler:
    - DataCollatorForCompletionOnlyLM ile yalnizca assistant yanitina loss verilir
    - Dinamik padding (batch-level) ile VRAM tasarrufu saglanir
    - warmup_ratio kullanilir (surumden bagimsiz guvenli kullanim)
    """
    from tool_calling_ft.training.collator import DataCollatorForCompletionOnlyLM

    from pathlib import Path
    train_file = Path("data/processed/train.jsonl")
    val_file = Path("data/processed/val.jsonl")
    if not train_file.exists() or not val_file.exists():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "data/processed/train.jsonl veya val.jsonl bulunamadi! "
            "Otomatik olarak prepare_and_process_dataset() calistiriliyor..."
        )
        from tool_calling_ft.data.prepare_dataset import prepare_and_process_dataset
        prepare_and_process_dataset()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "val": str(val_file),
        },
    )

    max_seq_len = config["training"]["max_seq_len"]

    # Tail-preserving truncation: assistant yanitini koruyarak prompt'u soldan kirp.
    # Standart truncation (soldan) uzun orneklerde assistant yanitini tamamen
    # siliyordu -> tum label'lar -100 -> eval_loss nan.
    response_template_str = "<|im_start|>assistant\n"
    _resp_token_ids = tokenizer.encode(response_template_str, add_special_tokens=False)

    def tokenize_fn(examples):
        results = {"input_ids": [], "attention_mask": []}

        for text in examples["text"]:
            full = tokenizer(text, truncation=False, padding=False)
            input_ids = full["input_ids"]

            if len(input_ids) <= max_seq_len:
                # Sigiyor, oldugu gibi kullan
                results["input_ids"].append(input_ids)
                results["attention_mask"].append(full["attention_mask"])
            else:
                # Son response template pozisyonunu bul (sondan basla)
                resp_len = len(_resp_token_ids)
                resp_pos = None
                for i in range(len(input_ids) - resp_len, -1, -1):
                    if input_ids[i : i + resp_len] == _resp_token_ids:
                        resp_pos = i
                        break

                if resp_pos is not None:
                    # Assistant yaniti (template dahil) korunacak
                    response_part = input_ids[resp_pos:]
                    prompt_budget = max_seq_len - len(response_part)

                    if prompt_budget > 0:
                        # Prompt'un SONUNDAN prompt_budget kadar token al
                        # (bastan kesilir -> tool sema basliklari gider, user sorusu kalir)
                        prompt_part = input_ids[:resp_pos][-prompt_budget:]
                        truncated = prompt_part + response_part
                    else:
                        # Response tek basina sigmiyor, sagdan kirp
                        truncated = response_part[:max_seq_len]
                else:
                    # Template bulunamadiysa (olagandisi), klasik sol truncation
                    truncated = input_ids[:max_seq_len]

                results["input_ids"].append(truncated)
                results["attention_mask"].append([1] * len(truncated))

        return results

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_train = tokenized["train"]
    tokenized_val = tokenized["val"]

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and config.get("quantization", {}).get("bnb_4bit_compute_dtype") != "float16"
    )

    # Config'ten opsiyonel degerler al, yoksa mantikli varsayilan kullan
    save_steps = config["training"].get("save_steps", 100)
    batch_size = config["training"]["batch_size"]
    grad_accum_steps = config["training"]["grad_accum_steps"]
    epochs = config["training"]["epochs"]
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.05))

    total_steps = (len(tokenized_train) // (batch_size * grad_accum_steps)) * epochs
    warmup_steps = config["training"].get("warmup_steps", max(1, int(total_steps * warmup_ratio)))

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=epochs,
        learning_rate=float(config["training"]["learning_rate"]),
        gradient_accumulation_steps=grad_accum_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        seed=config["training"]["seed"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=10,
        warmup_steps=warmup_steps,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # Assistant-only loss: Yalnizca "<|im_start|>assistant\n" sonrasi tokenlara
    # loss uygulanir. System prompt, tool semalari ve user mesaji maskelenir.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    trainer.tokenizer = tokenizer
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configs/qlora.yaml gibi")
    args = parser.parse_args()

    config = load_config(args.config)
    model, config = build_model(config)
    trainer = build_trainer(model, config)
    trainer.train()
    trainer.save_model(config["output_dir"])
    if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()