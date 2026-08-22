"""Yontem-agnostik egitim scripti. configs/*.yaml dosyasindaki 'method' alanina
gore LoRA/QLoRA/DoRA/Full FT arasinda dallanir.

Kullanim: uv run python -m tool_calling_ft.training.train --config configs/lora.yaml
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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_config(ft_method: str) -> dict:
    with open(f"configs/{ft_method}.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(ft_method: str):
    config = load_config(ft_method)

    if ft_method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
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
        dora_model.gradient_checkpointing_enable()
        dora_model.config.use_cache = False
        dora_model.print_trainable_parameters()
        return dora_model, config

    elif ft_method == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        # Full FT'de base model zaten tamamen trainable, enable_input_require_grads
        # gerekmez. 4GB VRAM'de dahi bu ayarlarla sinirda calisir; optimizer
        # state'leri (Adam: momentum+variance) tek basina birkac GB tutabilir.
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
    """
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "val": "data/processed/val.jsonl",
        },
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config["training"]["max_seq_len"],
        )

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_train = tokenized["train"]
    tokenized_val = tokenized["val"]

    use_bf16 = torch.cuda.is_available() and config.get("quantization", {}).get(
        "bnb_4bit_compute_dtype"
    ) != "float16"

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=float(config["training"]["learning_rate"]),
        gradient_accumulation_steps=config["training"]["grad_accum_steps"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=8,
        seed=config["training"]["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configs/lora.yaml gibi")
    args = parser.parse_args()

    ft_method = args.config.split("/")[-1].replace(".yaml", "")

    model, config = build_model(ft_method)
    trainer = build_trainer(model, config)
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
