"""Yontem-agnostik egitim scripti. configs/*.yaml dosyasindaki 'method' alanina
gore LoRA/QLoRA/DoRA/Full FT arasinda dallanir.

Kullanim: uv run python -m tool_calling_ft.training.train --config configs/lora.yaml

TODO: load_config(), build_model(config), build_trainer(model, dataset, config), main()
"""
from peft import TaskType
from transformers import AutoModelForCausalLM
from peft import get_peft_model,LoraConfig,prepare_model_for_kbit_training
import yaml
from transformers import BitsAndBytesConfig
import transformers
import torch

def load_config(ft_method: str) -> dict:
    with open(f"configs/{ft_method}.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
 
 
def build_model(ft_method: str):
    config = load_config(ft_method)
 
    if ft_method == "lora":
        model = AutoModelForCausalLM.from_pretrained(config["base_model"])
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        lora_model = get_peft_model(model, lora_config)
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
        model = AutoModelForCausalLM.from_pretrained(config["base_model"])
        dora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            use_dora=config["lora"]["use_dora"],
            task_type=TaskType.CAUSAL_LM,
        )
        dora_model = get_peft_model(model, dora_config)
        dora_model.print_trainable_parameters()
        return dora_model, config
 
    elif ft_method == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(config["base_model"])
        return model, config
 
    else:
        raise ValueError(f"Bilinmeyen fine-tuning yontemi: {ft_method}")
