"""Yontem-agnostik egitim scripti. configs/*.yaml dosyasindaki 'method' alanina
gore LoRA/QLoRA/DoRA/Full FT arasinda dallanir.

Kullanim: uv run python -m tool_calling_ft.training.train --config configs/lora.yaml

TODO: load_config(), build_model(config), build_trainer(model, dataset, config), main()
"""
