"""training/ modülü için pytest testleri.

GPU gerektirmeyen mock testler: config yükleme, model builder doğrulaması,
DataCollatorForCompletionOnlyLM unit testleri.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from tool_calling_ft.training.collator import DataCollatorForCompletionOnlyLM


# ─── Config Yükleme Testleri ───


def test_load_config_lora(tmp_path: Path):
    """LoRA YAML config dosyasının doğru yüklendiğini doğrular."""
    from tool_calling_ft.training.train import load_config

    config_content = textwrap.dedent("""\
        method: lora
        base_model: Qwen/Qwen2.5-0.5B
        output_dir: checkpoints/lora
        lora:
          r: 16
          alpha: 32
          dropout: 0.05
          target_modules: [q_proj, k_proj, v_proj, o_proj]
        training:
          epochs: 3
          batch_size: 2
          grad_accum_steps: 4
          learning_rate: 2e-4
          max_seq_len: 2048
          warmup_ratio: 0.05
          save_steps: 200
          seed: 42
    """)
    config_file = tmp_path / "lora.yaml"
    config_file.write_text(config_content)

    config = load_config(str(config_file))
    assert config["method"] == "lora"
    assert config["base_model"] == "Qwen/Qwen2.5-0.5B"
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["training"]["epochs"] == 3
    assert float(config["training"]["learning_rate"]) == 2e-4


def test_load_config_qlora(tmp_path: Path):
    """QLoRA YAML config dosyasında quantization ayarlarının doğruluğunu kontrol eder."""
    from tool_calling_ft.training.train import load_config

    config_content = textwrap.dedent("""\
        method: qlora
        base_model: Qwen/Qwen2.5-0.5B
        output_dir: checkpoints/qlora
        quantization:
          load_in_4bit: true
          bnb_4bit_quant_type: nf4
          bnb_4bit_compute_dtype: bfloat16
          bnb_4bit_use_double_quant: true
        lora:
          r: 16
          alpha: 32
          dropout: 0.05
          target_modules: [q_proj, k_proj, v_proj, o_proj]
        training:
          epochs: 3
          batch_size: 2
          grad_accum_steps: 4
          learning_rate: 2e-4
          max_seq_len: 2048
          warmup_ratio: 0.05
          save_steps: 100
          seed: 42
    """)
    config_file = tmp_path / "qlora.yaml"
    config_file.write_text(config_content)

    config = load_config(str(config_file))
    assert config["method"] == "qlora"
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert config["quantization"]["bnb_4bit_compute_dtype"] == "bfloat16"


def test_load_config_dora(tmp_path: Path):
    """DoRA config dosyasında use_dora=true ayarının varlığını doğrular."""
    from tool_calling_ft.training.train import load_config

    config_content = textwrap.dedent("""\
        method: dora
        base_model: Qwen/Qwen2.5-0.5B
        output_dir: checkpoints/dora
        lora:
          r: 16
          alpha: 32
          dropout: 0.05
          use_dora: true
          target_modules: [q_proj, k_proj, v_proj, o_proj]
        training:
          epochs: 3
          batch_size: 2
          grad_accum_steps: 4
          learning_rate: 2e-4
          max_seq_len: 2048
          warmup_ratio: 0.05
          save_steps: 200
          seed: 42
    """)
    config_file = tmp_path / "dora.yaml"
    config_file.write_text(config_content)

    config = load_config(str(config_file))
    assert config["method"] == "dora"
    assert config["lora"]["use_dora"] is True


def test_load_config_full_ft(tmp_path: Path):
    """Full FT config dosyasında lora bloğunun olmamasını kontrol eder."""
    from tool_calling_ft.training.train import load_config

    config_content = textwrap.dedent("""\
        method: full_ft
        base_model: Qwen/Qwen2.5-0.5B
        output_dir: checkpoints/full_ft
        training:
          epochs: 3
          batch_size: 1
          grad_accum_steps: 8
          learning_rate: 2e-5
          max_seq_len: 2048
          warmup_ratio: 0.05
          save_steps: 200
          seed: 42
    """)
    config_file = tmp_path / "full_ft.yaml"
    config_file.write_text(config_content)

    config = load_config(str(config_file))
    assert config["method"] == "full_ft"
    assert "lora" not in config
    assert float(config["training"]["learning_rate"]) == 2e-5


def test_load_config_missing_file():
    """Olmayan dosya ile FileNotFoundError fırlatılmalı."""
    from tool_calling_ft.training.train import load_config

    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")


# ─── build_model Mock Testleri ───


def test_build_model_unknown_method(tmp_path: Path):
    """Bilinmeyen fine-tuning yöntemi ValueError fırlatmalı."""
    from tool_calling_ft.training.train import build_model

    config = {
        "method": "unknown_method",
        "base_model": "Qwen/Qwen2.5-0.5B",
    }

    with pytest.raises(ValueError, match="Bilinmeyen fine-tuning"):
        build_model(config)


def test_build_model_lora_config_structure():
    """LoRA config'inin gerekli tüm alanları içerdiğini doğrular."""
    config_file = Path("configs/lora.yaml")
    if not config_file.exists():
        pytest.skip("configs/lora.yaml bulunamadı")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    assert "lora" in config
    assert "r" in config["lora"]
    assert "alpha" in config["lora"]
    assert "dropout" in config["lora"]
    assert "target_modules" in config["lora"]
    assert isinstance(config["lora"]["target_modules"], list)
    assert len(config["lora"]["target_modules"]) > 0


def test_build_model_qlora_quantization_structure():
    """QLoRA config'inin quantization bloğunun doğru yapıda olduğunu doğrular."""
    config_file = Path("configs/qlora.yaml")
    if not config_file.exists():
        pytest.skip("configs/qlora.yaml bulunamadı")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    assert "quantization" in config
    q = config["quantization"]
    assert q["load_in_4bit"] is True
    assert q["bnb_4bit_quant_type"] in ("nf4", "fp4")
    assert q["bnb_4bit_compute_dtype"] in ("float16", "bfloat16", "float32")


# ─── DataCollatorForCompletionOnlyLM Testleri ───


@pytest.fixture
def mock_tokenizer():
    """Test için sahte tokenizer oluşturur."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 151643

    # response template token ids: "<|im_start|>assistant\n"
    # Basitleştirilmiş: [100, 200, 300] olarak simüle et
    tok.encode.return_value = [100, 200, 300]

    def mock_pad(examples, padding=True, return_tensors="pt"):
        """Sahte padding: tüm örnekleri aynı uzunluğa getir."""
        max_len = max(len(ex["input_ids"]) for ex in examples)
        padded_ids = []
        padded_mask = []
        for ex in examples:
            ids = ex["input_ids"]
            pad_len = max_len - len(ids)
            padded_ids.append([0] * pad_len + ids)  # Sol padding
            padded_mask.append([0] * pad_len + [1] * len(ids))
        return {
            "input_ids": torch.tensor(padded_ids),
            "attention_mask": torch.tensor(padded_mask),
        }

    tok.pad = mock_pad
    return tok


def test_collator_basic_masking(mock_tokenizer):
    """Collator'ın prompt tokenlarını maskeleyip assistant tokenlarını açtığını doğrular."""
    # Simüle edilmiş input:
    # [10, 20, 30, 100, 200, 300, 40, 50, 60] (9 token)
    # response template = [100, 200, 300] → idx 3'te başlıyor
    # assistant yanıtı: [40, 50, 60] → idx 6-8

    collator = DataCollatorForCompletionOnlyLM(
        response_template=[100, 200, 300],
        tokenizer=mock_tokenizer,
    )

    examples = [
        {"input_ids": [10, 20, 30, 100, 200, 300, 40, 50, 60]},
    ]

    batch = collator(examples)

    assert "labels" in batch
    labels = batch["labels"][0].tolist()

    # İlk 6 token (prompt + template) maskelenmeli (-100)
    for i in range(6):
        assert labels[i] == -100, f"Token {i} maskelenmeli ama {labels[i]}"

    # Son 3 token (assistant yanıtı) açık olmalı
    assert labels[6] == 40
    assert labels[7] == 50
    assert labels[8] == 60


def test_collator_no_response_template(mock_tokenizer):
    """Response template bulunamadığında tüm tokenlar maskelenmeli."""
    collator = DataCollatorForCompletionOnlyLM(
        response_template=[100, 200, 300],
        tokenizer=mock_tokenizer,
    )

    # Template olmayan input
    examples = [
        {"input_ids": [10, 20, 30, 40, 50]},
    ]

    batch = collator(examples)
    labels = batch["labels"][0].tolist()

    # Tamamı maskelenmeli
    assert all(l == -100 for l in labels), f"Tüm tokenlar maskelenmeli: {labels}"


def test_collator_pad_tokens_in_response(mock_tokenizer):
    """Assistant yanıtı içindeki pad tokenlar da maskelenmeli."""
    collator = DataCollatorForCompletionOnlyLM(
        response_template=[100, 200, 300],
        tokenizer=mock_tokenizer,
    )

    # input: [10, 100, 200, 300, 40, 0, 50]  (0 = pad token)
    examples = [
        {"input_ids": [10, 100, 200, 300, 40, 0, 50]},
    ]

    batch = collator(examples)
    labels = batch["labels"][0].tolist()

    # idx 4 (40) ve idx 6 (50) açık olmalı
    assert labels[4] == 40
    assert labels[6] == 50
    # idx 5 (pad=0) maskelenmeli
    assert labels[5] == -100


def test_collator_multiple_examples_padding(mock_tokenizer):
    """Farklı uzunluktaki örneklerin doğru padding ile batch'lenmesini doğrular."""
    collator = DataCollatorForCompletionOnlyLM(
        response_template=[100, 200, 300],
        tokenizer=mock_tokenizer,
    )

    examples = [
        {"input_ids": [10, 100, 200, 300, 40, 50]},      # 6 token
        {"input_ids": [10, 20, 100, 200, 300, 60, 70, 80]},  # 8 token
    ]

    batch = collator(examples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape

    # İkinci örneğin assistant yanıtı
    labels_1 = batch["labels"][1].tolist()
    # Son 3 token (60, 70, 80) açık olmalı
    assert labels_1[-1] == 80
    assert labels_1[-2] == 70
    assert labels_1[-3] == 60


def test_collator_string_template():
    """response_template string olarak verildiğinde encode edilmeli."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.encode.return_value = [111, 222]

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tok,
    )

    assert collator.response_token_ids == [111, 222]
    tok.encode.assert_called_once_with("<|im_start|>assistant\n", add_special_tokens=False)


# ─── Config Dosya Bütünlüğü Testleri ───


@pytest.mark.parametrize("config_name", ["lora.yaml", "qlora.yaml", "dora.yaml", "full_ft.yaml"])
def test_config_files_have_required_fields(config_name: str):
    """Her config dosyasının zorunlu alanları içerdiğini doğrular."""
    config_file = Path("configs") / config_name
    if not config_file.exists():
        pytest.skip(f"{config_name} bulunamadı")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Tüm config'lerde olması gereken alanlar
    assert "method" in config
    assert "base_model" in config
    assert "output_dir" in config
    assert "training" in config

    training = config["training"]
    assert "epochs" in training
    assert "batch_size" in training
    assert "learning_rate" in training
    assert "max_seq_len" in training
    assert "seed" in training


@pytest.mark.parametrize("config_name", ["lora.yaml", "qlora.yaml", "dora.yaml"])
def test_adapter_configs_have_lora_block(config_name: str):
    """LoRA/QLoRA/DoRA config'lerinin lora bloğu içerdiğini doğrular."""
    config_file = Path("configs") / config_name
    if not config_file.exists():
        pytest.skip(f"{config_name} bulunamadı")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    assert "lora" in config
    lora = config["lora"]
    assert "r" in lora
    assert "alpha" in lora
    assert "target_modules" in lora
    assert isinstance(lora["target_modules"], list)
