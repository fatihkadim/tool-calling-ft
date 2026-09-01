# Google Colab QLoRA Değerlendirme Sorunu ve Çözüm Raporu

## 1. Problem Özeti ve Gözlemlenen Anomaliler

Google Colab üzerinde `Qwen/Qwen2.5-0.5B` modeli ile eğitilen **QLoRA** adaptörünün değerlendirilmesi (`tool_calling_ft.eval.harness`) sonucunda aşağıdaki beklenmedik metrikler elde edilmiştir:

```json
{
  "method": "qlora",
  "base_model": "Qwen/Qwen2.5-0.5B",
  "adapter_path": "checkpoints/qlora",
  "dataset_path": "data/processed/eval_subset.jsonl",
  "quality_metrics": {
    "total_examples": 100,
    "tool_required_count": 80,
    "no_tool_required_count": 20,
    "tool_selection_accuracy": 0.2,
    "argument_accuracy": 0.2,
    "json_validity_rate": 0.2,
    "positive_tool_selection_accuracy": 0.0,
    "positive_argument_accuracy": 0.0,
    "negative_rejection_accuracy": 1.0
  },
  "performance_metrics": {
    "total_samples": 100,
    "total_generated_tokens": 25600,
    "total_generation_seconds": 756.38,
    "throughput_tokens_per_sec": 33.85,
    "latency_ms_per_sample": 7563.83
  }
}
```

### Anomali Bulguları:
1. **%0 Pozitif Doğruluk (0/80):** Tool çağrısı yapması gereken 80 örneğin hiçbirinde geçerli `<tool_call>` tag'i ve JSON üretilememiştir.
2. **%100 Negatif Doğruluk (20/20):** Tool gerektirmeyen 20 örnekte model anlamsız metin ürettiği için sistem çıktıda tool çağrısı bulamamış ve bunu tesadüfen "doğru ret" olarak etiketlemiştir. Toplam başarı $(0 + 20)/100 = \%20$ çıkmıştır.
3. **Sonsuz Token Döngüsü ve Latency Patlaması:** 100 örneğin tamamında `max_new_tokens=256` sınırına takılınmış ($100 \times 256 = 25.600$ token) ve eval süresi **756 saniyeye (12.6 dakika)** ulaşmıştır. Model sürekli `\nLETE\nLETE...`, `\n Fälle\n...`, `\nFSIZE\n...` gibi tekrarlayan parçalar üretmiştir.

---

## 2. Kök Neden Analizi (Root Causes)

### Kök Neden 1: NVIDIA Tesla T4 ve `bfloat16` Donanım Uyuşmazlığı
* Google Colab ücretsiz ortamındaki GPU **NVIDIA Tesla T4**'tür (Turing CC 7.5 mimarisi).
* T4 GPU'lar **donanımsal olarak `bfloat16` desteklemez** (yalnızca `float16` destekler; native `bfloat16` Ampere CC 8.0+ mimarisinde gelmiştir).
* `src/tool_calling_ft/eval/harness.py` dosyasında `AutoModelForCausalLM` yüklenirken:
  ```python
  torch_dtype = torch_dtype if torch_dtype != "auto" else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
  ```
  mantığı çalışıyordu. Colab'da CUDA aktif olduğu için model **`bfloat16`** olarak yüklendi. T4 üzerinde `bfloat16` matris işlemlerinde taşmalar, sayısal kararsızlıklar ve bozuk logit dağılımları meydana gelmiş; greedy decoding (`do_sample=False`) aynı token'ları sürekli seçerek sonsuz döngüye girmiştir.

### Kök Neden 2: QLoRA Adapter'ının Kuantizasyon Olmadan (16-bit) Yüklenmesi
* Eğitim aşamasında taban model **4-bit NF4 (`BitsAndBytesConfig`)** ile kuantize edilmiş ve LoRA adaptörü bu 4-bit taban ağırlıklar üzerindeki artık hataları kompanse edecek şekilde optimize edilmiştir.
* Ancak `eval/harness.py` değerlendirme sırasında base modeli **herhangi bir `BitsAndBytesConfig` vermeden (16-bit unquantized)** yüklemiştir.
* 4-bit için eğitilmiş adaptör, kuantize edilmemiş 16-bit ağırlıklara takıldığında ağırlık skalaları ve aktivasyonlar tamamen uyumsuz hale gelmiş, modelin çıktı kalitesi sıfırlanmıştır.

---

## 3. Yapılan Düzeltmeler

[`src/tool_calling_ft/eval/harness.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/harness.py) modülünde kapsamlı iyileştirmeler yapılmıştır:

### 1. Akıllı Dtype Tespiti (`resolve_torch_dtype`):
* `torch.cuda.is_bf16_supported()` kontrolü eklenmiştir.
* GPU `bfloat16` desteklemiyorsa (T4 gibi) `auto` modu otomatik ve güvenli olarak `torch.float16` seçer. Ampere/Hopper (A100, H100, L4) üzerinde ise `torch.bfloat16` seçer.

```python
def resolve_torch_dtype(torch_dtype: str | torch.dtype = "auto") -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    dtype_str = str(torch_dtype).lower().strip()
    if dtype_str in ("float16", "fp16", "torch.float16"):
        return torch.float16
    elif dtype_str in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    elif dtype_str in ("float32", "fp32", "torch.float32"):
        return torch.float32

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
```

### 2. QLoRA İçin 4-Bit `BitsAndBytesConfig` Desteği:
* `load_model_and_tokenizer` fonksiyonuna `load_in_4bit: bool = False` parametresi eklendi.
* `run_eval` içinde `method_name.lower() == "qlora"` durumunda `load_in_4bit` otomatik olarak `True` olarak ayarlandı.
* Taban model 4-bit NF4 ve uygun `compute_dtype` (`float16` veya `bfloat16`) ile yüklenerek adaptör tam uyumlu şekilde entegre edildi:

```python
if load_in_4bit:
    from transformers import BitsAndBytesConfig

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=resolved_dtype,
        bnb_4bit_use_double_quant=True,
    )
```

### 3. CLI Argümanları Genişletildi:
* `--torch-dtype` (`auto`, `float16`, `bfloat16`, `float32`): İstenirse veri tipi elle belirlenebilir.
* `--load-in-4bit` / `--no-4bit`: 4-bit kuantizasyon yükleme davranışı açıkça kontrol edilebilir.

---

## 4. Test ve Doğrulama

Eklenen fonksiyonlar ve modül testleri için yeni unit testler eklendi:
* `test_resolve_torch_dtype`: Float16, bfloat16, float32 string/dtype dönüşümleri ve T4/A100 GPU simülasyonları test edildi.
* `test_load_model_and_tokenizer_mock`: 4-bit `BitsAndBytesConfig` yükleme mekanizması doğrulandı.

Tüm test paketi başarıyla çalıştırıldı:
```bash
uv run pytest
# 27 passed in 14.65s
```

---

## 5. Colab'da Değerlendirmeyi Yeniden Çalıştırma

Kodları GitHub'a pushladıktan sonra Google Colab notebook'unuzda aşağıdaki adımları izleyebilirsiniz:

1. **Repo Güncellemesini Çekin:**
   ```bash
   %cd /content/tool-calling-ft
   !git pull origin main
   ```

2. **QLoRA Değerlendirmesini Çalıştırın:**
   ```bash
   !python -m tool_calling_ft.eval.harness \
       --method qlora \
       --adapter checkpoints/qlora \
       --dataset data/processed/eval_subset.jsonl
   ```
   *(Artık script T4 üzerinde otomatik olarak `float16` + `4-bit NF4` modunda çalışacak, stop token'ları düzgün algılayacak ve gerçek başarı metriklerini raporlayacaktır.)*
