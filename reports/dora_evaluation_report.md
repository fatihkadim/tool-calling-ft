# DoRA (Weight-Decomposed LoRA) Fine-Tuning — Değerlendirme & Benchmark Raporu

**Tarih:** Eylül 2026  
**Model:** `Qwen/Qwen2.5-0.5B` (Base Model)  
**Yöntem:** DoRA (Weight-Decomposed Low-Rank Adaptation, 16-bit, `use_dora=true`)  
**Donanım:** Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM)  
**Veri Seti:** `NousResearch/hermes-function-calling-v1` (100 örnek eval subset: 80 pozitif, 20 negatif)  
**Eğitim Parametreleri:** $r=16, \alpha=32$, `target_modules=[q_proj, k_proj, v_proj, o_proj]`, `lr=2e-4`, `batch_size=2`, `grad_accum=4`, `max_seq_len=2048`, `epochs=3`, `dropout=0.05`  
**PEFT Versiyonu:** 0.19.1  

---

## 1. Yönetici Özeti (Executive Summary)

Google Colab ortamında NVIDIA Tesla T4 GPU üzerinde tamamlanan **DoRA (Weight-Decomposed Low-Rank Adaptation)** eğitimi ve 100 örneklik standart eval subset üzerindeki değerlendirme başarıyla kaydedilmiştir.

DoRA, adapter_config.json'da `"use_dora": true` flag'i ile PEFT kütüphanesi aracılığıyla uygulanmıştır. Model 684 adımda (3 epoch) eğitilmiş ve 3 ara checkpoint (400, 600, 684) kaydedilmiştir.

### Özet Sonuçlar

| Metrik Grubu | Metrik | Baseline | LoRA | QLoRA | **DoRA** | DoRA vs LoRA | DoRA vs QLoRA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pozitif Kalite** | Tool Selection Acc. | %0.0 | %8.75 | **%46.25** | **%13.75** | 🟢 +5.0% | 🔻 -32.5% |
| | Argument Accuracy | %0.0 | %7.71 | **%37.48** | **%11.63** | 🟢 +3.92% | 🔻 -25.85% |
| | JSON Validity | %0.0 | %11.25 | **%46.25** | **%15.0** | 🟢 +3.75% | 🔻 -31.25% |
| **Negatif Kalite** | Rejection Accuracy | %100.0 | %95.0 | %85.0 | **%95.0** | ➖ Eşit | 🟢 +10.0% |
| | Unnecessary Tool Call | %0.0 | %5.0 | %15.0 | **%5.0** | ➖ Eşit | 🟢 -10.0% |
| **Performans** | Throughput (tok/s) | 9.78 | 34.0 | **36.89** | **29.6** | 🔻 -4.4 | 🔻 -7.29 |
| | Latency (ms/sample) | 26,175 | 7,528 | **6,939** | **8,649** | 🔻 +1,121 | 🔻 +1,710 |
| | Peak VRAM (MB) | 0 | 8,922 | **8,416** | **8,922** | ➖ Eşit | 🔻 +506 |
| | Trainable Params | 494M | 2.16M | 2.16M | **2.21M** | 🔺 +49K | 🔺 +49K |

---

## 2. DoRA Ölçüm Değerleri (Doğrudan Çıktılar)

### 2.1. Kalite Metrikleri Tablosu
```
=====================================================
KALİTE METRİKLERİ (QUALITY METRICS)
=====================================================
total_examples                  : 100
tool_required_count             : 80
no_tool_required_count          : 20
tool_selection_accuracy         : 0.3000  (%30.0)
argument_accuracy               : 0.2830  (%28.3)
json_validity_rate              : 0.3200  (%32.0)
invalid_tool_call_rate          : 0.0100  (%1.0)
positive_tool_selection_accuracy: 0.1375  (%13.75 -> 80 örnekten 11'i doğru)
positive_argument_accuracy      : 0.1163  (%11.63)
positive_json_validity          : 0.1500  (%15.0  -> 80 örnekten 12'si geçerli JSON)
negative_rejection_accuracy     : 0.9500  (%95.0  -> 20 örnekten 19'u doğru ret)
unnecessary_tool_call_rate      : 0.0500  (%5.0   -> 20 örnekten 1'inde gereksiz çağrı)
```

### 2.2. Performans ve Donanım Metrikleri
```
=====================================================
PERFORMANS & DONANIM METRİKLERİ
=====================================================
total_samples                   : 100
total_generated_tokens          : 25600 (Her örnek tam 256 token limitine vurdu)
total_generation_seconds        : 864.92 saniye
throughput_tokens_per_sec       : 29.6 tok/s
latency_ms_per_sample           : 8649.17 ms
description                     : Eval: dora
elapsed_seconds                 : 869.7871 saniye
peak_vram_mb                    : 8922.28 MB (~8.92 GB)
```

### 2.3. Parametre İstatistikleri
```
=====================================================
PARAMETRE İSTATİSTİKLERİ
=====================================================
trainable_params                : 0 (eval modunda)
all_params                      : 496,244,608
trainable_percentage            : 0.0% (eval modunda — eğitimde %0.45)
lora_params                     : 2,211,840 (LoRA'dan +49,152 fazla — magnitude vektörleri)
adapter_size                    : ~8.5 MB (adapter_model.safetensors: 8,885,704 bytes)
```

---

## 3. DoRA Checkpoint Bilgileri

### Ara Checkpointler
| Checkpoint | Step | Açıklama |
|-----------|------|----------|
| `checkpoint-400` | 400 | İlk ara kayıt (~%58.5 eğitim) |
| `checkpoint-600` | 600 | İkinci ara kayıt (~%87.7 eğitim) |
| `checkpoint-684` | 684 | Final checkpoint (3 epoch tamamlandı) |

### Final Adapter Dosyaları
| Dosya | Boyut | Açıklama |
|-------|-------|----------|
| `adapter_model.safetensors` | 8.5 MB | DoRA adapter ağırlıkları |
| `adapter_config.json` | 1 KB | PEFT konfigürasyonu |
| `tokenizer.json` | 10.9 MB | Qwen2.5 tokenizer |
| `tokenizer_config.json` | 668 B | Tokenizer ayarları |
| `chat_template.jinja` | 2.4 KB | ChatML şablonu |
| `training_args.bin` | 5.2 KB | Eğitim parametreleri |
| `README.md` | 5.2 KB | HuggingFace model kartı |

---

## 4. Canlı Çıktı Anomalileri & Halüsinasyon Analizi

### Örnek 1: Negatif Test (Mona Lisa — Tool Gerekmez)
- **Beklenen:** Tool çağırmadan doğrudan yanıt
- **Model Çıktısı:**
```text
The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1506. It is 
currently housed at the Louvre Museum in Paris, France. sourceMapping
référencer
You are a function calling AI model...
```
- **Analiz:** ✅ Doğru ret (tool çağırmadı), ancak `sourceMapping` halüsinasyonu + sistem prompt'unun kısmen tekrarlanması. LoRA ile aynı patern.

### Örnek 2: Pozitif Test (Astronomi Uygulamaları)
- **Beklenen:** `find_astronomy_apps` tool çağrısı
- **Model Çıktısı:**
```text
"""""",
 """""",
 """""",
 ... (256 token boyunca tekrar)
```
- **Analiz:** ❌ Tam başarısızlık. Model tıkanma döngüsüne (repetition loop) girmiş, hiçbir anlamlı çıktı üretmemiş. LoRA'da da benzer patern var ancak farklı tokenlarla.

### Örnek 3: Pozitif Test (Film İncelemeleri — Kısmi Başarı)
- **Beklenen:** `fetch_movie_reviews(movie_title="Dune", limit=5)`
- **Model Çıktısı:**
```text
�
{"name": "fetch_movie_reviews", "arguments": {"movie_title": "Dune", "limit": 5}}
IntoConstraints
_Parms
{"name": "fetch_artist_details", "arguments": {"artist_name": "Timothée Chalamet"}}
...
ELLOW (tekrar)
```
- **Analiz:** ⚠️ Kısmi başarı! Tool adı ve argümanları **doğru JSON** formatında üretilmiş. Ancak:
  - `<tool_call>` XML etiketi yok
  - Birden fazla tool çağrısı (hallucination)
  - Tekrarlayan gürültü tokenları (`_Parms`, `ELLOW`)

### Örnek 4: Pozitif Test (Webhook — QLoRA ile Benzer)
- **Beklenen:** `create_task_completed_webhook(planner_id="abc123", task_id="task456")`
- **Model Çıktısı:**
```text
norge
절차 ل Initializing webhook و تفعيله في Microsoft Planner :
1. اختر سجل planner الخاص بك من الملفات الموجودة...
```
- **Analiz:** ❌ Model multi-language hallucination'a girmiş (Norveçce, Korece, Arapça). Tool çağrısı yapılamamış.

### Örnek 5: Pozitif Test (Market Analizi — Repetition Loop)
- **Beklenen:** `analyze_market_sentiment` tool çağrısı
- **Model Çıktısı:**
```text
_Parms:
_Parms:
_Parms:
... (256 token boyunca tekrar)
```
- **Analiz:** ❌ `_Parms:` token'ı tekrar döngüsüne girmiş. LoRA'da da aynı patern mevcut.

---

## 5. DoRA vs LoRA: Detaylı Karşılaştırma

### 5.1 DoRA'nın LoRA'ya Göre İyileştirmeleri

1. **+5% Pozitif Tool Selection:** 80 örnekte 7 yerine 11 doğru tool seçimi (4 ek doğru)
2. **+3.75% JSON Validity:** 9 yerine 12 geçerli JSON çıktısı (3 ek geçerli)
3. **Eşit Negatif Rejection:** Her ikisi de %95.0 — DoRA daha agresif tool çağırmıyor

### 5.2 DoRA'nın LoRA'ya Göre Dezavantajları

1. **-4.4 tok/s Throughput:** Magnitude decomposition inference overhead'i
2. **+1,121 ms Latency:** Her sample'da ek ~1.1 saniye gecikme
3. **+49K Parametre:** Magnitude vektörleri ek parametre gerektiriyor

### 5.3 Değerlendirme: İyileşme Anlamlı mı?

| Boyut | Değerlendirme |
|-------|--------------|
| İstatistiksel Anlamlılık | %5'lik fark, 80 örnekle sınırlı test setinde marginal (p > 0.1) |
| Pratik Değer | 4 ek doğru tool çağrısı, maliyet/faydaya göre sınırlı |
| Inference Maliyeti | %13 daha yavaş inference, production'da dezavantaj |
| **Sonuç** | DoRA bu setup'ta LoRA'ya göre anlamlı bir üstünlük sağlamamaktadır |

---

## 6. Tüm Yöntemlerin Ortak Halüsinasyon Paternleri

Üç 16-bit yöntemde (Baseline, LoRA, DoRA) ortak görülen sorunlar:

| Patern | Açıklama | Kaynak |
|--------|----------|--------|
| `sourceMapping` | JavaScript source-map referansı | Pre-training web verisi |
| `_Parms:` | Bilinmeyen parametrik referans | Pre-training kod verisi |
| `.createFrom("tools")` | TypeScript/JavaScript factory pattern | Pre-training kod verisi |
| `ITableView` / `quine` | Tekrarlayan token döngüsü | EOS token öğrenememe |
| Multi-language mixing | Norveçce/Korece/Arapça karışımı | Pre-training multilingual veri |
| `ELLOW` / `DERP` | Anlamsız token tekrarı | Degenerate generation |
| `<|endoftext|>` padding | EOS yerine padding | Tokenizer/training mismatch |

**QLoRA'da bu paternler daha az şiddetli** — kuantizasyon bu gürültülü ağırlık bileşenlerini bastırmaktadır.

---

## 7. Sonuçlar & Öneriler

### 7.1 DoRA Spesifik Sonuçlar

1. **DoRA, LoRA'ya göre marginal iyileşme sağlamıştır** (+5% tool selection, +3.75% JSON validity).
2. **Magnitude/direction ayrımı 0.5B modelde yetersiz kalmıştır** — bu teknik daha büyük modellerde (7B+) daha etkili olabilir.
3. **Inference overhead anlamlıdır** — throughput %13 düşmüş, production kullanımda önemli.
4. **Negatif rejection LoRA ile eşit** — DoRA daha fazla tool çağırma eğilimi göstermemiştir.

### 7.2 Üç Yöntem Sıralaması (Adapter Methods)

```
🥇 QLoRA  (%46.25 tool sel., %37.48 arg acc., 36.89 tok/s, 8416 MB)
🥈 DoRA   (%13.75 tool sel., %11.63 arg acc., 29.6 tok/s, 8922 MB)  
🥉 LoRA   (%8.75 tool sel., %7.71 arg acc., 34.0 tok/s, 8922 MB)
```

### 7.3 Sonraki Adımlar
1. **Full Fine-Tuning** — Son kalan yöntem. Tüm parametrelerin güncellenmesiyle format öğrenimi daha güçlü olabilir.
2. Tüm yöntemler tamamlandığında final karşılaştırma grafikleri ve radar chart oluşturulacak.

---

*Bu rapor `reports/dora_metrics.json` verilerine dayanmaktadır.*
