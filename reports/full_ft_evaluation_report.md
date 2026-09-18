# Full Fine-Tuning — Değerlendirme & Benchmark Raporu

**Tarih:** Eylül 2026  
**Model:** `Qwen/Qwen2.5-0.5B` (Base Model)  
**Yöntem:** Full Fine-Tuning (tüm 494M parametre güncellendi)  
**Donanım:** Google Colab / Kaggle NVIDIA Tesla T4 GPU (16 GB VRAM)  
**Veri Seti:** `NousResearch/hermes-function-calling-v1` (100 örnek eval subset: 80 pozitif, 20 negatif)  
**Eğitim Parametreleri:** `lr=2e-5`, `epochs=3`, `batch_size=1`, `grad_accum=8`, `max_seq_len=2048`, gradient checkpointing  

---

## 1. Yönetici Özeti (Executive Summary)

**Full Fine-Tuning**, tüm benchmark boyunca test edilen dört yöntem arasında **açık ara en iyi kalite sonuçlarını** elde etmiştir. Tüm 494M parametrenin güncellenmesi — özellikle MLP katmanlarının dahil olması — modelin `<tool_call>` formatını ve JSON yapısını öğrenmesini sağlamıştır.

### Özet Sonuçlar

| Metrik Grubu | Metrik | Baseline | LoRA | QLoRA | DoRA | **Full FT** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pozitif Kalite** | Tool Selection Acc. | %0.0 | %8.75 | %46.25 | %13.75 | **%92.5** |
| | Argument Accuracy | %0.0 | %7.71 | %37.48 | %11.63 | **%44.33** |
| | JSON Validity | %0.0 | %11.25 | %46.25 | %15.0 | **%83.75** |
| **Negatif Kalite** | Rejection Accuracy | %100.0 | %95.0 | %85.0 | %95.0 | **%95.0** |
| | Unnecessary Tool Call | %0.0 | %5.0 | %15.0 | %5.0 | **%5.0** |
| **Performans** | Throughput (tok/s) | 9.78 | 34.0 | 36.89 | 29.6 | **28.13** |
| | Latency (ms/sample) | 26,175 | 7,528 | 6,939 | 8,649 | **4,752** |
| | Peak VRAM (MB) | 0 | 8,922 | 8,416 | 8,922 | **8,914** |
| | Trainable Params | 494M | 2.16M | 2.16M | 2.21M | **494M** |

---

## 2. Full FT Ölçüm Değerleri (Doğrudan Çıktılar)

### 2.1. Kalite Metrikleri Tablosu
```
=====================================================
KALİTE METRİKLERİ (QUALITY METRICS)
=====================================================
total_examples                  : 100
tool_required_count             : 80
no_tool_required_count          : 20
tool_selection_accuracy         : 0.9300  (%93.0)
argument_accuracy               : 0.5447  (%54.47)
json_validity_rate              : 0.8700  (%87.0)
invalid_tool_call_rate          : 0.0200  (%2.0)
positive_tool_selection_accuracy: 0.9250  (%92.5  -> 80 örnekten 74'ü doğru)
positive_argument_accuracy      : 0.4433  (%44.33)
positive_json_validity          : 0.8375  (%83.75 -> 80 örnekten 67'si geçerli JSON)
negative_rejection_accuracy     : 0.9500  (%95.0  -> 20 örnekten 19'u doğru ret)
unnecessary_tool_call_rate      : 0.0500  (%5.0   -> 20 örnekten 1'inde gereksiz çağrı)
```

### 2.2. Performans ve Donanım Metrikleri
```
=====================================================
PERFORMANS & DONANIM METRİKLERİ
=====================================================
total_samples                   : 100
total_generated_tokens          : 13368
total_generation_seconds        : 475.23 saniye
throughput_tokens_per_sec       : 28.13 tok/s
latency_ms_per_sample           : 4752.32 ms
description                     : Eval: full_ft
elapsed_seconds                 : 478.1273 saniye
peak_vram_mb                    : 8913.67 MB (~8.91 GB)
```

### 2.3. Parametre İstatistikleri
```
=====================================================
PARAMETRE İSTATİSTİKLERİ
=====================================================
trainable_params                : 494,032,768 (%100)
all_params                      : 494,032,768
trainable_percentage            : 100.0%
model_size                      : ~988 MB (model.safetensors)
```

---

## 3. Eğitim Detayları

### 3.1 Training Loss Eğrisi

| Epoch | Step | Loss | Grad Norm |
|:-----:|:----:|:----:|:---------:|
| 0.04 | 10 | 1.5757 | 29.375 |
| 0.18 | 40 | 0.7101 | 12.313 |
| 0.44 | 100 | 0.4470 | 7.875 |
| 0.70 | 160 | 0.3710 | 8.188 |
| 1.00 | 228 | — | — |
| 1.00 | — | eval_loss: **0.5535** | — |
| 1.05 | 240 | 0.1888 | 7.031 |
| 1.49 | 340 | 0.2413 | 3.969 |
| 2.00 | 456 | — | — |
| 2.00 | — | eval_loss: **0.5558** | — |
| 2.11 | 480 | 0.2339 | 8.875 |
| 2.50 | 570 | 0.1884 | 4.094 |
| 2.90 | 660 | 0.1548 | 6.719 |
| 2.98 | 680 | 0.1828 | 1.125 |

**Gözlemler:**
- Loss, ilk epoch'ta hızlıca 1.57 → 0.28'e düştü
- 2. ve 3. epoch'larda 0.13–0.28 aralığında stabilize oldu
- Eval loss ~0.55'te sabit kaldı — hafif overfitting işareti ama kalite metrikleri çok iyi
- Final loss: **~0.18** (step 680)

### 3.2 Checkpoint Bilgileri

| Checkpoint | Step | Epoch | Açıklama |
|-----------|------|-------|----------|
| `checkpoint-400` | 400 | ~1.75 | İlk ara kayıt |
| `checkpoint-600` | 600 | ~2.63 | İkinci ara kayıt |
| `checkpoint-684` | 684 | 3.00 | Final checkpoint |

### 3.3 Training Konfigürasyonu
- **Total steps:** 684 (228 step/epoch × 3 epoch)
- **Learning rate:** 2e-5 (linear decay)
- **Warmup:** ~40 step
- **Batch size:** 1 (effective: 8 with grad_accum)
- **Gradient checkpointing:** Enabled (VRAM tasarrufu)
- **Total FLOPs:** 1.45 × 10¹⁶

---

## 4. Canlı Çıktı Analizi

### Örnek 1: ✅ Negatif Test (Mona Lisa — Tool Gerekmez)
- **Beklenen:** Tool çağırmadan doğrudan yanıt
- **Model Çıktısı:**
```text
The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1506. 
It is currently displayed at the Louvre Museum in Paris, France.
```
- **Analiz:** ✅ **Mükemmel ret.** Doğal dilde temiz yanıt, halüsinasyon yok, `sourceMapping` yok. LoRA/DoRA'daki gürültü paternleri tamamen giderilmiş.

### Örnek 2: ✅ Pozitif Test (Astronomi Uygulamaları)
- **Beklenen:** `find_astronomy_apps` tool çağrısı
- **Model Çıktısı:**
```xml
<tool_call>
{"name": "find_astronomy_apps", "arguments": {"features": ["real-time data", "celestial events", "constellation mapping"], "user_location": "34.0522N,118.2437W"}}
</tool_call>
```
- **Analiz:** ✅ **Başarılı!** `<tool_call>` etiketi var, JSON geçerli, tool adı doğru. Argümanlarda küçük farklar var (beklenen `augmented reality` yerine `celestial events`). LoRA/DoRA'nın tamamen başarısız olduğu bu örnek Full FT ile çözülmüş.

### Örnek 3: ⚠️ Pozitif Test (Film İncelemeleri — Multi-Tool)
- **Beklenen:** `fetch_movie_reviews(movie_title="Dune", limit=5)`
- **Model Çıktısı:**
```xml
<tool_call>
{"name": "fetch_movie_reviews", "arguments": {"movie_title": "Dune", "limit": 5}}
</tool_call>
<tool_call>
{"name": "fetch_artist_details", "arguments": {"artist_name": "Timothée Chalamet"}}
</tool_call>
<tool_call>
{"name": "fetch_box_office_figures", "arguments": {"movie_title": "Dune", "date_range": {"start_date": "2023-01-01", "end_date": "2023-04-30"}}}
</tool_call>
```
- **Analiz:** ⚠️ İlk tool çağrısı **tam doğru** (isim + argümanlar eşleşiyor). Ancak ek 2 tool çağrısı üretilmiş. Format öğrenimi mükemmel (`<tool_call>` etiketleri, temiz JSON), ama model "ne zaman durmalı" sorusunda aşırıya kaçıyor.

### Örnek 4: ✅ Pozitif Test (Webhook)
- **Beklenen:** `create_task_completed_webhook(planner_id="abc123", task_id="task456")`
- **Model Çıktısı:**
```xml
<tool_call>
{"name": "create_task_completed_webhook", "arguments": {"planner_id": "abc123", "task_id": "task456"}}
</tool_call>
```
- **Analiz:** ✅ **Birebir eşleşme.** Tool adı, argüman adları ve değerleri tamamen doğru. LoRA/DoRA'nın multi-language halüsinasyona girdiği bu örnek Full FT ile kusursuz çözülmüş.

### Örnek 5: ⚠️ Pozitif Test (Market Analizi — Multi-Tool)
- **Beklenen:** `analyze_market_sentiment` tool çağrısı
- **Model Çıktısı:** 4 adet `<tool_call>` bloğu (`analyze_market_sentiment`, `predict_price_movements`, `execute_trades`, `manage_diversified_portfolio`)
- **Analiz:** ⚠️ İlk tool doğru seçilmiş ve argümanlar büyük ölçüde eşleşiyor. Ancak 3 ek tool çağrısı üretilmiş. LoRA/DoRA'daki `_Parms:` tekrar döngüsü yerine burada yapısal ama gereksiz tool çağrıları var.

---

## 5. Full FT vs Adapter Yöntemleri: Detaylı Karşılaştırma

### 5.1 Full FT'nin Adapter Yöntemlerine Göre Üstünlükleri

| Metrik | LoRA | QLoRA | DoRA | **Full FT** | Full FT vs QLoRA |
|--------|:----:|:-----:|:----:|:-----------:|:---------------:|
| Poz. Tool Selection | %8.75 | %46.25 | %13.75 | **%92.5** | 🟢 **+46.25%** |
| Poz. Arg. Accuracy | %7.71 | %37.48 | %11.63 | **%44.33** | 🟢 **+6.85%** |
| Poz. JSON Validity | %11.25 | %46.25 | %15.0 | **%83.75** | 🟢 **+37.5%** |
| Neg. Rejection | %95.0 | %85.0 | %95.0 | **%95.0** | 🟢 **+10.0%** |
| Latency (ms) | 7,528 | 6,939 | 8,649 | **4,752** | 🟢 **-2,187 ms** |

### 5.2 Full FT'nin Dezavantajları

| Boyut | Adapter Yöntemleri | Full FT |
|-------|-------------------|---------|
| Model boyutu | ~8.5 MB adapter | **988 MB** tam model |
| Multi-task esnekliği | Adapter swap ile hızlı geçiş | Her görev için ayrı model |
| Overfitting riski | Düşük (az parametre) | Orta (494M param, 1822 örnek) |
| Training süresi | ~4-5 saat | ~8 saat (tahmini) |

### 5.3 Halüsinasyon Karşılaştırması

| Patern | LoRA | DoRA | QLoRA | **Full FT** |
|--------|:----:|:----:|:-----:|:-----------:|
| `sourceMapping` halüsinasyonu | ✓ Var | ✓ Var | ✓ Var (az) | **✗ Yok** |
| `_Parms:` tekrar döngüsü | ✓ Var | ✓ Var | ✓ Var (az) | **✗ Yok** |
| Multi-language mixing | ✓ Var | ✓ Var | ✓ Var (az) | **✗ Yok** |
| `<tool_call>` etiketi yok | ✓ Var | ✓ Var | ~Kısmen | **✗ Yok** |
| Gereksiz multi-tool çağrısı | ✗ Yok | ✗ Yok | ✗ Az | **⚠️ Var** |

**Kritik bulgu:** Full FT, pre-training artığı halüsinasyonları tamamen gidermiş, ancak yeni bir sorun ortaya çıkmış: **gereksiz multi-tool çağrısı** (%2 invalid tool call rate). Model format öğreniminde başarılı ama "ne zaman durmalı" konusunda ince ayar gerekiyor.

---

## 6. Neden Full FT Bu Kadar İyi?

### 6.1 MLP Katmanlarının Rolü

Adapter yöntemlerinde (LoRA/QLoRA/DoRA) yalnızca `q_proj, k_proj, v_proj, o_proj` (attention) hedeflenirken, Full FT'de **MLP katmanları dahil tüm katmanlar** güncellenir.

- **Attention katmanları:** "Hangi bilgiye dikkat etmeliyim?" → Tool seçimi
- **MLP katmanları:** "Bilgiyi nasıl dönüştürmeliyim?" → Format üretimi (`<tool_call>`, JSON yapısı)

Format öğrenimi büyük ölçüde MLP'lerde gerçekleştiğinden, adapter yöntemlerinin format başarısı sınırlı kalmıştır.

### 6.2 Kapasite Farkı

| | Adapter | Full FT | Oran |
|--|---------|---------|------|
| Trainable params | 2.16M | 494M | **~230x** |
| Hedeflenen katmanlar | 4 (attention) | Tümü | **Tam kapsam** |

### 6.3 Pre-training Artıklarının Bastırılması

Full FT'de tüm ağırlıklar güncellendiğinden, pre-training sırasında öğrenilen web/kod gürültüsü (`sourceMapping`, `_Parms`, multi-language mixing) **tamamen bastırılmıştır**. Adapter yöntemlerinde bu artıklar frozen ağırlıklarda kalmaya devam ediyordu.

---

## 7. Sonuçlar & Öneriler

### 7.1 Full FT Spesifik Sonuçlar

1. **%92.5 pozitif tool selection** — diğer tüm yöntemlerin 2-10x üzerinde.
2. **%83.75 JSON validity** — `<tool_call>` etiketi ve temiz JSON yapısı başarıyla öğrenilmiş.
3. **%95.0 negatif rejection** — QLoRA'nın (%85) aksine agresif tool çağrısı sorunu yok.
4. **Pre-training halüsinasyonları giderilmiş** — `sourceMapping`, `_Parms` gibi paternler tamamen yok.
5. **Yeni sorun: multi-tool çağrısı** — %2 invalid tool call rate, tek tool beklenirken birden fazla üretilmesi.

### 7.2 Final Yöntem Sıralaması (Dört Yöntem)

```
🥇 Full FT  (%92.5 tool sel., %44.33 arg acc., 28.13 tok/s, 8914 MB, 494M params)
🥈 QLoRA   (%46.25 tool sel., %37.48 arg acc., 36.89 tok/s, 8416 MB, 2.16M params)
🥉 DoRA    (%13.75 tool sel., %11.63 arg acc., 29.6 tok/s, 8922 MB, 2.21M params)
4️⃣ LoRA    (%8.75 tool sel., %7.71 arg acc., 34.0 tok/s, 8922 MB, 2.16M params)
```

### 7.3 Kullanım Önerisi

| Senaryo | Önerilen Yöntem |
|---------|----------------|
| Maksimum kalite, tek görev | **Full FT** |
| Sınırlı VRAM, hızlı iterasyon | **QLoRA** |
| Production'da adapter swap gerekli | **QLoRA** |
| Birden fazla görev, aynı base model | **QLoRA** (adapter'lar arası geçiş) |

---

*Bu rapor `reports/full_ft_metrics.json` verilerine dayanmaktadır.*
