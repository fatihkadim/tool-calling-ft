# 🏆 Tool Calling Fine-Tuning — Final Benchmark Raporu

**Tarih:** Eylül 2026  
**Model:** `Qwen/Qwen2.5-0.5B` (Base Model, Instruct değil)  
**Dataset:** `NousResearch/hermes-function-calling-v1` (1893 pozitif + 250 sentetik negatif)  
**Eval Seti:** 100 örnek (80 pozitif, 20 negatif)  
**Donanım:** NVIDIA Tesla T4 GPU (16 GB VRAM) — Google Colab / Kaggle  

---

## 1. Yönetici Özeti

Bu çalışma, **Qwen2.5-0.5B** base model üzerinde **tool calling (function calling)** görevi için **dört farklı fine-tuning yöntemini** sistematik olarak karşılaştıran bir benchmark'tır.

### Araştırma Sorusu
> *"Aynı model, aynı veri, aynı görev — farklı fine-tuning yöntemleri nasıl bir performans farkı yaratır?"*

### Sonuçlar

| Yöntem | Poz. Tool Selection | Poz. Arg. Accuracy | JSON Validity | Neg. Rejection | Peak VRAM |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Baseline (0-shot) | %0.0 | %0.0 | %0.0 | %100.0* | N/A (CPU) |
| LoRA (16-bit) | %8.75 | %7.71 | %11.25 | %95.0 | 8922 MB |
| QLoRA (4-bit NF4) | %46.25 | %37.48 | %46.25 | %85.0 | **8416 MB** |
| DoRA (use_dora=true) | %13.75 | %11.63 | %15.0 | %95.0 | 8922 MB |
| **Full Fine-Tuning** | **%92.5** | **%44.33** | **%83.75** | **%95.0** | 8914 MB |

> **🏅 Şampiyon: Full Fine-Tuning** — Tool selection'da %92.5 ile açık ara lider. QLoRA'nın 2 katı, negatif rejection'da da %95 ile en iyiler arasında.

*\* Baseline'da model hiç tool çağırmadığı için rejection %100'dür.*

---

## 2. Yöntem Açıklamaları

### 2.1 Baseline (Zero-Shot)
Pre-training sonrası hiçbir fine-tuning yapılmadan, doğrudan Hermes format prompt ile test.

### 2.2 LoRA (Low-Rank Adaptation)
- **Fikir:** Ağırlık matrislerini $W = W_0 + BA$ olarak ayrıştır; yalnızca düşük rank $B, A$ matrislerini eğit.
- **Config:** $r=16, \alpha=32$, `target_modules=[q_proj, k_proj, v_proj, o_proj]`
- **Trainable:** 2.16M parametre (%0.43)

### 2.3 QLoRA (Quantized LoRA)
- **Fikir:** Base modeli 4-bit NF4 kuantize et, üstüne LoRA adapter ekle.
- **Config:** NF4 + double quantization + LoRA ($r=16, \alpha=32$)
- **Trainable:** 2.16M parametre (%0.43), base model 4-bit

### 2.4 DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Fikir:** Ağırlık matrisini *magnitude* ve *direction* bileşenlerine ayır; LoRA yalnızca direction'ı günceller, magnitude ayrı öğrenilir.
- **Config:** LoRA + `use_dora=true`, $r=16, \alpha=32$, `dropout=0.05`
- **Trainable:** 2.21M parametre (%0.45)

### 2.5 Full Fine-Tuning
- **Fikir:** Tüm parametreleri ($\sim$494M) güncelle — MLP dahil.
- **Config:** `lr=2e-5`, `epochs=3`, `batch_size=1`, `grad_accum=8`, gradient checkpointing
- **Trainable:** 494M parametre (%100)
- **Training:** 684 step, final loss ~0.18

---

## 3. Kalite Metrikleri Karşılaştırması

### 3.1 Pozitif Örnekler (Tool Çağrısı Beklenen, n=80)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy | %0.0 | %8.75 | %46.25 | %13.75 | **%92.5** |
| Argument Accuracy | %0.0 | %7.71 | %37.48 | %11.63 | **%44.33** |
| JSON Validity | %0.0 | %11.25 | %46.25 | %15.0 | **%83.75** |

### 3.2 Negatif Örnekler (Tool Çağrılmaması Gereken, n=20)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Negative Rejection Acc. | %100.0* | **%95.0** | %85.0 | **%95.0** | **%95.0** |
| Unnecessary Tool Call Rate | %0.0 | %5.0 | %15.0 | %5.0 | %5.0 |

### 3.3 Genel Metrikler (Toplam 100 örnek)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy | %60.0* | %26.0 | %54.0 | %30.0 | **%93.0** |
| Argument Accuracy | %60.0* | %25.17 | %46.98 | %28.3 | **%54.47** |
| JSON Validity Rate | %100.0* | %29.0 | %57.0 | %32.0 | **%87.0** |
| Invalid Tool Call Rate | %0.0 | %1.0 | %1.0 | %1.0 | %2.0 |

*\* Baseline 5 sample ile çalıştırılmış (istatistiksel güvenilirlik düşük).*

---

## 4. Performans & Kaynak Metrikleri

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Training Time | — | 4h 36m | ~3h 48m | ~4h 50m | ~8h* |
| Peak VRAM (Eval) | 0 MB (CPU) | 8922 MB | **8416 MB** | 8922 MB | 8914 MB |
| Throughput (tok/s) | 9.78 | 34.0 | **36.89** | 29.6 | 28.13 |
| Latency (ms/sample) | 26,175 | 7,528 | **6,939** | 8,649 | 4,752 |
| Trainable Params | 494M (100%) | 2.16M (0.43%) | 2.16M (0.43%) | 2.21M (0.45%) | 494M (100%) |
| Model Size | 988 MB | ~8.5 MB adapter | ~8.5 MB adapter | ~8.5 MB adapter | 988 MB |

*\* Full FT training time tahmini (3 epoch × 684 step, gradient checkpointing ile).*

---

## 5. Yöntem Sıralaması (Final)

### 🥇 Full Fine-Tuning — Kalite Şampiyonu
- **%92.5** pozitif tool selection — diğer yöntemlerin 2-10x üzerinde
- **%83.75** JSON validity — format öğrenimi başarılı
- **%95.0** negatif rejection — agresif tool çağrısı sorunu yok
- MLP katmanları dahil olduğundan format dönüşümü tam başarılı
- **Dezavantaj:** Tüm model kaydedilmeli (988 MB vs 8.5 MB adapter), daha uzun training

### 🥈 QLoRA (4-bit NF4) — Verimlilik Şampiyonu
- **%46.25** pozitif tool selection — adapter yöntemlerinin en iyisi
- En düşük VRAM (8416 MB) ve en yüksek throughput (36.89 tok/s)
- **Dezavantaj:** Negatif rejection en düşük (%85.0)

### 🥉 DoRA (use_dora=true)
- LoRA'ya göre +5% iyileşme, ama QLoRA'nın çok gerisinde
- Magnitude decomposition küçük modellerde sınırlı etki

### 4️⃣ LoRA (16-bit)
- Sadece %8.75 pozitif tool selection
- Pre-training artıkları format öğrenimini sabote ediyor

---

## 6. Analiz & Çıkarımlar

### 6.1 Full FT Neden Bu Kadar İyi?

1. **MLP Katmanları Güncellenebilir:** Adapter yöntemlerinde (LoRA/QLoRA/DoRA) yalnızca attention katmanları hedeflenirken, Full FT'de MLP dahil tüm katmanlar güncellenir. `<tool_call>` etiketi ve JSON format yapısı MLP'lerde kodlanır.

2. **Kapasite Sınırı Yok:** 494M parametrenin tamamı kullanılabilir — adapter'ların 2.16M parametresiyle karşılaştırıldığında ~230x daha fazla öğrenme kapasitesi.

3. **Pre-training Artıkları Bastırılmış:** Tüm ağırlıklar güncellendiğinden `sourceMapping`, `_Parms` gibi halüsinasyonlar giderilmiş.

### 6.2 Kuantizasyon Etkisi Teyit Edildi

QLoRA'nın adapter yöntemleri arasında lider olması, NF4 kuantizasyonunun düzenlileştirme (regularization) etkisini doğrulamaya devam ediyor. 4-bit base model, pre-training artıklarını bastırarak adapter'ın format öğrenmesini kolaylaştırıyor.

### 6.3 Maliyet-Kalite Dengesi

| Senaryo | Önerilen Yöntem |
|---------|----------------|
| Maksimum kalite, kaynak önemli değil | **Full FT** |
| Sınırlı VRAM, hızlı iterasyon | **QLoRA** |
| Production'da adapter swap gerekli | **QLoRA** |
| Birden fazla görev için aynı base model | **QLoRA** (adapter'lar arası geçiş) |

---

## 7. Full FT Çıktı Örnekleri

### ✅ Başarılı Negatif Ret
**Soru:** *"Mona Lisa'yı kim yaptı?"*  
**Çıktı:** Doğal dilde yanıt (tool çağrısı yok) ✓

### ✅ Başarılı Pozitif Tool Call
```json
<tool_call>
{"name": "find_astronomy_apps", "arguments": {"features": ["real-time data", "celestial events", "constellation mapping"], "user_location": "34.0522N,118.2437W"}}
</tool_call>
```

### ⚠️ Agresif Multi-Tool Call
Tek tool beklenirken birden fazla tool çağrısı yapılması (invalid_tool_call_rate: %2.0)

---

## 8. Sonuç

### Ana Bulgular
1. **Full Fine-Tuning** küçük modellerde (0.5B) tool calling için **en iyi yöntem** — %92.5 tool selection.
2. **QLoRA** adapter yöntemleri arasında açık ara lider — kaynak verimliliği ile kaliteyi dengeliyor.
3. **LoRA ve DoRA** attention-only target ile küçük modellerde yetersiz kalıyor.
4. Kuantizasyonun "bilgi kaybı" varsayımı küçük modellerde **çürütülmüştür** — düzenlileştirme etkisi baskın.
5. MLP katmanlarının güncellenmesi, format dönüşümü (JSON + `<tool_call>` etiketi) için **kritik** öneme sahip.

### Gelecek Çalışmalar
1. `target_modules: all-linear` ile LoRA/QLoRA denemesi (MLP dahil)
2. Daha büyük model (Qwen2.5-1.5B veya 3B) üzerinde benchmark tekrarı
3. Multi-turn tool calling evaluation
4. Full FT + QLoRA hybrid: QLoRA ile pre-train, Full FT ile fine-tune

---

*Bu rapor, dört yöntemin tamamlanmasıyla nihai halini almıştır.*
