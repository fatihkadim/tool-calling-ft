# 🏆 Tool Calling Fine-Tuning — Final Benchmark Raporu

**Tarih:** Eylül 2026 (Ekim 2026'da ölçülen verilere göre revize edildi)  
**Model:** `Qwen/Qwen2.5-0.5B` (Base Model, Instruct değil)  
**Dataset:** `NousResearch/hermes-function-calling-v1` (1893 pozitif + 250 sentetik negatif)  
**Eval Seti:** 100 örnek (80 pozitif, 20 negatif)  
**Donanım:** NVIDIA Tesla T4 GPU (16 GB VRAM) — Google Colab / Kaggle  
**Çalıştırma:** Her yöntem için tek eğitim (seed 42); güven aralığı yok

---

## 1. Yönetici Özeti

Bu çalışma, **Qwen2.5-0.5B** base model üzerinde **tool calling (function calling)** görevi için **dört farklı fine-tuning yöntemini** karşılaştıran bir benchmark'tır.

### Araştırma Sorusu
> *"Aynı model, aynı veri, aynı görev — farklı fine-tuning yöntemleri nasıl bir performans farkı yaratır?"*

### Sonuçlar

| Yöntem | Poz. Tool Selection | Poz. Arg. Accuracy | JSON Validity | Neg. Rejection | Peak VRAM (Eval) |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Baseline (0-shot) | —† | —† | —† | —† | N/A (CPU) |
| LoRA (16-bit) | %8.75 | %7.71 | %11.25 | %95.0 | 8922 MB |
| QLoRA (4-bit NF4) | %46.25‡ | %37.48 | %46.25‡ | %85.0 | 8416 MB |
| DoRA (use_dora=true) | %13.75 | %11.63 | %15.0 | %95.0 | 8922 MB |
| **Full Fine-Tuning** | **%92.5** | **%44.33** | **%83.75** | **%95.0** | 8914 MB |

> **En iyi sonuç: Full Fine-Tuning.** Tool selection'da %92.5 ile açık ara önde; bu, QLoRA'nın yaklaşık 2 katı. Negatif rejection'da %95 (20 örnekte 1 hata).

*† Baseline, 100 örneklik eval setinde çalıştırılmadı. Yalnızca CPU'da 5 örneklik bir deneme var (`baseline_metrics.json`: 2 pozitif, 3 negatif; 0/2 doğru tool çağrısı, 3/3 doğru ret). Bu raporun önceki sürümündeki %0 / %100 değerleri ölçülmemiş, tahmin edilmişti. **Yapılacak:** `scripts/run_baseline.py` tüm eval setinde GPU'da yeniden çalıştırılmalı.*

*‡ Parser, `<tool_call>` etiketi olmayan çıplak `{"name": ...}` JSON'u da tool çağrısı olarak sayıyor. QLoRA çıktılarının çoğu bu biçimde olduğu için skor, sıkı bir parser'la daha düşük çıkar (bkz. Bölüm 7).*

*Peak VRAM, değerlendirme sırasında ölçüldü; eğitim belleğini göstermez (bkz. Bölüm 4).*

---

## 2. Yöntem Açıklamaları

### 2.1 Baseline (Zero-Shot)
Hiçbir fine-tuning yapılmadan, doğrudan Hermes formatındaki prompt ile test. Şu ana kadar yalnızca 5 örnekle CPU'da çalıştırıldı.

### 2.2 LoRA (Low-Rank Adaptation)
- **Fikir:** Ağırlık matrislerini $W = W_0 + BA$ olarak ayrıştır; yalnızca düşük rank'lı $B, A$ matrislerini eğit.
- **Config:** $r=16, \alpha=32$, `target_modules=[q_proj, k_proj, v_proj, o_proj]`
- **Trainable:** 2.16M parametre (%0.43)

### 2.3 QLoRA (Quantized LoRA)
- **Fikir:** Base modeli 4-bit NF4 ile kuantize et, üstüne LoRA adapter ekle.
- **Config:** NF4 + double quantization + LoRA ($r=16, \alpha=32$). Colab notebook'u `bnb_4bit_compute_dtype: float16` ile çalıştı (`configs/qlora.yaml` ise `bfloat16` diyor).
- **Trainable:** 2.16M parametre (%0.43), base model 4-bit

### 2.4 DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Fikir:** Ağırlık matrisini *magnitude* ve *direction* bileşenlerine ayır; LoRA yalnızca direction'ı günceller, magnitude ayrı öğrenilir.
- **Config:** LoRA + `use_dora=true`, $r=16, \alpha=32$, `dropout=0.05`
- **Trainable:** 2.21M parametre (%0.45)

### 2.5 Full Fine-Tuning
- **Fikir:** Tüm parametreleri ($\sim$494M) güncelle; MLP katmanları dahil.
- **Config:** `lr=2e-5`, `epochs=3`, `batch_size=1`, `grad_accum=8`, gradient checkpointing
- **Trainable:** 494M parametre (%100)
- **Training:** 684 step, final train loss ~0.18, eval loss ~0.55 (bkz. `full_ft_evaluation_report.md`)

---

## 3. Kalite Metrikleri Karşılaştırması

### 3.1 Pozitif Örnekler (Tool Çağrısı Beklenen, n=80)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy | —† | %8.75 (7/80) | %46.25 (37/80) | %13.75 (11/80) | **%92.5 (74/80)** |
| Argument Accuracy | —† | %7.71 | %37.48 | %11.63 | **%44.33** |
| JSON Validity | —† | %11.25 | %46.25 | %15.0 | **%83.75** |

### 3.2 Negatif Örnekler (Tool Çağrılmaması Gereken, n=20)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Negative Rejection Acc. | —† | %95.0 (19/20) | %85.0 (17/20) | %95.0 (19/20) | %95.0 (19/20) |
| Unnecessary Tool Call Rate | —† | %5.0 | %15.0 | %5.0 | %5.0 |

### 3.3 Genel Metrikler (Toplam 100 örnek)

| Metrik | Baseline (5 örnek)* | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy | %60.0* | %26.0 | %54.0 | %30.0 | **%93.0** |
| Argument Accuracy | %60.0* | %25.17 | %46.98 | %28.3 | **%54.47** |
| JSON Validity Rate | %100.0* | %29.0 | %57.0 | %32.0 | **%87.0** |
| Invalid Tool Call Rate | %0.0* | %1.0 | %1.0 | %1.0 | %2.0 |

*\* Baseline yalnızca 5 örnekle (2 pozitif, 3 negatif) çalıştırıldı; diğer sütunlarla karşılaştırılamaz. %60, 3 negatifin doğru reddinden geliyor.*

---

## 4. Performans & Kaynak Metrikleri

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Training Time | — | 4h 36m (kayıtlı) | ~3h 48m (kayıtlı değil) | ~4h 50m (kayıtlı değil) | ~8h (tahmin) |
| Peak VRAM (Eval) | 0 MB (CPU) | 8922 MB | 8416 MB | 8922 MB | 8914 MB |
| Üretilen token (toplam) | 1,280 (5 örnek) | 25,600 | 25,600 | 25,600 | 13,368 |
| Throughput (tok/s) | 9.78 | 34.0 | 36.89 | 29.6 | 28.13 |
| Latency (ms/sample) | 26,175 | 7,528 | 6,939 | 8,649 | **4,752** |
| Trainable Params | 0 (eğitim yok) | 2.16M (0.43%) | 2.16M (0.43%) | 2.21M (0.45%) | 494M (100%) |
| Model Size | 988 MB | ~8.5 MB adapter | ~8.5 MB adapter | ~8.7 MB adapter | 988 MB |

**Bu tablonun yorumu:**

- **Throughput/latency yöntemlerin hızını ölçmüyor.** LoRA, DoRA ve QLoRA her örnekte tam 256 token (`max_new_tokens`) üretti: hiçbiri `<|im_end|>` üretmedi ve cevaptan sonra tekrarlayan anlamsız metne kaydı. Bu modellerin yüksek tok/s değeri, kullanılabilir cevabı daha hızlı ürettikleri anlamına gelmiyor. Full FT durmayı öğrendi; örnek başına en düşük gecikmenin nedeni bu.
- **Peak VRAM (Eval)** batch 4 ve ~2k token'lık prompt'larla üretim sırasında ölçüldü. Değerin büyük kısmı aktivasyonlardan, KV cache'ten ve logit'lerden geliyor. QLoRA'daki ~500 MB'lık fark kabaca 4-bit ağırlık tasarrufu kadar. **Eğitim VRAM'i ölçülmedi**; QLoRA'nın asıl avantajı orada görülürdü.
- **Eğitim süresi** yalnızca LoRA için `lora_metrics.json`'da kayıtlı. QLoRA ve DoRA süreleri metrik dosyalarında yok; Full FT süresi tahmin.

---

## 5. Yöntem Sıralaması (Bu Kurulumda)

### 🥇 Full Fine-Tuning — En Yüksek Kalite
- **%92.5** pozitif tool selection; adapter yöntemlerinin 2 ile 10 katı arasında
- **%83.75** JSON validity; format başarıyla öğrenildi
- Durmayı öğrendi (`<|im_end|>`); kontrolsüz üretim yok
- **%95.0** negatif rejection (19/20)
- **Dezavantaj:** Tüm model kaydedilmeli (988 MB'a karşı 8.5 MB adapter), eğitim daha uzun

### 🥈 QLoRA (4-bit NF4) — En İyi Adapter Skoru
- **%46.25** pozitif tool selection; adapter yöntemlerinin en yükseği (gevşek parser'ın katkısıyla)
- Durmayı öğrenmedi: her örnekte 256 token üretti
- Negatif rejection en düşük (%85.0, 20 örnekte 3 hata)
- LoRA'ya karşı büyük farkın nedeni belirsiz (bkz. 6.2)

### 🥉 DoRA (use_dora=true)
- LoRA'dan 4 örnek fazla doğru (11/80'e karşı 7/80); tek çalıştırmada gürültü içinde

### 4️⃣ LoRA (16-bit)
- %8.75 pozitif tool selection; çıktılar cevaptan sonra `sourceMapping` gibi anlamsız metinle devam ediyor

---

## 6. Analiz & Çıkarımlar

### 6.1 Full FT Neden Bu Kadar İyi?

1. **Daha geniş güncelleme alanı:** Adapter yöntemlerinde (LoRA/QLoRA/DoRA) yalnızca attention projeksiyonları hedeflendi; Full FT'de MLP dahil tüm katmanlar güncelleniyor. Bu fark sonucun en olası açıklaması. Ancak MLP'yi de kapsayan bir LoRA (`all-linear`) denenmediği için nedensellik doğrulanmadı.
2. **Kapasite:** 494M parametrenin tamamı güncelleniyor; adapter'ların 2.16M parametresinin ~230 katı.
3. **Durmayı öğrenme:** Full FT, `<|im_end|>` üretip duruyor; adapter modelleri ise cevaptan sonra `sourceMapping`, `_Parms` gibi anlamsız metinle devam ediyor.

### 6.2 QLoRA > LoRA: Henüz Açıklanmadı

QLoRA ile LoRA arasındaki 5 katlık fark (%46.25'e karşı %8.75), bu raporun önceki sürümünde "kuantizasyonun düzenlileştirme etkisi" ile açıklanmıştı. **Mevcut veri bu açıklamayı desteklemiyor:**

- Her yöntem için tek seed var.
- Çalıştırmalar arasında kuantizasyon dışı farklar olabilir: QLoRA notebook'u `float16` compute dtype ile çalıştı, config ise `bfloat16` diyor. LoRA eğitimi QLoRA'dan uzun sürdü (4h 36m'ye karşı ~3h 48m); normalde tersi beklenir. Bu da hassasiyet ayarlarında bir fark olabileceğini düşündürüyor.
- QLoRA'nın skorunun bir kısmı, gevşek parser'ın `<tool_call>` etiketsiz JSON'u kabul etmesinden geliyor.

Önce LoRA ve QLoRA aynı dtype ayarlarıyla ve birkaç seed ile yeniden çalıştırılmalı; ancak ondan sonra bir açıklama yapılmalı.

### 6.3 Maliyet-Kalite Dengesi

| Senaryo | Bu benchmark'a göre |
|---------|----------------|
| 0.5B modelde maksimum kalite | **Full FT** |
| Adapter gerekiyorsa (adapter swap, çoklu görev) | Bu kurulumdaki adapter'lar production için yeterli kaliteye ulaşmadı; önce `all-linear` LoRA/QLoRA denenmeli |
| Sınırlı eğitim VRAM'i | QLoRA'nın avantajı beklenir, ancak bu benchmark'ta ölçülmedi |

---

## 7. Sınırlamalar

1. **Baseline ölçülmedi:** 100 örneklik eval setinde değil, yalnızca 5 örnekle ve CPU'da çalıştırıldı.
2. **İstatistiksel güç düşük:** tek seed, 80 pozitif ve 20 negatif örnek. Negatiflerde 1 örnek = %5.
3. **Karıştırıcı faktör:** adapter'lar attention-only; Full FT'nin öğrenme oranı ve batch düzeni de farklı.
4. **Gevşek parser:** `extract_raw_tool_call_json`, etiketsiz `{"name": ...}` JSON'u da kabul ediyor.
5. **Yalnızca ilk tool çağrısı puanlanıyor:** paralel çağrılarda fazla ya da eksik çağrılar cezalandırılmıyor.
6. **Sentetik negatifler:** ~20 sabit soru-cevap şablonundan üretiliyor ve train ile val aynı havuzu paylaşıyor; ret skoru kısmen ezberi ölçüyor olabilir.
7. **Kontrolsüz üretim:** adapter modelleri `<|im_end|>` üretmiyor; bu hem hız ve gecikme ölçümünü bozuyor hem de bir kalite sorunu.
8. **Verimlilik ölçümleri eksik:** eğitim VRAM'i yok; eğitim süresi yalnızca LoRA için kayıtlı.
9. **Kırpma:** eğitim örneklerinin ~%36.5'i 2048 token'ı aşıyor; tool şemalarının başı kesiliyor.

---

## 8. Full FT Çıktı Örnekleri

### ✅ Başarılı Negatif Ret
**Soru:** *"Mona Lisa'yı kim yaptı?"*  
**Çıktı:** Doğal dilde yanıt (tool çağrısı yok) ✓

### ✅ Doğru Tool, Kısmen Doğru Argümanlar
```json
<tool_call>
{"name": "find_astronomy_apps", "arguments": {"features": ["real-time data", "celestial events", "constellation mapping"], "user_location": "34.0522N,118.2437W"}}
</tool_call>
```
Tool doğru, ancak `features` listesi ve `user_location` formatı ground-truth ile birebir eşleşmiyor. Argument accuracy'nin tool selection'dan çok daha düşük olmasının (%44 ile %92.5) tipik nedeni bu.

### ⚠️ Fazladan Paralel Tool Çağrısı
Tek tool beklenirken birden fazla tool çağrılabiliyor. Metrik yalnızca ilk çağrıyı puanladığı için bu davranış skora yansımıyor.

---

## 9. Sonuç

### Ana Bulgular
1. **Full Fine-Tuning**, bu kurulumda 0.5B model için açık ara en iyi yöntem oldu (%92.5 tool selection).
2. **Attention-only r=16 adapter'lar** (LoRA, DoRA, QLoRA) formatı güvenilir şekilde öğrenemedi ve durmayı öğrenmedi.
3. **QLoRA** adapter'lar arasında en yüksek skoru aldı, ancak LoRA'ya karşı farkın nedeni belirsiz; kuantizasyon kaynaklı bir düzenlileştirme etkisi kanıtlanmadı.
4. **DoRA ile LoRA** arasında ölçülebilir bir fark yok.
5. Fine-tuning ciddi aşırı tetiklemeye yol açmadı (%5–15 gereksiz çağrı), ancak negatif set küçük ve sentetik.

### Gelecek Çalışmalar
1. Baseline'ı 100 örneklik eval setinde GPU'da yeniden çalıştırmak
2. `target_modules: all-linear` ile LoRA/QLoRA denemesi (MLP dahil)
3. LoRA ve QLoRA'yı aynı dtype ile ve birden fazla seed ile tekrarlamak
4. Sıkı (yalnızca `<tool_call>` etiketli) parser ile yeniden puanlamak; tüm paralel çağrıları puanlamak
5. Eğitim VRAM'ini ve süresini her yöntem için kaydetmek
6. Train ile örtüşmeyen, daha büyük bir negatif set oluşturmak
7. Daha büyük modellerde (Qwen2.5-1.5B veya 3B) benchmark'ı tekrarlamak

---

*Bu rapor, `reports/*_metrics.json` dosyalarındaki ölçülen değerlere göre revize edilmiştir.*
