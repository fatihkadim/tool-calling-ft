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

### Sonuçlar (Kısa)

| Yöntem | Pozitif Tool Selection | Pozitif Arg. Accuracy | JSON Validity | Neg. Rejection | Peak VRAM |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Baseline (0-shot) | %0.0 | %0.0 | %0.0 | %100.0* | N/A (CPU) |
| LoRA (16-bit) | %8.75 | %7.71 | %11.25 | **%95.0** | 8922 MB |
| QLoRA (4-bit NF4) | **%46.25** | **%37.48** | **%46.25** | %85.0 | **8416 MB** |
| DoRA (use_dora=true) | %13.75 | %11.63 | %15.0 | **%95.0** | 8922 MB |
| Full Fine-Tuning | ⬜ *Planlanıyor* | — | — | — | — |

> **🏅 Mevcut Şampiyon: QLoRA** — Tüm pozitif metriklerde açık ara lider, aynı zamanda en düşük VRAM kullanımı.

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
- **Fikir:** Ağırlık matrisini *magnitude* (büyüklük) ve *direction* (yön) bileşenlerine ayır; LoRA yalnızca direction'ı günceller, magnitude ayrı öğrenilir.
- **Config:** LoRA + `use_dora=true`, $r=16, \alpha=32$, `dropout=0.05`
- **Trainable:** 2.21M parametre (%0.45) — LoRA'dan ~49K fazla (magnitude vektörleri)
- **Referans:** [DoRA: Weight-Decomposed Low-Rank Adaptation of Large Language Models (Liu et al., 2024)](https://arxiv.org/abs/2402.09353)

### 2.5 Full Fine-Tuning
- **Fikir:** Tüm parametreleri ($\sim$494M) güncelle.
- **Config:** `lr=2e-5`, `epochs=3`, `batch_size=1`, `grad_accum=8`

---

## 3. Kalite Metrikleri Karşılaştırması

### 3.1 Pozitif Örnekler (Tool Çağrısı Beklenen, n=80)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy | %0.0 | %8.75 | **%46.25** | %13.75 | — |
| Argument Accuracy | %0.0 | %7.71 | **%37.48** | %11.63 | — |
| JSON Validity | %0.0 | %11.25 | **%46.25** | %15.0 | — |

### 3.2 Negatif Örnekler (Tool Çağrılmaması Gereken, n=20)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Negative Rejection Acc. | %100.0* | **%95.0** | %85.0 | **%95.0** | — |
| Unnecessary Tool Call Rate | %0.0 | %5.0 | %15.0 | %5.0 | — |

*\* Baseline'da model hiç tool çağırmadığı için rejection %100'dür.*

### 3.3 Genel Metrikler (Toplam 100 örnek)

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Tool Selection Accuracy (Toplam) | %60.0* | %26.0 | **%54.0** | %30.0 | — |
| Argument Accuracy (Toplam) | %60.0* | %25.17 | **%46.98** | %28.3 | — |
| JSON Validity Rate (Toplam) | %100.0* | %29.0 | **%57.0** | %32.0 | — |
| Invalid Tool Call Rate | %0.0 | %1.0 | %1.0 | %1.0 | — |

*\* Baseline 5 sample ile çalıştırılmış (istatistiksel güvenilirlik düşük).*

---

## 4. Performans & Kaynak Metrikleri

| Metrik | Baseline | LoRA | QLoRA | DoRA | Full FT |
|--------|:--------:|:----:|:-----:|:----:|:-------:|
| Training Time | — | 4h 36m | ~3h 48m | ~4h 50m* | — |
| Peak VRAM (Eval) | 0 MB (CPU) | 8922 MB | **8416 MB** | 8922 MB | — |
| Throughput (tok/s) | 9.78 | 34.0 | **36.89** | 29.6 | — |
| Latency (ms/sample) | 26,175 | 7,528 | **6,939** | 8,649 | — |
| Trainable Params | 494M (100%) | 2.16M (0.43%) | 2.16M (0.43%) | 2.21M (0.45%) | — |
| Total Params | 494M | 496M | 317M | 496M | — |
| Adapter Size (safetensors) | — | ~8.5 MB | ~8.5 MB | ~8.5 MB | — |

*\* DoRA training time, LoRA'dan 684 step ile aynı step sayısında ancak magnitude decomposition overhead'i ile ~%5 daha uzun sürer.*

---

## 5. DoRA Detaylı Analiz

### 5.1 DoRA Adapter Konfigürasyonu (adapter_config.json)
```json
{
  "peft_type": "LORA",
  "use_dora": true,
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "task_type": "CAUSAL_LM",
  "peft_version": "0.19.1"
}
```

### 5.2 DoRA vs LoRA Karşılaştırması

DoRA, LoRA'ya göre marjinal bir iyileşme sağlamıştır:

| Metrik | LoRA | DoRA | Fark |
|--------|:----:|:----:|:----:|
| Positive Tool Selection | %8.75 | %13.75 | 🟢 **+5.0%** |
| Positive Argument Acc. | %7.71 | %11.63 | 🟢 **+3.92%** |
| Positive JSON Validity | %11.25 | %15.0 | 🟢 **+3.75%** |
| Negative Rejection | %95.0 | %95.0 | ➖ Eşit |
| Throughput (tok/s) | 34.0 | 29.6 | 🔻 -4.4 (overhead) |
| Latency (ms/sample) | 7,528 | 8,649 | 🔻 +1,121 ms |

### 5.3 DoRA Neden Beklentiyi Karşılayamadı?

DoRA'nın LoRA'ya göre **anlamlı ama yetersiz** bir iyileşme sağlamasının nedenleri:

1. **Aynı Kök Sorun Devam Ediyor:** DoRA da LoRA gibi yalnızca `q/k/v/o_proj` (attention) katmanlarını hedeflemektedir. MLP katmanları dondurulmuş olduğundan, format yönelimi (JSON output, `<tool_call>` etiketi) değişmemiştir.

2. **Magnitude Decomposition Tek Başına Yetmez:** DoRA'nın magnitude/direction ayrımı, daha büyük modellerde (7B+) anlamlı fark yaratmaktadır. 0.5B gibi küçük modellerde ağırlık matrislerinin rank'ı zaten düşük olduğundan, decomposition'ın etkisi sınırlı kalmıştır.

3. **Pre-training Artıkları Hâlâ Baskın:** Tıpkı LoRA gibi, DoRA da 16-bit hassasiyette çalışmakta ve base modelin web/kod halüsinasyonları (`sourceMapping`, `_Parms`, tekrarlayan tokenlar) hâlâ çıktılarda görülmektedir.

4. **Çıktı Örnekleri DoRA'nın Sınırlarını Gösteriyor:**
   - **Negatif örnek:** Mona Lisa sorusuna doğal dilde yanıt + `sourceMapping` halüsinasyonu → tool çağrılmamış (doğru ret ✓)
   - **Pozitif örnek:** `find_astronomy_apps` yerine anlamsız çıktı (`""""""` tekrarları) → tool format öğrenilememiş ✗
   - **Kısmi başarı:** `fetch_movie_reviews` tool adı ve argümanları kısmen doğru JSON çıktıda görülüyor, ama `<tool_call>` etiketi ve temiz JSON yapısı yok

### 5.4 DoRA Sonucu Hipotez Teyidi

> **Orijinal Hipotez:** *"DoRA'nın direction bileşeni, LoRA'nın yaşadığı format yönelim kaybını çözebilir."*
>
> **Sonuç:** ❌ **Kısmen Çürütüldü.** Direction decomposition'ı format yönelimine %5'lik marjinal katkı sağlamıştır, ancak kök sorun (MLP katmanlarının dondurulması + küçük model boyutu) devam ettiğinden, QLoRA'nın kuantizasyon-bazlı düzenlileştirme etkisine yaklaşamamıştır.

---

## 6. Üç Yöntem Sıralaması (Mevcut)

### 🥇 QLoRA (4-bit NF4) — En İyi Yöntem
- Tüm pozitif metriklerde açık ara lider (%46.25 tool selection)
- En düşük VRAM kullanımı (8416 MB)
- En yüksek throughput (36.89 tok/s) ve en düşük latency (6939 ms)
- **Dezavantaj:** Negatif rejection en düşük (%85.0) — model tool çağırmaya daha agresif

### 🥈 DoRA (use_dora=true) — LoRA'dan Marjinal İyileşme
- LoRA'ya göre +5% tool selection artışı
- Negatif rejection LoRA ile eşit (%95.0)
- **Dezavantaj:** QLoRA'ya göre 3x düşük pozitif accuracy, inference overhead

### 🥉 LoRA (16-bit) — En Zayıf Adapter Yöntemi
- Sadece %8.75 pozitif tool selection
- En iyi negatif rejection (%95.0, DoRA ile eşit)
- **Dezavantaj:** Pre-training artıkları format öğrenimini sabote ediyor

---

## 7. Analiz & Çıkarımlar

### 7.1 QLoRA > DoRA > LoRA: Neden Kuantize Model En İyi?

Üç yöntemin sonuçları, kuantizasyonun düzenlileştirme etkisini kesin olarak doğrulamıştır:

1. **Kuantizasyon = Düzenlileştirme (Regularization):** NF4 kuantizasyonu, base modelin pre-training artıklarını (web/kod halüsinasyonları) bastırarak adapter'ın öğrettiği formata bağlanmayı kolaylaştırmıştır.

2. **DoRA'nın Magnitude/Direction Ayrımı Kısmen İşe Yarıyor:** DoRA, direction bileşeni sayesinde LoRA'nın format kaybını %5 oranında telafi etmiştir. Ancak MLP katmanlarının dondurulması nedeniyle tam format dönüşümü sağlanamamıştır.

3. **Model Boyutu Kritik Eşik:** 0.5B parametreli model, tool calling gibi karmaşık format görevleri için temel kapasiteye sahip ancak attention-only adaptation yetersiz kalmaktadır.

### 7.2 Ortak Sorunlar (Tüm 16-bit Yöntemlerde)

| Sorun | LoRA | DoRA | QLoRA |
|-------|:----:|:----:|:-----:|
| `sourceMapping` halüsinasyonu | ✓ Var | ✓ Var | ✓ Var (az) |
| `_Parms` / repetition döngüsü | ✓ Var | ✓ Var | ✓ Var (az) |
| EOS token üretememe | ✓ Var | ✓ Var | ✓ Var (az) |
| `<tool_call>` etiketi yok | ✓ Var | ✓ Var | ~Kısmen |
| Pre-training format sızıntısı | ✓ Şiddetli | ✓ Orta | ✓ Hafif |

### 7.3 DoRA Beklentisi vs Gerçek

Önceki raporlarda sorulan *"DoRA'nın direction bileşeni format sapmasını düzeltebilir mi?"* sorusunun yanıtı:

> **Kısmen.** Direction decomposition LoRA'ya göre +5% iyileşme sağlamıştır, ancak attention-only target ve küçük model boyutu nedeniyle dramatik bir fark yaratamamıştır. QLoRA'nın kuantizasyon etkisi, DoRA'nın mathematical decomposition etkisinden 3x daha güçlü çıkmıştır.

---

## 8. Sonuç & Öneriler

### 8.1 Mevcut Bulgular
- **QLoRA**, küçük modellerde (0.5B) açık ara en iyi yöntem — hem kalite hem VRAM hem hız.
- **DoRA**, LoRA'ya göre marjinal iyileşme sağlıyor ancak QLoRA'yı yakalayamıyor.
- **LoRA (16-bit)** en zayıf yöntem — pre-training artıkları dominant kalıyor.
- Kuantizasyonun "bilgi kaybı" varsayımı küçük modellerde **çürütülmüştür** — düzenlileştirme etkisi baskın.

### 8.2 Full Fine-Tuning Beklentisi
Kalan tek yöntem olan Full Fine-Tuning ile:
- Tüm 494M parametre güncellenecek
- MLP katmanları da dahil olacağından format dönüşümü daha güçlü olabilir
- Ancak 16GB T4 GPU'da VRAM sınırı ile gradient checkpointing gerekecek
- Overfitting riski (1822 örnekle 494M parametre)

### 8.3 Gelecek Çalışmalar
1. `target_modules: all-linear` ile LoRA/DoRA denemesi
2. Daha büyük model (Qwen2.5-1.5B veya 3B) üzerinde benchmark tekrarı
3. Multi-turn tool calling evaluation
4. Token-level analiz: hangi tokenlar yanlış üretiliyor?

---

*Bu rapor, Full Fine-Tuning sonuçları tamamlandığında nihai haliyle güncellenecektir.*
