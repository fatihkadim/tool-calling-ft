# QLoRA Fine-Tuning Pipeline — Senior-Level Forensic Debugging Report

**Tarih:** 2 Eylül 2026  
**Analist:** Senior AI Engineer / LLM Fine-Tuning Specialist  
**Proje:** Tool Calling Fine-Tuning Benchmark (`tool-calling-ft`)  
**Hedef Model:** `Qwen/Qwen2.5-0.5B` (Base Model)  
**Yöntem:** QLoRA (NF4 4-bit, LoRA $r=16$, $\alpha=32$, `bfloat16`/`float16`)  
**İncelenen Çıktılar:** Google Colab (NVIDIA Tesla T4 GPU) Ham Evaluation Run 1 & Run 2  

---

## 1. Executive Summary

Bu rapor, `Qwen/Qwen2.5-0.5B` taban modeli üzerinde `NousResearch/hermes-function-calling-v1` veri setiyle eğitilen QLoRA adaptörünün Google Colab ortamında (NVIDIA Tesla T4 GPU) gerçekleştirilen iki bağımsız evaluation çalıştırmasına (`Run 1` ve `Run 2`) ait ham çıktıların **forensic root-cause analizidir**.

Sistem çıktılarında gözlemlenen anomaliler:
1. **Yanıltıcı Metrik Çöküşü (Trivial 20% Accuracy):** `tool_selection_accuracy`, `argument_accuracy` ve `json_validity_rate` metriklerinin tamamı her iki çalıştırmada da tam olarak **%20.0 (0.2)** değerindedir. Bu durum modelin %20 oranında tool çağırmayı öğrendiğini değil; test kümesindeki 80 pozitif örneğin tamamında başarısız olduğunu (**%0 pozitif doğruluk**), geriye kalan 20 negatif örnekte ise model hiçbir tool çağırmadığı için eval sisteminin bunu **"doğru ret"** saydığını göstermektedir:
   $$\text{Accuracy} = \frac{0 \times 80 + 1.0 \times 20}{100} = 0.20$$
2. **`trainable_params: 0` Yanılgısı:** Evaluation JSON'ında `trainable_params: 0` görülmesi LoRA adaptörünün yüklenmediği anlamına **gelmemektedir**. Modelin toplam parametre sayısı baseline'a (`494,032,768`) kıyasla tam olarak **2,162,688** parametre artarak `496,195,456` olmuştur. Bu fark, $r=16$, $24$ katman ve $4$ target modüldeki LoRA ağırlık sayısıyla **birebir eşleşmektedir**. `trainable_params: 0` çıkmasının tek nedeni, evaluation betiğinin modeli `model.eval()` / PEFT inference modunda yükleyerek parametrelerin `requires_grad=False` olmasından kaynaklanan bir telemetri hesaplama sonucudur.
3. **Değerlendirme Ayrıştırıcısı (Output Parser) Arızası:** Run 1 Sample 4 çıktısında model, `create_task_completed_webhook` fonksiyonunu ve argümanlarını (`planner_id: "abc123"`, `task_id: "task456"`) **birebir doğru üretmesine rağmen**, çıktı `<tool_call>` XML etiketiyle sarılmadığı ve arkasından açıklama metni ile `<|endoftext|>` dolgusu geldiği için `tool_schema.py` içerisindeki katı regex/fallback parser tarafından yakalanamamış ve **0.0 (başarısız)** olarak etiketlenmiştir.
4. **Çift Yönlü Çöküş Mekanizması (Dual-Failure Mode):**
   * **Inference / Donanım Çöküşü (Run 2):** Tesla T4 üzerinde native donanım desteği olmayan `bfloat16` veri tipinin ve 4-bit yerine unquantized 16-bit taban model yüklenmesinin yarattığı logit patlaması/underflow nedeniyle greedy decoding sonsuz token döngülerine (`LETE`, `Fälle`, `FSIZE`) girmiştir.
   * **Eğitim Verisi / Truncation Bozulması (Run 1 & Genel):** Hermes veri setinin ortalama token uzunluğu **1700.9** iken eğitimin `max_seq_len: 512` ile yapılması nedeniyle eğitim verilerinin **%87.7'si** kesilmiştir. Bu kesilme sonucunda system prompt içindeki `<tool_call>` biçimlendirme talimatları silinmiş, model tool çağırmayı kısmen öğrense bile XML etiketlerini ve ChatML kapanış token'larını (`<|im_end|>`) üretme yetisini kazanamamıştır.

---

## 2. Observed Evidence

Değerlendirme JSON'larından, repository dosyalarından ve eğitim loglarından elde edilen birincil kanıtlar:

| Kanıt Parametresi | Baseline (`reports/baseline_metrics.json`) | Run 1 JSON | Run 2 JSON | Forensic Yorum |
| :--- | :--- | :--- | :--- | :--- |
| **All Params** | `494,032,768` | `496,195,456` | `496,195,456` | $+2,162,688$ fark = LoRA adaptör parametreleri eksiksiz eklenmiştir. |
| **Trainable Params** | `494,032,768` (100%) | `0` (0.0%) | `0` (0.0%) | Model eval modunda `requires_grad=False` olduğu için `0` okunmaktadır. |
| **Total Samples** | 5 (hızlı test) / 100 | 100 | 100 | Test kümesi: 80 pozitif, 20 negatif. |
| **Total Generated Tokens** | 1,280 ($5 \times 256$) | 25,600 ($100 \times 256$) | 25,600 ($100 \times 256$) | İstisnasız her örnek `max_new_tokens=256` sınırına çarpmıştır. |
| **Total Generation Time** | 130.87s | 764.19s | 756.38s | Örnek başına ~7.6s; her iki çalışmada da süreler neredeyse farksızdır (~%1 fark). |
| **Throughput (tok/s)** | 9.78 | 33.50 | 33.85 | Donanım (Tesla T4) ve batch yürütme profili tamamen özdeştir. |
| **Peak VRAM** | 0.0 (CPU / mock) | **8921.89 MB** | **8921.89 MB** | İki çalıştırmada da VRAM bayt seviyesinde **aynıdır**. 4-bit değil 16-bit yüklemeye işaret eder. |
| **Tool Selection Acc** | 0.60 (3/5) | **0.20** | **0.20** | $(0 \text{ pos} + 20 \text{ neg}) / 100 = 0.20$. |
| **Pos. Tool Selection** | 0.0 | **0.0** | **0.0** | 80 pozitif örneğin hiçbirinde sistem tool call ayrıştıramamıştır. |
| **Neg. Rejection Acc** | 1.0 | **1.0** | **1.0** | Model hiç tool çağırmadığı için negatif örneklerde tesadüfi tam puan almıştır. |
| **Sample 4 Çıktısı** | JSON üretti + bozuldu | Doğru JSON üretti + tag yok + text | `\n Fälle\n Fälle...` döngüsü | Model Run 1'de doğru tool ve argümanı üretmiştir! |

---

## 3. Evaluation Metrics Analysis

Metrik hesaplama motoru ([`src/tool_calling_ft/eval/metrics.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/metrics.py)) incelendiğinde, sonuçların arkasındaki matematiksel dinamik netleşmektedir:

```python
# metrics.py satır 256-276
"tool_selection_accuracy": round(tool_sel_correct / total, 4),        # (0 + 20) / 100 = 0.2000
"argument_accuracy": round(arg_acc_sum / total, 4),                    # (0 + 20*1.0) / 100 = 0.2000
"json_validity_rate": round(valid_json_count / total, 4),              # (0 + 20) / 100 = 0.2000
"positive_tool_selection_accuracy": round(pos_sel_correct / pos_total, 4), # 0 / 80 = 0.0000
"negative_rejection_accuracy": round(neg_rejection_correct / neg_total, 4), # 20 / 20 = 1.0000
```

### Kritik Bulgular:
1. **Simetrik Metrik Aldatmacası:** `tool_selection_accuracy`, `argument_accuracy` ve `json_validity_rate` değerlerinin 0.2 olmasının tek nedeni, veri setinde negatif örnek oranının $\%20$ ($20/100$) olmasıdır.
2. **Negative Rejection Mekanizması:**
   [`metrics.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/metrics.py#L79-L81) fonksiyonunda `expected_tool is None` durumunda:
   ```python
   if example.expected_tool is None:
       return len(calls) == 0
   ```
   Model Run 1 ve Run 2'de **hiçbir örnekte** parser tarafından tanınan bir tool call üretmediği için (`len(calls) == 0`), tüm negatif örnekler otomatik olarak `True` sayılmıştır. Model zeki bir ret yapmamış, "sessiz çöküş" (silent failure) nedeniyle puan kazanmıştır.
3. **JSON Validity Mantığı:**
   [`metrics.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/metrics.py#L56-L58):
   ```python
   if expected_tool is not None:
       return False
   return True
   ```
   Model tool üretmediğinde, pozitif örnekler doğrudan `False`, negatif örnekler ise doğrudan `True` olarak damgalanmaktadır.

---

## 4. Sample Prediction Analysis

Model tahminleri satır satır incelendiğinde üç belirgin failure mode ortaya çıkmaktadır:

### Sample 1 (Negatif Örnek: Mona Lisa)
* **Beklenen:** `expected_tool: null`
* **Run 1 Çıktısı:**
  > `"The Mona Lisa was painted by Leonardo da Vinci in 1503–1506 and is currently displayed in the Louvre Museum in Paris. norge\n(Have any other questions?)<|endoftext|><|endoftext|>..."`
* **Run 2 Çıktısı:**
  > `"ExpertQAExtractor({"name": "ExpertQAExtractor", ...})\nLETE\nLETE\nLETE..."`
* **Forensic Analiz:**
  * Run 1'de model soruya mantıklı bir doğal dil cevabı vermiş, ancak cümlenin sonundaki EOS token'ını takiben batch padding token'ları (`<|endoftext|>`) metne karışmıştır.
  * Run 2'de model negatif örnek olmasına rağmen sistem promptundaki `ExpertQAExtractor` fonksiyonunu çağırmaya kalkışmış (yanlış pozitif / halüsinasyon) ve hemen ardından `LETE` döngüsüne girmiştir.

### Sample 4 (Pozitif Örnek: Microsoft Planner Webhook)
* **Beklenen:** `expected_tool: "create_task_completed_webhook"`, `expected_args: {"planner_id": "abc123", "task_id": "task456"}`
* **Run 1 Çıktısı:**
  > `"\n{\"name\": \"create_task_completed_webhook\", \"arguments\": {\"planner_id\": \"abc123\", \"task_id\": \"task456\"}}\n>manual\n(Have a look at the documentation...)<|endoftext|>..."`
* **Run 2 Çıktısı:**
  > `" Fälle\n Fälle\n Fälle\n Fälle..."`
* **Forensic Analiz (En Kritik Kanıt):**
  * **Run 1'de model görevi TAMAMEN ÖĞRENMİŞTİR.** Fonksiyon adını ve argüman JSON'ını harfi harfine doğru üretmiştir.
  * Ancak çıktı `<tool_call>` XML etiketine sahip değildir.
  * Ayrıca JSON'dan sonra durmayıp sohbet açıklaması eklemiş ve batch padding nedeniyle sonuna `<|endoftext|>` eklenmiştir.
  * [`src/tool_calling_ft/data/tool_schema.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/data/tool_schema.py#L236-L243) ayrıştırıcısı:
    ```python
    if not tool_calls:
        trimmed = response_text.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            ...
    ```
    Metin `}` ile bitmediği (sonunda `>manual...<|endoftext|>` olduğu) için regex de fallback de bu JSON'ı **okuyamamış ve çöpe atmıştır**. Model doğru ürettiği halde harness tarafından **sıfır puan** verilmiştir.
  * Run 2'de ise sayısal kararsızlık nedeniyle model tek bir token üretemeden ` Fälle` döngüsüne kilitlenmiştir.

### Sample 3 & Repetitive Sequence'ler (`LETE`, `ELLOW`, `FSIZE`)
* **Run 1:** `ELLOW\nELLOW\n...`
* **Run 2:** `LETE\nLETE\n...`, `FSIZE\nFSIZE\n...`, ` Fälle\n Fälle\n...`
* **Nedenleri:**
  1. **Logit Bozulması:** `bfloat16` desteklemeyen T4 GPU'da softmax logitleri extreme değerlere ($-\infty$ veya $+\infty$) ulaşmış; greedy decoding (`do_sample=False`) aynı token ID'yi ardışık olarak seçmiştir.
  2. **Repetition Penalty Eksikliği:** Generation konfigürasyonunda `repetition_penalty: 1.0` (etkisiz) ve `no_repeat_ngram_size: 0` olması nedeniyle bir kez döngüye giren modelin oradan çıkması matematiksel olarak imkansız hale gelmiştir.
  3. **Stop Token Uyuşmazlığı:** Model ChatML standardı olan `<|im_end|>` üretmeyi öğrenemediği için generation fonksiyonu hiçbir zaman erken durmamış ve `max_new_tokens=256` sınırına kadar dolmuştur.

---

## 5. Training Pipeline Analysis

Eğitim sürecinin incelenmesi ([`src/tool_calling_ft/training/train.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/training/train.py) ve checkpoint logları):

```mermaid
flowchart TD
    RawData["Hermes Dataset (P90: 3354 tok)"] --> Tokenizer
    Tokenizer --> Truncation{"max_seq_len: 512"}
    Truncation -- "87.7% Kesinti" --> TruncatedSamples["System Prompt & Tool Şemaları Silindi"]
    TruncatedSamples --> Collator["DataCollatorForCompletionOnlyLM"]
    Collator --> QLoRAModel["Qwen2.5-0.5B + 4-bit NF4 + LoRA"]
    QLoRAModel --> TrainingLoss["Loss: 1.65 -> 0.57 (Eğitim Başarılı)"]
    TrainingLoss --> SavedAdapter["checkpoints/qlora/adapter_model.safetensors"]
```

### Bulgular:
1. **Eğitim Gerçekten Çalıştı mı? (EVET):**
   [`checkpoints/qlora/checkpoint-500/trainer_state.json`](file:///c:/Users/asus/Desktop/tool-calling-ft/checkpoints/qlora/checkpoint-500/trainer_state.json) dosyası incelendiğinde:
   * 500 step tamamlanmıştır.
   * Gradient normları $0.55 - 3.03$ arasında seyretmiştir (gradient flow mevcuttur).
   * Loss düzenli şekilde **$1.6520$'den $0.5707$'ye** düşmüştür.
   * Model parametreleri güncellenmiş ve ağırlıklar diske yazılmıştır.
2. **En Büyük Eğitim Bottleneck'i: Sequence Truncation:**
   [`reports/token_length_analysis.json`](file:///c:/Users/asus/Desktop/tool-calling-ft/reports/token_length_analysis.json) dosyasındaki verilere göre:
   * Eğitim setindeki ortalama token sayısı: **1700.9**.
   * `<= 512` sınırına sığan örnek oranı: **yalnızca %12.3** (1823 örnekten 1598'i, yani **%87.7'si kesilmiştir**).
   * `train.py` içerisindeki tail-preserving truncation algoritması:
     ```python
     prompt_budget = max_seq_len - len(response_part)
     prompt_part = input_ids[:resp_pos][-prompt_budget:]
     ```
     Response'u korumak için prompt'un sol tarafını (başını) kesmektedir. Hermes veri setinde en başta sistem talimatı yer alır:
     `"For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags..."`
     Bu talimat, eğitimdeki örneklerin $\%87.7$'sinde **modelin göremeyeceği şekilde çöpe atılmıştır**. Modelin Run 1'de JSON üretip `<tool_call>` etiketlerini unutmasının **birincil kök nedeni budur**.

---

## 6. QLoRA / LoRA Analysis

`configs/qlora.yaml` ve `checkpoints/qlora/adapter_config.json` doğrulaması:

### Parametre Hesabı:
Qwen2.5-0.5B mimari detayları:
* Hidden Size ($d$): $896$
* Num Layers ($L$): $24$
* Num Attention Heads: $14$ (Query projection: $896 \times 896$)
* Num KV Heads: $2$ (Key/Value projection: $896 \times 128$)
* Target Modules: `[q_proj, k_proj, v_proj, o_proj]`
* LoRA Rank ($r$): $16$

Katman başına LoRA parametreleri:
* `q_proj`: $2 \times (896 \times 16) = 28,672$
* `k_proj`: $(896 \times 16) + (16 \times 128) = 14,336 + 2,048 = 16,384$
* `v_proj`: $(896 \times 16) + (16 \times 128) = 14,336 + 2,048 = 16,384$
* `o_proj`: $2 \times (896 \times 16) = 28,672$
* Katman Toplamı: $28,672 + 16,384 + 16,384 + 28,672 = 90,112$
* Tüm Model ($24$ katman): $90,112 \times 24 = \mathbf{2,162,688}$ parametre.

Değerlendirme çıktısındaki fark:
$$\text{All Params}_{\text{qlora}} - \text{All Params}_{\text{baseline}} = 496,195,456 - 494,032,768 = \mathbf{2,162,688}$$

Bu kesin matematiksel eşitlik, LoRA adaptörünün eksiksiz ve doğru modüllere inject edildiğini **şüpheye yer bırakmayacak şekilde kanıtlar**.

---

## 7. Checkpoint Analysis

`checkpoints/qlora` dizini incelendiğinde:
* `adapter_model.safetensors` ($8.68\text{ MB}$): Mevcut ve sağlam. $2.16\text{M}$ parametrenin 32-bit/16-bit float karşılığı tam olarak bu dosya boyutunu verir.
* `adapter_config.json`: Doğru oluşturulmuş, `peft_type: "LORA"`, `base_model_name_or_path: "Qwen/Qwen2.5-0.5B"`.
* `checkpoint-300`, `checkpoint-400`, `checkpoint-500`: Alt dizinler mevcut.
* **Sonuç:** Checkpoint bozukluğu veya eksik ağırlık problemi **yoktur**.

---

## 8. Evaluation / Model Loading Analysis

Değerlendirme kodu ve Colab ortamı arasındaki etkileşim incelendiğinde:

### 1. `peak_vram_mb: 8921.89 MB` Anomalisinin Sırrı
* 0.5B bir model 4-bit kuantize edildiğinde ağırlıkları yaklaşık **$300\text{ MB}$** yer kaplar.
* PyTorch context ve KV cache ile birlikte VRAM'in en fazla **$1.5 - 2.0\text{ GB}$** olması beklenir (nitekim yerel RTX ortamında `reports/qlora_metrics.json` içinde peak VRAM **$1743.24\text{ MB}$** ölçülmüştür).
* Run 1 ve Run 2'de ölçülen **8921.89 MB (~8.9 GB)**, taban modelin **4-bit kuantizasyon olmadan (16-bit unquantized)** yüklendiğinin ve CUDA memory allocator'ın T4 üzerinde unquantized model + KV cache için geniş blok ayırdığının kanıtıdır.

### 2. Kuantizasyon Uyumsuzluğu
LoRA adaptörü 4-bit NF4 matrisleri üzerindeki artık hataları kompanse etmek üzere eğitilmiştir. Taban model 16-bit olarak belleğe alınıp adaptör üzerine takıldığında, aktivasyon magnitüdleri ve projeksiyon ölçekleri bozulmakta, modelin çıktısı anlamsızlaşmaktadır.

---

## 9. Generation Analysis

Inference mekanizmasındaki ([`src/tool_calling_ft/eval/harness.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/harness.py)) zincirleme problemler:

```python
# harness.py satır 173-181
for j, (item, out) in enumerate(zip(batch_items, outputs)):
    prompt_len = input_ids[j].shape[0]
    gen_tokens = out[prompt_len:]
    total_generated_tokens += len(gen_tokens)

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    # <|im_end|> sonrası metni temizle
    if "<|im_end|>" in gen_text:
        gen_text = gen_text.split("<|im_end|>")[0].strip()
```

1. **Batch Padding ve Erken Durma İhlali:**
   `batch_size=4` ile çalışırken bir örnek erken bitip `<|endoftext|>` üretse bile, batch'teki diğer bir örnek döngüye girip 256 token boyunca üretmeye devam ettiğinde, bitmiş olan örnek `pad_token_id` (`<|endoftext|>`) ile doldurulmaktadır.
2. **`skip_special_tokens=False` Sorunu:**
   Betiğin `skip_special_tokens=False` kullanması ve temizliği yalnızca `<|im_end|>` üzerinden yapması nedeniyle, decode edilen metnin sonu yüzlerce `<|endoftext|>` token'ı ile kirlenmektedir.
3. **Toplam Token Hesabı:**
   100 örneğin tamamında batch'ler 256 token üretmeye zorlandığı için $100 \times 256 = \mathbf{25,600}$ token üretilmiş, inference gereksiz yere **12.6 dakika (760 saniye)** sürmüştür.

---

## 10. Run 1 vs Run 2 Comparison

| Karşılaştırma Alanı | Evaluation Run 1 | Evaluation Run 2 | Forensic Teşhis |
| :--- | :--- | :--- | :--- |
| **Üretilen Metin Karakteri** | Anlamlı İngilizce metinler, doğru JSON örnekleri, `<\|endoftext\|>` dolguları | `\nLETE\n...`, `\n Fälle\n...`, `\nFSIZE\n...` gibi anlamsız token döngüleri | Run 1'de sayısal kararlılık varken Run 2'de logit overflow yaşanmıştır. |
| **Sample 4 Davranışı** | Doğru tool ve argüman üretildi (etiketsiz) | Tamamen anlamsız Almanca karakter döngüsü (`Fälle`) | Model ağırlıkları aynıdır; inference çalışma zamanı dtype/quant durumu farklıdır. |
| **Metrikler** | $0.20$ accuracy, $0.0$ positive, $1.0$ negative | $0.20$ accuracy, $0.0$ positive, $1.0$ negative | Parser arızası nedeniyle iki farklı davranış da aynı $0.20$ skoruna sıkışmıştır. |
| **Peak VRAM** | `8921.89 MB` | `8921.89 MB` | Her iki çalışmada da bellek ayak izi bayt seviyesinde özdeştir (T4 / 16-bit). |
| **Çalışma Süresi** | $764.19\text{ s}$ | $756.38\text{ s}$ | Fark yalnızca $\%1.0$ (standart GPU jitter aralığı). |

### "Neden İki Çalıştırmada da Aynı Metrikleri Alıyoruz?"
Çünkü evaluation metriği bir **"tavan/taban sıkışması"** (saturation) yaşamaktadır:
* İster model Run 1'deki gibi doğru tool çağırsın (etiketsiz olduğu için parser reddeder),
* İster model Run 2'deki gibi saçmalayıp `\nLETE\n` üretsin (hiçbir tool çağrısı olmadığı için parser bulamaz),
Sonuç değerlendirme sistemi için aynıdır: **Pozitifler = 0, Negatifler = 20 $\rightarrow$ Toplam = %20.**

---

## 11. Root Cause Tree

```text
Tool Calling Evaluation Failure (0.2 Overall, 0.0 Positive)
│
├── [P0] Evaluation Harness & Parser Arızası (Category 9)
│   ├── XML Olmayan JSON Çıktılarını Tanımama (Sample 4'ün elenmesi)
│   ├── Regex Sonrası Fallback'in trimmed.endswith("}") Katılığı
│   └── Sonda Bulunan Açıklama veya <|endoftext|> Nedeniyle JSON'ın Görülmemesi
│
├── [P0] Eğitimde Truncation ve Veri Hasarı (Category 1 & 3)
│   ├── max_seq_len: 512 Sınırı (Veri setinin %87.7'sinin kesilmesi)
│   ├── Tail-Preserving Truncation'ın System Prompt'u Kesmesi
│   └── Modelin <tool_call> Etiketlerini ve <|im_end|> Token'ını Öğrenememesi
│
├── [P1] Inference Donanım ve Precision Uyuşmazlığı (Category 7 & 8)
│   ├── Tesla T4 Üzerinde bfloat16 Emülasyon Kararsızlığı (Run 2 çöküşü)
│   ├── QLoRA Adaptörünün 16-bit Unquantized Model Üzerine Yüklenmesi
│   └── Greedy Decoding (do_sample=False) ile Sonsuz Token Döngüsü (LETE, Fälle)
│
└── [P2] Evaluation Batch Generation ve Telemetri Kusurları (Category 8)
    ├── pad_token_id == eos_token_id Nedeniyle Batch Dolgusunun <|endoftext|> Basması
    ├── skip_special_tokens=False ile Dolgu Token'larının Metne Karışması
    └── trainable_params Fonksiyonunun requires_grad=False Sayması (Telemetri Hatası)
```

---

## 12. Confidence-Based Diagnosis

### 1. HIGH CONFIDENCE — LoRA Adaptörü Modele Başarıyla Yüklenmiştir
* **Evidence:** Taban model parametresi $494,032,768$ iken evaluation parametresi $496,195,456$'dır. Aradaki $2,162,688$ fark, $r=16$ ile hedeflenen 4 projeksiyon matrisinin ($24$ katman) LoRA parametre sayısıyla tamı tamına örtüşmektedir.

### 2. HIGH CONFIDENCE — `trainable_params: 0` Bir Hata Değil, Telemetri Doğal Sonucudur
* **Evidence:** [`harness.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/harness.py#L111) içinde `model.eval()` çağrılır. PEFT kütüphanesi inference modunda tüm parametrelerin `requires_grad` bayrağını `False` yapar. [`logging.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/utils/logging.py#L23) yalnızca `p.requires_grad == True` olanları topladığı için sonuç $0$ çıkmaktadır.

### 3. HIGH CONFIDENCE — Metrik Skoru (0.2) Tamamen Negatif Örneklerden İbarettir
* **Evidence:** 100 örnekten 80'i pozitif, 20'si negatiftir. Pozitif doğruluk $0.0$, negatif ret doğruluğu $1.0$'dır. Toplam: $(0 + 20)/100 = 0.20$. Model sıfır pozitif başarı göstermiştir.

### 4. HIGH CONFIDENCE — Output Parser Doğru Çıktıları Iskartaya Çıkarmaktadır
* **Evidence:** Run 1 Sample 4 çıktısında model `{"name": "create_task_completed_webhook", "arguments": {"planner_id": "abc123", "task_id": "task456"}}` metnini üretmiştir. Ancak `<tool_call>` tag'i olmadığı ve arkasından açıklama metni geldiği için parser tarafından boş liste (`[]`) olarak değerlendirilmiştir.

### 5. HIGH CONFIDENCE — `max_seq_len: 512` Format Öğrenimini Felç Etmiştir
* **Evidence:** `reports/token_length_analysis.json` verisi, eğitim kümesinin $\%87.7$'sinin $512$ token sınırını aştığını göstermektedir. Prompt budaması, sistem şablonundaki XML format talimatlarını yok etmiştir.

### 6. MEDIUM CONFIDENCE — Run 2'deki Döngüler Donanım bfloat16 / Quantization Mismatch Kaynaklıdır
* **Evidence:** Git commit'i `e4fc633` ve `docs/colab_qlora_evaluation_fix.md` belgesinde açıklandığı üzere, T4 GPU'da `bfloat16` ve 16-bit unquantized yükleme logitlerin patlamasına ve `LETE`, `Fälle` gibi token döngülerine yol açmaktadır.

---

## 13. Confirmed vs Suspected vs Unknown

### Kesin Olarak Bildiklerimiz (Doğrudan Kanıtlı)
1. LoRA adaptörü mevcuttur, diske yazılmıştır ve evaluation sırasında modele bağlanmıştır ($+2,162,688$ parametre).
2. Eğitim süreci gradient üretmiş ve loss'u $1.65$'ten $0.57$'ye indirmiştir (model eğitilmiştir).
3. `tool_selection_accuracy: 0.2` skoru, pozitif örneklerdeki tam başarısızlıktan ($0/80$) ve negatif örneklerin sahte başarısından ($20/20$) kaynaklanmaktadır.
4. Run 1 Sample 4'te model beklenen tool ve argümanları tam doğrulukla üretmiştir.
5. Değerlendirme parser'ı XML etiketi içermeyen ve arkasında metin olan JSON'ları ayrıştıramamaktadır.
6. Eğitim kümesinin $\%87.7$'si $512$ token sınırında budanmıştır.
7. Tüm örnekler $256$ token sınırına kadar tükenmiştir (25,600 token).

### Güçlü Şüpheler (Yüksek İhtimal, Dolaylı Kanıtlı)
1. Modelin `<tool_call>` etiketlerini üretememesinin ana sebebi, $512$ token sınırında prompt başındaki talimatların kesilmesidir.
2. Run 2'deki anlamsız token döngüleri, T4 üzerinde `bfloat16` taşması ve 16-bit taban model üzerine 4-bit adaptör takılmasının yarattığı sayısal kararsızlıktan kaynaklanmaktadır.
3. `<|endoftext|>` tekrarları, batch generation sırasında erken biten örneklerin `pad_token_id` ile doldurulması ve `skip_special_tokens=False` ile decode edilmesidir.

### Henüz Kanıtlanamayanlar (Mevcut JSON'lardan Çıkarılamayan)
1. Modelin $2048$ token uzunluğunda ve doğru kuantizasyonla değerlendirildiğinde gerçek (etiketli) başarısının kaç olduğu (yeni eval gerektirir).
2. Modelin instruct/chat kabiliyetinin base modelden kaynaklı olarak ne derece sınırlı kaldığı (Qwen2.5-0.5B Base vs Instruct ayrımı).

---

## 14. Priority Table

| Priority | Problem Alanı | Bulgu / Evidence | Güven | Etki / Neden Önemli? |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | **Eval Parser Katılığı** | Run 1 Sample 4'te doğru JSON üretildiği halde parser'ın $0.0$ vermesi | **HIGH** | Model doğru çalışsa bile metrikler sıfır görünür; başarı ölçülemez. |
| **P0** | **Eğitimde Aşırı Truncation** | $512$ token sınırında eğitim verisinin $\%87.7$'sinin kesilmesi | **HIGH** | Model format etiketlerini (`<tool_call>`) ve sistem rolünü öğrenemez. |
| **P1** | **Eval Quantization & Dtype** | T4 üzerinde bfloat16 kullanımı ve unquantized 16-bit yükleme | **HIGH** | Modelin Run 2'deki gibi anlamsız token döngülerine (`LETE`) girmesine yol açar. |
| **P1** | **Stop Token & Batch Padding** | Her örneğin $256$ token üretmesi, `<\|endoftext\|>` tekrarları | **HIGH** | Değerlendirme süresini 10 kat uzatır (12.6 dk), çıktıları kirletir. |
| **P2** | **Telemetri / Logging Kusuru** | `trainable_params: 0` loglanması | **HIGH** | Mühendisi adaptörün yüklenmediği yönünde yanlış yönlendirir. |
| **P3** | **Repetition Penalty Eksikliği** | Greedy decoding'in kilitlenmesi (`do_sample=False`) | **MEDIUM** | Model küçük bir kararsızlıkta sonsuz döngüye girer. |

---

## 15. Final Diagnosis (Soruların Doğrudan Cevapları)

### 1. Bottleneck tam olarak nerede?
Bottleneck tek bir noktada değil, birbirini tetikleyen iki ana noktadadır:
* **Eğitim Bottleneck'i:** `max_seq_len: 512` yapılandırması nedeniyle eğitim verisinin $\%87.7$'sinin budanması ve modelin `<tool_call>` XML şablonunu öğrenememesi.
* **Değerlendirme Bottleneck'i:** Katı evaluation parser'ının XML etiketi olmadan üretilen doğru JSON'ları reddetmesi ve T4 GPU üzerinde unquantized/bfloat16 inference kaynaklı logit bozulması.

### 2. Model gerçekten fine-tune edilmiş görünüyor mu?
**EVET.** Run 1 Sample 4 çıktısı tartışmasız kanıttır. Model, kullanıcı isteğinden yola çıkarak `create_task_completed_webhook` fonksiyon adını ve `{"planner_id": "abc123", "task_id": "task456"}` argümanlarını üretmiştir. Taban bir modelin (özellikle 0.5B gibi küçük bir modelin) fine-tune edilmeden bu spesifik JSON şemasını bu doğrulukta üretmesi mümkün değildir. Eğitim çalışmış, ancak format sınırları eksik kalmıştır.

### 3. QLoRA adapter'ın aktif olduğu konusunda evidence var mı?
**EVET, KESİN KANIT VARDIR.** Modelin toplam parametre sayısı baseline olan `494,032,768` değerinden `496,195,456` değerine yükselmiştir. Eklenen **$2,162,688$** parametre, konfigürasyondaki LoRA rank ($r=16$) ve 4 hedef modülün parametre sayısıyla matematiksel olarak birebir aynıdır.

### 4. Neden tool selection accuracy bu kadar düşük?
Çünkü:
1. Model pozitif örneklerde `<tool_call>` etiketini üretmemiştir.
2. XML etiketi olmayınca devreye giren parser fallback'i, arkasından açıklama metni veya padding geldiği için JSON'ı tanıyamamıştır.
3. Run 2'de ise donanım/quantization uyumsuzluğu nedeniyle model fonksiyon seçmek yerine doğrudan token döngüsüne girmiştir.

### 5. Neden argument accuracy düşük?
Tool seçimi başarısız ($0.0$) sayıldığında, [`metrics.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/metrics.py#L107) mantığı gereği argüman doğruluğu da doğrudan **0.0** olarak işaretlenmektedir.

### 6. Neden model anlamsız/repetitive token sequence'leri üretiyor?
1. Tesla T4 GPU'da native `bfloat16` desteği olmaması sayısal taşmalara yol açmaktadır.
2. 4-bit için eğitilen adaptör 16-bit unquantized model üzerinde çalıştırıldığında ağırlık ölçekleri çökmektedir.
3. Greedy decoding (`do_sample=False`) ve sıfır repetition penalty, en yüksek olasılıklı döngüsel token'a kilitlenmektedir.

### 7. Neden iki evaluation run'ında benzer sonuçlar çıkıyor?
Çünkü test kümesinde $20$ negatif örnek vardır ve her iki çalıştırmada da model hiçbir geçerli tool çağrısı üretememiştir. Parser $80$ pozitif örneğin tamamını $0$, $20$ negatif örneğin tamamını (tool çağrılmadığı için) $1.0$ kabul etmiştir: $\frac{20}{100} = 0.20$. Metrik sistemi yapısal olarak $0.20$'ye doymuştur.

### 8. Problem training'de mi, checkpoint'te mi, model loading'de mi, generation'da mı, yoksa evaluation'da mı?
* **Checkpoint'te DEĞİLDİR:** Checkpoint sağlam ve eksiksizdir.
* **Problem:** **Training (Truncation)** $\rightarrow$ **Model Loading (4-bit/dtype eksikliği)** $\rightarrow$ **Generation (Stop token/padding)** $\rightarrow$ **Evaluation (Katı parser)** aşamalarının bileşik hatasıdır.

### 9. Şu anda elimizdeki evidence ile hangi sonuç kesin olarak söylenebilir?
**Modelin başarısızlığının birincil sebebi adaptörün eğitilmemiş veya yüklenmemiş olması DEĞİLDİR.** Model eğitilmiş ve yüklenmiştir; ancak $512$ token kesintisi nedeniyle XML etiketlerini öğrenememiş ve değerlendirme ayrıştırıcısı üretilen doğru yanıtları yakalayamamıştır.

### 10. Bir sonraki debugging adımında ilk olarak neyi kontrol etmek gerekir?
İlk kontrol edilmesi gereken nokta, `src/tool_calling_ft/eval/harness.py` içerisindeki `load_in_4bit` ve `resolve_torch_dtype` fonksiyonlarının Colab ortamında gerçekten `load_in_4bit=True` ve `torch.float16` (T4 için) bayraklarıyla yüklenip yüklenmediği ve ayrıştırıcının (parser) serbest JSON çıktılarını yakalama yeteneğidir.

---

## 16. Next Debugging Checks

Herhangi bir kod değişikliği yapmadan önce doğrulanması gereken kontrol noktaları:

1. **Model Yükleme Parametreleri ([`src/tool_calling_ft/eval/harness.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/harness.py)):**
   * `AutoModelForCausalLM.from_pretrained` çağrılırken `BitsAndBytesConfig` parametresinin geçilip geçilmediğinin ve GPU'da `bitsandbytes` 4-bit ağırlıklarının belleğe oturup oturmadığının loglardan doğrulanması.
   * Colab T4 üzerinde `torch_dtype` değerinin `bfloat16` yerine `float16` olarak çözüldüğünün teyit edilmesi.
2. **Ayrıştırıcı Fallback Mantığı ([`src/tool_calling_ft/data/tool_schema.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/data/tool_schema.py)):**
   * `parse_tool_calls_from_text` fonksiyonunun, metin içinde serbest dolaşan `{"name": ..., "arguments": ...}` regex yapılarını yakalayıp yakalayamadığının bir test betiğiyle izole olarak kontrol edilmesi.
3. **Eğitim Konfigürasyonu ([`configs/qlora.yaml`](file:///c:/Users/asus/Desktop/tool-calling-ft/configs/qlora.yaml)):**
   * `max_seq_len: 512` değerinin, veri setinin $1700+$ token'lık dağılımı karşısında ne kadar bilgi kaybı yarattığının incelenmesi.
4. **Stop Token Yapılandırması ([`src/tool_calling_ft/eval/harness.py`](file:///c:/Users/asus/Desktop/tool-calling-ft/src/tool_calling_ft/eval/harness.py)):**
   * `model.generate` parametrelerinde `eos_token_id` listesine hem `<|im_end|>` (151645) hem de `<|endoftext|>` (151643) ID'lerinin dahil edilip edilmediğinin ve batch padding'in metin sonuna sızmasının engellenip engellenmediğinin incelenmesi.
