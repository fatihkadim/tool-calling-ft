# Tool Calling Fine-Tuning Benchmark

**[🇬🇧 English](#english) | [🇹🇷 Türkçe](#türkçe)**

---

<a name="english"></a>

# 🇬🇧 English

A systematic benchmark for comparing parameter-efficient and full fine-tuning methods on LLM-based tool calling.

## Overview

This project evaluates how different fine-tuning strategies affect a language model's ability to select tools, generate valid arguments, and produce reliable structured tool calls.

The benchmark compares:

* **LoRA**
* **QLoRA**
* **DoRA**
* **Full Fine-Tuning**

The same datasets, evaluation pipeline, and task definitions are used across experiments to ensure a fair comparison.

## Models

The benchmark is designed to support multiple open-weight language models.

Initial experiments can include lightweight models such as:

* Qwen2.5-0.5B
* Qwen2.5-1.5B
* SmolLM2
* Other compatible open-weight models

Using multiple model sizes allows the project to investigate how fine-tuning methods behave across different model capacities.

## Dataset

The benchmark uses function-calling data containing both:

* Requests that require a tool call
* Requests where no tool should be called

The data is transformed into a standardized format containing the user request, available tool schemas, expected tool selection, and expected arguments.

## Evaluation

A reusable evaluation harness measures both model quality and system-level efficiency.

### Quality Metrics

* **Tool Selection Accuracy** — whether the correct tool is selected
* **Argument Accuracy** — field-level exact match of generated arguments
* **JSON Validity Rate** — percentage of syntactically valid tool-call outputs
* **Invalid Tool Call Rate** — malformed or unsupported tool calls
* **Unnecessary Tool Call Rate** — tool calls made when no tool was required

### Performance Metrics

* Training time
* Peak VRAM usage
* Trainable parameter count
* Inference latency
* Generation throughput (tokens/sec)

The evaluation pipeline is independent from the training implementation, allowing it to be reused for future experiments.

## Experimental Design

The benchmark investigates questions such as:

* Can parameter-efficient fine-tuning achieve performance comparable to Full Fine-Tuning?
* How much VRAM can QLoRA save compared with LoRA?
* Does DoRA provide measurable improvements over standard LoRA?
* How does model size affect tool-calling performance?
* What is the relationship between training cost and final accuracy?
* Which method provides the best quality-to-resource trade-off?

## Repository Structure

```text
configs/
  Fine-tuning configurations

data/
  raw/
  processed/

src/
  tool_calling_ft/
    data/
    training/
    eval/
    utils/

scripts/
  Baseline and experiment entry points

tests/
  Evaluation and metric tests

reports/
  Generated benchmark results and visualizations

notebooks/
  Exploratory experiments and analysis
```

Raw datasets, processed datasets, model checkpoints, and generated experiment outputs are excluded from version control.

## Reproducibility

Experiments are configuration-driven to ensure consistent comparisons.

Each experiment records:

* Model
* Fine-tuning method
* Dataset version
* Hyperparameters
* Trainable parameters
* Hardware/resource usage
* Evaluation results

This allows experiments to be reproduced and new models or fine-tuning methods to be added without redesigning the evaluation pipeline.

## Goal

The goal is not simply to fine-tune a model, but to build a **reproducible benchmark for evaluating fine-tuning methods on tool-calling tasks**.

The final analysis focuses on the trade-offs between:

**accuracy · reliability · compute · memory · training cost · inference performance**

[⬆️ Back to top](#tool-calling-fine-tuning-benchmark)

---

<a name="türkçe"></a>

# 🇹🇷 Türkçe

Bu proje, LLM'lerde **tool calling / function calling** yeteneği üzerinde farklı fine-tuning yöntemlerini sistematik olarak karşılaştırmak için geliştirilmiş bir benchmark çalışmasıdır.

## Genel Bakış

Proje, farklı fine-tuning yöntemlerinin bir dil modelinin doğru aracı seçme, doğru argümanları üretme ve geçerli yapılandırılmış tool call oluşturma yeteneği üzerindeki etkisini ölçer.

Karşılaştırılan yöntemler:

* **LoRA**
* **QLoRA**
* **DoRA**
* **Full Fine-Tuning**

Tüm yöntemler aynı veri, görev tanımları ve değerlendirme pipeline'ı kullanılarak karşılaştırılır.

## Modeller

Benchmark, farklı açık ağırlıklı dil modelleriyle çalışabilecek şekilde tasarlanmıştır.

Başlangıç deneylerinde aşağıdaki gibi küçük modeller kullanılabilir:

* Qwen2.5-0.5B
* Qwen2.5-1.5B
* SmolLM2
* Uyumlu diğer açık ağırlıklı modeller

Birden fazla model boyutunun kullanılması, fine-tuning yöntemlerinin model kapasitesi değiştikçe nasıl davrandığını incelemeyi sağlar.

## Dataset

Benchmark, hem tool kullanımının gerekli olduğu hem de tool kullanımının gerekli olmadığı örneklerden oluşan function-calling verilerini kullanır.

Veriler standart bir formata dönüştürülerek:

* Kullanıcı isteği
* Kullanılabilir tool schema'ları
* Beklenen tool seçimi
* Beklenen argümanlar

gibi bilgileri içerir.

## Değerlendirme

Yeniden kullanılabilir bir evaluation harness ile model kalitesi ve sistem performansı birlikte ölçülür.

### Kalite Metrikleri

* **Tool Selection Accuracy** — doğru tool'un seçilme oranı
* **Argument Accuracy** — argüman alanlarının doğru üretilme oranı
* **JSON Validity Rate** — geçerli JSON çıktılarının oranı
* **Invalid Tool Call Rate** — hatalı veya desteklenmeyen tool call oranı
* **Unnecessary Tool Call Rate** — tool gerekmediği halde yapılan çağrıların oranı

### Performans Metrikleri

* Eğitim süresi
* Peak VRAM kullanımı
* Eğitilebilir parametre sayısı
* Inference latency
* Generation throughput (tokens/sec)

Evaluation pipeline'ı training kodundan bağımsız tasarlanmıştır. Böylece ileride farklı modeller ve optimizasyon yöntemleriyle tekrar kullanılabilir.

## Deney Tasarımı

Benchmark aşağıdaki sorulara cevap vermeyi amaçlar:

* Parameter-efficient fine-tuning, Full Fine-Tuning seviyesinde performans sağlayabilir mi?
* QLoRA, LoRA'ya kıyasla ne kadar VRAM tasarrufu sağlar?
* DoRA, standart LoRA'ya göre ölçülebilir bir avantaj sağlıyor mu?
* Model boyutu tool-calling performansını nasıl etkiliyor?
* Eğitim maliyeti ile model performansı arasındaki ilişki nedir?
* Hangi yöntem performans ve kaynak kullanımı açısından en iyi dengeyi sunuyor?

## Proje Yapısı

```text
configs/
  Fine-tuning konfigürasyonları

data/
  raw/
  processed/

src/
  tool_calling_ft/
    data/
    training/
    eval/
    utils/

scripts/
  Baseline ve deney çalıştırma scriptleri

tests/
  Evaluation ve metric testleri

reports/
  Benchmark sonuçları ve görselleştirmeler

notebooks/
  Deneysel çalışmalar ve analizler
```

Ham datasetler, işlenmiş datasetler, model checkpoint'leri ve deney çıktıları Git repository'sine dahil edilmez.

## Reproducibility

Deneyler configuration-driven bir yapı kullanılarak yürütülür.

Her deney için:

* Model
* Fine-tuning yöntemi
* Dataset versiyonu
* Hyperparameter'lar
* Eğitilebilir parametre sayısı
* Donanım ve kaynak kullanımı
* Evaluation sonuçları

kaydedilir.

Bu yapı, deneylerin tekrar üretilebilmesini ve yeni modeller veya fine-tuning yöntemlerinin mevcut pipeline değiştirilmeden eklenebilmesini sağlar.

## Amaç

Projenin amacı yalnızca bir modeli fine-tune etmek değil, **tool-calling görevlerinde farklı fine-tuning yöntemlerini karşılaştıran yeniden üretilebilir bir benchmark oluşturmak**.

Sonuçlar özellikle şu değişkenler arasındaki trade-off'ları incelemeye odaklanır:

**doğruluk · güvenilirlik · hesaplama maliyeti · bellek kullanımı · eğitim maliyeti · inference performansı**

[⬆️ Başa dön](#tool-calling-fine-tuning-benchmark)
