# Tool Calling Fine-Tuning — Tüm yöntemleri sırayla eğitip eval eder.
# PowerShell versiyonu (Windows uyumlu)
#
# Kullanım:
#   uv run powershell -File scripts/run_all.ps1
#   uv run powershell -File scripts/run_all.ps1 -SkipTraining  # Sadece eval
#   uv run powershell -File scripts/run_all.ps1 -Methods "qlora,dora"

param(
    [switch]$SkipTraining,
    [switch]$SkipEval,
    [string]$Methods = "lora,qlora,dora,full_ft",
    [int]$MaxSamples = 0,
    [int]$BatchSize = 4,
    [string]$BaseModel = "Qwen/Qwen2.5-0.5B"
)

$ErrorActionPreference = "Stop"

$methodList = $Methods -split ","

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host " TOOL CALLING FINE-TUNING BENCHMARK" -ForegroundColor Cyan
Write-Host " Yöntemler: $($methodList -join ', ')" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

foreach ($method in $methodList) {
    $method = $method.Trim()
    $configFile = "configs/$method.yaml"
    $adapterPath = "checkpoints/$method"

    Write-Host ""
    Write-Host ">>> [$method] Başlatılıyor..." -ForegroundColor Yellow
    Write-Host "-" * 50

    # --- Training ---
    if (-not $SkipTraining) {
        if (Test-Path $configFile) {
            Write-Host "  🎯 Training: $configFile" -ForegroundColor Green
            uv run python -m tool_calling_ft.training.train --config $configFile
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ❌ Training başarısız: $method" -ForegroundColor Red
                continue
            }
            Write-Host "  ✅ Training tamamlandı: $method" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Config bulunamadı: $configFile — atlanıyor" -ForegroundColor DarkYellow
        }
    }

    # --- Evaluation ---
    if (-not $SkipEval) {
        $evalArgs = @(
            "-m", "tool_calling_ft.eval.harness",
            "--model", $BaseModel,
            "--method", $method,
            "--batch-size", $BatchSize
        )

        # Adapter yolu varsa ekle (baseline hariç)
        if ($method -ne "baseline" -and (Test-Path $adapterPath)) {
            $evalArgs += @("--adapter", $adapterPath)
        }

        # Max samples belirtilmişse
        if ($MaxSamples -gt 0) {
            $evalArgs += @("--max-samples", $MaxSamples)
        }

        Write-Host "  📊 Evaluation başlatılıyor: $method" -ForegroundColor Cyan
        uv run python @evalArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ❌ Eval başarısız: $method" -ForegroundColor Red
            continue
        }
        Write-Host "  ✅ Eval tamamlandı: reports/${method}_metrics.json" -ForegroundColor Green
    }
}

# --- Görselleştirme ---
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host " 📈 Görselleştirme üretiliyor..." -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

uv run python scripts/visualize_benchmark.py

Write-Host ""
Write-Host "🏆 Tüm işlemler tamamlandı!" -ForegroundColor Green
Write-Host ""
