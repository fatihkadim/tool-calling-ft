#!/usr/bin/env bash
# Tum yontemleri sirayla egitip eval eder.
# Kullanim:
#   bash scripts/run_all.sh
#   SKIP_TRAINING=1 bash scripts/run_all.sh   # Sadece eval
#   METHODS="qlora,dora" bash scripts/run_all.sh
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
METHODS="${METHODS:-lora,qlora,dora,full_ft}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

IFS=',' read -ra METHOD_LIST <<< "$METHODS"

echo ""
echo "============================================================"
echo " TOOL CALLING FINE-TUNING BENCHMARK"
echo " Yontemler: ${METHODS}"
echo "============================================================"
echo ""

for method in "${METHOD_LIST[@]}"; do
  method=$(echo "$method" | xargs)  # trim
  config="configs/${method}.yaml"
  adapter="checkpoints/${method}"

  echo ""
  echo ">>> [${method}] Baslatiliyor..."
  echo "--------------------------------------------------"

  # --- Training ---
  if [ "$SKIP_TRAINING" = "0" ]; then
    if [ -f "$config" ]; then
      echo "  Training: $config"
      uv run python -m tool_calling_ft.training.train --config "$config"
      echo "  ✅ Training tamamlandi: $method"
    else
      echo "  ⚠ Config bulunamadi: $config — atlaniyor"
    fi
  fi

  # --- Evaluation ---
  if [ "$SKIP_EVAL" = "0" ]; then
    eval_args=(
      -m tool_calling_ft.eval.harness
      --model "$BASE_MODEL"
      --method "$method"
      --batch-size 4
    )

    # Adapter yolu varsa ekle
    if [ "$method" != "baseline" ] && [ -d "$adapter" ]; then
      eval_args+=(--adapter "$adapter")
    fi

    echo "  Evaluation baslatiliyor: $method"
    uv run python "${eval_args[@]}"
    echo "  ✅ Eval tamamlandi: reports/${method}_metrics.json"
  fi
done

# --- Gorsellestirme ---
echo ""
echo "============================================================"
echo " Gorsellestirme uretiliyor..."
echo "============================================================"
uv run python scripts/visualize_benchmark.py

echo ""
echo "🏆 Tum islemler tamamlandi!"
echo ""
