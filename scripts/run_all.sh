#!/usr/bin/env bash
# Tum yontemleri sirayla egitip eval eder.
set -euo pipefail

for cfg in configs/lora.yaml configs/qlora.yaml configs/dora.yaml configs/full_ft.yaml; do
  echo ">>> Training with $cfg"
  uv run python -m tool_calling_ft.training.train --config "$cfg"
done
