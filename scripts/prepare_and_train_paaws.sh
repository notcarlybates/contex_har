#!/bin/bash
# Prepare PAAWS data and train DeepConvContext model.
#
# Usage:
#   bash scripts/prepare_and_train_paaws.sh

set -e

echo "=== Step 1: Processing PAAWS data and generating config ==="
python prepare_paaws.py --paaws-dir ./data/paaws

echo ""
echo "=== Step 2: Training DeepConvContext model ==="
python main.py \
    --config configs/main_experiments/deepconvcontext/paaws_loso_lstm.yaml \
    --seed 1 \
    --ckpt-freq 10
