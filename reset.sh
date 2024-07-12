#!/bin/bash

# Paths
BACKUP_DIR="/home/solomon/data/lose_data/backup_models"
CACHE_DIR="$HOME/.cache/huggingface/hub"
MODELS=("deepset--roberta-base-squad2" "distilbert--distilbert-base-cased-distilled-squad" "sentence-transformers--distiluse-base-multilingual-cased" "xlm-roberta-base")

# Function to download and cache model
download_model() {
  local model=$1
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('${model/--//}')"  # Replace -- with /
}

# Download models if not already in backup
for model in "${MODELS[@]}"; do
  if [ ! -d "$BACKUP_DIR/models--$model" ]; then
    echo "Downloading model: $model"
    download_model $model
    mkdir -p "$BACKUP_DIR/models--$model"
    cp -r "$CACHE_DIR/models--$model/"* "$BACKUP_DIR/models--$model/"
  else
    echo "Model already exists in backup: $model"
  fi
done

# Remove models from cache
for model in "${MODELS[@]}"; do
  echo "Removing model from cache: $model"
  rm -rf "$CACHE_DIR/models--$model"
done

# Restore pre-trained models from backup
for model in "${MODELS[@]}"; do
  echo "Restoring model from backup: $model"
  mkdir -p "$CACHE_DIR/models--$model"
  cp -r "$BACKUP_DIR/models--$model/"* "$CACHE_DIR/models--$model/"
done

echo "Model preparation completed."
