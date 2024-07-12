#!/bin/bash

# Paths
BACKUP_DIR="/home/solomon/data/lose_data/backup_models"
CACHE_DIR="$HOME/.cache/huggingface/hub"
WORKING_DIR="/home/solomon/data/lose_data"
MODELS_DIR="$WORKING_DIR/models"
MODELS=("deepset-roberta-base-squad2" "distilbert-base-cased-distilled-squad" "sentence-transformers-distiluse-base-multilingual-cased" "xlm-roberta-base")

# Function to download and cache model
download_model() {
  local model=$1
  python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; AutoTokenizer.from_pretrained('$model'); AutoModelForQuestionAnswering.from_pretrained('$model')" 
}

# Download models if not already in backup
for model in "${MODELS[@]}"; do
  if [! -d "$BACKUP_DIR/$model" ]; then
    echo "Downloading model: $model"
    download_model $model
    mkdir -p "$BACKUP_DIR/$model"
    cp -r "$CACHE_DIR/$model/"* "$BACKUP_DIR/$model/"
  else
    echo "Model already exists in backup: $model"
  fi
done

# Remove models from cache
for model in "${MODELS[@]}"; do
  echo "Removing model from cache: $model"
  rm -rf "$CACHE_DIR/$model"
done

# Remove models from working directory
for model in "${MODELS[@]}"; do
  echo "Removing model from working directory: $model"
  rm -rf "$MODELS_DIR/$model"
done

# Restore pre-trained models from backup to working directory
for model in "${MODELS[@]}"; do
  if [! -d "$MODELS_DIR/$model" ]; then
    echo "Restoring model from backup to working directory: $model"
    mkdir -p "$MODELS_DIR/$model"
    cp -r "$BACKUP_DIR/$model/"* "$MODELS_DIR/$model/"
  else
    echo "Model already exists in working directory: $model"
  fi
done

echo "Model preparation completed."