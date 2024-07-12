#!/bin/bash

# Paths
BACKUP_DIR="/home/solomon/data/lose_data/backup_models"
CACHE_DIR="$HOME/.cache/huggingface/hub/models--"
WORKING_DIR="/home/solomon/data/lose_data"
MODELS_DIR="$WORKING_DIR/models"
MODEL="xlm-roberta-base"
CACHE_MODEL_DIR="$CACHE_DIR$MODEL"

# Function to download and cache model
download_model() {
  local model=$1
  python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; AutoTokenizer.from_pretrained('$model'); AutoModelForQuestionAnswering.from_pretrained('$model')" 
}

# Download model if not already in backup
backup_model_dir="$BACKUP_DIR/models--$MODEL"
if [[ ! -d "$backup_model_dir" ]]; then
  echo "Downloading model: $MODEL"
  download_model $MODEL
  mkdir -p "$backup_model_dir"
  cp -r "$CACHE_MODEL_DIR"/* "$backup_model_dir/"
else
  echo "Model already exists in backup: $MODEL"
fi

# Clear working directory
echo "Clearing models from working directory: $MODELS_DIR"
rm -rf "$MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Restore models from backup to working directory
if [[ -d "$backup_model_dir" ]]; then
  echo "Restoring model from backup to working directory: $MODEL"
  cp -r "$backup_model_dir"/* "$MODELS_DIR/"
else
  echo "Backup not found for model: $MODEL"
fi

echo "Model preparation completed."
