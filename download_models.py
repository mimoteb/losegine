import os
from transformers import AutoTokenizer, AutoModel

def download_model(model_name, cache_dir):
    if not os.path.exists(os.path.join(cache_dir, model_name)):
        print(f"Downloading {model_name} model...")
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        print(f"{model_name} model already exists. Skipping download.")

if __name__ == '__main__':
    cache_dir = '/home/solomon/data/lose_data/models'
    os.makedirs(cache_dir, exist_ok=True)
    download_model('xlm-roberta-base', cache_dir)
